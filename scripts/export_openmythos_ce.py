"""Train/evaluate OpenMythos cross-entropy scores by recurrent loop depth.

This script is intended to run inside the ``openmythos-distill`` Docker image.
It trains or loads an OpenMythos recurrent-depth model, evaluates examples at
multiple loop depths, and writes the JSONL score file consumed by
``scripts/openmythos_diagnostics.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


APP_ROOT = Path("/app")
if APP_ROOT.exists() and str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

try:
    from distill_openmythos.tokenizer import ByteTokenizer
    from distill_openmythos.train_tiny_distill import (
        lr_at_step,
        make_batch,
    )
    from open_mythos.main import MythosConfig, OpenMythos
except ImportError as exc:  # pragma: no cover - exercised inside Docker.
    raise SystemExit(
        "OpenMythos image modules were not found. Run this script inside the "
        "openmythos-distill Docker image or mount /app onto PYTHONPATH."
    ) from exc


DEFAULT_DATA = Path("/app/distill_openmythos/data/teacher_pairs.jsonl")
DEFAULT_ARTIFACT_DIR = Path("/workspace/artifacts/openmythos-ce-export")
DEFAULT_SCORES_OUT = Path("/workspace/data/openmythos_loop_scores.jsonl")
DEFAULT_REPORT_OUT = Path("/workspace/data/reports/openmythos_ce_export_report.json")
SYSTEM_PROMPT = (
    "You are OpenMythos-Coder, a compact recurrent-depth coding assistant. "
    "Answer with correct, concise code and brief reasoning when useful."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/load OpenMythos and export CE loop-depth diagnostics."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--scores-out", type=Path, default=DEFAULT_SCORES_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--train-loop", type=int, default=4)
    parser.add_argument("--eval-loops", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--holdout-ratio", type=float, default=0.35)
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "medium", "full"],
        default="tiny",
        help="Scaled OpenMythos config. `full` is intended for long GPU runs.",
    )
    parser.add_argument("--dim", type=int, help="Override hidden dimension.")
    parser.add_argument("--heads", type=int, help="Override attention heads.")
    parser.add_argument("--kv-heads", type=int, help="Override grouped KV heads.")
    parser.add_argument("--experts", type=int, help="Override recurrent MoE experts.")
    parser.add_argument("--expert-dim", type=int, help="Override recurrent expert dimension.")
    parser.add_argument("--prelude-layers", type=int, help="Override prelude layers.")
    parser.add_argument("--coda-layers", type=int, help="Override coda layers.")
    parser.add_argument(
        "--train-stages",
        action="store_true",
        help="Expand staged-refinement examples into loop-specific training pairs.",
    )
    parser.add_argument(
        "--max-eval-per-split",
        type=int,
        help="Cap evaluated examples per split for quick validation runs.",
    )
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision training.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device, require_cuda=args.require_cuda)
    tokenizer = ByteTokenizer()
    records = load_records(args.data)
    split_records = split_dataset(records, args.seed, args.holdout_ratio)
    train_pairs = expand_training_pairs(split_records["train"], train_stages=args.train_stages)
    eval_pairs = {
        split: normalize_eval_pairs(items, limit=args.max_eval_per_split)
        for split, items in split_records.items()
    }
    max_loops = max([args.train_loop, *args.eval_loops])

    if args.checkpoint:
        model, checkpoint_metadata = load_checkpoint(args.checkpoint, device)
        if model.cfg.max_seq_len < args.seq_len:
            raise ValueError(
                f"Checkpoint max_seq_len={model.cfg.max_seq_len} is smaller than requested "
                f"seq_len={args.seq_len}."
            )
    else:
        model = OpenMythos(
            build_openmythos_config(
                vocab_size=tokenizer.vocab_size,
                seq_len=args.seq_len,
                loops=max_loops,
                args=args,
            )
        ).to(device)
        checkpoint_metadata = train_model(
            model=model,
            tokenizer=tokenizer,
            pairs=train_pairs,
            args=args,
            device=device,
        )
        save_checkpoint(model, tokenizer, args, args.artifact_dir, checkpoint_metadata)

    rows = export_scores(
        model=model,
        tokenizer=tokenizer,
        split_pairs=eval_pairs,
        seq_len=min(args.seq_len, model.cfg.max_seq_len),
        eval_loops=sorted(set(args.eval_loops)),
        device=device,
        metadata={
            "source": "openmythos_ce_export",
            "data_path": str(args.data),
            "checkpoint": str(args.checkpoint) if args.checkpoint else str(args.artifact_dir / "checkpoint.pt"),
            "train_steps": args.steps if not args.checkpoint else checkpoint_metadata.get("step"),
            "train_loop": args.train_loop if not args.checkpoint else checkpoint_metadata.get("train_loop"),
            "model": f"openmythos-distill-{args.model_size}",
            "train_stages": args.train_stages,
        },
    )

    write_jsonl(args.scores_out, rows)
    report = build_report(
        args=args,
        rows=rows,
        split_pairs=eval_pairs,
        device=device,
        checkpoint_metadata=checkpoint_metadata,
        elapsed_seconds=time.perf_counter() - started,
    )
    write_json(args.report_out, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


def resolve_device(selection: str, require_cuda: bool) -> torch.device:
    if selection == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(selection)

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was required but is not visible. Run Docker with `--gpus all`."
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but torch cannot see a CUDA device.")
    return device


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "prompt" not in row:
                raise ValueError(f"{path}: row missing prompt")
            if "response" not in row and "final_answer" not in row:
                raise ValueError(f"{path}: row missing response/final_answer")
            records.append(row)
    if not records:
        raise ValueError(f"No training records found in {path}")
    return records


def build_openmythos_config(
    vocab_size: int,
    seq_len: int,
    loops: int,
    args: argparse.Namespace,
) -> MythosConfig:
    presets = {
        "tiny": {
            "dim": 64,
            "n_heads": 4,
            "n_kv_heads": 2,
            "prelude_layers": 1,
            "coda_layers": 1,
            "n_experts": 8,
            "n_shared_experts": 1,
            "n_experts_per_tok": 2,
            "expert_dim": 64,
            "lora_rank": 4,
        },
        "small": {
            "dim": 128,
            "n_heads": 4,
            "n_kv_heads": 2,
            "prelude_layers": 1,
            "coda_layers": 1,
            "n_experts": 12,
            "n_shared_experts": 1,
            "n_experts_per_tok": 2,
            "expert_dim": 128,
            "lora_rank": 8,
        },
        "medium": {
            "dim": 256,
            "n_heads": 8,
            "n_kv_heads": 2,
            "prelude_layers": 2,
            "coda_layers": 2,
            "n_experts": 16,
            "n_shared_experts": 2,
            "n_experts_per_tok": 2,
            "expert_dim": 256,
            "lora_rank": 8,
        },
        "full": {
            "dim": 512,
            "n_heads": 8,
            "n_kv_heads": 4,
            "prelude_layers": 2,
            "coda_layers": 2,
            "n_experts": 32,
            "n_shared_experts": 2,
            "n_experts_per_tok": 4,
            "expert_dim": 512,
            "lora_rank": 16,
        },
    }
    cfg = presets[args.model_size].copy()
    overrides = {
        "dim": args.dim,
        "n_heads": args.heads,
        "n_kv_heads": args.kv_heads,
        "n_experts": args.experts,
        "expert_dim": args.expert_dim,
        "prelude_layers": args.prelude_layers,
        "coda_layers": args.coda_layers,
    }
    cfg.update({key: value for key, value in overrides.items() if value is not None})
    if cfg["dim"] % cfg["n_heads"] != 0:
        raise ValueError("dim must be divisible by n_heads")
    if cfg["n_heads"] % cfg["n_kv_heads"] != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")

    return MythosConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        max_loop_iters=loops,
        attn_type="gqa",
        act_threshold=0.95,
        rope_theta=10000.0,
        dropout=0.0,
        **cfg,
    )


def split_dataset(
    pairs: list[dict[str, Any]],
    seed: int,
    holdout_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    indexed = [{**pair, "_index": index} for index, pair in enumerate(pairs)]
    if any(pair.get("split") for pair in indexed):
        train = [pair for pair in indexed if pair.get("split", "train") != "holdout"]
        holdout = [pair for pair in indexed if pair.get("split") == "holdout"]
        return {
            "train": sorted(train, key=lambda row: row["_index"]),
            "holdout": sorted(holdout, key=lambda row: row["_index"]),
        }

    by_difficulty: dict[str, list[dict[str, str]]] = {"easy": [], "medium": [], "hard": []}
    for pair in indexed:
        by_difficulty[infer_difficulty(pair["prompt"])].append(pair)

    train: list[dict[str, str]] = []
    holdout: list[dict[str, str]] = []
    for items in by_difficulty.values():
        rng.shuffle(items)
        if not items:
            continue
        holdout_count = max(1, round(len(items) * holdout_ratio))
        holdout.extend(items[:holdout_count])
        train.extend(items[holdout_count:])

    if not train and holdout:
        train.append(holdout.pop())
    if len(holdout) < min(3, len(indexed)) and len(train) > 1:
        holdout.append(train.pop())

    return {
        "train": sorted(train, key=lambda row: row["_index"]),
        "holdout": sorted(holdout, key=lambda row: row["_index"]),
    }


def expand_training_pairs(
    records: list[dict[str, Any]],
    train_stages: bool,
) -> list[dict[str, Any]]:
    pairs = []
    for record in records:
        final = response_for(record)
        pairs.append({**record, "response": final})
        if not train_stages:
            continue
        for stage in record.get("stages", []) or []:
            stage_prompt = (
                f"{record['prompt']}\n\n"
                f"OpenMythos refinement loop {stage['loop']} ({stage['label']}): "
                "produce only this stage target."
            )
            pairs.append(
                {
                    **record,
                    "prompt": stage_prompt,
                    "response": stage["target"],
                    "_stage_loop": stage["loop"],
                    "_stage_label": stage["label"],
                }
            )
    return pairs


def normalize_eval_pairs(
    records: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected = records[:limit] if limit else records
    return [{**record, "response": response_for(record)} for record in selected]


def response_for(record: dict[str, Any]) -> str:
    return str(record.get("response") or record.get("final_answer") or "")


def train_model(
    model: OpenMythos,
    tokenizer: ByteTokenizer,
    pairs: list[dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    rng = random.Random(args.seed)
    examples = [make_supervised_example(tokenizer, pair, args.seq_len) for pair in pairs]
    if not examples:
        raise ValueError("No training examples were available after split/stage expansion.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    model.train()
    metrics_path = args.artifact_dir / "training_metrics.jsonl"
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        metrics_path.unlink()

    last_loss = None
    for step in range(1, args.steps + 1):
        lr = lr_at_step(step, args.steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(max(1, args.grad_accum_steps)):
            x, y = make_batch(examples, args.batch_size, rng, device)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(x, n_loops=args.train_loop)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-100,
                ) / max(1, args.grad_accum_steps)
            total_loss += float(loss.detach().cpu())
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        last_loss = total_loss

        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            record = {
                "step": step,
                "loss": last_loss,
                "lr": lr,
                "grad_norm": float(grad_norm.detach().cpu()),
                "device": str(device),
                "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "model_size": args.model_size,
                "effective_batch_size": args.batch_size * max(1, args.grad_accum_steps),
            }
            append_jsonl(metrics_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)

    return {
        "step": args.steps,
        "loss": last_loss,
        "train_loop": args.train_loop,
        "train_pairs": len(pairs),
        "model_size": args.model_size,
        "effective_batch_size": args.batch_size * max(1, args.grad_accum_steps),
    }


def load_checkpoint(path: Path, device: torch.device) -> tuple[OpenMythos, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg_keys = {field.name for field in fields(MythosConfig)}
    cfg = MythosConfig(**{key: value for key, value in checkpoint["cfg"].items() if key in cfg_keys})
    model = OpenMythos(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metadata = {
        "step": checkpoint.get("step"),
        "loss": checkpoint.get("loss"),
        "train_loop": checkpoint.get("training_args", {}).get("n_loops"),
        "training_args": checkpoint.get("training_args", {}),
    }
    return model, metadata


def save_checkpoint(
    model: OpenMythos,
    tokenizer: ByteTokenizer,
    args: argparse.Namespace,
    artifact_dir: Path,
    metadata: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "cfg": asdict(model.cfg),
        "tokenizer": tokenizer.to_dict(),
        "step": metadata.get("step"),
        "loss": metadata.get("loss"),
        "training_args": {
            "data": str(args.data),
            "seq_len": args.seq_len,
            "n_loops": args.train_loop,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "model_size": args.model_size,
            "grad_accum_steps": args.grad_accum_steps,
            "train_stages": args.train_stages,
        },
    }
    torch.save(checkpoint, artifact_dir / "checkpoint.pt")
    write_json(artifact_dir / "config.json", asdict(model.cfg))
    write_json(artifact_dir / "tokenizer.json", tokenizer.to_dict())


@torch.no_grad()
def export_scores(
    model: OpenMythos,
    tokenizer: ByteTokenizer,
    split_pairs: dict[str, list[dict[str, str]]],
    seq_len: int,
    eval_loops: Iterable[int],
    device: torch.device,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []

    for split, pairs in split_pairs.items():
        for pair in pairs:
            example = make_supervised_example(tokenizer, pair, seq_len)
            input_ids, labels = example
            x = input_ids.unsqueeze(0).to(device)
            y = labels.unsqueeze(0).to(device)
            previous_ce = None

            for loop in eval_loops:
                logits = model(x, n_loops=loop)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-100,
                )
                ce_value = float(ce.detach().cpu())
                row = {
                    "task_id": pair.get("task_id") or task_id(pair),
                    "split": split,
                    "difficulty": pair.get("difficulty") or infer_difficulty(pair["prompt"]),
                    "loop": loop,
                    "cross_entropy": ce_value,
                    "category": pair.get("category") or infer_category(pair["prompt"]),
                    "agent_id": "code_primary",
                    "metadata": {
                        **metadata,
                        **pair.get("metadata", {}),
                        "teacher_pair_index": pair.get("_index"),
                        "valid_target_tokens": int((labels != -100).sum().item()),
                    },
                }
                if previous_ce is not None:
                    row["refinement_loss"] = abs(ce_value - previous_ce)
                rows.append(row)
                previous_ce = ce_value

    return rows


def build_report(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    split_pairs: dict[str, list[dict[str, str]]],
    device: torch.device,
    checkpoint_metadata: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "scores_out": str(args.scores_out),
        "report_out": str(args.report_out),
        "artifact_dir": str(args.artifact_dir),
        "data": str(args.data),
        "rows": len(rows),
        "splits": {split: len(pairs) for split, pairs in split_pairs.items()},
        "eval_loops": sorted(set(args.eval_loops)),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint else str(args.artifact_dir / "checkpoint.pt"),
        "checkpoint_metadata": checkpoint_metadata,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "source": "openmythos_ce_export",
    }


def infer_category(prompt: str) -> str:
    text = prompt.lower()
    if "pytest" in text or "test" in text:
        return "test_writing"
    if "fix" in text or "bug" in text or "keyerror" in text:
        return "debugging"
    if "refactor" in text:
        return "refactoring"
    if "explain" in text:
        return "coding"
    if "sql" in text or "api" in text:
        return "api"
    return "code_generation"


def infer_difficulty(prompt: str) -> str:
    text = prompt.lower()
    hard_markers = {
        "thread-safe",
        "concurrent",
        "balanced",
        "database",
        "authentication",
        "rate limiter",
        "async",
    }
    medium_markers = {
        "debounce",
        "decorator",
        "retry",
        "pytest",
        "refactor",
        "keyerror",
        "sql",
        "api",
    }
    if any(marker in text for marker in hard_markers) or len(prompt) > 180:
        return "hard"
    if any(marker in text for marker in medium_markers) or len(prompt) > 80:
        return "medium"
    return "easy"


def task_id(pair: dict[str, str]) -> str:
    index = pair.get("_index", 0)
    slug = "".join(
        char if char.isalnum() else "_"
        for char in pair["prompt"].lower().strip()[:48]
    ).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"teacher_{index:03d}_{slug or 'task'}"


def make_supervised_example(
    tokenizer: ByteTokenizer,
    pair: dict[str, Any],
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a causal LM example while preserving answer tokens.

    The original Docker trainer truncates from the right after concatenating
    prompt and answer. That works for tiny prompts, but scaled staged prompts
    can consume the full byte-level context and leave every label masked. This
    variant keeps the assistant/prompt tail and reserves at least half the
    window for supervised answer bytes.
    """

    prefix = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{pair['prompt']}\n"
        "<|assistant|>\n"
    )
    response = response_for(pair) + "\n<|end|>\n"
    max_ids = seq_len + 1

    prefix_body = tokenizer.encode(prefix, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]

    min_prefix_budget = min(len(prefix_body) + 1, max(16, max_ids // 4))
    response_budget = max(1, max_ids - min_prefix_budget)
    if len(response_ids) > response_budget:
        response_ids = response_ids[:response_budget]

    prefix_budget = max_ids - len(response_ids) - 1
    prefix_body = prefix_body[-max(0, prefix_budget):]
    ids = [tokenizer.bos_token_id, *prefix_body, *response_ids]

    if len(ids) > max_ids:
        ids = ids[-max_ids:]
    pad_len = max_ids - len(ids)
    ids = ids + [tokenizer.pad_token_id] * pad_len

    input_ids = torch.tensor(ids[:-1], dtype=torch.long)
    labels = torch.tensor(ids[1:], dtype=torch.long)
    labels[input_ids == tokenizer.pad_token_id] = -100

    prefix_len = min(1 + len(prefix_body), seq_len)
    labels[: max(0, prefix_len - 1)] = -100
    if int((labels != -100).sum().item()) == 0:
        raise ValueError(
            f"No supervised target tokens survived truncation for task {task_id(pair)}; "
            f"increase --seq-len or shorten prompts."
        )
    return input_ids, labels


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
