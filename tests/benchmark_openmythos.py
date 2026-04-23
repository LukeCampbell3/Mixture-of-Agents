"""
OpenMythos+Ollama vs plain Ollama benchmark.

Tests the correct integration: OpenMythos as a reasoning pre-processor
that enriches prompts before Ollama generates the final response.

Pipeline under test:
  Plain Ollama:  prompt → Ollama → response
  OpenMythos:    prompt → RDT(n_loops) → enriched_prompt → Ollama → response

Run:
    python tests/benchmark_openmythos.py
    python tests/benchmark_openmythos.py --loops 1 2 4 8
    python tests/benchmark_openmythos.py --ollama-only
    python tests/benchmark_openmythos.py --artifact poc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

# ── Tasks ─────────────────────────────────────────────────────────────────────

TASKS = [
    {
        "id":      "binary_search",
        "prompt":  "Write a complete binary search function in Python with edge case handling.",
        "signals": ["def", "while", "mid", "return", "low", "high", "not found"],
    },
    {
        "id":      "fizzbuzz",
        "prompt":  "Write FizzBuzz in Python for 1 to 20.",
        "signals": ["fizz", "buzz", "for", "range", "print", "if"],
    },
    {
        "id":      "linked_list",
        "prompt":  "Implement a singly linked list in Python with insert and search.",
        "signals": ["class Node", "class", "def insert", "def search", "self.next", "head"],
    },
    {
        "id":      "explain_async",
        "prompt":  "Explain async/await in Python with a concrete example.",
        "signals": ["async", "await", "asyncio", "coroutine", "event loop", "def"],
    },
    {
        "id":      "graph_bfs",
        "prompt":  "Implement breadth-first search for a graph in Python.",
        "signals": ["def bfs", "queue", "deque", "visited", "neighbors", "append"],
    },
]


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: str
    provider: str
    n_loops: Optional[int]
    total_s: float
    tok_per_sec: float
    response: str
    quality_score: float
    signals_hit: list
    reasoning_chars: int = 0
    error: str = ""


@dataclass
class BenchmarkReport:
    timestamp: str
    results: list = field(default_factory=list)

    def add(self, r: TaskResult):
        self.results.append(r)

    def table(self) -> str:
        lines = [
            f"\n{'─'*95}",
            f"{'Provider':<38} {'Task':<16} {'Quality':>8} {'Tok/s':>7} {'Time(s)':>8} {'Loops':>6} {'Reason':>7}",
            f"{'─'*95}",
        ]
        for r in sorted(self.results, key=lambda x: (x.provider, x.task_id)):
            loops = str(r.n_loops) if r.n_loops is not None else "N/A"
            reason = f"{r.reasoning_chars}c" if r.reasoning_chars else "-"
            err = f"  ERR: {r.error[:20]}" if r.error else ""
            lines.append(
                f"{r.provider:<38} {r.task_id:<16} {r.quality_score:>7.0%} "
                f"{r.tok_per_sec:>7.1f} {r.total_s:>8.2f} {loops:>6} {reason:>7}{err}"
            )
        lines.append(f"{'─'*95}")

        # Per-provider averages
        providers = sorted(set(r.provider for r in self.results))
        lines.append("\nAverages:")
        for p in providers:
            pr = [r for r in self.results if r.provider == p and not r.error]
            if not pr:
                continue
            aq  = sum(r.quality_score for r in pr) / len(pr)
            at  = sum(r.tok_per_sec   for r in pr) / len(pr)
            as_ = sum(r.total_s       for r in pr) / len(pr)
            lines.append(f"  {p:<38} quality={aq:.0%}  tok/s={at:.1f}  avg_time={as_:.2f}s")

        # Loop scaling
        mythos = [r for r in self.results if "openmythos" in r.provider and not r.error]
        loop_counts = sorted(set(r.n_loops for r in mythos if r.n_loops))
        if len(loop_counts) > 1:
            lines.append("\nOpenMythos loop depth scaling:")
            for lc in loop_counts:
                lt = [r for r in mythos if r.n_loops == lc]
                aq  = sum(r.quality_score for r in lt) / len(lt)
                at  = sum(r.tok_per_sec   for r in lt) / len(lt)
                as_ = sum(r.total_s       for r in lt) / len(lt)
                lines.append(
                    f"  loops={lc:<3}  quality={aq:.0%}  tok/s={at:.1f}  avg_time={as_:.2f}s"
                )

        return "\n".join(lines)

    def verdict(self) -> str:
        mythos = [r for r in self.results if "openmythos" in r.provider and not r.error]
        ollama = [r for r in self.results if r.provider.startswith("ollama") and not r.error]
        if not (mythos and ollama):
            return ""

        aq_m = sum(r.quality_score for r in mythos) / len(mythos)
        aq_o = sum(r.quality_score for r in ollama) / len(ollama)
        at_m = sum(r.total_s       for r in mythos) / len(mythos)
        at_o = sum(r.total_s       for r in ollama) / len(ollama)
        q_delta = aq_m - aq_o
        t_delta = at_m - at_o

        lines = [
            "\n\033[1mVerdict — OpenMythos+Ollama vs plain Ollama:\033[0m",
            f"  Quality  Ollama: {aq_o:.0%}  →  OpenMythos+Ollama: {aq_m:.0%}  ({q_delta:+.0%})",
            f"  Latency  Ollama: {at_o:.2f}s  →  OpenMythos+Ollama: {at_m:.2f}s  ({t_delta:+.2f}s)",
        ]

        if q_delta > 0.05:
            lines.append(
                f"  \033[32m✓ OpenMythos reasoning layer improved quality by {q_delta:+.0%}\033[0m"
            )
        elif q_delta >= -0.05:
            lines.append(
                "  \033[33m~ Quality is similar — reasoning layer adds context without hurting output\033[0m"
            )
        else:
            lines.append(
                f"  \033[31m✗ Quality dropped {q_delta:.0%} — reasoning context may be noisy\033[0m"
            )

        if t_delta < 2.0:
            lines.append(
                f"  \033[32m✓ Latency overhead is acceptable ({t_delta:+.2f}s)\033[0m"
            )
        else:
            lines.append(
                f"  \033[33m~ Latency overhead: {t_delta:+.2f}s (RDT reasoning adds compute)\033[0m"
            )

        lines.append(
            "\n  Architecture note: OpenMythos RDT runs n_loops of latent reasoning"
        )
        lines.append(
            "  before Ollama generates. More loops = deeper reasoning, more latency."
        )
        lines.append(
            "  The quality gain scales with task complexity (proj.txt Stage 3)."
        )

        return "\n".join(lines)


# ── Quality scorer ────────────────────────────────────────────────────────────

def score(response: str, signals: list) -> tuple:
    if not response or len(response.strip()) < 10:
        return 0.0, []
    text = response.lower()
    found = [s for s in signals if s.lower() in text]
    base = len(found) / len(signals) if signals else 0.5
    if len(response.strip()) > 50:
        base = min(1.0, base + 0.05)
    return round(base, 3), found


# ── Runners ───────────────────────────────────────────────────────────────────

def run_task(client, task: dict, n_loops: Optional[int] = None) -> TaskResult:
    if n_loops is not None and hasattr(client, "set_loops"):
        client.set_loops(n_loops)

    t0 = time.perf_counter()
    error = ""
    response = ""
    try:
        response = client.generate(task["prompt"], max_tokens=400, temperature=0.3)
    except Exception as e:
        error = str(e)[:150]
    elapsed = time.perf_counter() - t0

    metrics = getattr(client, "last_metrics", lambda: {})()
    tps = metrics.get("tok_per_sec", len(response.split()) / elapsed if elapsed > 0 else 0)
    loops = n_loops or metrics.get("n_loops")
    reasoning_chars = metrics.get("reasoning_chars", 0)

    q, found = score(response, task["signals"])
    return TaskResult(
        task_id=task["id"],
        provider=client.get_model_name(),
        n_loops=loops,
        total_s=round(elapsed, 3),
        tok_per_sec=round(tps, 1),
        response=response[:400],
        quality_score=q,
        signals_hit=found,
        reasoning_chars=reasoning_chars,
        error=error,
    )


def benchmark_ollama(report: BenchmarkReport, tasks: list):
    print("\n\033[1mBenchmarking plain Ollama (baseline)...\033[0m")
    try:
        import requests as _req
        r = _req.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        if not models:
            print("  ⚠ No models pulled — skipping")
            return
        model = models[0]
        print(f"  Model: {model}")
    except Exception as e:
        print(f"  ⚠ Ollama not available: {e}")
        return

    from app.models.local_llm_client import OllamaClient
    client = OllamaClient(model=model)

    for task in tasks:
        print(f"  {task['id']}...", end=" ", flush=True)
        r = run_task(client, task)
        report.add(r)
        status = "✓" if not r.error else "✗"
        print(f"{status} quality={r.quality_score:.0%} tok/s={r.tok_per_sec:.1f} ({r.total_s:.1f}s)")
        if r.response and not r.error:
            print(f"    → {r.response[:80].replace(chr(10), ' ')}")


def benchmark_openmythos(
    report: BenchmarkReport,
    tasks: list,
    loop_counts: list,
    artifact: str = "gpu",
):
    print(f"\n\033[1mBenchmarking OpenMythos+Ollama ({artifact} checkpoint)...\033[0m")
    print("  Pipeline: prompt → RDT reasoning → enriched_prompt → Ollama → response")

    from app.models.openmythos_client import OpenMythosClient, OpenMythosConfig
    from app.models.openmythos_client import ARTIFACTS_GPU, ARTIFACTS_POC, ARTIFACTS_LARGE

    artifact_map = {"gpu": ARTIFACTS_GPU, "poc": ARTIFACTS_POC, "large": ARTIFACTS_LARGE}
    artifact_dir = artifact_map.get(artifact, ARTIFACTS_LARGE)

    # Get Ollama model
    try:
        import requests as _req
        r = _req.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        ollama_model = models[0] if models else "qwen2.5:1.5b"
    except Exception:
        ollama_model = "qwen2.5:1.5b"

    cfg = OpenMythosConfig(
        artifact_dir=artifact_dir,
        n_loops=loop_counts[0],
        ollama_model=ollama_model,
    )
    client = OpenMythosClient(cfg)

    if client._loaded:
        params = sum(p.numel() for p in client._rdt.parameters())
        print(f"  RDT: dim={client._rdt.cfg.dim} params={params:,} device={client._device}")
    else:
        print(f"  ⚠ RDT not loaded: {client._load_error}")
        print("  Running with Ollama only (no reasoning enrichment)")

    print(f"  Ollama model: {ollama_model}")

    for loops in loop_counts:
        print(f"\n  Loop depth: {loops}")
        for task in tasks:
            print(f"    {task['id']}...", end=" ", flush=True)
            r = run_task(client, task, n_loops=loops)
            report.add(r)
            status = "✓" if not r.error else "✗"
            reason_info = f" reason={r.reasoning_chars}c" if r.reasoning_chars else ""
            print(f"{status} quality={r.quality_score:.0%} tok/s={r.tok_per_sec:.1f} ({r.total_s:.1f}s){reason_info}")
            if r.response and not r.error:
                print(f"      → {r.response[:80].replace(chr(10), ' ')}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenMythos+Ollama vs plain Ollama benchmark"
    )
    parser.add_argument("--loops",       nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--ollama-only", action="store_true")
    parser.add_argument("--mythos-only", action="store_true")
    parser.add_argument("--artifact",    default="large", choices=["gpu", "poc", "large"])
    parser.add_argument("--tasks",       nargs="+",
                        choices=[t["id"] for t in TASKS] + ["all"],
                        default=["all"])
    args = parser.parse_args()

    selected = TASKS if "all" in args.tasks else [t for t in TASKS if t["id"] in args.tasks]
    report = BenchmarkReport(timestamp=datetime.now(timezone.utc).isoformat())

    print("\n\033[1;34m╔══════════════════════════════════════════════════════════╗\033[0m")
    print("\033[1;34m║  OpenMythos+Ollama vs Ollama — Benchmark                  ║\033[0m")
    print("\033[1;34m╚══════════════════════════════════════════════════════════╝\033[0m")
    print(f"\nTasks: {[t['id'] for t in selected]}")
    print(f"OpenMythos artifact: {args.artifact}  loops: {args.loops}")
    print("\nCorrect integration: OpenMythos RDT enriches prompts, Ollama generates text.")

    if not args.mythos_only:
        benchmark_ollama(report, selected)

    if not args.ollama_only:
        benchmark_openmythos(report, selected, args.loops, args.artifact)

    print(report.table())
    print(report.verdict())

    out = ROOT / "data" / "benchmark_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"timestamp": report.timestamp, "results": [asdict(r) for r in report.results]},
        indent=2,
    ))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
