"""Run OpenMythos recurrent-refinement diagnostics from JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation.openmythos import (
    OpenMythosAgentOptimizer,
    OpenMythosLoopDiagnostics,
    load_loop_scores_jsonl,
    load_staged_examples_jsonl,
    validate_staged_dataset,
)
from app.schemas.registry import AgentRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose OpenMythos loop-depth generalization and dataset readiness."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        help="JSONL with task_id, split, difficulty, loop, cross_entropy, and optional refinement_loss.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="JSONL staged-refinement dataset to validate.",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=1000,
        help="Minimum dataset size before recurrence conclusions are considered meaningful.",
    )
    parser.add_argument(
        "--target-examples",
        type=int,
        default=5000,
        help="Preferred dataset size for robust recurrence experiments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional agent_registry.json path for promotion/subagent recommendations.",
    )
    parser.add_argument(
        "--apply-promotions",
        action="store_true",
        help="Apply recommended lifecycle promotions to --registry.",
    )
    parser.add_argument(
        "--create-subagents",
        action="store_true",
        help="Create recommended probationary subagents in --registry.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.scores and not args.dataset:
        raise SystemExit("Provide --scores, --dataset, or both.")

    report: Dict[str, Any] = {}

    if args.dataset:
        examples = load_staged_examples_jsonl(args.dataset)
        dataset_report = validate_staged_dataset(
            examples,
            min_examples=args.min_examples,
            target_examples=args.target_examples,
        )
        report["dataset_readiness"] = dataset_report.to_dict()

    if args.scores:
        scores = load_loop_scores_jsonl(args.scores)
        diagnostics = OpenMythosLoopDiagnostics(scores)
        report["loop_diagnostics"] = diagnostics.report()

        if args.registry:
            registry = _load_registry(args.registry)
            optimizer = OpenMythosAgentOptimizer(diagnostics, registry)
            recommendations = optimizer.recommend()
            report["agent_optimization"] = {
                "recommendations": [
                    recommendation.to_dict() for recommendation in recommendations
                ]
            }

            registry_changed = False
            if args.apply_promotions:
                applied = optimizer.apply_promotions()
                report["agent_optimization"]["applied_promotions"] = [
                    recommendation.to_dict() for recommendation in applied
                ]
                registry_changed = bool(applied)

            if args.create_subagents:
                created = optimizer.create_recommended_subagents()
                report["agent_optimization"]["created_subagents"] = [
                    agent.model_dump() for agent in created
                ]
                registry_changed = registry_changed or bool(created)

            if registry_changed:
                _save_registry(args.registry, registry)

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    return 0


def _load_registry(path: Path) -> AgentRegistry:
    return AgentRegistry(**json.loads(path.read_text(encoding="utf-8-sig")))


def _save_registry(path: Path, registry: AgentRegistry) -> None:
    path.write_text(
        json.dumps(registry.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
