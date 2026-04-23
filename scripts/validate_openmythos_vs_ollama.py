"""Validate old Ollama orchestration against OpenMythos adaptation.

This is intentionally a live-model validator. It does not claim improvement
from stubs; it checks Ollama availability, applies OpenMythos score-based
adaptation, runs both systems on the same existing coding prompts, and writes
a comparison report.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.openmythos_runtime import OpenMythosRuntimeAdapter
from app.orchestrator import Orchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original Ollama orchestrator against OpenMythos-adapted orchestrator."
    )
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--router-model", default="qwen2.5:0.5b")
    parser.add_argument("--scores", default="data/openmythos_loop_scores.jsonl")
    parser.add_argument("--sample-size", type=int, default=6)
    parser.add_argument("--output", default="data/reports/openmythos_vs_ollama_report.json")
    parser.add_argument("--max-tokens", type=int, default=800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prereq = check_prerequisites(args.base_url, args.model, args.router_model, Path(args.scores))
    if not prereq["ready"]:
        report = {
            "validated": False,
            "verdict": "blocked",
            "reason": "Live Ollama/OpenMythos prerequisites are not satisfied.",
            "prerequisites": prereq,
        }
        write_report(output_path, report)
        print(json.dumps(report, indent=2))
        return 2

    with tempfile.TemporaryDirectory(prefix="ollama_baseline_") as baseline_dir, tempfile.TemporaryDirectory(
        prefix="openmythos_adapted_"
    ) as adapted_dir:
        baseline = make_orchestrator(args, baseline_dir)
        adapted = make_orchestrator(args, adapted_dir)
        adapter = OpenMythosRuntimeAdapter(adapted, data_dir=adapted_dir)

        adaptation = adapter.adapt(scores_path=args.scores)
        comparison = adapter.compare_against_baseline(
            baseline,
            sample_size=args.sample_size,
        )

    report = {
        "validated": True,
        "verdict": comparison["verdict"],
        "openmythos_better": comparison["openmythos_better"],
        "adaptation": adaptation,
        "comparison": comparison,
    }
    write_report(output_path, report)
    print(json.dumps(report, indent=2))
    return 0 if comparison["openmythos_better"] else 1


def make_orchestrator(args: argparse.Namespace, data_dir: str) -> Orchestrator:
    return Orchestrator(
        llm_provider="ollama",
        llm_model=args.model,
        llm_base_url=args.base_url,
        router_model=args.router_model,
        budget_mode="balanced",
        data_dir=data_dir,
        enable_parallel=True,
        max_parallel_agents=3,
        max_tokens=args.max_tokens,
        auto_approve_file_ops=False,
    )


def check_prerequisites(
    base_url: str,
    model: str,
    router_model: str,
    scores_path: Path,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "router_model": router_model,
        "scores_path": str(scores_path),
        "ollama_http_available": False,
        "model_available": False,
        "router_model_available": False,
        "scores_available": scores_path.exists(),
        "errors": [],
    }

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        checks["ollama_http_available"] = True
        models = [item.get("name", "") for item in response.json().get("models", [])]
        checks["available_models"] = models
        checks["model_available"] = model in models or any(name.startswith(model) for name in models)
        checks["router_model_available"] = router_model in models or any(
            name.startswith(router_model) for name in models
        )
    except Exception as exc:
        checks["errors"].append(f"Ollama HTTP unavailable: {exc}")

    if not checks["scores_available"]:
        checks["errors"].append(
            "OpenMythos score file is missing; run training/evaluation to produce loop scores first."
        )
    if checks["ollama_http_available"] and not checks["model_available"]:
        checks["errors"].append(f"Worker model not available in Ollama: {model}")
    if checks["ollama_http_available"] and not checks["router_model_available"]:
        checks["errors"].append(f"Router model not available in Ollama: {router_model}")

    checks["ready"] = (
        checks["ollama_http_available"]
        and checks["model_available"]
        and checks["router_model_available"]
        and checks["scores_available"]
    )
    return checks


def write_report(path: Path, report: Dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
