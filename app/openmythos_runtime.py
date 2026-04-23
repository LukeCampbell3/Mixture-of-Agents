"""Runtime integration for OpenMythos adaptation.

This module bridges the offline OpenMythos diagnostics with the live
orchestrator used by claude_integrated.py.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.evaluation.openmythos import (
    OpenMythosAgentOptimizer,
    OpenMythosLoopDiagnostics,
    load_loop_scores_jsonl,
)
from app.evaluation.realistic_prompts import RealisticPromptDataset


@dataclass(frozen=True)
class PromptValidationResult:
    """Result from validating one pre-existing prompt."""

    prompt_id: str
    category: str
    complexity: str
    success: bool
    final_state: str
    active_agents: List[str]
    spawned_agents: List[str]
    promoted_agents: List[str]
    created_files: List[str]
    pending_file_ops: List[str]
    elapsed_seconds: float


class OpenMythosRuntimeAdapter:
    """Apply OpenMythos recommendations to a live orchestrator."""

    def __init__(
        self,
        orchestrator,
        data_dir: str = "data",
        scores_filename: str = "openmythos_loop_scores.jsonl",
    ):
        self.orchestrator = orchestrator
        self.data_dir = Path(data_dir)
        self.scores_path = self.data_dir / scores_filename
        self.validation_path = self.data_dir / "openmythos_prompt_validation.jsonl"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> Dict[str, Any]:
        """Return current OpenMythos runtime status."""

        registry = self.orchestrator.registry
        optimized = [
            agent.agent_id
            for agent in registry.agents.values()
            if "openmythos_optimized" in agent.tags
        ]
        subagents = [
            agent.agent_id
            for agent in registry.agents.values()
            if "openmythos" in agent.tags and agent.parent_lineage
        ]
        return {
            "scores_path": str(self.scores_path),
            "scores_available": self.scores_path.exists(),
            "validation_path": str(self.validation_path),
            "agent_count": len(registry.agents),
            "optimized_agents": optimized,
            "openmythos_subagents": subagents,
        }

    def adapt(
        self,
        scores_path: Optional[str | Path] = None,
        apply_promotions: bool = True,
        create_subagents: bool = True,
        save_registry: bool = True,
    ) -> Dict[str, Any]:
        """Apply OpenMythos score-based recommendations to the registry."""

        path = Path(scores_path) if scores_path else self.scores_path
        if not path.exists():
            return {
                "adapted": False,
                "reason": f"No OpenMythos score file found at {path}",
                "recommendations": [],
                "applied_promotions": [],
                "created_subagents": [],
            }

        scores = load_loop_scores_jsonl(path)
        diagnostics = OpenMythosLoopDiagnostics(scores)
        optimizer = OpenMythosAgentOptimizer(diagnostics, self.orchestrator.registry)

        recommendations = optimizer.recommend()
        applied = optimizer.apply_promotions() if apply_promotions else []
        created = optimizer.create_recommended_subagents() if create_subagents else []

        if save_registry and (applied or created):
            self.orchestrator.registry_store.save_registry(self.orchestrator.registry)

        return {
            "adapted": bool(applied or created),
            "scores_path": str(path),
            "recommendations": [item.to_dict() for item in recommendations],
            "applied_promotions": [item.to_dict() for item in applied],
            "created_subagents": [agent.model_dump() for agent in created],
        }

    def validate_existing_prompts(
        self,
        sample_size: int = 6,
        include_mixed: bool = True,
        workspace_root: str = ".",
    ) -> Dict[str, Any]:
        """Validate coding behavior using the repo's existing prompt dataset."""

        dataset = RealisticPromptDataset()
        categories = {"coding", "mixed"} if include_mixed else {"coding"}
        prompts = [
            prompt
            for prompt in dataset.get_all_prompts()
            if prompt.category in categories
        ][:sample_size]

        results = []
        for prompt in prompts:
            started = datetime.utcnow()
            run_state = self.orchestrator.run_task(
                prompt.text,
                workspace_root=workspace_root,
                conversation_history=(
                    "OPENMYTHOS VALIDATION MODE:\n"
                    "Prioritize complete coding work, tests, and durable file outputs when useful."
                ),
            )
            elapsed = (datetime.utcnow() - started).total_seconds()
            result = PromptValidationResult(
                prompt_id=prompt.prompt_id,
                category=prompt.category,
                complexity=prompt.complexity.value,
                success=run_state.final_state == "success",
                final_state=run_state.final_state,
                active_agents=run_state.active_agents,
                spawned_agents=run_state.spawned_agents,
                promoted_agents=run_state.promoted_agents,
                created_files=self._created_files_from_run(run_state),
                pending_file_ops=self._pending_file_ops_from_run(run_state),
                elapsed_seconds=elapsed,
            )
            results.append(result)
            self._append_validation_record(result, run_state)

        return self._summarize_validation(results)

    def compare_against_baseline(
        self,
        baseline_orchestrator,
        sample_size: int = 6,
        include_mixed: bool = True,
        workspace_root: str = ".",
    ) -> Dict[str, Any]:
        """Run the same existing prompts against baseline and OpenMythos."""

        baseline_adapter = OpenMythosRuntimeAdapter(
            baseline_orchestrator,
            data_dir=str(self.data_dir),
            scores_filename=self.scores_path.name,
        )

        baseline = baseline_adapter.validate_existing_prompts(
            sample_size=sample_size,
            include_mixed=include_mixed,
            workspace_root=workspace_root,
        )
        adapted = self.validate_existing_prompts(
            sample_size=sample_size,
            include_mixed=include_mixed,
            workspace_root=workspace_root,
        )

        baseline_score = self._effectiveness_score(baseline)
        adapted_score = self._effectiveness_score(adapted)
        return {
            "baseline": baseline,
            "openmythos": adapted,
            "baseline_effectiveness_score": baseline_score,
            "openmythos_effectiveness_score": adapted_score,
            "score_delta": round(adapted_score - baseline_score, 4),
            "openmythos_better": adapted_score > baseline_score,
            "verdict": (
                "improved"
                if adapted_score > baseline_score
                else "not_yet_improved"
            ),
        }

    def _append_validation_record(self, result: PromptValidationResult, run_state) -> None:
        record = {
            **asdict(result),
            "timestamp": datetime.utcnow().isoformat(),
            "task_id": run_state.task_id,
            "pool_size_before": run_state.pool_size_before,
            "pool_size_after": run_state.pool_size_after,
            "lifecycle_events": run_state.lifecycle_events,
        }
        with self.validation_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _summarize_validation(results: List[PromptValidationResult]) -> Dict[str, Any]:
        if not results:
            return {
                "total_prompts": 0,
                "success_rate": 0.0,
                "code_output_rate": 0.0,
                "specialist_usage_rate": 0.0,
                "lifecycle_activity_rate": 0.0,
                "agents_used": [],
                "spawned_agents": [],
                "promoted_agents": [],
                "created_files": [],
                "pending_file_ops": [],
                "long_term_code_creation": "no_prompts",
                "results": [],
            }

        agents_used = sorted({agent for row in results for agent in row.active_agents})
        spawned_agents = sorted({agent for row in results for agent in row.spawned_agents})
        promoted_agents = sorted({agent for row in results for agent in row.promoted_agents})
        created_files = sorted({path for row in results for path in row.created_files})
        pending_file_ops = sorted({path for row in results for path in row.pending_file_ops})
        success_count = sum(1 for row in results if row.success)
        code_output_count = sum(
            1 for row in results if row.created_files or row.pending_file_ops
        )
        specialist_count = sum(
            1
            for row in results
            if any(agent not in {"code_primary", "web_research", "critic_verifier"} for agent in row.active_agents)
        )
        lifecycle_count = sum(
            1 for row in results if row.spawned_agents or row.promoted_agents
        )

        return {
            "total_prompts": len(results),
            "success_rate": success_count / len(results),
            "code_output_rate": code_output_count / len(results),
            "specialist_usage_rate": specialist_count / len(results),
            "lifecycle_activity_rate": lifecycle_count / len(results),
            "agents_used": agents_used,
            "spawned_agents": spawned_agents,
            "promoted_agents": promoted_agents,
            "created_files": created_files,
            "pending_file_ops": pending_file_ops,
            "long_term_code_creation": (
                "file_outputs_detected" if created_files or pending_file_ops else "no_file_outputs"
            ),
            "results": [asdict(row) for row in results],
        }

    @staticmethod
    def _effectiveness_score(summary: Dict[str, Any]) -> float:
        """Single comparison score for coding-task orchestration."""

        return round(
            0.55 * summary.get("success_rate", 0.0)
            + 0.25 * summary.get("code_output_rate", 0.0)
            + 0.10 * summary.get("specialist_usage_rate", 0.0)
            + 0.10 * summary.get("lifecycle_activity_rate", 0.0),
            4,
        )

    @staticmethod
    def _pending_file_ops_from_run(run_state) -> List[str]:
        paths = []
        for call in getattr(run_state, "pending_tool_calls", []) or []:
            if isinstance(call, dict):
                path = call.get("path") or call.get("file_path")
                if path:
                    paths.append(path)
        return paths

    @classmethod
    def _created_files_from_run(cls, run_state) -> List[str]:
        files = list(getattr(run_state, "final_files", []) or [])
        for path in cls._pending_file_ops_from_run(run_state):
            if path not in files:
                files.append(path)
        return files
