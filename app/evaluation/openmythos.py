"""Diagnostics for OpenMythos-style recurrent refinement experiments.

The helpers in this module are intentionally model-agnostic. They do not
train a network; they make the next experiment harder to fool by checking
dataset readiness, loop-depth generalization, and best-of-loop behavior.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from app.schemas.registry import AgentRegistry, AgentSpec, LifecycleState


DIFFICULTY_ORDER = ("easy", "medium", "hard")


@dataclass(frozen=True)
class StageTarget:
    """Supervised target for one refinement loop."""

    loop: int
    label: str
    target: str

    def validate(self) -> List[str]:
        errors = []
        if self.loop < 1:
            errors.append("stage loop must be >= 1")
        if not self.label.strip():
            errors.append("stage label is required")
        if not self.target.strip():
            errors.append("stage target is required")
        return errors


@dataclass(frozen=True)
class StagedRefinementExample:
    """One instruction-code example with optional intermediate targets."""

    task_id: str
    prompt: str
    final_answer: str
    category: str
    difficulty: str
    split: str = "development"
    stages: List[StageTarget] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        errors = []
        if not self.task_id.strip():
            errors.append("task_id is required")
        if not self.prompt.strip():
            errors.append(f"{self.task_id}: prompt is required")
        if not self.final_answer.strip():
            errors.append(f"{self.task_id}: final_answer is required")
        if self.difficulty not in DIFFICULTY_ORDER:
            errors.append(
                f"{self.task_id}: difficulty must be one of {', '.join(DIFFICULTY_ORDER)}"
            )
        seen_loops = set()
        for stage in self.stages:
            errors.extend(f"{self.task_id}: {error}" for error in stage.validate())
            if stage.loop in seen_loops:
                errors.append(f"{self.task_id}: duplicate stage loop {stage.loop}")
            seen_loops.add(stage.loop)
        return errors


@dataclass(frozen=True)
class DatasetReadinessReport:
    """Readiness summary for recurrent-refinement training data."""

    ready: bool
    total_examples: int
    staged_examples: int
    category_counts: Dict[str, int]
    difficulty_counts: Dict[str, int]
    split_counts: Dict[str, int]
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoopScore:
    """Cross-entropy and optional refinement loss for one task at one loop."""

    task_id: str
    split: str
    difficulty: str
    loop: int
    cross_entropy: float
    category: str = "unknown"
    agent_id: Optional[str] = None
    refinement_loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        errors = []
        if not self.task_id.strip():
            errors.append("task_id is required")
        if self.difficulty not in DIFFICULTY_ORDER:
            errors.append(
                f"{self.task_id}: difficulty must be one of {', '.join(DIFFICULTY_ORDER)}"
            )
        if self.loop < 1:
            errors.append(f"{self.task_id}: loop must be >= 1")
        if self.cross_entropy < 0:
            errors.append(f"{self.task_id}: cross_entropy must be >= 0")
        if self.refinement_loss is not None and self.refinement_loss < 0:
            errors.append(f"{self.task_id}: refinement_loss must be >= 0")
        return errors


def validate_staged_dataset(
    examples: Iterable[StagedRefinementExample],
    min_examples: int = 1000,
    target_examples: int = 5000,
    min_categories: int = 4,
    require_intermediate_stages: bool = True,
) -> DatasetReadinessReport:
    """Check whether a dataset can credibly test loop-specific refinement."""

    example_list = list(examples)
    errors: List[str] = []
    warnings: List[str] = []

    for example in example_list:
        errors.extend(example.validate())

    category_counts = Counter(example.category for example in example_list)
    difficulty_counts = Counter(example.difficulty for example in example_list)
    split_counts = Counter(example.split for example in example_list)
    staged_examples = sum(1 for example in example_list if example.stages)

    if len(example_list) < min_examples:
        errors.append(
            f"dataset has {len(example_list)} examples; use at least {min_examples} "
            "before treating loop-depth generalization as meaningful"
        )
    elif len(example_list) < target_examples:
        warnings.append(
            f"dataset has {len(example_list)} examples; {target_examples}+ is a better "
            "target for robust recurrence experiments"
        )

    if len(category_counts) < min_categories:
        errors.append(
            f"dataset has {len(category_counts)} categories; use at least {min_categories} "
            "diverse task families"
        )

    missing_difficulties = [
        difficulty for difficulty in DIFFICULTY_ORDER if difficulty_counts[difficulty] == 0
    ]
    if missing_difficulties:
        errors.append(f"dataset is missing difficulty buckets: {', '.join(missing_difficulties)}")

    if "holdout" not in split_counts:
        errors.append("dataset needs a holdout split")

    if require_intermediate_stages and staged_examples < len(example_list):
        warnings.append(
            f"{len(example_list) - staged_examples} examples have no intermediate stage targets; "
            "the model will have to invent staged refinement without direct supervision"
        )

    return DatasetReadinessReport(
        ready=not errors,
        total_examples=len(example_list),
        staged_examples=staged_examples,
        category_counts=dict(category_counts),
        difficulty_counts=dict(difficulty_counts),
        split_counts=dict(split_counts),
        errors=errors,
        warnings=warnings,
    )


class OpenMythosLoopDiagnostics:
    """Analyze loop-depth behavior from held-out and training CE scores."""

    def __init__(self, scores: Iterable[LoopScore]):
        self.scores = list(scores)
        errors = [error for score in self.scores for error in score.validate()]
        if errors:
            raise ValueError("; ".join(errors))

    def mean_ce_by_split_and_loop(self) -> Dict[str, Dict[int, float]]:
        grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        for score in self.scores:
            grouped[score.split][score.loop].append(score.cross_entropy)
        return {
            split: {loop: mean(values) for loop, values in sorted(loop_map.items())}
            for split, loop_map in sorted(grouped.items())
        }

    def mean_ce_by_difficulty_and_loop(self, split: Optional[str] = None) -> Dict[str, Dict[int, float]]:
        grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        for score in self._filter(split=split):
            grouped[score.difficulty][score.loop].append(score.cross_entropy)
        return {
            difficulty: {loop: mean(values) for loop, values in sorted(loop_map.items())}
            for difficulty, loop_map in sorted(grouped.items())
        }

    def best_loop_by_task(self, split: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, List[LoopScore]] = defaultdict(list)
        for score in self._filter(split=split):
            grouped[score.task_id].append(score)

        best: Dict[str, Dict[str, Any]] = {}
        for task_id, task_scores in grouped.items():
            winner = min(task_scores, key=lambda score: (score.cross_entropy, score.loop))
            first_loop = min(task_scores, key=lambda score: score.loop)
            last_loop = max(task_scores, key=lambda score: score.loop)
            best[task_id] = {
                "best_loop": winner.loop,
                "best_cross_entropy": winner.cross_entropy,
                "loop_1_cross_entropy": first_loop.cross_entropy,
                "final_loop": last_loop.loop,
                "final_cross_entropy": last_loop.cross_entropy,
                "best_vs_loop_1_delta": winner.cross_entropy - first_loop.cross_entropy,
                "final_vs_loop_1_delta": last_loop.cross_entropy - first_loop.cross_entropy,
                "difficulty": winner.difficulty,
                "category": winner.category,
            }
        return best

    def best_of_loops_summary(self, split: Optional[str] = None) -> Dict[str, Any]:
        best = self.best_loop_by_task(split=split)
        if not best:
            return {"tasks": 0, "best_loop_counts": {}, "final_degrades_rate": 0.0}

        best_loop_counts = Counter(row["best_loop"] for row in best.values())
        final_degrades = sum(1 for row in best.values() if row["final_vs_loop_1_delta"] > 0)
        any_deeper_wins = sum(
            1
            for row in best.values()
            if row["best_loop"] > 1 and row["best_vs_loop_1_delta"] < 0
        )

        return {
            "tasks": len(best),
            "best_loop_counts": dict(sorted(best_loop_counts.items())),
            "final_degrades_rate": final_degrades / len(best),
            "any_deeper_loop_win_rate": any_deeper_wins / len(best),
            "mean_best_vs_loop_1_delta": mean(
                row["best_vs_loop_1_delta"] for row in best.values()
            ),
            "mean_final_vs_loop_1_delta": mean(
                row["final_vs_loop_1_delta"] for row in best.values()
            ),
        }

    def conditional_usefulness(self, split: Optional[str] = "holdout") -> Dict[str, Dict[str, Any]]:
        """Report where deeper loops help, grouped by difficulty."""

        best = self.best_loop_by_task(split=split)
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in best.values():
            grouped[row["difficulty"]].append(row)

        report: Dict[str, Dict[str, Any]] = {}
        for difficulty in DIFFICULTY_ORDER:
            rows = grouped.get(difficulty, [])
            if not rows:
                continue

            deeper_wins = [
                row
                for row in rows
                if row["best_loop"] > 1 and row["best_vs_loop_1_delta"] < 0
            ]
            final_degrades = [row for row in rows if row["final_vs_loop_1_delta"] > 0]
            report[difficulty] = {
                "tasks": len(rows),
                "deeper_loop_win_rate": len(deeper_wins) / len(rows),
                "final_degrades_rate": len(final_degrades) / len(rows),
                "mean_best_vs_loop_1_delta": mean(
                    row["best_vs_loop_1_delta"] for row in rows
                ),
                "mean_final_vs_loop_1_delta": mean(
                    row["final_vs_loop_1_delta"] for row in rows
                ),
                "recommended_default_loop": self._recommended_loop(rows),
            }
        return report

    def refinement_loss_summary(self, split: Optional[str] = None) -> Dict[int, Dict[str, float]]:
        grouped: Dict[int, List[float]] = defaultdict(list)
        for score in self._filter(split=split):
            if score.refinement_loss is not None:
                grouped[score.loop].append(score.refinement_loss)

        return {
            loop: {
                "mean": mean(values),
                "min": min(values),
                "max": max(values),
            }
            for loop, values in sorted(grouped.items())
        }

    def generalization_gap_by_loop(self) -> Dict[int, float]:
        """Return holdout CE minus training CE for loops present in both splits."""

        by_split = self.mean_ce_by_split_and_loop()
        train = by_split.get("train") or by_split.get("training") or {}
        holdout = by_split.get("holdout", {})
        loops = sorted(set(train) & set(holdout))
        return {loop: holdout[loop] - train[loop] for loop in loops}

    def report(self) -> Dict[str, Any]:
        """Return a compact diagnostic report suitable for JSON output."""

        return {
            "mean_ce_by_split_and_loop": self.mean_ce_by_split_and_loop(),
            "holdout_ce_by_difficulty_and_loop": self.mean_ce_by_difficulty_and_loop(
                split="holdout"
            ),
            "best_of_loops_holdout": self.best_of_loops_summary(split="holdout"),
            "conditional_usefulness_holdout": self.conditional_usefulness(split="holdout"),
            "refinement_loss_holdout": self.refinement_loss_summary(split="holdout"),
            "generalization_gap_by_loop": self.generalization_gap_by_loop(),
        }

    def _filter(self, split: Optional[str] = None) -> Iterable[LoopScore]:
        if split is None:
            return list(self.scores)
        return [score for score in self.scores if score.split == split]

    @staticmethod
    def _recommended_loop(rows: List[Dict[str, Any]]) -> int:
        loop_counts = Counter(row["best_loop"] for row in rows)
        return min(
            loop_counts,
            key=lambda loop: (-loop_counts[loop], loop),
        )


@dataclass(frozen=True)
class AgentOptimizationRecommendation:
    """Lifecycle recommendation derived from OpenMythos coding diagnostics."""

    agent_id: str
    action: str
    target_state: Optional[str]
    reason: str
    score: float
    metrics: Dict[str, Any]
    subagent_specs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OpenMythosAgentOptimizer:
    """Use loop diagnostics to promote coding agents and suggest subagents.

    The optimizer rewards held-out coding performance, especially cases where
    recurrence is conditionally useful on medium or hard tasks. It keeps easy
    task degradation visible so always-deeper policies do not get promoted by
    accident.
    """

    CODING_CATEGORIES = {
        "coding",
        "code",
        "code_generation",
        "debugging",
        "bug_repair",
        "refactoring",
        "test_writing",
        "algorithm",
        "api",
    }

    def __init__(
        self,
        diagnostics: OpenMythosLoopDiagnostics,
        registry: AgentRegistry,
        min_holdout_tasks: int = 3,
        promote_threshold: float = 0.65,
        hot_threshold: float = 0.82,
        max_easy_final_degrade_rate: float = 0.35,
    ):
        self.diagnostics = diagnostics
        self.registry = registry
        self.min_holdout_tasks = min_holdout_tasks
        self.promote_threshold = promote_threshold
        self.hot_threshold = hot_threshold
        self.max_easy_final_degrade_rate = max_easy_final_degrade_rate

    def recommend(self) -> List[AgentOptimizationRecommendation]:
        """Return lifecycle and subagent recommendations for coding agents."""

        grouped_scores = self._coding_holdout_scores_by_agent()
        recommendations = []

        for agent_id, scores in sorted(grouped_scores.items()):
            metrics = self._agent_metrics(scores)
            score = self._optimization_score(metrics)
            action, target_state, reason = self._action_for(agent_id, metrics, score)
            subagent_specs = self._subagent_specs(agent_id, metrics)

            recommendations.append(
                AgentOptimizationRecommendation(
                    agent_id=agent_id,
                    action=action,
                    target_state=target_state,
                    reason=reason,
                    score=score,
                    metrics=metrics,
                    subagent_specs=subagent_specs,
                )
            )

        recommendations.sort(key=lambda item: item.score, reverse=True)
        return recommendations

    def apply_promotions(self) -> List[AgentOptimizationRecommendation]:
        """Promote recommended agents in the registry and return applied items."""

        applied = []
        for recommendation in self.recommend():
            if recommendation.action not in {"promote_to_warm", "promote_to_hot"}:
                continue

            agent = self.registry.get_agent(recommendation.agent_id)
            if not agent:
                continue

            agent.lifecycle_state = LifecycleState(recommendation.target_state)
            agent.average_quality_lift = recommendation.metrics["mean_best_vs_loop_1_gain"]
            agent.calibration_score = recommendation.score
            if "openmythos_optimized" not in agent.tags:
                agent.tags.append("openmythos_optimized")
            if "coding_optimizer" not in agent.tags:
                agent.tags.append("coding_optimizer")
            self.registry.add_agent(agent)
            applied.append(recommendation)

        return applied

    def create_recommended_subagents(self) -> List[AgentSpec]:
        """Create probationary subagent specs recommended by diagnostics."""

        created = []
        for recommendation in self.recommend():
            for spec in recommendation.subagent_specs:
                if self.registry.get_agent(spec["agent_id"]):
                    continue

                agent = AgentSpec(
                    agent_id=spec["agent_id"],
                    name=spec["name"],
                    description=spec["spawn_reason"],
                    domain=spec["domain"],
                    lifecycle_state=LifecycleState.PROBATIONARY,
                    tools=spec.get("tools", []),
                    tags=spec.get("tags", []),
                    parent_lineage=spec.get("parent_lineage"),
                    target_cluster=spec.get("target_cluster"),
                    expected_activation_rate=0.2,
                )
                self.registry.add_agent(agent)
                created.append(agent)

        return created

    def _coding_holdout_scores_by_agent(self) -> Dict[str, List[LoopScore]]:
        grouped: Dict[str, List[LoopScore]] = defaultdict(list)
        for score in self.diagnostics.scores:
            if score.split != "holdout":
                continue
            if not score.agent_id:
                continue
            if not self._is_coding_score(score):
                continue
            grouped[score.agent_id].append(score)
        return grouped

    def _agent_metrics(self, scores: List[LoopScore]) -> Dict[str, Any]:
        task_groups: Dict[str, List[LoopScore]] = defaultdict(list)
        for score in scores:
            task_groups[score.task_id].append(score)

        rows = []
        for task_id, task_scores in task_groups.items():
            winner = min(task_scores, key=lambda score: (score.cross_entropy, score.loop))
            first_loop = min(task_scores, key=lambda score: score.loop)
            last_loop = max(task_scores, key=lambda score: score.loop)
            rows.append(
                {
                    "task_id": task_id,
                    "difficulty": winner.difficulty,
                    "category": winner.category,
                    "best_loop": winner.loop,
                    "best_vs_loop_1_gain": first_loop.cross_entropy - winner.cross_entropy,
                    "final_vs_loop_1_delta": last_loop.cross_entropy - first_loop.cross_entropy,
                    "final_loop": last_loop.loop,
                }
            )

        tasks = len(rows)
        deeper_wins = [
            row for row in rows if row["best_loop"] > 1 and row["best_vs_loop_1_gain"] > 0
        ]
        final_degrades = [row for row in rows if row["final_vs_loop_1_delta"] > 0]
        medium_hard = [row for row in rows if row["difficulty"] in {"medium", "hard"}]
        medium_hard_deeper_wins = [
            row
            for row in medium_hard
            if row["best_loop"] > 1 and row["best_vs_loop_1_gain"] > 0
        ]
        easy = [row for row in rows if row["difficulty"] == "easy"]
        easy_final_degrades = [row for row in easy if row["final_vs_loop_1_delta"] > 0]

        return {
            "tasks": tasks,
            "difficulty_counts": dict(Counter(row["difficulty"] for row in rows)),
            "category_counts": dict(Counter(row["category"] for row in rows)),
            "best_loop_counts": dict(Counter(row["best_loop"] for row in rows)),
            "deeper_loop_win_rate": len(deeper_wins) / tasks if tasks else 0.0,
            "medium_hard_deeper_loop_win_rate": (
                len(medium_hard_deeper_wins) / len(medium_hard) if medium_hard else 0.0
            ),
            "final_degrades_rate": len(final_degrades) / tasks if tasks else 0.0,
            "easy_final_degrades_rate": (
                len(easy_final_degrades) / len(easy) if easy else 0.0
            ),
            "mean_best_vs_loop_1_gain": (
                mean(row["best_vs_loop_1_gain"] for row in rows) if rows else 0.0
            ),
            "mean_final_vs_loop_1_delta": (
                mean(row["final_vs_loop_1_delta"] for row in rows) if rows else 0.0
            ),
            "recommended_loop": self._recommended_loop(rows),
        }

    def _optimization_score(self, metrics: Dict[str, Any]) -> float:
        if metrics["tasks"] < self.min_holdout_tasks:
            return 0.0

        gain = max(0.0, min(metrics["mean_best_vs_loop_1_gain"] / 0.10, 1.0))
        deeper = metrics["deeper_loop_win_rate"]
        medium_hard = metrics["medium_hard_deeper_loop_win_rate"]
        stability = 1.0 - min(metrics["final_degrades_rate"], 1.0)
        easy_stability = 1.0 - min(metrics["easy_final_degrades_rate"], 1.0)

        return round(
            0.35 * gain
            + 0.20 * deeper
            + 0.25 * medium_hard
            + 0.10 * stability
            + 0.10 * easy_stability,
            4,
        )

    def _action_for(
        self,
        agent_id: str,
        metrics: Dict[str, Any],
        score: float,
    ) -> tuple[str, Optional[str], str]:
        agent = self.registry.get_agent(agent_id)
        if not agent:
            return "missing_agent", None, "No registry entry exists for this agent."

        current_state = self._state_value(agent.lifecycle_state)

        if metrics["tasks"] < self.min_holdout_tasks:
            return (
                "hold",
                None,
                f"Only {metrics['tasks']} held-out coding tasks; need {self.min_holdout_tasks}.",
            )

        if metrics["easy_final_degrades_rate"] > self.max_easy_final_degrade_rate:
            return (
                "hold",
                None,
                "Easy coding tasks degrade too often at the final loop; prefer adaptive halting first.",
            )

        if score >= self.hot_threshold and current_state != LifecycleState.HOT.value:
            return (
                "promote_to_hot",
                LifecycleState.HOT.value,
                "Strong held-out coding gains with useful recurrence on harder tasks.",
            )

        if score >= self.promote_threshold and current_state == LifecycleState.PROBATIONARY.value:
            return (
                "promote_to_warm",
                LifecycleState.WARM.value,
                "Held-out coding diagnostics justify promoting this probationary agent.",
            )

        if score >= self.promote_threshold:
            return (
                "keep_active",
                current_state,
                "Coding diagnostics support keeping this agent routable.",
            )

        return (
            "hold",
            None,
            "Coding diagnostics do not yet show enough held-out loop-specific lift.",
        )

    def _subagent_specs(self, agent_id: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        specs = []
        difficulty_counts = metrics["difficulty_counts"]

        if (
            difficulty_counts.get("hard", 0) >= 2
            and metrics["medium_hard_deeper_loop_win_rate"] >= 0.5
        ):
            specs.append(
                {
                    "agent_id": f"{agent_id}_hard_code_refiner",
                    "name": "Hard Code Refiner Subagent",
                    "domain": "coding",
                    "parent_lineage": agent_id,
                    "target_cluster": "hard_coding_refinement",
                    "tools": ["repo_tool", "test_runner", "code_analyzer"],
                    "tags": ["openmythos", "subagent", "hard_coding", "refinement"],
                    "recommended_loop": max(2, metrics["recommended_loop"]),
                    "spawn_reason": "Held-out hard coding tasks benefit from deeper refinement loops.",
                }
            )

        if (
            not specs
            and metrics["tasks"] >= self.min_holdout_tasks
            and metrics["deeper_loop_win_rate"] >= 0.5
            and metrics["medium_hard_deeper_loop_win_rate"] >= 0.5
            and metrics["mean_best_vs_loop_1_gain"] >= 0.05
            and metrics["easy_final_degrades_rate"] <= self.max_easy_final_degrade_rate
        ):
            specs.append(
                {
                    "agent_id": f"{agent_id}_loop{metrics['recommended_loop']}_code_refiner",
                    "name": "OpenMythos Loop Code Refiner",
                    "domain": "coding",
                    "parent_lineage": agent_id,
                    "target_cluster": "openmythos_coding_refinement",
                    "tools": ["repo_tool", "test_runner", "code_analyzer"],
                    "tags": [
                        "openmythos",
                        "subagent",
                        "coding",
                        "code",
                        "implementation",
                        "debugging",
                        "testing",
                        "refinement",
                    ],
                    "recommended_loop": max(2, metrics["recommended_loop"]),
                    "spawn_reason": (
                        "Held-out coding tasks show reliable loop-depth gains, "
                        "but the hard-task bucket is still too small for a hard-only refiner."
                    ),
                }
            )

        if metrics["easy_final_degrades_rate"] > 0:
            specs.append(
                {
                    "agent_id": f"{agent_id}_adaptive_halting_guard",
                    "name": "Adaptive Halting Guard Subagent",
                    "domain": "coding",
                    "parent_lineage": agent_id,
                    "target_cluster": "coding_loop_halt",
                    "tools": ["test_runner", "code_analyzer"],
                    "tags": ["openmythos", "subagent", "adaptive_halting"],
                    "recommended_loop": 1,
                    "spawn_reason": "Easy coding tasks show final-loop degradation; gate depth before routing.",
                }
            )

        return specs

    def _is_coding_score(self, score: LoopScore) -> bool:
        category = score.category.lower()
        if category in self.CODING_CATEGORIES:
            return True
        return any(token in category for token in self.CODING_CATEGORIES)

    @staticmethod
    def _recommended_loop(rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 1
        loop_counts = Counter(row["best_loop"] for row in rows)
        return min(loop_counts, key=lambda loop: (-loop_counts[loop], loop))

    @staticmethod
    def _state_value(state: Any) -> str:
        return state.value if hasattr(state, "value") else str(state)


def load_staged_examples_jsonl(path: str | Path) -> List[StagedRefinementExample]:
    """Load staged refinement examples from JSONL."""

    examples = []
    for row in _read_jsonl(path):
        stages = [StageTarget(**stage) for stage in row.get("stages", [])]
        examples.append(
            StagedRefinementExample(
                task_id=row["task_id"],
                prompt=row["prompt"],
                final_answer=row["final_answer"],
                category=row["category"],
                difficulty=row["difficulty"],
                split=row.get("split", "development"),
                stages=stages,
                metadata=row.get("metadata", {}),
            )
        )
    return examples


def load_loop_scores_jsonl(path: str | Path) -> List[LoopScore]:
    """Load loop CE/refinement-loss rows from JSONL."""

    return [LoopScore(**row) for row in _read_jsonl(path)]


def _read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
