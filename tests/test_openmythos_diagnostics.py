"""Tests for OpenMythos loop-depth diagnostics."""

import json
import subprocess
import sys

from app.evaluation.openmythos import (
    LoopScore,
    OpenMythosAgentOptimizer,
    OpenMythosLoopDiagnostics,
    StageTarget,
    StagedRefinementExample,
    load_loop_scores_jsonl,
    validate_staged_dataset,
)
from app.schemas.registry import AgentRegistry, AgentSpec, LifecycleState


def test_dataset_readiness_flags_tiny_unstaged_dataset():
    examples = [
        StagedRefinementExample(
            task_id=f"task_{i}",
            prompt="Write a parser",
            final_answer="def parse(value): return value",
            category="code_generation",
            difficulty="easy",
            split="train",
        )
        for i in range(44)
    ]

    report = validate_staged_dataset(examples)

    assert not report.ready
    assert report.total_examples == 44
    assert any("at least 1000" in error for error in report.errors)
    assert any("intermediate stage targets" in warning for warning in report.warnings)


def test_dataset_readiness_accepts_diverse_staged_holdout_data():
    categories = ["code_generation", "debugging", "refactoring", "test_writing"]
    difficulties = ["easy", "medium", "hard"]
    examples = []

    for i in range(120):
        examples.append(
            StagedRefinementExample(
                task_id=f"task_{i}",
                prompt=f"Task {i}",
                final_answer="final",
                category=categories[i % len(categories)],
                difficulty=difficulties[i % len(difficulties)],
                split="holdout" if i % 5 == 0 else "train",
                stages=[
                    StageTarget(loop=1, label="draft", target="draft"),
                    StageTarget(loop=2, label="valid", target="valid"),
                    StageTarget(loop=3, label="corrected", target="corrected"),
                    StageTarget(loop=4, label="final", target="final"),
                ],
            )
        )

    report = validate_staged_dataset(examples, min_examples=100, target_examples=200)

    assert report.ready
    assert report.staged_examples == 120
    assert report.category_counts["debugging"] == 30
    assert "holdout" in report.split_counts


def test_loop_diagnostics_detects_final_loop_degradation():
    diagnostics = OpenMythosLoopDiagnostics(
        [
            LoopScore("easy_1", "holdout", "easy", 1, 0.10),
            LoopScore("easy_1", "holdout", "easy", 2, 0.12),
            LoopScore("easy_1", "holdout", "easy", 4, 0.18),
            LoopScore("hard_1", "holdout", "hard", 1, 0.35),
            LoopScore("hard_1", "holdout", "hard", 2, 0.30),
            LoopScore("hard_1", "holdout", "hard", 4, 0.28),
        ]
    )

    summary = diagnostics.best_of_loops_summary(split="holdout")
    conditional = diagnostics.conditional_usefulness(split="holdout")

    assert summary["tasks"] == 2
    assert summary["best_loop_counts"] == {1: 1, 4: 1}
    assert summary["final_degrades_rate"] == 0.5
    assert conditional["easy"]["recommended_default_loop"] == 1
    assert conditional["hard"]["recommended_default_loop"] == 4


def test_generalization_gap_by_loop_prefers_common_training_names():
    diagnostics = OpenMythosLoopDiagnostics(
        [
            LoopScore("task_a", "train", "medium", 1, 0.01),
            LoopScore("task_a", "train", "medium", 2, 0.01),
            LoopScore("task_a", "holdout", "medium", 1, 0.20),
            LoopScore("task_a", "holdout", "medium", 2, 0.30),
        ]
    )

    assert diagnostics.generalization_gap_by_loop() == {1: 0.19, 2: 0.29}


def test_load_loop_scores_jsonl(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "task_a",
                "split": "holdout",
                "difficulty": "medium",
                "loop": 1,
                "cross_entropy": 0.2,
                "refinement_loss": 0.015,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    scores = load_loop_scores_jsonl(path)

    assert scores == [
        LoopScore(
            task_id="task_a",
            split="holdout",
            difficulty="medium",
            loop=1,
            cross_entropy=0.2,
            refinement_loss=0.015,
        )
    ]


def test_openmythos_diagnostics_cli_outputs_report(tmp_path):
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "task_a",
                        "split": "train",
                        "difficulty": "hard",
                        "loop": 1,
                        "cross_entropy": 0.01,
                    }
                ),
                json.dumps(
                    {
                        "task_id": "task_a",
                        "split": "holdout",
                        "difficulty": "hard",
                        "loop": 1,
                        "cross_entropy": 0.2,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/openmythos_diagnostics.py",
            "--scores",
            str(scores_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert "loop_diagnostics" in payload
    assert payload["loop_diagnostics"]["generalization_gap_by_loop"] == {"1": 0.19}


def test_agent_optimizer_promotes_powerful_coding_agent():
    registry = AgentRegistry()
    registry.add_agent(
        AgentSpec(
            agent_id="openmythos_code",
            name="OpenMythos Code",
            description="Coding refinement agent",
            domain="coding",
            lifecycle_state=LifecycleState.PROBATIONARY,
            tools=["repo_tool", "test_runner"],
        )
    )
    diagnostics = OpenMythosLoopDiagnostics(
        [
            LoopScore("hard_1", "holdout", "hard", 1, 0.40, category="coding", agent_id="openmythos_code"),
            LoopScore("hard_1", "holdout", "hard", 2, 0.28, category="coding", agent_id="openmythos_code"),
            LoopScore("hard_1", "holdout", "hard", 4, 0.25, category="coding", agent_id="openmythos_code"),
            LoopScore("hard_2", "holdout", "hard", 1, 0.38, category="debugging", agent_id="openmythos_code"),
            LoopScore("hard_2", "holdout", "hard", 2, 0.26, category="debugging", agent_id="openmythos_code"),
            LoopScore("hard_2", "holdout", "hard", 4, 0.24, category="debugging", agent_id="openmythos_code"),
            LoopScore("medium_1", "holdout", "medium", 1, 0.30, category="refactoring", agent_id="openmythos_code"),
            LoopScore("medium_1", "holdout", "medium", 2, 0.21, category="refactoring", agent_id="openmythos_code"),
            LoopScore("medium_1", "holdout", "medium", 4, 0.20, category="refactoring", agent_id="openmythos_code"),
        ]
    )

    optimizer = OpenMythosAgentOptimizer(
        diagnostics,
        registry,
        min_holdout_tasks=3,
        promote_threshold=0.65,
        hot_threshold=0.95,
    )
    recommendations = optimizer.recommend()
    applied = optimizer.apply_promotions()
    created = optimizer.create_recommended_subagents()
    agent = registry.get_agent("openmythos_code")

    assert recommendations[0].action == "promote_to_hot"
    assert recommendations[0].subagent_specs[0]["target_cluster"] == "hard_coding_refinement"
    assert applied[0].agent_id == "openmythos_code"
    assert created[0].agent_id == "openmythos_code_hard_code_refiner"
    assert registry.get_agent("openmythos_code_hard_code_refiner").parent_lineage == "openmythos_code"
    assert agent.lifecycle_state == LifecycleState.HOT
    assert "openmythos_optimized" in agent.tags


def test_agent_optimizer_holds_when_easy_tasks_degrade():
    registry = AgentRegistry()
    registry.add_agent(
        AgentSpec(
            agent_id="openmythos_code",
            name="OpenMythos Code",
            description="Coding refinement agent",
            domain="coding",
            lifecycle_state=LifecycleState.PROBATIONARY,
        )
    )
    diagnostics = OpenMythosLoopDiagnostics(
        [
            LoopScore("easy_1", "holdout", "easy", 1, 0.10, category="coding", agent_id="openmythos_code"),
            LoopScore("easy_1", "holdout", "easy", 2, 0.12, category="coding", agent_id="openmythos_code"),
            LoopScore("easy_1", "holdout", "easy", 4, 0.20, category="coding", agent_id="openmythos_code"),
            LoopScore("hard_1", "holdout", "hard", 1, 0.40, category="coding", agent_id="openmythos_code"),
            LoopScore("hard_1", "holdout", "hard", 2, 0.25, category="coding", agent_id="openmythos_code"),
            LoopScore("hard_1", "holdout", "hard", 4, 0.23, category="coding", agent_id="openmythos_code"),
            LoopScore("hard_2", "holdout", "hard", 1, 0.40, category="coding", agent_id="openmythos_code"),
            LoopScore("hard_2", "holdout", "hard", 2, 0.25, category="coding", agent_id="openmythos_code"),
            LoopScore("hard_2", "holdout", "hard", 4, 0.23, category="coding", agent_id="openmythos_code"),
        ]
    )

    optimizer = OpenMythosAgentOptimizer(diagnostics, registry, min_holdout_tasks=3)
    recommendation = optimizer.recommend()[0]

    assert recommendation.action == "hold"
    assert "adaptive halting" in recommendation.reason.lower()
    assert any(
        spec["target_cluster"] == "coding_loop_halt"
        for spec in recommendation.subagent_specs
    )
