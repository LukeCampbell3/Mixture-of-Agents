"""Runtime validation tests for OpenMythos orchestration integration."""

from types import SimpleNamespace

from app.openmythos_runtime import OpenMythosRuntimeAdapter
from app.schemas.registry import AgentRegistry, AgentSpec, LifecycleState


class FakeRegistryStore:
    def __init__(self):
        self.saved = False

    def save_registry(self, registry):
        self.saved = True


class FakeOrchestrator:
    def __init__(self, mode: str):
        self.mode = mode
        self.registry = AgentRegistry()
        self.registry.add_agent(
            AgentSpec(
                agent_id="code_primary",
                name="Code Primary",
                description="General coding agent",
                domain="coding",
                lifecycle_state=LifecycleState.HOT,
                tags=["coding"],
            )
        )
        if mode == "openmythos":
            self.registry.add_agent(
                AgentSpec(
                    agent_id="openmythos_code_hard_code_refiner",
                    name="Hard Code Refiner",
                    description="OpenMythos hard coding subagent",
                    domain="coding",
                    lifecycle_state=LifecycleState.PROBATIONARY,
                    tags=["openmythos", "subagent", "hard_coding"],
                    parent_lineage="code_primary",
                )
            )
        self.registry_store = FakeRegistryStore()
        self.calls = 0

    def run_task(self, prompt, workspace_root=".", conversation_history=""):
        self.calls += 1
        if self.mode == "baseline":
            return SimpleNamespace(
                task_id=f"base_{self.calls}",
                final_state="failure",
                active_agents=["code_primary"],
                spawned_agents=[],
                promoted_agents=[],
                final_files=[],
                pending_tool_calls=[],
                pool_size_before=1,
                pool_size_after=1,
                lifecycle_events=[],
            )

        return SimpleNamespace(
            task_id=f"openmythos_{self.calls}",
            final_state="success",
            active_agents=["code_primary", "openmythos_code_hard_code_refiner"],
            spawned_agents=["openmythos_code_hard_code_refiner"] if self.calls == 1 else [],
            promoted_agents=["code_primary"] if self.calls == 1 else [],
            final_files=[],
            pending_tool_calls=[
                {"tool": "write_file", "path": f"src/generated_{self.calls}.py"}
            ],
            pool_size_before=1,
            pool_size_after=2,
            lifecycle_events=[],
        )


def test_openmythos_comparison_must_beat_original_orchestrator(tmp_path):
    baseline = FakeOrchestrator("baseline")
    openmythos = FakeOrchestrator("openmythos")
    adapter = OpenMythosRuntimeAdapter(openmythos, data_dir=str(tmp_path))

    report = adapter.compare_against_baseline(
        baseline,
        sample_size=3,
    )

    assert report["openmythos_better"] is True
    assert report["verdict"] == "improved"
    assert report["score_delta"] > 0
    assert report["openmythos"]["long_term_code_creation"] == "file_outputs_detected"
    assert "openmythos_code_hard_code_refiner" in report["openmythos"]["agents_used"]
