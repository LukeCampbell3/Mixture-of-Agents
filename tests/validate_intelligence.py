"""
Agent intelligence validation — verifies knowledge enrichment, persistence,
and end-to-end agent behaviour through a stub LLM.

Tests:
  1. WebFetcher — real HTTP fetches (PyPI, npm, URL)
  2. KnowledgeEnricher — query detection + context block assembly
  3. Registry persistence — save → reload → agents intact
  4. Artifact persistence — RunState written and readable
  5. Spawned agent persistence — specialist survives restart
  6. Knowledge injection — enriched context reaches agent prompt
  7. Language detection — correct language extracted from task text
  8. End-to-end stub run — full orchestrator cycle with stub LLM

Run:
    python tests/validate_intelligence.py
    python tests/validate_intelligence.py --no-network   # skip live fetches
"""

import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Result tracking ───────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

_results: List[tuple] = []   # (status, name, detail)


def record(status: str, name: str, detail: str = ""):
    _results.append((status, name, detail))
    icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "○"}.get(status, "?")
    color = {"PASS": "\033[32m", "FAIL": "\033[31m", "SKIP": "\033[33m"}.get(status, "")
    reset = "\033[0m"
    line = f"  {color}{icon}{reset} {name}"
    if detail:
        line += f"  \033[2m({detail})\033[0m"
    print(line)


def section(title: str):
    print(f"\n\033[1m{title}\033[0m")
    print("─" * 50)


# ── Stub LLM ──────────────────────────────────────────────────────────────────

class StubLLM:
    """Deterministic LLM that echoes the knowledge block back."""

    def __init__(self):
        self.last_prompt = ""
        self.call_count = 0

    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        self.last_prompt = prompt
        self.call_count += 1
        # Return something that looks like a code response
        return (
            "Here is the implementation:\n\n"
            "```python\n"
            "def solution():\n"
            "    # Complete implementation\n"
            "    return 42\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    print(solution())\n"
            "```\n\n"
            "**How it works:** This function returns 42.\n"
        )

    def get_model_name(self) -> str:
        return "stub-model"


# ── Test 1: WebFetcher ────────────────────────────────────────────────────────

def test_web_fetcher_pypi(no_network: bool):
    section("1. WebFetcher — PyPI")
    from app.tools.web_fetcher import WebFetcher

    fetcher = WebFetcher()

    if no_network:
        record("SKIP", "PyPI fetch (requests)", "--no-network")
        return

    result = fetcher.fetch_pypi("requests")
    if result.ok and "requests" in result.content.lower():
        record("PASS", "fetch_pypi(requests)", f"version found: {result.content[:60]}")
    else:
        record("FAIL", "fetch_pypi(requests)", result.error or "no content")

    # Cache hit
    t0 = time.perf_counter()
    result2 = fetcher.fetch_pypi("requests")
    elapsed = time.perf_counter() - t0
    if elapsed < 0.01:
        record("PASS", "PyPI cache hit", f"{elapsed*1000:.1f}ms")
    else:
        record("FAIL", "PyPI cache hit", f"took {elapsed:.2f}s (expected <10ms)")


def test_web_fetcher_npm(no_network: bool):
    section("2. WebFetcher — npm")
    from app.tools.web_fetcher import WebFetcher

    if no_network:
        record("SKIP", "npm fetch (axios)", "--no-network")
        return

    fetcher = WebFetcher()
    result = fetcher.fetch_npm("axios")
    if result.ok and "axios" in result.content.lower():
        record("PASS", "fetch_npm(axios)", result.content[:60])
    else:
        record("FAIL", "fetch_npm(axios)", result.error or "no content")


def test_knowledge_query_builder():
    section("3. Knowledge Query Builder")
    from app.tools.web_fetcher import build_knowledge_queries

    cases = [
        ("build a fastapi server",          ["url"],   "fastapi"),
        ("implement pytorch neural net",    ["url"],   "pytorch"),
        ("create react typescript app",     ["url"],   "react"),
        ("write a binary search in python", [],        None),
        ("use pandas for data analysis",    ["url"],   "pandas"),
    ]

    for task, expected_types, expected_keyword in cases:
        queries = build_knowledge_queries(task)
        types = [q[0] for q in queries]
        keywords = [q[1] for q in queries]

        if expected_types and not any(t in types for t in expected_types):
            record("FAIL", f"query builder: {task[:40]}", f"got {queries}")
        elif expected_keyword and not any(expected_keyword in str(k) for k in keywords):
            record("FAIL", f"query builder: {task[:40]}", f"missing {expected_keyword} in {keywords}")
        else:
            record("PASS", f"query builder: {task[:40]}", str(queries[:2]))


def test_knowledge_enricher(no_network: bool):
    section("4. KnowledgeEnricher")
    from app.agents.knowledge_enricher import KnowledgeEnricher

    enricher = KnowledgeEnricher()

    # Task with no relevant knowledge
    result = enricher.enrich("explain what a variable is")
    if not result.has_content:
        record("PASS", "no-op for generic task", "no sources fetched")
    else:
        record("FAIL", "no-op for generic task", f"unexpectedly fetched {len(result.sources)} sources")

    if no_network:
        record("SKIP", "enricher with fastapi task", "--no-network")
        return

    # Task with relevant knowledge
    result2 = enricher.enrich("build a fastapi server with pydantic models")
    if result2.has_content:
        block = result2.as_context_block()
        record("PASS", "enricher fetches fastapi docs",
               f"{len(result2.sources)} sources, {len(block)} chars")
        if "RETRIEVED KNOWLEDGE" in block:
            record("PASS", "context block has correct header")
        else:
            record("FAIL", "context block missing header")
    else:
        record("FAIL", "enricher found no content for fastapi task",
               str([s.error for s in result2.sources]))


# ── Test 2: Registry persistence ─────────────────────────────────────────────

def test_registry_persistence():
    section("5. Registry Persistence")
    from app.storage.registry_store import RegistryStore
    from app.schemas.registry import AgentRegistry, AgentSpec, LifecycleState

    with tempfile.TemporaryDirectory() as tmpdir:
        store = RegistryStore(data_dir=tmpdir)

        # Create registry with 3 agents
        reg = AgentRegistry()
        for i in range(3):
            reg.add_agent(AgentSpec(
                agent_id=f"agent_{i}",
                name=f"Agent {i}",
                description=f"Test agent {i}",
                domain="coding",
                lifecycle_state=LifecycleState.WARM,
                tools=["repo_tool"],
                tags=["test"],
            ))

        store.save_registry(reg)
        reg_path = Path(tmpdir) / "agent_registry.json"

        if reg_path.exists():
            record("PASS", "registry file created", str(reg_path))
        else:
            record("FAIL", "registry file not created")
            return

        # Reload
        loaded = store.load_registry()
        if loaded is None:
            record("FAIL", "registry loaded as None")
            return

        if len(loaded.agents) == 3:
            record("PASS", "all 3 agents reloaded")
        else:
            record("FAIL", f"expected 3 agents, got {len(loaded.agents)}")

        # Verify agent data integrity
        a0 = loaded.get_agent("agent_0")
        if a0 and a0.name == "Agent 0" and a0.domain == "coding":
            record("PASS", "agent data integrity preserved")
        else:
            record("FAIL", "agent data corrupted on reload")

        # Atomic write — verify no .tmp file left behind
        tmp_files = list(Path(tmpdir).glob("*.tmp"))
        if not tmp_files:
            record("PASS", "no temp files left after atomic write")
        else:
            record("FAIL", f"temp files not cleaned up: {tmp_files}")


def test_spawned_agent_persistence():
    section("6. Spawned Agent Persistence (Cross-Session)")
    from app.storage.registry_store import RegistryStore
    from app.schemas.registry import AgentRegistry, AgentSpec, LifecycleState

    with tempfile.TemporaryDirectory() as tmpdir:
        store = RegistryStore(data_dir=tmpdir)

        # Session 1: create base registry + spawn a specialist
        reg = AgentRegistry()
        reg.add_agent(AgentSpec(
            agent_id="code_primary",
            name="Code Primary",
            description="Base coding agent",
            domain="coding",
            lifecycle_state=LifecycleState.HOT,
            tools=["repo_tool"],
            tags=["coding"],
        ))
        reg.add_agent(AgentSpec(
            agent_id="specialist_ml_engineering",
            name="ML Engineering Specialist",
            description="Spawned specialist for ML tasks",
            domain="ml_engineering",
            lifecycle_state=LifecycleState.PROBATIONARY,
            tools=["repo_tool"],
            tags=["ml_engineering", "spawned", "dynamic"],
        ))
        store.save_registry(reg)

        # Session 2: reload and verify specialist survived
        store2 = RegistryStore(data_dir=tmpdir)
        loaded = store2.load_registry()

        if loaded is None:
            record("FAIL", "registry not loaded in session 2")
            return

        specialist = loaded.get_agent("specialist_ml_engineering")
        if specialist:
            record("PASS", "spawned specialist survived restart",
                   f"state={specialist.lifecycle_state}")
        else:
            record("FAIL", "spawned specialist lost on restart")

        # Verify lifecycle state preserved
        if specialist and str(specialist.lifecycle_state) in ("probationary", "LifecycleState.PROBATIONARY"):
            record("PASS", "lifecycle state preserved across restart")
        else:
            state = str(specialist.lifecycle_state) if specialist else "missing"
            record("FAIL", f"lifecycle state wrong: {state}")

        # Verify base agent still present
        base = loaded.get_agent("code_primary")
        if base and str(base.lifecycle_state) in ("hot", "LifecycleState.HOT"):
            record("PASS", "base agent state preserved")
        else:
            record("FAIL", "base agent state wrong")


# ── Test 3: Artifact persistence ─────────────────────────────────────────────

def test_artifact_persistence():
    section("7. Artifact Persistence (RunState)")
    from app.storage.artifact_store import ArtifactStore
    from app.schemas.run_state import RunState

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArtifactStore(data_dir=tmpdir)
        task_id = str(uuid.uuid4())

        run_state = RunState(
            task_id=task_id,
            task_frame={"normalized_request": "test task", "task_type": "coding_stable"},
            active_agents=["code_primary"],
            suppressed_agents=[],
            budget_usage={"max_active_agents": 3, "active_agents": 1},
            base_model_version="stub-model",
            final_answer="Here is the solution.",
            final_state="success",
        )

        store.save_run_state(run_state)
        store.save_context(task_id, "## Context\nShared context here.")
        store.save_analysis(task_id, {"routing": "code_primary", "score": 0.85})

        # Verify files exist
        episodes_dir = Path(tmpdir) / "episodes"
        state_file = episodes_dir / f"{task_id}.json"
        ctx_file = episodes_dir / f"{task_id}_context.md"
        analysis_file = episodes_dir / f"{task_id}_analysis.json"

        for f, name in [(state_file, "RunState"), (ctx_file, "Context"), (analysis_file, "Analysis")]:
            if f.exists():
                record("PASS", f"{name} file written", str(f.name))
            else:
                record("FAIL", f"{name} file missing")

        # Reload RunState
        loaded = store.load_run_state(task_id)
        if loaded.task_id == task_id:
            record("PASS", "RunState reloaded correctly")
        else:
            record("FAIL", f"RunState task_id mismatch: {loaded.task_id}")

        if loaded.final_state == "success":
            record("PASS", "RunState final_state preserved")
        else:
            record("FAIL", f"RunState final_state wrong: {loaded.final_state}")

        if loaded.active_agents == ["code_primary"]:
            record("PASS", "RunState active_agents preserved")
        else:
            record("FAIL", f"RunState active_agents wrong: {loaded.active_agents}")


# ── Test 4: Knowledge injection into agent prompt ─────────────────────────────

def test_knowledge_injection():
    section("8. Knowledge Injection into Agent Prompt")
    from app.agents.code_primary import CodePrimaryAgent
    from app.schemas.task_frame import TaskFrame, TaskType

    stub = StubLLM()
    agent = CodePrimaryAgent(
        agent_id="code_primary",
        name="Code Primary",
        description="Test agent",
        llm_client=stub,
        tools=["repo_tool"],
    )

    knowledge_block = (
        "=== RETRIEVED KNOWLEDGE ===\n"
        "### FastAPI Documentation\n"
        "Source: https://fastapi.tiangolo.com/\n"
        "```\nFastAPI is a modern web framework...\n```\n"
        "=== END RETRIEVED KNOWLEDGE ===\n"
    )

    tf = TaskFrame(
        task_id=str(uuid.uuid4()),
        normalized_request="build a fastapi endpoint",
        task_type=TaskType.CODING_STABLE,
        hard_constraints=[],
        likely_tools=["repo_tool"],
        difficulty_estimate=0.5,
        initial_uncertainty=0.5,
        novelty_score=0.5,
        freshness_requirement=0.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        agent.execute({
            "task_frame": tf,
            "shared_context": "Task context here.",
            "iteration": 1,
            "max_tokens": 200,
            "workspace_root": tmpdir,
            "knowledge_block": knowledge_block,
            "language_preference": "python",
        })

    prompt = stub.last_prompt
    if "RETRIEVED KNOWLEDGE" in prompt:
        record("PASS", "knowledge block injected into prompt")
    else:
        record("FAIL", "knowledge block NOT in prompt")

    if "FastAPI Documentation" in prompt:
        record("PASS", "fetched doc title present in prompt")
    else:
        record("FAIL", "fetched doc title missing from prompt")

    if "LANGUAGE: Write all code in python" in prompt:
        record("PASS", "language preference injected")
    else:
        record("FAIL", "language preference NOT in prompt")

    if stub.call_count == 1:
        record("PASS", "LLM called exactly once")
    else:
        record("FAIL", f"LLM called {stub.call_count} times (expected 1)")


# ── Test 5: Language detection ────────────────────────────────────────────────

def test_language_detection():
    section("9. Language Detection")
    sys.path.insert(0, str(ROOT))

    try:
        from claude_integrated import _detect_language_from_text, _is_large_codebase_task
    except ImportError as e:
        record("FAIL", "import claude_integrated", str(e))
        return

    lang_cases = [
        ("build a REST API in python",        "python"),
        ("create a typescript react app",     "typescript"),
        ("implement a rust cli tool",         "rust"),
        ("write a go microservice",           "go"),
        ("build a spring boot java app",      "java"),
        ("explain what a variable is",        ""),
        ("implement a linked list",           ""),
    ]

    for text, expected in lang_cases:
        got = _detect_language_from_text(text)
        if got == expected:
            record("PASS", f"lang detect: {text[:40]}", f"→ {got!r}")
        else:
            record("FAIL", f"lang detect: {text[:40]}", f"expected {expected!r}, got {got!r}")

    codebase_cases = [
        ("build a REST API application",      True),
        ("create a typescript react app",     True),
        ("implement a linked list",           False),
        ("explain binary search",             False),
        ("develop a microservice backend",    True),
    ]

    for text, expected in codebase_cases:
        got = _is_large_codebase_task(text)
        if got == expected:
            record("PASS", f"codebase detect: {text[:40]}", f"→ {got}")
        else:
            record("FAIL", f"codebase detect: {text[:40]}", f"expected {expected}, got {got}")


# ── Test 6: End-to-end stub orchestrator run ──────────────────────────────────

def test_e2e_stub_run():
    section("10. End-to-End Stub Orchestrator Run")

    from app.orchestrator import Orchestrator
    from app.schemas.registry import AgentRegistry, AgentSpec, LifecycleState
    from app.storage.registry_store import RegistryStore
    from app.storage.artifact_store import ArtifactStore
    from app.models.embeddings import EmbeddingGenerator
    from app.models.uncertainty import UncertaintyEstimator
    from app.calibration import ThreeLevelCalibrator
    from app.lifecycle import LifecycleManager
    from app.gap_analyzer import GapAnalyzer
    from app.parallel_executor import ParallelExecutor
    from app.skill_packs import get_skill_pack_registry
    from app.lead_agent_pattern import LeadAgentCoordinator
    from app.validator import Validator
    from app.router import Router
    from app.agent_factory import AgentFactory

    stub = StubLLM()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build orchestrator manually with stub LLM
        orch = Orchestrator.__new__(Orchestrator)
        orch.llm_client = stub
        orch.router_llm_client = stub
        orch.embedding_generator = EmbeddingGenerator()
        orch.uncertainty_estimator = UncertaintyEstimator()
        orch.registry_store = RegistryStore(tmpdir)
        orch.artifact_store = ArtifactStore(tmpdir)

        # Fresh registry
        registry = AgentRegistry()
        registry.add_agent(AgentSpec(
            agent_id="code_primary", name="Code Primary",
            description="Primary coding agent",
            domain="coding", lifecycle_state=LifecycleState.HOT,
            tools=["repo_tool", "test_runner"],
            tags=["coding", "implementation"],
        ))
        registry.add_agent(AgentSpec(
            agent_id="web_research", name="Web Research",
            description="Research agent",
            domain="research", lifecycle_state=LifecycleState.WARM,
            tools=["web_tool"], tags=["research"],
        ))
        registry.add_agent(AgentSpec(
            agent_id="critic_verifier", name="Critic Verifier",
            description="Verification agent",
            domain="verification", lifecycle_state=LifecycleState.WARM,
            tools=["test_runner"], tags=["verification"],
        ))
        orch.registry = registry
        orch.registry_store.save_registry(registry)

        orch.calibrator = ThreeLevelCalibrator(
            task_families=["coding", "research", "reasoning", "mixed"]
        )
        orch.router = Router(
            registry, stub, orch.embedding_generator,
            orch.uncertainty_estimator, calibrator=orch.calibrator
        )
        orch.validator = Validator()
        orch.use_lead_agent_pattern = False
        orch.lead_coordinator = LeadAgentCoordinator(max_supporting_agents=2)
        orch.enable_parallel = False
        orch.parallel_executor = None
        orch.skill_pack_registry = get_skill_pack_registry()
        orch.lifecycle_manager = LifecycleManager(registry, orch.embedding_generator)
        orch.budget_mode = "balanced"
        orch.max_parallel_agents = 2
        orch.max_tokens = 400
        orch.auto_approve_file_ops = True
        orch.agent_factory = AgentFactory(stub)
        orch._language_preference = "python"

        class _NoOpGapAnalyzer:
            agent_embeddings = {}
            def analyze_gap(self, *a, **kw): return {}
            class embedding_generator:
                @staticmethod
                def embed(t): return [0.1] * 10

        orch.gap_analyzer = _NoOpGapAnalyzer()

        # Run a task
        try:
            run_state = orch.run_task(
                "implement a binary search function",
                workspace_root=tmpdir,
                conversation_history="",
            )
            record("PASS", "run_task completed without exception")
        except Exception as e:
            record("FAIL", "run_task raised exception", str(e)[:100])
            return

        # Verify run state
        if run_state.task_id:
            record("PASS", "RunState has task_id", run_state.task_id[:8])
        else:
            record("FAIL", "RunState missing task_id")

        if run_state.active_agents:
            record("PASS", "agents were activated", str(run_state.active_agents))
        else:
            record("FAIL", "no agents activated")

        if run_state.final_answer:
            record("PASS", "final_answer produced",
                   run_state.final_answer[:60].replace("\n", " "))
        else:
            record("FAIL", "no final_answer")

        # Verify artifacts written
        episodes = Path(tmpdir) / "episodes"
        episode_files = list(episodes.glob(f"{run_state.task_id}*.json"))
        if episode_files:
            record("PASS", f"artifacts written ({len(episode_files)} file(s))")
        else:
            record("FAIL", "no artifact files written")

        # Verify registry still intact after run
        loaded_reg = orch.registry_store.load_registry()
        if loaded_reg and len(loaded_reg.agents) >= 3:
            record("PASS", f"registry intact after run ({len(loaded_reg.agents)} agents)")
        else:
            count = len(loaded_reg.agents) if loaded_reg else 0
            record("FAIL", f"registry damaged after run ({count} agents)")

        # Verify LLM was called
        if stub.call_count > 0:
            record("PASS", f"LLM called {stub.call_count} time(s)")
        else:
            record("FAIL", "LLM never called")

        # Verify knowledge block was injected (check last prompt)
        # The enricher may or may not find queries for "binary search"
        # but the prompt should always have the task
        if "binary search" in stub.last_prompt.lower():
            record("PASS", "task text present in LLM prompt")
        else:
            record("FAIL", "task text missing from LLM prompt")

        if "LANGUAGE: Write all code in python" in stub.last_prompt:
            record("PASS", "language preference in LLM prompt")
        else:
            record("FAIL", "language preference missing from LLM prompt")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate agent intelligence and persistence")
    parser.add_argument("--no-network", action="store_true",
                        help="Skip tests that require internet access")
    args = parser.parse_args()

    print("\n\033[1;34m╔══════════════════════════════════════════════════════════╗\033[0m")
    print("\033[1;34m║  Agent Intelligence & Persistence Validation              ║\033[0m")
    print("\033[1;34m╚══════════════════════════════════════════════════════════╝\033[0m")

    test_web_fetcher_pypi(args.no_network)
    test_web_fetcher_npm(args.no_network)
    test_knowledge_query_builder()
    test_knowledge_enricher(args.no_network)
    test_registry_persistence()
    test_spawned_agent_persistence()
    test_artifact_persistence()
    test_knowledge_injection()
    test_language_detection()
    test_e2e_stub_run()

    # Summary
    passed  = sum(1 for s, _, _ in _results if s == "PASS")
    failed  = sum(1 for s, _, _ in _results if s == "FAIL")
    skipped = sum(1 for s, _, _ in _results if s == "SKIP")
    total   = len(_results)

    print(f"\n{'─' * 60}")
    print(f"Results: \033[32m{passed} passed\033[0m  "
          f"\033[31m{failed} failed\033[0m  "
          f"\033[33m{skipped} skipped\033[0m  "
          f"({total} checks)")

    if failed:
        print("\n\033[31mValidation FAILED — see failures above.\033[0m")
        sys.exit(1)
    else:
        print("\n\033[32mAll checks passed.\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    main()
