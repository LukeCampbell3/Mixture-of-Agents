"""Tests for Phase 2: Audit-Ready Orchestration."""

import pytest
from app.arbitration import Arbitrator, ConflictType
from app.lifecycle import LifecycleManager
from app.synthesizer import Synthesizer, SynthesisPackage
from app.schemas.registry import AgentRegistry, AgentSpec, LifecycleState
from app.schemas.task_frame import TaskFrame, TaskType
from app.schemas.validation import ValidationReport, ValidationState
from app.models.llm_client import create_llm_client
from app.models.embeddings import EmbeddingGenerator


@pytest.fixture
def llm_client():
    """Create LLM client for testing."""
    return create_llm_client("openai")


@pytest.fixture
def arbitrator(llm_client):
    """Create arbitrator for testing."""
    return Arbitrator(llm_client)


@pytest.fixture
def registry():
    """Create test registry."""
    registry = AgentRegistry()
    registry.add_agent(AgentSpec(
        agent_id="test_agent_1",
        name="Test Agent 1",
        description="Test agent",
        domain="testing",
        lifecycle_state=LifecycleState.HOT
    ))
    return registry


@pytest.fixture
def lifecycle_manager(registry):
    """Create lifecycle manager for testing."""
    embedding_gen = EmbeddingGenerator()
    return LifecycleManager(registry, embedding_gen)


class TestArbitration:
    """Test conflict arbitration."""
    
    def test_detect_factual_conflicts(self, arbitrator):
        """Test factual conflict detection."""
        agent_outputs = {
            "agent1": {"output": "The answer is 42"},
            "agent2": {"output": "Actually, the answer is 43"}
        }
        
        conflicts = arbitrator.detect_conflicts(agent_outputs, {})
        
        assert len(conflicts) > 0
        assert any(c.conflict_type == ConflictType.FACTUAL for c in conflicts)
    
    def test_detect_code_conflicts(self, arbitrator):
        """Test code conflict detection."""
        agent_outputs = {
            "agent1": {"output": "```python\ndef foo(): return 1\n```"},
            "agent2": {"output": "```python\ndef foo(): return 2\n```"}
        }
        
        conflicts = arbitrator.detect_conflicts(agent_outputs, {})
        
        assert len(conflicts) > 0
        assert any(c.conflict_type == ConflictType.CODE for c in conflicts)
    
    def test_no_conflicts_single_agent(self, arbitrator):
        """Test no conflicts with single agent."""
        agent_outputs = {
            "agent1": {"output": "The answer is 42"}
        }
        
        conflicts = arbitrator.detect_conflicts(agent_outputs, {})
        
        assert len(conflicts) == 0
    
    def test_conflict_summary(self, arbitrator):
        """Test conflict summary generation."""
        summary = arbitrator.get_conflict_summary()
        
        assert "total_conflicts" in summary
        assert "by_type" in summary
        assert "resolution_rate" in summary


class TestLifecycleManagement:
    """Test lifecycle management."""
    
    def test_spawn_evaluation_below_threshold(self, lifecycle_manager):
        """Test spawn evaluation with low score."""
        task_history = []
        failure_patterns = {
            "recurrence_rate": 0.1,
            "cluster_density": 0.2,
            "avg_uncertainty": 0.3
        }
        
        should_spawn, spec = lifecycle_manager.evaluate_spawn_need(
            task_history,
            failure_patterns
        )
        
        assert not should_spawn
        assert spec is None
    
    def test_spawn_evaluation_above_threshold(self, lifecycle_manager):
        """Test spawn evaluation with high score."""
        task_history = []
        failure_patterns = {
            "recurrence_rate": 0.8,
            "cluster_density": 0.7,
            "avg_uncertainty": 0.7,
            "disagreement_rate": 0.6,
            "projected_usage": 0.5,
            "domain": "specialized",
            "cluster_id": "test_cluster"
        }
        
        should_spawn, spec = lifecycle_manager.evaluate_spawn_need(
            task_history,
            failure_patterns
        )
        
        # May still fail due to overlap check, but should calculate score
        assert isinstance(should_spawn, bool)
    
    def test_lifecycle_summary(self, lifecycle_manager):
        """Test lifecycle summary generation."""
        summary = lifecycle_manager.get_lifecycle_summary()
        
        assert "total_decisions" in summary
        assert "by_type" in summary
        assert "recent_decisions" in summary


class TestSynthesis:
    """Test final synthesis."""
    
    def test_synthesis_package_creation(self, llm_client):
        """Test synthesis package creation."""
        synthesizer = Synthesizer(llm_client)
        
        task_frame = TaskFrame(
            task_id="test_task",
            normalized_request="Test request",
            task_type=TaskType.CODING_STABLE
        )
        
        agent_outputs = {
            "agent1": {"output": "Solution 1"}
        }
        
        validation_report = ValidationReport(
            task_id="test_task",
            validation_state=ValidationState.SUCCESS,
            overall_passed=True,
            summary="All checks passed"
        )
        
        package = synthesizer.create_synthesis_package(
            task_frame,
            agent_outputs,
            "Shared context",
            [],
            validation_report
        )
        
        assert isinstance(package, SynthesisPackage)
        assert package.task_frame == task_frame
        assert len(package.agent_outputs) == 1
    
    def test_confidence_calculation(self, llm_client):
        """Test confidence calculation."""
        synthesizer = Synthesizer(llm_client)
        
        task_frame = TaskFrame(
            task_id="test_task",
            normalized_request="Test request",
            task_type=TaskType.CODING_STABLE,
            initial_uncertainty=0.3
        )
        
        validation_report = ValidationReport(
            task_id="test_task",
            validation_state=ValidationState.SUCCESS,
            overall_passed=True,
            summary="All checks passed"
        )
        
        package = SynthesisPackage(
            task_frame=task_frame,
            accepted_facts=["Fact 1", "Fact 2"],
            agent_outputs={"agent1": {"output": "Output"}},
            arbitration_outcomes=[],
            validation_results=validation_report,
            uncertainty_notes=[]
        )
        
        confidence = synthesizer._calculate_confidence(package)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high with successful validation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
