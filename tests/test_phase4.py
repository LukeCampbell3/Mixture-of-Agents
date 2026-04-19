"""Tests for Phase 4: Spawn Recommendation."""

import pytest
import numpy as np
from app.cluster_analyzer import ClusterAnalyzer, TaskCluster
from app.gap_analyzer import GapAnalyzer
from app.agent_factory import AgentFactory, DynamicAgent
from app.schemas.registry import AgentRegistry, AgentSpec, LifecycleState
from app.schemas.task_frame import TaskFrame, TaskType
from app.models.embeddings import EmbeddingGenerator
from app.models.llm_client import create_llm_client


@pytest.fixture
def embedding_generator():
    """Create embedding generator for testing."""
    return EmbeddingGenerator()


@pytest.fixture
def cluster_analyzer(embedding_generator):
    """Create cluster analyzer for testing."""
    return ClusterAnalyzer(embedding_generator, min_cluster_size=3)


@pytest.fixture
def registry():
    """Create test registry."""
    registry = AgentRegistry()
    registry.add_agent(AgentSpec(
        agent_id="code_primary",
        name="Code Primary",
        description="Primary coding agent",
        domain="coding",
        lifecycle_state=LifecycleState.HOT,
        tools=["repo_tool", "test_runner"]
    ))
    return registry


@pytest.fixture
def gap_analyzer(registry, embedding_generator):
    """Create gap analyzer for testing."""
    return GapAnalyzer(registry, embedding_generator)


@pytest.fixture
def llm_client():
    """Create LLM client for testing."""
    return create_llm_client("openai")


@pytest.fixture
def agent_factory(llm_client):
    """Create agent factory for testing."""
    return AgentFactory(llm_client)


class TestClusterAnalyzer:
    """Test cluster analysis."""
    
    def test_add_task_result(self, cluster_analyzer):
        """Test adding task results."""
        task_frame = TaskFrame(
            task_id="task1",
            normalized_request="Write a function",
            task_type=TaskType.CODING_STABLE
        )
        
        cluster_analyzer.add_task_result(
            task_frame,
            "success",
            {},
            0.8
        )
        
        assert len(cluster_analyzer.task_history) == 1
    
    def test_cluster_creation(self, cluster_analyzer):
        """Test cluster creation from similar tasks."""
        # Add similar tasks
        for i in range(5):
            task_frame = TaskFrame(
                task_id=f"task{i}",
                normalized_request="Write a Python function to sort a list",
                task_type=TaskType.CODING_STABLE
            )
            
            cluster_analyzer.add_task_result(
                task_frame,
                "validation_failure" if i % 2 == 0 else "success",
                {},
                0.5
            )
        
        # Should create at least one cluster
        assert len(cluster_analyzer.clusters) > 0
    
    def test_spawn_opportunity_detection(self, cluster_analyzer):
        """Test spawn opportunity detection."""
        # Add tasks with high failure rate
        for i in range(6):
            task_frame = TaskFrame(
                task_id=f"task{i}",
                normalized_request="Complex specialized task",
                task_type=TaskType.CODING_STABLE
            )
            
            cluster_analyzer.add_task_result(
                task_frame,
                "validation_failure",  # All failures
                {},
                0.3
            )
        
        opportunities = cluster_analyzer.detect_spawn_opportunities()
        
        # May or may not detect opportunities depending on clustering
        assert isinstance(opportunities, list)
    
    def test_cluster_summary(self, cluster_analyzer):
        """Test cluster summary generation."""
        summary = cluster_analyzer.get_cluster_summary()
        
        assert "total_clusters" in summary
        assert "total_tasks" in summary
        assert "clusters" in summary


class TestGapAnalyzer:
    """Test gap analysis."""
    
    def test_gap_analysis_new_domain(self, gap_analyzer):
        """Test gap analysis for new domain."""
        cluster_info = {
            "cluster_id": "test_cluster",
            "domain": "new_domain",
            "projected_usage": 0.5,
            "cluster_size": 10
        }
        
        analysis = gap_analyzer.analyze_gap(
            "new_domain",
            ["new_tool"],
            cluster_info
        )
        
        assert "gap_score" in analysis
        assert "coverage_gap" in analysis
        assert "recommendation" in analysis
        assert 0.0 <= analysis["gap_score"] <= 1.0
    
    def test_gap_analysis_existing_domain(self, gap_analyzer):
        """Test gap analysis for existing domain."""
        cluster_info = {
            "cluster_id": "test_cluster",
            "domain": "coding",  # Same as existing agent
            "projected_usage": 0.3,
            "cluster_size": 5
        }
        
        analysis = gap_analyzer.analyze_gap(
            "coding",
            ["repo_tool"],  # Same tool as existing agent
            cluster_info
        )
        
        # Should have high overlap
        assert analysis["max_overlap"] > 0.5
    
    def test_merge_candidates(self, gap_analyzer):
        """Test merge candidate suggestion."""
        candidates = gap_analyzer.suggest_merge_candidates("code_primary")
        
        # Should return list (may be empty)
        assert isinstance(candidates, list)


class TestAgentFactory:
    """Test agent factory."""
    
    def test_synthesize_agent_spec(self, agent_factory):
        """Test agent spec synthesis."""
        cluster_info = {
            "cluster_id": "test_cluster",
            "domain": "specialized",
            "required_tools": ["tool1", "tool2"],
            "projected_usage": 0.5,
            "cluster_size": 10,
            "spawn_reason": "High failure rate"
        }
        
        gap_analysis = {
            "gap_score": 0.8,
            "coverage_gap": 0.7
        }
        
        spec = agent_factory.synthesize_agent_spec(cluster_info, gap_analysis)
        
        assert "agent_id" in spec
        assert "name" in spec
        assert "description" in spec
        assert spec["domain"] == "specialized"
        assert len(spec["tools"]) == 2
    
    def test_create_agent_from_spec(self, agent_factory):
        """Test agent creation from spec."""
        spawn_spec = {
            "agent_id": "test_spawned",
            "name": "Test Spawned Agent",
            "description": "Test description",
            "domain": "testing",
            "tools": ["test_tool"],
            "tags": ["test"],
            "target_cluster": "test_cluster",
            "expected_activation_rate": 0.1
        }
        
        agent_spec = agent_factory.create_agent_from_spec(spawn_spec)
        
        assert agent_spec.agent_id == "test_spawned"
        assert agent_spec.lifecycle_state == LifecycleState.PROBATIONARY
    
    def test_create_agent_instance(self, agent_factory):
        """Test agent instance creation."""
        agent_spec = AgentSpec(
            agent_id="test_agent",
            name="Test Agent",
            description="Test description",
            domain="testing",
            lifecycle_state=LifecycleState.PROBATIONARY
        )
        
        agent = agent_factory.create_agent_instance(agent_spec)
        
        assert isinstance(agent, DynamicAgent)
        assert agent.agent_id == "test_agent"
        assert agent.get_system_prompt() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
