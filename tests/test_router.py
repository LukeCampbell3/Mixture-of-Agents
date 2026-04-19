"""Tests for router."""

import pytest
from app.router import Router
from app.schemas.registry import AgentRegistry, AgentSpec, LifecycleState
from app.models.llm_client import create_llm_client
from app.models.embeddings import EmbeddingGenerator
from app.models.uncertainty import UncertaintyEstimator


@pytest.fixture
def router():
    """Create router for testing."""
    registry = AgentRegistry()
    registry.add_agent(AgentSpec(
        agent_id="test_agent",
        name="Test Agent",
        description="Test agent",
        domain="coding",
        lifecycle_state=LifecycleState.HOT,
        tools=["test_tool"]
    ))
    
    llm_client = create_llm_client("openai")
    embedding_gen = EmbeddingGenerator()
    uncertainty_est = UncertaintyEstimator()
    
    return Router(registry, llm_client, embedding_gen, uncertainty_est)


def test_task_framing(router):
    """Test task framing."""
    task_frame = router.frame_task("Write a function")
    
    assert task_frame.task_id is not None
    assert task_frame.normalized_request == "Write a function"
    assert task_frame.task_type is not None


def test_task_classification(router):
    """Test task type classification."""
    from app.schemas.task_frame import TaskType
    
    # Coding task
    task_type = router._classify_task_type("Write a function to parse JSON")
    assert task_type == TaskType.CODING_STABLE
    
    # Research task
    task_type = router._classify_task_type("What is the latest version of Python?")
    assert task_type in [TaskType.RESEARCH_LOW_AMBIGUITY, TaskType.CODING_CURRENT]


def test_agent_scoring(router):
    """Test agent scoring."""
    task_frame = router.frame_task("Write code")
    agent = router.registry.agents["test_agent"]
    
    score = router._score_agent(agent, task_frame)
    
    assert score.agent_id == "test_agent"
    assert 0.0 <= score.activation_score <= 1.0
    assert score.reason is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
