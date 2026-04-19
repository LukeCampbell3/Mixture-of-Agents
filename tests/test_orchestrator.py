"""Tests for orchestrator."""

import pytest
from app.orchestrator import Orchestrator
from app.schemas.validation import ValidationState


def test_orchestrator_initialization():
    """Test orchestrator can be initialized."""
    orchestrator = Orchestrator(llm_provider="openai", budget_mode="low")
    assert orchestrator is not None
    assert orchestrator.registry is not None
    assert len(orchestrator.registry.agents) == 3  # Default agents


def test_task_execution_coding():
    """Test task execution for coding task."""
    orchestrator = Orchestrator(llm_provider="openai", budget_mode="balanced")
    
    # This will fail without API key, but tests the flow
    try:
        result = orchestrator.run_task("Write a function to calculate fibonacci numbers")
        assert result.task_id is not None
        assert result.final_state in [state.value for state in ValidationState]
    except Exception as e:
        # Expected if no API key
        assert "api" in str(e).lower() or "key" in str(e).lower()


def test_task_framing():
    """Test task framing."""
    orchestrator = Orchestrator(llm_provider="openai")
    task_frame = orchestrator.router.frame_task("Write a Python function to sort a list")
    
    assert task_frame.task_id is not None
    assert task_frame.normalized_request == "Write a Python function to sort a list"
    assert task_frame.task_type is not None


def test_agent_routing():
    """Test agent routing."""
    orchestrator = Orchestrator(llm_provider="openai")
    task_frame = orchestrator.router.frame_task("Debug this code")
    
    routing_decision = orchestrator.router.route(task_frame, max_agents=2)
    
    assert routing_decision.task_id == task_frame.task_id
    assert len(routing_decision.selected_agents) <= 2
    assert len(routing_decision.candidate_agents) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
