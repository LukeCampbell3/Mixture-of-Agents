"""Tests for Phase 5 & 6: Shadow Evaluation and Memory Management."""

import pytest
from app.shadow_evaluator import ShadowEvaluator, ShadowEvaluation
from app.memory_manager import MemoryManager
from app.schemas.memory import MemoryItem, MemoryType
from app.schemas.task_frame import TaskFrame, TaskType
from app.agents.base_agent import BaseAgent
from app.models.llm_client import create_llm_client


@pytest.fixture
def shadow_evaluator():
    """Create shadow evaluator for testing."""
    return ShadowEvaluator()


@pytest.fixture
def memory_manager(tmp_path):
    """Create memory manager for testing."""
    return MemoryManager(data_dir=str(tmp_path))


@pytest.fixture
def llm_client():
    """Create LLM client for testing."""
    return create_llm_client("openai")


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def get_system_prompt(self) -> str:
        return "Test agent"
    
    def execute(self, task_context):
        return {
            "output": "Test output",
            "confidence": 0.8,
            "tool_calls": [],
            "reasoning": "Test reasoning"
        }


class TestShadowEvaluator:
    """Test shadow evaluation."""
    
    def test_shadow_evaluation_creation(self, shadow_evaluator):
        """Test shadow evaluation creation."""
        evaluation = ShadowEvaluation(
            agent_id="test_agent",
            task_id="test_task",
            shadow_output={"output": "Shadow output"},
            active_outputs={"agent1": {"output": "Active output"}}
        )
        
        assert evaluation.agent_id == "test_agent"
        assert evaluation.task_id == "test_task"
    
    def test_quality_lift_calculation(self, shadow_evaluator):
        """Test quality lift calculation."""
        shadow_output = {"output": "High quality output", "confidence": 0.9}
        active_outputs = {"agent1": {"output": "Lower quality", "confidence": 0.6}}
        
        quality_lift = shadow_evaluator._evaluate_quality_lift(
            shadow_output,
            active_outputs,
            {}
        )
        
        assert 0.0 <= quality_lift <= 1.0
    
    def test_redundancy_calculation(self, shadow_evaluator):
        """Test redundancy calculation."""
        shadow_output = {"output": "The answer is 42"}
        active_outputs = {"agent1": {"output": "The answer is 42"}}
        
        redundancy = shadow_evaluator._evaluate_redundancy(
            shadow_output,
            active_outputs
        )
        
        # Should be high redundancy (identical outputs)
        assert redundancy > 0.5
    
    def test_promotion_readiness_insufficient_data(self, shadow_evaluator):
        """Test promotion readiness with insufficient data."""
        readiness = shadow_evaluator.get_promotion_readiness("unknown_agent")
        
        assert not readiness["ready"]
        assert "Insufficient" in readiness["reason"]
    
    def test_evaluation_summary(self, shadow_evaluator):
        """Test evaluation summary generation."""
        summary = shadow_evaluator.get_evaluation_summary("test_agent")
        
        assert "agent_id" in summary
        assert "evaluations" in summary


class TestMemoryManager:
    """Test memory management."""
    
    def test_memory_admission_high_quality(self, memory_manager):
        """Test admission of high-quality memory."""
        memory = MemoryItem(
            memory_id="test_memory",
            memory_type=MemoryType.PROCEDURAL,
            content="Always validate inputs before processing",
            source_type="validation_report",
            confidence=0.9
        )
        
        task_context = {"validation_state": "success"}
        
        decision = memory_manager.evaluate_admission(memory, task_context)
        
        # Should admit high-quality procedural knowledge
        assert isinstance(decision["admit"], bool)
        assert "reason" in decision
    
    def test_memory_admission_ephemeral(self, memory_manager):
        """Test rejection of ephemeral memory."""
        memory = MemoryItem(
            memory_id="test_memory",
            memory_type=MemoryType.PROCEDURAL,
            content="For this task_id only, use temporary approach",
            source_type="agent_output",
            confidence=0.8
        )
        
        task_context = {"validation_state": "success"}
        
        decision = memory_manager.evaluate_admission(memory, task_context)
        
        # Should reject ephemeral content
        assert not decision["admit"]
        assert "ephemeral" in decision["reason"].lower()
    
    def test_memory_retrieval(self, memory_manager):
        """Test memory retrieval."""
        # Add some memories
        memory1 = MemoryItem(
            memory_id="mem1",
            memory_type=MemoryType.CODE_PATTERN,
            content="Use list comprehension for filtering",
            source_type="validated",
            confidence=0.9
        )
        
        memory_manager.admit_memory(memory1)
        
        # Retrieve
        results = memory_manager.retrieve_relevant_memories(
            "How to filter a list",
            MemoryType.CODE_PATTERN
        )
        
        assert len(results) > 0
    
    def test_freshness_checking(self, memory_manager):
        """Test freshness checking."""
        memory = MemoryItem(
            memory_id="test_memory",
            memory_type=MemoryType.PROCEDURAL,
            content="Test content",
            source_type="test",
            confidence=0.8,
            freshness_horizon_days=30
        )
        
        # Fresh memory
        assert memory_manager._is_fresh(memory)
    
    def test_memory_stats(self, memory_manager):
        """Test memory statistics."""
        stats = memory_manager.get_memory_stats()
        
        assert "total_memories" in stats
        assert "by_type" in stats
        assert "fresh" in stats
        assert "stale" in stats


class TestIntegration:
    """Integration tests for Phase 5 & 6."""
    
    def test_shadow_to_memory_pipeline(
        self,
        shadow_evaluator,
        memory_manager,
        llm_client
    ):
        """Test pipeline from shadow evaluation to memory admission."""
        # Create mock agent
        agent = MockAgent(
            agent_id="test_agent",
            name="Test Agent",
            description="Test",
            llm_client=llm_client
        )
        
        # Run shadow evaluation
        task_frame = TaskFrame(
            task_id="test_task",
            normalized_request="Test request",
            task_type=TaskType.CODING_STABLE
        )
        
        task_context = {
            "task_frame": task_frame,
            "shared_context": "",
            "constraints": []
        }
        
        active_outputs = {
            "agent1": {"output": "Active output", "confidence": 0.7}
        }
        
        evaluation = shadow_evaluator.run_shadow_evaluation(
            agent,
            task_context,
            active_outputs
        )
        
        # Create memory from evaluation
        if evaluation.quality_lift > 0.5:
            memory = MemoryItem(
                memory_id=f"eval_{evaluation.task_id}",
                memory_type=MemoryType.AGENT_PERFORMANCE,
                content=f"Agent {evaluation.agent_id} showed quality lift of {evaluation.quality_lift:.2f}",
                source_type="shadow_evaluation",
                confidence=0.8
            )
            
            # Evaluate admission
            decision = memory_manager.evaluate_admission(
                memory,
                {"validation_state": "success"}
            )
            
            assert "admit" in decision


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
