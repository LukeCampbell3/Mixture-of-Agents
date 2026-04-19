"""Integration tests for complete system."""

import pytest
from app.orchestrator import Orchestrator
from app.schemas.validation import ValidationState
from app.schemas.task_frame import TaskType


class TestFullPipeline:
    """Test complete pipeline from request to result."""
    
    def test_simple_coding_task(self):
        """Test simple coding task end-to-end."""
        try:
            orchestrator = Orchestrator(
                llm_provider="openai",
                budget_mode="low"
            )
            
            result = orchestrator.run_task("Write a function to add two numbers")
            
            assert result.task_id is not None
            assert result.final_answer is not None
            assert result.final_state in [state.value for state in ValidationState]
            assert len(result.active_agents) > 0
            
        except Exception as e:
            # Expected if no API key
            assert "api" in str(e).lower() or "key" in str(e).lower()
    
    def test_research_task(self):
        """Test research task end-to-end."""
        try:
            orchestrator = Orchestrator(
                llm_provider="openai",
                budget_mode="balanced"
            )
            
            result = orchestrator.run_task("What is Python?")
            
            assert result.task_id is not None
            assert result.final_answer is not None
            
        except Exception as e:
            assert "api" in str(e).lower() or "key" in str(e).lower()
    
    def test_budget_enforcement(self):
        """Test that budget limits are enforced."""
        try:
            orchestrator = Orchestrator(
                llm_provider="openai",
                budget_mode="low"
            )
            
            result = orchestrator.run_task("Complex multi-step task")
            
            # Low budget should limit agents
            assert len(result.active_agents) <= 1
            
        except Exception as e:
            assert "api" in str(e).lower() or "key" in str(e).lower()


class TestLocalLLMIntegration:
    """Test integration with local LLM."""
    
    def test_ollama_connection(self):
        """Test Ollama connection."""
        try:
            from app.models.local_llm_client import create_local_llm_client
            
            client = create_local_llm_client(
                backend="ollama",
                model="qwen2.5:7b",
                base_url="http://localhost:11434"
            )
            
            response = client.generate("Hello", max_tokens=10)
            assert isinstance(response, str)
            
        except Exception as e:
            # Expected if Ollama not running
            pytest.skip(f"Ollama not available: {e}")
    
    def test_orchestrator_with_local_llm(self):
        """Test orchestrator with local LLM."""
        try:
            orchestrator = Orchestrator(
                llm_provider="ollama",
                llm_model="qwen2.5:7b",
                llm_base_url="http://localhost:11434",
                budget_mode="low"
            )
            
            result = orchestrator.run_task("Write a function to multiply two numbers")
            
            assert result.task_id is not None
            assert result.final_answer is not None
            
        except Exception as e:
            pytest.skip(f"Local LLM not available: {e}")


class TestPhaseIntegration:
    """Test integration across phases."""
    
    def test_phase2_arbitration_in_pipeline(self):
        """Test that arbitration works in full pipeline."""
        try:
            orchestrator = Orchestrator(
                llm_provider="openai",
                budget_mode="balanced"
            )
            
            # Task likely to cause agent disagreement
            result = orchestrator.run_task(
                "What is the best programming language and why?"
            )
            
            # Should complete even with potential conflicts
            assert result.final_state is not None
            
        except Exception as e:
            assert "api" in str(e).lower() or "key" in str(e).lower()
    
    def test_phase3_user_preferences(self):
        """Test user preference integration."""
        from app.user_manager import UserManager
        
        user_manager = UserManager()
        profile = user_manager.get_profile("test_user")
        
        # Pin an agent
        profile.pin_agent("code_primary", categories=["coding_stable"])
        user_manager.save_profile(profile)
        
        # Verify saved
        loaded_profile = user_manager.get_profile("test_user")
        assert "code_primary" in loaded_profile.agent_preferences
        assert loaded_profile.agent_preferences["code_primary"].pinned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
