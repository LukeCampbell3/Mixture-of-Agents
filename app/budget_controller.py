"""Budget and execution controller."""

from typing import Dict, Any
from app.schemas.analysis import BudgetStatus
import time


class BudgetMode:
    """Budget profiles for different execution modes."""
    
    LOW = {
        "max_active_agents": 1,
        "max_prompt_tokens": 5000,
        "max_retrieval_calls": 2,
        "max_validation_passes": 1,
        "max_wall_clock_seconds": 300.0,
        "reserved_synthesis_tokens": 1000  # 20% reserved for synthesis
    }
    
    BALANCED = {
        "max_active_agents": 3,
        "max_prompt_tokens": 15000,
        "max_retrieval_calls": 5,
        "max_validation_passes": 2,
        "max_wall_clock_seconds": 180.0,
        "reserved_synthesis_tokens": 3000  # 20% reserved for synthesis
    }
    
    THOROUGH = {
        "max_active_agents": 5,
        "max_prompt_tokens": 30000,
        "max_retrieval_calls": 10,
        "max_validation_passes": 3,
        "max_wall_clock_seconds": 600.0,
        "reserved_synthesis_tokens": 6000  # 20% reserved for synthesis
    }


class BudgetController:
    """Enforce hard operational limits on task execution."""
    
    def __init__(self, mode: str = "balanced"):
        """Initialize budget controller.
        
        Args:
            mode: Budget mode - "low", "balanced", or "thorough"
        """
        self.mode = mode
        limits = self._get_limits(mode)
        
        self.status = BudgetStatus(
            max_active_agents=limits["max_active_agents"],
            active_agents=0,
            max_prompt_tokens=limits["max_prompt_tokens"],
            used_prompt_tokens=0,
            max_retrieval_calls=limits["max_retrieval_calls"],
            used_retrieval_calls=0,
            max_validation_passes=limits["max_validation_passes"],
            used_validation_passes=0,
            max_wall_clock_seconds=limits["max_wall_clock_seconds"],
            elapsed_seconds=0.0
        )
        
        self.reserved_synthesis_tokens = limits["reserved_synthesis_tokens"]
        self.start_time = time.time()
    
    def _get_limits(self, mode: str) -> Dict[str, Any]:
        """Get limits for mode."""
        if mode == "low":
            return BudgetMode.LOW
        elif mode == "thorough":
            return BudgetMode.THOROUGH
        else:
            return BudgetMode.BALANCED
    
    def can_activate_agent(self) -> bool:
        """Check if another agent can be activated."""
        self._check_exhaustion()
        return (
            self.status.active_agents < self.status.max_active_agents
            and not self.status.budget_exhausted
        )
    
    def can_make_retrieval_call(self) -> bool:
        """Check if another retrieval call is allowed."""
        return (
            self.status.used_retrieval_calls < self.status.max_retrieval_calls
            and not self.status.budget_exhausted
        )
    
    def can_run_validation(self) -> bool:
        """Check if another validation pass is allowed."""
        return (
            self.status.used_validation_passes < self.status.max_validation_passes
            and not self.status.budget_exhausted
        )
    
    def can_use_tokens(self, token_count: int) -> bool:
        """Check if token usage is within budget (with synthesis reserve)."""
        # Reserve tokens for final synthesis
        available = self.status.max_prompt_tokens - self.reserved_synthesis_tokens
        return (
            self.status.used_prompt_tokens + token_count <= available
            and not self.status.budget_exhausted
        )
    
    def can_afford_synthesis(self) -> bool:
        """Check if synthesis budget is available."""
        remaining = self.status.max_prompt_tokens - self.status.used_prompt_tokens
        return remaining >= self.reserved_synthesis_tokens
    
    def should_early_exit(self, confidence: float, threshold: float = 0.9) -> bool:
        """Check if execution should exit early based on calibrated confidence.
        
        Args:
            confidence: Calibrated confidence score
            threshold: Confidence threshold for early exit
        
        Returns:
            True if should exit early
        """
        # Only exit early if we have budget remaining and confidence is high
        if self.status.budget_exhausted:
            return False
        
        return confidence >= threshold
    
    def predict_overrun(self, projected_tokens: int) -> bool:
        """Predict if execution will overrun budget.
        
        Args:
            projected_tokens: Projected additional token usage
        
        Returns:
            True if overrun predicted
        """
        total_projected = self.status.used_prompt_tokens + projected_tokens
        return total_projected > self.status.max_prompt_tokens
    
    def downgrade_mode(self):
        """Downgrade budget mode to prevent overrun mid-execution."""
        if self.mode == "thorough":
            self.mode = "balanced"
        elif self.mode == "balanced":
            self.mode = "low"
        else:
            return  # Already at lowest
        
        # Update limits
        limits = self._get_limits(self.mode)
        self.status.max_active_agents = limits["max_active_agents"]
        self.status.max_prompt_tokens = limits["max_prompt_tokens"]
        self.status.max_retrieval_calls = limits["max_retrieval_calls"]
        self.status.max_validation_passes = limits["max_validation_passes"]
        self.status.max_wall_clock_seconds = limits["max_wall_clock_seconds"]
        self.reserved_synthesis_tokens = limits["reserved_synthesis_tokens"]
    
    def activate_agent(self) -> None:
        """Record agent activation."""
        self.status.active_agents += 1
        self._check_exhaustion()
    
    def deactivate_agent(self) -> None:
        """Record agent deactivation."""
        self.status.active_agents = max(0, self.status.active_agents - 1)
    
    def record_retrieval_call(self) -> None:
        """Record a retrieval call."""
        self.status.used_retrieval_calls += 1
        self._check_exhaustion()
    
    def record_validation_pass(self) -> None:
        """Record a validation pass."""
        self.status.used_validation_passes += 1
        self._check_exhaustion()
    
    def record_token_usage(self, token_count: int) -> None:
        """Record token usage."""
        self.status.used_prompt_tokens += token_count
        self._check_exhaustion()
    
    def update_elapsed_time(self) -> None:
        """Update elapsed time."""
        self.status.elapsed_seconds = time.time() - self.start_time
    
    def _check_exhaustion(self) -> None:
        """Check if budget is exhausted.
        
        Budget is exhausted when any hard limit is reached.
        Note: active_agents hitting max is normal operation for low-budget mode,
        not an overrun. We track it separately.
        """
        # Update elapsed time without recursion
        self.status.elapsed_seconds = time.time() - self.start_time
        
        if (
            self.status.used_prompt_tokens >= self.status.max_prompt_tokens
            or self.status.used_retrieval_calls >= self.status.max_retrieval_calls
            or self.status.used_validation_passes >= self.status.max_validation_passes
            or self.status.elapsed_seconds >= self.status.max_wall_clock_seconds
        ):
            self.status.budget_exhausted = True
    
    def get_status(self) -> BudgetStatus:
        """Get current budget status."""
        self._check_exhaustion()
        return self.status
    
    def get_remaining_agents(self) -> int:
        """Get number of agents that can still be activated."""
        return max(0, self.status.max_active_agents - self.status.active_agents)
    
    def start_execution(self) -> None:
        """Start execution timer."""
        self.start_time = time.time()
        self.status.elapsed_seconds = 0.0
