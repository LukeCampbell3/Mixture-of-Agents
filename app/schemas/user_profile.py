"""User profile schema for user-aware control."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class TaskCategoryStats(BaseModel):
    """Statistics for a task category."""
    
    category: str
    count: int = 0
    recent_count: int = 0  # Last 30 days
    avg_quality_gain: Dict[str, float] = Field(default_factory=dict)  # agent_id -> gain
    avg_cost: Dict[str, float] = Field(default_factory=dict)  # agent_id -> cost
    last_seen: Optional[str] = None


class AgentPreference(BaseModel):
    """User preference for an agent."""
    
    agent_id: str
    pinned: bool = False  # Always activate
    disabled: bool = False  # Never activate
    preferred_for_categories: List[str] = Field(default_factory=list)
    custom_activation_threshold: Optional[float] = None


class BudgetPreference(BaseModel):
    """User budget preferences."""
    
    default_mode: str = "balanced"  # low, balanced, thorough
    max_agents_override: Optional[int] = None
    max_tokens_override: Optional[int] = None
    max_time_override: Optional[float] = None
    latency_tolerance: str = "normal"  # low, normal, high


class UserProfile(BaseModel):
    """User-specific profile for adaptive agent control."""
    
    user_id: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Task distribution
    task_category_stats: Dict[str, TaskCategoryStats] = Field(default_factory=dict)
    total_tasks: int = 0
    
    # Agent preferences
    agent_preferences: Dict[str, AgentPreference] = Field(default_factory=dict)
    
    # Budget preferences
    budget_preference: BudgetPreference = Field(default_factory=BudgetPreference)
    
    # Validation preferences
    validation_thoroughness: str = "normal"  # minimal, normal, thorough
    
    # Learning rate (how quickly to adapt)
    adaptation_rate: float = 0.1  # 0-1, higher = faster adaptation
    
    def update_task_stats(
        self,
        category: str,
        agent_quality_gains: Dict[str, float],
        agent_costs: Dict[str, float]
    ) -> None:
        """Update task category statistics."""
        if category not in self.task_category_stats:
            self.task_category_stats[category] = TaskCategoryStats(category=category)
        
        stats = self.task_category_stats[category]
        stats.count += 1
        stats.recent_count += 1
        stats.last_seen = datetime.utcnow().isoformat()
        
        # Update agent-specific stats with exponential moving average
        for agent_id, gain in agent_quality_gains.items():
            if agent_id not in stats.avg_quality_gain:
                stats.avg_quality_gain[agent_id] = gain
            else:
                # Exponential moving average
                stats.avg_quality_gain[agent_id] = (
                    (1 - self.adaptation_rate) * stats.avg_quality_gain[agent_id] +
                    self.adaptation_rate * gain
                )
        
        for agent_id, cost in agent_costs.items():
            if agent_id not in stats.avg_cost:
                stats.avg_cost[agent_id] = cost
            else:
                stats.avg_cost[agent_id] = (
                    (1 - self.adaptation_rate) * stats.avg_cost[agent_id] +
                    self.adaptation_rate * cost
                )
        
        self.total_tasks += 1
        self.updated_at = datetime.utcnow().isoformat()
    
    def get_agent_preference(self, agent_id: str) -> AgentPreference:
        """Get preference for an agent."""
        if agent_id not in self.agent_preferences:
            self.agent_preferences[agent_id] = AgentPreference(agent_id=agent_id)
        return self.agent_preferences[agent_id]
    
    def pin_agent(self, agent_id: str, categories: Optional[List[str]] = None) -> None:
        """Pin an agent for automatic activation."""
        pref = self.get_agent_preference(agent_id)
        pref.pinned = True
        if categories:
            pref.preferred_for_categories = categories
        self.updated_at = datetime.utcnow().isoformat()
    
    def disable_agent(self, agent_id: str) -> None:
        """Disable an agent from automatic activation."""
        pref = self.get_agent_preference(agent_id)
        pref.disabled = True
        pref.pinned = False
        self.updated_at = datetime.utcnow().isoformat()
    
    def get_category_distribution(self) -> Dict[str, float]:
        """Get normalized task category distribution."""
        if not self.task_category_stats:
            return {}
        
        total = sum(stats.count for stats in self.task_category_stats.values())
        if total == 0:
            return {}
        
        return {
            category: stats.count / total
            for category, stats in self.task_category_stats.items()
        }
    
    def get_agent_quality_for_category(self, agent_id: str, category: str) -> float:
        """Get agent's average quality gain for a category."""
        if category not in self.task_category_stats:
            return 0.5  # Default
        
        stats = self.task_category_stats[category]
        return stats.avg_quality_gain.get(agent_id, 0.5)
    
    def get_agent_cost_for_category(self, agent_id: str, category: str) -> float:
        """Get agent's average cost for a category."""
        if category not in self.task_category_stats:
            return 1000.0  # Default
        
        stats = self.task_category_stats[category]
        return stats.avg_cost.get(agent_id, 1000.0)
