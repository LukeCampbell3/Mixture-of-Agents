"""Agent registry schema."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class LifecycleState(str, Enum):
    """Agent lifecycle states."""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    DORMANT = "dormant"
    MANUAL_ONLY = "manual_only"
    PROBATIONARY = "probationary"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class AgentSpec(BaseModel):
    """Specification for a single agent."""
    
    agent_id: str
    name: str
    description: str
    domain: str
    lifecycle_state: LifecycleState
    tools: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    parent_lineage: Optional[str] = None
    target_cluster: Optional[str] = None
    expected_activation_rate: float = 0.0
    
    # Performance metrics
    total_activations: int = 0
    successful_activations: int = 0
    average_quality_lift: float = 0.0
    average_token_cost: float = 0.0
    calibration_score: float = 0.5
    
    # Overlap metrics
    overlap_with: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class AgentRegistry(BaseModel):
    """Registry of all agents in the system."""
    
    version: str = "1.0.0"
    agents: Dict[str, AgentSpec] = Field(default_factory=dict)
    last_updated: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def get_agent(self, agent_id: str) -> Optional[AgentSpec]:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    def add_agent(self, agent: AgentSpec) -> None:
        """Add or update agent."""
        agent.updated_at = datetime.utcnow().isoformat()
        self.agents[agent.agent_id] = agent
        self.last_updated = datetime.utcnow().isoformat()
    
    def get_active_agents(self) -> List[AgentSpec]:
        """Get all non-archived agents."""
        return [
            agent for agent in self.agents.values()
            if agent.lifecycle_state not in [LifecycleState.ARCHIVED, LifecycleState.DEPRECATED]
        ]
    
    def get_routable_agents(self) -> List[AgentSpec]:
        """Get agents eligible for auto-routing."""
        return [
            agent for agent in self.agents.values()
            if agent.lifecycle_state in [
                LifecycleState.HOT, LifecycleState.WARM,
                LifecycleState.COLD, LifecycleState.PROBATIONARY
            ]
        ]
