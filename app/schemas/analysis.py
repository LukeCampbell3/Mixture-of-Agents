"""Analysis schema for router and lifecycle state."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class AgentScore(BaseModel):
    """Score for a single agent candidate."""
    
    agent_id: str
    activation_score: float
    capability_match: float
    expected_quality_gain: float
    token_cost: float
    latency_cost: float
    overlap_penalty: float
    reason: str


class RoutingDecision(BaseModel):
    """Router decision output."""
    
    task_id: str
    candidate_agents: List[AgentScore]
    selected_agents: List[str]
    suppressed_agents: List[Dict[str, str]]  # agent_id -> reason
    budget_plan: Dict[str, float]
    routing_reasons: List[str]
    uncertainty_summary: str
    spawn_recommendation: Optional[str] = None
    no_spawn_reason: Optional[str] = None
    arbitration_needed: bool = False


class BudgetStatus(BaseModel):
    """Current budget consumption."""
    
    max_active_agents: int
    active_agents: int
    max_prompt_tokens: int
    used_prompt_tokens: int
    max_retrieval_calls: int
    used_retrieval_calls: int
    max_validation_passes: int
    used_validation_passes: int
    max_wall_clock_seconds: float
    elapsed_seconds: float
    budget_exhausted: bool = False


class Analysis(BaseModel):
    """Machine-readable router and lifecycle state."""
    
    task_id: str
    routing_decision: RoutingDecision
    budget_status: BudgetStatus
    conflict_markers: List[str] = Field(default_factory=list)
    validation_status: str = "pending"
    spawn_signals: List[str] = Field(default_factory=list)
    prune_signals: List[str] = Field(default_factory=list)
