"""Run state schema for episode logging."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ToolCall(BaseModel):
    """Record of a tool invocation."""
    
    tool_name: str
    agent_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None
    duration_seconds: float = 0.0


class ConflictEvent(BaseModel):
    """Record of a conflict between agents."""
    
    conflict_type: str  # factual, code, plan, citation, validation
    agents_involved: List[str]
    description: str
    resolution: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class RunState(BaseModel):
    """Complete state of a task execution episode."""
    
    task_id: str
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    
    # Input
    task_frame: Dict
    
    # Execution
    active_agents: List[str]
    suppressed_agents: List[Dict[str, str]]
    tool_calls: List[ToolCall] = Field(default_factory=list)
    conflict_events: List[ConflictEvent] = Field(default_factory=list)
    
    # Budget
    budget_usage: Dict[str, Any]
    
    # Outputs
    synthesis_package: Optional[Dict] = None
    validation_report: Optional[Dict] = None
    final_state: str = "running"  # running, success, partial_success, failure
    final_answer: Optional[str] = None
    
    # Lifecycle tracking (NEW)
    spawn_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    spawned_agents: List[str] = Field(default_factory=list)
    probationary_agents_used: List[str] = Field(default_factory=list)
    promoted_agents: List[str] = Field(default_factory=list)
    pruned_agents: List[str] = Field(default_factory=list)
    demoted_agents: List[str] = Field(default_factory=list)
    lifecycle_events: List[Dict[str, Any]] = Field(default_factory=list)
    pool_size_before: int = 0
    pool_size_after: int = 0
    lifecycle_recommendations: List[str] = Field(default_factory=list)
    
    # Versioning
    base_model_version: str = "unknown"
    router_version: str = "1.0.0"
    agent_versions: Dict[str, str] = Field(default_factory=dict)
