"""Task frame schema for structured task representation."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum


class TaskType(str, Enum):
    """Task category classification."""
    CODING_STABLE = "coding_stable"
    CODING_CURRENT = "coding_current"
    RESEARCH_LOW_AMBIGUITY = "research_low_ambiguity"
    RESEARCH_HIGH_AMBIGUITY = "research_high_ambiguity"
    HYBRID = "hybrid"
    PLANNING = "planning"
    UNKNOWN = "unknown"


class TaskFrame(BaseModel):
    """Structured representation of a task."""
    
    task_id: str = Field(description="Unique task identifier")
    normalized_request: str = Field(description="Cleaned user request")
    task_type: TaskType = Field(description="Task category")
    required_outputs: List[str] = Field(default_factory=list, description="Expected output types")
    hard_constraints: List[str] = Field(default_factory=list, description="Must-satisfy constraints")
    soft_preferences: List[str] = Field(default_factory=list, description="Nice-to-have preferences")
    likely_tools: List[str] = Field(default_factory=list, description="Tools likely needed")
    difficulty_estimate: float = Field(default=0.5, ge=0.0, le=1.0, description="Task difficulty 0-1")
    initial_uncertainty: float = Field(default=0.5, ge=0.0, le=1.0, description="Initial uncertainty 0-1")
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Task novelty 0-1")
    retrieval_gap_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Need for retrieval 0-1")
    freshness_requirement: float = Field(default=0.0, ge=0.0, le=1.0, description="Need for current info 0-1")
    
    class Config:
        use_enum_values = True
