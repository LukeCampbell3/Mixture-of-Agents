"""Memory schema for persistent storage."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class MemoryType(str, Enum):
    """Types of memory items."""
    PROCEDURAL = "procedural"
    USER_PREFERENCE = "user_preference"
    AGENT_PERFORMANCE = "agent_performance"
    TASK_CLUSTER = "task_cluster"
    CODE_PATTERN = "code_pattern"


class MemoryItem(BaseModel):
    """Single memory item."""
    
    memory_id: str
    memory_type: MemoryType
    content: str
    source_type: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    freshness_horizon_days: Optional[int] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    invalidation_trigger: Optional[str] = None
    provenance: str = ""
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


class MemoryCandidates(BaseModel):
    """Items proposed for persistent storage."""
    
    task_id: str
    candidates: List[MemoryItem] = Field(default_factory=list)
    admission_decisions: List[Dict[str, str]] = Field(default_factory=list)
