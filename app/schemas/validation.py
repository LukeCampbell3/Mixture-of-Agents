"""Validation schema."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ValidationState(str, Enum):
    """Validation outcome states."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    VALIDATION_FAILURE = "validation_failure"
    TOOL_FAILURE = "tool_failure"
    BUDGET_CUTOFF = "budget_cutoff"
    SYNTHESIS_FAILURE = "synthesis_failure"


class ValidationCheck(BaseModel):
    """Single validation check result."""
    
    check_name: str
    passed: bool
    severity: str  # "error", "warning", "info"
    message: str
    details: Optional[Dict] = None


class ValidationReport(BaseModel):
    """Per-run validation summary."""
    
    task_id: str
    validation_state: ValidationState
    checks: List[ValidationCheck] = Field(default_factory=list)
    
    # Task-specific validation
    compile_result: Optional[bool] = None
    test_results: Optional[Dict] = None
    citation_coverage: Optional[float] = None
    contradiction_count: int = 0
    unsupported_claim_count: int = 0
    
    # Agent-specific validation
    agent_validations: Dict[str, Dict] = Field(default_factory=dict)
    
    overall_passed: bool = False
    summary: str = ""
    
    class Config:
        use_enum_values = True
