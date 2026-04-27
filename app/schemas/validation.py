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


# ── Completion Contract ──────────────────────────────────────────────────────

class CompletionContract(BaseModel):
    """Scored completion gate — every primary result must pass through this
    before the orchestrator considers the task done.

    Each dimension is scored 0.0–1.0.  The ``overall`` score is a weighted
    aggregate.  The orchestrator compares ``overall`` against a threshold
    (default 0.6) to decide whether to escalate or stop.
    """

    explicit_requirements_satisfied: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="How well the explicit user asks are covered",
    )
    implied_requirements_satisfied: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Correctness / usability requirements the user didn't state",
    )
    edge_cases_addressed: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Coverage of failure modes and boundary conditions",
    )
    validation_evidence_present: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Tests, assertions, or other proof of correctness",
    )
    abstraction_opportunity_addressed: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Reusable design / generalisation where appropriate",
    )
    assumptions_declared: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Explicit listing of what was assumed",
    )
    stopping_justified: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence that further work would not materially improve the result",
    )

    overall: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Weighted aggregate of all dimensions",
    )

    escalation_reasons: List[str] = Field(
        default_factory=list,
        description="Why the contract is not yet satisfied",
    )

    def compute_overall(self) -> float:
        """Recompute the weighted aggregate and return it."""
        self.overall = round(
            0.25 * self.explicit_requirements_satisfied
            + 0.20 * self.implied_requirements_satisfied
            + 0.15 * self.edge_cases_addressed
            + 0.15 * self.validation_evidence_present
            + 0.10 * self.abstraction_opportunity_addressed
            + 0.08 * self.assumptions_declared
            + 0.07 * self.stopping_justified,
            3,
        )
        return self.overall


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

    # Completion contract (new)
    completion_contract: Optional[CompletionContract] = None
    
    overall_passed: bool = False
    summary: str = ""
    
    class Config:
        use_enum_values = True
