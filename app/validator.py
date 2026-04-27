"""Validation layer for outputs and agents.

Checks both surface-level answer presence AND engineering completeness
via a scored CompletionContract.
"""

import re
from typing import Dict, Any, List

from app.schemas.validation import (
    ValidationReport,
    ValidationCheck,
    ValidationState,
    CompletionContract,
)
from app.schemas.task_frame import TaskFrame, TaskType


class Validator:
    """Validate task outputs and agent performance."""

    def validate_output(
        self,
        task_frame: TaskFrame,
        agent_outputs: Dict[str, Any],
        shared_context: str,
    ) -> ValidationReport:
        checks: List[ValidationCheck] = []

        task_type = task_frame.task_type
        task_type_str = task_type if isinstance(task_type, str) else task_type.value

        # ── Task-specific surface checks ─────────────────────────────────
        if "coding" in task_type_str:
            checks.extend(self._validate_code_output(agent_outputs, shared_context))
        if "research" in task_type_str:
            checks.extend(self._validate_research_output(agent_outputs, shared_context))
        if task_type_str == "hybrid":
            checks.extend(self._validate_code_output(agent_outputs, shared_context))
            checks.extend(self._validate_research_output(agent_outputs, shared_context))
        if task_type_str == "planning":
            checks.extend(self._validate_reasoning_output(agent_outputs, shared_context, task_frame))

        # ── General checks (always) ──────────────────────────────────────
        checks.extend(self._validate_completeness(task_frame, shared_context))
        checks.extend(self._validate_consistency(agent_outputs))

        # ── Engineering completeness checks ──────────────────────────────
        checks.extend(self._validate_engineering_completeness(task_frame, shared_context))

        # ── Completion contract ──────────────────────────────────────────
        contract = self._score_completion_contract(task_frame, shared_context)

        # ── Determine overall state ──────────────────────────────────────
        total_checks = len(checks)
        passed_checks = sum(1 for c in checks if c.passed)
        error_count = sum(1 for c in checks if not c.passed and c.severity == "error")
        warning_count = sum(1 for c in checks if not c.passed and c.severity == "warning")

        if error_count > 0:
            state = ValidationState.VALIDATION_FAILURE
            overall_passed = False
        elif total_checks > 0 and passed_checks * 2 <= total_checks:
            state = ValidationState.PARTIAL_SUCCESS
            overall_passed = False
        elif warning_count > 2:
            state = ValidationState.PARTIAL_SUCCESS
            overall_passed = True
        else:
            state = ValidationState.SUCCESS
            overall_passed = True

        return ValidationReport(
            task_id=task_frame.task_id,
            validation_state=state,
            checks=checks,
            overall_passed=overall_passed,
            completion_contract=contract,
            summary=self._generate_summary(checks, state),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Completion contract scoring
    # ──────────────────────────────────────────────────────────────────────

    def score_completion(
        self,
        task_frame: TaskFrame,
        output_text: str,
    ) -> CompletionContract:
        """Public entry point — score an output against the completion contract."""
        return self._score_completion_contract(task_frame, output_text)

    def _score_completion_contract(
        self,
        task_frame: TaskFrame,
        context: str,
    ) -> CompletionContract:
        """Score the output against every dimension of the completion contract."""
        text = context.lower()
        request = task_frame.normalized_request.lower()
        reasons: List[str] = []

        # 1. Explicit requirements satisfied
        request_words = set(request.split())
        context_words = set(text.split())
        overlap = len(request_words & context_words) / max(1, len(request_words))
        explicit = min(1.0, overlap * 1.3)  # slight boost
        if explicit < 0.5:
            reasons.append("Low overlap with explicit user requirements")

        # 2. Implied requirements satisfied
        implied_markers = [
            "error handling", "exception", "try", "except", "catch",
            "validate", "validation", "input check", "type check",
            "encoding", "cleanup", "close", "finally", "dispose",
            "timeout", "retry", "logging", "log",
        ]
        implied_hits = sum(1 for m in implied_markers if m in text)
        implied = min(1.0, implied_hits / 4.0)
        if implied < 0.3:
            reasons.append("Few implied requirements addressed (error handling, validation, cleanup)")

        # 3. Edge cases addressed
        edge_markers = [
            "edge case", "boundary", "empty", "none", "null", "zero",
            "negative", "overflow", "underflow", "unicode", "special char",
            "large input", "concurrent", "race condition", "timeout",
        ]
        edge_hits = sum(1 for m in edge_markers if m in text)
        edge = min(1.0, edge_hits / 3.0)
        if edge < 0.3:
            reasons.append("Edge cases not visibly addressed")

        # 4. Validation evidence present
        test_markers = [
            "test", "assert", "expect", "pytest", "unittest", "jest",
            "verify", "check", "should return", "should raise",
            "test case", "test_", "def test",
        ]
        test_hits = sum(1 for m in test_markers if m in text)
        validation = min(1.0, test_hits / 3.0)
        if validation < 0.3:
            reasons.append("No tests or validation evidence found")

        # 5. Abstraction opportunity addressed
        abstraction_markers = [
            "interface", "abstract", "protocol", "base class", "generic",
            "configurable", "parameter", "factory", "strategy", "plugin",
            "extensible", "reusable", "modular",
        ]
        abs_hits = sum(1 for m in abstraction_markers if m in text)
        abstraction = min(1.0, abs_hits / 2.0)
        # Only penalise if the task has high abstraction opportunity
        spec_density = getattr(task_frame, "specification_density", "medium")
        abs_opp = getattr(task_frame, "abstraction_opportunity", "medium")
        if abs_opp == "high" and abstraction < 0.3:
            reasons.append("High abstraction opportunity but solution is narrow")

        # 6. Assumptions declared
        assumption_markers = [
            "assumption", "assumes", "assuming", "presume", "given that",
            "we expect", "requirement:", "note:", "caveat", "limitation",
            "intentionally", "left out", "out of scope", "not included",
        ]
        asm_hits = sum(1 for m in assumption_markers if m in text)
        assumptions = min(1.0, asm_hits / 2.0)
        if assumptions < 0.3:
            reasons.append("Assumptions not explicitly declared")

        # 7. Stopping justified
        stopping_markers = [
            "complete", "done", "finished", "ready", "production",
            "how it works", "summary", "conclusion", "in summary",
        ]
        stop_hits = sum(1 for m in stopping_markers if m in text)
        is_substantive = len(context.strip()) > 200
        stopping = min(1.0, (stop_hits / 2.0) * 0.6 + (0.4 if is_substantive else 0.0))

        contract = CompletionContract(
            explicit_requirements_satisfied=round(explicit, 3),
            implied_requirements_satisfied=round(implied, 3),
            edge_cases_addressed=round(edge, 3),
            validation_evidence_present=round(validation, 3),
            abstraction_opportunity_addressed=round(abstraction, 3),
            assumptions_declared=round(assumptions, 3),
            stopping_justified=round(stopping, 3),
            escalation_reasons=reasons,
        )
        contract.compute_overall()
        return contract

    # ──────────────────────────────────────────────────────────────────────
    # Engineering completeness checks (new)
    # ──────────────────────────────────────────────────────────────────────

    def _validate_engineering_completeness(
        self,
        task_frame: TaskFrame,
        context: str,
    ) -> List[ValidationCheck]:
        """Check engineering robustness, not just answer presence."""
        checks: List[ValidationCheck] = []
        text = context.lower()
        task_type_str = (
            task_frame.task_type
            if isinstance(task_frame.task_type, str)
            else task_frame.task_type.value
        )
        is_code = "coding" in task_type_str or "hybrid" in task_type_str

        # ── Assumptions declared ─────────────────────────────────────────
        has_assumptions = any(
            m in text
            for m in ["assumption", "assumes", "caveat", "limitation", "note:", "out of scope"]
        )
        checks.append(ValidationCheck(
            check_name="assumptions_declared",
            passed=has_assumptions,
            severity="warning" if not has_assumptions else "info",
            message="Assumptions explicitly stated" if has_assumptions else "No explicit assumptions found",
        ))

        # ── Edge cases addressed ─────────────────────────────────────────
        has_edge = any(
            m in text
            for m in ["edge case", "boundary", "empty", "none", "null", "error", "exception", "invalid"]
        )
        checks.append(ValidationCheck(
            check_name="edge_cases_addressed",
            passed=has_edge,
            severity="warning" if (not has_edge and is_code) else "info",
            message="Edge cases addressed" if has_edge else "No visible edge case handling",
        ))

        # ── Tests or validation present ──────────────────────────────────
        has_tests = any(
            m in text
            for m in ["test", "assert", "pytest", "unittest", "expect(", "should return"]
        )
        checks.append(ValidationCheck(
            check_name="tests_or_validation_present",
            passed=has_tests,
            severity="warning" if (not has_tests and is_code) else "info",
            message="Tests or validation present" if has_tests else "No tests or validation found",
        ))

        # ── Failure modes discussed ──────────────────────────────────────
        has_failure = any(
            m in text
            for m in [
                "failure", "error handling", "exception", "timeout", "retry",
                "fallback", "graceful", "recover", "raise", "catch",
            ]
        )
        checks.append(ValidationCheck(
            check_name="failure_modes_discussed",
            passed=has_failure,
            severity="warning" if (not has_failure and is_code) else "info",
            message="Failure modes discussed" if has_failure else "No failure mode discussion found",
        ))

        # ── Extensibility considered (only for medium/high abstraction) ──
        abs_opp = getattr(task_frame, "abstraction_opportunity", "medium")
        if abs_opp in ("medium", "high"):
            has_extensibility = any(
                m in text
                for m in [
                    "interface", "abstract", "extensible", "configurable",
                    "parameter", "generic", "reusable", "modular", "plugin",
                ]
            )
            checks.append(ValidationCheck(
                check_name="extensibility_considered",
                passed=has_extensibility,
                severity="warning" if (not has_extensibility and abs_opp == "high") else "info",
                message="Extensibility considered" if has_extensibility else "No extensibility design found",
            ))

        # ── Intentional omissions stated ─────────────────────────────────
        has_omissions = any(
            m in text
            for m in ["intentionally", "left out", "not included", "out of scope", "omitted", "beyond scope"]
        )
        checks.append(ValidationCheck(
            check_name="intentional_omissions_stated",
            passed=has_omissions,
            severity="info",
            message="Intentional omissions stated" if has_omissions else "No explicit omissions stated",
        ))

        return checks

    # ──────────────────────────────────────────────────────────────────────
    # Existing surface-level checks (preserved)
    # ──────────────────────────────────────────────────────────────────────

    def _validate_code_output(
        self, agent_outputs: Dict[str, Any], context: str,
    ) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []
        has_code = "```" in context or "def " in context or "function " in context
        checks.append(ValidationCheck(
            check_name="code_present",
            passed=has_code,
            severity="error" if not has_code else "info",
            message="Code present in output" if has_code else "No code found in output",
        ))
        if has_code:
            has_structure = any(kw in context for kw in ["def ", "class ", "function ", "const ", "let "])
            checks.append(ValidationCheck(
                check_name="code_structure",
                passed=has_structure,
                severity="warning" if not has_structure else "info",
                message="Code has proper structure" if has_structure else "Code may lack proper structure",
            ))
        return checks

    def _validate_research_output(
        self, agent_outputs: Dict[str, Any], context: str,
    ) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []
        has_sources = any(m in context.lower() for m in ["source:", "reference:", "http", "according to"])
        checks.append(ValidationCheck(
            check_name="sources_present",
            passed=has_sources,
            severity="warning" if not has_sources else "info",
            message="Sources/citations present" if has_sources else "No clear sources or citations found",
        ))
        claim_markers = ["definitely", "certainly", "always", "never", "all", "none"]
        strong_claims = sum(1 for m in claim_markers if m in context.lower())
        checks.append(ValidationCheck(
            check_name="claim_strength",
            passed=strong_claims < 3,
            severity="warning" if strong_claims >= 3 else "info",
            message=f"Found {strong_claims} strong claim markers - verify support",
        ))
        return checks

    def _validate_reasoning_output(
        self, agent_outputs: Dict[str, Any], context: str, task_frame: TaskFrame,
    ) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []
        conclusion_markers = [
            "therefore", "thus", "so", "the answer", "result is",
            "conclusion", "in total", "equals", "is **", "= ",
        ]
        has_conclusion = any(m in context.lower() for m in conclusion_markers)
        checks.append(ValidationCheck(
            check_name="has_conclusion",
            passed=has_conclusion,
            severity="error" if not has_conclusion else "info",
            message="Contains a clear conclusion" if has_conclusion else "No clear conclusion found",
        ))
        reasoning_markers = [
            "because", "since", "step", "first", "then",
            "applying", "using", "principle", "formula",
            "calculation", "we know", "given that",
        ]
        has_reasoning = any(m in context.lower() for m in reasoning_markers)
        checks.append(ValidationCheck(
            check_name="shows_reasoning",
            passed=has_reasoning,
            severity="warning" if not has_reasoning else "info",
            message="Shows reasoning steps" if has_reasoning else "No clear reasoning steps found",
        ))
        request_lower = task_frame.normalized_request.lower()
        asks_for_number = any(w in request_lower for w in ["how many", "how much", "calculate", "what is the"])
        if asks_for_number:
            has_number = bool(re.search(r'\b\d+\b', context))
            checks.append(ValidationCheck(
                check_name="numeric_answer",
                passed=has_number,
                severity="error" if not has_number else "info",
                message="Contains numeric answer" if has_number else "No numeric answer found for quantitative question",
            ))
        return checks

    def _validate_completeness(
        self, task_frame: TaskFrame, context: str,
    ) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []
        request_words = set(task_frame.normalized_request.lower().split())
        context_words = set(context.lower().split())
        overlap = len(request_words & context_words) / len(request_words) if request_words else 0
        min_length = 50
        is_substantive = len(context.strip()) >= min_length
        passed = overlap > 0.3 or is_substantive
        checks.append(ValidationCheck(
            check_name="addresses_request",
            passed=passed,
            severity="error" if not passed else "info",
            message=f"Output addresses request (overlap: {overlap:.2f}, length: {len(context.strip())} chars)",
        ))
        if task_frame.hard_constraints:
            checks.append(ValidationCheck(
                check_name="constraints_acknowledged",
                passed=True,
                severity="info",
                message=f"Checked {len(task_frame.hard_constraints)} constraints",
            ))
        return checks

    def _validate_consistency(
        self, agent_outputs: Dict[str, Any],
    ) -> List[ValidationCheck]:
        checks: List[ValidationCheck] = []
        if len(agent_outputs) > 1:
            outputs_text = " ".join(str(o.get("output", "")) for o in agent_outputs.values())
            contradiction_markers = ["however", "but", "contradicts", "disagrees", "conflict"]
            has_contradictions = any(m in outputs_text.lower() for m in contradiction_markers)
            checks.append(ValidationCheck(
                check_name="agent_consistency",
                passed=not has_contradictions,
                severity="warning" if has_contradictions else "info",
                message="Potential contradictions between agents" if has_contradictions else "Agents appear consistent",
            ))
        return checks

    def _generate_summary(
        self, checks: List[ValidationCheck], state: ValidationState,
    ) -> str:
        passed = sum(1 for c in checks if c.passed)
        total = len(checks)
        return f"Validation {state.value}: {passed}/{total} checks passed"
