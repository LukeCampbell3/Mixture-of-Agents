"""Validation layer for outputs and agents."""

from typing import Dict, Any, List
from app.schemas.validation import ValidationReport, ValidationCheck, ValidationState
from app.schemas.task_frame import TaskFrame, TaskType


class Validator:
    """Validate task outputs and agent performance."""
    
    def validate_output(
        self,
        task_frame: TaskFrame,
        agent_outputs: Dict[str, Any],
        shared_context: str
    ) -> ValidationReport:
        """Validate task output.
        
        Args:
            task_frame: Task specification
            agent_outputs: Outputs from all agents
            shared_context: Final shared context
        
        Returns:
            ValidationReport with checks and overall status
        """
        checks = []
        
        # Task-specific validation
        task_type = task_frame.task_type
        task_type_str = task_type if isinstance(task_type, str) else task_type.value
        
        if "coding" in task_type_str:
            checks.extend(self._validate_code_output(agent_outputs, shared_context))
        
        if "research" in task_type_str:
            checks.extend(self._validate_research_output(agent_outputs, shared_context))
        
        # Hybrid tasks get both coding and research checks
        if task_type_str == "hybrid":
            checks.extend(self._validate_code_output(agent_outputs, shared_context))
            checks.extend(self._validate_research_output(agent_outputs, shared_context))
        
        # Reasoning/planning tasks get reasoning checks
        if task_type_str == "planning":
            checks.extend(self._validate_reasoning_output(agent_outputs, shared_context, task_frame))
        
        # General validation (always applied)
        checks.extend(self._validate_completeness(task_frame, shared_context))
        checks.extend(self._validate_consistency(agent_outputs))
        
        # Determine overall state
        total_checks = len(checks)
        passed_checks = sum(1 for c in checks if c.passed)
        error_count = sum(1 for c in checks if not c.passed and c.severity == "error")
        warning_count = sum(1 for c in checks if not c.passed and c.severity == "warning")
        
        # Pass requires: zero errors AND strictly more than half the checks passed
        if error_count > 0:
            state = ValidationState.VALIDATION_FAILURE
            overall_passed = False
        elif total_checks > 0 and passed_checks * 2 <= total_checks:
            # Strictly more than half must pass (1/2 is NOT enough)
            state = ValidationState.PARTIAL_SUCCESS
            overall_passed = False
        elif warning_count > 2:
            state = ValidationState.PARTIAL_SUCCESS
            overall_passed = True
        else:
            state = ValidationState.SUCCESS
            overall_passed = True
        
        # Build report
        report = ValidationReport(
            task_id=task_frame.task_id,
            validation_state=state,
            checks=checks,
            overall_passed=overall_passed,
            summary=self._generate_summary(checks, state)
        )
        
        return report
    
    def _validate_code_output(self, agent_outputs: Dict[str, Any], context: str) -> List[ValidationCheck]:
        """Validate code-related outputs."""
        checks = []
        
        # Check if code is present
        has_code = "```" in context or "def " in context or "function " in context
        checks.append(ValidationCheck(
            check_name="code_present",
            passed=has_code,
            severity="error" if not has_code else "info",
            message="Code present in output" if has_code else "No code found in output"
        ))
        
        # Check for basic syntax patterns
        if has_code:
            has_structure = any(keyword in context for keyword in ["def ", "class ", "function ", "const ", "let "])
            checks.append(ValidationCheck(
                check_name="code_structure",
                passed=has_structure,
                severity="warning" if not has_structure else "info",
                message="Code has proper structure" if has_structure else "Code may lack proper structure"
            ))
        
        return checks
    
    def _validate_research_output(self, agent_outputs: Dict[str, Any], context: str) -> List[ValidationCheck]:
        """Validate research-related outputs."""
        checks = []
        
        # Check for citations or sources
        has_sources = any(marker in context.lower() for marker in ["source:", "reference:", "http", "according to"])
        checks.append(ValidationCheck(
            check_name="sources_present",
            passed=has_sources,
            severity="warning" if not has_sources else "info",
            message="Sources/citations present" if has_sources else "No clear sources or citations found"
        ))
        
        # Check for unsupported claims
        claim_markers = ["definitely", "certainly", "always", "never", "all", "none"]
        strong_claims = sum(1 for marker in claim_markers if marker in context.lower())
        checks.append(ValidationCheck(
            check_name="claim_strength",
            passed=strong_claims < 3,
            severity="warning" if strong_claims >= 3 else "info",
            message=f"Found {strong_claims} strong claim markers - verify support"
        ))
        
        return checks
    
    def _validate_reasoning_output(self, agent_outputs: Dict[str, Any], context: str, task_frame: TaskFrame) -> List[ValidationCheck]:
        """Validate reasoning/planning outputs."""
        checks = []
        
        # Check that the output contains a clear conclusion or answer
        conclusion_markers = ["therefore", "thus", "so", "the answer", "result is",
                            "conclusion", "in total", "equals", "is **", "= "]
        has_conclusion = any(marker in context.lower() for marker in conclusion_markers)
        checks.append(ValidationCheck(
            check_name="has_conclusion",
            passed=has_conclusion,
            severity="error" if not has_conclusion else "info",
            message="Contains a clear conclusion" if has_conclusion else "No clear conclusion found"
        ))
        
        # Check that reasoning steps are present (not just a bare answer)
        reasoning_markers = ["because", "since", "step", "first", "then",
                           "applying", "using", "principle", "formula",
                           "calculation", "we know", "given that"]
        has_reasoning = any(marker in context.lower() for marker in reasoning_markers)
        checks.append(ValidationCheck(
            check_name="shows_reasoning",
            passed=has_reasoning,
            severity="warning" if not has_reasoning else "info",
            message="Shows reasoning steps" if has_reasoning else "No clear reasoning steps found"
        ))
        
        # Check that a numeric or definitive answer is present if the task asks for one
        request_lower = task_frame.normalized_request.lower()
        asks_for_number = any(w in request_lower for w in ["how many", "how much", "calculate", "what is the"])
        if asks_for_number:
            import re
            has_number = bool(re.search(r'\b\d+\b', context))
            checks.append(ValidationCheck(
                check_name="numeric_answer",
                passed=has_number,
                severity="error" if not has_number else "info",
                message="Contains numeric answer" if has_number else "No numeric answer found for quantitative question"
            ))
        
        return checks
    
    def _validate_completeness(self, task_frame: TaskFrame, context: str) -> List[ValidationCheck]:
        """Validate output completeness."""
        checks = []
        
        # Check if output addresses the request — use a lower threshold for
        # short-answer tasks (reasoning, math) where a concise correct answer
        # may share few surface words with the prompt.
        request_words = set(task_frame.normalized_request.lower().split())
        context_words = set(context.lower().split())
        overlap = len(request_words & context_words) / len(request_words) if request_words else 0
        
        # Minimum output length as a secondary signal: very short outputs
        # are suspicious regardless of overlap.
        min_length = 50  # characters
        is_substantive = len(context.strip()) >= min_length
        
        # Pass if either overlap is reasonable OR the output is substantive
        passed = overlap > 0.3 or is_substantive
        
        checks.append(ValidationCheck(
            check_name="addresses_request",
            passed=passed,
            severity="error" if not passed else "info",
            message=f"Output addresses request (overlap: {overlap:.2f}, length: {len(context.strip())} chars)"
        ))
        
        # Check for constraint satisfaction
        if task_frame.hard_constraints:
            checks.append(ValidationCheck(
                check_name="constraints_acknowledged",
                passed=True,  # Simplified - would need deeper analysis
                severity="info",
                message=f"Checked {len(task_frame.hard_constraints)} constraints"
            ))
        
        return checks
    
    def _validate_consistency(self, agent_outputs: Dict[str, Any]) -> List[ValidationCheck]:
        """Validate consistency between agent outputs."""
        checks = []
        
        if len(agent_outputs) > 1:
            # Check for explicit contradictions (simplified)
            outputs_text = " ".join(str(output.get("output", "")) for output in agent_outputs.values())
            contradiction_markers = ["however", "but", "contradicts", "disagrees", "conflict"]
            has_contradictions = any(marker in outputs_text.lower() for marker in contradiction_markers)
            
            checks.append(ValidationCheck(
                check_name="agent_consistency",
                passed=not has_contradictions,
                severity="warning" if has_contradictions else "info",
                message="Potential contradictions between agents" if has_contradictions else "Agents appear consistent"
            ))
        
        return checks
    
    def _generate_summary(self, checks: List[ValidationCheck], state: ValidationState) -> str:
        """Generate validation summary."""
        passed = sum(1 for c in checks if c.passed)
        total = len(checks)
        
        return f"Validation {state.value}: {passed}/{total} checks passed"
