"""Final synthesis stage for integrating agent outputs."""

from typing import Dict, Any, List, Optional
from app.models.llm_client import LLMClient
from app.schemas.task_frame import TaskFrame
from app.schemas.validation import ValidationReport


class SynthesisPackage:
    """Package of validated information for final synthesis."""
    
    def __init__(
        self,
        task_frame: TaskFrame,
        accepted_facts: List[str],
        agent_outputs: Dict[str, Any],
        arbitration_outcomes: List[Dict[str, Any]],
        validation_results: ValidationReport,
        uncertainty_notes: List[str],
        budget_notes: Optional[str] = None
    ):
        self.task_frame = task_frame
        self.accepted_facts = accepted_facts
        self.agent_outputs = agent_outputs
        self.arbitration_outcomes = arbitration_outcomes
        self.validation_results = validation_results
        self.uncertainty_notes = uncertainty_notes
        self.budget_notes = budget_notes


class Synthesizer:
    """Synthesize final answer from validated agent outputs."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def synthesize(
        self,
        synthesis_package: SynthesisPackage
    ) -> Dict[str, Any]:
        """Synthesize final answer from package.
        
        Args:
            synthesis_package: Validated information package
        
        Returns:
            Dictionary with final_answer, confidence, and metadata
        """
        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(synthesis_package)
        
        # Generate final answer
        final_answer = self.llm_client.generate(
            prompt,
            max_tokens=1500,
            temperature=0.5
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(synthesis_package)
        
        # Check for unsupported claims
        validation_passed = self._validate_synthesis(
            final_answer,
            synthesis_package
        )
        
        return {
            "final_answer": final_answer,
            "confidence": confidence,
            "validation_passed": validation_passed,
            "synthesis_method": "evidence_constrained",
            "sources_used": len(synthesis_package.agent_outputs),
            "conflicts_resolved": len(synthesis_package.arbitration_outcomes)
        }
    
    def _build_synthesis_prompt(self, package: SynthesisPackage) -> str:
        """Build prompt for final synthesis."""
        prompt = f"""You are synthesizing the final answer to a user request based on validated agent analyses.

USER REQUEST:
{package.task_frame.normalized_request}

TASK TYPE: {package.task_frame.task_type}

ACCEPTED FACTS:
"""
        if package.accepted_facts:
            for fact in package.accepted_facts:
                prompt += f"- {fact}\n"
        else:
            prompt += "None explicitly validated.\n"
        
        prompt += "\nAGENT ANALYSES:\n"
        for agent_id, output in package.agent_outputs.items():
            prompt += f"\n{agent_id}:\n{output.get('output', '')}\n"
        
        if package.arbitration_outcomes:
            prompt += "\nCONFLICT RESOLUTIONS:\n"
            for outcome in package.arbitration_outcomes:
                prompt += f"- {outcome.get('resolution', '')}\n"
        
        if package.uncertainty_notes:
            prompt += "\nUNCERTAINTIES:\n"
            for note in package.uncertainty_notes:
                prompt += f"- {note}\n"
        
        if package.budget_notes:
            prompt += f"\nBUDGET NOTES:\n{package.budget_notes}\n"
        
        prompt += """
SYNTHESIS REQUIREMENTS:
1. Directly address the user's request
2. Use ONLY information from accepted facts and agent analyses
3. Integrate insights from all agents coherently
4. Acknowledge any remaining uncertainties
5. Do NOT invent unsupported facts
6. If information is incomplete, state what is missing

Provide a clear, concise final answer:"""
        
        return prompt
    
    def _calculate_confidence(self, package: SynthesisPackage) -> float:
        """Calculate confidence in synthesis."""
        # Factors affecting confidence
        factors = []
        
        # Validation state
        if package.validation_results.overall_passed:
            factors.append(0.8)
        else:
            factors.append(0.4)
        
        # Number of agents (more agreement = higher confidence)
        num_agents = len(package.agent_outputs)
        if num_agents >= 3:
            factors.append(0.9)
        elif num_agents == 2:
            factors.append(0.7)
        else:
            factors.append(0.5)
        
        # Conflicts (fewer = higher confidence)
        num_conflicts = len(package.arbitration_outcomes)
        if num_conflicts == 0:
            factors.append(0.9)
        elif num_conflicts <= 2:
            factors.append(0.7)
        else:
            factors.append(0.5)
        
        # Uncertainties (fewer = higher confidence)
        num_uncertainties = len(package.uncertainty_notes)
        if num_uncertainties == 0:
            factors.append(0.9)
        elif num_uncertainties <= 2:
            factors.append(0.7)
        else:
            factors.append(0.5)
        
        # Task uncertainty
        task_uncertainty = package.task_frame.initial_uncertainty
        factors.append(1.0 - task_uncertainty)
        
        # Average all factors
        confidence = sum(factors) / len(factors)
        return max(0.0, min(confidence, 1.0))
    
    def _validate_synthesis(
        self,
        final_answer: str,
        package: SynthesisPackage
    ) -> bool:
        """Validate that synthesis stays within accepted evidence."""
        # Check for common unsupported claim markers
        unsupported_markers = [
            "definitely", "certainly", "always", "never",
            "all", "none", "every", "no one"
        ]
        
        final_answer_lower = final_answer.lower()
        
        # Count strong claims
        strong_claims = sum(
            1 for marker in unsupported_markers
            if marker in final_answer_lower
        )
        
        # If many strong claims, check if they're supported
        if strong_claims > 3:
            # Would need more sophisticated validation
            return False
        
        # Check if answer addresses the request
        request_words = set(package.task_frame.normalized_request.lower().split())
        answer_words = set(final_answer_lower.split())
        overlap = len(request_words & answer_words) / len(request_words) if request_words else 0
        
        if overlap < 0.2:
            return False
        
        return True
    
    def create_synthesis_package(
        self,
        task_frame: TaskFrame,
        agent_outputs: Dict[str, Any],
        shared_context: str,
        arbitration_outcomes: List[Dict[str, Any]],
        validation_report: ValidationReport,
        budget_exhausted: bool = False
    ) -> SynthesisPackage:
        """Create synthesis package from execution results.
        
        Args:
            task_frame: Task specification
            agent_outputs: Agent outputs
            shared_context: Accumulated context
            arbitration_outcomes: Conflict resolutions
            validation_report: Validation results
            budget_exhausted: Whether budget was exhausted
        
        Returns:
            SynthesisPackage ready for final synthesis
        """
        # Extract accepted facts from context
        accepted_facts = self._extract_accepted_facts(shared_context)
        
        # Extract uncertainty notes
        uncertainty_notes = self._extract_uncertainties(agent_outputs, shared_context)
        
        # Budget notes
        budget_notes = None
        if budget_exhausted:
            budget_notes = "Budget was exhausted. Some analysis may be incomplete."
        
        return SynthesisPackage(
            task_frame=task_frame,
            accepted_facts=accepted_facts,
            agent_outputs=agent_outputs,
            arbitration_outcomes=arbitration_outcomes,
            validation_results=validation_report,
            uncertainty_notes=uncertainty_notes,
            budget_notes=budget_notes
        )
    
    def _extract_accepted_facts(self, shared_context: str) -> List[str]:
        """Extract accepted facts from shared context."""
        facts = []
        
        # Look for explicit fact statements
        lines = shared_context.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                facts.append(line[2:])
            elif line and not line.startswith('#') and len(line) > 20:
                # Potential fact statement
                if any(marker in line.lower() for marker in ['is', 'are', 'can', 'will', 'should']):
                    facts.append(line)
        
        return facts[:10]  # Limit to top 10 facts
    
    def _extract_uncertainties(
        self,
        agent_outputs: Dict[str, Any],
        shared_context: str
    ) -> List[str]:
        """Extract uncertainty notes from outputs."""
        uncertainties = []
        
        uncertainty_markers = [
            "uncertain", "unclear", "not sure", "might", "maybe",
            "possibly", "perhaps", "could be", "unknown"
        ]
        
        # Check agent outputs
        for agent_id, output in agent_outputs.items():
            output_text = str(output.get("output", "")).lower()
            for marker in uncertainty_markers:
                if marker in output_text:
                    # Extract sentence containing uncertainty
                    sentences = output_text.split('.')
                    for sentence in sentences:
                        if marker in sentence:
                            uncertainties.append(f"{agent_id}: {sentence.strip()}")
                            break
                    break
        
        return uncertainties[:5]  # Limit to top 5 uncertainties
