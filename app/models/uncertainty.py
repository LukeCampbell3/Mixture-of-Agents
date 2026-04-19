"""Uncertainty estimation for model outputs."""

from typing import List, Dict, Any
import re


class UncertaintyEstimator:
    """Estimate uncertainty in model outputs."""
    
    # Uncertainty markers in text
    UNCERTAINTY_MARKERS = [
        "might", "maybe", "possibly", "perhaps", "could be",
        "not sure", "unclear", "uncertain", "probably", "likely",
        "I think", "I believe", "seems like", "appears to"
    ]
    
    def estimate_from_text(self, text: str) -> float:
        """Estimate uncertainty from text content (0=certain, 1=uncertain)."""
        if not text:
            return 1.0
        
        text_lower = text.lower()
        
        # Count uncertainty markers
        marker_count = sum(1 for marker in self.UNCERTAINTY_MARKERS if marker in text_lower)
        
        # Count question marks
        question_count = text.count("?")
        
        # Check for explicit uncertainty statements
        explicit_uncertainty = any(phrase in text_lower for phrase in [
            "don't know", "can't determine", "need more information",
            "insufficient information", "ambiguous"
        ])
        
        # Combine signals
        base_uncertainty = min(marker_count * 0.1 + question_count * 0.05, 0.8)
        if explicit_uncertainty:
            base_uncertainty = max(base_uncertainty, 0.7)
        
        return min(base_uncertainty, 1.0)
    
    def estimate_from_logprobs(self, logprobs: List[float]) -> float:
        """Estimate uncertainty from token log probabilities."""
        if not logprobs:
            return 0.5
        
        # Use entropy as uncertainty measure
        import numpy as np
        probs = np.exp(logprobs)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Normalize to 0-1 range (assuming max entropy ~3 for typical vocab)
        return min(entropy / 3.0, 1.0)
    
    def estimate_task_uncertainty(self, task_description: str) -> float:
        """Estimate initial task uncertainty."""
        # Check for ambiguous language
        ambiguous_terms = ["any", "some", "various", "multiple", "different", "several"]
        ambiguity_score = sum(1 for term in ambiguous_terms if term in task_description.lower())
        
        # Check for specificity
        has_numbers = bool(re.search(r'\d+', task_description))
        has_specific_names = bool(re.search(r'[A-Z][a-z]+', task_description))
        
        specificity = 0.0
        if has_numbers:
            specificity += 0.3
        if has_specific_names:
            specificity += 0.3
        
        # Combine
        uncertainty = 0.5 + (ambiguity_score * 0.1) - specificity
        return max(0.0, min(uncertainty, 1.0))
