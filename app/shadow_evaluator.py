"""Shadow evaluator for probationary agent evaluation (Phase 5)."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from app.schemas.registry import AgentSpec
from app.agents.base_agent import BaseAgent


class ShadowEvaluation:
    """Record of a shadow evaluation."""
    
    def __init__(
        self,
        agent_id: str,
        task_id: str,
        shadow_output: Dict[str, Any],
        active_outputs: Dict[str, Any],
        timestamp: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.task_id = task_id
        self.shadow_output = shadow_output
        self.active_outputs = active_outputs
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        
        # Evaluation metrics
        self.quality_lift: Optional[float] = None
        self.cost: Optional[float] = None
        self.redundancy_score: Optional[float] = None
        self.unique_contribution: Optional[float] = None


class ShadowEvaluator:
    """Evaluate probationary agents in shadow mode."""
    
    def __init__(self):
        self.evaluations: List[ShadowEvaluation] = []
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
    
    def run_shadow_evaluation(
        self,
        agent: BaseAgent,
        task_context: Dict[str, Any],
        active_agent_outputs: Dict[str, Any]
    ) -> ShadowEvaluation:
        """Run agent in shadow mode and compare with active agents.
        
        Args:
            agent: Probationary agent to evaluate
            task_context: Task context
            active_agent_outputs: Outputs from active agents
        
        Returns:
            ShadowEvaluation with comparison results
        """
        # Execute shadow agent
        shadow_output = agent.execute(task_context)
        
        # Create evaluation record
        evaluation = ShadowEvaluation(
            agent_id=agent.agent_id,
            task_id=task_context.get("task_frame").task_id,
            shadow_output=shadow_output,
            active_outputs=active_agent_outputs
        )
        
        # Evaluate quality
        evaluation.quality_lift = self._evaluate_quality_lift(
            shadow_output,
            active_agent_outputs,
            task_context
        )
        
        # Evaluate cost
        evaluation.cost = self._evaluate_cost(shadow_output)
        
        # Evaluate redundancy
        evaluation.redundancy_score = self._evaluate_redundancy(
            shadow_output,
            active_agent_outputs
        )
        
        # Evaluate unique contribution
        evaluation.unique_contribution = self._evaluate_unique_contribution(
            shadow_output,
            active_agent_outputs
        )
        
        # Store evaluation
        self.evaluations.append(evaluation)
        self._update_agent_stats(agent.agent_id, evaluation)
        
        return evaluation
    
    def get_promotion_readiness(self, agent_id: str) -> Dict[str, Any]:
        """Calculate promotion readiness for probationary agent.
        
        Args:
            agent_id: Agent to evaluate
        
        Returns:
            Promotion readiness metrics
        """
        if agent_id not in self.agent_stats:
            return {
                "ready": False,
                "reason": "Insufficient evaluation data",
                "evaluations_count": 0
            }
        
        stats = self.agent_stats[agent_id]
        
        # Require minimum evaluations
        if stats["evaluation_count"] < 10:
            return {
                "ready": False,
                "reason": f"Need more evaluations ({stats['evaluation_count']}/10)",
                "evaluations_count": stats["evaluation_count"]
            }
        
        # Calculate promotion score
        promotion_score = (
            0.4 * stats["avg_quality_lift"] +
            0.3 * stats["success_rate"] +
            0.2 * (1.0 - stats["avg_cost"] / 2000.0) +  # Normalize cost
            0.1 * stats["avg_unique_contribution"]
        )
        
        # Check thresholds
        ready = (
            promotion_score >= 0.7 and
            stats["avg_quality_lift"] >= 0.5 and
            stats["success_rate"] >= 0.7 and
            stats["avg_redundancy"] < 0.7
        )
        
        return {
            "ready": ready,
            "promotion_score": promotion_score,
            "avg_quality_lift": stats["avg_quality_lift"],
            "success_rate": stats["success_rate"],
            "avg_cost": stats["avg_cost"],
            "avg_redundancy": stats["avg_redundancy"],
            "avg_unique_contribution": stats["avg_unique_contribution"],
            "evaluations_count": stats["evaluation_count"],
            "reason": self._generate_readiness_reason(ready, stats, promotion_score)
        }
    
    def _evaluate_quality_lift(
        self,
        shadow_output: Dict[str, Any],
        active_outputs: Dict[str, Any],
        task_context: Dict[str, Any]
    ) -> float:
        """Evaluate quality improvement from shadow agent."""
        # Simplified quality evaluation
        # In production, would use more sophisticated metrics
        
        shadow_confidence = shadow_output.get("confidence", 0.5)
        shadow_length = len(str(shadow_output.get("output", "")))
        
        # Compare with active agents
        if not active_outputs:
            return shadow_confidence
        
        avg_active_confidence = sum(
            output.get("confidence", 0.5)
            for output in active_outputs.values()
        ) / len(active_outputs)
        
        # Quality lift is relative improvement
        quality_lift = (shadow_confidence - avg_active_confidence) / max(avg_active_confidence, 0.1)
        
        # Normalize to 0-1
        return max(0.0, min(quality_lift + 0.5, 1.0))
    
    def _evaluate_cost(self, shadow_output: Dict[str, Any]) -> float:
        """Evaluate computational cost of shadow agent."""
        # Estimate based on output length and tool calls
        output_length = len(str(shadow_output.get("output", "")))
        tool_calls = len(shadow_output.get("tool_calls", []))
        
        # Rough cost estimate (tokens)
        estimated_tokens = output_length / 4 + tool_calls * 500
        
        return estimated_tokens
    
    def _evaluate_redundancy(
        self,
        shadow_output: Dict[str, Any],
        active_outputs: Dict[str, Any]
    ) -> float:
        """Evaluate how redundant shadow output is with active agents."""
        if not active_outputs:
            return 0.0
        
        shadow_text = str(shadow_output.get("output", "")).lower()
        
        # Calculate overlap with each active agent
        overlaps = []
        for active_output in active_outputs.values():
            active_text = str(active_output.get("output", "")).lower()
            
            # Simple word overlap
            shadow_words = set(shadow_text.split())
            active_words = set(active_text.split())
            
            if shadow_words and active_words:
                overlap = len(shadow_words & active_words) / len(shadow_words | active_words)
                overlaps.append(overlap)
        
        # Return maximum overlap
        return max(overlaps) if overlaps else 0.0
    
    def _evaluate_unique_contribution(
        self,
        shadow_output: Dict[str, Any],
        active_outputs: Dict[str, Any]
    ) -> float:
        """Evaluate unique contribution of shadow agent."""
        # Inverse of redundancy
        redundancy = self._evaluate_redundancy(shadow_output, active_outputs)
        return 1.0 - redundancy
    
    def _update_agent_stats(self, agent_id: str, evaluation: ShadowEvaluation) -> None:
        """Update running statistics for agent."""
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = {
                "evaluation_count": 0,
                "avg_quality_lift": 0.0,
                "avg_cost": 0.0,
                "avg_redundancy": 0.0,
                "avg_unique_contribution": 0.0,
                "success_rate": 0.0,
                "successes": 0
            }
        
        stats = self.agent_stats[agent_id]
        n = stats["evaluation_count"]
        
        # Update averages with exponential moving average
        alpha = 0.2  # Learning rate
        stats["avg_quality_lift"] = (
            (1 - alpha) * stats["avg_quality_lift"] +
            alpha * evaluation.quality_lift
        )
        stats["avg_cost"] = (
            (1 - alpha) * stats["avg_cost"] +
            alpha * evaluation.cost
        )
        stats["avg_redundancy"] = (
            (1 - alpha) * stats["avg_redundancy"] +
            alpha * evaluation.redundancy_score
        )
        stats["avg_unique_contribution"] = (
            (1 - alpha) * stats["avg_unique_contribution"] +
            alpha * evaluation.unique_contribution
        )
        
        # Update success rate
        if evaluation.quality_lift > 0.5:
            stats["successes"] += 1
        stats["evaluation_count"] += 1
        stats["success_rate"] = stats["successes"] / stats["evaluation_count"]
    
    def _generate_readiness_reason(
        self,
        ready: bool,
        stats: Dict[str, Any],
        promotion_score: float
    ) -> str:
        """Generate human-readable readiness reason."""
        if ready:
            return f"Ready for promotion: score {promotion_score:.2f}, quality lift {stats['avg_quality_lift']:.2f}, success rate {stats['success_rate']:.1%}"
        
        reasons = []
        if promotion_score < 0.7:
            reasons.append(f"promotion score {promotion_score:.2f} below 0.7")
        if stats["avg_quality_lift"] < 0.5:
            reasons.append(f"quality lift {stats['avg_quality_lift']:.2f} below 0.5")
        if stats["success_rate"] < 0.7:
            reasons.append(f"success rate {stats['success_rate']:.1%} below 70%")
        if stats["avg_redundancy"] >= 0.7:
            reasons.append(f"redundancy {stats['avg_redundancy']:.2f} too high")
        
        return f"Not ready: {', '.join(reasons)}"
    
    def get_evaluation_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get summary of evaluations for an agent."""
        agent_evals = [e for e in self.evaluations if e.agent_id == agent_id]
        
        if not agent_evals:
            return {"agent_id": agent_id, "evaluations": 0}
        
        return {
            "agent_id": agent_id,
            "evaluations": len(agent_evals),
            "stats": self.agent_stats.get(agent_id, {}),
            "recent_evaluations": [
                {
                    "task_id": e.task_id,
                    "quality_lift": e.quality_lift,
                    "cost": e.cost,
                    "redundancy": e.redundancy_score,
                    "timestamp": e.timestamp
                }
                for e in agent_evals[-5:]  # Last 5
            ]
        }
