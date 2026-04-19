"""Router for agent selection and task framing with calibrated confidence."""

from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
from app.schemas.task_frame import TaskFrame, TaskType
from app.schemas.analysis import RoutingDecision, AgentScore
from app.schemas.registry import AgentRegistry, AgentSpec
from app.models.embeddings import EmbeddingGenerator
from app.models.uncertainty import UncertaintyEstimator
from app.models.llm_client import LLMClient
from app.calibration import ThreeLevelCalibrator
import uuid
from pathlib import Path

if TYPE_CHECKING:
    from app.evaluation.data_splits import CounterfactualStore


class Router:
    """Control component for task classification and agent selection with calibration."""
    
    def __init__(
        self,
        registry: AgentRegistry,
        llm_client: LLMClient,
        embedding_generator: EmbeddingGenerator,
        uncertainty_estimator: UncertaintyEstimator,
        calibrator: Optional[ThreeLevelCalibrator] = None,
        counterfactual_store: Optional["CounterfactualStore"] = None
    ):
        self.registry = registry
        self.llm_client = llm_client
        self.embedding_generator = embedding_generator
        self.uncertainty_estimator = uncertainty_estimator
        self.version = "2.0.0"
        
        # Calibration system
        self.calibrator = calibrator
        if self.calibrator is None:
            # Try to load existing calibrator
            calibrator_path = Path("data/calibration")
            if calibrator_path.exists():
                try:
                    self.calibrator = ThreeLevelCalibrator(
                        task_families=["coding", "research", "reasoning", "mixed"]
                    )
                    self.calibrator.load(str(calibrator_path))
                except Exception:
                    pass
        
        # Counterfactual store for oracle-based training
        self.counterfactual_store = counterfactual_store
        if self.counterfactual_store is None:
            # Lazy import to avoid circular dependency
            from app.evaluation.data_splits import CounterfactualStore
            self.counterfactual_store = CounterfactualStore()
            try:
                self.counterfactual_store.load()
            except Exception:
                pass
        
        # Historical performance for counterfactual lift
        self.agent_performance_history: Dict[str, List[float]] = {}
    
    def frame_task(self, user_request: str) -> TaskFrame:
        """Convert user request into structured task frame."""
        # Classify task type
        task_type = self._classify_task_type(user_request)
        
        # Estimate uncertainty
        initial_uncertainty = self.uncertainty_estimator.estimate_task_uncertainty(user_request)
        
        # Estimate novelty (simplified - could use retrieval)
        novelty_score = 0.5  # Default
        
        # Determine freshness requirement
        freshness_requirement = self._estimate_freshness_need(user_request)
        
        # Extract constraints (simplified)
        hard_constraints = self._extract_constraints(user_request)
        
        # Identify likely tools
        likely_tools = self._identify_tools(user_request, task_type)
        
        task_frame = TaskFrame(
            task_id=str(uuid.uuid4()),
            normalized_request=user_request.strip(),
            task_type=task_type,
            hard_constraints=hard_constraints,
            likely_tools=likely_tools,
            difficulty_estimate=0.5,  # Could be enhanced
            initial_uncertainty=initial_uncertainty,
            novelty_score=novelty_score,
            freshness_requirement=freshness_requirement
        )
        
        return task_frame
    
    def route(self, task_frame: TaskFrame, max_agents: int = 3) -> RoutingDecision:
        """Select agents for task execution with calibrated confidence.
        
        Args:
            task_frame: Structured task representation
            max_agents: Maximum number of agents to activate
        
        Returns:
            RoutingDecision with selected and suppressed agents
        """
        # Get routable agents
        candidates = self.registry.get_routable_agents()
        
        # Check for oracle subset from counterfactual store
        oracle_subset = None
        if self.counterfactual_store:
            oracle_subset = self.counterfactual_store.get_oracle_subset(task_frame.task_id)
        
        # Score each candidate
        agent_scores = []
        for agent in candidates:
            score = self._score_agent(agent, task_frame, oracle_subset)
            agent_scores.append(score)
        
        # Sort by activation score
        agent_scores.sort(key=lambda x: x.activation_score, reverse=True)
        
        # Select top agents within budget with top-k cap
        selected = []
        suppressed = []
        
        # Top-k routing: consider only top candidates
        top_k = min(5, len(agent_scores))
        top_candidates = agent_scores[:top_k]
        
        # Dynamic threshold based on task characteristics
        base_threshold = 0.3
        
        # Lower threshold for complex/uncertain tasks to encourage collaboration
        if task_frame.initial_uncertainty > 0.5:
            base_threshold = 0.2  # Very uncertain - need multiple perspectives
        elif task_frame.difficulty_estimate > 0.6:
            base_threshold = 0.25  # Hard task - benefit from collaboration
        
        # For hybrid/unknown tasks, be even more aggressive
        task_type_str = task_frame.task_type if isinstance(task_frame.task_type, str) else task_frame.task_type.value
        if "hybrid" in task_type_str or task_type_str == "unknown":
            base_threshold = 0.2  # Encourage multi-agent for complex tasks
        
        # Ensure at least 2 agents for tasks that could benefit from collaboration
        min_agents_for_collaboration = 1
        if task_frame.initial_uncertainty > 0.4 or task_frame.difficulty_estimate > 0.5:
            min_agents_for_collaboration = 2
        
        for score in top_candidates:
            threshold = base_threshold
            
            # If we haven't met minimum collaboration threshold, be more lenient
            if len(selected) < min_agents_for_collaboration:
                threshold = max(0.15, base_threshold - 0.1)
            
            if len(selected) < max_agents and score.activation_score > threshold:
                selected.append(score.agent_id)
            else:
                suppressed.append({
                    "agent_id": score.agent_id,
                    "reason": score.reason if score.activation_score <= threshold else "budget_exclusion"
                })
        
        # Suppress remaining candidates
        for score in agent_scores[top_k:]:
            suppressed.append({
                "agent_id": score.agent_id,
                "reason": "below_top_k_threshold"
            })
        
        # Build routing decision
        decision = RoutingDecision(
            task_id=task_frame.task_id,
            candidate_agents=agent_scores,
            selected_agents=selected,
            suppressed_agents=suppressed,
            budget_plan={
                "expected_tokens": len(selected) * 1000,
                "expected_latency": len(selected) * 2.0
            },
            routing_reasons=[
                f"Selected {len(selected)} agents based on calibrated confidence and counterfactual lift",
                f"Task type: {task_frame.task_type}",
                f"Uncertainty: {task_frame.initial_uncertainty:.2f}",
                f"Oracle guidance: {'Yes' if oracle_subset else 'No'}"
            ],
            uncertainty_summary=f"Initial uncertainty: {task_frame.initial_uncertainty:.2f}",
            spawn_recommendation=None,
            no_spawn_reason="Existing agents sufficient for task",
            arbitration_needed=len(selected) > 1
        )
        
        return decision
    
    def _classify_task_type(self, request: str) -> TaskType:
        """Classify task into category."""
        request_lower = request.lower()
        
        # Check for coding signals first (strongest signal)
        coding_keywords = ["code", "function", "class", "implement", "debug", "fix", "write a", "create a", "build a"]
        has_coding = any(word in request_lower for word in coding_keywords)
        
        # Check for research signals
        research_keywords = ["compare", "difference", "explain", "what is", "how does",
                           "trade-off", "tradeoff", "pros and cons", "characteristics",
                           "performance of", "advantages", "disadvantages"]
        has_research = any(word in request_lower for word in research_keywords)
        
        # Check for reasoning signals
        reasoning_keywords = ["how many", "calculate", "if all", "can we conclude",
                            "logic", "prove", "deduce", "probability", "how much",
                            "employees", "balls", "puzzle"]
        has_reasoning = any(word in request_lower for word in reasoning_keywords)
        
        # Check for freshness signals
        freshness_keywords = ["latest", "current", "new", "recent", "today", "now", "2024", "2025", "2026"]
        has_freshness = any(word in request_lower for word in freshness_keywords)
        
        # Classify based on signal combinations
        if has_coding and has_research:
            return TaskType.HYBRID
        
        if has_coding:
            if has_freshness:
                return TaskType.CODING_CURRENT
            return TaskType.CODING_STABLE
        
        if has_reasoning and not has_research:
            return TaskType.PLANNING  # Closest to reasoning in our enum
        
        if has_research:
            if any(word in request_lower for word in ["maybe", "possibly", "unclear", "ambiguous"]):
                return TaskType.RESEARCH_HIGH_AMBIGUITY
            return TaskType.RESEARCH_LOW_AMBIGUITY
        
        # Fallback: check for question patterns (likely research)
        if request_lower.startswith(("what", "how", "why", "when", "where", "which")):
            return TaskType.RESEARCH_LOW_AMBIGUITY
        
        return TaskType.UNKNOWN
    
    def _estimate_freshness_need(self, request: str) -> float:
        """Estimate need for current information."""
        freshness_keywords = ["latest", "current", "new", "recent", "today", "now", "2024", "2025", "2026"]
        request_lower = request.lower()
        
        count = sum(1 for keyword in freshness_keywords if keyword in request_lower)
        return min(count * 0.3, 1.0)
    
    def _extract_constraints(self, request: str) -> List[str]:
        """Extract hard constraints from request."""
        constraints = []
        
        # Look for explicit constraints
        if "must" in request.lower():
            # Simple extraction - could be enhanced
            constraints.append("Contains explicit 'must' requirements")
        
        return constraints
    
    def _identify_tools(self, request: str, task_type: TaskType) -> List[str]:
        """Identify likely tools needed."""
        tools = []
        
        if task_type in [TaskType.CODING_STABLE, TaskType.CODING_CURRENT]:
            tools.extend(["repo_tool", "test_runner"])
        
        if task_type in [TaskType.RESEARCH_LOW_AMBIGUITY, TaskType.RESEARCH_HIGH_AMBIGUITY, TaskType.CODING_CURRENT]:
            tools.append("web_tool")
        
        if task_type == TaskType.HYBRID:
            tools.extend(["repo_tool", "web_tool", "test_runner"])
        
        return tools
    
    def _score_agent(
        self,
        agent: AgentSpec,
        task_frame: TaskFrame,
        oracle_subset: Optional[List[str]] = None
    ) -> AgentScore:
        """Score agent for task activation with calibrated confidence and counterfactual lift.
        
        Args:
            agent: Agent specification
            task_frame: Task frame
            oracle_subset: Oracle agent subset from counterfactual store (if available)
        """
        # Capability match based on domain and task type
        capability_match = self._compute_capability_match(agent, task_frame)
        
        # Counterfactual lift: use historical performance if available
        counterfactual_lift = self._compute_counterfactual_lift(
            agent.agent_id,
            task_frame,
            oracle_subset
        )
        
        # Expected quality gain (weighted by counterfactual lift)
        expected_quality_gain = capability_match * 0.5 + counterfactual_lift * 0.5
        
        # Calibrate confidence if calibrator available
        raw_confidence = expected_quality_gain
        calibrated_confidence = raw_confidence
        if self.calibrator:
            task_family = self._get_task_family(task_frame.task_type)
            calibrated_confidence = self.calibrator.calibrate_router(
                raw_confidence,
                task_family=task_family
            )
        
        # Token cost estimate
        token_cost = 1000.0  # Base estimate
        
        # Latency cost
        latency_cost = 2.0  # seconds
        
        # Overlap penalty (simplified - would check against already selected)
        overlap_penalty = 0.0
        
        # Compute activation score with calibrated confidence
        # Increase weight on capability match and reduce cost penalties
        activation_score = (
            0.5 * capability_match  # Increased from 0.3
            + 0.3 * calibrated_confidence  # Decreased from 0.4
            + 0.2 * counterfactual_lift
            - 0.0 * (token_cost / 5000.0)  # Remove token cost penalty
            - 0.0 * (latency_cost / 10.0)  # Remove latency penalty
            - 0.0 * overlap_penalty  # Remove overlap penalty
        )
        
        # Generate reason
        reason = self._generate_score_reason(
            agent,
            capability_match,
            calibrated_confidence,
            counterfactual_lift,
            activation_score
        )
        
        return AgentScore(
            agent_id=agent.agent_id,
            activation_score=max(0.0, activation_score),
            capability_match=capability_match,
            expected_quality_gain=expected_quality_gain,
            token_cost=token_cost,
            latency_cost=latency_cost,
            overlap_penalty=overlap_penalty,
            reason=reason
        )
    
    def _compute_counterfactual_lift(
        self,
        agent_id: str,
        task_frame: TaskFrame,
        oracle_subset: Optional[List[str]] = None
    ) -> float:
        """Compute counterfactual lift for agent.
        
        This estimates how much the agent actually improves outcomes,
        not just semantic similarity.
        
        Args:
            agent_id: Agent identifier
            task_frame: Task frame
            oracle_subset: Oracle subset from counterfactual store
        
        Returns:
            Counterfactual lift score [0, 1]
        """
        # If oracle subset available, boost agents in oracle
        if oracle_subset and agent_id in oracle_subset:
            return 0.9
        
        # Use historical performance if available
        if agent_id in self.agent_performance_history:
            history = self.agent_performance_history[agent_id]
            if history:
                return sum(history) / len(history)
        
        # Default: moderate lift
        return 0.5
    
    def record_agent_performance(
        self,
        agent_id: str,
        quality_score: float
    ):
        """Record agent performance for counterfactual lift estimation.
        
        Args:
            agent_id: Agent identifier
            quality_score: Quality score achieved [0, 1]
        """
        if agent_id not in self.agent_performance_history:
            self.agent_performance_history[agent_id] = []
        
        self.agent_performance_history[agent_id].append(quality_score)
        
        # Keep only recent history (last 100 tasks)
        if len(self.agent_performance_history[agent_id]) > 100:
            self.agent_performance_history[agent_id] = \
                self.agent_performance_history[agent_id][-100:]
    
    def _get_task_family(self, task_type: TaskType) -> str:
        """Map task type to task family for calibration."""
        if task_type in [TaskType.CODING_STABLE, TaskType.CODING_CURRENT]:
            return "coding"
        elif task_type in [TaskType.RESEARCH_LOW_AMBIGUITY, TaskType.RESEARCH_HIGH_AMBIGUITY]:
            return "research"
        elif task_type == TaskType.PLANNING:
            return "reasoning"
        elif task_type == TaskType.HYBRID:
            return "mixed"
        else:
            return "mixed"
    
    def _compute_capability_match(self, agent: AgentSpec, task_frame: TaskFrame) -> float:
        """Compute how well agent capabilities match task needs."""
        # Domain-based matching - MORE AGGRESSIVE activation
        domain_match = 0.5  # Default: moderate baseline
        
        task_type = task_frame.task_type
        if isinstance(task_type, str):
            task_type_str = task_type
        else:
            task_type_str = task_type.value
        
        # Primary domain matches - very high scores
        if agent.domain == "coding" and "coding" in task_type_str:
            domain_match = 0.95
        elif agent.domain == "research" and "research" in task_type_str:
            domain_match = 0.95
        elif agent.domain == "verification":
            # Critic is ALWAYS useful for quality assurance
            if "planning" in task_type_str:
                domain_match = 0.9  # Reasoning tasks benefit from verification
            elif task_frame.initial_uncertainty > 0.3 or task_frame.difficulty_estimate > 0.5:
                domain_match = 0.85  # High value for uncertain/complex tasks
            else:
                domain_match = 0.6  # Still valuable for simple tasks
        
        # Specialist domain matches (spawned agents)
        # Check if the task text matches the agent's target domain
        task_text_lower = task_frame.normalized_request.lower()
        if agent.domain == "api_migration":
            if any(kw in task_text_lower for kw in ["api", "migrate", "migration", "endpoint", "rest", "graphql"]):
                domain_match = 0.95
        elif agent.domain == "security_audit":
            if any(kw in task_text_lower for kw in ["security", "audit", "vulnerability", "injection"]):
                domain_match = 0.95
        
        # Cross-domain synergies - ENABLE rather than penalize
        if agent.domain == "coding" and "research" in task_type_str:
            domain_match = 0.4  # Coding perspective on research can help
        if agent.domain == "coding" and task_type_str in ("planning", "unknown"):
            domain_match = 0.4  # Implementation perspective on planning
        if agent.domain == "research" and "coding" in task_type_str:
            # Research can provide context and best practices
            domain_match = 0.5  # Increased from 0.15
        
        # Hybrid tasks benefit from ALL agents
        if "hybrid" in task_type_str or task_type_str == "unknown":
            domain_match = max(domain_match, 0.7)  # Boost all agents for hybrid
        
        # Tool match
        tool_match = 0.0
        if agent.tools and task_frame.likely_tools:
            matching_tools = set(agent.tools) & set(task_frame.likely_tools)
            tool_match = len(matching_tools) / len(task_frame.likely_tools) if task_frame.likely_tools else 0.0
        
        return 0.7 * domain_match + 0.3 * tool_match
    
    def _generate_score_reason(
        self,
        agent: AgentSpec,
        capability_match: float,
        calibrated_confidence: float,
        counterfactual_lift: float,
        score: float
    ) -> str:
        """Generate human-readable reason for score."""
        if score > 0.6:
            return (f"Strong match: {agent.name} - capability {capability_match:.2f}, "
                   f"calibrated confidence {calibrated_confidence:.2f}, "
                   f"counterfactual lift {counterfactual_lift:.2f}")
        elif score > 0.3:
            return (f"Moderate match: {agent.name} - "
                   f"calibrated confidence {calibrated_confidence:.2f}")
        else:
            return f"Weak match: {agent.name} not well-suited for this task"
