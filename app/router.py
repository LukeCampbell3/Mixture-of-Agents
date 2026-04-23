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
        """Select agents for task execution.

        Selection strategy:
        - Score all routable agents.
        - Always select the top-scoring agent (the specialist or best match).
        - Fill remaining slots up to max_agents with agents that score above
          a low relevance floor (0.15), so genuinely irrelevant agents are
          excluded but good supporting agents are included.
        - Never select fewer than min(max_agents, 1) agents.
        """
        candidates = self.registry.get_routable_agents()

        oracle_subset = None
        if self.counterfactual_store:
            oracle_subset = self.counterfactual_store.get_oracle_subset(task_frame.task_id)

        # Score every candidate
        agent_scores = []
        for agent in candidates:
            score = self._score_agent(agent, task_frame, oracle_subset)
            agent_scores.append(score)

        agent_scores.sort(key=lambda x: x.activation_score, reverse=True)

        # Relevance floor — only exclude agents that are clearly irrelevant
        RELEVANCE_FLOOR = 0.15

        selected = []
        suppressed = []

        for score in agent_scores:
            if len(selected) >= max_agents:
                suppressed.append({"agent_id": score.agent_id, "reason": "budget_exclusion"})
                continue

            # Always take the top agent regardless of score
            if len(selected) == 0:
                selected.append(score.agent_id)
            elif score.activation_score >= RELEVANCE_FLOOR:
                selected.append(score.agent_id)
            else:
                suppressed.append({"agent_id": score.agent_id, "reason": "below_relevance_floor"})

        decision = RoutingDecision(
            task_id=task_frame.task_id,
            candidate_agents=agent_scores,
            selected_agents=selected,
            suppressed_agents=suppressed,
            budget_plan={
                "expected_tokens": len(selected) * 400,
                "expected_latency": len(selected) * 30.0,
            },
            routing_reasons=[
                f"Selected {len(selected)}/{max_agents} agents (relevance floor={RELEVANCE_FLOOR})",
                f"Task type: {task_frame.task_type}",
                f"Uncertainty: {task_frame.initial_uncertainty:.2f}",
            ],
            uncertainty_summary=f"Initial uncertainty: {task_frame.initial_uncertainty:.2f}",
            spawn_recommendation=None,
            no_spawn_reason="Existing agents sufficient for task",
            arbitration_needed=len(selected) > 1,
        )

        return decision

    def route_primary(self, task_frame: TaskFrame) -> RoutingDecision:
        """Return only the single best agent for lazy-chain execution.

        The caller runs this agent first, then calls needs_escalation() on
        its output to decide whether to warm up additional agents.
        """
        candidates = self.registry.get_routable_agents()

        oracle_subset = None
        if self.counterfactual_store:
            oracle_subset = self.counterfactual_store.get_oracle_subset(task_frame.task_id)

        agent_scores = [
            self._score_agent(a, task_frame, oracle_subset) for a in candidates
        ]
        agent_scores.sort(key=lambda x: x.activation_score, reverse=True)

        best = agent_scores[0].agent_id if agent_scores else "code_primary"
        suppressed = [
            {"agent_id": s.agent_id, "reason": "lazy_chain_deferred"}
            for s in agent_scores[1:]
        ]

        return RoutingDecision(
            task_id=task_frame.task_id,
            candidate_agents=agent_scores,
            selected_agents=[best],
            suppressed_agents=suppressed,
            budget_plan={"expected_tokens": 400, "expected_latency": 30.0},
            routing_reasons=[
                f"Lazy-chain primary: {best} (score {agent_scores[0].activation_score:.2f})",
                f"Task type: {task_frame.task_type}",
            ],
            uncertainty_summary=f"Initial uncertainty: {task_frame.initial_uncertainty:.2f}",
            spawn_recommendation=None,
            no_spawn_reason="Lazy chain — escalate only if needed",
            arbitration_needed=False,
        )

    def needs_escalation(self, output: str, task_frame: TaskFrame) -> Tuple[bool, List[str]]:
        """Inspect a primary agent's output for signals that sub-agents are needed.

        Returns (should_escalate, list_of_agent_ids_to_add).

        Escalation triggers:
        - Explicit uncertainty phrases ("I'm not sure", "you may want to verify", etc.)
        - Output is very short relative to task complexity
        - Task type is HYBRID and output lacks one of the two required parts
        - Output contains domain keywords that belong to a different specialist
        """
        text = output.lower()
        task_type_str = (
            task_frame.task_type
            if isinstance(task_frame.task_type, str)
            else task_frame.task_type.value
        )

        # ── Uncertainty phrases ──────────────────────────────────────────────
        UNCERTAINTY_PHRASES = [
            "i'm not sure", "i am not sure", "not certain", "you may want to verify",
            "you should verify", "i cannot confirm", "i don't know", "unclear",
            "consult a", "double-check", "double check", "may be incorrect",
            "might be wrong", "please verify", "i would recommend checking",
        ]
        has_uncertainty = any(p in text for p in UNCERTAINTY_PHRASES)

        # ── Short output on a complex task ───────────────────────────────────
        is_short = len(output.split()) < 60 and task_frame.difficulty_estimate > 0.5

        # ── Hybrid task missing a part ───────────────────────────────────────
        hybrid_gap = False
        if "hybrid" in task_type_str:
            has_code  = "```" in output
            has_prose = len([w for w in output.split() if w.isalpha()]) > 40
            hybrid_gap = not (has_code and has_prose)

        # ── Domain keywords that belong to a different specialist ────────────
        DOMAIN_ESCALATION = {
            "web_research": [
                "latest", "current", "recent", "as of 2024", "as of 2025",
                "according to", "documentation says", "official docs",
            ],
            "critic_verifier": [
                "should be reviewed", "needs testing", "edge case",
                "potential bug", "security concern", "race condition",
            ],
        }
        escalate_to: List[str] = []
        for agent_id, phrases in DOMAIN_ESCALATION.items():
            if any(p in text for p in phrases):
                escalate_to.append(agent_id)

        openmythos_refiners: List[str] = []
        if "coding" in task_type_str or "hybrid" in task_type_str:
            task_text = task_frame.normalized_request.lower()
            refiner_triggers = [
                "implement",
                "thread",
                "concurrent",
                "test",
                "edge case",
                "debug",
                "fix",
                "refactor",
                "performance",
                "error handling",
                "database",
                "api",
                "cache",
                "decorator",
            ]
            output_triggers = [
                "test",
                "edge case",
                "potential bug",
                "race condition",
                "timed out",
                "max iterations",
                "needs testing",
            ]
            if (
                any(trigger in task_text for trigger in refiner_triggers)
                or any(trigger in text for trigger in output_triggers)
            ):
                openmythos_refiners = self._openmythos_refinement_agents()
                for refiner_id in reversed(openmythos_refiners):
                    if refiner_id not in escalate_to:
                        escalate_to.insert(0, refiner_id)

        # ── Decision ─────────────────────────────────────────────────────────
        should_escalate = (
            has_uncertainty
            or is_short
            or hybrid_gap
            or bool(escalate_to)
            or bool(openmythos_refiners)
        )

        if should_escalate and not escalate_to:
            # Default escalation: add critic_verifier for quality check
            escalate_to = ["critic_verifier"]

        return should_escalate, escalate_to

    def _openmythos_refinement_agents(self) -> List[str]:
        """Return routable OpenMythos coding refiners, strongest first."""

        refiners = []
        for agent in self.registry.get_routable_agents():
            tags = set(agent.tags or [])
            if "openmythos" not in tags or "subagent" not in tags:
                continue
            if agent.domain != "coding" and "coding" not in tags and "code" not in tags:
                continue
            if "refinement" not in tags and "refiner" not in agent.agent_id:
                continue
            refiners.append(agent)

        refiners.sort(
            key=lambda agent: (
                agent.calibration_score,
                agent.average_quality_lift,
                agent.expected_activation_rate,
            ),
            reverse=True,
        )
        return [agent.agent_id for agent in refiners]

    def build_sub_task(self, primary_output: str, task_frame: TaskFrame, sub_agent_id: str) -> str:
        """Build a focused sub-task prompt for a sub-agent.

        The sub-agent receives only what it needs — the primary output as
        context and a narrow instruction matching its role — not the full
        original prompt with all its overhead.
        """
        task = task_frame.normalized_request

        if sub_agent_id == "critic_verifier":
            return (
                f"Review the following response to this task and identify any issues:\n\n"
                f"TASK: {task}\n\n"
                f"RESPONSE TO REVIEW:\n{primary_output}\n\n"
                f"Identify: factual errors, missing edge cases, security issues, "
                f"or logical gaps. Be concise. If the response is correct, say so."
            )

        if sub_agent_id == "web_research":
            return (
                f"The following response may contain outdated information. "
                f"Verify or supplement it with current facts.\n\n"
                f"TASK: {task}\n\n"
                f"RESPONSE TO VERIFY:\n{primary_output}\n\n"
                f"Provide only corrections or additions. Do not repeat what is already correct."
            )

        # Generic sub-task for specialist agents
        domain = sub_agent_id.replace("specialist_", "").replace("_", " ")
        return (
            f"You are a {domain} specialist. The following response needs your expertise.\n\n"
            f"TASK: {task}\n\n"
            f"PRIMARY RESPONSE:\n{primary_output}\n\n"
            f"Add or correct anything in your domain ({domain}). Be concise."
        )

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
        domain_match = 0.5

        task_type = task_frame.task_type
        task_type_str = task_type if isinstance(task_type, str) else task_type.value
        task_text_lower = task_frame.normalized_request.lower()

        # Base agents
        if agent.agent_id == "code_primary":
            if "coding" in task_type_str:
                domain_match = 0.85
            elif task_type_str in ("planning", "unknown"):
                domain_match = 0.5
            else:
                domain_match = 0.3

        elif agent.agent_id == "web_research":
            if "research" in task_type_str:
                domain_match = 0.95
            elif "coding" in task_type_str:
                domain_match = 0.2
            else:
                domain_match = 0.4

        elif agent.agent_id == "critic_verifier":
            if any(kw in task_text_lower for kw in
                   ["review", "check", "verify", "validate", "audit", "critique"]):
                domain_match = 0.85
            else:
                domain_match = 0.15

        # Dynamically spawned specialists
        else:
            agent_keywords = set(agent.tags or []) | {agent.domain}
            agent_keywords -= {"spawned", "dynamic", "probationary"}
            hits = sum(
                1 for kw in agent_keywords
                if kw.replace("_", " ") in task_text_lower or kw in task_text_lower
            )
            if hits > 0:
                domain_match = min(0.95, 0.6 + hits * 0.1)
            else:
                domain_words = agent.domain.replace("_", " ").split()
                if any(w in task_text_lower for w in domain_words if len(w) > 3):
                    domain_match = 0.75
                else:
                    domain_match = 0.2

        # Tool match bonus
        tool_match = 0.0
        if agent.tools and task_frame.likely_tools:
            common = set(agent.tools) & set(task_frame.likely_tools)
            tool_match = len(common) / len(set(agent.tools) | set(task_frame.likely_tools))

        return min(1.0, 0.8 * domain_match + 0.2 * tool_match)

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
