"""Main orchestrator for the agentic network with parallel execution and skill packs."""

from typing import Dict, Any, Optional, List
from app.router import Router
from app.budget_controller import BudgetController
from app.validator import Validator
from app.schemas.registry import AgentRegistry, AgentSpec, LifecycleState
from app.schemas.run_state import RunState, ToolCall
from app.models.llm_client import LLMClient, create_llm_client
from app.models.embeddings import EmbeddingGenerator
from app.models.uncertainty import UncertaintyEstimator
from app.agents.code_primary import CodePrimaryAgent
from app.agents.web_research import WebResearchAgent
from app.agents.critic_verifier import CriticVerifierAgent
from app.storage.registry_store import RegistryStore
from app.storage.artifact_store import ArtifactStore
from app.calibration import ThreeLevelCalibrator
from app.lead_agent_pattern import (
    prevent_free_form_collaboration,
    LeadAgentCoordinator,
    BoundedOutput,
    AgentRole
)
from app.parallel_executor import ParallelExecutor, SharedContext, AgentTask
from app.skill_packs import get_skill_pack_registry
from app.lifecycle import LifecycleManager
from app.gap_analyzer import GapAnalyzer
from app.agent_factory import AgentFactory, DynamicAgent
from datetime import datetime
import json
from pathlib import Path


class Orchestrator:
    """Main orchestration engine for sparse agent network."""
    
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        router_model: Optional[str] = None,
        budget_mode: str = "balanced",
        data_dir: str = "data",
        use_lead_agent_pattern: bool = True,
        enable_parallel: bool = True,
        max_parallel_agents: int = 3
    ):
        """Initialize orchestrator.
        
        Args:
            llm_provider:        LLM provider ("openai", "anthropic", "ollama", "local")
            llm_model:           Worker model name — used for all agent execution
            llm_base_url:        Base URL for local LLM API (optional)
            router_model:        Lightweight model for routing/classification only.
                                 When None the worker model is reused for routing.
                                 On constrained hardware set to "qwen2.5:0.5b".
            budget_mode:         Budget mode ("low", "balanced", "thorough")
            data_dir:            Directory for persistent data
            use_lead_agent_pattern: Whether to use lead-agent pattern (default: True)
            enable_parallel:     Enable parallel agent execution (default: True)
            max_parallel_agents: Maximum parallel agents (default: 3)
        """
        # ── worker model (all agent execution) ──────────────────────────────
        self.llm_client = create_llm_client(llm_provider, llm_model, llm_base_url)

        # ── router model (classification only, may be smaller) ───────────────
        if router_model and router_model != llm_model:
            self.router_llm_client = create_llm_client(
                llm_provider, router_model, llm_base_url
            )
        else:
            self.router_llm_client = self.llm_client   # reuse worker

        self.embedding_generator = EmbeddingGenerator()
        self.uncertainty_estimator = UncertaintyEstimator()
        
        # Initialize storage
        self.registry_store = RegistryStore(data_dir)
        self.artifact_store = ArtifactStore(data_dir)
        
        # Load or create agent registry
        self.registry = self._initialize_registry()
        
        # Initialize calibrator
        self.calibrator = self._initialize_calibrator(data_dir)
        
        # Initialize components — router uses the lightweight router_llm_client
        self.router = Router(
            self.registry,
            self.router_llm_client,   # small model for classification
            self.embedding_generator,
            self.uncertainty_estimator,
            calibrator=self.calibrator
        )
        self.validator = Validator()
        
        # Lead-agent pattern
        self.use_lead_agent_pattern = use_lead_agent_pattern
        self.lead_coordinator = LeadAgentCoordinator(max_supporting_agents=2)
        
        # Parallel execution
        self.enable_parallel = enable_parallel
        self.parallel_executor = ParallelExecutor(max_workers=max_parallel_agents) if enable_parallel else None
        
        # Skill pack registry
        self.skill_pack_registry = get_skill_pack_registry()
        
        # Budget mode + device-profile agent cap
        self.budget_mode = budget_mode
        self.max_parallel_agents = max_parallel_agents  # used as hard cap in run_task
        
        # Lifecycle manager
        self.lifecycle_manager = LifecycleManager(
            self.registry,
            self.embedding_generator
        )
        
        # Agent factory for dynamic agent creation
        self.agent_factory = AgentFactory(self.llm_client)
        
        # Gap analyzer
        self.gap_analyzer = GapAnalyzer(
            self.registry,
            self.embedding_generator
        )
    
    def _initialize_calibrator(self, data_dir: str) -> Optional[ThreeLevelCalibrator]:
        """Initialize or load calibrator."""
        calibrator_path = Path(data_dir) / "calibration"
        
        calibrator = ThreeLevelCalibrator(
            task_families=["coding", "research", "reasoning", "mixed"]
        )
        
        if calibrator_path.exists():
            try:
                calibrator.load(str(calibrator_path))
            except Exception:
                pass  # Use fresh calibrator
        
        return calibrator
    
    def run_task(self, user_request: str) -> RunState:
        """Execute a task through the agent network.

        Before routing, checks whether any existing agent is a good fit.
        If not, spawns a specialist on-demand using the LLM, registers it,
        and routes to it immediately — no waiting for history to accumulate.
        """
        # Initialize budget controller — honour device profile's agent cap
        budget_controller = BudgetController(
            mode=self.budget_mode,
            max_agents_override=self.max_parallel_agents
        )
        budget_controller.start_execution()

        # Frame the task
        task_frame = self.router.frame_task(user_request)
        task_family = self.router._get_task_family(task_frame.task_type)

        # ── On-demand spawn: detect gap before routing ────────────────────
        self._maybe_spawn_specialist(task_frame)

        # Initialize run state
        run_state = RunState(
            task_id=task_frame.task_id,
            task_frame=task_frame.model_dump(),
            active_agents=[],
            suppressed_agents=[],
            budget_usage={},
            base_model_version=self.llm_client.get_model_name(),
            pool_size_before=len(self.registry.agents)  # Track pool size
        )
        
        # Route to agents
        routing_decision = self.router.route(
            task_frame,
            max_agents=budget_controller.get_remaining_agents()
        )
        
        # Use lead-agent pattern if enabled
        if self.use_lead_agent_pattern and len(routing_decision.selected_agents) > 1:
            final_answer = self._execute_with_lead_pattern(
                task_frame,
                routing_decision,
                budget_controller,
                run_state,
                task_family
            )
        else:
            # Fall back to traditional execution
            final_answer = self._execute_traditional(
                task_frame,
                routing_decision,
                budget_controller,
                run_state
            )
        
        # Validate output
        agent_outputs = {agent_id: {"output": final_answer} for agent_id in run_state.active_agents}
        validation_report = self.validator.validate_output(
            task_frame,
            agent_outputs,
            final_answer
        )
        
        # Update run state
        run_state.final_answer = final_answer
        run_state.validation_report = validation_report.model_dump()
        run_state.final_state = validation_report.validation_state if isinstance(validation_report.validation_state, str) else validation_report.validation_state.value
        run_state.completed_at = datetime.utcnow().isoformat()
        run_state.budget_usage = budget_controller.get_status().model_dump()
        run_state.pool_size_after = len(self.registry.agents)  # Track pool size after
        
        # Save artifacts
        self._save_artifacts(run_state, final_answer, routing_decision)
        
        # Record agent performance for counterfactual lift
        quality_score = 0.8 if run_state.final_state == "success" else 0.3
        for agent_id in run_state.active_agents:
            self.router.record_agent_performance(agent_id, quality_score)
            # Track probationary agents that were activated
            agent_spec = self.registry.get_agent(agent_id)
            if agent_spec and agent_spec.lifecycle_state == LifecycleState.PROBATIONARY:
                run_state.probationary_agents_used.append(agent_id)
        
        # Record task execution in lifecycle manager for history tracking
        self.lifecycle_manager.record_task_execution(
            task_frame.model_dump(),
            routing_decision.model_dump(),
            {
                "task_id": run_state.task_id,
                "final_state": run_state.final_state,
                "active_agents": run_state.active_agents
            }
        )
        
        # Lifecycle evaluation - check if spawning is needed
        lifecycle_recommendations = self._evaluate_lifecycle(
            task_frame,
            routing_decision,
            run_state
        )
        run_state.spawn_recommendations = lifecycle_recommendations.get("spawn_recommendations", [])
        run_state.spawned_agents = lifecycle_recommendations.get("spawned_agents", [])
        run_state.promoted_agents = lifecycle_recommendations.get("promoted_agents", [])
        run_state.pruned_agents = lifecycle_recommendations.get("pruned_agents", [])
        run_state.lifecycle_events = lifecycle_recommendations.get("lifecycle_events", [])
        
        return run_state
    
    def _evaluate_lifecycle(
        self,
        task_frame,
        routing_decision,
        run_state
    ) -> Dict[str, Any]:
        """Evaluate lifecycle needs and make recommendations.
        
        Returns recommendations for:
        - Spawning new specialists for recurring unmet demand
        - Promoting probationary agents
        - Demoting/pruning underperforming agents
        """
        recommendations = {
            "spawn_recommendations": [],
            "spawned_agents": [],
            "promoted_agents": [],
            "pruned_agents": [],
            "lifecycle_events": []
        }
        
        # Get task history for lifecycle evaluation
        task_history = self._get_task_history()
        
        # Check if we should spawn a specialist for this task
        should_spawn, spawn_spec = self.lifecycle_manager.evaluate_spawn_need(
            task_history,
            {"current_task": task_frame.model_dump()}
        )
        
        if should_spawn and spawn_spec:
            spawn_recommendation = {
                "agent_id": spawn_spec["agent_id"],
                "spawn_score": spawn_spec.get("spawn_score", 0),
                "reason": spawn_spec.get("reason", "Unmet demand detected"),
                "domain": spawn_spec.get("domain", "unknown"),
                "tools": spawn_spec.get("tools", [])
            }
            recommendations["spawn_recommendations"].append(spawn_recommendation)
            
            # If spawn score meets threshold, actually spawn
            if spawn_recommendation.get("spawn_score", 0) >= self.lifecycle_manager.min_spawn_score:
                # Create a new agent spec
                new_agent_id = spawn_spec["agent_id"]
                new_agent_spec = AgentSpec(
                    agent_id=new_agent_id,
                    name=spawn_spec.get("name", f"Specialist for {spawn_spec.get('domain', 'unknown')}"),
                    description=spawn_spec.get("description", "Specialized agent for specific task types"),
                    domain=spawn_spec.get("domain", "unknown"),
                    lifecycle_state="probationary",
                    tools=spawn_spec.get("tools", ["repo_tool", "web_tool"]),
                    tags=["spawned", "specialist"]
                )
                
                self.registry.add_agent(new_agent_spec)
                self.registry_store.save_registry(self.registry)  # Persist immediately
                recommendations["spawned_agents"].append(new_agent_id)
                
                # Initialize performance tracking for new agent
                self.lifecycle_manager.agent_performance[new_agent_id] = {
                    "activation_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "task_families": set()
                }
                
                # Record spawn time for grace period
                self.lifecycle_manager.spawned_agents[new_agent_id] = {
                    "spawn_task_count": len(self.lifecycle_manager.task_history),
                    "spawn_time": datetime.utcnow().isoformat()
                }
                
                recommendations["lifecycle_events"].append({
                    "event_type": "spawn",
                    "agent_id": new_agent_id,
                    "reason": spawn_recommendation.get("reason", "Unmet demand detected"),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Check for promotion/demotion
        promotion_recommendations = self.lifecycle_manager.evaluate_promotions()
        recommendations["promoted_agents"] = promotion_recommendations.get("promoted", [])
        recommendations["lifecycle_events"].extend(promotion_recommendations.get("events", []))
        
        # Check for pruning
        pruning_recommendations = self.lifecycle_manager.evaluate_pruning()
        recommendations["pruned_agents"] = pruning_recommendations.get("pruned", [])
        recommendations["lifecycle_events"].extend(pruning_recommendations.get("events", []))
        
        return recommendations
    
    def _get_task_history(self) -> List[Dict[str, Any]]:
        """Get recent task execution history for lifecycle evaluation."""
        # Return history from lifecycle manager
        return self.lifecycle_manager.task_history
    
    def _execute_with_lead_pattern(
        self,
        task_frame,
        routing_decision,
        budget_controller,
        run_state,
        task_family: str
    ) -> str:
        """Execute using lead-agent pattern to prevent negative synergy."""
        # Get agent scores and calibrated confidences
        agent_scores = {
            score.agent_id: score.activation_score
            for score in routing_decision.candidate_agents
        }
        
        calibrated_confidences = {}
        for agent_id in routing_decision.selected_agents:
            raw_conf = agent_scores.get(agent_id, 0.5)
            calibrated = self.calibrator.calibrate_router(raw_conf, task_family) \
                if self.calibrator else raw_conf
            calibrated_confidences[agent_id] = calibrated
        
        # Select lead and supporting agents
        plan = prevent_free_form_collaboration(
            task=task_frame.normalized_request,
            available_agents=routing_decision.selected_agents,
            agent_scores=agent_scores,
            calibrated_confidences=calibrated_confidences,
            task_type=task_frame.task_type if isinstance(task_frame.task_type, str) else task_frame.task_type.value
        )
        
        lead_agent_id = plan["lead_agent"]
        supporting_agents = plan["supporting_agents"]
        
        # Execute lead agent
        if not budget_controller.can_activate_agent():
            return "Budget exhausted before lead agent execution"
        
        budget_controller.activate_agent()
        run_state.active_agents.append(lead_agent_id)
        
        lead_agent_spec = self.registry.get_agent(lead_agent_id)
        lead_agent = self._create_agent_instance(lead_agent_spec)
        
        task_context = {
            "task_frame": task_frame,
            "constraints": task_frame.hard_constraints,
            "available_tools": lead_agent_spec.tools
        }
        
        lead_output = lead_agent.execute(task_context)
        budget_controller.deactivate_agent()
        
        # Collect bounded outputs from supporting agents
        bounded_outputs = {}
        for agent_id, role in supporting_agents.items():
            if not budget_controller.can_activate_agent():
                break
            
            budget_controller.activate_agent()
            run_state.active_agents.append(agent_id)
            
            agent_spec = self.registry.get_agent(agent_id)
            agent = self._create_agent_instance(agent_spec)
            
            # Execute with role-specific instructions
            role_context = {
                "task_frame": task_frame,
                "lead_output": lead_output["output"],
                "role": role.value,
                "available_tools": agent_spec.tools
            }
            
            output = agent.execute(role_context)
            
            # Create bounded output
            bounded_outputs[agent_id] = BoundedOutput(
                agent_id=agent_id,
                role=role,
                content=output["output"],
                confidence=calibrated_confidences.get(agent_id, 0.5),
                has_objection=(role == AgentRole.CRITIC and "concern" in output["output"].lower())
            )
            
            budget_controller.deactivate_agent()
        
        # Synthesize with lead having authority
        collected = self.lead_coordinator.collect_bounded_outputs(
            lead_output["output"],
            bounded_outputs
        )
        
        final_answer = self.lead_coordinator.synthesize_final_answer(
            lead_output["output"],
            collected,
            []  # Resolutions would come from arbitration
        )
        
        return final_answer
    
    def _execute_traditional(
        self,
        task_frame,
        routing_decision,
        budget_controller,
        run_state
    ) -> str:
        """Collaborative execution with parallel processing, iterative refinement, and skill packs."""
        from app.arbitration import Arbitrator
        
        # Initialize arbitrator
        arbitrator = Arbitrator(self.llm_client)
        
        # Initialize shared context for parallel execution
        shared_context_obj = SharedContext()
        shared_context_text = f"Task: {task_frame.normalized_request}\n\n"
        shared_context_obj.update("task_description", task_frame.normalized_request)
        
        # Find relevant skill packs for this task
        task_type_str = task_frame.task_type if isinstance(task_frame.task_type, str) else task_frame.task_type.value
        keywords = task_frame.normalized_request.lower().split()
        
        # Map difficulty estimate to string
        difficulty = "medium"
        if task_frame.difficulty_estimate < 0.4:
            difficulty = "easy"
        elif task_frame.difficulty_estimate > 0.6:
            difficulty = "hard"
        
        skill_packs = self.skill_pack_registry.find_packs_for_task(
            task_type_str,
            keywords,
            difficulty
        )
        
        # PHASE 1: Initial execution (parallel if enabled)
        agent_outputs = {}
        
        if self.enable_parallel and len(routing_decision.selected_agents) > 1:
            # PARALLEL EXECUTION
            tasks = []
            for agent_id in routing_decision.selected_agents:
                if not budget_controller.can_activate_agent():
                    break
                
                budget_controller.activate_agent()
                run_state.active_agents.append(agent_id)
                
                agent_spec = self.registry.get_agent(agent_id)
                if not agent_spec:
                    continue
                
                agent = self._create_agent_instance(agent_spec)
                
                # Assign relevant skill packs to agent
                agent_skill_packs = self._select_skill_packs_for_agent(
                    agent_spec,
                    skill_packs
                )
                
                task_context = {
                    "task_frame": task_frame,
                    "shared_context": shared_context_text,
                    "constraints": task_frame.hard_constraints,
                    "available_tools": agent_spec.tools,
                    "iteration": 1
                }
                
                tasks.append(AgentTask(
                    agent_id=agent_id,
                    agent_instance=agent,
                    context=task_context,
                    skill_packs=agent_skill_packs
                ))
            
            # Execute all agents in parallel
            results = self.parallel_executor.execute_parallel(
                tasks,
                shared_context_obj,
                timeout=60.0
            )
            
            # Collect outputs
            for result in results:
                if result.success:
                    agent_outputs[result.agent_id] = result.output
                    agent_spec = self.registry.get_agent(result.agent_id)
                    shared_context_text += f"\n## {agent_spec.name if agent_spec else result.agent_id} Analysis:\n{result.output['output']}\n"
                budget_controller.deactivate_agent()
        
        else:
            # SEQUENTIAL EXECUTION (fallback or single agent)
            for agent_id in routing_decision.selected_agents:
                if not budget_controller.can_activate_agent():
                    break
                
                budget_controller.activate_agent()
                run_state.active_agents.append(agent_id)
                
                agent_spec = self.registry.get_agent(agent_id)
                if not agent_spec:
                    continue
                
                agent = self._create_agent_instance(agent_spec)
                
                # Assign relevant skill packs
                agent_skill_packs = self._select_skill_packs_for_agent(
                    agent_spec,
                    skill_packs
                )
                
                # Execute agent WITH CONTEXT from other agents
                task_context = {
                    "task_frame": task_frame,
                    "shared_context": shared_context_text,
                    "constraints": task_frame.hard_constraints,
                    "available_tools": agent_spec.tools,
                    "other_agent_outputs": agent_outputs,
                    "iteration": 1,
                    "skill_packs": agent_skill_packs
                }
                
                output = agent.execute(task_context)
                agent_outputs[agent_id] = output
                
                # Update shared context IMMEDIATELY
                shared_context_text += f"\n## {agent_spec.name} Analysis:\n{output['output']}\n"
                
                budget_controller.deactivate_agent()
        
        # Store suppressed agents
        run_state.suppressed_agents = routing_decision.suppressed_agents
        
        # PHASE 2: Conflict detection and arbitration
        conflicts = []
        arbitration_results = []
        if len(agent_outputs) > 1:
            conflicts = arbitrator.detect_conflicts(
                agent_outputs,
                {"task_frame": task_frame}
            )
            
            # Arbitrate each conflict
            for conflict in conflicts:
                result = arbitrator.arbitrate(
                    conflict,
                    agent_outputs,
                    {"task_frame": task_frame}
                )
                arbitration_results.append(result)
                
                # Add arbitration to shared context
                shared_context_text += f"\n## Arbitration: {conflict.description}\n{result['resolution']}\n"
        
        # PHASE 3: Iterative refinement (if multiple agents and conflicts found)
        if len(agent_outputs) > 1 and conflicts:
            refined_outputs = {}
            
            # Prepare refinement tasks
            refinement_tasks = []
            for agent_id in routing_decision.selected_agents:
                if not budget_controller.can_activate_agent():
                    break
                
                budget_controller.activate_agent()
                
                agent_spec = self.registry.get_agent(agent_id)
                if not agent_spec:
                    continue
                
                agent = self._create_agent_instance(agent_spec)
                
                # Refinement context includes arbitration results
                refinement_context = {
                    "task_frame": task_frame,
                    "shared_context": shared_context_text,
                    "constraints": task_frame.hard_constraints,
                    "available_tools": agent_spec.tools,
                    "other_agent_outputs": agent_outputs,
                    "conflicts_detected": conflicts,
                    "arbitration_results": arbitration_results,
                    "iteration": 2,
                    "refinement_mode": True
                }
                
                refinement_tasks.append(AgentTask(
                    agent_id=agent_id,
                    agent_instance=agent,
                    context=refinement_context,
                    skill_packs=self._select_skill_packs_for_agent(agent_spec, skill_packs)
                ))
            
            # Execute refinement (parallel if enabled)
            if self.enable_parallel and len(refinement_tasks) > 1:
                refinement_results = self.parallel_executor.execute_parallel(
                    refinement_tasks,
                    shared_context_obj,
                    timeout=60.0
                )
                
                for result in refinement_results:
                    if result.success:
                        refined_outputs[result.agent_id] = result.output
                        agent_spec = self.registry.get_agent(result.agent_id)
                        shared_context_text += f"\n## {agent_spec.name if agent_spec else result.agent_id} Refinement:\n{result.output['output']}\n"
                    budget_controller.deactivate_agent()
            else:
                # Sequential refinement
                for task in refinement_tasks:
                    refined_output = task.agent_instance.execute(task.context)
                    refined_outputs[task.agent_id] = refined_output
                    
                    agent_spec = self.registry.get_agent(task.agent_id)
                    shared_context_text += f"\n## {agent_spec.name if agent_spec else task.agent_id} Refinement:\n{refined_output['output']}\n"
                    
                    budget_controller.deactivate_agent()
            
            # Use refined outputs if available
            if refined_outputs:
                agent_outputs = refined_outputs
        
        # PHASE 4: Collaborative synthesis
        if len(agent_outputs) > 1:
            final_answer = self._collaborative_synthesis(
                task_frame,
                agent_outputs,
                shared_context_text,
                conflicts,
                arbitration_results
            )
        else:
            # Single agent - just return its output
            final_answer = list(agent_outputs.values())[0]["output"]
        
        return final_answer
    
    def _select_skill_packs_for_agent(
        self,
        agent_spec: AgentSpec,
        available_packs: List[Any]
    ) -> List[Any]:
        """Select relevant skill packs for an agent based on its domain and tags."""
        selected = []
        
        domain_pack_map = {
            "coding":         ["algorithm_optimization", "code_review", "debugging_mode", "security_focused"],
            "research":       ["fact_checking", "sorting_comparison", "architecture_comparison"],
            "verification":   ["logical_analysis", "quantitative_analysis", "code_review"],
            "security_audit": ["security_focused", "code_review"],
            "api_migration":  ["code_review", "implementation_with_research"],
            "devops":         ["code_review", "security_focused", "implementation_with_research"],
            "data":           ["quantitative_analysis", "algorithm_optimization"],
            "database":       ["algorithm_optimization", "code_review"],
            "testing":        ["code_review", "logical_analysis"],
            "api":            ["code_review", "implementation_with_research", "security_focused"],
            "documentation":  ["code_review"],
            "refactoring":    ["code_review", "algorithm_optimization", "security_focused"],
        }
        
        # Universal packs that apply to all agents
        universal_packs = {"implementation_with_research", "security_focused"}
        
        domain_packs = set(domain_pack_map.get(agent_spec.domain, []))
        
        for pack in available_packs:
            if pack.pack_id in domain_packs or pack.pack_id in universal_packs:
                if pack not in selected:
                    selected.append(pack)
        
        return selected
    
    def _maybe_spawn_specialist(self, task_frame) -> None:
        """
        Spawn a specialist agent on-demand when the task has clear specialist
        signals but no matching specialist exists yet.

        Logic:
        1. Detect the specialist domain from the task text.
        2. If no specialist for that domain exists, spawn one immediately.
        3. If a specialist already exists, do nothing (it will be routed to).
        4. If no specialist domain is detected, fall back to base agents.
        """
        # Detect the specialist domain from the task text
        domain = self._detect_specialist_domain(task_frame.normalized_request)
        if domain is None:
            return  # No specialist signal — base agents are fine

        # Check if we already have a specialist for this domain
        agent_id = f"specialist_{domain}"
        if self.registry.get_agent(agent_id) is not None:
            return  # Already exists — router will select it

        # Spawn a new specialist
        print(f"\n  [spawn] Creating specialist: {domain}")
        spec, _instance = self.agent_factory.spawn_for_task(
            task_text=task_frame.normalized_request,
            domain=domain,
            agent_id=agent_id,
        )

        # Register immediately so the router sees it on this request
        self.registry.add_agent(spec)
        self.registry_store.save_registry(self.registry)

        # Refresh gap analyzer embeddings
        self.gap_analyzer.agent_embeddings[agent_id] = (
            self.gap_analyzer.embedding_generator.embed(
                f"{spec.name}: {spec.description}. Domain: {spec.domain}"
            )
        )

        # Track in lifecycle manager
        self.lifecycle_manager.spawned_agents[agent_id] = {
            "spawn_task_count": len(self.lifecycle_manager.task_history),
            "spawn_time": __import__("datetime").datetime.utcnow().isoformat(),
        }
        self.lifecycle_manager.agent_performance[agent_id] = {
            "activation_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "task_families": set(),
        }

    # Domain keyword map — used to detect what specialist to spawn
    _DOMAIN_KEYWORDS: dict = {
        "devops":         ["docker", "kubernetes", "k8s", "ci/cd", "pipeline", "terraform",
                           "ansible", "helm", "deploy", "nginx", "cloud", "aws", "gcp", "azure",
                           "container", "pod", "service mesh", "github actions", "gitlab ci"],
        "data_analysis":  ["pandas", "numpy", "dataframe", "csv", "dataset", "plot", "chart",
                           "visualize", "visualization", "statistics", "regression", "correlation",
                           "machine learning", "sklearn", "seaborn", "matplotlib", "jupyter"],
        "security":       ["security", "vulnerability", "exploit", "injection", "xss", "csrf",
                           "authentication", "authorization", "oauth", "jwt", "encrypt", "hash",
                           "penetration", "owasp", "cve", "audit", "hardening"],
        "database":       ["sql", "query", "database", "schema", "table", "index", "join",
                           "postgres", "postgresql", "mysql", "sqlite", "mongodb", "orm",
                           "migration", "transaction", "stored procedure"],
        "api":            ["fastapi", "flask", "express", "django rest", "rest api", "graphql",
                           "openapi", "swagger", "endpoint", "webhook", "http client",
                           "rate limit", "pagination", "versioning"],
        "testing":        ["unit test", "integration test", "pytest", "jest", "vitest",
                           "mock", "stub", "fixture", "tdd", "bdd", "coverage", "assert",
                           "test suite", "end-to-end", "e2e"],
        "documentation":  ["docstring", "jsdoc", "tsdoc", "readme", "sphinx", "mkdocs",
                           "write docs", "document this", "add comments", "api reference",
                           "architecture doc"],
        "refactoring":    ["refactor", "code smell", "technical debt", "clean up",
                           "solid principle", "dry principle", "design pattern",
                           "restructure", "simplify", "decouple", "extract method"],
        "mobile":         ["react native", "flutter", "swift", "kotlin", "ios", "android",
                           "mobile app", "xcode", "gradle", "expo"],
        "frontend":       ["react", "vue", "angular", "svelte", "css", "tailwind",
                           "component", "ui", "ux", "html", "dom", "typescript frontend",
                           "next.js", "nuxt", "vite"],
        "blockchain":     ["solidity", "smart contract", "ethereum", "web3", "nft",
                           "defi", "blockchain", "wallet", "token", "erc20"],
        "ml_engineering": ["pytorch", "tensorflow", "keras", "model training", "fine-tune",
                           "neural network", "transformer model", "llm", "embedding",
                           "inference", "gpu", "cuda", "huggingface"],
    }

    def _detect_specialist_domain(self, task_text: str) -> Optional[str]:
        """
        Return the best-matching specialist domain for a task, or None if
        no domain has enough keyword signal.
        """
        text = task_text.lower()
        scores: dict[str, int] = {}
        for domain, keywords in self._DOMAIN_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits > 0:
                scores[domain] = hits
        if not scores:
            return None
        return max(scores, key=scores.__getitem__)

    def _initialize_registry(self) -> AgentRegistry:
        """Initialize or load agent registry.

        Only three universal base agents are seeded here.
        All specialist agents are spawned dynamically at runtime
        when the router detects no good match for a task.
        """
        try:
            registry = self.registry_store.load_registry()
        except Exception:
            registry = None

        if registry and registry.agents:
            return registry

        registry = AgentRegistry()

        registry.add_agent(AgentSpec(
            agent_id="code_primary",
            name="Code Primary",
            description="General-purpose coding agent: implementation, debugging, architecture",
            domain="coding",
            lifecycle_state=LifecycleState.HOT,
            tools=["repo_tool", "test_runner"],
            tags=["coding", "implementation", "debugging"],
        ))

        registry.add_agent(AgentSpec(
            agent_id="web_research",
            name="Web Research",
            description="Research agent for current documentation and fact validation",
            domain="research",
            lifecycle_state=LifecycleState.WARM,
            tools=["web_tool", "citation_checker"],
            tags=["research", "documentation", "validation"],
        ))

        registry.add_agent(AgentSpec(
            agent_id="critic_verifier",
            name="Critic Verifier",
            description="Verification agent for consistency checking and risk assessment",
            domain="verification",
            lifecycle_state=LifecycleState.WARM,
            tools=["test_runner", "citation_checker"],
            tags=["verification", "testing", "quality"],
        ))

        self.registry_store.save_registry(registry)
        return registry
    
    def _create_agent_instance(self, agent_spec: AgentSpec):
        """Create agent instance from spec.

        The three universal base agents use their hand-written classes.
        Every other agent (spawned specialists) is instantiated via
        AgentFactory, which uses the LLM to generate a focused system prompt.
        """
        base_agents = {
            "code_primary":   CodePrimaryAgent,
            "web_research":   WebResearchAgent,
            "critic_verifier": CriticVerifierAgent,
        }

        agent_cls = base_agents.get(agent_spec.agent_id)
        if agent_cls:
            return agent_cls(
                agent_spec.agent_id,
                agent_spec.name,
                agent_spec.description,
                self.llm_client,
                agent_spec.tools,
            )

        # All spawned / dynamic agents go through the factory
        return self.agent_factory.create_agent_instance(agent_spec)
    
    def _collaborative_synthesis(
        self,
        task_frame,
        agent_outputs: Dict[str, Any],
        shared_context: str,
        conflicts: List[Any] = None,
        arbitration_results: List[Dict[str, Any]] = None
    ) -> str:
        """Collaborative synthesis that leverages multiple perspectives with conflict resolution."""
        if not agent_outputs:
            return "No agents were activated to answer this task."
        
        # Build synthesis prompt that emphasizes collaboration AND conflict resolution
        synthesis_prompt = f"""You are synthesizing insights from multiple specialized agents working together.

USER REQUEST:
{task_frame.normalized_request}

COLLABORATIVE ANALYSIS:
{shared_context}

"""
        
        # Add conflict resolution context if available
        if conflicts and arbitration_results:
            synthesis_prompt += f"""
CONFLICTS DETECTED AND RESOLVED:
{len(conflicts)} conflicts were identified and arbitrated:
"""
            for i, (conflict, resolution) in enumerate(zip(conflicts, arbitration_results)):
                synthesis_prompt += f"\n{i+1}. {conflict.description}\n   Resolution: {resolution['resolution'][:200]}...\n"
        
        synthesis_prompt += """
The agents have worked together through multiple phases:
1. Initial parallel analysis from different perspectives
2. Conflict detection and arbitration where disagreements arose
3. Iterative refinement based on arbitration results

Your task is to:
1. Integrate the BEST insights from each agent's refined analysis
2. Identify where agents COMPLEMENT each other (not just agree)
3. Incorporate arbitration decisions to resolve contradictions
4. Create a BETTER answer than any single agent could provide
5. Highlight synergies where combining perspectives adds unique value
6. Show how the multi-agent collaboration produced insights no single agent would have

Focus on creating an answer that is MORE COMPLETE, MORE ACCURATE, and MORE INSIGHTFUL than any individual agent's contribution. The answer should demonstrate clear value from the collaborative process.

Final synthesized answer:"""
        
        try:
            final_answer = self.llm_client.generate(synthesis_prompt, max_tokens=500, temperature=0.4)
        except Exception:
            # Fallback: return the best single-agent output
            final_answer = max(agent_outputs.values(), key=lambda o: len(o.get("output", "")))["output"]
        return final_answer
    
    def _synthesize_answer(
        self,
        task_frame,
        agent_outputs: Dict[str, Any],
        shared_context: str
    ) -> str:
        """Synthesize final answer from agent outputs."""
        # Delegate to collaborative synthesis
        return self._collaborative_synthesis(task_frame, agent_outputs, shared_context)
    
    def _save_artifacts(self, run_state: RunState, shared_context: str, routing_decision):
        """Save execution artifacts."""
        # Save run state
        self.artifact_store.save_run_state(run_state)
        
        # Save context
        self.artifact_store.save_context(run_state.task_id, shared_context)
        
        # Save analysis
        analysis_data = {
            "task_id": run_state.task_id,
            "routing_decision": routing_decision.model_dump(),
            "budget_status": run_state.budget_usage
        }
        self.artifact_store.save_analysis(run_state.task_id, analysis_data)
