"""Main orchestrator for the agentic network with parallel execution and skill packs."""

from typing import Dict, Any, Optional, List
from app.router import Router
from app.budget_controller import BudgetController
from app.validator import Validator
from app.schemas.registry import AgentRegistry, AgentSpec, LifecycleState
from app.schemas.run_state import RunState, ToolCall
from app.schemas.validation import ValidationCheck, ValidationState
from app.models.llm_client import LLMClient, create_llm_client
from app.models.embeddings import EmbeddingGenerator
from app.models.uncertainty import UncertaintyEstimator
from app.agents.code_primary import CodePrimaryAgent
from app.agents.web_research import WebResearchAgent
from app.agents.critic_verifier import CriticVerifierAgent
from app.tools.filesystem import FilesystemExecutor, parse_tool_calls, FileOperation, OperationResult
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


# ANSI color codes for terminal output
class Color:
    """Simple ANSI color codes for terminal output."""
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    
    @classmethod
    def red(cls, text: str) -> str:
        return f"{cls.RED}{text}{cls.RESET}"
    
    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls.GREEN}{text}{cls.RESET}"
    
    @classmethod
    def yellow(cls, text: str) -> str:
        return f"{cls.YELLOW}{text}{cls.RESET}"
    
    @classmethod
    def blue(cls, text: str) -> str:
        return f"{cls.BLUE}{text}{cls.RESET}"
    
    @classmethod
    def dim(cls, text: str) -> str:
        return f"{cls.DIM}{text}{cls.RESET}"
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
        max_parallel_agents: int = 3,
        max_tokens: int = 2000,
        auto_approve_file_ops: bool = True
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
            max_tokens:          Max tokens per agent call (default: 2000)
            auto_approve_file_ops: Automatically approve file operations without user input (default: True)
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
        self.max_tokens = max_tokens  # per-agent token budget
        
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
        
        # File operation auto-approval
        self.auto_approve_file_ops = auto_approve_file_ops
    
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
    
    def run_task(self, user_request: str, workspace_root: str = ".",
                 conversation_history: str = "") -> RunState:
        """Execute a task through the agent network.

        Args:
            user_request:          The user's natural-language request.
            workspace_root:        Absolute path to the workspace root.
            conversation_history:  Pre-formatted history block to inject as
                                   shared context for all agents.
        """
        self.workspace_root = workspace_root
        self._conversation_history = conversation_history
        
        # Frame the task FIRST to get task context for spawning
        task_frame = self.router.frame_task(user_request)
        task_family = self.router._get_task_family(task_frame.task_type)
        
        # Store task frame for use in spawning (needed for proper context)
        self._current_task_frame = task_frame

        # Initialize budget controller — honour device profile's agent cap
        budget_controller = BudgetController(
            mode=self.budget_mode,
            max_agents_override=self.max_parallel_agents
        )
        budget_controller.start_execution()

        # ── On-demand spawn: detect gap before routing ────────────────────
        # Pass the full task frame with conversation history for proper context
        self._maybe_spawn_specialist(task_frame, conversation_history)

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
        
        # Route to agents — lazy chain: start with primary only
        routing_decision = self.router.route_primary(task_frame)

        # Execute via lazy chain (primary first, sub-agents only if needed)
        final_answer = self._execute_lazy_chain(
            task_frame,
            routing_decision,
            budget_controller,
            run_state,
        )
        
        # Validate output
        agent_outputs = {agent_id: {"output": final_answer} for agent_id in run_state.active_agents}
        validation_report = self.validator.validate_output(
            task_frame,
            agent_outputs,
            final_answer
        )
        self._merge_build_validation(validation_report, run_state)

        # Update run state
        run_state.final_answer = final_answer

        # Execute file operations (with approval if enabled)
        pending_ops = list(getattr(self, "_pending_tool_calls", []))
        if pending_ops:
            execution_results = self._execute_file_operations(
                pending_ops, 
                workspace_root,
                auto_approve=self.auto_approve_file_ops
            )
            for result in execution_results:
                if result.success and result.op.path not in run_state.final_files:
                    run_state.final_files.append(result.op.path)
            run_state.pending_tool_calls = []
        else:
            run_state.pending_tool_calls = []
        
        self._pending_tool_calls = []  # reset for next run
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
    
    def _execute_lazy_chain(
        self,
        task_frame,
        routing_decision,
        budget_controller,
        run_state,
    ) -> str:
        """Lazy-chain execution: run the primary agent, then escalate only if needed.

        Cost model:
        - Simple tasks: 1 agent, 1 LLM call.
        - Uncertain/complex tasks: primary + 1-2 focused sub-agents.
        - Sub-agents receive a narrow sub-task (not the full prompt) so their
          token budget is much smaller than a full parallel fan-out.

        The primary agent's output is inspected for escalation signals before
        any additional agents are warmed up.
        """
        history = getattr(self, "_conversation_history", "")
        workspace = getattr(self, "workspace_root", ".")

        # ── Step 0: Pre-fetch knowledge for grounded responses ───────────────
        from app.agents.knowledge_enricher import KnowledgeEnricher
        enricher = KnowledgeEnricher()
        enrichment = enricher.enrich(task_frame.normalized_request, verbose=True)
        knowledge_block = enrichment.as_context_block()
        if enrichment.has_content:
            print(f"  [knowledge] {len([s for s in enrichment.sources if s.ok])} source(s) ready")

        # Language preference (set by CLI before calling run_task)
        lang_pref = getattr(self, "_language_preference", "")

        # ── Step 1: Run primary agent ────────────────────────────────────────
        primary_id = routing_decision.selected_agents[0]
        budget_controller.activate_agent()
        run_state.active_agents.append(primary_id)

        primary_spec = self.registry.get_agent(primary_id)
        primary_agent = self._create_agent_instance(primary_spec, history)

        shared_ctx = (
            f"{history}\n\nTask: {task_frame.normalized_request}\n\n"
            if history else
            f"Task: {task_frame.normalized_request}\n\n"
        )

        primary_output = primary_agent.execute({
            "task_frame": task_frame,
            "shared_context": shared_ctx,
            "constraints": task_frame.hard_constraints,
            "available_tools": primary_spec.tools,
            "iteration": 1,
            "max_tokens": self.max_tokens,
            "workspace_root": workspace,
            "knowledge_block": knowledge_block,
            "language_preference": lang_pref,
        })
        budget_controller.deactivate_agent()
        self._collect_tool_calls({primary_id: primary_output})

        primary_text = primary_output.get("output", "")
        print(f"  [chain] Primary: {primary_id}")

        # ── Step 1b: Run build-test-fix loop for coding tasks ────────────────
        task_type_str = (
            task_frame.task_type
            if isinstance(task_frame.task_type, str)
            else task_frame.task_type.value
        )
        is_coding = "coding" in task_type_str or "hybrid" in task_type_str
        written_ops = primary_output.get("tool_calls", [])

        if is_coding and written_ops:
            from app.tools.codebase_builder import CodebaseBuilder, BuildConfig
            build_cfg = BuildConfig(
                workspace_root=workspace,
                max_iterations=5,
                tokens_per_iteration=self.max_tokens,
                run_entry_points=True,
                run_tests=True,
                test_scope="workspace" if self.budget_mode == "codebase" else "written",
                auto_generate_tests=True,
                verbose=True,
            )
            builder = CodebaseBuilder(
                config=build_cfg,
                agent_fn=lambda prompt, max_tok: self.llm_client.generate(
                    prompt, max_tokens=max_tok, temperature=0.3
                ),
            )
            session = builder.build(
                task_text=task_frame.normalized_request,
                initial_response=primary_output.get("raw_response", primary_text),
                existing_tool_calls=written_ops,
                existing_tool_results=primary_output.get("tool_results", []),
            )
            run_state.build_report = self._build_report_payload(session)
            if self.budget_mode == "codebase":
                primary_text = primary_text + "\n\n" + session.summary()
            else:
                verification_note = self._format_build_verification(session)
                if verification_note:
                    primary_text = primary_text.rstrip() + "\n\n" + verification_note
            run_state.final_files = session.final_files if hasattr(run_state, "final_files") else []
            self._pending_tool_calls = []

        # ── Step 2: Check if escalation is needed ────────────────────────────
        if not budget_controller.can_activate_agent():
            return primary_text

        should_escalate, escalate_to = self.router.needs_escalation(
            primary_text, task_frame
        )

        if not should_escalate:
            print(f"  [chain] No escalation needed — done in 1 agent")
            run_state.suppressed_agents = routing_decision.suppressed_agents
            return primary_text

        # ── Step 3: Run sub-agents with focused sub-tasks ────────────────────
        sub_outputs: List[str] = [primary_text]
        max_sub = min(len(escalate_to), budget_controller.get_remaining_agents())

        for sub_id in escalate_to[:max_sub]:
            if not budget_controller.can_activate_agent():
                break

            # Ensure the sub-agent exists (spawn if needed)
            sub_spec = self.registry.get_agent(sub_id)
            if sub_spec is None:
                domain = sub_id.replace("specialist_", "")
                spec, _ = self.agent_factory.spawn_for_task(
                    task_text=task_frame.normalized_request,
                    domain=domain,
                    agent_id=sub_id,
                    conversation_history=history,
                )
                self.registry.add_agent(spec)
                self.registry_store.save_registry(self.registry)
                sub_spec = spec

            budget_controller.activate_agent()
            run_state.active_agents.append(sub_id)

            sub_agent = self._create_agent_instance(sub_spec, history)

            # Focused sub-task — much smaller prompt than the full original
            sub_task_text = self.router.build_sub_task(primary_text, task_frame, sub_id)

            # Minimal TaskFrame for the sub-task
            from app.schemas.task_frame import TaskFrame as TF
            import uuid as _uuid
            sub_frame = TF(
                task_id=str(_uuid.uuid4()),
                normalized_request=sub_task_text,
                task_type=task_frame.task_type,
                hard_constraints=[],
                likely_tools=sub_spec.tools,
                difficulty_estimate=0.3,   # sub-tasks are narrower
                initial_uncertainty=0.3,
                novelty_score=0.3,
                freshness_requirement=task_frame.freshness_requirement,
            )

            sub_output = sub_agent.execute({
                "task_frame": sub_frame,
                "shared_context": f"Primary response:\n{primary_text}",
                "constraints": [],
                "available_tools": sub_spec.tools,
                "iteration": 1,
                "max_tokens": self.max_tokens // 2,   # sub-agents get half the budget
                "workspace_root": workspace,
                "knowledge_block": knowledge_block,
                "language_preference": lang_pref,
            })
            budget_controller.deactivate_agent()
            self._collect_tool_calls({sub_id: sub_output})

            sub_text = sub_output.get("output", "")
            sub_outputs.append(sub_text)
            print(f"  [chain] Sub-agent: {sub_id}")

        run_state.suppressed_agents = routing_decision.suppressed_agents

        # ── Step 4: Merge — primary answer + sub-agent additions ─────────────
        if len(sub_outputs) == 1:
            return sub_outputs[0]

        # Lightweight merge: append non-redundant sub-agent content
        merged = primary_text
        for extra in sub_outputs[1:]:
            if extra and extra.strip() and extra.strip() != primary_text.strip():
                merged += f"\n\n---\n{extra.strip()}"
        return merged

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
        lead_agent = self._create_agent_instance(lead_agent_spec, getattr(self, "_conversation_history", ""))
        
        task_context = {
            "task_frame": task_frame,
            "constraints": task_frame.hard_constraints,
            "available_tools": lead_agent_spec.tools,
            "max_tokens": self.max_tokens,
            "shared_context": getattr(self, "_conversation_history", ""),
            "workspace_root": getattr(self, "workspace_root", "."),
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
            agent = self._create_agent_instance(agent_spec, getattr(self, "_conversation_history", ""))
            
            # Execute with role-specific instructions
            role_context = {
                "task_frame": task_frame,
                "lead_output": lead_output["output"],
                "role": role.value,
                "available_tools": agent_spec.tools,
                "max_tokens": self.max_tokens,
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
        history = getattr(self, "_conversation_history", "")
        shared_context_text = (
            f"{history}\n\nTask: {task_frame.normalized_request}\n\n"
            if history else
            f"Task: {task_frame.normalized_request}\n\n"
        )
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
                
                agent = self._create_agent_instance(agent_spec, getattr(self, "_conversation_history", ""))
                
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
                    "iteration": 1,
                    "max_tokens": self.max_tokens,
                    "workspace_root": getattr(self, "workspace_root", "."),
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
            self._collect_tool_calls(agent_outputs)
        
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
                
                agent = self._create_agent_instance(agent_spec, getattr(self, "_conversation_history", ""))
                
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
                    "skill_packs": agent_skill_packs,
                    "max_tokens": self.max_tokens,
                    "workspace_root": getattr(self, "workspace_root", "."),
                }
                
                output = agent.execute(task_context)
                agent_outputs[agent_id] = output
                self._collect_tool_calls({agent_id: output})
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
                
                agent = self._create_agent_instance(agent_spec, getattr(self, "_conversation_history", ""))
                
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
                    "refinement_mode": True,
                    "max_tokens": self.max_tokens,
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
    
    def _collect_tool_calls(self, agent_outputs: dict) -> None:
        """Accumulate FileOperation objects from agent outputs that were NOT
        already executed inline during streaming."""
        if not hasattr(self, "_pending_tool_calls"):
            self._pending_tool_calls = []
        for output in agent_outputs.values():
            # tool_results present means streaming already executed these ops
            if output.get("tool_results"):
                continue
            calls = output.get("tool_calls", [])
            self._pending_tool_calls.extend(calls)

    def _build_report_payload(self, session) -> Dict[str, Any]:
        """Serialize a build session into a lightweight run-state payload."""
        iterations = []
        for record in session.iterations:
            iterations.append({
                "iteration": record.iteration,
                "files_written": record.files_written,
                "syntax_errors": len(record.syntax_errors),
                "execution_failures": sum(
                    1 for result in record.execution_results if not result.success
                ),
                "command_failures": sum(
                    1 for result in record.command_results if not result.success
                ),
                "tests": None if record.test_result is None else {
                    "framework": record.test_result.framework,
                    "passed": record.test_result.passed,
                    "failed": record.test_result.failed,
                    "errors": record.test_result.errors,
                    "success": record.test_result.success,
                },
                "passed": record.passed,
            })

        return {
            "success": session.success,
            "summary": session.summary(),
            "final_files": session.final_files,
            "iterations": iterations,
        }

    def _format_build_verification(self, session) -> str:
        """Return a short user-facing verification note for a build session."""
        if not session.iterations:
            return ""

        last = session.iterations[-1]
        parts = [
            "Verification: passed" if session.success else "Verification: failed",
        ]
        if last.syntax_errors:
            parts.append(f"{len(last.syntax_errors)} syntax error(s)")
        failed_exec = sum(1 for result in last.execution_results if not result.success)
        if failed_exec:
            parts.append(f"{failed_exec} runtime check(s) failed")
        failed_commands = sum(1 for result in last.command_results if not result.success)
        if failed_commands:
            parts.append(f"{failed_commands} build/test command(s) failed")
        if last.test_result is not None:
            if last.test_result.framework == "none":
                parts.append("no tests found")
            elif last.test_result.success:
                parts.append(
                    f"{last.test_result.framework} {last.test_result.passed} passed"
                )
            else:
                parts.append(
                    f"{last.test_result.framework} "
                    f"{last.test_result.failed} failed / {last.test_result.errors} errors"
                )
        return " ".join(parts) + "."

    def _merge_build_validation(self, validation_report, run_state) -> None:
        """Fold build-loop validation into the final validation report."""
        build_report = getattr(run_state, "build_report", None)
        if not build_report:
            return

        passed = bool(build_report.get("success"))
        summary = build_report.get("summary", "Build validation completed")
        validation_report.checks.append(
            ValidationCheck(
                check_name="build_validation",
                passed=passed,
                severity="info" if passed else "error",
                message=summary,
            )
        )

        if not passed:
            validation_report.validation_state = ValidationState.VALIDATION_FAILURE
            validation_report.overall_passed = False

        passed_checks = sum(1 for check in validation_report.checks if check.passed)
        total_checks = len(validation_report.checks)
        state_value = (
            validation_report.validation_state
            if isinstance(validation_report.validation_state, str)
            else validation_report.validation_state.value
        )
        validation_report.summary = (
            f"Validation {state_value}: {passed_checks}/{total_checks} checks passed"
        )

    def _execute_file_operations(
        self,
        pending_ops: List[FileOperation],
        workspace_root: str,
        auto_approve: bool = False
    ) -> List[OperationResult]:
        """Execute file operations with optional approval flow.
        
        Args:
            pending_ops: List of file operations to execute
            workspace_root: Root directory for file operations
            auto_approve: If True, execute without user approval
            
        Returns:
            List of operation results
        """
        from app.tools.filesystem import FilesystemExecutor
        
        executor = FilesystemExecutor(workspace_root=workspace_root)
        
        if not pending_ops:
            return []
        
        print(Color.yellow(f"\n  {len(pending_ops)} file operation(s) proposed:\n"))
        
        approved_ops = []
        for i, op in enumerate(pending_ops, 1):
            print(Color.blue(f"  [{i}/{len(pending_ops)}] {op.tool.upper()}: {op.path}"))
            if op.description:
                print(Color.dim(f"  {op.description}"))

            # Show diff/preview
            preview = executor.preview(op)
            if preview and preview != "(no changes)":
                # Colour the diff lines
                for line in preview.splitlines():
                    if line.startswith("+") and not line.startswith("+++"):
                        print(Color.green(f"  {line}"))
                    elif line.startswith("-") and not line.startswith("---"):
                        print(Color.red(f"  {line}"))
                    else:
                        print(Color.dim(f"  {line}"))
            print()

            # Auto-approve or ask for approval
            if auto_approve:
                approved_ops.append(op)
                print(Color.green(f"  [AUTO-APPROVED]"))
            else:
                try:
                    choice = input(
                        Color.yellow("  Apply? [y]es / [n]o / [a]ll / [q]uit: ")
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = "q"

                if choice in ("a", "all"):
                    approved_ops.extend(pending_ops[i - 1:])
                    print(Color.green(f"  Approved all remaining {len(pending_ops)-i+1} operation(s)."))
                    break
                elif choice in ("y", "yes", ""):
                    approved_ops.append(op)
                elif choice in ("q", "quit"):
                    print(Color.yellow("  Aborted remaining operations."))
                    break
                else:
                    print(Color.dim("  Skipped."))

        # Execute approved operations
        if approved_ops:
            print()
            # Use batch mode for faster execution when auto-approving
            results = executor.execute_all(approved_ops, batch_mode=auto_approve)
            for res in results:
                if res.success:
                    print(Color.green(f"  ✓ {res.message}"))
                else:
                    print(Color.red(f"  ✗ {res.message}"))
            print()
            return results
        
        return []

    def _maybe_spawn_specialist(self, task_frame, conversation_history: str = "") -> None:
        """
        Spawn a specialist agent on-demand when the task has clear specialist
        signals but no matching specialist exists yet.

        Logic:
        1. Detect the specialist domain from the task text.
        2. If no specialist for that domain exists, spawn one immediately.
        3. If a specialist already exists, do nothing (it will be routed to).
        4. If no specialist domain is detected, fall back to base agents.
        
        The spawned agent receives the conversation history so it has prior context.
        """
        # Detect the specialist domain from the task text
        domain = self._detect_specialist_domain(task_frame.normalized_request)
        if domain is None:
            return  # No specialist signal — base agents are fine

        # Check if we already have a specialist for this domain
        agent_id = f"specialist_{domain}"
        if self.registry.get_agent(agent_id) is not None:
            return  # Already exists — router will select it

        # Create context-aware task text that includes conversation history
        full_task_context = task_frame.normalized_request
        if conversation_history:
            full_task_context = f"{conversation_history}\n\nCURRENT TASK:\n{task_frame.normalized_request}"

        # Spawn a new specialist
        print(f"\n  [spawn] Creating specialist: {domain}")
        spec, _instance = self.agent_factory.spawn_for_task(
            task_text=full_task_context,
            domain=domain,
            agent_id=agent_id,
            conversation_history=conversation_history,
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
                           "fine tuning", "neural network", "transformer model", "transformer node",
                           "switch transformer", "mixture of experts", "moe", "attention mechanism",
                           "self-attention", "multi-head attention", "feed forward network",
                           "activation function", "softmax", "layer norm", "batch norm",
                           "llm", "embedding", "inference", "gpu", "cuda", "huggingface",
                           "backpropagation", "gradient descent", "loss function", "epoch",
                           "bert", "gpt", "t5", "vit", "diffusion model", "autoencoder"],
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
    
    def _create_agent_instance(
        self,
        agent_spec: AgentSpec,
        conversation_history: str = "",
    ):
        """Create agent instance from spec.

        The three universal base agents use their hand-written classes.
        Every other agent (spawned specialists) is instantiated via
        AgentFactory, which uses the LLM to generate a focused system prompt.
        
        The conversation history is passed so spawned agents have prior context.
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
        return self.agent_factory.create_agent_instance(agent_spec, conversation_history)
    
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

Be detailed and production-oriented. If code or file changes were involved, explain what was changed, why it was done that way, notable edge cases, and any validation performed.

Final synthesized answer:"""
        
        try:
            synthesis_tokens = max(1200, min(self.max_tokens, 4000))
            final_answer = self.llm_client.generate(
                synthesis_prompt,
                max_tokens=synthesis_tokens,
                temperature=0.4,
            )
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
