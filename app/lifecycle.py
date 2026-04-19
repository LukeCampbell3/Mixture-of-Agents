"""Lifecycle manager for agent spawn, promotion, and pruning decisions."""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from app.schemas.registry import AgentRegistry, AgentSpec, LifecycleState
from app.schemas.task_frame import TaskFrame
from app.models.embeddings import EmbeddingGenerator
import numpy as np


class LifecycleDecision:
    """Record of a lifecycle decision."""
    
    def __init__(
        self,
        decision_type: str,
        agent_id: str,
        reason: str,
        scores: Dict[str, float],
        timestamp: Optional[str] = None
    ):
        self.decision_type = decision_type  # spawn, promote, demote, prune, no_spawn, no_promote
        self.agent_id = agent_id
        self.reason = reason
        self.scores = scores
        self.timestamp = timestamp or datetime.utcnow().isoformat()


class LifecycleManager:
    """Manage agent lifecycle: spawn, promote, demote, and prune."""
    
    def __init__(
        self,
        registry: AgentRegistry,
        embedding_generator: EmbeddingGenerator
    ):
        self.registry = registry
        self.embedding_generator = embedding_generator
        self.decision_history: List[LifecycleDecision] = []
        
        # Persistent lifecycle memory
        self.task_history: List[Dict[str, Any]] = []
        self.cluster_stats: Dict[str, Dict[str, Any]] = {}
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        self.spawned_agents: Dict[str, Dict[str, Any]] = {}
        
        # Thresholds (from spec)
        self.min_spawn_score = 0.6
        self.max_overlap_for_spawn = 0.7
        self.min_cluster_size_for_spawn = 5
        self.min_promotion_score = 0.7
        self.min_retention_score = 0.3
        
        # Weights for scoring (from spec)
        self.spawn_weights = {
            "recurring_failure_score": 0.15,  # Reduced from 0.3
            "task_cluster_density": 0.35,      # Increased from 0.2
            "uncertainty_persistence": 0.1,     # Reduced from 0.2
            "disagreement_score": 0.1,          # Reduced from 0.15
            "projected_future_usage": 0.30,     # Increased from 0.15
            "overlap_penalty": -0.3,
            "maintenance_cost": -0.1
        }
        
        self.retention_weights = {
            "long_run_quality_lift": 0.4,
            "unique_coverage_value": 0.3,
            "user_preference_weight": 0.2,
            "rare_but_high_value_bonus": 0.1,
            "maintenance_cost": -0.3,
            "redundancy_penalty": -0.3
        }
        
        self.warmth_weights = {
            "predicted_near_term_usage": 0.4,
            "recent_quality_lift": 0.3,
            "readiness_value": 0.2,
            "idle_cost": -0.05,
            "overlap_penalty": -0.05
        }
    
    def record_task_execution(
        self,
        task_frame: Dict[str, Any],
        routing_decision: Dict[str, Any],
        run_state: Dict[str, Any]
    ):
        """Record task execution for lifecycle analysis.
        
        Args:
            task_frame: Task frame data
            routing_decision: Routing decision data
            run_state: Run state data
        """
        # Extract key information
        task_type = task_frame.get("task_type", "unknown")
        task_text = task_frame.get("normalized_request", "")
        task_family = self._infer_task_family(task_type, task_text)
        success = run_state.get("final_state") == "success"
        active_agents = run_state.get("active_agents", [])
        
        # Record in task history
        task_record = {
            "task_id": run_state.get("task_id"),
            "task_type": task_type,
            "task_family": task_family,
            "success": success,
            "active_agents": active_agents,
            "timestamp": datetime.utcnow().isoformat(),
            "difficulty": task_frame.get("difficulty_estimate", 0.5)
        }
        self.task_history.append(task_record)
        
        # Keep only recent history (last 100 tasks)
        if len(self.task_history) > 100:
            self.task_history = self.task_history[-100:]
        
        # Update cluster stats
        if task_family not in self.cluster_stats:
            self.cluster_stats[task_family] = {
                "count": 0,
                "success_count": 0,
                "failure_count": 0,
                "recent_tasks": []
            }
        
        cluster = self.cluster_stats[task_family]
        cluster["count"] += 1
        if success:
            cluster["success_count"] += 1
        else:
            cluster["failure_count"] += 1
        cluster["recent_tasks"].append(task_record)
        
        # Keep only recent tasks per cluster
        if len(cluster["recent_tasks"]) > 20:
            cluster["recent_tasks"] = cluster["recent_tasks"][-20:]
        
        # Update agent performance
        for agent_id in active_agents:
            if agent_id not in self.agent_performance:
                self.agent_performance[agent_id] = {
                    "activation_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "task_families": set()
                }
            
            perf = self.agent_performance[agent_id]
            perf["activation_count"] += 1
            if success:
                perf["success_count"] += 1
            else:
                perf["failure_count"] += 1
            perf["task_families"].add(task_family)
    
    def _infer_task_family(self, task_type: str, task_text: str = "") -> str:
        """Infer task family from task type and task text.
        
        Uses both the classified task_type and the raw text to determine
        the most specific cluster. Text-based signals take priority when
        they indicate a specialist domain.
        """
        text_lower = task_text.lower()
        
        # Text-based classification first (more specific)
        if any(kw in text_lower for kw in ["api", "migrate", "migration", "endpoint", "rest", "graphql", "grpc", "soap"]):
            return "api_migration"
        if any(kw in text_lower for kw in ["security", "audit", "vulnerability", "injection", "authentication attack", "penetration"]):
            return "security_audit"
        
        # Fall back to task_type classification
        task_type_lower = task_type.lower()
        
        if any(kw in task_type_lower for kw in ["code", "coding", "implement", "debug"]):
            return "coding"
        elif any(kw in task_type_lower for kw in ["research", "find", "search"]):
            return "research"
        elif any(kw in task_type_lower for kw in ["verify", "check", "validate"]):
            return "verification"
        elif any(kw in task_type_lower for kw in ["security", "audit"]):
            return "security_audit"
        elif any(kw in task_type_lower for kw in ["api", "migration"]):
            return "api_migration"
        elif any(kw in task_type_lower for kw in ["planning", "reason"]):
            return "reasoning"
        else:
            return "general"
    
    def evaluate_spawn_need(
        self,
        task_history: List[Dict[str, Any]],
        failure_patterns: Dict[str, Any]
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Evaluate if a new agent should be spawned.
        
        Args:
            task_history: Recent task execution history (ignored, uses internal history)
            failure_patterns: Detected failure patterns (contains current_task)
        
        Returns:
            (should_spawn, spawn_spec) tuple
        """
        # Use internal task history instead of passed parameter
        if len(self.task_history) < self.min_cluster_size_for_spawn:
            # Not enough history to make spawn decision
            return False, None
        
        # Analyze current task to identify cluster
        current_task = failure_patterns.get("current_task", {})
        task_type = current_task.get("task_type", "unknown")
        task_text = current_task.get("normalized_request", "")
        task_family = self._infer_task_family(task_type, task_text)
        
        # Check if this cluster has recurring issues
        if task_family not in self.cluster_stats:
            return False, None
        
        cluster = self.cluster_stats[task_family]
        
        # Build failure patterns from cluster stats
        enriched_patterns = {
            "domain": task_family,
            "cluster_id": task_family,
            "recurrence_rate": cluster["count"] / len(self.task_history),
            "cluster_density": min(1.0, cluster["count"] / 5.0),  # Changed from 10.0 to 5.0
            "avg_uncertainty": 0.3,  # Baseline uncertainty
            "disagreement_rate": 0.2 if cluster["failure_count"] > 2 else 0.1,
            "projected_usage": min(1.0, cluster["count"] / len(self.task_history) * 1.5),  # Boost projected usage
            "spawn_reason": f"Recurring {task_family} tasks detected ({cluster['count']} occurrences)",
            "required_tools": self._infer_required_tools(task_family)
        }
        
        # Calculate spawn score
        spawn_score, components = self._calculate_spawn_score(self.task_history, enriched_patterns)
        
        # Add spawn score to spec
        enriched_patterns["spawn_score"] = spawn_score
        
        # Check if spawn is warranted
        if spawn_score < self.min_spawn_score:
            decision = LifecycleDecision(
                decision_type="no_spawn",
                agent_id="proposed",
                reason=f"Spawn score {spawn_score:.3f} below threshold {self.min_spawn_score}",
                scores=components
            )
            self.decision_history.append(decision)
            return False, None
        
        # Check overlap with existing agents
        max_overlap = self._calculate_max_overlap(enriched_patterns)
        if max_overlap > self.max_overlap_for_spawn:
            decision = LifecycleDecision(
                decision_type="no_spawn",
                agent_id="proposed",
                reason=f"Overlap {max_overlap:.3f} exceeds threshold {self.max_overlap_for_spawn}",
                scores=components
            )
            self.decision_history.append(decision)
            return False, None
        
        # Generate spawn spec
        spawn_spec = self._generate_spawn_spec(enriched_patterns, components)
        spawn_spec["spawn_score"] = spawn_score
        
        decision = LifecycleDecision(
            decision_type="spawn_recommended",
            agent_id=spawn_spec["agent_id"],
            reason=f"Spawn score {spawn_score:.3f} meets threshold, overlap {max_overlap:.3f} acceptable",
            scores=components
        )
        self.decision_history.append(decision)
        
        return True, spawn_spec
    
    def _infer_required_tools(self, task_family: str) -> List[str]:
        """Infer required tools based on task family."""
        tool_map = {
            "coding": ["repo_tool", "test_runner", "code_analyzer"],
            "research": ["web_tool", "citation_checker", "doc_retriever"],
            "verification": ["test_runner", "citation_checker", "validator"],
            "security_audit": ["security_scanner", "code_analyzer", "test_runner"],
            "api_migration": ["api_tool", "code_analyzer", "test_runner"],
            "general": ["repo_tool", "web_tool"]
        }
        return tool_map.get(task_family, ["repo_tool", "web_tool"])
    
    def evaluate_promotion(self, agent_id: str, performance_data: Dict[str, Any]) -> bool:
        """Evaluate if a probationary agent should be promoted.
        
        Args:
            agent_id: Agent to evaluate
            performance_data: Performance metrics
        
        Returns:
            True if agent should be promoted
        """
        agent = self.registry.get_agent(agent_id)
        if not agent or agent.lifecycle_state != LifecycleState.PROBATIONARY:
            return False
        
        # Calculate promotion score
        promotion_score = self._calculate_promotion_score(agent, performance_data)
        
        should_promote = promotion_score >= self.min_promotion_score
        
        decision = LifecycleDecision(
            decision_type="promote" if should_promote else "no_promote",
            agent_id=agent_id,
            reason=f"Promotion score {promotion_score:.3f} {'meets' if should_promote else 'below'} threshold {self.min_promotion_score}",
            scores={"promotion_score": promotion_score, **performance_data}
        )
        self.decision_history.append(decision)
        
        if should_promote:
            # Promote to warm state
            agent.lifecycle_state = LifecycleState.WARM
            agent.updated_at = datetime.utcnow().isoformat()
            self.registry.add_agent(agent)
        
        return should_promote
    
    def evaluate_demotion(self, agent_id: str, performance_data: Dict[str, Any]) -> Optional[LifecycleState]:
        """Evaluate if an agent should be demoted.
        
        Args:
            agent_id: Agent to evaluate
            performance_data: Performance metrics
        
        Returns:
            New lifecycle state if demotion warranted, None otherwise
        """
        agent = self.registry.get_agent(agent_id)
        if not agent:
            return None
        
        # Calculate retention score
        retention_score = self._calculate_retention_score(agent, performance_data)
        
        # Determine new state based on retention score
        current_state = agent.lifecycle_state
        new_state = None
        
        if retention_score < 0.2:
            new_state = LifecycleState.ARCHIVED
            reason = f"Retention score {retention_score:.3f} critically low"
        elif retention_score < 0.3:
            if current_state in [LifecycleState.HOT, LifecycleState.WARM]:
                new_state = LifecycleState.COLD
                reason = f"Retention score {retention_score:.3f} below threshold"
            elif current_state == LifecycleState.COLD:
                new_state = LifecycleState.DORMANT
                reason = f"Retention score {retention_score:.3f} remains low"
        elif retention_score < 0.5:
            if current_state == LifecycleState.HOT:
                new_state = LifecycleState.WARM
                reason = f"Retention score {retention_score:.3f} indicates reduced utility"
        
        if new_state and new_state != current_state:
            decision = LifecycleDecision(
                decision_type="demote",
                agent_id=agent_id,
                reason=reason,
                scores={"retention_score": retention_score, **performance_data}
            )
            self.decision_history.append(decision)
            
            agent.lifecycle_state = new_state
            agent.updated_at = datetime.utcnow().isoformat()
            self.registry.add_agent(agent)
            
            return new_state
        
        return None
    
    def update_warmth(self, agent_id: str, usage_data: Dict[str, Any]) -> Optional[LifecycleState]:
        """Update agent warmth based on usage patterns.
        
        Args:
            agent_id: Agent to update
            usage_data: Recent usage statistics
        
        Returns:
            New lifecycle state if changed, None otherwise
        """
        agent = self.registry.get_agent(agent_id)
        if not agent or agent.lifecycle_state in [LifecycleState.ARCHIVED, LifecycleState.DEPRECATED]:
            return None
        
        # Calculate warmth score
        warmth_score = self._calculate_warmth_score(agent, usage_data)
        
        # Determine appropriate warmth state
        current_state = agent.lifecycle_state
        new_state = None
        
        if warmth_score > 0.7:
            if current_state != LifecycleState.HOT:
                new_state = LifecycleState.HOT
        elif warmth_score > 0.5:
            if current_state not in [LifecycleState.HOT, LifecycleState.WARM]:
                new_state = LifecycleState.WARM
        elif warmth_score > 0.3:
            if current_state in [LifecycleState.HOT, LifecycleState.DORMANT]:
                new_state = LifecycleState.WARM
        else:
            if current_state in [LifecycleState.HOT, LifecycleState.WARM]:
                new_state = LifecycleState.COLD
        
        if new_state and new_state != current_state:
            decision = LifecycleDecision(
                decision_type="warmth_update",
                agent_id=agent_id,
                reason=f"Warmth score {warmth_score:.3f} indicates {new_state.value} state",
                scores={"warmth_score": warmth_score, **usage_data}
            )
            self.decision_history.append(decision)
            
            agent.lifecycle_state = new_state
            agent.updated_at = datetime.utcnow().isoformat()
            self.registry.add_agent(agent)
            
            return new_state
        
        return None
    
    def evaluate_promotions(self) -> Dict[str, Any]:
        """Evaluate all probationary agents for promotion.
        
        Returns:
            Dictionary with promoted agents and events
        """
        promoted = []
        events = []
        
        # Find all probationary agents - iterate over values, not keys
        for agent in self.registry.agents.values():
            if not isinstance(agent, AgentSpec):
                continue
            
            if agent.lifecycle_state == LifecycleState.PROBATIONARY:
                # Get performance data
                perf = self.agent_performance.get(agent.agent_id, {})
                
                if perf.get("activation_count", 0) < 3:
                    # Not enough data yet
                    continue
                
                performance_data = {
                    "quality_lift": perf.get("success_count", 0) / max(1, perf.get("activation_count", 1)),
                    "success_rate": perf.get("success_count", 0) / max(1, perf.get("activation_count", 1)),
                    "relative_cost": 0.5,  # Would be computed from actual usage
                    "overlap_score": 0.3,  # Would be computed from similarity
                }
                
                # Debug: Calculate expected promotion score
                expected_score = (
                    0.4 * performance_data["quality_lift"] +
                    0.3 * performance_data["success_rate"] +
                    0.2 * (1.0 - performance_data["relative_cost"]) +
                    0.1 * (1.0 - performance_data["overlap_score"])
                )
                
                # Evaluate promotion
                if self.evaluate_promotion(agent.agent_id, performance_data):
                    promoted.append(agent.agent_id)
                    events.append({
                        "event_type": "promotion",
                        "agent_id": agent.agent_id,
                        "reason": f"Promoted after {perf.get('activation_count', 0)} successful activations (score: {expected_score:.3f})",
                        "timestamp": datetime.utcnow().isoformat()
                    })
        
        return {
            "promoted": promoted,
            "events": events
        }
    
    def evaluate_pruning(self) -> Dict[str, Any]:
        """Evaluate all agents for pruning/demotion.
        
        Returns:
            Dictionary with pruned agents and events
        """
        pruned = []
        events = []
        
        # Only prune if we have enough history
        if len(self.task_history) < 10:
            return {"pruned": [], "events": []}
        
        # Evaluate each agent - iterate over values, not keys
        for agent in self.registry.agents.values():
            if not isinstance(agent, AgentSpec):
                continue
            
            # Skip base agents
            if agent.agent_id in ["code_primary", "web_research", "critic_verifier"]:
                continue
            
            # Grace period: skip agents spawned recently (fewer than 10 tasks ago)
            if agent.agent_id in self.spawned_agents:
                spawn_task_count = self.spawned_agents[agent.agent_id].get("spawn_task_count", 0)
                tasks_since_spawn = len(self.task_history) - spawn_task_count
                if tasks_since_spawn < 10:
                    continue  # Too early to prune
            
            # Get performance data
            perf = self.agent_performance.get(agent.agent_id, {})
            
            # Calculate usage rate
            activation_count = perf.get("activation_count", 0)
            usage_rate = activation_count / len(self.task_history)
            
            # Calculate success rate
            success_rate = 0.0
            if activation_count > 0:
                success_rate = perf.get("success_count", 0) / activation_count
            
            performance_data = {
                "quality_lift": success_rate * 0.5,  # Simplified
                "unique_coverage": 0.3,  # Would be computed from overlap
                "user_preference": 0.5,  # Would come from user feedback
                "high_value_bonus": 0.0,
                "maintenance_cost": 0.1,
                "redundancy": 0.3 if usage_rate < 0.05 else 0.0
            }
            
            # Check for demotion/pruning
            new_state = self.evaluate_demotion(agent.agent_id, performance_data)
            
            if new_state == LifecycleState.ARCHIVED:
                pruned.append(agent.agent_id)
                events.append({
                    "event_type": "prune",
                    "agent_id": agent.agent_id,
                    "reason": f"Low usage ({usage_rate:.1%}) and performance ({success_rate:.1%})",
                    "timestamp": datetime.utcnow().isoformat()
                })
            elif new_state:
                events.append({
                    "event_type": "demote",
                    "agent_id": agent.agent_id,
                    "reason": f"Demoted to {new_state.value}",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return {
            "pruned": pruned,
            "events": events
        }
    
    def _calculate_spawn_score(
        self,
        task_history: List[Dict[str, Any]],
        failure_patterns: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate spawn score based on multiple factors."""
        components = {
            "recurring_failure_score": failure_patterns.get("recurrence_rate", 0.0),
            "task_cluster_density": failure_patterns.get("cluster_density", 0.0),
            "uncertainty_persistence": failure_patterns.get("avg_uncertainty", 0.0),
            "disagreement_score": failure_patterns.get("disagreement_rate", 0.0),
            "projected_future_usage": failure_patterns.get("projected_usage", 0.0),
            "overlap_penalty": 0.0,  # Calculated separately
            "maintenance_cost": 0.1  # Estimated
        }
        
        # Calculate weighted score
        score = sum(
            self.spawn_weights[key] * value
            for key, value in components.items()
        )
        
        return max(0.0, min(score, 1.0)), components
    
    def _calculate_max_overlap(self, failure_patterns: Dict[str, Any]) -> float:
        """Calculate maximum overlap with existing agents."""
        # Simplified: would use embeddings and behavioral similarity
        proposed_domain = failure_patterns.get("domain", "")
        
        max_overlap = 0.0
        for agent in self.registry.get_active_agents():
            # Domain similarity (simplified)
            domain_match = 1.0 if agent.domain == proposed_domain else 0.3
            max_overlap = max(max_overlap, domain_match)
        
        return max_overlap
    
    def _generate_spawn_spec(
        self,
        failure_patterns: Dict[str, Any],
        spawn_components: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate specification for new agent."""
        domain = failure_patterns.get("domain", "specialized")
        cluster_id = failure_patterns.get("cluster_id", "unknown")
        
        return {
            "agent_id": f"spawned_{domain}_{cluster_id}",
            "name": f"Specialized {domain.title()} Agent",
            "description": f"Agent spawned to handle {domain} tasks in cluster {cluster_id}",
            "domain": domain,
            "lifecycle_state": LifecycleState.PROBATIONARY,
            "tools": failure_patterns.get("required_tools", []),
            "tags": [domain, "spawned", f"cluster_{cluster_id}"],
            "target_cluster": cluster_id,
            "expected_activation_rate": spawn_components.get("projected_future_usage", 0.1),
            "spawn_reason": failure_patterns.get("spawn_reason", "Recurring failures detected")
        }
    
    def _calculate_promotion_score(
        self,
        agent: AgentSpec,
        performance_data: Dict[str, Any]
    ) -> float:
        """Calculate promotion score for probationary agent."""
        # Factors: quality lift, cost efficiency, low overlap
        quality_lift = performance_data.get("quality_lift", 0.0)
        cost_efficiency = 1.0 - performance_data.get("relative_cost", 0.5)
        low_overlap = 1.0 - performance_data.get("overlap_score", 0.5)
        activation_success = performance_data.get("success_rate", 0.0)
        
        score = (
            0.4 * quality_lift +
            0.3 * activation_success +
            0.2 * cost_efficiency +
            0.1 * low_overlap
        )
        
        return max(0.0, min(score, 1.0))
    
    def _calculate_retention_score(
        self,
        agent: AgentSpec,
        performance_data: Dict[str, Any]
    ) -> float:
        """Calculate retention score for agent."""
        components = {
            "long_run_quality_lift": performance_data.get("quality_lift", 0.0),
            "unique_coverage_value": performance_data.get("unique_coverage", 0.0),
            "user_preference_weight": performance_data.get("user_preference", 0.0),
            "rare_but_high_value_bonus": performance_data.get("high_value_bonus", 0.0),
            "maintenance_cost": performance_data.get("maintenance_cost", 0.1),
            "redundancy_penalty": performance_data.get("redundancy", 0.0)
        }
        
        score = sum(
            self.retention_weights[key] * value
            for key, value in components.items()
        )
        
        return max(0.0, min(score, 1.0))
    
    def _calculate_warmth_score(
        self,
        agent: AgentSpec,
        usage_data: Dict[str, Any]
    ) -> float:
        """Calculate warmth score based on usage patterns."""
        components = {
            "predicted_near_term_usage": usage_data.get("predicted_usage", 0.0),
            "recent_quality_lift": usage_data.get("recent_quality", 0.0),
            "readiness_value": usage_data.get("readiness", 0.5),
            "idle_cost": usage_data.get("idle_cost", 0.1),
            "overlap_penalty": usage_data.get("overlap", 0.0)
        }
        
        score = sum(
            self.warmth_weights[key] * value
            for key, value in components.items()
        )
        
        return max(0.0, min(score, 1.0))
    
    def get_lifecycle_summary(self) -> Dict[str, Any]:
        """Get summary of lifecycle decisions."""
        return {
            "total_decisions": len(self.decision_history),
            "by_type": self._count_decisions_by_type(),
            "recent_decisions": [
                {
                    "type": d.decision_type,
                    "agent_id": d.agent_id,
                    "reason": d.reason,
                    "timestamp": d.timestamp
                }
                for d in self.decision_history[-10:]
            ]
        }
    
    def _count_decisions_by_type(self) -> Dict[str, int]:
        """Count decisions by type."""
        counts = {}
        for decision in self.decision_history:
            decision_type = decision.decision_type
            counts[decision_type] = counts.get(decision_type, 0) + 1
        return counts
