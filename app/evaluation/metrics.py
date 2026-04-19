"""Comprehensive metrics for evaluating the agentic network."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict


@dataclass
class TaskMetrics:
    """Metrics for a single task execution."""
    
    task_id: str
    success: bool
    quality_score: float  # 0-1
    
    # Efficiency
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    energy_joules: Optional[float] = None
    
    # Sparsity
    active_agents: List[str] = field(default_factory=list)
    total_available_agents: int = 0
    
    # Router
    selected_agents: List[str] = field(default_factory=list)
    suppressed_agents: List[str] = field(default_factory=list)
    routing_scores: Dict[str, float] = field(default_factory=dict)
    
    # Quality
    unsupported_claims: int = 0
    contradictions: int = 0
    citation_count: int = 0
    constraint_violations: int = 0
    
    # Individual agent scores
    agent_scores: Dict[str, float] = field(default_factory=dict)
    best_individual_score: float = 0.0
    team_score: float = 0.0
    
    # Lifecycle
    spawned_agents: List[str] = field(default_factory=list)
    promoted_agents: List[str] = field(default_factory=list)
    pruned_agents: List[str] = field(default_factory=list)
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class MetricsCollector:
    """Collects and aggregates metrics across tasks."""
    
    def __init__(self):
        self.task_metrics: List[TaskMetrics] = []
        self.agent_utilization: Dict[str, int] = defaultdict(int)
        self.agent_quality: Dict[str, List[float]] = defaultdict(list)
        self.routing_decisions: List[Dict[str, Any]] = []
        
    def add_task_metrics(self, metrics: TaskMetrics) -> None:
        """Add metrics from a task execution."""
        self.task_metrics.append(metrics)
        
        # Update agent utilization
        for agent_id in metrics.active_agents:
            self.agent_utilization[agent_id] += 1
        
        # Update agent quality
        for agent_id, score in metrics.agent_scores.items():
            self.agent_quality[agent_id].append(score)
    
    # ========================================
    # A. OUTCOME QUALITY METRICS
    # ========================================
    
    def task_success_rate(self) -> float:
        """Primary outcome metric: task success rate."""
        if not self.task_metrics:
            return 0.0
        successful = sum(1 for m in self.task_metrics if m.success)
        return successful / len(self.task_metrics)
    
    def code_pass_rate(self) -> float:
        """Pass rate for coding tasks."""
        coding_tasks = [m for m in self.task_metrics if "coding" in m.task_id.lower()]
        if not coding_tasks:
            return 0.0
        passed = sum(1 for m in coding_tasks if m.success)
        return passed / len(coding_tasks)
    
    def citation_supported_correctness(self) -> float:
        """Percentage of claims with citation support."""
        research_tasks = [m for m in self.task_metrics if "research" in m.task_id.lower()]
        if not research_tasks:
            return 0.0
        
        total_claims = sum(m.citation_count + m.unsupported_claims for m in research_tasks)
        if total_claims == 0:
            return 1.0
        
        supported = sum(m.citation_count for m in research_tasks)
        return supported / total_claims
    
    def constraint_satisfaction_rate(self) -> float:
        """Percentage of tasks with no constraint violations."""
        if not self.task_metrics:
            return 0.0
        satisfied = sum(1 for m in self.task_metrics if m.constraint_violations == 0)
        return satisfied / len(self.task_metrics)
    
    def unsupported_claim_rate(self) -> float:
        """Rate of unsupported claims (lower is better)."""
        if not self.task_metrics:
            return 0.0
        total_unsupported = sum(m.unsupported_claims for m in self.task_metrics)
        return total_unsupported / len(self.task_metrics)
    
    def contradiction_rate(self) -> float:
        """Rate of contradictions (lower is better)."""
        if not self.task_metrics:
            return 0.0
        total_contradictions = sum(m.contradictions for m in self.task_metrics)
        return total_contradictions / len(self.task_metrics)
    
    # ========================================
    # B. EFFICIENCY AND SPARSITY METRICS
    # ========================================
    
    def quality_per_compute(self) -> float:
        """Flagship metric: quality per unit of compute."""
        if not self.task_metrics:
            return 0.0
        
        total_quality = sum(m.quality_score for m in self.task_metrics)
        total_tokens = sum(m.total_tokens for m in self.task_metrics)
        
        if total_tokens == 0:
            return 0.0
        
        # Normalize: quality per 1000 tokens
        return (total_quality / len(self.task_metrics)) / (total_tokens / len(self.task_metrics) / 1000)
    
    def tokens_per_task(self) -> float:
        """Average tokens per task."""
        if not self.task_metrics:
            return 0.0
        return sum(m.total_tokens for m in self.task_metrics) / len(self.task_metrics)
    
    def tokens_per_successful_task(self) -> float:
        """Tokens per successful task (key efficiency metric)."""
        successful = [m for m in self.task_metrics if m.success]
        if not successful:
            return 0.0
        return sum(m.total_tokens for m in successful) / len(successful)
    
    def latency_stats(self) -> Dict[str, float]:
        """Latency statistics (median, P95)."""
        if not self.task_metrics:
            return {"median": 0.0, "p95": 0.0, "mean": 0.0}
        
        latencies = [m.latency_seconds for m in self.task_metrics]
        return {
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "mean": float(np.mean(latencies))
        }
    
    def latency_per_successful_task(self) -> Dict[str, float]:
        """Latency for successful tasks only."""
        successful = [m for m in self.task_metrics if m.success]
        if not successful:
            return {"median": 0.0, "p95": 0.0}
        
        latencies = [m.latency_seconds for m in successful]
        return {
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95))
        }
    
    def energy_per_successful_task(self) -> Optional[float]:
        """Energy per successful task (if measured)."""
        successful = [m for m in self.task_metrics if m.success and m.energy_joules is not None]
        if not successful:
            return None
        return sum(m.energy_joules for m in successful) / len(successful)
    
    def average_active_agents_per_task(self) -> float:
        """Average number of active agents (sparsity proof)."""
        if not self.task_metrics:
            return 0.0
        return sum(len(m.active_agents) for m in self.task_metrics) / len(self.task_metrics)
    
    def budget_overrun_rate(self) -> float:
        """Rate of tasks that exceeded budget."""
        if not self.task_metrics:
            return 0.0
        
        # Check if any task has budget_exhausted flag in metadata
        overruns = 0
        for m in self.task_metrics:
            # Assume budget overrun if tokens exceed typical limits
            if m.total_tokens > 30000:  # Thorough budget limit
                overruns += 1
        
        return overruns / len(self.task_metrics)
    
    # ========================================
    # C. ROUTER QUALITY METRICS
    # ========================================
    
    def routing_regret(self) -> float:
        """Performance loss vs oracle router."""
        if not self.task_metrics:
            return 0.0
        
        total_regret = 0.0
        for m in self.task_metrics:
            # Oracle would pick agents with highest scores
            if m.agent_scores:
                oracle_score = max(m.agent_scores.values())
                actual_score = m.team_score
                total_regret += max(0, oracle_score - actual_score)
        
        return total_regret / len(self.task_metrics)
    
    def top_k_routing_hit_rate(self, k: int = 3) -> float:
        """How often router picks agents in top-k by score."""
        if not self.task_metrics:
            return 0.0
        
        hits = 0
        total = 0
        
        for m in self.task_metrics:
            if not m.agent_scores or not m.selected_agents:
                continue
            
            # Get top-k agents by score
            sorted_agents = sorted(m.agent_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_agents = set(agent for agent, _ in sorted_agents[:k])
            
            # Check overlap
            selected = set(m.selected_agents)
            hits += len(selected & top_k_agents)
            total += len(selected)
        
        return hits / total if total > 0 else 0.0
    
    def activation_precision(self) -> float:
        """Precision of agent activation."""
        # Activated agents that contributed positively
        if not self.task_metrics:
            return 0.0
        
        useful_activations = 0
        total_activations = 0
        
        for m in self.task_metrics:
            for agent_id in m.active_agents:
                total_activations += 1
                if agent_id in m.agent_scores and m.agent_scores[agent_id] > 0.5:
                    useful_activations += 1
        
        return useful_activations / total_activations if total_activations > 0 else 0.0
    
    # ========================================
    # D. SPECIALIZATION AND OVERLAP METRICS
    # ========================================
    
    def routing_entropy(self) -> float:
        """Entropy of routing distribution (lower = more specialized)."""
        if not self.agent_utilization:
            return 0.0
        
        total = sum(self.agent_utilization.values())
        if total == 0:
            return 0.0
        
        probs = [count / total for count in self.agent_utilization.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # Normalize by max entropy
        max_entropy = np.log2(len(self.agent_utilization))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def expert_utilization_balance(self) -> Dict[str, float]:
        """Utilization balance metrics."""
        if not self.agent_utilization:
            return {"gini": 0.0, "top1_share": 0.0, "concentration": 0.0}
        
        counts = sorted(self.agent_utilization.values())
        total = sum(counts)
        
        if total == 0:
            return {"gini": 0.0, "top1_share": 0.0, "concentration": 0.0}
        
        # Gini coefficient
        n = len(counts)
        cumsum = np.cumsum(counts)
        gini = (2 * sum((i + 1) * count for i, count in enumerate(counts))) / (n * total) - (n + 1) / n
        
        # Top-1 share
        top1_share = max(counts) / total
        
        # Concentration (HHI)
        shares = [count / total for count in counts]
        concentration = sum(s ** 2 for s in shares)
        
        return {
            "gini": float(gini),
            "top1_share": float(top1_share),
            "concentration": float(concentration)
        }
    
    def routing_consistency_within_cluster(self) -> float:
        """Consistency of routing for similar tasks."""
        if len(self.task_metrics) < 2:
            return 1.0
        
        # Group tasks by type (simple heuristic)
        task_groups = defaultdict(list)
        for m in self.task_metrics:
            task_type = m.task_id.split('_')[0] if '_' in m.task_id else 'general'
            task_groups[task_type].append(m.selected_agents)
        
        # Calculate consistency within each group
        consistencies = []
        for group_agents in task_groups.values():
            if len(group_agents) < 2:
                continue
            
            # Calculate pairwise overlap
            overlaps = []
            for i in range(len(group_agents)):
                for j in range(i + 1, len(group_agents)):
                    set_i = set(group_agents[i])
                    set_j = set(group_agents[j])
                    if set_i or set_j:
                        overlap = len(set_i & set_j) / len(set_i | set_j)
                        overlaps.append(overlap)
            
            if overlaps:
                consistencies.append(np.mean(overlaps))
        
        return float(np.mean(consistencies)) if consistencies else 1.0
    
    # ========================================
    # E. LIFECYCLE QUALITY METRICS
    # ========================================
    
    def spawn_success_rate(self) -> float:
        """Rate of spawned agents that get promoted."""
        spawned = set()
        promoted = set()
        
        for m in self.task_metrics:
            spawned.update(m.spawned_agents)
            promoted.update(m.promoted_agents)
        
        if not spawned:
            return 0.0
        
        successful_spawns = len(spawned & promoted)
        return successful_spawns / len(spawned)
    
    def probation_to_promotion_rate(self) -> float:
        """Rate of probationary agents that get promoted."""
        probationary = set()
        promoted = set()
        
        for m in self.task_metrics:
            # Track agents in probationary state
            for agent_id in m.spawned_agents:
                if agent_id not in m.promoted_agents:
                    probationary.add(agent_id)
            
            # Track promoted agents
            promoted.update(m.promoted_agents)
        
        if not probationary:
            return 0.0
        
        successful_probations = len(probationary & promoted)
        return successful_probations / len(probationary)
    
    def net_active_pool_growth_rate(self) -> float:
        """Net growth in active agent pool."""
        if len(self.task_metrics) < 2:
            return 0.0
        
        initial_pool = self.task_metrics[0].total_available_agents
        final_pool = self.task_metrics[-1].total_available_agents
        
        if initial_pool == 0:
            return 0.0
        
        return (final_pool - initial_pool) / initial_pool
    
    # ========================================
    # F. RELIABILITY AND SAFETY METRICS
    # ========================================
    
    def calibration_error(self) -> float:
        """Expected Calibration Error (ECE)."""
        if not self.task_metrics:
            return 0.0
        
        # Use quality scores as confidence proxy
        # Bin predictions and calculate calibration
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            bin_lower = bins[i]
            bin_upper = bins[i + 1]
            
            # Find tasks in this confidence bin
            in_bin = [
                m for m in self.task_metrics
                if bin_lower <= m.quality_score < bin_upper or (i == n_bins - 1 and m.quality_score == 1.0)
            ]
            
            if not in_bin:
                continue
            
            # Calculate accuracy (success rate) and average confidence
            accuracy = sum(1 for m in in_bin if m.success) / len(in_bin)
            confidence = np.mean([m.quality_score for m in in_bin])
            
            bin_accuracies.append(accuracy)
            bin_confidences.append(confidence)
            bin_counts.append(len(in_bin))
        
        if not bin_counts:
            return 0.0
        
        # Calculate ECE
        total_samples = sum(bin_counts)
        ece = sum(
            (count / total_samples) * abs(acc - conf)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
        )
        
        return float(ece)
    
    def run_to_run_variance(self) -> float:
        """Variance in quality across runs."""
        if len(self.task_metrics) < 2:
            return 0.0
        
        qualities = [m.quality_score for m in self.task_metrics]
        return float(np.var(qualities))
    
    # ========================================
    # G. COORDINATION QUALITY METRICS
    # ========================================
    
    def strong_synergy_gap(self) -> float:
        """Team score - best individual score (key metric!)."""
        if not self.task_metrics:
            return 0.0
        
        gaps = []
        for m in self.task_metrics:
            if m.best_individual_score > 0:
                gap = m.team_score - m.best_individual_score
                gaps.append(gap)
        
        return float(np.mean(gaps)) if gaps else 0.0
    
    def weak_synergy_gap(self) -> float:
        """Team score - average individual score."""
        if not self.task_metrics:
            return 0.0
        
        gaps = []
        for m in self.task_metrics:
            if m.agent_scores:
                avg_individual = np.mean(list(m.agent_scores.values()))
                gap = m.team_score - avg_individual
                gaps.append(gap)
        
        return float(np.mean(gaps)) if gaps else 0.0
    
    def contribution_concentration(self) -> float:
        """How concentrated are agent contributions (Gini coefficient)."""
        if not self.agent_quality:
            return 0.0
        
        # Calculate average quality per agent
        agent_avg_quality = {
            agent_id: np.mean(scores)
            for agent_id, scores in self.agent_quality.items()
        }
        
        if not agent_avg_quality:
            return 0.0
        
        # Calculate Gini coefficient
        qualities = sorted(agent_avg_quality.values())
        n = len(qualities)
        total = sum(qualities)
        
        if total == 0:
            return 0.0
        
        gini = (2 * sum((i + 1) * q for i, q in enumerate(qualities))) / (n * total) - (n + 1) / n
        
        return float(gini)
    
    # ========================================
    # H. PERSONALIZATION METRICS
    # ========================================
    
    def per_user_quality_lift(self, user_id: str) -> float:
        """Quality lift for specific user vs global."""
        # Filter tasks by user (would need user_id in TaskMetrics)
        # For now, return 0 as placeholder since we need schema update
        return 0.0
    
    # ========================================
    # DASHBOARD METRICS (TOP 9)
    # ========================================
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get the 9 key metrics for proof dashboard."""
        return {
            "task_success_rate": self.task_success_rate(),
            "quality_per_compute": self.quality_per_compute(),
            "tokens_per_successful_task": self.tokens_per_successful_task(),
            "p95_latency": self.latency_per_successful_task()["p95"],
            "energy_per_successful_task": self.energy_per_successful_task(),
            "avg_active_agents_per_task": self.average_active_agents_per_task(),
            "routing_regret": self.routing_regret(),
            "strong_synergy_gap": self.strong_synergy_gap(),
            "spawn_success_rate": self.spawn_success_rate()
        }
    
    def get_full_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        return {
            "summary": {
                "total_tasks": len(self.task_metrics),
                "successful_tasks": sum(1 for m in self.task_metrics if m.success),
                "total_agents": len(self.agent_utilization)
            },
            "outcome_quality": {
                "task_success_rate": self.task_success_rate(),
                "code_pass_rate": self.code_pass_rate(),
                "citation_supported_correctness": self.citation_supported_correctness(),
                "constraint_satisfaction_rate": self.constraint_satisfaction_rate(),
                "unsupported_claim_rate": self.unsupported_claim_rate(),
                "contradiction_rate": self.contradiction_rate()
            },
            "efficiency_sparsity": {
                "quality_per_compute": self.quality_per_compute(),
                "tokens_per_task": self.tokens_per_task(),
                "tokens_per_successful_task": self.tokens_per_successful_task(),
                "latency_stats": self.latency_stats(),
                "latency_per_successful_task": self.latency_per_successful_task(),
                "energy_per_successful_task": self.energy_per_successful_task(),
                "avg_active_agents_per_task": self.average_active_agents_per_task()
            },
            "router_quality": {
                "routing_regret": self.routing_regret(),
                "top_k_hit_rate": self.top_k_routing_hit_rate(),
                "activation_precision": self.activation_precision()
            },
            "specialization": {
                "routing_entropy": self.routing_entropy(),
                "utilization_balance": self.expert_utilization_balance()
            },
            "lifecycle": {
                "spawn_success_rate": self.spawn_success_rate(),
                "net_pool_growth": self.net_active_pool_growth_rate()
            },
            "reliability": {
                "run_to_run_variance": self.run_to_run_variance()
            },
            "coordination": {
                "strong_synergy_gap": self.strong_synergy_gap(),
                "weak_synergy_gap": self.weak_synergy_gap()
            },
            "dashboard": self.get_dashboard_metrics()
        }
