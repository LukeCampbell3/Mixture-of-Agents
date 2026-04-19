"""Cluster analyzer for detecting task patterns and spawn opportunities."""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from app.models.embeddings import EmbeddingGenerator
from app.schemas.task_frame import TaskFrame


class TaskCluster:
    """Represents a cluster of similar tasks."""
    
    def __init__(self, cluster_id: str, centroid: np.ndarray):
        self.cluster_id = cluster_id
        self.centroid = centroid
        self.task_ids: List[str] = []
        self.task_embeddings: List[np.ndarray] = []
        self.failure_count = 0
        self.success_count = 0
        self.avg_uncertainty = 0.0
        self.avg_quality = 0.0
        self.domain = "unknown"
        self.required_tools: List[str] = []
        self.created_at = datetime.utcnow().isoformat()
        self.last_updated = datetime.utcnow().isoformat()
    
    def add_task(
        self,
        task_id: str,
        embedding: np.ndarray,
        failed: bool,
        uncertainty: float,
        quality: float
    ) -> None:
        """Add a task to this cluster."""
        self.task_ids.append(task_id)
        self.task_embeddings.append(embedding)
        
        if failed:
            self.failure_count += 1
        else:
            self.success_count += 1
        
        # Update averages
        n = len(self.task_ids)
        self.avg_uncertainty = (
            (self.avg_uncertainty * (n - 1) + uncertainty) / n
        )
        self.avg_quality = (
            (self.avg_quality * (n - 1) + quality) / n
        )
        
        self.last_updated = datetime.utcnow().isoformat()
    
    def get_density(self) -> float:
        """Calculate cluster density (cohesion)."""
        if len(self.task_embeddings) < 2:
            return 0.0
        
        # Calculate average distance to centroid
        distances = [
            np.linalg.norm(emb - self.centroid)
            for emb in self.task_embeddings
        ]
        avg_distance = np.mean(distances)
        
        # Convert to density (inverse of distance)
        # Higher density = more cohesive cluster
        return 1.0 / (1.0 + avg_distance)
    
    def get_failure_rate(self) -> float:
        """Calculate failure rate for this cluster."""
        total = self.failure_count + self.success_count
        if total == 0:
            return 0.0
        return self.failure_count / total


class ClusterAnalyzer:
    """Analyze task history to detect clusters and spawn opportunities."""
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        min_cluster_size: int = 5,
        similarity_threshold: float = 0.7
    ):
        self.embedding_generator = embedding_generator
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.clusters: Dict[str, TaskCluster] = {}
        self.task_history: List[Dict[str, Any]] = []
    
    def add_task_result(
        self,
        task_frame: TaskFrame,
        validation_state: str,
        agent_outputs: Dict[str, Any],
        quality_score: float
    ) -> None:
        """Add a task result to history for analysis."""
        task_result = {
            "task_id": task_frame.task_id,
            "task_frame": task_frame,
            "validation_state": validation_state,
            "agent_outputs": agent_outputs,
            "quality_score": quality_score,
            "failed": validation_state in ["validation_failure", "unresolved_conflict"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.task_history.append(task_result)
        
        # Update clusters
        self._update_clusters(task_result)
    
    def detect_spawn_opportunities(self) -> List[Dict[str, Any]]:
        """Detect clusters that warrant spawning a new agent."""
        opportunities = []
        
        for cluster_id, cluster in self.clusters.items():
            # Check if cluster meets spawn criteria
            if not self._meets_spawn_criteria(cluster):
                continue
            
            # Calculate spawn score components
            spawn_info = self._analyze_cluster_for_spawn(cluster)
            
            if spawn_info["spawn_score"] >= 0.6:  # Threshold from spec
                opportunities.append(spawn_info)
        
        # Sort by spawn score
        opportunities.sort(key=lambda x: x["spawn_score"], reverse=True)
        
        return opportunities
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of all clusters."""
        return {
            "total_clusters": len(self.clusters),
            "total_tasks": len(self.task_history),
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "size": len(cluster.task_ids),
                    "failure_rate": cluster.get_failure_rate(),
                    "density": cluster.get_density(),
                    "avg_uncertainty": cluster.avg_uncertainty,
                    "domain": cluster.domain
                }
                for cluster in self.clusters.values()
            ]
        }
    
    def _update_clusters(self, task_result: Dict[str, Any]) -> None:
        """Update clusters with new task result."""
        task_frame = task_result["task_frame"]
        
        # Generate embedding for task
        embedding = self.embedding_generator.embed(task_frame.normalized_request)
        
        # Find closest cluster
        closest_cluster = None
        min_distance = float('inf')
        
        for cluster in self.clusters.values():
            distance = np.linalg.norm(embedding - cluster.centroid)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster
        
        # Check if task belongs to existing cluster
        similarity = 1.0 / (1.0 + min_distance) if min_distance < float('inf') else 0.0
        
        if similarity >= self.similarity_threshold and closest_cluster:
            # Add to existing cluster
            closest_cluster.add_task(
                task_result["task_id"],
                embedding,
                task_result["failed"],
                task_frame.initial_uncertainty,
                task_result["quality_score"]
            )
            
            # Update centroid
            closest_cluster.centroid = np.mean(closest_cluster.task_embeddings, axis=0)
        else:
            # Create new cluster
            cluster_id = f"cluster_{len(self.clusters)}"
            new_cluster = TaskCluster(cluster_id, embedding)
            new_cluster.add_task(
                task_result["task_id"],
                embedding,
                task_result["failed"],
                task_frame.initial_uncertainty,
                task_result["quality_score"]
            )
            new_cluster.domain = task_frame.task_type.value
            new_cluster.required_tools = task_frame.likely_tools
            
            self.clusters[cluster_id] = new_cluster
    
    def _meets_spawn_criteria(self, cluster: TaskCluster) -> bool:
        """Check if cluster meets basic spawn criteria."""
        # Must have minimum size
        if len(cluster.task_ids) < self.min_cluster_size:
            return False
        
        # Must have significant failures or uncertainty
        if cluster.get_failure_rate() < 0.3 and cluster.avg_uncertainty < 0.5:
            return False
        
        return True
    
    def _analyze_cluster_for_spawn(self, cluster: TaskCluster) -> Dict[str, Any]:
        """Analyze cluster to determine spawn worthiness."""
        # Calculate spawn score components
        recurring_failure_score = cluster.get_failure_rate()
        task_cluster_density = cluster.get_density()
        uncertainty_persistence = cluster.avg_uncertainty
        
        # Calculate disagreement score (simplified)
        disagreement_score = 0.5  # Would need agent output analysis
        
        # Estimate projected usage
        recent_tasks = self._get_recent_cluster_tasks(cluster, days=30)
        projected_usage = len(recent_tasks) / 30.0  # Tasks per day
        projected_usage = min(projected_usage / 5.0, 1.0)  # Normalize to 0-1
        
        # Calculate spawn score (from spec)
        spawn_score = (
            0.3 * recurring_failure_score +
            0.2 * task_cluster_density +
            0.2 * uncertainty_persistence +
            0.15 * disagreement_score +
            0.15 * projected_usage
        )
        
        return {
            "cluster_id": cluster.cluster_id,
            "spawn_score": spawn_score,
            "cluster_size": len(cluster.task_ids),
            "failure_rate": recurring_failure_score,
            "density": task_cluster_density,
            "uncertainty": uncertainty_persistence,
            "projected_usage": projected_usage,
            "domain": cluster.domain,
            "required_tools": cluster.required_tools,
            "spawn_reason": self._generate_spawn_reason(cluster, spawn_score)
        }
    
    def _get_recent_cluster_tasks(self, cluster: TaskCluster, days: int = 30) -> List[str]:
        """Get tasks in cluster from recent time period."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        recent = []
        for task_result in self.task_history:
            if task_result["task_id"] in cluster.task_ids:
                task_time = datetime.fromisoformat(task_result["timestamp"])
                if task_time >= cutoff:
                    recent.append(task_result["task_id"])
        
        return recent
    
    def _generate_spawn_reason(self, cluster: TaskCluster, spawn_score: float) -> str:
        """Generate human-readable spawn reason."""
        reasons = []
        
        if cluster.get_failure_rate() > 0.5:
            reasons.append(f"high failure rate ({cluster.get_failure_rate():.1%})")
        
        if cluster.avg_uncertainty > 0.6:
            reasons.append(f"persistent uncertainty ({cluster.avg_uncertainty:.2f})")
        
        if cluster.get_density() > 0.7:
            reasons.append(f"cohesive cluster (density {cluster.get_density():.2f})")
        
        if len(cluster.task_ids) >= self.min_cluster_size * 2:
            reasons.append(f"large cluster ({len(cluster.task_ids)} tasks)")
        
        if reasons:
            return f"Spawn recommended: {', '.join(reasons)}"
        else:
            return f"Spawn score {spawn_score:.2f} meets threshold"
