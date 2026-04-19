"""Lifecycle benchmark with recurring task clusters for testing creation/pruning."""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class TaskCluster(str, Enum):
    """Task cluster types for lifecycle testing."""
    BROAD_CODING = "broad_coding"
    BROAD_RESEARCH = "broad_research"
    BROAD_REASONING = "broad_reasoning"
    
    # Specialized clusters that should trigger spawning
    API_MIGRATION = "api_migration"
    CONFLICTING_RESEARCH = "conflicting_research"
    HYBRID_DOCS = "hybrid_docs"
    SECURITY_AUDIT = "security_audit"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


@dataclass
class LifecycleTask:
    """A task for lifecycle testing."""
    
    task_id: str
    text: str
    cluster: TaskCluster
    epoch: int  # Which phase of the benchmark
    expected_specialist: str = None  # Expected specialist agent if any
    notes: str = ""


class LifecycleBenchmark:
    """Benchmark designed to test agent creation and pruning."""
    
    def __init__(self):
        self.tasks = self._create_lifecycle_tasks()
    
    def _create_lifecycle_tasks(self) -> List[LifecycleTask]:
        """Create lifecycle benchmark tasks."""
        tasks = []
        
        # ============================================================
        # PHASE A: WARM-UP (Epoch 0-1)
        # Broad distribution to establish baseline
        # ============================================================
        
        # Epoch 0: Initial broad tasks
        tasks.extend([
            LifecycleTask(
                task_id="warmup_code_001",
                text="Write a function to reverse a string",
                cluster=TaskCluster.BROAD_CODING,
                epoch=0
            ),
            LifecycleTask(
                task_id="warmup_code_002",
                text="Implement binary search",
                cluster=TaskCluster.BROAD_CODING,
                epoch=0
            ),
            LifecycleTask(
                task_id="warmup_research_001",
                text="What is the difference between REST and GraphQL?",
                cluster=TaskCluster.BROAD_RESEARCH,
                epoch=0
            ),
            LifecycleTask(
                task_id="warmup_research_002",
                text="Explain async/await in Python",
                cluster=TaskCluster.BROAD_RESEARCH,
                epoch=0
            ),
            LifecycleTask(
                task_id="warmup_reasoning_001",
                text="If all A are B and some B are C, can we conclude some A are C?",
                cluster=TaskCluster.BROAD_REASONING,
                epoch=0
            ),
        ])
        
        # Epoch 1: More broad tasks
        tasks.extend([
            LifecycleTask(
                task_id="warmup_code_003",
                text="Create a stack data structure",
                cluster=TaskCluster.BROAD_CODING,
                epoch=1
            ),
            LifecycleTask(
                task_id="warmup_research_003",
                text="Compare SQL vs NoSQL databases",
                cluster=TaskCluster.BROAD_RESEARCH,
                epoch=1
            ),
            LifecycleTask(
                task_id="warmup_reasoning_002",
                text="Calculate: A train leaves at 60mph, another at 40mph, 180 miles apart. When do they meet?",
                cluster=TaskCluster.BROAD_REASONING,
                epoch=1
            ),
        ])
        
        # ============================================================
        # PHASE B: RECURRING CLUSTER 1 (Epoch 2-4)
        # API Migration tasks - should trigger specialist
        # ============================================================
        
        # Epoch 2: First API migration cluster
        tasks.extend([
            LifecycleTask(
                task_id="api_mig_001",
                text="I'm migrating from Express 4 to Express 5. What breaking changes should I know about?",
                cluster=TaskCluster.API_MIGRATION,
                epoch=2,
                expected_specialist="api_migration_specialist",
                notes="First occurrence - should trigger spawn consideration"
            ),
            LifecycleTask(
                task_id="api_mig_002",
                text="Help me update my React app from class components to hooks. What patterns change?",
                cluster=TaskCluster.API_MIGRATION,
                epoch=2,
                expected_specialist="api_migration_specialist"
            ),
            LifecycleTask(
                task_id="api_mig_003",
                text="Migrating from Python 2 to Python 3. What are the main syntax changes?",
                cluster=TaskCluster.API_MIGRATION,
                epoch=2,
                expected_specialist="api_migration_specialist"
            ),
            # Mix in some broad tasks
            LifecycleTask(
                task_id="warmup_code_004",
                text="Write a function to check if a string is a palindrome",
                cluster=TaskCluster.BROAD_CODING,
                epoch=2
            ),
        ])
        
        # Epoch 3: More API migration
        tasks.extend([
            LifecycleTask(
                task_id="api_mig_004",
                text="Upgrading from Django 2.2 to Django 4.0. What deprecated features should I replace?",
                cluster=TaskCluster.API_MIGRATION,
                epoch=3,
                expected_specialist="api_migration_specialist",
                notes="Should use spawned specialist if created"
            ),
            LifecycleTask(
                task_id="api_mig_005",
                text="Moving from Vue 2 to Vue 3. How does the composition API differ from options API?",
                cluster=TaskCluster.API_MIGRATION,
                epoch=3,
                expected_specialist="api_migration_specialist"
            ),
            LifecycleTask(
                task_id="api_mig_006",
                text="Migrating from MongoDB 4 to MongoDB 5. What query syntax changes?",
                cluster=TaskCluster.API_MIGRATION,
                epoch=3,
                expected_specialist="api_migration_specialist"
            ),
        ])
        
        # Epoch 4: Final API migration burst
        tasks.extend([
            LifecycleTask(
                task_id="api_mig_007",
                text="Upgrading from Node 14 to Node 18. What new features can I use?",
                cluster=TaskCluster.API_MIGRATION,
                epoch=4,
                expected_specialist="api_migration_specialist"
            ),
            LifecycleTask(
                task_id="api_mig_008",
                text="Moving from Jest 26 to Jest 29. What test syntax changed?",
                cluster=TaskCluster.API_MIGRATION,
                epoch=4,
                expected_specialist="api_migration_specialist"
            ),
        ])
        
        # ============================================================
        # PHASE C: DISTRIBUTION SHIFT (Epoch 5-7)
        # Switch to different cluster - test cooling/pruning
        # ============================================================
        
        # Epoch 5: Introduce security audit cluster
        tasks.extend([
            LifecycleTask(
                task_id="security_001",
                text="Review this authentication code for SQL injection vulnerabilities",
                cluster=TaskCluster.SECURITY_AUDIT,
                epoch=5,
                expected_specialist="security_specialist",
                notes="New cluster - API migration specialist should cool down"
            ),
            LifecycleTask(
                task_id="security_002",
                text="Audit this API endpoint for CSRF protection",
                cluster=TaskCluster.SECURITY_AUDIT,
                epoch=5,
                expected_specialist="security_specialist"
            ),
            LifecycleTask(
                task_id="security_003",
                text="Check this password hashing implementation for security issues",
                cluster=TaskCluster.SECURITY_AUDIT,
                epoch=5,
                expected_specialist="security_specialist"
            ),
        ])
        
        # Epoch 6: More security tasks
        tasks.extend([
            LifecycleTask(
                task_id="security_004",
                text="Review this JWT implementation for security best practices",
                cluster=TaskCluster.SECURITY_AUDIT,
                epoch=6,
                expected_specialist="security_specialist"
            ),
            LifecycleTask(
                task_id="security_005",
                text="Audit this file upload handler for path traversal vulnerabilities",
                cluster=TaskCluster.SECURITY_AUDIT,
                epoch=6,
                expected_specialist="security_specialist"
            ),
            # Mix in broad task
            LifecycleTask(
                task_id="warmup_research_004",
                text="What are the main principles of secure coding?",
                cluster=TaskCluster.BROAD_RESEARCH,
                epoch=6
            ),
        ])
        
        # Epoch 7: Final security burst
        tasks.extend([
            LifecycleTask(
                task_id="security_006",
                text="Check this session management code for security flaws",
                cluster=TaskCluster.SECURITY_AUDIT,
                epoch=7,
                expected_specialist="security_specialist"
            ),
            LifecycleTask(
                task_id="security_007",
                text="Review this API rate limiting implementation",
                cluster=TaskCluster.SECURITY_AUDIT,
                epoch=7,
                expected_specialist="security_specialist"
            ),
        ])
        
        # ============================================================
        # PHASE D: RETURN TO BROAD (Epoch 8-9)
        # Test pruning of unused specialists
        # ============================================================
        
        # Epoch 8: Back to broad distribution
        tasks.extend([
            LifecycleTask(
                task_id="cooldown_code_001",
                text="Implement a queue data structure",
                cluster=TaskCluster.BROAD_CODING,
                epoch=8,
                notes="Specialists should cool down if not used"
            ),
            LifecycleTask(
                task_id="cooldown_research_001",
                text="Explain the CAP theorem",
                cluster=TaskCluster.BROAD_RESEARCH,
                epoch=8
            ),
            LifecycleTask(
                task_id="cooldown_reasoning_003",
                text="You have 8 balls, one is heavier. Using a balance scale twice, find it.",
                cluster=TaskCluster.BROAD_REASONING,
                epoch=8
            ),
        ])
        
        # Epoch 9: More broad tasks
        tasks.extend([
            LifecycleTask(
                task_id="cooldown_code_002",
                text="Write a function to find the longest common substring",
                cluster=TaskCluster.BROAD_CODING,
                epoch=9
            ),
            LifecycleTask(
                task_id="cooldown_research_002",
                text="Compare microservices vs monolithic architecture",
                cluster=TaskCluster.BROAD_RESEARCH,
                epoch=9
            ),
        ])
        
        # ============================================================
        # PHASE E: REACTIVATION TEST (Epoch 10)
        # One-off return to old cluster
        # ============================================================
        
        # Epoch 10: Single API migration task
        tasks.extend([
            LifecycleTask(
                task_id="reactivation_001",
                text="Quick question: migrating from Angular 14 to 15, any breaking changes?",
                cluster=TaskCluster.API_MIGRATION,
                epoch=10,
                notes="Test if dormant specialist can be reactivated"
            ),
        ])
        
        return tasks
    
    def get_tasks_by_epoch(self, epoch: int) -> List[LifecycleTask]:
        """Get all tasks for a specific epoch."""
        return [t for t in self.tasks if t.epoch == epoch]
    
    def get_tasks_by_cluster(self, cluster: TaskCluster) -> List[LifecycleTask]:
        """Get all tasks in a cluster."""
        return [t for t in self.tasks if t.cluster == cluster]
    
    def get_all_tasks(self) -> List[LifecycleTask]:
        """Get all tasks in order."""
        return sorted(self.tasks, key=lambda t: (t.epoch, t.task_id))
    
    def get_epoch_count(self) -> int:
        """Get number of epochs."""
        return max(t.epoch for t in self.tasks) + 1
    
    def get_cluster_distribution(self) -> Dict[int, Dict[str, int]]:
        """Get cluster distribution by epoch."""
        distribution = {}
        for epoch in range(self.get_epoch_count()):
            epoch_tasks = self.get_tasks_by_epoch(epoch)
            distribution[epoch] = {}
            for task in epoch_tasks:
                cluster = task.cluster.value
                distribution[epoch][cluster] = distribution[epoch].get(cluster, 0) + 1
        return distribution
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        return {
            "total_tasks": len(self.tasks),
            "epochs": self.get_epoch_count(),
            "clusters": list(set(t.cluster.value for t in self.tasks)),
            "cluster_distribution": self.get_cluster_distribution(),
            "expected_specialists": list(set(
                t.expected_specialist 
                for t in self.tasks 
                if t.expected_specialist
            ))
        }
