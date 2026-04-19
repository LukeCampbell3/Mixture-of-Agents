"""Data split management for preventing overfitting in evaluation."""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum
import random
import json
from pathlib import Path


class SplitType(str, Enum):
    """Types of data splits."""
    DEVELOPMENT = "development"  # For initial development
    TUNING = "tuning"  # For tuning router weights, thresholds, calibration
    HOLDOUT = "holdout"  # Report once, never tune on
    STRESS = "stress"  # Edge cases, adversarial, distribution shift


@dataclass
class TaskSplit:
    """A task assigned to a specific split."""
    task_id: str
    split: SplitType
    category: str
    difficulty: str
    user_id: Optional[str] = None  # For user-based splits


class SplitManager:
    """Manages train/tune/holdout/stress splits to prevent overfitting."""
    
    def __init__(
        self,
        split_ratios: Optional[Dict[str, float]] = None,
        seed: int = 42
    ):
        """Initialize split manager.
        
        Args:
            split_ratios: Ratios for each split (default: 30/30/30/10)
            seed: Random seed for reproducibility
        """
        self.split_ratios = split_ratios or {
            "development": 0.30,
            "tuning": 0.30,
            "holdout": 0.30,
            "stress": 0.10
        }
        self.seed = seed
        self.splits: Dict[str, TaskSplit] = {}
        self.holdout_reported: Set[str] = set()
        random.seed(seed)
    
    def create_splits(
        self,
        task_ids: List[str],
        categories: Dict[str, str],
        difficulties: Dict[str, str],
        stratify: bool = True
    ) -> Dict[str, List[str]]:
        """Create stratified splits across tasks.
        
        Args:
            task_ids: List of task IDs
            categories: Map of task_id -> category
            difficulties: Map of task_id -> difficulty
            stratify: Whether to stratify by category and difficulty
        
        Returns:
            Dictionary mapping split type to task IDs
        """
        if stratify:
            # Group by category and difficulty
            groups = {}
            for task_id in task_ids:
                key = (categories[task_id], difficulties[task_id])
                if key not in groups:
                    groups[key] = []
                groups[key].append(task_id)
            
            # Split each group proportionally
            split_assignments = {split: [] for split in SplitType}
            
            for group_tasks in groups.values():
                random.shuffle(group_tasks)
                n = len(group_tasks)
                
                dev_end = int(n * self.split_ratios["development"])
                tune_end = dev_end + int(n * self.split_ratios["tuning"])
                holdout_end = tune_end + int(n * self.split_ratios["holdout"])
                
                split_assignments[SplitType.DEVELOPMENT].extend(group_tasks[:dev_end])
                split_assignments[SplitType.TUNING].extend(group_tasks[dev_end:tune_end])
                split_assignments[SplitType.HOLDOUT].extend(group_tasks[tune_end:holdout_end])
                split_assignments[SplitType.STRESS].extend(group_tasks[holdout_end:])
        else:
            # Simple random split
            shuffled = task_ids.copy()
            random.shuffle(shuffled)
            n = len(shuffled)
            
            dev_end = int(n * self.split_ratios["development"])
            tune_end = dev_end + int(n * self.split_ratios["tuning"])
            holdout_end = tune_end + int(n * self.split_ratios["holdout"])
            
            split_assignments = {
                SplitType.DEVELOPMENT: shuffled[:dev_end],
                SplitType.TUNING: shuffled[dev_end:tune_end],
                SplitType.HOLDOUT: shuffled[tune_end:holdout_end],
                SplitType.STRESS: shuffled[holdout_end:]
            }
        
        # Store splits
        for split_type, tasks in split_assignments.items():
            for task_id in tasks:
                self.splits[task_id] = TaskSplit(
                    task_id=task_id,
                    split=split_type,
                    category=categories[task_id],
                    difficulty=difficulties[task_id]
                )
        
        return {k.value: v for k, v in split_assignments.items()}
    
    def create_user_splits(
        self,
        user_ids: List[str],
        holdout_ratio: float = 0.2
    ) -> Dict[str, List[str]]:
        """Create held-out user slice for personalization testing.
        
        Args:
            user_ids: List of user IDs
            holdout_ratio: Ratio of users to hold out
        
        Returns:
            Dictionary with 'train' and 'holdout' user lists
        """
        shuffled = user_ids.copy()
        random.shuffle(shuffled)
        
        split_point = int(len(shuffled) * (1 - holdout_ratio))
        
        return {
            "train": shuffled[:split_point],
            "holdout": shuffled[split_point:]
        }
    
    def get_split(self, task_id: str) -> Optional[SplitType]:
        """Get the split assignment for a task."""
        if task_id in self.splits:
            return self.splits[task_id].split
        return None
    
    def can_use_for_tuning(self, task_id: str) -> bool:
        """Check if task can be used for tuning."""
        split = self.get_split(task_id)
        return split in [SplitType.DEVELOPMENT, SplitType.TUNING]
    
    def can_report_holdout(self, task_id: str) -> bool:
        """Check if holdout task can be reported (only once)."""
        if task_id in self.holdout_reported:
            return False
        
        split = self.get_split(task_id)
        if split == SplitType.HOLDOUT:
            self.holdout_reported.add(task_id)
            return True
        
        return False
    
    def get_tasks_by_split(self, split: SplitType) -> List[str]:
        """Get all tasks in a specific split."""
        return [
            task_id for task_id, task_split in self.splits.items()
            if task_split.split == split
        ]
    
    def save_splits(self, filepath: str):
        """Save splits to disk for reproducibility."""
        data = {
            "seed": self.seed,
            "split_ratios": self.split_ratios,
            "splits": {
                task_id: {
                    "split": split.split.value,
                    "category": split.category,
                    "difficulty": split.difficulty
                }
                for task_id, split in self.splits.items()
            },
            "holdout_reported": list(self.holdout_reported)
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_splits(self, filepath: str):
        """Load splits from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.seed = data["seed"]
        self.split_ratios = data["split_ratios"]
        self.holdout_reported = set(data["holdout_reported"])
        
        self.splits = {
            task_id: TaskSplit(
                task_id=task_id,
                split=SplitType(split_data["split"]),
                category=split_data["category"],
                difficulty=split_data["difficulty"]
            )
            for task_id, split_data in data["splits"].items()
        }


class CounterfactualStore:
    """Store counterfactual routing results for oracle comparison."""
    
    def __init__(self, storage_dir: str = "data/counterfactuals"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.counterfactuals: Dict[str, Dict] = {}
    
    def store_counterfactual(
        self,
        task_id: str,
        agent_subset: List[str],
        quality_score: float,
        tokens_used: int,
        latency: float,
        success: bool
    ):
        """Store a counterfactual result for a task with specific agent subset.
        
        Args:
            task_id: Task identifier
            agent_subset: List of agent IDs used
            quality_score: Quality score achieved
            tokens_used: Tokens consumed
            latency: Execution time
            success: Whether task succeeded
        """
        if task_id not in self.counterfactuals:
            self.counterfactuals[task_id] = {
                "task_id": task_id,
                "results": []
            }
        
        subset_key = tuple(sorted(agent_subset))
        
        self.counterfactuals[task_id]["results"].append({
            "agent_subset": list(subset_key),
            "quality_score": quality_score,
            "tokens_used": tokens_used,
            "latency": latency,
            "success": success
        })
    
    def get_oracle_subset(self, task_id: str) -> Optional[List[str]]:
        """Get the oracle (best) agent subset for a task.
        
        Returns the subset that achieved highest quality with lowest cost.
        """
        if task_id not in self.counterfactuals:
            return None
        
        results = self.counterfactuals[task_id]["results"]
        if not results:
            return None
        
        # Score by quality per token
        best = max(
            results,
            key=lambda r: r["quality_score"] / max(r["tokens_used"], 1)
        )
        
        return best["agent_subset"]
    
    def get_oracle_quality(self, task_id: str) -> Optional[float]:
        """Get the oracle quality score for a task."""
        if task_id not in self.counterfactuals:
            return None
        
        results = self.counterfactuals[task_id]["results"]
        if not results:
            return None
        
        return max(r["quality_score"] for r in results)
    
    def save(self):
        """Save counterfactuals to disk."""
        filepath = self.storage_dir / "counterfactuals.json"
        with open(filepath, 'w') as f:
            json.dump(self.counterfactuals, f, indent=2)
    
    def load(self):
        """Load counterfactuals from disk."""
        filepath = self.storage_dir / "counterfactuals.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                self.counterfactuals = json.load(f)
