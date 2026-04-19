"""Benchmark tasks for evaluation."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class TaskCategory(str, Enum):
    """Task categories for benchmarking."""
    CODING = "coding"
    RESEARCH = "research"
    REASONING = "reasoning"
    MIXED = "mixed"


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    
    task_id: str
    category: TaskCategory
    description: str
    expected_agents: List[str]  # Expected agents to be activated
    success_criteria: str
    difficulty: str  # "easy", "medium", "hard"
    ground_truth: Optional[str] = None


class BenchmarkSuite:
    """Collection of benchmark tasks."""
    
    def __init__(self):
        self.tasks = self._create_tasks()
    
    def _create_tasks(self) -> List[BenchmarkTask]:
        """Create benchmark task suite."""
        return [
            # CODING TASKS
            BenchmarkTask(
                task_id="coding_easy_1",
                category=TaskCategory.CODING,
                description="Write a Python function to check if a number is prime",
                expected_agents=["code_primary"],
                success_criteria="Function works correctly for test cases",
                difficulty="easy"
            ),
            BenchmarkTask(
                task_id="coding_easy_2",
                category=TaskCategory.CODING,
                description="Write a function to reverse a string",
                expected_agents=["code_primary"],
                success_criteria="Function correctly reverses strings",
                difficulty="easy"
            ),
            BenchmarkTask(
                task_id="coding_medium_1",
                category=TaskCategory.CODING,
                description="Implement a binary search algorithm with error handling",
                expected_agents=["code_primary", "critic_verifier"],
                success_criteria="Correct implementation with edge cases handled",
                difficulty="medium"
            ),
            BenchmarkTask(
                task_id="coding_medium_2",
                category=TaskCategory.CODING,
                description="Create a class for a stack data structure with push, pop, and peek methods",
                expected_agents=["code_primary"],
                success_criteria="All methods work correctly",
                difficulty="medium"
            ),
            BenchmarkTask(
                task_id="coding_hard_1",
                category=TaskCategory.CODING,
                description="Design and implement a LRU cache with O(1) operations",
                expected_agents=["code_primary", "critic_verifier"],
                success_criteria="Correct implementation with optimal complexity",
                difficulty="hard"
            ),
            
            # RESEARCH TASKS
            BenchmarkTask(
                task_id="research_easy_1",
                category=TaskCategory.RESEARCH,
                description="What is the difference between REST and GraphQL?",
                expected_agents=["web_research"],
                success_criteria="Accurate comparison with key differences",
                difficulty="easy"
            ),
            BenchmarkTask(
                task_id="research_easy_2",
                category=TaskCategory.RESEARCH,
                description="Explain what async/await does in Python",
                expected_agents=["web_research"],
                success_criteria="Clear explanation with examples",
                difficulty="easy"
            ),
            BenchmarkTask(
                task_id="research_medium_1",
                category=TaskCategory.RESEARCH,
                description="Compare the performance characteristics of different sorting algorithms",
                expected_agents=["web_research", "critic_verifier"],
                success_criteria="Accurate complexity analysis with citations",
                difficulty="medium"
            ),
            BenchmarkTask(
                task_id="research_medium_2",
                category=TaskCategory.RESEARCH,
                description="What are the trade-offs between microservices and monolithic architecture?",
                expected_agents=["web_research"],
                success_criteria="Balanced analysis of pros and cons",
                difficulty="medium"
            ),
            BenchmarkTask(
                task_id="research_hard_1",
                category=TaskCategory.RESEARCH,
                description="Explain the CAP theorem and provide real-world examples of systems making different trade-offs",
                expected_agents=["web_research", "critic_verifier"],
                success_criteria="Accurate explanation with concrete examples",
                difficulty="hard"
            ),
            
            # REASONING TASKS
            BenchmarkTask(
                task_id="reasoning_easy_1",
                category=TaskCategory.REASONING,
                description="If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
                expected_agents=["critic_verifier"],
                success_criteria="Correct logical reasoning",
                difficulty="easy"
            ),
            BenchmarkTask(
                task_id="reasoning_medium_1",
                category=TaskCategory.REASONING,
                description="A company has 100 employees. 60% work in engineering, 30% in sales, and 20% in both. How many work in neither?",
                expected_agents=["critic_verifier"],
                success_criteria="Correct calculation with explanation",
                difficulty="medium"
            ),
            BenchmarkTask(
                task_id="reasoning_hard_1",
                category=TaskCategory.REASONING,
                description="You have 8 balls, one is heavier. Using a balance scale only twice, how do you find the heavy ball?",
                expected_agents=["critic_verifier"],
                success_criteria="Correct strategy with proof",
                difficulty="hard"
            ),
            
            # MIXED TASKS
            BenchmarkTask(
                task_id="mixed_medium_1",
                category=TaskCategory.MIXED,
                description="Implement a rate limiter in Python and explain when you would use different rate limiting algorithms",
                expected_agents=["code_primary", "web_research"],
                success_criteria="Working code with accurate explanation",
                difficulty="medium"
            ),
            BenchmarkTask(
                task_id="mixed_medium_2",
                category=TaskCategory.MIXED,
                description="Write a function to validate email addresses and explain the RFC 5322 standard",
                expected_agents=["code_primary", "web_research"],
                success_criteria="Correct implementation with standard explanation",
                difficulty="medium"
            ),
            BenchmarkTask(
                task_id="mixed_hard_1",
                category=TaskCategory.MIXED,
                description="Design a distributed task queue system, implement a basic version in Python, and explain the trade-offs",
                expected_agents=["code_primary", "web_research", "critic_verifier"],
                success_criteria="Complete design with working prototype",
                difficulty="hard"
            ),
            BenchmarkTask(
                task_id="mixed_hard_2",
                category=TaskCategory.MIXED,
                description="Implement a simple blockchain in Python and explain the cryptographic principles behind it",
                expected_agents=["code_primary", "web_research", "critic_verifier"],
                success_criteria="Working implementation with accurate explanation",
                difficulty="hard"
            ),
        ]
    
    def get_tasks_by_category(self, category: TaskCategory) -> List[BenchmarkTask]:
        """Get all tasks in a category."""
        return [t for t in self.tasks if t.category == category]
    
    def get_tasks_by_difficulty(self, difficulty: str) -> List[BenchmarkTask]:
        """Get all tasks of a difficulty level."""
        return [t for t in self.tasks if t.difficulty == difficulty]
    
    def get_task(self, task_id: str) -> Optional[BenchmarkTask]:
        """Get a specific task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_all_tasks(self) -> List[BenchmarkTask]:
        """Get all benchmark tasks."""
        return self.tasks
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the benchmark suite."""
        return {
            "total_tasks": len(self.tasks),
            "by_category": {
                category.value: len(self.get_tasks_by_category(category))
                for category in TaskCategory
            },
            "by_difficulty": {
                difficulty: len(self.get_tasks_by_difficulty(difficulty))
                for difficulty in ["easy", "medium", "hard"]
            }
        }


# Predefined test cases for validation
VALIDATION_TEST_CASES = {
    "coding_easy_1": [
        {"input": 2, "expected": True},
        {"input": 4, "expected": False},
        {"input": 17, "expected": True},
        {"input": 1, "expected": False},
    ],
    "coding_easy_2": [
        {"input": "hello", "expected": "olleh"},
        {"input": "python", "expected": "nohtyp"},
        {"input": "", "expected": ""},
    ],
    "reasoning_medium_1": {
        "answer": 10,
        "explanation": "60 + 30 - 20 = 70 work in at least one dept, so 100 - 70 = 30 in neither"
    }
}
