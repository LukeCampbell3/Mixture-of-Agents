"""Tests for the evaluation system."""

import pytest
from app.evaluation.metrics import MetricsCollector, TaskMetrics
from app.evaluation.baselines import (
    SingleAgentBaseline,
    AlwaysOnMultiAgentBaseline,
    StaticRoutedBaseline,
    BestIndividualBaseline,
    BaselineComparator
)
from app.evaluation.benchmarks import BenchmarkSuite, TaskCategory
from app.models.llm_client import MockLLMClient


class TestMetricsCollector:
    """Test metrics collection and calculation."""
    
    def test_task_success_rate(self):
        """Test task success rate calculation."""
        collector = MetricsCollector()
        
        # Add successful tasks
        collector.add_task_metrics(TaskMetrics(
            task_id="task1",
            success=True,
            quality_score=0.9,
            total_tokens=1000,
            prompt_tokens=600,
            completion_tokens=400,
            latency_seconds=2.0
        ))
        
        collector.add_task_metrics(TaskMetrics(
            task_id="task2",
            success=False,
            quality_score=0.3,
            total_tokens=800,
            prompt_tokens=500,
            completion_tokens=300,
            latency_seconds=1.5
        ))
        
        assert collector.task_success_rate() == 0.5
    
    def test_quality_per_compute(self):
        """Test quality per compute metric."""
        collector = MetricsCollector()
        
        collector.add_task_metrics(TaskMetrics(
            task_id="task1",
            success=True,
            quality_score=0.8,
            total_tokens=1000,
            prompt_tokens=600,
            completion_tokens=400,
            latency_seconds=2.0
        ))
        
        qpc = collector.quality_per_compute()
        assert qpc > 0
    
    def test_average_active_agents(self):
        """Test average active agents calculation."""
        collector = MetricsCollector()
        
        collector.add_task_metrics(TaskMetrics(
            task_id="task1",
            success=True,
            quality_score=0.8,
            total_tokens=1000,
            prompt_tokens=600,
            completion_tokens=400,
            latency_seconds=2.0,
            active_agents=["agent1", "agent2"]
        ))
        
        collector.add_task_metrics(TaskMetrics(
            task_id="task2",
            success=True,
            quality_score=0.9,
            total_tokens=1200,
            prompt_tokens=700,
            completion_tokens=500,
            latency_seconds=2.5,
            active_agents=["agent1"]
        ))
        
        assert collector.average_active_agents_per_task() == 1.5
    
    def test_strong_synergy_gap(self):
        """Test strong synergy gap calculation."""
        collector = MetricsCollector()
        
        collector.add_task_metrics(TaskMetrics(
            task_id="task1",
            success=True,
            quality_score=0.9,
            total_tokens=1000,
            prompt_tokens=600,
            completion_tokens=400,
            latency_seconds=2.0,
            agent_scores={"agent1": 0.7, "agent2": 0.8},
            best_individual_score=0.8,
            team_score=0.9
        ))
        
        gap = collector.strong_synergy_gap()
        assert abs(gap - 0.1) < 0.001  # 0.9 - 0.8 (with floating point tolerance)
    
    def test_routing_entropy(self):
        """Test routing entropy calculation."""
        collector = MetricsCollector()
        
        # Add tasks with different agent usage
        for i in range(10):
            collector.add_task_metrics(TaskMetrics(
                task_id=f"task{i}",
                success=True,
                quality_score=0.8,
                total_tokens=1000,
                prompt_tokens=600,
                completion_tokens=400,
                latency_seconds=2.0,
                active_agents=["agent1"] if i < 5 else ["agent2"]
            ))
        
        entropy = collector.routing_entropy()
        assert 0 <= entropy <= 1
    
    def test_dashboard_metrics(self):
        """Test dashboard metrics generation."""
        collector = MetricsCollector()
        
        collector.add_task_metrics(TaskMetrics(
            task_id="task1",
            success=True,
            quality_score=0.9,
            total_tokens=1000,
            prompt_tokens=600,
            completion_tokens=400,
            latency_seconds=2.0,
            active_agents=["agent1", "agent2"],
            agent_scores={"agent1": 0.8, "agent2": 0.85},
            best_individual_score=0.85,
            team_score=0.9
        ))
        
        dashboard = collector.get_dashboard_metrics()
        
        assert "task_success_rate" in dashboard
        assert "quality_per_compute" in dashboard
        assert "strong_synergy_gap" in dashboard
        assert len(dashboard) == 9
    
    def test_full_report(self):
        """Test full report generation."""
        collector = MetricsCollector()
        
        collector.add_task_metrics(TaskMetrics(
            task_id="task1",
            success=True,
            quality_score=0.9,
            total_tokens=1000,
            prompt_tokens=600,
            completion_tokens=400,
            latency_seconds=2.0
        ))
        
        report = collector.get_full_report()
        
        assert "summary" in report
        assert "outcome_quality" in report
        assert "efficiency_sparsity" in report
        assert "router_quality" in report
        assert "dashboard" in report


class TestBaselines:
    """Test baseline systems."""
    
    def test_single_agent_baseline(self):
        """Test single agent baseline."""
        llm_client = MockLLMClient()
        baseline = SingleAgentBaseline(llm_client)
        
        result = baseline.run("Test task")
        
        assert result.baseline_name == "single_agent"
        assert result.agents_activated == 1
        assert result.tokens_used > 0
        assert result.latency_seconds >= 0  # Can be 0 for mock
    
    def test_always_on_baseline(self):
        """Test always-on multi-agent baseline."""
        llm_client = MockLLMClient()
        baseline = AlwaysOnMultiAgentBaseline(
            llm_client,
            ["agent1", "agent2", "agent3"]
        )
        
        result = baseline.run("Test task")
        
        assert result.baseline_name == "always_on_multi_agent"
        assert result.agents_activated == 3
        assert result.tokens_used > 0
    
    def test_static_routed_baseline(self):
        """Test static routed baseline."""
        llm_client = MockLLMClient()
        baseline = StaticRoutedBaseline(llm_client)
        
        # Test code task
        result = baseline.run("Write a function to sort a list")
        assert result.agents_activated >= 1
        
        # Test research task
        result = baseline.run("What is machine learning?")
        assert result.agents_activated >= 1
    
    def test_best_individual_baseline(self):
        """Test best individual baseline."""
        llm_client = MockLLMClient()
        baseline = BestIndividualBaseline(
            llm_client,
            ["agent1", "agent2", "agent3"]
        )
        
        result = baseline.run("Test task")
        
        assert result.baseline_name == "best_individual"
        assert result.agents_activated == 3  # All evaluated
    
    def test_baseline_comparator(self):
        """Test baseline comparator."""
        llm_client = MockLLMClient()
        comparator = BaselineComparator(llm_client)
        
        results = comparator.run_all_baselines("Test task")
        
        assert "single_agent" in results
        assert "always_on" in results
        assert "static_routed" in results
        assert "best_individual" in results
    
    def test_comparison_calculation(self):
        """Test comparison calculation."""
        llm_client = MockLLMClient()
        comparator = BaselineComparator(llm_client)
        
        system_result = {
            "tokens_used": 1000,
            "latency_seconds": 2.0,
            "agents_activated": 2,
            "quality_score": 0.9
        }
        
        baseline_results = comparator.run_all_baselines("Test task")
        
        comparison = comparator.compare_to_baselines(
            "Test task",
            system_result,
            baseline_results
        )
        
        assert "system" in comparison
        assert "baselines" in comparison
        assert "improvements" in comparison


class TestBenchmarkSuite:
    """Test benchmark suite."""
    
    def test_suite_creation(self):
        """Test benchmark suite creation."""
        suite = BenchmarkSuite()
        
        tasks = suite.get_all_tasks()
        assert len(tasks) > 0
    
    def test_filter_by_category(self):
        """Test filtering by category."""
        suite = BenchmarkSuite()
        
        coding_tasks = suite.get_tasks_by_category(TaskCategory.CODING)
        assert all(t.category == TaskCategory.CODING for t in coding_tasks)
        
        research_tasks = suite.get_tasks_by_category(TaskCategory.RESEARCH)
        assert all(t.category == TaskCategory.RESEARCH for t in research_tasks)
    
    def test_filter_by_difficulty(self):
        """Test filtering by difficulty."""
        suite = BenchmarkSuite()
        
        easy_tasks = suite.get_tasks_by_difficulty("easy")
        assert all(t.difficulty == "easy" for t in easy_tasks)
        
        hard_tasks = suite.get_tasks_by_difficulty("hard")
        assert all(t.difficulty == "hard" for t in hard_tasks)
    
    def test_get_task_by_id(self):
        """Test getting task by ID."""
        suite = BenchmarkSuite()
        
        task = suite.get_task("coding_easy_1")
        assert task is not None
        assert task.task_id == "coding_easy_1"
        
        task = suite.get_task("nonexistent")
        assert task is None
    
    def test_suite_summary(self):
        """Test suite summary."""
        suite = BenchmarkSuite()
        
        summary = suite.get_summary()
        
        assert "total_tasks" in summary
        assert "by_category" in summary
        assert "by_difficulty" in summary
        assert summary["total_tasks"] > 0


def test_calibration_error():
    """Test calibration error calculation."""
    collector = MetricsCollector()
    
    # Add tasks with varying confidence and success
    for i in range(10):
        collector.add_task_metrics(TaskMetrics(
            task_id=f"task{i}",
            success=i < 8,  # 80% success
            quality_score=0.8,  # 80% confidence
            total_tokens=1000,
            prompt_tokens=600,
            completion_tokens=400,
            latency_seconds=2.0
        ))
    
    ece = collector.calibration_error()
    assert 0 <= ece <= 1


def test_routing_consistency():
    """Test routing consistency calculation."""
    collector = MetricsCollector()
    
    # Add similar tasks with consistent routing
    for i in range(5):
        collector.add_task_metrics(TaskMetrics(
            task_id=f"coding_{i}",
            success=True,
            quality_score=0.8,
            total_tokens=1000,
            prompt_tokens=600,
            completion_tokens=400,
            latency_seconds=2.0,
            selected_agents=["code_primary", "critic_verifier"]
        ))
    
    consistency = collector.routing_consistency_within_cluster()
    assert 0 <= consistency <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
