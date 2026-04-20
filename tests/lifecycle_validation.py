"""Lifecycle-specific validation harness for testing agent creation and pruning."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from app.orchestrator import Orchestrator
from app.evaluation.lifecycle_benchmark import LifecycleBenchmark, TaskCluster


class LifecycleValidator:
    """Validator specifically for lifecycle (creation/pruning) testing."""
    
    def __init__(self, output_dir: str = "lifecycle_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark = LifecycleBenchmark()
    
    def run_lifecycle_validation(self, seed: int = 42, max_tasks: int = 10):
        """Run lifecycle validation with deterministic seed.
        
        Args:
            seed: Random seed for reproducibility
            max_tasks: Maximum tasks to run (default: 10 for quick test)
        """
        
        print("=" * 80)
        print("LIFECYCLE VALIDATION - Agent Creation & Pruning")
        print("=" * 80)
        print()
        
        # Show benchmark summary
        summary = self.benchmark.get_summary()
        print("Benchmark Summary:")
        print(f"  Total tasks: {summary['total_tasks']}")
        print(f"  Epochs: {summary['epochs']}")
        print(f"  Clusters: {', '.join(summary['clusters'])}")
        print(f"  Expected specialists: {', '.join(summary['expected_specialists'])}")
        print()
        
        print("Cluster Distribution by Epoch:")
        for epoch, dist in summary['cluster_distribution'].items():
            print(f"  Epoch {epoch}: {dist}")
        print()
        
        # Limit to max_tasks for quick test
        all_tasks = self.benchmark.get_all_tasks()
        test_tasks = all_tasks[:max_tasks]
        print(f"Running {len(test_tasks)} tasks for quick validation...")
        print()
        
        # Initialize orchestrator
        orchestrator = Orchestrator(
            llm_provider="ollama",
            llm_model="qwen2.5:7b",
            budget_mode="balanced",
            enable_parallel=True,
            max_parallel_agents=3
        )
        
        # Track lifecycle metrics
        lifecycle_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "seed": seed,
            "max_tasks": max_tasks,
            "benchmark_summary": summary,
            "epochs": [],
            "lifecycle_events": [],
            "registry_snapshots": []
        }
        
        # Initial registry snapshot
        lifecycle_log["registry_snapshots"].append({
            "epoch": -1,
            "task_count": 0,
            "pool_size": len(orchestrator.registry.agents),
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "domain": agent.domain,
                    "lifecycle_state": str(agent.lifecycle_state)
                }
                for agent in orchestrator.registry.agents.values()
            ]
        })
        
        # Run tasks
        task_count = 0
        
        for task in test_tasks:
            task_count += 1
            print(f"[{task_count}/{len(test_tasks)}] {task.task_id} ({task.cluster.value})")
            print(f"    Text: {task.text[:80]}...")
            
            start_time = time.time()
            
            try:
                # Run task
                run_state = orchestrator.run_task(task.text)
                
                elapsed = time.time() - start_time
                
                # Extract lifecycle metrics
                result = {
                    "task_id": task.task_id,
                    "cluster": task.cluster.value,
                    "epoch": task.epoch,
                    "success": run_state.final_state == "success",
                    "elapsed_seconds": elapsed,
                    "agents_activated": run_state.active_agents,
                    "num_agents": len(run_state.active_agents),
                    
                    # Lifecycle metrics (NEW)
                    "spawn_recommendations": run_state.spawn_recommendations,
                    "spawned_agents": run_state.spawned_agents,
                    "probationary_agents_used": run_state.probationary_agents_used,
                    "promoted_agents": run_state.promoted_agents,
                    "pruned_agents": run_state.pruned_agents,
                    "demoted_agents": run_state.demoted_agents,
                    "lifecycle_events": run_state.lifecycle_events,
                    "pool_size_before": run_state.pool_size_before,
                    "pool_size_after": run_state.pool_size_after,
                    
                    "expected_specialist": task.expected_specialist
                }
                
                # Check if expected specialist was used
                if task.expected_specialist:
                    result["used_expected_specialist"] = task.expected_specialist in run_state.active_agents
                
                print(f"    ✓ {run_state.final_state} | {elapsed:.1f}s | {len(run_state.active_agents)} agents")
                
                # Log lifecycle events
                if run_state.spawn_recommendations:
                    print(f"    📝 Spawn recommendations: {len(run_state.spawn_recommendations)}")
                if run_state.spawned_agents:
                    print(f"    ✨ Spawned: {', '.join(run_state.spawned_agents)}")
                if run_state.promoted_agents:
                    print(f"    ⬆️  Promoted: {', '.join(run_state.promoted_agents)}")
                if run_state.pruned_agents:
                    print(f"    🗑️  Pruned: {', '.join(run_state.pruned_agents)}")
                
            except Exception as e:
                elapsed = time.time() - start_time
                result = {
                    "task_id": task.task_id,
                    "cluster": task.cluster.value,
                    "epoch": task.epoch,
                    "success": False,
                    "error": str(e),
                    "elapsed_seconds": elapsed
                }
                print(f"    ✗ Error: {str(e)[:100]}")
            
            # Add to lifecycle log
            lifecycle_log["epochs"].append(result)
        
        # Final registry snapshot
        lifecycle_log["registry_snapshots"].append({
            "epoch": test_tasks[-1].epoch if test_tasks else 0,
            "task_count": task_count,
            "pool_size": len(orchestrator.registry.agents),
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "domain": agent.domain,
                    "lifecycle_state": str(agent.lifecycle_state)
                }
                for agent in orchestrator.registry.agents.values()
            ]
        })
        
        # Final analysis
        print(f"\n{'='*80}")
        print("LIFECYCLE ANALYSIS")
        print(f"{'='*80}\n")
        
        analysis = self._analyze_lifecycle(lifecycle_log)
        self._print_analysis(analysis)
        
        # Save results
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"lifecycle_validation_{timestamp}.json"
        
        lifecycle_log["analysis"] = analysis
        
        with open(output_file, "w") as f:
            json.dump(lifecycle_log, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        return lifecycle_log, analysis
    
    def _analyze_lifecycle(self, lifecycle_log: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze lifecycle behavior across all epochs."""
        
        epochs = lifecycle_log["epochs"]
        
        # Overall metrics
        total_spawn_recs = sum(len(e.get("spawn_recommendations", [])) for e in epochs)
        total_spawned = sum(len(e.get("spawned_agents", [])) for e in epochs)
        total_promoted = sum(len(e.get("promoted_agents", [])) for e in epochs)
        total_pruned = sum(len(e.get("pruned_agents", [])) for e in epochs)
        
        # Pool growth
        snapshots = lifecycle_log["registry_snapshots"]
        initial_pool = snapshots[0]["pool_size"]
        final_pool = snapshots[-1]["pool_size"]
        pool_growth = final_pool - initial_pool
        
        return {
            "overall": {
                "total_spawn_recommendations": total_spawn_recs,
                "total_spawned": total_spawned,
                "total_promoted": total_promoted,
                "total_pruned": total_pruned,
                "initial_pool_size": initial_pool,
                "final_pool_size": final_pool,
                "pool_growth": pool_growth,
                "pool_growth_rate": pool_growth / initial_pool if initial_pool > 0 else 0
            },
            "spawn_behavior": {
                "spawn_rate": total_spawned / len(epochs) if epochs else 0
            },
            "prune_behavior": {
                "prune_rate": total_pruned / len(epochs) if epochs else 0
            },
            "lifecycle_health": {
                "spawn_to_promotion_ratio": total_promoted / total_spawned if total_spawned > 0 else 0,
                "pool_stability": 1.0 - abs(pool_growth) / initial_pool if initial_pool > 0 else 1.0
            }
        }
    
    def _print_analysis(self, analysis: Dict[str, Any]):
        """Print lifecycle analysis."""
        
        overall = analysis["overall"]
        spawn = analysis["spawn_behavior"]
        prune = analysis["prune_behavior"]
        health = analysis["lifecycle_health"]
        
        print("Overall Lifecycle Metrics:")
        print(f"  Spawn recommendations: {overall['total_spawn_recommendations']}")
        print(f"  Agents spawned: {overall['total_spawned']}")
        print(f"  Agents promoted: {overall['total_promoted']}")
        print(f"  Agents pruned: {overall['total_pruned']}")
        print(f"  Pool size: {overall['initial_pool_size']} → {overall['final_pool_size']} ({overall['pool_growth']:+d})")
        print(f"  Pool growth rate: {overall['pool_growth_rate']:+.1%}")
        print()
        
        print("Spawn Behavior:")
        print(f"  Average spawn rate: {spawn['spawn_rate']:.2f} per task")
        print()
        
        print("Prune Behavior:")
        print(f"  Average prune rate: {prune['prune_rate']:.2f} per task")
        print()
        
        print("Lifecycle Health:")
        print(f"  Spawn-to-promotion ratio: {health['spawn_to_promotion_ratio']:.1%}")
        print(f"  Pool stability: {health['pool_stability']:.1%}")
        print()
        
        # Verdict
        print("Lifecycle Validation:")
        
        if overall['total_spawned'] > 0:
            print("  ✓ Spawning mechanism active")
        else:
            print("  ⚠ No agents spawned (mechanism may be disabled)")
        
        if overall['total_pruned'] > 0:
            print("  ✓ Pruning mechanism active")
        else:
            print("  ⚠ No agents pruned (mechanism may be disabled)")
        
        if abs(overall['pool_growth']) <= 3:
            print("  ✓ Pool size stable (not exploding)")
        else:
            print(f"  ⚠ Pool grew by {overall['pool_growth']} agents")


def main():
    """Run lifecycle validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lifecycle validation")
    parser.add_argument("--max-tasks", type=int, default=10, help="Maximum tasks to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    validator = LifecycleValidator()
    lifecycle_log, analysis = validator.run_lifecycle_validation(seed=args.seed, max_tasks=args.max_tasks)
    
    print("\n" + "="*80)
    print("LIFECYCLE VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
