"""Large-scale validation with realistic, diverse prompts."""

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from app.orchestrator import Orchestrator
from app.evaluation.realistic_prompts import (
    RealisticPromptDataset,
    PromptQuality,
    PromptComplexity
)


class LargeScaleValidator:
    """Large-scale validation with diverse realistic prompts."""
    
    def __init__(self, output_dir: str = "large_scale_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dataset = RealisticPromptDataset()
    
    def run_full_validation(
        self,
        sample_size: int = 30,
        test_parallel: bool = True,
        test_sequential: bool = True,
        test_single: bool = True
    ):
        """Run full large-scale validation.
        
        Args:
            sample_size: Number of prompts to test (default: 30)
            test_parallel: Test parallel multi-agent
            test_sequential: Test sequential multi-agent
            test_single: Test single-agent baseline
        """
        print("=" * 80)
        print("LARGE-SCALE ARCHITECTURE VALIDATION")
        print("=" * 80)
        print()
        
        # Get dataset statistics
        stats = self.dataset.get_statistics()
        print("Dataset Statistics:")
        print(f"  Total prompts: {stats['total_prompts']}")
        print(f"  By quality: {stats['by_quality']}")
        print(f"  By complexity: {stats['by_complexity']}")
        print(f"  By category: {stats['by_category']}")
        print()
        
        # Select diverse sample
        print(f"Selecting {sample_size} diverse prompts...")
        test_prompts = self.dataset.get_sample(sample_size, diverse=True)
        print(f"Selected {len(test_prompts)} prompts")
        print()
        
        # Show sample breakdown
        quality_counts = {}
        complexity_counts = {}
        category_counts = {}
        
        for prompt in test_prompts:
            quality_counts[prompt.quality.value] = quality_counts.get(prompt.quality.value, 0) + 1
            complexity_counts[prompt.complexity.value] = complexity_counts.get(prompt.complexity.value, 0) + 1
            category_counts[prompt.category] = category_counts.get(prompt.category, 0) + 1
        
        print("Sample Breakdown:")
        print(f"  Quality: {quality_counts}")
        print(f"  Complexity: {complexity_counts}")
        print(f"  Category: {category_counts}")
        print()
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Test configurations
        configs_to_test = []
        
        if test_single:
            configs_to_test.append({
                "name": "single_agent",
                "description": "Single agent baseline",
                "enable_parallel": False,
                "budget_mode": "low"
            })
        
        if test_sequential:
            configs_to_test.append({
                "name": "sequential_multi",
                "description": "Multi-agent sequential",
                "enable_parallel": False,
                "budget_mode": "balanced"
            })
        
        if test_parallel:
            configs_to_test.append({
                "name": "parallel_multi",
                "description": "Multi-agent parallel (NEW)",
                "enable_parallel": True,
                "budget_mode": "balanced"
            })
        
        # Run tests
        all_results = {}
        
        for config in configs_to_test:
            print(f"\n{'='*80}")
            print(f"TESTING: {config['description']}")
            print(f"{'='*80}\n")
            
            results = self._test_configuration(config, test_prompts)
            all_results[config["name"]] = results
            
            # Print summary
            self._print_config_summary(config["name"], results)
        
        # Comprehensive analysis
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}\n")
        
        analysis = self._analyze_results(all_results, test_prompts)
        self._print_analysis(analysis)
        
        # Save results
        output_file = self.output_dir / f"large_scale_validation_{timestamp}.json"
        self._save_results(all_results, analysis, test_prompts, output_file)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}\n")
        
        # Final verdict
        self._print_final_verdict(analysis)
        
        return all_results, analysis
    
    def _test_configuration(
        self,
        config: Dict[str, Any],
        prompts: List[Any]
    ) -> Dict[str, Any]:
        """Test a configuration with all prompts."""
        
        # Initialize orchestrator
        orchestrator = Orchestrator(
            llm_provider="ollama",
            llm_model="qwen2.5:7b",
            budget_mode=config.get("budget_mode", "balanced"),
            enable_parallel=config.get("enable_parallel", False),
            max_parallel_agents=3
        )
        
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"  [{i}/{len(prompts)}] {prompt.prompt_id} ({prompt.quality.value}, {prompt.complexity.value})")
            print(f"      Text: {prompt.text[:80]}...")
            
            start_time = time.time()
            
            try:
                # Run task
                run_state = orchestrator.run_task(prompt.text)
                
                elapsed = time.time() - start_time
                
                # Analyze result
                result = {
                    "prompt_id": prompt.prompt_id,
                    "category": prompt.category,
                    "quality": prompt.quality.value,
                    "complexity": prompt.complexity.value,
                    "success": run_state.final_state == "success",
                    "final_state": run_state.final_state,
                    "elapsed_seconds": elapsed,
                    "agents_activated": run_state.active_agents,
                    "num_agents": len(run_state.active_agents),
                    "suppressed_agents": [s["agent_id"] for s in run_state.suppressed_agents],
                    "answer_length": len(run_state.final_answer) if run_state.final_answer else 0,
                    "expected_agents": prompt.expected_agents,
                    "expected_challenges": prompt.expected_challenges
                }
                
                # Check if expected agents were used
                result["used_expected_agents"] = any(
                    agent in run_state.active_agents 
                    for agent in prompt.expected_agents
                )
                
                # Check for multi-agent collaboration
                if len(run_state.active_agents) > 1:
                    result["multi_agent_used"] = True
                    
                    # Check for collaboration evidence
                    answer_lower = run_state.final_answer.lower() if run_state.final_answer else ""
                    collaboration_keywords = [
                        "perspective", "analysis", "synthesis", "integrate", 
                        "combine", "together", "complement", "both", "multiple"
                    ]
                    result["collaboration_keywords_found"] = [
                        kw for kw in collaboration_keywords if kw in answer_lower
                    ]
                    result["has_collaboration_evidence"] = len(result["collaboration_keywords_found"]) > 0
                else:
                    result["multi_agent_used"] = False
                    result["collaboration_keywords_found"] = []
                    result["has_collaboration_evidence"] = False
                
                # Quality assessment based on prompt quality
                if prompt.quality in [PromptQuality.INCOMPLETE, PromptQuality.VAGUE, PromptQuality.AMBIGUOUS]:
                    # For incomplete/vague prompts, check if agent asks for clarification
                    answer_lower = run_state.final_answer.lower() if run_state.final_answer else ""
                    clarification_keywords = [
                        "clarif", "specify", "which", "what do you mean", 
                        "could you", "need more", "unclear", "ambiguous"
                    ]
                    result["asks_for_clarification"] = any(
                        kw in answer_lower for kw in clarification_keywords
                    )
                
                if prompt.quality == PromptQuality.CONFLICTING:
                    # Check if agent identifies conflicts
                    answer_lower = run_state.final_answer.lower() if run_state.final_answer else ""
                    conflict_keywords = [
                        "conflict", "contradict", "impossible", "cannot", 
                        "trade-off", "tradeoff", "mutually exclusive"
                    ]
                    result["identifies_conflicts"] = any(
                        kw in answer_lower for kw in conflict_keywords
                    )
                
                print(f"      ✓ {run_state.final_state} | {elapsed:.1f}s | {len(run_state.active_agents)} agents")
                
            except Exception as e:
                elapsed = time.time() - start_time
                result = {
                    "prompt_id": prompt.prompt_id,
                    "category": prompt.category,
                    "quality": prompt.quality.value,
                    "complexity": prompt.complexity.value,
                    "success": False,
                    "error": str(e),
                    "error_trace": traceback.format_exc(),
                    "elapsed_seconds": elapsed,
                    "num_agents": 0
                }
                print(f"      ✗ Error: {str(e)[:100]}")
            
            results.append(result)
        
        return {
            "config_name": config["name"],
            "config_params": config,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _print_config_summary(self, config_name: str, results: Dict[str, Any]):
        """Print summary for a configuration."""
        task_results = results["results"]
        
        total = len(task_results)
        successful = sum(1 for r in task_results if r.get("success", False))
        multi_agent = sum(1 for r in task_results if r.get("multi_agent_used", False))
        with_collaboration = sum(1 for r in task_results if r.get("has_collaboration_evidence", False))
        avg_time = sum(r.get("elapsed_seconds", 0) for r in task_results) / total if total > 0 else 0
        avg_agents = sum(r.get("num_agents", 0) for r in task_results) / total if total > 0 else 0
        
        # By quality
        by_quality = {}
        for r in task_results:
            quality = r.get("quality", "unknown")
            if quality not in by_quality:
                by_quality[quality] = {"total": 0, "success": 0}
            by_quality[quality]["total"] += 1
            if r.get("success", False):
                by_quality[quality]["success"] += 1
        
        print(f"\n  Summary for {config_name}:")
        print(f"    Overall success: {successful}/{total} ({successful/total*100:.1f}%)")
        print(f"    Multi-agent tasks: {multi_agent}/{total} ({multi_agent/total*100:.1f}%)")
        print(f"    With collaboration evidence: {with_collaboration}/{multi_agent if multi_agent > 0 else 1}")
        print(f"    Avg time: {avg_time:.1f}s")
        print(f"    Avg agents: {avg_agents:.1f}")
        print(f"\n    Success by prompt quality:")
        for quality, stats in sorted(by_quality.items()):
            success_rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"      {quality:15s}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
    
    def _analyze_results(
        self,
        all_results: Dict[str, Dict[str, Any]],
        prompts: List[Any]
    ) -> Dict[str, Any]:
        """Analyze results across configurations."""
        
        analysis = {
            "by_config": {},
            "by_quality": {},
            "by_complexity": {},
            "by_category": {},
            "comparisons": {}
        }
        
        # Analyze each configuration
        for config_name, config_results in all_results.items():
            results = config_results["results"]
            
            metrics = {
                "total": len(results),
                "successful": sum(1 for r in results if r.get("success", False)),
                "success_rate": 0.0,
                "multi_agent_used": sum(1 for r in results if r.get("multi_agent_used", False)),
                "multi_agent_rate": 0.0,
                "with_collaboration": sum(1 for r in results if r.get("has_collaboration_evidence", False)),
                "collaboration_rate": 0.0,
                "avg_time": 0.0,
                "avg_agents": 0.0,
                "total_time": sum(r.get("elapsed_seconds", 0) for r in results),
                "errors": sum(1 for r in results if "error" in r)
            }
            
            if metrics["total"] > 0:
                metrics["success_rate"] = metrics["successful"] / metrics["total"]
                metrics["multi_agent_rate"] = metrics["multi_agent_used"] / metrics["total"]
                metrics["avg_time"] = metrics["total_time"] / metrics["total"]
                metrics["avg_agents"] = sum(r.get("num_agents", 0) for r in results) / metrics["total"]
            
            if metrics["multi_agent_used"] > 0:
                metrics["collaboration_rate"] = metrics["with_collaboration"] / metrics["multi_agent_used"]
            
            # By quality
            by_quality = {}
            for r in results:
                quality = r.get("quality", "unknown")
                if quality not in by_quality:
                    by_quality[quality] = {"total": 0, "success": 0, "multi_agent": 0}
                by_quality[quality]["total"] += 1
                if r.get("success", False):
                    by_quality[quality]["success"] += 1
                if r.get("multi_agent_used", False):
                    by_quality[quality]["multi_agent"] += 1
            
            metrics["by_quality"] = by_quality
            
            analysis["by_config"][config_name] = metrics
        
        # Comparisons
        if "parallel_multi" in analysis["by_config"] and "sequential_multi" in analysis["by_config"]:
            parallel = analysis["by_config"]["parallel_multi"]
            sequential = analysis["by_config"]["sequential_multi"]
            
            if sequential["avg_time"] > 0:
                analysis["comparisons"]["parallel_speedup"] = sequential["avg_time"] / parallel["avg_time"]
            
            analysis["comparisons"]["parallel_vs_sequential"] = {
                "quality_delta": parallel["success_rate"] - sequential["success_rate"],
                "time_ratio": parallel["avg_time"] / sequential["avg_time"] if sequential["avg_time"] > 0 else 0,
                "multi_agent_delta": parallel["multi_agent_rate"] - sequential["multi_agent_rate"]
            }
        
        if "parallel_multi" in analysis["by_config"] and "single_agent" in analysis["by_config"]:
            multi = analysis["by_config"]["parallel_multi"]
            single = analysis["by_config"]["single_agent"]
            
            analysis["comparisons"]["multi_vs_single"] = {
                "quality_delta": multi["success_rate"] - single["success_rate"],
                "time_ratio": multi["avg_time"] / single["avg_time"] if single["avg_time"] > 0 else 0,
                "multi_agent_activation": multi["multi_agent_rate"]
            }
        
        return analysis
    
    def _print_analysis(self, analysis: Dict[str, Any]):
        """Print comprehensive analysis."""
        
        # Configuration comparison table
        print("Configuration Comparison:")
        print(f"{'Config':<20} {'Success':<12} {'Multi-Agent':<12} {'Collab':<12} {'Avg Time':<12} {'Avg Agents':<12}")
        print("-" * 80)
        
        for config_name, metrics in analysis["by_config"].items():
            success_pct = metrics["success_rate"] * 100
            multi_pct = metrics["multi_agent_rate"] * 100
            collab_pct = metrics["collaboration_rate"] * 100
            avg_time = metrics["avg_time"]
            avg_agents = metrics["avg_agents"]
            
            print(f"{config_name:<20} {success_pct:>8.1f}%    {multi_pct:>8.1f}%    {collab_pct:>8.1f}%    {avg_time:>8.1f}s    {avg_agents:>8.1f}")
        
        print()
        
        # Key findings
        print("Key Findings:")
        
        comparisons = analysis.get("comparisons", {})
        
        if "parallel_speedup" in comparisons:
            speedup = comparisons["parallel_speedup"]
            print(f"  • Parallel speedup: {speedup:.2f}x")
            if speedup > 1.5:
                print(f"    ✓ EXCELLENT parallel improvement")
            elif speedup > 1.3:
                print(f"    ✓ SIGNIFICANT parallel improvement")
            elif speedup > 1.1:
                print(f"    ✓ Moderate parallel improvement")
            else:
                print(f"    ⚠ Limited parallel benefit")
        
        if "multi_vs_single" in comparisons:
            comp = comparisons["multi_vs_single"]
            print(f"  • Multi-agent vs Single-agent:")
            print(f"    - Quality delta: {comp['quality_delta']:+.1%}")
            print(f"    - Time ratio: {comp['time_ratio']:.2f}x")
            print(f"    - Multi-agent activation: {comp['multi_agent_activation']:.1%}")
            
            if comp["quality_delta"] >= 0 and comp["time_ratio"] < 2.5:
                print(f"    ✓ Multi-agent provides good value")
            elif comp["quality_delta"] >= 0:
                print(f"    ⚠ Multi-agent maintains quality but slower")
            else:
                print(f"    ✗ Multi-agent needs improvement")
        
        print()
        
        # Performance by prompt quality
        print("Performance by Prompt Quality:")
        
        if "parallel_multi" in analysis["by_config"]:
            by_quality = analysis["by_config"]["parallel_multi"].get("by_quality", {})
            
            for quality in sorted(by_quality.keys()):
                stats = by_quality[quality]
                success_rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
                multi_rate = stats["multi_agent"] / stats["total"] * 100 if stats["total"] > 0 else 0
                
                print(f"  {quality:15s}: {stats['success']}/{stats['total']} success ({success_rate:.1f}%), {multi_rate:.1f}% multi-agent")
    
    def _print_final_verdict(self, analysis: Dict[str, Any]):
        """Print final verdict."""
        
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)
        print()
        
        if "parallel_multi" not in analysis["by_config"]:
            print("⚠ Parallel multi-agent not tested")
            return
        
        metrics = analysis["by_config"]["parallel_multi"]
        comparisons = analysis.get("comparisons", {})
        
        passed = 0
        total = 0
        
        # Quality check
        total += 1
        if metrics["success_rate"] >= 0.7:
            print("  ✓ Quality: PASS (≥70% success rate)")
            passed += 1
        else:
            print(f"  ✗ Quality: FAIL ({metrics['success_rate']*100:.1f}% < 70%)")
        
        # Parallel efficiency check
        if "parallel_speedup" in comparisons:
            total += 1
            if comparisons["parallel_speedup"] > 1.2:
                print("  ✓ Parallel Efficiency: PASS (>1.2x speedup)")
                passed += 1
            else:
                print(f"  ⚠ Parallel Efficiency: NEEDS IMPROVEMENT ({comparisons['parallel_speedup']:.2f}x)")
        
        # Multi-agent activation check
        total += 1
        if metrics["multi_agent_rate"] > 0.3:
            print("  ✓ Multi-Agent Activation: PASS (>30% of tasks)")
            passed += 1
        else:
            print(f"  ⚠ Multi-Agent Activation: NEEDS IMPROVEMENT ({metrics['multi_agent_rate']*100:.1f}%)")
        
        # Collaboration evidence check
        total += 1
        if metrics["collaboration_rate"] > 0.5:
            print("  ✓ Collaboration Evidence: PASS (>50% of multi-agent tasks)")
            passed += 1
        else:
            print(f"  ⚠ Collaboration Evidence: NEEDS IMPROVEMENT ({metrics['collaboration_rate']*100:.1f}%)")
        
        # Multi-agent value check
        if "multi_vs_single" in comparisons:
            total += 1
            comp = comparisons["multi_vs_single"]
            if comp["quality_delta"] >= -0.05 and comp["time_ratio"] < 3.0:
                print("  ✓ Multi-Agent Value: PASS (quality maintained, reasonable efficiency)")
                passed += 1
            else:
                print(f"  ⚠ Multi-Agent Value: NEEDS IMPROVEMENT")
        
        print()
        print(f"Overall: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("\n🎉 VALIDATION PASSED - Architecture improvements validated!")
        elif passed >= total * 0.7:
            print("\n✓ VALIDATION MOSTLY PASSED - Minor improvements needed")
        else:
            print("\n⚠ VALIDATION NEEDS WORK - Significant improvements required")
    
    def _save_results(
        self,
        all_results: Dict[str, Any],
        analysis: Dict[str, Any],
        prompts: List[Any],
        output_file: Path
    ):
        """Save results to file."""
        
        output_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "validation_type": "large_scale_realistic",
            "dataset_stats": self.dataset.get_statistics(),
            "prompts_tested": [
                {
                    "prompt_id": p.prompt_id,
                    "text": p.text,
                    "category": p.category,
                    "quality": p.quality.value,
                    "complexity": p.complexity.value
                }
                for p in prompts
            ],
            "results": all_results,
            "analysis": analysis
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)


def main():
    """Run large-scale validation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Large-scale architecture validation")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of prompts to test")
    parser.add_argument("--no-parallel", action="store_true", help="Skip parallel testing")
    parser.add_argument("--no-sequential", action="store_true", help="Skip sequential testing")
    parser.add_argument("--no-single", action="store_true", help="Skip single-agent testing")
    
    args = parser.parse_args()
    
    validator = LargeScaleValidator()
    
    results, analysis = validator.run_full_validation(
        sample_size=args.sample_size,
        test_parallel=not args.no_parallel,
        test_sequential=not args.no_sequential,
        test_single=not args.no_single
    )
    
    print("\n" + "="*80)
    print("LARGE-SCALE VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
