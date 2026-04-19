"""Baseline systems for comparison."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from app.schemas.task_frame import TaskFrame
from app.models.llm_client import LLMClient
import time


@dataclass
class BaselineResult:
    """Result from a baseline system."""
    
    baseline_name: str
    output: str
    tokens_used: int
    latency_seconds: float
    agents_activated: int
    quality_score: float = 0.0


class SingleAgentBaseline:
    """Single best agent baseline."""
    
    def __init__(self, llm_client: LLMClient, agent_type: str = "generalist"):
        self.llm_client = llm_client
        self.agent_type = agent_type
    
    def run(self, task: str) -> BaselineResult:
        """Run single agent on task."""
        start_time = time.time()
        
        prompt = f"""You are a general-purpose AI assistant.

Task: {task}

Provide a comprehensive answer:"""
        
        output = self.llm_client.generate(prompt, max_tokens=2000, temperature=0.7)
        
        latency = time.time() - start_time
        
        # Estimate tokens (rough approximation)
        tokens = len(prompt.split()) + len(output.split())
        
        return BaselineResult(
            baseline_name="single_agent",
            output=output,
            tokens_used=tokens,
            latency_seconds=latency,
            agents_activated=1
        )


class AlwaysOnMultiAgentBaseline:
    """Always activate all agents baseline."""
    
    def __init__(self, llm_client: LLMClient, agent_ids: List[str]):
        self.llm_client = llm_client
        self.agent_ids = agent_ids
    
    def run(self, task: str) -> BaselineResult:
        """Run all agents on every task."""
        start_time = time.time()
        
        outputs = []
        total_tokens = 0
        
        for agent_id in self.agent_ids:
            prompt = f"""You are agent: {agent_id}

Task: {task}

Provide your analysis:"""
            
            output = self.llm_client.generate(prompt, max_tokens=1000, temperature=0.7)
            outputs.append(f"[{agent_id}]: {output}")
            total_tokens += len(prompt.split()) + len(output.split())
        
        # Synthesize
        synthesis_prompt = f"""Synthesize these agent outputs:

{chr(10).join(outputs)}

Final answer:"""
        
        final_output = self.llm_client.generate(synthesis_prompt, max_tokens=1000, temperature=0.5)
        total_tokens += len(synthesis_prompt.split()) + len(final_output.split())
        
        latency = time.time() - start_time
        
        return BaselineResult(
            baseline_name="always_on_multi_agent",
            output=final_output,
            tokens_used=total_tokens,
            latency_seconds=latency,
            agents_activated=len(self.agent_ids)
        )


class StaticRoutedBaseline:
    """Static rule-based routing baseline."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.routing_rules = {
            "code": ["code_primary", "critic_verifier"],
            "research": ["web_research", "critic_verifier"],
            "general": ["code_primary"]
        }
    
    def run(self, task: str) -> BaselineResult:
        """Route based on simple keyword matching."""
        start_time = time.time()
        
        # Simple keyword-based routing
        task_lower = task.lower()
        if any(kw in task_lower for kw in ["code", "function", "implement", "debug"]):
            task_type = "code"
        elif any(kw in task_lower for kw in ["research", "explain", "what", "how"]):
            task_type = "research"
        else:
            task_type = "general"
        
        selected_agents = self.routing_rules[task_type]
        
        outputs = []
        total_tokens = 0
        
        for agent_id in selected_agents:
            prompt = f"""You are agent: {agent_id}

Task: {task}

Provide your analysis:"""
            
            output = self.llm_client.generate(prompt, max_tokens=1000, temperature=0.7)
            outputs.append(f"[{agent_id}]: {output}")
            total_tokens += len(prompt.split()) + len(output.split())
        
        # Synthesize
        synthesis_prompt = f"""Synthesize these agent outputs:

{chr(10).join(outputs)}

Final answer:"""
        
        final_output = self.llm_client.generate(synthesis_prompt, max_tokens=1000, temperature=0.5)
        total_tokens += len(synthesis_prompt.split()) + len(final_output.split())
        
        latency = time.time() - start_time
        
        return BaselineResult(
            baseline_name="static_routed",
            output=final_output,
            tokens_used=total_tokens,
            latency_seconds=latency,
            agents_activated=len(selected_agents)
        )


class BestIndividualBaseline:
    """Oracle that picks the best individual agent."""
    
    def __init__(self, llm_client: LLMClient, agent_ids: List[str]):
        self.llm_client = llm_client
        self.agent_ids = agent_ids
    
    def run(self, task: str, ground_truth: Optional[str] = None) -> BaselineResult:
        """Run all agents and pick the best output."""
        start_time = time.time()
        
        outputs = []
        total_tokens = 0
        
        for agent_id in self.agent_ids:
            prompt = f"""You are agent: {agent_id}

Task: {task}

Provide your analysis:"""
            
            output = self.llm_client.generate(prompt, max_tokens=1000, temperature=0.7)
            outputs.append((agent_id, output))
            total_tokens += len(prompt.split()) + len(output.split())
        
        # Pick best (in practice, would use quality scoring)
        # For now, just pick the longest output as proxy
        best_agent, best_output = max(outputs, key=lambda x: len(x[1]))
        
        latency = time.time() - start_time
        
        return BaselineResult(
            baseline_name="best_individual",
            output=best_output,
            tokens_used=total_tokens,
            latency_seconds=latency,
            agents_activated=len(self.agent_ids)  # All evaluated
        )


class BaselineComparator:
    """Compare full system against baselines."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.baselines = {
            "single_agent": SingleAgentBaseline(llm_client),
            "always_on": AlwaysOnMultiAgentBaseline(
                llm_client,
                ["code_primary", "web_research", "critic_verifier"]
            ),
            "static_routed": StaticRoutedBaseline(llm_client),
            "best_individual": BestIndividualBaseline(
                llm_client,
                ["code_primary", "web_research", "critic_verifier"]
            )
        }
    
    def run_all_baselines(self, task: str) -> Dict[str, BaselineResult]:
        """Run all baselines on a task."""
        results = {}
        
        for name, baseline in self.baselines.items():
            try:
                result = baseline.run(task)
                results[name] = result
            except Exception as e:
                print(f"Baseline {name} failed: {e}")
                continue
        
        return results
    
    def compare_to_baselines(
        self,
        task: str,
        system_result: Dict[str, Any],
        baseline_results: Dict[str, BaselineResult]
    ) -> Dict[str, Any]:
        """Compare system performance to baselines."""
        comparison = {
            "task": task,
            "system": {
                "tokens": system_result.get("tokens_used", 0),
                "latency": system_result.get("latency_seconds", 0),
                "agents": system_result.get("agents_activated", 0),
                "quality": system_result.get("quality_score", 0)
            },
            "baselines": {},
            "improvements": {}
        }
        
        for name, result in baseline_results.items():
            comparison["baselines"][name] = {
                "tokens": result.tokens_used,
                "latency": result.latency_seconds,
                "agents": result.agents_activated,
                "quality": result.quality_score
            }
            
            # Calculate improvements
            system_tokens = comparison["system"]["tokens"]
            system_quality = comparison["system"]["quality"]
            
            if result.tokens_used > 0:
                token_reduction = (result.tokens_used - system_tokens) / result.tokens_used
            else:
                token_reduction = 0.0
            
            quality_gain = system_quality - result.quality_score
            
            comparison["improvements"][name] = {
                "token_reduction_pct": token_reduction * 100,
                "quality_gain": quality_gain,
                "efficiency_gain": (quality_gain / (system_tokens / 1000)) if system_tokens > 0 else 0
            }
        
        return comparison
