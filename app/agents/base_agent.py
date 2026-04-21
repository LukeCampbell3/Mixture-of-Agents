"""Base agent class."""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from app.models.llm_client import LLMClient


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        llm_client: LLMClient,
        tools: Optional[List[str]] = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.llm_client = llm_client
        self.tools = tools or []
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass
    
    def execute(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent on task.
        
        Args:
            task_context: Dictionary containing:
                - task_frame: TaskFrame object
                - shared_context: Current shared context
                - constraints: Hard constraints
                - available_tools: List of available tool names
                - other_agent_outputs: Outputs from other agents (optional)
                - iteration: Iteration number (1 for initial, 2+ for refinement)
                - refinement_mode: Whether this is a refinement pass (optional)
                - conflicts_detected: List of conflicts (optional)
                - arbitration_results: Arbitration results (optional)
                - skill_packs: List of SkillPack objects to apply (optional)
        
        Returns:
            Dictionary containing:
                - output: Agent's response
                - confidence: Confidence score 0-1
                - tool_calls: List of tool calls made
                - reasoning: Explanation of approach
        """
        task_frame = task_context.get("task_frame")
        shared_context = task_context.get("shared_context", "")
        other_outputs = task_context.get("other_agent_outputs", {})
        iteration = task_context.get("iteration", 1)
        refinement_mode = task_context.get("refinement_mode", False)
        conflicts = task_context.get("conflicts_detected", [])
        arbitration = task_context.get("arbitration_results", [])
        skill_packs = task_context.get("skill_packs", [])
        
        # Apply skill packs to modify behavior
        modified_context = self._apply_skill_packs(task_context, skill_packs)
        
        # Build prompt based on mode
        if refinement_mode and conflicts:
            prompt = self._build_refinement_prompt(
                task_frame,
                shared_context,
                other_outputs,
                conflicts,
                arbitration
            )
        else:
            prompt = self._build_prompt(
                task_frame,
                shared_context,
                other_outputs,
                iteration
            )
        
        # Add skill pack modifications to prompt
        if skill_packs:
            prompt = self._enhance_prompt_with_skill_packs(prompt, skill_packs)
        
        # Get generation parameters (possibly modified by skill packs)
        temperature = modified_context.get("temperature", 0.7)
        max_tokens = modified_context.get("max_tokens", 600)  # conservative default for local models
        
        # Generate response
        response = self.llm_client.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        
        # Parse response
        result = self._parse_response(response)
        
        # Add skill pack metadata
        if skill_packs:
            result["skill_packs_applied"] = [pack.pack_id for pack in skill_packs]
        
        return result
    
    def _apply_skill_packs(
        self,
        task_context: Dict[str, Any],
        skill_packs: List[Any]
    ) -> Dict[str, Any]:
        """Apply skill packs to modify task context and parameters."""
        modified = task_context.copy()
        
        for pack in skill_packs:
            # Apply temperature override
            if pack.temperature is not None:
                modified["temperature"] = pack.temperature
            
            # Apply max_tokens override
            if pack.max_tokens is not None:
                modified["max_tokens"] = pack.max_tokens
            
            # Add preferred tools
            if pack.preferred_tools:
                current_tools = modified.get("available_tools", [])
                # Prioritize preferred tools
                modified["available_tools"] = pack.preferred_tools + [
                    t for t in current_tools if t not in pack.preferred_tools
                ]
            
            # Add domain context
            if pack.domain_context:
                current_context = modified.get("domain_context", "")
                modified["domain_context"] = current_context + "\n\n" + pack.domain_context
        
        return modified
    
    def _enhance_prompt_with_skill_packs(
        self,
        base_prompt: str,
        skill_packs: List[Any]
    ) -> str:
        """Enhance prompt with skill pack additions."""
        enhanced = base_prompt
        
        # Add system prompt additions
        for pack in skill_packs:
            if pack.system_prompt_addition:
                enhanced = pack.system_prompt_addition + "\n\n" + enhanced
            
            # Add domain context
            if pack.domain_context:
                enhanced += f"\n\nDOMAIN CONTEXT ({pack.name}):\n{pack.domain_context}\n"
            
            # Add examples
            if pack.examples:
                enhanced += f"\n\nEXAMPLES ({pack.name}):\n"
                for i, example in enumerate(pack.examples, 1):
                    enhanced += f"\nExample {i}:\n"
                    enhanced += f"Q: {example.get('question', '')}\n"
                    enhanced += f"A: {example.get('answer', '')}\n"
            
            # Add focus areas
            if pack.focus_areas:
                enhanced += f"\n\nFOCUS AREAS: {', '.join(pack.focus_areas)}\n"
        
        return enhanced
    
    def _build_prompt(
        self,
        task_frame: Any,
        shared_context: str,
        other_outputs: Dict[str, Any] = None,
        iteration: int = 1
    ) -> str:
        """Build prompt for agent execution."""
        system_prompt = self.get_system_prompt()
        
        prompt = f"""{system_prompt}

TASK:
{task_frame.normalized_request}

TASK TYPE: {task_frame.task_type}

ITERATION: {iteration}

SHARED CONTEXT:
{shared_context if shared_context else "No prior context available."}

CONSTRAINTS:
{chr(10).join(f"- {c}" for c in task_frame.hard_constraints) if task_frame.hard_constraints else "None"}
"""
        
        # Add other agent outputs if available
        if other_outputs and len(other_outputs) > 0:
            prompt += "\n\nOTHER AGENT PERSPECTIVES:\n"
            for agent_id, output in other_outputs.items():
                if agent_id != self.agent_id:  # Don't show own output
                    prompt += f"\n{agent_id}: {output.get('output', '')[:500]}...\n"
            
            prompt += """
COLLABORATION INSTRUCTIONS:
- Build on insights from other agents where they add value
- Challenge assumptions if you see potential issues
- Identify gaps or blind spots in other perspectives
- Provide your unique expertise to complement the team
- Be specific about where you agree, disagree, or extend others' analysis
"""
        
        prompt += "\n\nProvide your analysis and recommendations. Be specific and actionable."
        
        return prompt
    
    def _build_refinement_prompt(
        self,
        task_frame: Any,
        shared_context: str,
        other_outputs: Dict[str, Any],
        conflicts: List[Any],
        arbitration_results: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for refinement iteration."""
        system_prompt = self.get_system_prompt()
        
        prompt = f"""{system_prompt}

TASK:
{task_frame.normalized_request}

REFINEMENT MODE: You are refining your analysis based on conflicts and arbitration.

SHARED CONTEXT:
{shared_context}

CONFLICTS DETECTED:
"""
        for i, conflict in enumerate(conflicts):
            prompt += f"\n{i+1}. {conflict.description}"
            if i < len(arbitration_results):
                prompt += f"\n   Arbitration: {arbitration_results[i]['resolution'][:300]}...\n"
        
        prompt += """

REFINEMENT INSTRUCTIONS:
- Review the arbitration decisions carefully
- Refine your analysis to incorporate resolved conflicts
- Strengthen areas where arbitration supported your position
- Adjust or clarify areas where arbitration suggested improvements
- Maintain your unique perspective while integrating team insights
- Focus on creating a more robust, well-rounded analysis

Provide your refined analysis:"""
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse agent response into structured format."""
        # Simple parsing - can be enhanced with structured output
        return {
            "output": response,
            "confidence": 0.7,  # Default confidence
            "tool_calls": [],
            "reasoning": "Generated response based on task context"
        }
    
    def get_capability_description(self) -> str:
        """Get description of agent capabilities."""
        return f"{self.name}: {self.description}"
