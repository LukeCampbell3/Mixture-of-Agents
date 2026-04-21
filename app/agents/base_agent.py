"""Base agent class."""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from app.models.llm_client import LLMClient
from app.tools.filesystem import parse_tool_calls


# Tool call instruction block injected into every prompt when file tools are available
_TOOL_CALL_INSTRUCTIONS = """
FILE OPERATIONS:
You can create, edit, or delete files by emitting tool calls in your response.
Use this format for each operation (one per block):

To CREATE or OVERWRITE a file:
<tool_call>
{"tool": "write_file", "path": "relative/path/to/file.py", "content": "full file content here"}
</tool_call>

To EDIT part of an existing file (surgical patch):
<tool_call>
{"tool": "edit_file", "path": "relative/path/to/file.py", "old_str": "exact text to replace", "new_str": "replacement text"}
</tool_call>

To DELETE a file:
<tool_call>
{"tool": "delete_file", "path": "relative/path/to/file.py"}
</tool_call>

To CREATE a directory:
<tool_call>
{"tool": "mkdir", "path": "relative/path/to/dir"}
</tool_call>

RULES:
- Always use relative paths from the workspace root.
- For write_file, include the COMPLETE file content — never truncate.
- For edit_file, old_str must match exactly (including whitespace).
- Emit tool calls AFTER your explanation, not before.
- You may emit multiple tool calls in one response.
"""


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        llm_client: LLMClient,
        tools: Optional[List[str]] = None,
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
        """Execute agent on task."""
        task_frame    = task_context.get("task_frame")
        shared_context = task_context.get("shared_context", "")
        other_outputs  = task_context.get("other_agent_outputs", {})
        iteration      = task_context.get("iteration", 1)
        refinement_mode = task_context.get("refinement_mode", False)
        conflicts      = task_context.get("conflicts_detected", [])
        arbitration    = task_context.get("arbitration_results", [])
        skill_packs    = task_context.get("skill_packs", [])
        workspace_root = task_context.get("workspace_root", ".")

        modified_context = self._apply_skill_packs(task_context, skill_packs)

        if refinement_mode and conflicts:
            prompt = self._build_refinement_prompt(
                task_frame, shared_context, other_outputs, conflicts, arbitration
            )
        else:
            prompt = self._build_prompt(
                task_frame, shared_context, other_outputs, iteration, workspace_root
            )

        if skill_packs:
            prompt = self._enhance_prompt_with_skill_packs(prompt, skill_packs)

        temperature = modified_context.get("temperature", 0.7)
        max_tokens  = modified_context.get("max_tokens", 800)

        response = self.llm_client.generate(
            prompt, max_tokens=max_tokens, temperature=temperature
        )

        result = self._parse_response(response)

        if skill_packs:
            result["skill_packs_applied"] = [pack.pack_id for pack in skill_packs]

        return result

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        task_frame: Any,
        shared_context: str,
        other_outputs: Dict[str, Any] = None,
        iteration: int = 1,
        workspace_root: str = ".",
    ) -> str:
        system_prompt = self.get_system_prompt()

        # Include file tool instructions when repo_tool is available
        file_tools_block = ""
        if "repo_tool" in self.tools:
            file_tools_block = _TOOL_CALL_INSTRUCTIONS

        prompt = (
            f"{system_prompt}\n"
            f"{file_tools_block}\n"
            f"TASK:\n{task_frame.normalized_request}\n\n"
            f"TASK TYPE: {task_frame.task_type}\n\n"
            f"ITERATION: {iteration}\n\n"
            f"SHARED CONTEXT:\n"
            f"{shared_context if shared_context else 'No prior context available.'}\n\n"
            f"CONSTRAINTS:\n"
        )

        if task_frame.hard_constraints:
            prompt += "\n".join(f"- {c}" for c in task_frame.hard_constraints)
        else:
            prompt += "None"

        if other_outputs:
            prompt += "\n\nOTHER AGENT PERSPECTIVES:\n"
            for agent_id, output in other_outputs.items():
                if agent_id != self.agent_id:
                    prompt += f"\n{agent_id}: {output.get('output', '')[:400]}...\n"
            prompt += (
                "\nCOLLABORATION INSTRUCTIONS:\n"
                "- Build on insights from other agents where they add value.\n"
                "- Challenge assumptions if you see potential issues.\n"
                "- Provide your unique expertise to complement the team.\n"
            )

        prompt += "\n\nProvide your analysis and any file operations needed."
        return prompt

    def _build_refinement_prompt(
        self,
        task_frame: Any,
        shared_context: str,
        other_outputs: Dict[str, Any],
        conflicts: List[Any],
        arbitration_results: List[Dict[str, Any]],
    ) -> str:
        system_prompt = self.get_system_prompt()

        file_tools_block = ""
        if "repo_tool" in self.tools:
            file_tools_block = _TOOL_CALL_INSTRUCTIONS

        prompt = (
            f"{system_prompt}\n"
            f"{file_tools_block}\n"
            f"TASK:\n{task_frame.normalized_request}\n\n"
            f"REFINEMENT MODE: Refine your analysis based on conflicts and arbitration.\n\n"
            f"SHARED CONTEXT:\n{shared_context}\n\n"
            f"CONFLICTS DETECTED:\n"
        )

        for i, conflict in enumerate(conflicts):
            prompt += f"\n{i+1}. {conflict.description}"
            if i < len(arbitration_results):
                prompt += f"\n   Arbitration: {arbitration_results[i]['resolution'][:300]}...\n"

        prompt += (
            "\n\nREFINEMENT INSTRUCTIONS:\n"
            "- Incorporate arbitration decisions.\n"
            "- Strengthen areas where arbitration supported your position.\n"
            "- Emit updated file operations if the code needs to change.\n\n"
            "Provide your refined analysis:"
        )
        return prompt

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse agent response, extracting any tool calls."""
        tool_calls = parse_tool_calls(response)

        # Strip tool_call blocks from the display text
        import re
        display = re.sub(
            r'<tool_call>.*?</tool_call>', '', response,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()

        return {
            "output": display if display else response,
            "raw_response": response,
            "confidence": 0.7,
            "tool_calls": tool_calls,   # List[FileOperation]
            "reasoning": "Generated response based on task context",
        }

    # ------------------------------------------------------------------
    # Skill pack helpers
    # ------------------------------------------------------------------

    def _apply_skill_packs(
        self, task_context: Dict[str, Any], skill_packs: List[Any]
    ) -> Dict[str, Any]:
        modified = task_context.copy()
        for pack in skill_packs:
            if pack.temperature is not None:
                modified["temperature"] = pack.temperature
            if pack.max_tokens is not None:
                modified["max_tokens"] = pack.max_tokens
            if pack.preferred_tools:
                current = modified.get("available_tools", [])
                modified["available_tools"] = pack.preferred_tools + [
                    t for t in current if t not in pack.preferred_tools
                ]
            if pack.domain_context:
                modified["domain_context"] = (
                    modified.get("domain_context", "") + "\n\n" + pack.domain_context
                )
        return modified

    def _enhance_prompt_with_skill_packs(
        self, base_prompt: str, skill_packs: List[Any]
    ) -> str:
        enhanced = base_prompt
        for pack in skill_packs:
            if pack.system_prompt_addition:
                enhanced = pack.system_prompt_addition + "\n\n" + enhanced
            if pack.domain_context:
                enhanced += f"\n\nDOMAIN CONTEXT ({pack.name}):\n{pack.domain_context}\n"
            if pack.examples:
                enhanced += f"\n\nEXAMPLES ({pack.name}):\n"
                for i, example in enumerate(pack.examples, 1):
                    enhanced += f"\nExample {i}:\nQ: {example.get('question','')}\nA: {example.get('answer','')}\n"
            if pack.focus_areas:
                enhanced += f"\n\nFOCUS AREAS: {', '.join(pack.focus_areas)}\n"
        return enhanced

    def get_capability_description(self) -> str:
        return f"{self.name}: {self.description}"
