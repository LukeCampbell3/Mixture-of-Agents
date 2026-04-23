"""Base agent class."""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from app.models.llm_client import LLMClient
from app.tools.filesystem import parse_tool_calls, CodeExtractor


# Tool call instructions — injected when repo_tool is available.
# Directive: agents MUST write code to files, not just markdown blocks.
_TOOL_CALL_INSTRUCTIONS = """
FILE OPERATIONS — you MUST use these for any code you produce:
Always emit write_file BEFORE the markdown block. Use a descriptive filename.
<tool_call>{"tool": "write_file", "path": "filename.py", "content": "complete file content here"}</tool_call>
<tool_call>{"tool": "edit_file", "path": "filename.py", "old_str": "exact text to replace", "new_str": "new text"}</tool_call>
<tool_call>{"tool": "delete_file", "path": "filename.py"}</tool_call>
<tool_call>{"tool": "mkdir", "path": "dirname"}</tool_call>
Rules: relative paths only. write_file requires COMPLETE file content. edit_file old_str must match exactly.
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
        """Execute agent on task.

        If the LLM client supports streaming (has stream_tokens()) AND
        the agent has repo_tool, tool calls are executed as they arrive
        rather than after the full response — cutting perceived latency
        to near-zero for file operations.

        After execution, CodeExtractor runs as a model-agnostic fallback:
        any markdown code blocks that weren't already written via tool_calls
        are extracted and written to files automatically.
        """
        task_frame    = task_context.get("task_frame")
        shared_context = task_context.get("shared_context", "")
        other_outputs  = task_context.get("other_agent_outputs", {})
        iteration      = task_context.get("iteration", 1)
        refinement_mode = task_context.get("refinement_mode", False)
        conflicts      = task_context.get("conflicts_detected", [])
        arbitration    = task_context.get("arbitration_results", [])
        skill_packs    = task_context.get("skill_packs", [])
        workspace_root = task_context.get("workspace_root", ".")
        # Enriched knowledge block (fetched docs, package metadata, etc.)
        knowledge_block = task_context.get("knowledge_block", "")
        # Language preference (e.g. "python", "typescript")
        lang_pref = task_context.get("language_preference", "")

        modified_context = self._apply_skill_packs(task_context, skill_packs)

        if refinement_mode and conflicts:
            prompt = self._build_refinement_prompt(
                task_frame, shared_context, other_outputs, conflicts, arbitration
            )
        else:
            prompt = self._build_prompt(
                task_frame, shared_context, other_outputs, iteration,
                workspace_root, knowledge_block=knowledge_block,
                language_preference=lang_pref,
            )

        if skill_packs:
            prompt = self._enhance_prompt_with_skill_packs(prompt, skill_packs)

        temperature = modified_context.get("temperature", 0.7)
        max_tokens  = modified_context.get("max_tokens", 800)

        # Use file tools only when the user asks for durable code changes.
        # Explanatory coding questions should not create files or build logs.
        wants_file_output = self._wants_file_output(
            getattr(task_frame, "normalized_request", "") if task_frame is not None else ""
        )
        has_file_tools = "repo_tool" in self.tools and wants_file_output
        has_streaming  = hasattr(self.llm_client, "stream_tokens")

        if has_file_tools and has_streaming:
            result = self._execute_streaming(
                prompt, max_tokens, temperature, workspace_root
            )
        else:
            response = self.llm_client.generate(
                prompt, max_tokens=max_tokens, temperature=temperature
            )
            result = self._parse_response(response)

        # ── Model-agnostic fallback: extract & write code blocks ─────────────
        if has_file_tools and task_frame is not None:
            extractor = CodeExtractor(workspace_root=workspace_root)
            extra_results = extractor.extract_and_write(
                response=result.get("raw_response", result.get("output", "")),
                task_text=task_frame.normalized_request,
                existing_tool_calls=result.get("tool_calls", []),
            )
            if extra_results:
                result.setdefault("tool_calls", [])
                result.setdefault("tool_results", [])
                for r in extra_results:
                    result["tool_calls"].append(r.op)
                    result["tool_results"].append(r)
                    status = "✓" if r.success else "✗"
                    print(f"  {status} auto-write: {r.op.path}")

        if skill_packs:
            result["skill_packs_applied"] = [pack.pack_id for pack in skill_packs]

        return result

    def _execute_streaming(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        workspace_root: str,
    ) -> Dict[str, Any]:
        """
        Stream tokens from the LLM and execute tool calls as each block closes.

        Tokens are collected into full_parts for the final response text while
        the streaming parser fires file operations the moment each closing
        </tool_call> tag arrives.
        """
        from app.tools.filesystem import FilesystemExecutor, stream_parse_tool_calls

        executor = FilesystemExecutor(workspace_root=workspace_root)
        full_parts: list[str] = []
        executed_ops: list = []

        # Tee: collect all tokens AND feed the streaming parser
        def tee_stream():
            for tok in self.llm_client.stream_tokens(
                prompt, max_tokens=max_tokens, temperature=temperature
            ):
                full_parts.append(tok)
                yield tok

        for op in stream_parse_tool_calls(tee_stream()):
            result = executor.execute(op)
            executed_ops.append(result)
            status = "✓" if result.success else "✗"
            print(f"  {status} {op.tool}: {op.path}")

        full_response = "".join(full_parts)

        # Strip tool_call blocks from display text
        import re
        display = re.sub(
            r'<tool_call>.*?</tool_call>', '', full_response,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()

        return {
            "output": display if display else full_response,
            "raw_response": full_response,
            "confidence": 0.7,
            "tool_calls": [r.op for r in executed_ops],
            "tool_results": executed_ops,
            "reasoning": "Streaming execution with inline tool call dispatch",
        }

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
        knowledge_block: str = "",
        language_preference: str = "",
    ) -> str:
        system_prompt = self.get_system_prompt()

        # Language preference override
        lang_directive = ""
        if language_preference:
            lang_directive = (
                f"\nLANGUAGE: Write all code in {language_preference}. "
                f"Use {language_preference} idioms, conventions, and standard library.\n"
            )

        # Include file tool instructions when repo_tool is available
        file_tools_block = ""
        wants_file_output = self._wants_file_output(
            getattr(task_frame, "normalized_request", "")
        )
        if "repo_tool" in self.tools and wants_file_output:
            file_tools_block = _TOOL_CALL_INSTRUCTIONS
        elif "repo_tool" in self.tools:
            file_tools_block = (
                "FILE OPERATIONS: Do not emit tool calls, do not write files, "
                "and do not include fake write_file helpers or sections about "
                "saving code to disk. Answer with explanation and markdown code only.\n"
            )

        # Knowledge block (fetched docs) goes before the task
        knowledge_section = ""
        if knowledge_block:
            knowledge_section = f"{knowledge_block}\n"

        prompt = (
            f"{system_prompt}\n"
            f"{lang_directive}"
            f"{file_tools_block}\n"
            f"{knowledge_section}"
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

        prompt += (
            "\n\nProvide a detailed, production-ready response."
            " If implementation work is required, include any file operations needed,"
            " explain the changes clearly, call out edge cases, and mention how the"
            " result should be validated."
        )
        return prompt

    @staticmethod
    def _wants_file_output(request: str) -> bool:
        """Decide whether this task should write files.

        Questions asking how something would be implemented should receive
        readable code and explanation, not automatic workspace writes.
        """

        text = (request or "").strip().lower()
        if not text:
            return False

        explanatory_prefixes = (
            "how would you",
            "how would i",
            "how do you",
            "how do i",
            "explain",
            "describe",
            "what is",
            "show me how",
        )
        explicit_file_terms = (
            "write file",
            "save",
            "create file",
            "edit",
            "modify",
            "update",
            "patch",
            "fix this file",
            "in the repo",
            "in this repo",
            "add a file",
        )
        if any(term in text for term in explicit_file_terms):
            return True
        if text.startswith(explanatory_prefixes):
            return False

        artifact_verbs = (
            "implement",
            "create",
            "build",
            "write",
            "generate",
            "fix",
            "debug",
            "refactor",
            "add",
        )
        return text.startswith(artifact_verbs)

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
