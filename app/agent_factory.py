"""Agent factory for creating new agents dynamically."""

from typing import Dict, Any, Optional
from app.schemas.registry import AgentSpec, LifecycleState
from app.agents.base_agent import BaseAgent
from app.models.llm_client import LLMClient


class DynamicAgent(BaseAgent):
    """Dynamically created agent with an LLM-generated system prompt."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        llm_client: LLMClient,
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
        domain: str = "specialized",
    ):
        super().__init__(agent_id, name, description, llm_client, tools)
        self.custom_system_prompt = system_prompt
        self.domain = domain

    def get_system_prompt(self) -> str:
        if self.custom_system_prompt:
            return self.custom_system_prompt
        # Fallback — should rarely be hit because factory always generates one
        return (
            f"You are a specialized {self.domain} agent.\n"
            f"Role: {self.description}\n\n"
            "RULES:\n"
            "- Always write complete, working code when asked to implement something.\n"
            "- Use markdown code blocks with the correct language tag.\n"
            "- Be specific and actionable.\n"
        )


class AgentFactory:
    """Factory for creating new agents dynamically using the LLM."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_agent_from_spec(self, spawn_spec: Dict[str, Any]) -> AgentSpec:
        """Build an AgentSpec from a spawn recommendation dict."""
        return AgentSpec(
            agent_id=spawn_spec["agent_id"],
            name=spawn_spec["name"],
            description=spawn_spec["description"],
            domain=spawn_spec["domain"],
            lifecycle_state=LifecycleState.PROBATIONARY,
            tools=spawn_spec.get("tools", []),
            tags=spawn_spec.get("tags", []),
            target_cluster=spawn_spec.get("target_cluster"),
            expected_activation_rate=spawn_spec.get("expected_activation_rate", 0.1),
        )

    def create_agent_instance(self, agent_spec: AgentSpec) -> DynamicAgent:
        """Instantiate a DynamicAgent from a spec, generating its system prompt."""
        system_prompt = self._generate_system_prompt(agent_spec)
        return DynamicAgent(
            agent_id=agent_spec.agent_id,
            name=agent_spec.name,
            description=agent_spec.description,
            llm_client=self.llm_client,
            tools=agent_spec.tools,
            system_prompt=system_prompt,
            domain=agent_spec.domain,
        )

    def spawn_for_task(
        self,
        task_text: str,
        domain: str,
        agent_id: str,
    ):
        """
        Immediately create a new specialist agent for a task that has no good match.

        Uses the LLM to write a focused system prompt, then returns both the
        AgentSpec (for the registry) and a ready-to-use DynamicAgent instance.
        Returns: (AgentSpec, DynamicAgent)
        """
        # Ask the LLM to write the system prompt
        system_prompt = self._llm_generate_system_prompt(task_text, domain)

        name = f"{domain.replace('_', ' ').title()} Specialist"
        description = (
            f"Dynamically spawned specialist for {domain} tasks. "
            f"Created in response to: {task_text[:120]}"
        )

        spec = AgentSpec(
            agent_id=agent_id,
            name=name,
            description=description,
            domain=domain,
            lifecycle_state=LifecycleState.PROBATIONARY,
            tools=self._infer_tools(domain),
            tags=[domain, "spawned", "dynamic"],
        )

        agent = DynamicAgent(
            agent_id=agent_id,
            name=name,
            description=description,
            llm_client=self.llm_client,
            tools=spec.tools,
            system_prompt=system_prompt,
            domain=domain,
        )

        return spec, agent

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _llm_generate_system_prompt(self, task_text: str, domain: str) -> str:
        """
        Ask the LLM to write a tight, code-first system prompt for a new specialist.
        Falls back to a template if the LLM call fails.
        """
        meta_prompt = f"""Write a system prompt for a specialist AI agent.

The agent's domain: {domain}
Example task it will handle: {task_text}

Requirements for the system prompt:
- Start with "You are a {domain} specialist agent."
- Include a RULES section that says: always write complete working code in markdown code blocks
- List 4-6 specific areas of expertise relevant to {domain}
- Describe the expected output format (code block first, then brief explanation)
- Keep it under 200 words
- Do NOT include meta-commentary — write the prompt itself, nothing else

System prompt:"""

        try:
            prompt = self.llm_client.generate(
                meta_prompt, max_tokens=300, temperature=0.4
            )
            # Sanity check — must mention the domain
            if domain.split("_")[0].lower() in prompt.lower() or len(prompt) > 100:
                return prompt.strip()
        except Exception:
            pass

        # Template fallback
        return self._template_system_prompt(domain, task_text)

    def _template_system_prompt(self, domain: str, task_text: str) -> str:
        """Deterministic fallback system prompt."""
        domain_display = domain.replace("_", " ")
        return f"""You are a {domain_display} specialist agent.

RULES:
- ALWAYS write complete, working code — never just describe what code should do.
- Use markdown code blocks with the correct language tag.
- After the code, add a brief explanation (2-5 sentences).

Expertise: {domain_display} tasks including implementation, debugging, and best practices.

Example task: {task_text[:100]}

Output format:
```<language>
# complete code here
```

**How it works:** brief explanation.
"""

    def _generate_system_prompt(self, agent_spec: AgentSpec) -> str:
        """Generate system prompt for an agent spec (used by create_agent_instance)."""
        return self._llm_generate_system_prompt(
            agent_spec.target_cluster or agent_spec.description,
            agent_spec.domain,
        )

    def _infer_tools(self, domain: str) -> list:
        """Infer sensible default tools for a domain."""
        tool_map = {
            "coding":         ["repo_tool", "test_runner"],
            "research":       ["web_tool", "citation_checker"],
            "verification":   ["test_runner", "citation_checker"],
            "security":       ["repo_tool", "web_tool"],
            "data":           ["repo_tool", "data_tool"],
            "database":       ["repo_tool", "data_tool"],
            "devops":         ["repo_tool", "shell_tool"],
            "api":            ["repo_tool", "web_tool"],
            "testing":        ["repo_tool", "test_runner"],
            "documentation":  ["repo_tool"],
            "refactoring":    ["repo_tool"],
        }
        return tool_map.get(domain, ["repo_tool", "web_tool"])

    def synthesize_agent_spec(
        self,
        cluster_info: Dict[str, Any],
        gap_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Synthesize agent specification from cluster and gap analysis."""
        cluster_id = cluster_info["cluster_id"]
        domain = cluster_info["domain"]
        tools = cluster_info.get("required_tools", self._infer_tools(domain))

        agent_id = f"spawned_{domain}_{cluster_id}"
        name = f"{domain.replace('_', ' ').title()} Specialist"
        description = self._generate_description(cluster_info, gap_analysis)
        tags = [domain, "spawned", f"cluster_{cluster_id}"]
        if gap_analysis["coverage_gap"] > 0.7:
            tags.append("high_coverage")

        return {
            "agent_id": agent_id,
            "name": name,
            "description": description,
            "domain": domain,
            "tools": tools,
            "tags": tags,
            "target_cluster": cluster_id,
            "expected_activation_rate": cluster_info.get("projected_usage", 0.1),
            "spawn_reason": cluster_info.get("spawn_reason", "Cluster analysis"),
            "gap_score": gap_analysis["gap_score"],
            "coverage_gap": gap_analysis["coverage_gap"],
        }

    def _generate_description(
        self,
        cluster_info: Dict[str, Any],
        gap_analysis: Dict[str, Any],
    ) -> str:
        domain = cluster_info["domain"]
        cluster_size = cluster_info.get("cluster_size", 0)
        failure_rate = cluster_info.get("failure_rate", 0.0)
        description = f"Specialist agent for {domain} tasks"
        if cluster_size > 0:
            description += f" (cluster of {cluster_size} similar tasks)"
        if failure_rate > 0.5:
            description += f". Addresses high failure rate ({failure_rate:.0%})"
        if gap_analysis["coverage_gap"] > 0.7:
            description += ". Fills significant coverage gap"
        return description
