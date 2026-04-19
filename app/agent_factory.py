"""Agent factory for creating new agents dynamically (Phase 5)."""

from typing import Dict, Any, Optional
from app.schemas.registry import AgentSpec, LifecycleState
from app.agents.base_agent import BaseAgent
from app.models.llm_client import LLMClient


class DynamicAgent(BaseAgent):
    """Dynamically created agent with custom system prompt."""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        llm_client: LLMClient,
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
        domain: str = "specialized"
    ):
        super().__init__(agent_id, name, description, llm_client, tools)
        self.custom_system_prompt = system_prompt
        self.domain = domain
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        if self.custom_system_prompt:
            return self.custom_system_prompt
        
        # Generate default prompt based on description
        return f"""You are a specialized agent with expertise in {self.domain}.

Your role: {self.description}

Available tools: {', '.join(self.tools) if self.tools else 'None'}

When responding:
1. Focus on your area of specialization
2. Provide specific, actionable recommendations
3. Use available tools when appropriate
4. Acknowledge limitations outside your domain
5. Be concise and precise

Provide your analysis and recommendations."""


class AgentFactory:
    """Factory for creating new agents dynamically."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def create_agent_from_spec(self, spawn_spec: Dict[str, Any]) -> AgentSpec:
        """Create agent specification from spawn recommendation.
        
        Args:
            spawn_spec: Spawn specification from lifecycle manager
        
        Returns:
            AgentSpec ready to be added to registry
        """
        agent_spec = AgentSpec(
            agent_id=spawn_spec["agent_id"],
            name=spawn_spec["name"],
            description=spawn_spec["description"],
            domain=spawn_spec["domain"],
            lifecycle_state=LifecycleState.PROBATIONARY,
            tools=spawn_spec.get("tools", []),
            tags=spawn_spec.get("tags", []),
            target_cluster=spawn_spec.get("target_cluster"),
            expected_activation_rate=spawn_spec.get("expected_activation_rate", 0.1)
        )
        
        return agent_spec
    
    def create_agent_instance(self, agent_spec: AgentSpec) -> DynamicAgent:
        """Create agent instance from specification.
        
        Args:
            agent_spec: Agent specification
        
        Returns:
            DynamicAgent instance ready for execution
        """
        # Generate system prompt
        system_prompt = self._generate_system_prompt(agent_spec)
        
        agent = DynamicAgent(
            agent_id=agent_spec.agent_id,
            name=agent_spec.name,
            description=agent_spec.description,
            llm_client=self.llm_client,
            tools=agent_spec.tools,
            system_prompt=system_prompt,
            domain=agent_spec.domain
        )
        
        return agent
    
    def _generate_system_prompt(self, agent_spec: AgentSpec) -> str:
        """Generate system prompt for agent based on specification."""
        prompt = f"""You are {agent_spec.name}, a specialized agent.

DOMAIN: {agent_spec.domain}

DESCRIPTION: {agent_spec.description}

AVAILABLE TOOLS: {', '.join(agent_spec.tools) if agent_spec.tools else 'None'}

TARGET TASKS: {agent_spec.target_cluster or 'General tasks in your domain'}

YOUR RESPONSIBILITIES:
1. Provide expert analysis in your domain
2. Use available tools effectively
3. Acknowledge when tasks fall outside your expertise
4. Provide specific, actionable recommendations
5. Maintain high quality standards

APPROACH:
- Be thorough but concise
- Focus on your area of specialization
- Cite evidence when making claims
- Acknowledge uncertainties
- Collaborate with other agents when needed

Provide your analysis:"""
        
        return prompt
    
    def synthesize_agent_spec(
        self,
        cluster_info: Dict[str, Any],
        gap_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize agent specification from cluster and gap analysis.
        
        Args:
            cluster_info: Information about task cluster
            gap_analysis: Gap analysis results
        
        Returns:
            Complete agent specification ready for creation
        """
        cluster_id = cluster_info["cluster_id"]
        domain = cluster_info["domain"]
        tools = cluster_info.get("required_tools", [])
        
        # Generate agent ID
        agent_id = f"spawned_{domain}_{cluster_id}"
        
        # Generate name
        name = f"Specialized {domain.replace('_', ' ').title()} Agent"
        
        # Generate description
        description = self._generate_description(cluster_info, gap_analysis)
        
        # Determine tags
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
            "coverage_gap": gap_analysis["coverage_gap"]
        }
    
    def _generate_description(
        self,
        cluster_info: Dict[str, Any],
        gap_analysis: Dict[str, Any]
    ) -> str:
        """Generate agent description from analysis."""
        domain = cluster_info["domain"]
        cluster_size = cluster_info.get("cluster_size", 0)
        failure_rate = cluster_info.get("failure_rate", 0.0)
        
        description = f"Specialized agent for {domain} tasks"
        
        if cluster_size > 0:
            description += f" (handling cluster of {cluster_size} similar tasks)"
        
        if failure_rate > 0.5:
            description += f". Created to address high failure rate ({failure_rate:.1%})"
        
        if gap_analysis["coverage_gap"] > 0.7:
            description += ". Fills significant coverage gap in agent pool"
        
        return description
