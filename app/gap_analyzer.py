"""Gap analyzer for identifying coverage gaps in agent pool."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from app.schemas.registry import AgentRegistry, AgentSpec
from app.models.embeddings import EmbeddingGenerator


class GapAnalyzer:
    """Analyze gaps in agent pool coverage."""
    
    def __init__(
        self,
        registry: AgentRegistry,
        embedding_generator: EmbeddingGenerator
    ):
        self.registry = registry
        self.embedding_generator = embedding_generator
        
        # Cache agent embeddings
        self.agent_embeddings: Dict[str, np.ndarray] = {}
        self._compute_agent_embeddings()
    
    def _compute_agent_embeddings(self) -> None:
        """Compute embeddings for all agents."""
        for agent in self.registry.get_active_agents():
            # Create agent description for embedding
            description = f"{agent.name}: {agent.description}. Domain: {agent.domain}. Tools: {', '.join(agent.tools)}"
            self.agent_embeddings[agent.agent_id] = self.embedding_generator.embed(description)
    
    def analyze_gap(
        self,
        proposed_domain: str,
        proposed_tools: List[str],
        cluster_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze if proposed agent fills a gap.
        
        Args:
            proposed_domain: Domain of proposed agent
            proposed_tools: Tools for proposed agent
            cluster_info: Information about task cluster
        
        Returns:
            Gap analysis with overlap scores and recommendations
        """
        # Create proposed agent description
        proposed_desc = f"Specialized agent for {proposed_domain}. Tools: {', '.join(proposed_tools)}"
        proposed_embedding = self.embedding_generator.embed(proposed_desc)
        
        # Calculate overlap with existing agents
        overlaps = self._calculate_overlaps(proposed_embedding, proposed_domain, proposed_tools)
        
        # Calculate coverage gap
        coverage_gap = self._calculate_coverage_gap(overlaps)
        
        # Estimate maintenance cost
        maintenance_cost = self._estimate_maintenance_cost(proposed_tools)
        
        # Check for taxonomy fragmentation
        fragmentation_risk = self._assess_fragmentation_risk(proposed_domain, overlaps)
        
        # Calculate expected future usage
        expected_usage = cluster_info.get("projected_usage", 0.0)
        
        # Overall gap score
        gap_score = (
            0.4 * coverage_gap +
            0.3 * expected_usage +
            0.2 * (1.0 - fragmentation_risk) -
            0.1 * maintenance_cost
        )
        
        return {
            "gap_score": max(0.0, min(gap_score, 1.0)),
            "coverage_gap": coverage_gap,
            "max_overlap": max(overlaps.values()) if overlaps else 0.0,
            "overlapping_agents": overlaps,
            "maintenance_cost": maintenance_cost,
            "fragmentation_risk": fragmentation_risk,
            "expected_usage": expected_usage,
            "recommendation": self._generate_recommendation(gap_score, overlaps, fragmentation_risk)
        }
    
    def _calculate_overlaps(
        self,
        proposed_embedding: np.ndarray,
        proposed_domain: str,
        proposed_tools: List[str]
    ) -> Dict[str, float]:
        """Calculate overlap with each existing agent."""
        overlaps = {}
        
        for agent in self.registry.get_active_agents():
            # Semantic overlap (embedding similarity)
            agent_embedding = self.agent_embeddings[agent.agent_id]
            semantic_similarity = float(
                np.dot(proposed_embedding, agent_embedding) /
                (np.linalg.norm(proposed_embedding) * np.linalg.norm(agent_embedding))
            )
            
            # Domain overlap
            domain_overlap = 1.0 if agent.domain == proposed_domain else 0.3
            
            # Tool overlap
            if agent.tools and proposed_tools:
                common_tools = set(agent.tools) & set(proposed_tools)
                tool_overlap = len(common_tools) / len(set(agent.tools) | set(proposed_tools))
            else:
                tool_overlap = 0.0
            
            # Combined overlap
            overlap = (
                0.5 * semantic_similarity +
                0.3 * domain_overlap +
                0.2 * tool_overlap
            )
            
            overlaps[agent.agent_id] = overlap
        
        return overlaps
    
    def _calculate_coverage_gap(self, overlaps: Dict[str, float]) -> float:
        """Calculate how much new coverage the agent provides."""
        if not overlaps:
            return 1.0  # Complete gap if no existing agents
        
        max_overlap = max(overlaps.values())
        
        # Gap is inverse of overlap
        # High overlap = low gap
        # Low overlap = high gap
        return 1.0 - max_overlap
    
    def _estimate_maintenance_cost(self, tools: List[str]) -> float:
        """Estimate maintenance cost (0-1, higher = more costly)."""
        # More tools = higher maintenance
        tool_cost = min(len(tools) / 10.0, 1.0)
        
        # Base cost
        base_cost = 0.2
        
        return base_cost + 0.8 * tool_cost
    
    def _assess_fragmentation_risk(
        self,
        proposed_domain: str,
        overlaps: Dict[str, float]
    ) -> float:
        """Assess risk of taxonomy fragmentation."""
        # Count agents in same domain
        domain_agents = [
            agent for agent in self.registry.get_active_agents()
            if agent.domain == proposed_domain
        ]
        
        # More agents in domain = higher fragmentation risk
        domain_count_risk = min(len(domain_agents) / 5.0, 1.0)
        
        # High overlap with multiple agents = fragmentation risk
        high_overlap_count = sum(1 for overlap in overlaps.values() if overlap > 0.6)
        overlap_risk = min(high_overlap_count / 3.0, 1.0)
        
        return 0.6 * domain_count_risk + 0.4 * overlap_risk
    
    def _generate_recommendation(
        self,
        gap_score: float,
        overlaps: Dict[str, float],
        fragmentation_risk: float
    ) -> str:
        """Generate recommendation based on gap analysis."""
        if gap_score < 0.3:
            return "Not recommended: Low gap score, existing agents provide sufficient coverage"
        
        if fragmentation_risk > 0.7:
            return "Not recommended: High fragmentation risk, consider merging with existing agent"
        
        max_overlap = max(overlaps.values()) if overlaps else 0.0
        if max_overlap > 0.7:
            overlapping_agent = max(overlaps.items(), key=lambda x: x[1])[0]
            return f"Not recommended: High overlap with {overlapping_agent}, consider enhancing existing agent"
        
        if gap_score > 0.6:
            return "Recommended: Fills significant coverage gap with acceptable overlap"
        
        return "Conditional: Moderate gap, monitor cluster growth before spawning"
    
    def suggest_merge_candidates(self, agent_id: str) -> List[Tuple[str, float]]:
        """Suggest agents that could be merged with given agent."""
        agent = self.registry.get_agent(agent_id)
        if not agent or agent_id not in self.agent_embeddings:
            return []
        
        agent_embedding = self.agent_embeddings[agent_id]
        candidates = []
        
        for other_agent in self.registry.get_active_agents():
            if other_agent.agent_id == agent_id:
                continue
            
            # Calculate similarity
            other_embedding = self.agent_embeddings[other_agent.agent_id]
            similarity = float(
                np.dot(agent_embedding, other_embedding) /
                (np.linalg.norm(agent_embedding) * np.linalg.norm(other_embedding))
            )
            
            # High similarity = merge candidate
            if similarity > 0.7:
                candidates.append((other_agent.agent_id, similarity))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates
