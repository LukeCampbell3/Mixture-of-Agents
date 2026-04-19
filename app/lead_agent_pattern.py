"""Lead agent pattern to prevent negative synergy.

Based on research showing self-organizing multi-agent LLM teams often
underperform their best individual member due to integrative compromise.
https://arxiv.org/abs/2410.02907
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class AgentRole(str, Enum):
    """Agent roles in lead-agent pattern."""
    LEAD = "lead"  # Makes final decision
    EVIDENCE = "evidence"  # Provides evidence/research
    CRITIC = "critic"  # Provides critique/objections
    TESTER = "tester"  # Provides test results


@dataclass
class BoundedOutput:
    """Bounded output from supporting agent."""
    agent_id: str
    role: AgentRole
    content: str
    confidence: float
    evidence_quality: float = 0.0  # For evidence providers
    
    # For critics
    has_objection: bool = False
    objection_claim: Optional[str] = None
    objection_evidence: Optional[str] = None


@dataclass
class DisagreementStructure:
    """Structured disagreement for arbitration."""
    claim: str
    evidence_for: List[str]
    objections: List[Dict[str, str]]  # {agent_id, objection, evidence}
    resolution: Optional[str] = None


class LeadAgentCoordinator:
    """Coordinates agents using lead-agent pattern to prevent negative synergy."""
    
    def __init__(self, max_supporting_agents: int = 2):
        """Initialize coordinator.
        
        Args:
            max_supporting_agents: Max number of supporting agents (2-3 recommended)
        """
        self.max_supporting_agents = max_supporting_agents
    
    def select_lead_agent(
        self,
        task_type: str,
        agent_scores: Dict[str, float],
        calibrated_confidences: Dict[str, float]
    ) -> str:
        """Select lead agent based on task type and calibrated confidence.
        
        Args:
            task_type: Type of task (coding, research, etc.)
            agent_scores: Routing scores for each agent
            calibrated_confidences: Calibrated confidence scores
        
        Returns:
            Lead agent ID
        """
        # Weight by both routing score and calibrated confidence
        weighted_scores = {
            agent_id: score * calibrated_confidences.get(agent_id, 0.5)
            for agent_id, score in agent_scores.items()
        }
        
        return max(weighted_scores, key=weighted_scores.get)
    
    def select_supporting_agents(
        self,
        lead_agent: str,
        available_agents: List[str],
        task_type: str,
        agent_scores: Dict[str, float]
    ) -> Dict[str, AgentRole]:
        """Select supporting agents and assign roles.
        
        Args:
            lead_agent: ID of lead agent
            available_agents: List of available agent IDs
            task_type: Type of task
            agent_scores: Routing scores
        
        Returns:
            Dictionary mapping agent ID to role
        """
        supporting = {}
        
        # Remove lead from available
        candidates = [a for a in available_agents if a != lead_agent]
        
        # Sort by score
        candidates.sort(key=lambda a: agent_scores.get(a, 0), reverse=True)
        
        # Assign roles based on agent type and task
        for agent_id in candidates[:self.max_supporting_agents]:
            if "critic" in agent_id or "verifier" in agent_id:
                supporting[agent_id] = AgentRole.CRITIC
            elif "research" in agent_id or "web" in agent_id:
                supporting[agent_id] = AgentRole.EVIDENCE
            elif "test" in agent_id:
                supporting[agent_id] = AgentRole.TESTER
            else:
                supporting[agent_id] = AgentRole.EVIDENCE
        
        return supporting
    
    def collect_bounded_outputs(
        self,
        lead_output: str,
        supporting_outputs: Dict[str, BoundedOutput]
    ) -> Dict[str, Any]:
        """Collect bounded outputs from supporting agents.
        
        Args:
            lead_output: Output from lead agent
            supporting_outputs: Bounded outputs from supporting agents
        
        Returns:
            Structured collection of outputs
        """
        evidence = []
        critiques = []
        tests = []
        
        for agent_id, output in supporting_outputs.items():
            if output.role == AgentRole.EVIDENCE:
                evidence.append({
                    "agent_id": agent_id,
                    "content": output.content,
                    "quality": output.evidence_quality
                })
            elif output.role == AgentRole.CRITIC:
                if output.has_objection:
                    critiques.append({
                        "agent_id": agent_id,
                        "claim": output.objection_claim,
                        "evidence": output.objection_evidence
                    })
            elif output.role == AgentRole.TESTER:
                tests.append({
                    "agent_id": agent_id,
                    "content": output.content
                })
        
        return {
            "lead_output": lead_output,
            "evidence": evidence,
            "critiques": critiques,
            "tests": tests
        }
    
    def resolve_disagreements(
        self,
        disagreements: List[DisagreementStructure],
        lead_agent: str,
        calibrated_expertise: Dict[str, float],
        evidence_quality: Dict[str, float]
    ) -> List[str]:
        """Resolve disagreements using weighted arbitration.
        
        NOT majority vote - weighted by calibrated expertise and evidence quality.
        
        Args:
            disagreements: List of structured disagreements
            lead_agent: ID of lead agent
            calibrated_expertise: Calibrated expertise scores per agent
            evidence_quality: Evidence quality scores per agent
        
        Returns:
            List of resolutions
        """
        resolutions = []
        
        for disagreement in disagreements:
            # Weight lead agent's position
            lead_weight = calibrated_expertise.get(lead_agent, 0.5)
            
            # Weight objections by expertise and evidence quality
            objection_weights = []
            for obj in disagreement.objections:
                agent_id = obj["agent_id"]
                expertise = calibrated_expertise.get(agent_id, 0.5)
                quality = evidence_quality.get(agent_id, 0.5)
                objection_weights.append(expertise * quality)
            
            # If objections are strong enough, modify lead's position
            if objection_weights and max(objection_weights) > lead_weight:
                # Strongest objection wins
                strongest_idx = objection_weights.index(max(objection_weights))
                resolution = f"Modified based on {disagreement.objections[strongest_idx]['agent_id']}: {disagreement.objections[strongest_idx]['objection']}"
            else:
                # Lead's position stands
                resolution = f"Lead agent's position maintained: {disagreement.claim}"
            
            resolutions.append(resolution)
        
        return resolutions
    
    def synthesize_final_answer(
        self,
        lead_output: str,
        bounded_outputs: Dict[str, Any],
        resolutions: List[str]
    ) -> str:
        """Synthesize final answer with lead agent having primary authority.
        
        Args:
            lead_output: Lead agent's output
            bounded_outputs: Collected bounded outputs
            resolutions: Resolved disagreements
        
        Returns:
            Final synthesized answer
        """
        # Start with lead output
        final = lead_output
        
        # Incorporate high-quality evidence
        high_quality_evidence = [
            e for e in bounded_outputs.get("evidence", [])
            if e["quality"] > 0.7
        ]
        
        if high_quality_evidence:
            final += "\n\nSupporting evidence:\n"
            for e in high_quality_evidence:
                final += f"- {e['content']}\n"
        
        # Incorporate resolved critiques
        if resolutions:
            final += "\n\nResolved considerations:\n"
            for r in resolutions:
                final += f"- {r}\n"
        
        # Incorporate test results
        if bounded_outputs.get("tests"):
            final += "\n\nValidation:\n"
            for t in bounded_outputs["tests"]:
                final += f"- {t['content']}\n"
        
        return final


def prevent_free_form_collaboration(
    task: str,
    available_agents: List[str],
    agent_scores: Dict[str, float],
    calibrated_confidences: Dict[str, float],
    task_type: str
) -> Dict[str, Any]:
    """Prevent free-form collaboration using lead-agent pattern.
    
    This is the main entry point for preventing negative synergy.
    
    Args:
        task: Task description
        available_agents: List of available agent IDs
        agent_scores: Routing scores
        calibrated_confidences: Calibrated confidence scores
        task_type: Type of task
    
    Returns:
        Execution plan with lead and supporting agents
    """
    coordinator = LeadAgentCoordinator(max_supporting_agents=2)
    
    # Select lead agent
    lead = coordinator.select_lead_agent(
        task_type,
        agent_scores,
        calibrated_confidences
    )
    
    # Select supporting agents with roles
    supporting = coordinator.select_supporting_agents(
        lead,
        available_agents,
        task_type,
        agent_scores
    )
    
    return {
        "lead_agent": lead,
        "supporting_agents": supporting,
        "pattern": "lead_agent",
        "max_supporting": coordinator.max_supporting_agents
    }
