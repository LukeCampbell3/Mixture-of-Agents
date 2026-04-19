"""Conflict arbitration protocol for resolving agent disagreements."""

from typing import Dict, Any, List, Optional
from enum import Enum
from app.schemas.run_state import ConflictEvent
from app.models.llm_client import LLMClient


class ConflictType(str, Enum):
    """Types of conflicts between agents."""
    FACTUAL = "factual"
    CODE = "code"
    PLAN = "plan"
    CITATION = "citation"
    VALIDATION = "validation"


class ArbitrationStrategy(str, Enum):
    """Strategies for resolving conflicts."""
    EVIDENCE_BASED = "evidence_based"
    TEST_BASED = "test_based"
    CONSTRAINT_BASED = "constraint_based"
    TRUST_BASED = "trust_based"
    SYNTHESIS = "synthesis"


class Arbitrator:
    """Arbitrate conflicts between agent outputs."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.conflict_history: List[ConflictEvent] = []
    
    def detect_conflicts(
        self,
        agent_outputs: Dict[str, Any],
        task_context: Dict[str, Any]
    ) -> List[ConflictEvent]:
        """Detect conflicts between agent outputs.
        
        Args:
            agent_outputs: Dictionary of agent_id -> output
            task_context: Task context including constraints
        
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        if len(agent_outputs) < 2:
            return conflicts
        
        # Check for explicit contradictions
        conflicts.extend(self._detect_factual_conflicts(agent_outputs))
        
        # Check for code conflicts
        conflicts.extend(self._detect_code_conflicts(agent_outputs))
        
        # Check for plan conflicts
        conflicts.extend(self._detect_plan_conflicts(agent_outputs))
        
        # Check for citation conflicts
        conflicts.extend(self._detect_citation_conflicts(agent_outputs))
        
        return conflicts
    
    def arbitrate(
        self,
        conflict: ConflictEvent,
        agent_outputs: Dict[str, Any],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Arbitrate a conflict and determine resolution.
        
        Args:
            conflict: Conflict to resolve
            agent_outputs: Agent outputs involved in conflict
            task_context: Task context
        
        Returns:
            Arbitration result with resolution and reasoning
        """
        # Select strategy based on conflict type
        strategy = self._select_strategy(conflict.conflict_type)
        
        # Apply strategy
        if strategy == ArbitrationStrategy.EVIDENCE_BASED:
            result = self._arbitrate_by_evidence(conflict, agent_outputs, task_context)
        elif strategy == ArbitrationStrategy.TEST_BASED:
            result = self._arbitrate_by_tests(conflict, agent_outputs, task_context)
        elif strategy == ArbitrationStrategy.CONSTRAINT_BASED:
            result = self._arbitrate_by_constraints(conflict, agent_outputs, task_context)
        elif strategy == ArbitrationStrategy.TRUST_BASED:
            result = self._arbitrate_by_trust(conflict, agent_outputs, task_context)
        else:
            result = self._arbitrate_by_synthesis(conflict, agent_outputs, task_context)
        
        # Update conflict with resolution
        conflict.resolution = result.get("resolution", "unresolved")
        self.conflict_history.append(conflict)
        
        return result
    
    def _detect_factual_conflicts(self, agent_outputs: Dict[str, Any]) -> List[ConflictEvent]:
        """Detect factual contradictions between agents."""
        conflicts = []
        
        # Look for contradiction markers
        contradiction_markers = [
            "however", "but", "contradicts", "disagrees", "conflict",
            "incorrect", "wrong", "actually", "in fact"
        ]
        
        agents = list(agent_outputs.keys())
        for i, agent1 in enumerate(agents):
            output1 = str(agent_outputs[agent1].get("output", "")).lower()
            
            for agent2 in agents[i+1:]:
                output2 = str(agent_outputs[agent2].get("output", "")).lower()
                
                # Check if either output mentions the other or uses contradiction markers
                has_contradiction = any(marker in output1 or marker in output2 
                                       for marker in contradiction_markers)
                
                if has_contradiction:
                    conflicts.append(ConflictEvent(
                        conflict_type=ConflictType.FACTUAL,
                        agents_involved=[agent1, agent2],
                        description=f"Potential factual disagreement between {agent1} and {agent2}"
                    ))
        
        return conflicts
    
    def _detect_code_conflicts(self, agent_outputs: Dict[str, Any]) -> List[ConflictEvent]:
        """Detect code-related conflicts."""
        conflicts = []
        
        # Check if multiple agents provided different code solutions
        code_outputs = {}
        for agent_id, output in agent_outputs.items():
            output_text = str(output.get("output", ""))
            if "```" in output_text or "def " in output_text or "function " in output_text:
                code_outputs[agent_id] = output_text
        
        if len(code_outputs) > 1:
            agents = list(code_outputs.keys())
            conflicts.append(ConflictEvent(
                conflict_type=ConflictType.CODE,
                agents_involved=agents,
                description=f"Multiple code solutions provided by {', '.join(agents)}"
            ))
        
        return conflicts
    
    def _detect_plan_conflicts(self, agent_outputs: Dict[str, Any]) -> List[ConflictEvent]:
        """Detect planning conflicts."""
        conflicts = []
        
        # Check for different approaches or strategies
        plan_markers = ["approach", "strategy", "plan", "steps", "method"]
        
        agents_with_plans = []
        for agent_id, output in agent_outputs.items():
            output_text = str(output.get("output", "")).lower()
            if any(marker in output_text for marker in plan_markers):
                agents_with_plans.append(agent_id)
        
        if len(agents_with_plans) > 1:
            conflicts.append(ConflictEvent(
                conflict_type=ConflictType.PLAN,
                agents_involved=agents_with_plans,
                description=f"Different approaches suggested by {', '.join(agents_with_plans)}"
            ))
        
        return conflicts
    
    def _detect_citation_conflicts(self, agent_outputs: Dict[str, Any]) -> List[ConflictEvent]:
        """Detect citation or source conflicts."""
        conflicts = []
        
        # Check for conflicting sources or citations
        citation_markers = ["source", "according to", "reference", "citation"]
        
        agents_with_citations = []
        for agent_id, output in agent_outputs.items():
            output_text = str(output.get("output", "")).lower()
            if any(marker in output_text for marker in citation_markers):
                agents_with_citations.append(agent_id)
        
        if len(agents_with_citations) > 1:
            # Could have conflicting sources
            conflicts.append(ConflictEvent(
                conflict_type=ConflictType.CITATION,
                agents_involved=agents_with_citations,
                description=f"Multiple sources cited by {', '.join(agents_with_citations)}"
            ))
        
        return conflicts
    
    def _select_strategy(self, conflict_type: str) -> ArbitrationStrategy:
        """Select arbitration strategy based on conflict type."""
        if conflict_type == ConflictType.FACTUAL:
            return ArbitrationStrategy.EVIDENCE_BASED
        elif conflict_type == ConflictType.CODE:
            return ArbitrationStrategy.TEST_BASED
        elif conflict_type == ConflictType.PLAN:
            return ArbitrationStrategy.CONSTRAINT_BASED
        elif conflict_type == ConflictType.CITATION:
            return ArbitrationStrategy.TRUST_BASED
        else:
            return ArbitrationStrategy.SYNTHESIS
    
    def _arbitrate_by_evidence(
        self,
        conflict: ConflictEvent,
        agent_outputs: Dict[str, Any],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Arbitrate based on evidence strength and freshness."""
        # Collect outputs from conflicting agents
        conflicting_outputs = {
            agent_id: agent_outputs[agent_id]
            for agent_id in conflict.agents_involved
            if agent_id in agent_outputs
        }
        
        # Build arbitration prompt
        prompt = f"""You are arbitrating a factual conflict between multiple agents.

CONFLICT: {conflict.description}

AGENT OUTPUTS:
"""
        for agent_id, output in conflicting_outputs.items():
            prompt += f"\n{agent_id}:\n{output.get('output', '')}\n"
        
        prompt += """
Analyze the evidence provided by each agent and determine:
1. Which claims are best supported by evidence
2. Which sources are more authoritative or recent
3. Whether the conflict can be resolved or remains unresolved

Provide your arbitration decision and reasoning."""
        
        decision = self.llm_client.generate(prompt, max_tokens=500, temperature=0.3)
        
        return {
            "strategy": ArbitrationStrategy.EVIDENCE_BASED,
            "resolution": decision,
            "reasoning": "Arbitrated based on evidence strength and source quality"
        }
    
    def _arbitrate_by_tests(
        self,
        conflict: ConflictEvent,
        agent_outputs: Dict[str, Any],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Arbitrate based on test results or code correctness."""
        # For code conflicts, prefer solutions that:
        # 1. Pass tests
        # 2. Have better error handling
        # 3. Are more maintainable
        
        conflicting_outputs = {
            agent_id: agent_outputs[agent_id]
            for agent_id in conflict.agents_involved
            if agent_id in agent_outputs
        }
        
        prompt = f"""You are arbitrating a code conflict between multiple agents.

CONFLICT: {conflict.description}

CODE SOLUTIONS:
"""
        for agent_id, output in conflicting_outputs.items():
            prompt += f"\n{agent_id}:\n{output.get('output', '')}\n"
        
        prompt += """
Evaluate each code solution based on:
1. Correctness and likely test pass rate
2. Error handling and edge cases
3. Code quality and maintainability
4. Performance considerations

Recommend the best solution or suggest a synthesis."""
        
        decision = self.llm_client.generate(prompt, max_tokens=500, temperature=0.3)
        
        return {
            "strategy": ArbitrationStrategy.TEST_BASED,
            "resolution": decision,
            "reasoning": "Arbitrated based on code quality and correctness"
        }
    
    def _arbitrate_by_constraints(
        self,
        conflict: ConflictEvent,
        agent_outputs: Dict[str, Any],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Arbitrate based on constraint satisfaction."""
        task_frame = task_context.get("task_frame")
        constraints = task_frame.hard_constraints if task_frame else []
        
        conflicting_outputs = {
            agent_id: agent_outputs[agent_id]
            for agent_id in conflict.agents_involved
            if agent_id in agent_outputs
        }
        
        prompt = f"""You are arbitrating a planning conflict between multiple agents.

CONFLICT: {conflict.description}

HARD CONSTRAINTS:
{chr(10).join(f"- {c}" for c in constraints) if constraints else "None specified"}

PROPOSED PLANS:
"""
        for agent_id, output in conflicting_outputs.items():
            prompt += f"\n{agent_id}:\n{output.get('output', '')}\n"
        
        prompt += """
Evaluate each plan based on:
1. Satisfaction of hard constraints
2. Risk level and failure modes
3. Feasibility and resource requirements

Recommend the best plan or suggest a hybrid approach."""
        
        decision = self.llm_client.generate(prompt, max_tokens=500, temperature=0.3)
        
        return {
            "strategy": ArbitrationStrategy.CONSTRAINT_BASED,
            "resolution": decision,
            "reasoning": "Arbitrated based on constraint satisfaction and risk"
        }
    
    def _arbitrate_by_trust(
        self,
        conflict: ConflictEvent,
        agent_outputs: Dict[str, Any],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Arbitrate based on source trust and authority."""
        # Prefer higher-trust sources and more authoritative citations
        
        conflicting_outputs = {
            agent_id: agent_outputs[agent_id]
            for agent_id in conflict.agents_involved
            if agent_id in agent_outputs
        }
        
        prompt = f"""You are arbitrating a citation conflict between multiple agents.

CONFLICT: {conflict.description}

CITED SOURCES:
"""
        for agent_id, output in conflicting_outputs.items():
            prompt += f"\n{agent_id}:\n{output.get('output', '')}\n"
        
        prompt += """
Evaluate the sources based on:
1. Authority and credibility
2. Recency and relevance
3. Primary vs secondary sources

Determine which sources should be preferred."""
        
        decision = self.llm_client.generate(prompt, max_tokens=500, temperature=0.3)
        
        return {
            "strategy": ArbitrationStrategy.TRUST_BASED,
            "resolution": decision,
            "reasoning": "Arbitrated based on source authority and trust"
        }
    
    def _arbitrate_by_synthesis(
        self,
        conflict: ConflictEvent,
        agent_outputs: Dict[str, Any],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Arbitrate by synthesizing multiple perspectives."""
        conflicting_outputs = {
            agent_id: agent_outputs[agent_id]
            for agent_id in conflict.agents_involved
            if agent_id in agent_outputs
        }
        
        prompt = f"""You are arbitrating a conflict between multiple agents.

CONFLICT: {conflict.description}

AGENT PERSPECTIVES:
"""
        for agent_id, output in conflicting_outputs.items():
            prompt += f"\n{agent_id}:\n{output.get('output', '')}\n"
        
        prompt += """
Synthesize the different perspectives into a coherent resolution that:
1. Acknowledges valid points from each agent
2. Resolves contradictions where possible
3. Notes remaining uncertainties
4. Provides a balanced recommendation"""
        
        decision = self.llm_client.generate(prompt, max_tokens=500, temperature=0.3)
        
        return {
            "strategy": ArbitrationStrategy.SYNTHESIS,
            "resolution": decision,
            "reasoning": "Arbitrated by synthesizing multiple perspectives"
        }
    
    def get_conflict_summary(self) -> Dict[str, Any]:
        """Get summary of all conflicts and resolutions."""
        return {
            "total_conflicts": len(self.conflict_history),
            "by_type": self._count_by_type(),
            "resolution_rate": self._calculate_resolution_rate(),
            "conflicts": [c.model_dump() for c in self.conflict_history]
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count conflicts by type."""
        counts = {}
        for conflict in self.conflict_history:
            conflict_type = conflict.conflict_type
            counts[conflict_type] = counts.get(conflict_type, 0) + 1
        return counts
    
    def _calculate_resolution_rate(self) -> float:
        """Calculate percentage of conflicts resolved."""
        if not self.conflict_history:
            return 1.0
        
        resolved = sum(1 for c in self.conflict_history if c.resolution and c.resolution != "unresolved")
        return resolved / len(self.conflict_history)
