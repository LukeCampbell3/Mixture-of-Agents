"""Skill packs for soft agent specialization without spawning new agents."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class SkillPackType(str, Enum):
    """Types of skill packs."""
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    TOOL_POLICY = "tool_policy"
    PROMPT_VARIANT = "prompt_variant"
    RETRIEVAL_PACK = "retrieval_pack"
    MODE = "mode"


@dataclass
class SkillPack:
    """A skill pack that modifies agent behavior without creating new agents."""
    
    pack_id: str
    name: str
    pack_type: SkillPackType
    description: str
    
    # Prompt modifications
    system_prompt_addition: Optional[str] = None
    task_prompt_template: Optional[str] = None
    
    # Tool policies
    preferred_tools: Optional[List[str]] = None
    tool_constraints: Optional[Dict[str, Any]] = None
    
    # Domain knowledge
    domain_context: Optional[str] = None
    examples: Optional[List[Dict[str, str]]] = None
    
    # Retrieval context
    retrieval_queries: Optional[List[str]] = None
    reference_docs: Optional[List[str]] = None
    
    # Behavioral parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    focus_areas: Optional[List[str]] = None


class SkillPackRegistry:
    """Registry of available skill packs for soft specialization."""
    
    def __init__(self):
        self.packs: Dict[str, SkillPack] = {}
        self._initialize_default_packs()
    
    def _initialize_default_packs(self):
        """Initialize default skill packs."""
        
        # CODING SKILL PACKS
        self.register(SkillPack(
            pack_id="algorithm_optimization",
            name="Algorithm Optimization",
            pack_type=SkillPackType.MODE,
            description="Focus on algorithmic efficiency and complexity analysis",
            system_prompt_addition="""
You are in ALGORITHM OPTIMIZATION mode. Focus on:
- Time and space complexity analysis
- Optimal data structures
- Performance bottlenecks
- Big-O notation and scalability
- Trade-offs between different approaches
""",
            temperature=0.3,
            focus_areas=["complexity", "performance", "optimization"]
        ))
        
        self.register(SkillPack(
            pack_id="code_review",
            name="Code Review",
            pack_type=SkillPackType.MODE,
            description="Critical code review focusing on quality and maintainability",
            system_prompt_addition="""
You are in CODE REVIEW mode. Critically evaluate:
- Code correctness and edge cases
- Readability and maintainability
- Security vulnerabilities
- Best practices and patterns
- Potential bugs or issues
Be constructive but thorough in identifying problems.
""",
            temperature=0.4,
            focus_areas=["quality", "security", "maintainability"]
        ))
        
        self.register(SkillPack(
            pack_id="debugging_mode",
            name="Debugging Mode",
            pack_type=SkillPackType.MODE,
            description="Systematic debugging and error analysis",
            system_prompt_addition="""
You are in DEBUGGING mode. Approach systematically:
- Identify error symptoms and patterns
- Trace execution flow
- Check assumptions and invariants
- Propose hypotheses and tests
- Suggest fixes with explanations
""",
            temperature=0.5,
            focus_areas=["debugging", "error_analysis", "root_cause"]
        ))
        
        # RESEARCH SKILL PACKS
        self.register(SkillPack(
            pack_id="sorting_comparison",
            name="Sorting Algorithm Comparison",
            pack_type=SkillPackType.DOMAIN_KNOWLEDGE,
            description="Specialized knowledge for comparing sorting algorithms",
            domain_context="""
Key sorting algorithms to compare:
- Bubble Sort: O(n²), simple, stable
- Quick Sort: O(n log n) avg, O(n²) worst, in-place
- Merge Sort: O(n log n), stable, requires O(n) space
- Heap Sort: O(n log n), in-place, not stable
- Radix Sort: O(nk), non-comparison, integer-specific
- Tim Sort: O(n log n), hybrid, Python's default

Focus on: time complexity, space complexity, stability, use cases
""",
            focus_areas=["algorithms", "complexity", "comparison"],
            examples=[
                {
                    "question": "When to use Quick Sort vs Merge Sort?",
                    "answer": "Quick Sort for in-place sorting with good average performance. Merge Sort when stability is required or worst-case O(n log n) is critical."
                }
            ]
        ))
        
        self.register(SkillPack(
            pack_id="architecture_comparison",
            name="Architecture Comparison",
            pack_type=SkillPackType.DOMAIN_KNOWLEDGE,
            description="Compare software architecture patterns",
            domain_context="""
Architecture patterns to analyze:
- Monolithic: Single deployment, tight coupling, simple ops
- Microservices: Independent services, loose coupling, complex ops
- Serverless: Event-driven, auto-scaling, vendor lock-in
- Event-driven: Async, decoupled, eventual consistency
- Layered: Separation of concerns, clear boundaries

Focus on: scalability, maintainability, complexity, team structure
""",
            focus_areas=["architecture", "design", "trade-offs"]
        ))
        
        self.register(SkillPack(
            pack_id="fact_checking",
            name="Fact Checking",
            pack_type=SkillPackType.MODE,
            description="Rigorous fact verification and source validation",
            system_prompt_addition="""
You are in FACT CHECKING mode. Be rigorous:
- Verify claims against authoritative sources
- Check for outdated information
- Identify unsupported assertions
- Note confidence levels
- Flag contradictions or uncertainties
Cite sources for all factual claims.
""",
            temperature=0.2,
            focus_areas=["verification", "accuracy", "sources"]
        ))
        
        # REASONING SKILL PACKS
        self.register(SkillPack(
            pack_id="logical_analysis",
            name="Logical Analysis",
            pack_type=SkillPackType.MODE,
            description="Formal logical reasoning and proof",
            system_prompt_addition="""
You are in LOGICAL ANALYSIS mode. Apply formal reasoning:
- Identify premises and conclusions
- Check logical validity
- Detect fallacies
- Use formal logic notation when helpful
- Provide step-by-step proofs
Be precise and rigorous.
""",
            temperature=0.3,
            focus_areas=["logic", "reasoning", "proof"]
        ))
        
        self.register(SkillPack(
            pack_id="quantitative_analysis",
            name="Quantitative Analysis",
            pack_type=SkillPackType.MODE,
            description="Mathematical and statistical analysis",
            system_prompt_addition="""
You are in QUANTITATIVE ANALYSIS mode. Focus on:
- Precise calculations
- Statistical reasoning
- Probability analysis
- Mathematical proofs
- Numerical accuracy
Show all work and verify calculations.
""",
            temperature=0.2,
            focus_areas=["math", "statistics", "calculation"]
        ))
        
        # HYBRID SKILL PACKS
        self.register(SkillPack(
            pack_id="implementation_with_research",
            name="Implementation with Research",
            pack_type=SkillPackType.MODE,
            description="Combine implementation with best practices research",
            system_prompt_addition="""
You are in IMPLEMENTATION WITH RESEARCH mode:
- Research best practices before implementing
- Cite relevant patterns and standards
- Explain design decisions
- Consider real-world examples
- Balance theory and practice
""",
            temperature=0.5,
            focus_areas=["implementation", "research", "best_practices"]
        ))
        
        self.register(SkillPack(
            pack_id="security_focused",
            name="Security Focused",
            pack_type=SkillPackType.MODE,
            description="Security-first analysis and implementation",
            system_prompt_addition="""
You are in SECURITY FOCUSED mode. Prioritize:
- Input validation and sanitization
- Authentication and authorization
- Encryption and data protection
- Common vulnerabilities (OWASP Top 10)
- Secure coding practices
- Threat modeling
Assume adversarial context.
""",
            temperature=0.3,
            focus_areas=["security", "vulnerabilities", "protection"]
        ))
    
    def register(self, pack: SkillPack):
        """Register a skill pack."""
        self.packs[pack.pack_id] = pack
    
    def get(self, pack_id: str) -> Optional[SkillPack]:
        """Get a skill pack by ID."""
        return self.packs.get(pack_id)
    
    def find_packs_for_task(
        self,
        task_type: str,
        keywords: List[str],
        difficulty: str = "medium"
    ) -> List[SkillPack]:
        """Find relevant skill packs for a task."""
        relevant_packs = []
        
        # Match by task type
        if "coding" in task_type.lower():
            if difficulty == "hard":
                relevant_packs.append(self.get("algorithm_optimization"))
            relevant_packs.append(self.get("code_review"))
        
        if "research" in task_type.lower():
            relevant_packs.append(self.get("fact_checking"))
        
        if "reasoning" in task_type.lower() or "planning" in task_type.lower():
            relevant_packs.append(self.get("logical_analysis"))
        
        # Match by keywords
        keyword_lower = [k.lower() for k in keywords]
        
        if any(k in keyword_lower for k in ["sort", "algorithm", "complexity"]):
            pack = self.get("sorting_comparison")
            if pack and pack not in relevant_packs:
                relevant_packs.append(pack)
        
        if any(k in keyword_lower for k in ["architecture", "microservice", "design"]):
            pack = self.get("architecture_comparison")
            if pack and pack not in relevant_packs:
                relevant_packs.append(pack)
        
        if any(k in keyword_lower for k in ["security", "vulnerability", "auth"]):
            pack = self.get("security_focused")
            if pack and pack not in relevant_packs:
                relevant_packs.append(pack)
        
        if any(k in keyword_lower for k in ["debug", "error", "bug"]):
            pack = self.get("debugging_mode")
            if pack and pack not in relevant_packs:
                relevant_packs.append(pack)
        
        if any(k in keyword_lower for k in ["math", "calculate", "probability"]):
            pack = self.get("quantitative_analysis")
            if pack and pack not in relevant_packs:
                relevant_packs.append(pack)
        
        return [p for p in relevant_packs if p is not None]
    
    def get_all_packs(self) -> List[SkillPack]:
        """Get all registered skill packs."""
        return list(self.packs.values())
    
    def get_packs_by_type(self, pack_type: SkillPackType) -> List[SkillPack]:
        """Get all packs of a specific type."""
        return [p for p in self.packs.values() if p.pack_type == pack_type]


# Global registry instance
_skill_pack_registry = None


def get_skill_pack_registry() -> SkillPackRegistry:
    """Get the global skill pack registry."""
    global _skill_pack_registry
    if _skill_pack_registry is None:
        _skill_pack_registry = SkillPackRegistry()
    return _skill_pack_registry
