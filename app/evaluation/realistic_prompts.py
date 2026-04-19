"""Realistic prompt dataset for large-scale validation."""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class PromptQuality(str, Enum):
    """Quality/completeness of prompt."""
    DETAILED = "detailed"           # Clear, specific, well-defined
    NORMAL = "normal"               # Typical user request
    INCOMPLETE = "incomplete"       # Missing key information
    AMBIGUOUS = "ambiguous"         # Multiple interpretations
    VAGUE = "vague"                 # Unclear requirements
    CONFLICTING = "conflicting"     # Contradictory requirements
    MINIMAL = "minimal"             # Bare minimum information


class PromptComplexity(str, Enum):
    """Complexity level of task."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class RealisticPrompt:
    """A realistic user prompt for testing."""
    
    prompt_id: str
    text: str
    category: str  # coding, research, reasoning, mixed
    quality: PromptQuality
    complexity: PromptComplexity
    expected_agents: List[str]
    expected_challenges: List[str]
    success_criteria: str
    notes: str = ""


class RealisticPromptDataset:
    """Large-scale dataset of realistic prompts."""
    
    def __init__(self):
        self.prompts = self._create_prompts()
    
    def _create_prompts(self) -> List[RealisticPrompt]:
        """Create comprehensive prompt dataset."""
        prompts = []
        
        # ============================================================
        # CODING PROMPTS
        # ============================================================
        
        # Detailed coding prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="code_detailed_001",
                text="""Implement a thread-safe LRU cache in Python with the following requirements:
                - O(1) time complexity for get and put operations
                - Support for a configurable maximum size
                - Thread-safe using appropriate locking mechanisms
                - Include comprehensive error handling
                - Add type hints and docstrings
                - Write unit tests covering edge cases (empty cache, capacity 1, concurrent access)
                Please explain your design choices and any trade-offs.""",
                category="coding",
                quality=PromptQuality.DETAILED,
                complexity=PromptComplexity.COMPLEX,
                expected_agents=["code_primary", "critic_verifier"],
                expected_challenges=["complexity_analysis", "thread_safety", "testing"],
                success_criteria="Complete implementation with tests and explanation"
            ),
            
            RealisticPrompt(
                prompt_id="code_detailed_002",
                text="""Create a Python decorator that:
                1. Measures execution time of the decorated function
                2. Logs the function name, arguments, and execution time
                3. Implements retry logic with exponential backoff for exceptions
                4. Has configurable max retries and initial delay
                5. Preserves function metadata using functools.wraps
                Include examples of usage and explain how it works.""",
                category="coding",
                quality=PromptQuality.DETAILED,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["code_primary"],
                expected_challenges=["decorator_pattern", "error_handling"],
                success_criteria="Working decorator with examples"
            ),
        ])
        
        # Normal coding prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="code_normal_001",
                text="Write a function to find the longest palindromic substring in a string. Include time complexity analysis.",
                category="coding",
                quality=PromptQuality.NORMAL,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["code_primary"],
                expected_challenges=["algorithm_design"],
                success_criteria="Working function with complexity analysis"
            ),
            
            RealisticPrompt(
                prompt_id="code_normal_002",
                text="Implement a binary search tree with insert, delete, and search operations. Make it balanced.",
                category="coding",
                quality=PromptQuality.NORMAL,
                complexity=PromptComplexity.COMPLEX,
                expected_agents=["code_primary", "critic_verifier"],
                expected_challenges=["data_structures", "balancing"],
                success_criteria="Complete BST implementation"
            ),
            
            RealisticPrompt(
                prompt_id="code_normal_003",
                text="Create a REST API endpoint in Python using Flask that accepts JSON data, validates it, and stores it in a database.",
                category="coding",
                quality=PromptQuality.NORMAL,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["code_primary", "web_research"],
                expected_challenges=["api_design", "validation"],
                success_criteria="Working API endpoint with validation"
            ),
        ])
        
        # Incomplete coding prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="code_incomplete_001",
                text="Write a sorting function that's fast.",
                category="coding",
                quality=PromptQuality.INCOMPLETE,
                complexity=PromptComplexity.SIMPLE,
                expected_agents=["code_primary"],
                expected_challenges=["ambiguity", "requirements_clarification"],
                success_criteria="Reasonable sorting implementation with assumptions stated",
                notes="Missing: language, input type, stability requirements, space constraints"
            ),
            
            RealisticPrompt(
                prompt_id="code_incomplete_002",
                text="I need a cache. Make it work with my application.",
                category="coding",
                quality=PromptQuality.INCOMPLETE,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["code_primary", "web_research"],
                expected_challenges=["requirements_gathering", "assumptions"],
                success_criteria="Generic cache implementation with stated assumptions",
                notes="Missing: cache type, size, eviction policy, persistence, language"
            ),
            
            RealisticPrompt(
                prompt_id="code_incomplete_003",
                text="Fix the bug in my authentication code.",
                category="coding",
                quality=PromptQuality.INCOMPLETE,
                complexity=PromptComplexity.COMPLEX,
                expected_agents=["code_primary", "critic_verifier"],
                expected_challenges=["missing_context", "debugging_without_code"],
                success_criteria="General debugging advice and common auth issues",
                notes="Missing: actual code, error message, framework, symptoms"
            ),
        ])
        
        # Ambiguous coding prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="code_ambiguous_001",
                text="Build a system that handles user data efficiently and securely.",
                category="coding",
                quality=PromptQuality.AMBIGUOUS,
                complexity=PromptComplexity.VERY_COMPLEX,
                expected_agents=["code_primary", "web_research", "critic_verifier"],
                expected_challenges=["scope_definition", "multiple_interpretations"],
                success_criteria="Clarifying questions and high-level architecture",
                notes="Could mean: database, API, encryption, access control, etc."
            ),
            
            RealisticPrompt(
                prompt_id="code_ambiguous_002",
                text="Make my code faster without breaking anything.",
                category="coding",
                quality=PromptQuality.AMBIGUOUS,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["code_primary"],
                expected_challenges=["no_code_provided", "optimization_without_context"],
                success_criteria="General optimization strategies and profiling advice",
                notes="Missing: code, performance metrics, constraints"
            ),
        ])
        
        # Vague coding prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="code_vague_001",
                text="Help me with my project.",
                category="coding",
                quality=PromptQuality.VAGUE,
                complexity=PromptComplexity.SIMPLE,
                expected_agents=["code_primary"],
                expected_challenges=["extreme_vagueness"],
                success_criteria="Clarifying questions and general guidance",
                notes="Extremely vague - needs significant clarification"
            ),
            
            RealisticPrompt(
                prompt_id="code_vague_002",
                text="Something is wrong with the database.",
                category="coding",
                quality=PromptQuality.VAGUE,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["code_primary", "critic_verifier"],
                expected_challenges=["diagnostic_without_info"],
                success_criteria="Systematic debugging approach and common issues",
                notes="Missing: symptoms, database type, error messages"
            ),
        ])
        
        # Conflicting coding prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="code_conflicting_001",
                text="Create a highly secure authentication system that doesn't require passwords and is completely stateless but remembers users across sessions.",
                category="coding",
                quality=PromptQuality.CONFLICTING,
                complexity=PromptComplexity.COMPLEX,
                expected_agents=["code_primary", "critic_verifier"],
                expected_challenges=["contradictory_requirements"],
                success_criteria="Identify conflicts and propose trade-offs",
                notes="Stateless + remember users = contradiction"
            ),
            
            RealisticPrompt(
                prompt_id="code_conflicting_002",
                text="Build a real-time system with zero latency that processes unlimited data on a single server with no database.",
                category="coding",
                quality=PromptQuality.CONFLICTING,
                complexity=PromptComplexity.VERY_COMPLEX,
                expected_agents=["code_primary", "critic_verifier"],
                expected_challenges=["impossible_requirements"],
                success_criteria="Explain impossibilities and realistic alternatives",
                notes="Multiple physical impossibilities"
            ),
        ])
        
        # Minimal coding prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="code_minimal_001",
                text="fibonacci",
                category="coding",
                quality=PromptQuality.MINIMAL,
                complexity=PromptComplexity.SIMPLE,
                expected_agents=["code_primary"],
                expected_challenges=["minimal_context"],
                success_criteria="Fibonacci implementation with reasonable assumptions",
                notes="Single word - assume implementation request"
            ),
            
            RealisticPrompt(
                prompt_id="code_minimal_002",
                text="api",
                category="coding",
                quality=PromptQuality.MINIMAL,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["code_primary", "web_research"],
                expected_challenges=["extreme_minimal"],
                success_criteria="Clarifying questions or basic API example",
                notes="One word - could mean many things"
            ),
        ])
        
        # ============================================================
        # RESEARCH PROMPTS
        # ============================================================
        
        # Detailed research prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="research_detailed_001",
                text="""Compare and contrast microservices and monolithic architectures across the following dimensions:
                1. Scalability (horizontal and vertical)
                2. Development velocity and team organization
                3. Deployment complexity and DevOps requirements
                4. Fault isolation and resilience
                5. Technology stack flexibility
                6. Performance and latency considerations
                7. Cost implications (infrastructure and operational)
                
                For each dimension, provide concrete examples and cite industry case studies where available. 
                Conclude with decision criteria for choosing between the two approaches.""",
                category="research",
                quality=PromptQuality.DETAILED,
                complexity=PromptComplexity.COMPLEX,
                expected_agents=["web_research", "critic_verifier"],
                expected_challenges=["comprehensive_analysis", "multiple_dimensions"],
                success_criteria="Detailed comparison with examples and decision framework"
            ),
            
            RealisticPrompt(
                prompt_id="research_detailed_002",
                text="""Explain the CAP theorem in distributed systems:
                - Define Consistency, Availability, and Partition tolerance
                - Explain why you can only choose 2 of 3
                - Provide real-world examples of systems that prioritize CP, AP, and CA
                - Discuss how modern systems handle this trade-off (eventual consistency, CRDTs, etc.)
                - Include references to academic papers or authoritative sources""",
                category="research",
                quality=PromptQuality.DETAILED,
                complexity=PromptComplexity.COMPLEX,
                expected_agents=["web_research", "critic_verifier"],
                expected_challenges=["theoretical_concepts", "real_world_examples"],
                success_criteria="Comprehensive explanation with examples and sources"
            ),
        ])
        
        # Normal research prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="research_normal_001",
                text="What are the main differences between SQL and NoSQL databases? When should I use each?",
                category="research",
                quality=PromptQuality.NORMAL,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["web_research"],
                expected_challenges=["comparison", "decision_criteria"],
                success_criteria="Clear comparison with use case guidance"
            ),
            
            RealisticPrompt(
                prompt_id="research_normal_002",
                text="Explain how OAuth 2.0 works and the different grant types.",
                category="research",
                quality=PromptQuality.NORMAL,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["web_research"],
                expected_challenges=["technical_explanation"],
                success_criteria="Clear OAuth explanation with grant type descriptions"
            ),
            
            RealisticPrompt(
                prompt_id="research_normal_003",
                text="What are the pros and cons of using Docker containers vs virtual machines?",
                category="research",
                quality=PromptQuality.NORMAL,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["web_research"],
                expected_challenges=["technology_comparison"],
                success_criteria="Balanced pros/cons analysis"
            ),
        ])
        
        # Incomplete research prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="research_incomplete_001",
                text="Tell me about that new JavaScript framework.",
                category="research",
                quality=PromptQuality.INCOMPLETE,
                complexity=PromptComplexity.SIMPLE,
                expected_agents=["web_research"],
                expected_challenges=["unspecified_subject"],
                success_criteria="Ask for clarification or cover popular frameworks",
                notes="Which framework? React, Vue, Svelte, etc.?"
            ),
            
            RealisticPrompt(
                prompt_id="research_incomplete_002",
                text="How does it compare to the old way?",
                category="research",
                quality=PromptQuality.INCOMPLETE,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["web_research"],
                expected_challenges=["missing_context"],
                success_criteria="Request clarification on what 'it' refers to",
                notes="No context on what is being compared"
            ),
        ])
        
        # Ambiguous research prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="research_ambiguous_001",
                text="What's the best programming language?",
                category="research",
                quality=PromptQuality.AMBIGUOUS,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["web_research", "critic_verifier"],
                expected_challenges=["subjective_question", "context_dependent"],
                success_criteria="Explain context-dependence and provide framework for choosing",
                notes="'Best' depends on use case, team, requirements"
            ),
            
            RealisticPrompt(
                prompt_id="research_ambiguous_002",
                text="Should I use cloud or on-premise?",
                category="research",
                quality=PromptQuality.AMBIGUOUS,
                complexity=PromptComplexity.COMPLEX,
                expected_agents=["web_research", "critic_verifier"],
                expected_challenges=["decision_without_context"],
                success_criteria="Present trade-offs and decision factors",
                notes="Depends on budget, scale, compliance, expertise"
            ),
        ])
        
        # Vague research prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="research_vague_001",
                text="Explain AI.",
                category="research",
                quality=PromptQuality.VAGUE,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["web_research"],
                expected_challenges=["extremely_broad"],
                success_criteria="High-level overview or request for specific focus",
                notes="Extremely broad topic"
            ),
            
            RealisticPrompt(
                prompt_id="research_vague_002",
                text="What about security?",
                category="research",
                quality=PromptQuality.VAGUE,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["web_research", "critic_verifier"],
                expected_challenges=["no_context"],
                success_criteria="General security principles or request for context",
                notes="Security of what? Web, network, application?"
            ),
        ])
        
        # ============================================================
        # REASONING PROMPTS
        # ============================================================
        
        # Detailed reasoning prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="reasoning_detailed_001",
                text="""A company has 200 employees. 
                - 60% work in engineering
                - 40% work in sales
                - 25% work in both engineering and sales
                - 15% work in marketing
                - 10% work in both marketing and engineering
                - 5% work in all three departments
                
                How many employees work in none of these three departments? 
                Show your work step by step and explain your reasoning using set theory.""",
                category="reasoning",
                quality=PromptQuality.DETAILED,
                complexity=PromptComplexity.COMPLEX,
                expected_agents=["critic_verifier"],
                expected_challenges=["set_theory", "overlapping_sets"],
                success_criteria="Correct answer with step-by-step explanation"
            ),
            
            RealisticPrompt(
                prompt_id="reasoning_detailed_002",
                text="""You have 12 balls that look identical. One ball is either heavier or lighter than the others (you don't know which). 
                You have a balance scale and can use it exactly 3 times. 
                Devise a strategy to:
                1. Identify which ball is different
                2. Determine whether it's heavier or lighter
                Explain your strategy step by step and prove it works for all cases.""",
                category="reasoning",
                quality=PromptQuality.DETAILED,
                complexity=PromptComplexity.VERY_COMPLEX,
                expected_agents=["critic_verifier"],
                expected_challenges=["logic_puzzle", "proof"],
                success_criteria="Complete strategy with proof"
            ),
        ])
        
        # Normal reasoning prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="reasoning_normal_001",
                text="If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
                category="reasoning",
                quality=PromptQuality.NORMAL,
                complexity=PromptComplexity.SIMPLE,
                expected_agents=["critic_verifier"],
                expected_challenges=["logical_validity"],
                success_criteria="Correct logical analysis"
            ),
            
            RealisticPrompt(
                prompt_id="reasoning_normal_002",
                text="A train leaves Station A at 60 mph heading to Station B, 180 miles away. Another train leaves Station B at the same time heading to Station A at 40 mph. When do they meet?",
                category="reasoning",
                quality=PromptQuality.NORMAL,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["critic_verifier"],
                expected_challenges=["word_problem"],
                success_criteria="Correct calculation with explanation"
            ),
        ])
        
        # Incomplete reasoning prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="reasoning_incomplete_001",
                text="If A is greater than B, what can we conclude?",
                category="reasoning",
                quality=PromptQuality.INCOMPLETE,
                complexity=PromptComplexity.SIMPLE,
                expected_agents=["critic_verifier"],
                expected_challenges=["insufficient_information"],
                success_criteria="Explain need for more information",
                notes="Need to know what we're trying to conclude"
            ),
        ])
        
        # ============================================================
        # MIXED PROMPTS
        # ============================================================
        
        # Detailed mixed prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="mixed_detailed_001",
                text="""Design and implement a distributed rate limiter system:
                
                Requirements:
                - Support multiple rate limiting algorithms (token bucket, leaky bucket, sliding window)
                - Distributed across multiple servers
                - Handle 100,000+ requests per second
                - Provide per-user and per-API-key rate limiting
                - Include monitoring and alerting
                
                Please:
                1. Explain the trade-offs between different rate limiting algorithms
                2. Design the system architecture (include diagrams if possible)
                3. Implement a basic version in Python
                4. Discuss scalability and failure modes
                5. Suggest how to test it""",
                category="mixed",
                quality=PromptQuality.DETAILED,
                complexity=PromptComplexity.VERY_COMPLEX,
                expected_agents=["code_primary", "web_research", "critic_verifier"],
                expected_challenges=["system_design", "implementation", "analysis"],
                success_criteria="Complete design with implementation and analysis"
            ),
            
            RealisticPrompt(
                prompt_id="mixed_detailed_002",
                text="""Create a Python library for A/B testing:
                - Implement statistical significance testing (chi-square, t-test)
                - Support multiple variants (not just A/B)
                - Include confidence intervals and p-values
                - Provide clear API for defining experiments
                - Add visualization of results
                
                Explain the statistical concepts used and when each test is appropriate.""",
                category="mixed",
                quality=PromptQuality.DETAILED,
                complexity=PromptComplexity.COMPLEX,
                expected_agents=["code_primary", "web_research", "critic_verifier"],
                expected_challenges=["statistics", "implementation", "explanation"],
                success_criteria="Working library with statistical explanation"
            ),
        ])
        
        # Normal mixed prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="mixed_normal_001",
                text="Implement a simple blockchain in Python and explain how it ensures data integrity.",
                category="mixed",
                quality=PromptQuality.NORMAL,
                complexity=PromptComplexity.COMPLEX,
                expected_agents=["code_primary", "web_research"],
                expected_challenges=["implementation", "explanation"],
                success_criteria="Working blockchain with explanation"
            ),
            
            RealisticPrompt(
                prompt_id="mixed_normal_002",
                text="Write a function to validate email addresses according to RFC 5322 and explain the standard.",
                category="mixed",
                quality=PromptQuality.NORMAL,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["code_primary", "web_research"],
                expected_challenges=["regex", "standards"],
                success_criteria="Validation function with standard explanation"
            ),
        ])
        
        # Incomplete mixed prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="mixed_incomplete_001",
                text="Build me a web scraper that works.",
                category="mixed",
                quality=PromptQuality.INCOMPLETE,
                complexity=PromptComplexity.MODERATE,
                expected_agents=["code_primary", "web_research"],
                expected_challenges=["missing_requirements"],
                success_criteria="Generic scraper with stated assumptions",
                notes="Missing: target site, data to extract, frequency, storage"
            ),
        ])
        
        # Ambiguous mixed prompts
        prompts.extend([
            RealisticPrompt(
                prompt_id="mixed_ambiguous_001",
                text="Create something that analyzes data and shows insights.",
                category="mixed",
                quality=PromptQuality.AMBIGUOUS,
                complexity=PromptComplexity.COMPLEX,
                expected_agents=["code_primary", "web_research", "critic_verifier"],
                expected_challenges=["extreme_ambiguity"],
                success_criteria="Clarifying questions and generic example",
                notes="What data? What insights? What format?"
            ),
        ])
        
        # Edge cases and stress tests
        prompts.extend([
            RealisticPrompt(
                prompt_id="edge_empty_001",
                text="",
                category="mixed",
                quality=PromptQuality.MINIMAL,
                complexity=PromptComplexity.TRIVIAL,
                expected_agents=["code_primary"],
                expected_challenges=["empty_input"],
                success_criteria="Handle gracefully with error or prompt for input",
                notes="Empty prompt - edge case"
            ),
            
            RealisticPrompt(
                prompt_id="edge_long_001",
                text="Write a function " + "that does something " * 100 + "and make it work well.",
                category="coding",
                quality=PromptQuality.VAGUE,
                complexity=PromptComplexity.SIMPLE,
                expected_agents=["code_primary"],
                expected_challenges=["repetitive_input"],
                success_criteria="Extract intent despite repetition",
                notes="Extremely repetitive prompt"
            ),
        ])
        
        return prompts
    
    def get_all_prompts(self) -> List[RealisticPrompt]:
        """Get all prompts."""
        return self.prompts
    
    def get_by_quality(self, quality: PromptQuality) -> List[RealisticPrompt]:
        """Get prompts by quality level."""
        return [p for p in self.prompts if p.quality == quality]
    
    def get_by_complexity(self, complexity: PromptComplexity) -> List[RealisticPrompt]:
        """Get prompts by complexity."""
        return [p for p in self.prompts if p.complexity == complexity]
    
    def get_by_category(self, category: str) -> List[RealisticPrompt]:
        """Get prompts by category."""
        return [p for p in self.prompts if p.category == category]
    
    def get_sample(self, n: int = 10, diverse: bool = True) -> List[RealisticPrompt]:
        """Get a sample of prompts.
        
        Args:
            n: Number of prompts to return
            diverse: If True, ensure diversity across quality and complexity
        """
        if not diverse:
            import random
            return random.sample(self.prompts, min(n, len(self.prompts)))
        
        # Ensure diversity
        sample = []
        qualities = list(PromptQuality)
        complexities = list(PromptComplexity)
        
        # Try to get at least one from each quality level
        for quality in qualities:
            candidates = self.get_by_quality(quality)
            if candidates and len(sample) < n:
                import random
                sample.append(random.choice(candidates))
        
        # Fill remaining with random selection
        if len(sample) < n:
            remaining = [p for p in self.prompts if p not in sample]
            import random
            sample.extend(random.sample(remaining, min(n - len(sample), len(remaining))))
        
        return sample[:n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "total_prompts": len(self.prompts),
            "by_quality": {
                quality.value: len(self.get_by_quality(quality))
                for quality in PromptQuality
            },
            "by_complexity": {
                complexity.value: len(self.get_by_complexity(complexity))
                for complexity in PromptComplexity
            },
            "by_category": {
                category: len(self.get_by_category(category))
                for category in ["coding", "research", "reasoning", "mixed"]
            }
        }
