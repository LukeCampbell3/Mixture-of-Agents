"""Memory manager for persistent memory governance (Phase 6)."""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
from app.schemas.memory import MemoryItem, MemoryType, MemoryCandidates


class MemoryManager:
    """Manage persistent memory with admission and freshness policies."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.memory_dir = self.data_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_store: Dict[str, MemoryItem] = {}
        self._load_memories()
        
        # Admission thresholds
        self.min_confidence = 0.7
        self.min_reusability_score = 0.6
    
    def _load_memories(self) -> None:
        """Load memories from disk."""
        memory_file = self.memory_dir / "memories.json"
        if memory_file.exists():
            with open(memory_file, 'r') as f:
                data = json.load(f)
                for item_data in data.get("memories", []):
                    memory = MemoryItem(**item_data)
                    self.memory_store[memory.memory_id] = memory
    
    def _save_memories(self) -> None:
        """Save memories to disk."""
        memory_file = self.memory_dir / "memories.json"
        data = {
            "memories": [m.model_dump() for m in self.memory_store.values()],
            "last_updated": datetime.utcnow().isoformat()
        }
        with open(memory_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def evaluate_admission(
        self,
        candidate: MemoryItem,
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate if memory candidate should be admitted.
        
        Args:
            candidate: Memory item to evaluate
            task_context: Context from task execution
        
        Returns:
            Admission decision with reasoning
        """
        # Check reusability
        reusability_score = self._assess_reusability(candidate, task_context)
        
        # Check if validated
        is_validated = self._check_validation(candidate, task_context)
        
        # Check if ephemeral
        is_ephemeral = self._check_ephemeral(candidate)
        
        # Check for contradictions
        has_contradictions = self._check_contradictions(candidate)
        
        # Check usefulness beyond current episode
        future_usefulness = self._assess_future_usefulness(candidate)
        
        # Admission decision
        should_admit = (
            reusability_score >= self.min_reusability_score and
            candidate.confidence >= self.min_confidence and
            is_validated and
            not is_ephemeral and
            not has_contradictions and
            future_usefulness > 0.5
        )
        
        return {
            "admit": should_admit,
            "reusability_score": reusability_score,
            "is_validated": is_validated,
            "is_ephemeral": is_ephemeral,
            "has_contradictions": has_contradictions,
            "future_usefulness": future_usefulness,
            "reason": self._generate_admission_reason(
                should_admit,
                reusability_score,
                is_validated,
                is_ephemeral,
                has_contradictions
            )
        }
    
    def admit_memory(self, memory: MemoryItem) -> None:
        """Admit memory to persistent store."""
        self.memory_store[memory.memory_id] = memory
        self._save_memories()
    
    def retrieve_relevant_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        max_results: int = 5
    ) -> List[MemoryItem]:
        """Retrieve relevant memories for a query.
        
        Args:
            query: Query string
            memory_type: Filter by memory type
            max_results: Maximum number of results
        
        Returns:
            List of relevant memories
        """
        # Filter by type if specified
        candidates = [
            m for m in self.memory_store.values()
            if memory_type is None or m.memory_type == memory_type
        ]
        
        # Filter by freshness
        candidates = [m for m in candidates if self._is_fresh(m)]
        
        # Score by relevance (simplified - would use embeddings)
        query_lower = query.lower()
        scored = []
        for memory in candidates:
            # Simple keyword matching
            content_lower = memory.content.lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)
            score = score / max(len(query_lower.split()), 1)
            scored.append((memory, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [m for m, _ in scored[:max_results]]
    
    def invalidate_stale_memories(self) -> List[str]:
        """Invalidate memories that are no longer fresh.
        
        Returns:
            List of invalidated memory IDs
        """
        invalidated = []
        
        for memory_id, memory in list(self.memory_store.items()):
            if not self._is_fresh(memory):
                # Mark as invalidated or remove
                del self.memory_store[memory_id]
                invalidated.append(memory_id)
        
        if invalidated:
            self._save_memories()
        
        return invalidated
    
    def _assess_reusability(
        self,
        candidate: MemoryItem,
        task_context: Dict[str, Any]
    ) -> float:
        """Assess how reusable the memory is."""
        # Factors affecting reusability
        factors = []
        
        # Procedural knowledge is highly reusable
        if candidate.memory_type == MemoryType.PROCEDURAL:
            factors.append(0.9)
        
        # Code patterns are reusable
        if candidate.memory_type == MemoryType.CODE_PATTERN:
            factors.append(0.8)
        
        # User preferences are reusable
        if candidate.memory_type == MemoryType.USER_PREFERENCE:
            factors.append(0.9)
        
        # Task-specific memories less reusable
        if "task_id" in candidate.content.lower():
            factors.append(0.3)
        
        # General statements more reusable
        if any(word in candidate.content.lower() for word in ["always", "generally", "typically"]):
            factors.append(0.7)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def _check_validation(
        self,
        candidate: MemoryItem,
        task_context: Dict[str, Any]
    ) -> bool:
        """Check if memory is validated."""
        # Check confidence threshold
        if candidate.confidence < self.min_confidence:
            return False
        
        # Check if from validated source
        validated_sources = ["validation_report", "test_result", "verified_output"]
        if any(source in candidate.source_type for source in validated_sources):
            return True
        
        # Check task validation state
        validation_state = task_context.get("validation_state", "unknown")
        if validation_state in ["success", "partial_success"]:
            return True
        
        return False
    
    def _check_ephemeral(self, candidate: MemoryItem) -> bool:
        """Check if memory is ephemeral (temporary)."""
        ephemeral_markers = [
            "temporary", "for now", "currently", "at the moment",
            "task_id", "session", "this time"
        ]
        
        content_lower = candidate.content.lower()
        return any(marker in content_lower for marker in ephemeral_markers)
    
    def _check_contradictions(self, candidate: MemoryItem) -> bool:
        """Check if memory contradicts existing memories."""
        # Simplified contradiction detection
        # Would need more sophisticated NLI in production
        
        for existing in self.memory_store.values():
            if existing.memory_type != candidate.memory_type:
                continue
            
            # Check for explicit contradictions
            contradiction_markers = ["not", "never", "incorrect", "wrong", "false"]
            
            candidate_lower = candidate.content.lower()
            existing_lower = existing.content.lower()
            
            # If both mention similar topics but with contradiction markers
            common_words = set(candidate_lower.split()) & set(existing_lower.split())
            if len(common_words) > 3:
                if any(marker in candidate_lower for marker in contradiction_markers):
                    return True
        
        return False
    
    def _assess_future_usefulness(self, candidate: MemoryItem) -> float:
        """Assess usefulness beyond current episode."""
        # Procedural knowledge is useful
        if candidate.memory_type == MemoryType.PROCEDURAL:
            return 0.9
        
        # Code patterns are useful
        if candidate.memory_type == MemoryType.CODE_PATTERN:
            return 0.8
        
        # Agent performance data is useful
        if candidate.memory_type == MemoryType.AGENT_PERFORMANCE:
            return 0.7
        
        # Task clusters are useful
        if candidate.memory_type == MemoryType.TASK_CLUSTER:
            return 0.7
        
        # User preferences are useful
        if candidate.memory_type == MemoryType.USER_PREFERENCE:
            return 0.9
        
        return 0.5
    
    def _is_fresh(self, memory: MemoryItem) -> bool:
        """Check if memory is still fresh."""
        if memory.freshness_horizon_days is None:
            return True  # No expiration
        
        created_at = datetime.fromisoformat(memory.created_at)
        age_days = (datetime.utcnow() - created_at).days
        
        return age_days <= memory.freshness_horizon_days
    
    def _generate_admission_reason(
        self,
        should_admit: bool,
        reusability_score: float,
        is_validated: bool,
        is_ephemeral: bool,
        has_contradictions: bool
    ) -> str:
        """Generate human-readable admission reason."""
        if should_admit:
            return f"Admitted: reusability {reusability_score:.2f}, validated, non-ephemeral"
        
        reasons = []
        if reusability_score < self.min_reusability_score:
            reasons.append(f"low reusability ({reusability_score:.2f})")
        if not is_validated:
            reasons.append("not validated")
        if is_ephemeral:
            reasons.append("ephemeral content")
        if has_contradictions:
            reasons.append("contradicts existing memories")
        
        return f"Rejected: {', '.join(reasons)}"
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory store."""
        by_type = {}
        for memory in self.memory_store.values():
            memory_type = memory.memory_type.value
            by_type[memory_type] = by_type.get(memory_type, 0) + 1
        
        # Count fresh vs stale
        fresh_count = sum(1 for m in self.memory_store.values() if self._is_fresh(m))
        stale_count = len(self.memory_store) - fresh_count
        
        return {
            "total_memories": len(self.memory_store),
            "by_type": by_type,
            "fresh": fresh_count,
            "stale": stale_count,
            "avg_confidence": sum(m.confidence for m in self.memory_store.values()) / len(self.memory_store) if self.memory_store else 0.0
        }
