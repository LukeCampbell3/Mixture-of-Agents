"""Embedding generation for task and agent similarity."""

import numpy as np
from typing import List, Optional
import hashlib
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text.
    
    Falls back to a deterministic hash-based embedding when
    sentence-transformers is not installed.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._cache = {}
        self._fallback = False
    
    def _get_model(self):
        """Lazy load the embedding model."""
        if self._model is None and not self._fallback:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Using hash-based fallback embeddings. "
                    "Install with: pip install sentence-transformers"
                )
                self._fallback = True
        return self._model
    
    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic hash-based embedding fallback (384-dim)."""
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._cache:
            return self._cache[text_hash]
        
        model = self._get_model()
        if model is not None:
            embedding = model.encode(text, convert_to_numpy=True)
        else:
            embedding = self._hash_embed(text)
        
        self._cache[text_hash] = embedding
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        model = self._get_model()
        if model is not None:
            return model.encode(texts, convert_to_numpy=True)
        return np.array([self._hash_embed(t) for t in texts])
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def get_version(self) -> str:
        """Get embedding model version."""
        if self._fallback:
            return "hash-fallback/sha256-384"
        return f"sentence-transformers/{self.model_name}"
