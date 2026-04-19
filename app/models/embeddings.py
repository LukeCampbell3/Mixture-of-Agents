"""Embedding generation for task and agent similarity."""

import numpy as np
from typing import List, Optional
import hashlib


class EmbeddingGenerator:
    """Generate embeddings for text."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._cache = {}
    
    def _get_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install with: pip install sentence-transformers"
                )
        return self._model
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        # Simple cache based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._cache:
            return self._cache[text_hash]
        
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        self._cache[text_hash] = embedding
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        model = self._get_model()
        return model.encode(texts, convert_to_numpy=True)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def get_version(self) -> str:
        """Get embedding model version."""
        return f"sentence-transformers/{self.model_name}"
