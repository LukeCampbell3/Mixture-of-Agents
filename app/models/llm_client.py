"""LLM client for base model inference."""

import os
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured output matching schema."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model identifier."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured output matching schema."""
        import json
        
        schema_str = json.dumps(schema, indent=2)
        full_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema_str}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    def get_model_name(self) -> str:
        """Get model identifier."""
        return f"openai/{self.model}"


class AnthropicClient(LLMClient):
    """Anthropic API client."""
    
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: Optional[str] = None):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        self.model = model
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured output matching schema."""
        import json
        
        schema_str = json.dumps(schema, indent=2)
        full_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema_str}"
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": full_prompt}]
        )
        
        text = response.content[0].text
        # Extract JSON from markdown code blocks if present
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
        except IndexError:
            pass  # Use text as-is
        
        return json.loads(text)
    
    def get_model_name(self) -> str:
        """Get model identifier."""
        return f"anthropic/{self.model}"


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""
    
    def __init__(self, model: str = "mock-model"):
        self.model = model
        self.call_count = 0
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate mock text."""
        self.call_count += 1
        return f"Mock response {self.call_count} for prompt: {prompt[:50]}..."
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock structured output."""
        self.call_count += 1
        # Return a simple mock structure
        return {"result": "mock", "count": self.call_count}
    
    def get_model_name(self) -> str:
        """Get model identifier."""
        return f"mock/{self.model}"


def create_llm_client(provider: str = "openai", model: Optional[str] = None, base_url: Optional[str] = None) -> LLMClient:
    """Factory function to create LLM client."""
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini")
    elif provider == "anthropic":
        return AnthropicClient(model=model or "claude-3-haiku-20240307")
    elif provider == "ollama":
        from app.models.local_llm_client import OllamaClient
        return OllamaClient(model=model or "qwen2.5:latest", base_url=base_url or "http://localhost:11434")
    elif provider == "local":
        from app.models.local_llm_client import LocalLLMClient
        return LocalLLMClient(model=model or "qwen2.5", base_url=base_url or "http://localhost:8000")
    elif provider == "mock":
        return MockLLMClient(model=model or "mock-model")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def auto_select_provider() -> Dict[str, str]:
    """
    Auto-select the fastest available provider for constrained systems.

    Priority order (fastest first):
      1. Anthropic API  — claude-haiku-3-5 (fast, cheap, cloud)
      2. OpenAI API     — gpt-4o-mini (fast, cheap, cloud)
      3. Ollama local   — qwen2.5:1.5b (smallest viable local model)

    Returns a dict with keys: provider, model, base_url (may be None)
    """
    # 1. Anthropic — fastest for coding tasks, no local GPU needed
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key and anthropic_key != "your-anthropic-api-key-here":
        return {
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
            "base_url": None,
        }

    # 2. OpenAI — gpt-4o-mini is very fast and cheap
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and openai_key != "your-openai-api-key-here":
        return {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "base_url": None,
        }

    # 3. Ollama local — use the smallest model that still works well
    try:
        import requests as _req
        resp = _req.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            # Prefer smallest models first for speed on constrained hardware
            for preferred in ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:latest"]:
                if any(preferred in m for m in models):
                    return {
                        "provider": "ollama",
                        "model": preferred,
                        "base_url": "http://localhost:11434",
                    }
            # Fall back to whatever is available
            if models:
                return {
                    "provider": "ollama",
                    "model": models[0],
                    "base_url": "http://localhost:11434",
                }
    except Exception:
        pass

    # Default — will prompt user to configure
    return {
        "provider": "ollama",
        "model": "qwen2.5:1.5b",
        "base_url": "http://localhost:11434",
    }
