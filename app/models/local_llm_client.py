"""Local LLM client for running models like Qwen2.5 locally."""

import json
import requests
from typing import Optional, Dict, Any
from app.models.llm_client import LLMClient


class OllamaClient(LLMClient):
    """Client for Ollama local LLM server."""
    
    def __init__(self, model: str = "qwen2.5:latest", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text from prompt using Ollama."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured output matching schema."""
        schema_str = json.dumps(schema, indent=2)
        full_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema_str}\n\nJSON:"
        
        response = self.generate(full_prompt, max_tokens=2000, temperature=0.3)
        
        # Extract JSON from response
        try:
            # Try to parse directly
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                # Try to find JSON object
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end > start:
                    json_str = response[start:end]
                else:
                    raise ValueError(f"Could not extract JSON from response: {response[:200]}")
            
            return json.loads(json_str)
    
    def get_model_name(self) -> str:
        """Get model identifier."""
        return f"ollama/{self.model}"


class LocalLLMClient(LLMClient):
    """Client for local LLM inference via HTTP API."""
    
    def __init__(
        self,
        model: str = "qwen2.5",
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        url = f"{self.base_url}/v1/completions"
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if "choices" in result:
                return result["choices"][0].get("text", "")
            elif "response" in result:
                return result["response"]
            else:
                return str(result)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Local LLM API error: {e}")
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured output matching schema."""
        schema_str = json.dumps(schema, indent=2)
        full_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema_str}\n\nJSON:"
        
        response = self.generate(full_prompt, max_tokens=2000, temperature=0.3)
        
        # Extract JSON from response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end > start:
                    json_str = response[start:end]
                else:
                    raise ValueError(f"Could not extract JSON from response: {response[:200]}")
            
            return json.loads(json_str)
    
    def get_model_name(self) -> str:
        """Get model identifier."""
        return f"local/{self.model}"


def create_local_llm_client(
    backend: str = "ollama",
    model: Optional[str] = None,
    base_url: Optional[str] = None
) -> LLMClient:
    """Factory function to create local LLM client.
    
    Args:
        backend: Backend type ("ollama" or "local")
        model: Model name (optional)
        base_url: Base URL for API (optional)
    
    Returns:
        LLMClient instance
    """
    if backend == "ollama":
        return OllamaClient(
            model=model or "qwen2.5:latest",
            base_url=base_url or "http://localhost:11434"
        )
    elif backend == "local":
        return LocalLLMClient(
            model=model or "qwen2.5",
            base_url=base_url or "http://localhost:8000"
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
