"""Local LLM client for running models like Qwen2.5 locally."""

import json
import time
import requests
from typing import Optional, Dict, Any, Iterator
from app.models.llm_client import LLMClient


class OllamaClient(LLMClient):
    """Client for Ollama local LLM server — uses streaming for low latency."""

    def __init__(self, model: str = "qwen2.5:latest", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    # ------------------------------------------------------------------
    # Internal streaming helper
    # ------------------------------------------------------------------

    def _stream_chunks(self, prompt: str, max_tokens: int, temperature: float) -> Iterator[dict]:
        """Yield raw Ollama stream chunks."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        # connect_timeout=10s, read_timeout scales with token budget (≥30s floor)
        read_timeout = max(30, max_tokens * 0.25)
        with requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=(10, read_timeout),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    yield json.loads(line)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        print_stream: bool = False,
    ) -> str:
        """
        Generate text from prompt.

        When print_stream=True the tokens are printed to stdout as they
        arrive (used by the CLI for interactive feel), and the full text
        is still returned as a string.
        """
        t_start = time.perf_counter()
        t_first_token: Optional[float] = None
        parts: list[str] = []
        eval_count = 0

        try:
            for chunk in self._stream_chunks(prompt, max_tokens, temperature):
                token = chunk.get("response", "")
                if token:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    parts.append(token)
                    if print_stream:
                        print(token, end="", flush=True)
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", len(parts))
                    break
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")

        if print_stream:
            print()  # newline after streamed output

        t_end = time.perf_counter()
        ttft  = (t_first_token - t_start) if t_first_token else None
        gen_s = (t_end - t_first_token) if t_first_token else (t_end - t_start)
        tps   = eval_count / gen_s if gen_s > 0 else 0

        # Always emit timing so callers can log it
        self._last_metrics = {
            "ttft_s":      round(ttft, 3) if ttft else None,
            "total_s":     round(t_end - t_start, 3),
            "tokens":      eval_count,
            "tok_per_sec": round(tps, 1),
        }

        return "".join(parts)

    def stream_tokens(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """
        Yield individual text tokens as they arrive from Ollama.

        Use this for streaming execution — callers can parse and act on
        tool calls as soon as each block closes, without waiting for the
        full response.
        """
        try:
            for chunk in self._stream_chunks(prompt, max_tokens, temperature):
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured output matching schema."""
        schema_str = json.dumps(schema, indent=2)
        full_prompt = (
            f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema_str}\n\nJSON:"
        )
        response = self.generate(full_prompt, max_tokens=1000, temperature=0.3)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                start = response.find("{")
                end   = response.rfind("}") + 1
                if start != -1 and end > start:
                    json_str = response[start:end]
                else:
                    raise ValueError(f"Could not extract JSON from response: {response[:200]}")
            return json.loads(json_str)

    def get_model_name(self) -> str:
        return f"ollama/{self.model}"

    def last_metrics(self) -> dict:
        """Return timing metrics from the most recent generate() call."""
        return getattr(self, "_last_metrics", {})


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
