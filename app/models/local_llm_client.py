"""Local LLM client for running models like Qwen2.5 locally."""

import json
import time
import requests
from typing import Optional, Dict, Any, Iterator, List
from app.models.llm_client import LLMClient


class OllamaClient(LLMClient):
    """Client for Ollama local LLM server — uses /api/generate with streaming.

    This is the prompt-based transport.  For role-aware multi-turn
    conversations and tool calling, use :class:`ChatOllamaClient` instead.
    """

    def __init__(
        self,
        model: str = "qwen2.5:latest",
        base_url: str = "http://localhost:11434",
        keep_alive: str | int | None = None,
    ):
        self.model = model
        self.base_url = base_url
        # keep_alive: None → Ollama default (5m), -1 → forever, "15m" etc.
        self.keep_alive = keep_alive

    # ------------------------------------------------------------------
    # Internal streaming helper
    # ------------------------------------------------------------------

    def _stream_chunks(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        format_spec: dict | str | None = None,
    ) -> Iterator[dict]:
        """Yield raw Ollama /api/generate stream chunks."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if format_spec is not None:
            payload["format"] = format_spec
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
        """Generate structured output using Ollama's native ``format`` field.

        Falls back to prompt-based JSON extraction when the native path
        returns unparseable output (e.g. older Ollama versions).
        """
        schema_str = json.dumps(schema, indent=2)
        full_prompt = (
            f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema_str}\n\nJSON:"
        )

        # ── Try native structured output first ──────────────────────────
        try:
            parts: list[str] = []
            for chunk in self._stream_chunks(
                full_prompt, max_tokens=1200, temperature=0.2,
                format_spec=schema,
            ):
                token = chunk.get("response", "")
                if token:
                    parts.append(token)
                if chunk.get("done"):
                    break
            native_text = "".join(parts)
            return json.loads(native_text)
        except (json.JSONDecodeError, requests.exceptions.RequestException):
            pass  # fall through to legacy path

        # ── Legacy: prompt-only extraction ───────────────────────────────
        response = self.generate(full_prompt, max_tokens=1000, temperature=0.3)
        return _extract_json(response)

    def get_model_name(self) -> str:
        return f"ollama/{self.model}"

    def last_metrics(self) -> dict:
        """Return timing metrics from the most recent generate() call."""
        return getattr(self, "_last_metrics", {})

    # ------------------------------------------------------------------
    # Warmup / preload
    # ------------------------------------------------------------------

    def warmup(self, timeout: float = 30) -> bool:
        """Send a tiny request to preload the model into VRAM/RAM.

        Returns True if the model responded, False on timeout or error.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": "hi",
            "stream": False,
            "options": {"num_predict": 1, "temperature": 0.0},
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False


# ──────────────────────────────────────────────────────────────────────────────
# Chat-native Ollama client  (/api/chat)
# ──────────────────────────────────────────────────────────────────────────────

class ChatOllamaClient(LLMClient):
    """Role-aware Ollama client using the /api/chat endpoint.

    Advantages over the prompt-based :class:`OllamaClient`:
    * Preserves message roles (system / user / assistant / tool).
    * Supports Ollama-native tool calling.
    * Better multi-turn state for agentic workflows.

    The ``generate()`` method accepts a plain prompt string for backward
    compatibility (wraps it in a single user message).  For full control,
    use ``chat()`` directly.
    """

    def __init__(
        self,
        model: str = "qwen2.5:latest",
        base_url: str = "http://localhost:11434",
        keep_alive: str | int | None = None,
        system_prompt: str | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.keep_alive = keep_alive
        self.system_prompt = system_prompt
        self._last_metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Core chat transport
    # ------------------------------------------------------------------

    def _stream_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: List[Dict[str, Any]] | None = None,
        format_spec: dict | str | None = None,
    ) -> Iterator[dict]:
        """Yield raw chunks from Ollama /api/chat."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if tools:
            payload["tools"] = tools
        if format_spec is not None:
            payload["format"] = format_spec

        read_timeout = max(30, max_tokens * 0.25)
        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            stream=True,
            timeout=(10, read_timeout),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    yield json.loads(line)

    # ------------------------------------------------------------------
    # High-level chat method
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        tools: List[Dict[str, Any]] | None = None,
        print_stream: bool = False,
    ) -> Dict[str, Any]:
        """Send a multi-turn conversation and return the assistant reply.

        Returns a dict with keys:
        * ``content``   – the assistant's text reply
        * ``tool_calls`` – list of tool-call dicts (may be empty)
        * ``metrics``   – timing information
        """
        if self.system_prompt:
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": self.system_prompt}] + messages

        t_start = time.perf_counter()
        t_first_token: Optional[float] = None
        parts: list[str] = []
        tool_calls: list[dict] = []
        eval_count = 0

        try:
            for chunk in self._stream_chat(messages, max_tokens, temperature, tools):
                msg = chunk.get("message", {})
                token = msg.get("content", "")
                if token:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    parts.append(token)
                    if print_stream:
                        print(token, end="", flush=True)
                # Collect tool calls from the message
                if msg.get("tool_calls"):
                    tool_calls.extend(msg["tool_calls"])
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", len(parts))
                    break
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama chat API error: {e}")

        if print_stream:
            print()

        t_end = time.perf_counter()
        ttft = (t_first_token - t_start) if t_first_token else None
        gen_s = (t_end - t_first_token) if t_first_token else (t_end - t_start)
        tps = eval_count / gen_s if gen_s > 0 else 0

        self._last_metrics = {
            "ttft_s":      round(ttft, 3) if ttft else None,
            "total_s":     round(t_end - t_start, 3),
            "tokens":      eval_count,
            "tok_per_sec": round(tps, 1),
        }

        return {
            "content": "".join(parts),
            "tool_calls": tool_calls,
            "metrics": self._last_metrics,
        }

    # ------------------------------------------------------------------
    # LLMClient interface (backward-compatible prompt-based)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        print_stream: bool = False,
    ) -> str:
        """Wrap a plain prompt in a user message and call chat()."""
        result = self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            print_stream=print_stream,
        )
        return result["content"]

    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Structured output via /api/chat with native ``format`` field."""
        schema_str = json.dumps(schema, indent=2)
        full_prompt = (
            f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema_str}\n\nJSON:"
        )
        messages = [{"role": "user", "content": full_prompt}]

        # ── Try native structured output ─────────────────────────────────
        try:
            parts: list[str] = []
            for chunk in self._stream_chat(
                messages, max_tokens=1200, temperature=0.2,
                format_spec=schema,
            ):
                token = chunk.get("message", {}).get("content", "")
                if token:
                    parts.append(token)
                if chunk.get("done"):
                    break
            return json.loads("".join(parts))
        except (json.JSONDecodeError, requests.exceptions.RequestException):
            pass

        # ── Fallback: prompt-only extraction ─────────────────────────────
        response = self.generate(full_prompt, max_tokens=1000, temperature=0.3)
        return _extract_json(response)

    def get_model_name(self) -> str:
        return f"ollama-chat/{self.model}"

    def last_metrics(self) -> dict:
        return self._last_metrics

    # ------------------------------------------------------------------
    # Warmup / preload
    # ------------------------------------------------------------------

    def warmup(self, timeout: float = 30) -> bool:
        """Preload the model into VRAM/RAM via a tiny chat request."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "options": {"num_predict": 1, "temperature": 0.0},
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=timeout,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False


# ──────────────────────────────────────────────────────────────────────────────
# Session metrics tracker
# ──────────────────────────────────────────────────────────────────────────────

class SessionMetrics:
    """Persistent per-session counters for OpenMythos vs fallback tracking.

    Tracks:
    * total requests
    * primary (OpenMythos) successes / failures
    * fallback count + reason histogram
    * average TTFT and tok/sec by source
    * coherence-guard rejection rate
    """

    def __init__(self):
        self.total_requests: int = 0
        self.primary_successes: int = 0
        self.primary_failures: int = 0
        self.fallback_count: int = 0
        self.fallback_reasons: Dict[str, int] = {}
        self._ttft_by_source: Dict[str, list] = {"openmythos": [], "fallback": []}
        self._tps_by_source: Dict[str, list] = {"openmythos": [], "fallback": []}
        self.coherence_rejections: int = 0

    def record(self, metrics: Dict[str, Any]) -> None:
        """Record metrics from a single generate() call."""
        self.total_requests += 1
        source = metrics.get("source", "unknown")

        if source == "openmythos":
            self.primary_successes += 1
        elif source == "fallback":
            self.fallback_count += 1
            reason = metrics.get("fallback_reason", "unknown")
            self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
            if "coherence" in reason.lower():
                self.coherence_rejections += 1
        else:
            self.primary_failures += 1

        ttft = metrics.get("ttft_s")
        tps = metrics.get("tok_per_sec")
        bucket = "openmythos" if source == "openmythos" else "fallback"
        if ttft is not None:
            self._ttft_by_source[bucket].append(ttft)
        if tps is not None:
            self._tps_by_source[bucket].append(tps)

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of the session."""
        def _avg(lst: list) -> float | None:
            return round(sum(lst) / len(lst), 3) if lst else None

        return {
            "total_requests": self.total_requests,
            "primary_successes": self.primary_successes,
            "primary_failures": self.primary_failures,
            "fallback_count": self.fallback_count,
            "fallback_rate": round(
                self.fallback_count / max(1, self.total_requests), 3
            ),
            "fallback_reason_histogram": dict(self.fallback_reasons),
            "coherence_rejection_rate": round(
                self.coherence_rejections / max(1, self.total_requests), 3
            ),
            "avg_ttft_openmythos": _avg(self._ttft_by_source["openmythos"]),
            "avg_ttft_fallback": _avg(self._ttft_by_source["fallback"]),
            "avg_tps_openmythos": _avg(self._tps_by_source["openmythos"]),
            "avg_tps_fallback": _avg(self._tps_by_source["fallback"]),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Shared JSON extraction helper
# ──────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from LLM text output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```json" in text:
        json_str = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        json_str = text.split("```")[1].split("```")[0].strip()
    else:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            json_str = text[start:end]
        else:
            raise ValueError(f"Could not extract JSON from response: {text[:200]}")
    return json.loads(json_str)


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
    base_url: Optional[str] = None,
    keep_alive: str | int | None = None,
) -> LLMClient:
    """Factory function to create local LLM client.
    
    Args:
        backend: Backend type ("ollama", "ollama_chat", or "local")
        model: Model name (optional)
        base_url: Base URL for API (optional)
        keep_alive: Ollama keep_alive value (optional)
    
    Returns:
        LLMClient instance
    """
    if backend == "ollama":
        return OllamaClient(
            model=model or "qwen2.5:latest",
            base_url=base_url or "http://localhost:11434",
            keep_alive=keep_alive,
        )
    elif backend == "ollama_chat":
        return ChatOllamaClient(
            model=model or "qwen2.5:latest",
            base_url=base_url or "http://localhost:11434",
            keep_alive=keep_alive,
        )
    elif backend == "local":
        return LocalLLMClient(
            model=model or "qwen2.5",
            base_url=base_url or "http://localhost:8000"
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
