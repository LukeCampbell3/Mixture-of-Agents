"""
OpenMythos reasoning layer — uses the Recurrent-Depth Transformer as a
prompt enrichment pre-processor before Ollama generation.

Correct integration (per README + proj.txt):
  OpenMythos is NOT a standalone text generator.
  It is a reasoning architecture that runs a prompt through recurrent
  depth loops (h_{t+1} = A·h_t + B·e + Transformer(h_t, e)) to produce
  a reasoning-enriched context, which is then passed to Ollama for
  actual text generation.

Pipeline:
  User prompt
      ↓
  [OpenMythos RDT — n_loops iterations of latent reasoning]
      ↓  (produces reasoning summary from hidden states)
  [Ollama Qwen2.5 — generates final text with enriched context]
      ↓
  Response

The n_loops parameter controls reasoning depth:
  - loops=1: minimal reasoning (fast, simple tasks)
  - loops=2: default (trained depth)
  - loops=4: deeper reasoning (harder tasks, more compute)
  - loops=8+: depth extrapolation (emergent capability on novel problems)

Architecture facts (from config.json):
  dim=64, vocab=259 (byte-level), max_seq_len=256, n_loops=2
  GQA attention, MoE FFN (8 experts, top-2), prelude+coda=1 layer each
  Trained: 1000 steps on RTX 4080 SUPER, final loss=0.162
  Checkpoint: OpenMythos/artifacts/openmythos-distill-gpu/checkpoint.pt
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

from app.models.llm_client import LLMClient

ROOT = Path(__file__).parent.parent.parent.resolve()
ARTIFACTS_GPU   = ROOT / "OpenMythos" / "artifacts" / "openmythos-distill-gpu"
ARTIFACTS_POC   = ROOT / "OpenMythos" / "artifacts" / "openmythos-distill-poc"
ARTIFACTS_LARGE = ROOT / "OpenMythos" / "artifacts" / "openmythos-distill-large"


@dataclass
class OpenMythosConfig:
    """Configuration for the OpenMythos reasoning layer."""
    artifact_dir: Path = ARTIFACTS_LARGE   # prefer large loop-aware model
    n_loops: int = 4                       # trained depth for large model
    device: str = "auto"
    ollama_model: str = "qwen2.5:1.5b"
    ollama_base_url: str = "http://localhost:11434"
    max_new_tokens: int = 512
    reasoning_tokens: int = 64


class OpenMythosClient(LLMClient):
    """
    OpenMythos + Ollama hybrid client.

    Uses OpenMythos's recurrent-depth transformer to enrich prompts with
    latent reasoning before passing them to Ollama for text generation.

    The reasoning layer:
    1. Encodes the prompt through the RDT prelude
    2. Runs n_loops recurrent iterations (h_{t+1} = A·h_t + B·e + Transformer)
    3. Extracts a reasoning summary from the final hidden states
    4. Prepends this summary to the original prompt
    5. Sends the enriched prompt to Ollama

    This gives Ollama a "pre-reasoned" context, improving response quality
    on complex tasks without changing the generation model.
    """

    def __init__(self, config: Optional[OpenMythosConfig] = None):
        self.config = config or OpenMythosConfig()
        self._rdt = None          # OpenMythos model
        self._tokenizer = None    # ByteTokenizer
        self._device = None
        self._loaded = False
        self._load_error: str = ""
        self._last_metrics: dict = {}

        self._try_load_rdt()

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        print_stream: bool = False,
    ) -> str:
        t0 = time.perf_counter()

        # Step 1: Run OpenMythos reasoning enrichment (if loaded)
        enriched_prompt = prompt
        reasoning_summary = ""
        if self._loaded:
            reasoning_summary = self._reason(prompt)
            if reasoning_summary:
                enriched_prompt = (
                    f"[Reasoning context: {reasoning_summary}]\n\n{prompt}"
                )

        # Step 2: Generate with Ollama using enriched prompt
        from app.models.local_llm_client import OllamaClient
        ollama = OllamaClient(
            model=self.config.ollama_model,
            base_url=self.config.ollama_base_url,
        )
        response = ollama.generate(
            enriched_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            print_stream=print_stream,
        )

        elapsed = time.perf_counter() - t0
        ollama_metrics = ollama.last_metrics() if hasattr(ollama, "last_metrics") else {}
        self._last_metrics = {
            "total_s":          round(elapsed, 3),
            "tokens":           ollama_metrics.get("tokens", 0),
            "tok_per_sec":      ollama_metrics.get("tok_per_sec", 0),
            "n_loops":          self.config.n_loops,
            "reasoning_loaded": self._loaded,
            "reasoning_chars":  len(reasoning_summary),
            "ollama_model":     self.config.ollama_model,
        }

        return response

    def stream_tokens(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Stream tokens from Ollama with reasoning-enriched prompt."""
        enriched_prompt = prompt
        if self._loaded:
            reasoning_summary = self._reason(prompt)
            if reasoning_summary:
                enriched_prompt = (
                    f"[Reasoning context: {reasoning_summary}]\n\n{prompt}"
                )

        from app.models.local_llm_client import OllamaClient
        ollama = OllamaClient(
            model=self.config.ollama_model,
            base_url=self.config.ollama_base_url,
        )
        yield from ollama.stream_tokens(
            enriched_prompt, max_tokens=max_tokens, temperature=temperature
        )

    def generate_structured(self, prompt: str, schema: Dict) -> Dict:
        schema_str = json.dumps(schema, indent=2)
        full = f"{prompt}\n\nRespond with valid JSON:\n{schema_str}\n\nJSON:"
        response = self.generate(full, max_tokens=512, temperature=0.2)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
            raise ValueError(f"Could not extract JSON: {response[:200]}")

    def get_model_name(self) -> str:
        rdt_status = f"rdt@{self.config.n_loops}loops" if self._loaded else "rdt=off"
        return f"openmythos/{rdt_status}+{self.config.ollama_model}"

    def last_metrics(self) -> dict:
        return self._last_metrics

    # ------------------------------------------------------------------
    # Reasoning layer
    # ------------------------------------------------------------------

    def _reason(self, prompt: str) -> str:
        """
        Run the prompt through OpenMythos's recurrent loops and extract
        a reasoning signal from the hidden state trajectory.

        Uses the *delta* between the initial encoded state and the final
        recurrent state — this captures what the loops added, not the
        raw (noisy) final state.

        Key fix: caps n_loops at max_loop_iters (the trained depth).
        Running beyond the trained depth causes the LoRA adapter to clamp
        to the last trained index, accumulating noise without adding signal.
        """
        try:
            import torch

            # Cap loops at the trained maximum — never extrapolate on this tiny model
            n_loops = min(self.config.n_loops, self._rdt.cfg.max_loop_iters)

            # Tokenize prompt (byte-level, capped at max_seq_len)
            ids = self._tokenizer.encode(prompt, add_special_tokens=True)
            max_len = self._rdt.cfg.max_seq_len
            if len(ids) > max_len:
                ids = ids[:max_len]

            input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
            T = input_ids.shape[1]

            with torch.no_grad():
                # Run through prelude to get encoded representation
                x = self._rdt.embed(input_ids)
                freqs_cis = self._rdt.freqs_cis[:T]
                mask = self._rdt._causal_mask(T, self._device) if T > 1 else None

                for i, layer in enumerate(self._rdt.prelude):
                    x = layer(x, freqs_cis, mask, cache_key=f"prelude_{i}")

                e = x.clone()  # encoded input (baseline)

                # Run recurrent block for n_loops iterations
                h_final = self._rdt.recurrent(
                    x.clone(), e, freqs_cis, mask, n_loops=n_loops
                )

                # Use the DELTA: what did the loops add beyond the initial encoding?
                # This is the actual reasoning signal — the difference between
                # the initial state and the refined state after recurrence.
                delta = h_final - e  # (1, T, dim)

                # Mean-pool across sequence positions
                delta_mean = delta[0].mean(dim=0)  # (dim,)

                # Normalize to unit sphere for stable projection
                delta_norm = delta_mean / (delta_mean.norm() + 1e-8)

                # Project through the output head to get vocabulary signal
                h_normed = self._rdt.norm(delta_norm.unsqueeze(0).unsqueeze(0))
                logits = self._rdt.head(h_normed)[0, 0]  # (vocab_size,)

                # Use top-k tokens by *positive* logit (things the delta emphasizes)
                top_k = min(self.config.reasoning_tokens, logits.shape[0])
                top_ids = logits.topk(top_k).indices.tolist()

                # Decode the top tokens as a reasoning hint
                reasoning_bytes = bytes([
                    max(0, min(255, tid - 3))
                    for tid in top_ids
                    if tid >= 3  # skip special tokens
                ])
                reasoning_text = reasoning_bytes.decode("utf-8", errors="replace")

                # Keep only printable ASCII — the byte tokenizer maps bytes to chars
                clean = "".join(
                    c for c in reasoning_text
                    if c.isprintable() and ord(c) < 128
                ).strip()

                return clean[:200] if clean else ""

        except Exception:
            return ""

    def set_loops(self, n: int):
        """Adjust recurrent depth at runtime."""
        self.config.n_loops = n

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _try_load_rdt(self):
        """Load the OpenMythos RDT checkpoint. Fails gracefully."""
        try:
            import torch
        except ImportError:
            self._load_error = "torch not installed — run: pip install torch"
            return

        # Add OpenMythos to path
        openmythos_root = ROOT / "OpenMythos"
        if str(openmythos_root) not in sys.path:
            sys.path.insert(0, str(openmythos_root))

        try:
            from open_mythos.main import OpenMythos, MythosConfig
            from distill_openmythos.tokenizer import ByteTokenizer
        except ImportError as e:
            self._load_error = f"open-mythos not importable: {e}"
            return

        checkpoint = self.config.artifact_dir / "checkpoint.pt"
        if not checkpoint.exists():
            # Try POC fallback
            checkpoint = ARTIFACTS_POC / "checkpoint.pt"
            if not checkpoint.exists():
                self._load_error = f"No checkpoint at {self.config.artifact_dir}"
                return

        try:
            if self.config.device == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self.config.device)

            from dataclasses import fields
            ckpt = torch.load(checkpoint, map_location=self._device, weights_only=False)
            cfg_keys = {f.name for f in fields(MythosConfig)}
            cfg = MythosConfig(**{k: v for k, v in ckpt["cfg"].items() if k in cfg_keys})

            self._rdt = OpenMythos(cfg).to(self._device)
            self._rdt.load_state_dict(ckpt["model"])
            self._rdt.eval()
            self._tokenizer = ByteTokenizer()
            self._loaded = True

            params = sum(p.numel() for p in self._rdt.parameters())
            print(f"  [openmythos] RDT loaded: dim={cfg.dim} vocab={cfg.vocab_size} "
                  f"loops={cfg.max_loop_iters} params={params:,} device={self._device}")

        except Exception as e:
            self._load_error = str(e)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_openmythos_client(
    artifact: str = "gpu",
    n_loops: int = 2,
    ollama_model: str = "qwen2.5:1.5b",
    device: str = "auto",
) -> OpenMythosClient:
    """
    Create an OpenMythos+Ollama hybrid client.

    Args:
        artifact:      "gpu" (RTX 4080 checkpoint) or "poc" (CPU checkpoint)
        n_loops:       Recurrent reasoning depth (2=default, 4=deeper, 8=extrapolation)
        ollama_model:  Ollama model for text generation
        device:        "auto", "cpu", or "cuda"
    """
    artifact_dir = ARTIFACTS_GPU if artifact == "gpu" else ARTIFACTS_POC
    cfg = OpenMythosConfig(
        artifact_dir=artifact_dir,
        n_loops=n_loops,
        device=device,
        ollama_model=ollama_model,
    )
    return OpenMythosClient(cfg)
