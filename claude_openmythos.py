#!/usr/bin/env python3
"""Run claude_integrated.py against the trained OpenMythos checkpoint.

This is a test launcher. It keeps claude_integrated.py unchanged, starts the
OpenMythos Docker inference adapter, and points the integrated CLI at that
Ollama-compatible /api/generate endpoint instead of the normal Ollama model.

Examples:
    python claude_openmythos.py --smoke "Say hello"
    python claude_openmythos.py
    python claude_openmythos.py --prompt "Write a tiny Python add function"
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

import claude_integrated as ci
import app.device_profile as device_profile
import app.models.llm_client as llm_client_module
import app.orchestrator as orchestrator_module
from app.models.llm_client import LLMClient
from app.models.local_llm_client import OllamaClient, ChatOllamaClient, SessionMetrics


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "openmythos-scaled-ce-export" / "checkpoint.pt"
DEFAULT_IMAGE = "openmythos-distill:local"
DEFAULT_MODEL_NAME = "openmythos-scaled-medium"
DEFAULT_ROUTER_MODEL = "qwen2.5:0.5b"
DEFAULT_FALLBACK_MODEL = "qwen2.5-coder:1.5b"
DEFAULT_FALLBACK_URL = "http://localhost:11434"
DEFAULT_MAX_TOKENS = 4000
DEFAULT_KEEP_ALIVE_ROUTER = -1          # pin router forever (tiny model)
DEFAULT_KEEP_ALIVE_WORKER = "15m"       # keep worker warm for 15 min
DEFAULT_KEEP_ALIVE_FALLBACK = "10m"     # keep fallback warm for 10 min

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class OpenMythosServer:
    """Owns a local OpenMythos Docker serving process."""

    def __init__(
        self,
        checkpoint: Path,
        image: str,
        host: str,
        port: int,
        model_name: str,
        loops: int,
        device: str,
        start: bool = True,
    ):
        self.checkpoint = checkpoint.resolve()
        self.image = image
        self.host = host
        self.port = port
        self.model_name = model_name
        self.loops = loops
        self.device = device
        self.should_start = start
        self.process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_running(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def start(self) -> bool:
        if self.is_running():
            return True
        if not self.should_start:
            return False
        docker_ready, docker_error = self._docker_ready()
        if not docker_ready:
            raise RuntimeError(docker_error)
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"OpenMythos checkpoint not found: {self.checkpoint}\n"
                "Run the scaled CE export first, or pass --checkpoint."
            )

        container_checkpoint = self._container_path(self.checkpoint)
        command = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-p",
            f"127.0.0.1:{self.port}:{self.port}",
            "-v",
            f"{ROOT}:/workspace",
            "--entrypoint",
            "python",
            self.image,
            "-m",
            "distill_openmythos.serve_moa_adapter",
            "--checkpoint",
            container_checkpoint,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--device",
            self.device,
            "--model-name",
            self.model_name,
            "--n-loops",
            str(self.loops),
        ]

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self.process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )

        deadline = time.time() + 90
        last_line = ""
        while time.time() < deadline:
            if self.is_running():
                return True
            if self.process.poll() is not None:
                if self.process.stdout:
                    last_line = self.process.stdout.read()[-1000:]
                raise RuntimeError(
                    "OpenMythos server exited before becoming healthy.\n"
                    f"{last_line}"
                )
            if self.process.stdout:
                line = self.process.stdout.readline()
                if line:
                    last_line = line.strip()
                    print(ci.Color.dim(f"  [openmythos] {last_line}"))
            time.sleep(0.5)

        raise TimeoutError(
            f"OpenMythos server did not become healthy at {self.base_url}/health. "
            f"Last output: {last_line}"
        )

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()

    @staticmethod
    def _container_path(path: Path) -> str:
        try:
            rel = path.resolve().relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(
                f"Checkpoint must live under the repo so Docker can see it: {path}"
            ) from exc
        return "/workspace/" + rel.as_posix()

    @staticmethod
    def _docker_ready() -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except FileNotFoundError:
            return False, (
                "Docker is not installed or not on PATH, so the OpenMythos server "
                "cannot be launched from the local image."
            )
        except Exception as exc:
            return False, f"Unable to probe Docker before starting OpenMythos: {exc}"

        if result.returncode == 0:
            return True, ""

        detail = (result.stderr or result.stdout or "").strip()
        if "dockerDesktopLinuxEngine" in detail or "The system cannot find the file specified" in detail:
            return False, (
                "Docker Desktop is not running, so the OpenMythos server cannot be "
                "started from the local image."
            )
        if detail:
            detail = detail[-400:]
            return False, f"Docker is installed but not ready for OpenMythos startup.\n{detail}"
        return False, "Docker is installed but not ready for OpenMythos startup."


class OpenMythosManager:
    """Drop-in replacement for claude_integrated.OllamaManager."""

    def __init__(
        self,
        server: OpenMythosServer,
        fallback_model: str = DEFAULT_FALLBACK_MODEL,
        fallback_base_url: str = DEFAULT_FALLBACK_URL,
    ):
        self.server = server
        self.fallback_model = fallback_model
        self.fallback_base_url = fallback_base_url.rstrip("/")
        self.active_model = server.model_name
        self.active_base_url = server.base_url
        self.using_fallback = False
        self.startup_warning = ""
        self.ollama_installed = True
        self.ollama_running = False
        self.model_available = False
        self.setup_complete = False
        self.ollama_process = None

    def check_ollama_installed(self) -> bool:
        self.ollama_installed = True
        return True

    def check_ollama_running(self) -> bool:
        self.ollama_running = self.server.is_running() if not self.using_fallback else self._fallback_ready()
        return self.ollama_running

    def start_ollama_server(self) -> bool:
        if self.using_fallback and self._fallback_ready():
            self.ollama_running = True
            return True

        print(ci.Color.yellow("Starting OpenMythos inference server..."))
        try:
            self.ollama_running = self.server.start()
            self.active_model = self.server.model_name
            self.active_base_url = self.server.base_url
            self.using_fallback = False
            self.startup_warning = ""
            if self.ollama_running:
                print(ci.Color.green(f"OpenMythos server is running at {self.server.base_url}"))
            return self.ollama_running
        except Exception as exc:
            if self._fallback_ready():
                self.using_fallback = True
                self.ollama_running = True
                self.model_available = True
                self.setup_complete = True
                self.active_model = self.fallback_model
                self.active_base_url = self.fallback_base_url
                self.startup_warning = str(exc)
                print(
                    ci.Color.yellow(
                        "OpenMythos server could not start; continuing with fallback "
                        f"{self.fallback_model} at {self.fallback_base_url}."
                    )
                )
                print(ci.Color.dim(f"  [openmythos] startup issue: {self.startup_warning}"))
                return True
            raise

    def check_model_available(self, model_name: str) -> bool:
        if self.using_fallback:
            self.model_available = self._fallback_ready()
        else:
            self.model_available = self.server.is_running()
        return self.model_available

    def pull_model(self, model_name: str) -> bool:
        if self.using_fallback:
            print(
                ci.Color.dim(
                    f"Fallback model {self.fallback_model} is expected to be managed by Ollama."
                )
            )
            self.model_available = self._fallback_ready()
            return self.model_available
        print(ci.Color.dim(f"OpenMythos checkpoint is local; no Ollama pull needed for {model_name}."))
        self.model_available = True
        return True

    def setup_ollama(self, model_name: str = DEFAULT_MODEL_NAME) -> bool:
        self.ollama_running = self.start_ollama_server()
        self.model_available = self.ollama_running
        self.setup_complete = self.ollama_running
        return self.setup_complete

    def is_ready(self) -> bool:
        return self._fallback_ready() if self.using_fallback else self.server.is_running()

    def cleanup(self) -> None:
        self.server.stop()

    def _fallback_ready(self) -> bool:
        try:
            response = requests.get(f"{self.fallback_base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False


class QualityGuardedOpenMythosClient(LLMClient):
    """OpenMythos-first client with Ollama-compatible fallback.

    The current checkpoint is useful for validating recurrent adaptation, but
    it can emit byte-fragmented text. This guard preserves the claude_integrated
    response experience while still exercising OpenMythos whenever it produces
    coherent output.

    Now tracks per-session metrics via :class:`SessionMetrics` so you can
    compare strict vs guarded OpenMythos across a whole run.
    """

    def __init__(
        self,
        model: str,
        openmythos_base_url: str,
        fallback_model: str,
        fallback_base_url: str,
        strict_openmythos: bool = False,
        show_fallback: bool = False,
        transport: str = "generate",
        keep_alive_worker: str | int | None = DEFAULT_KEEP_ALIVE_WORKER,
        keep_alive_fallback: str | int | None = DEFAULT_KEEP_ALIVE_FALLBACK,
        session_metrics: SessionMetrics | None = None,
    ):
        self.model = model
        self.fallback_model = fallback_model
        self.strict_openmythos = strict_openmythos
        self.show_fallback = show_fallback
        self._last_metrics: dict[str, Any] = {}
        self._warned_fallback = False
        self.session_metrics = session_metrics or SessionMetrics()

        # Build primary and fallback clients based on transport choice
        if transport == "chat":
            self.primary: LLMClient = ChatOllamaClient(
                model=model,
                base_url=openmythos_base_url,
                keep_alive=keep_alive_worker,
            )
            self.fallback: LLMClient = ChatOllamaClient(
                model=fallback_model,
                base_url=fallback_base_url,
                keep_alive=keep_alive_fallback,
            )
        else:
            self.primary = OllamaClient(
                model=model,
                base_url=openmythos_base_url,
                keep_alive=keep_alive_worker,
            )
            self.fallback = OllamaClient(
                model=fallback_model,
                base_url=fallback_base_url,
                keep_alive=keep_alive_fallback,
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        print_stream: bool = False,
    ) -> str:
        primary_text = ""
        primary_error = None
        try:
            primary_text = self.primary.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                print_stream=False,
            )
        except Exception as exc:
            primary_error = exc

        if self.strict_openmythos:
            if primary_error:
                raise RuntimeError(f"OpenMythos inference failed: {primary_error}") from primary_error
            self._last_metrics = {
                "source": "openmythos",
                **getattr(self.primary, "last_metrics", lambda: {})(),
            }
            self.session_metrics.record(self._last_metrics)
            if print_stream:
                print(primary_text)
            return primary_text

        if primary_error is None and not self._is_fragmented(primary_text, max_tokens=max_tokens):
            self._last_metrics = {
                "source": "openmythos",
                "fallback_used": False,
                **self.primary.last_metrics(),
            }
            self.session_metrics.record(self._last_metrics)
            if print_stream:
                print(primary_text)
            return primary_text

        fallback_reason = (
            f"OpenMythos error: {primary_error}"
            if primary_error
            else "OpenMythos output failed coherence guard"
        )
        if self.show_fallback and not self._warned_fallback:
            print(
                ci.Color.yellow(
                    f"  [openmythos] {fallback_reason}; using {self.fallback_model} for this response."
                )
            )
            self._warned_fallback = True

        fallback_text = self.fallback.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            print_stream=print_stream,
        )
        self._last_metrics = {
            "source": "fallback",
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "openmythos_preview": primary_text[:160],
            **self.fallback.last_metrics(),
        }
        self.session_metrics.record(self._last_metrics)
        return fallback_text

    def generate_structured(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Structured output — delegates to the underlying client's native path."""
        try:
            return self.primary.generate_structured(prompt, schema)
        except Exception:
            if self.strict_openmythos:
                raise
            return self.fallback.generate_structured(prompt, schema)

    def get_model_name(self) -> str:
        if self.strict_openmythos:
            return f"openmythos/{self.model}"
        return f"openmythos-guarded/{self.model}+fallback/{self.fallback_model}"

    def last_metrics(self) -> dict[str, Any]:
        return self._last_metrics

    @staticmethod
    def _is_fragmented(text: str, max_tokens: int) -> bool:
        stripped = text.strip()
        if not stripped:
            return True

        visible = [ch for ch in stripped if not ch.isspace()]
        letters = [ch for ch in visible if ch.isalpha()]
        if len(visible) < 8:
            return max_tokens > 24

        if len(letters) >= 20:
            vowels = sum(1 for ch in letters if ch.lower() in "aeiou")
            vowel_ratio = vowels / len(letters)
            if vowel_ratio < 0.18:
                return True

        if stripped[0] in "|<~_^" and "```" not in stripped:
            return True
        if stripped.startswith("````"):
            return True

        symbols = [
            ch for ch in visible
            if not ch.isalnum() and ch not in "._:-/#()[]{}=,+*'\"`"
        ]
        if len(symbols) / max(1, len(visible)) > 0.18 and "```" not in stripped:
            return True

        whitespace_ratio = sum(1 for ch in text if ch.isspace()) / max(1, len(text))
        if whitespace_ratio > 0.45:
            return True

        words = [word for word in stripped.replace("`", " ").split() if word]
        if len(words) >= 8:
            short_words = sum(1 for word in words if len(word) <= 2)
            if short_words / len(words) > 0.55:
                return True

        if stripped.count("<") > stripped.count(">") + 1:
            return True
        if "�" in stripped:
            return True

        alpha_words = re.findall(r"[A-Za-z]{3,}", stripped.lower())
        if len(alpha_words) >= 4:
            anchors = {
                "the", "and", "for", "that", "with", "this", "you", "are",
                "from", "hello", "openmythos", "python", "implement", "list",
                "node", "class", "function", "return", "self", "none", "prev",
                "next", "value", "head", "tail", "insert", "delete", "remove",
                "append", "method", "example", "would", "use", "can", "will",
                "doubly", "linked",
            }
            anchor_hits = sum(1 for word in alpha_words if word in anchors)
            code_anchor_hits = sum(
                1
                for token in [
                    "def ",
                    "class ",
                    "return ",
                    "self.",
                    "None",
                    "import ",
                    "```python",
                ]
                if token in stripped
            )
            if anchor_hits / len(alpha_words) < 0.15:
                return code_anchor_hits == 0
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch claude_integrated.py with OpenMythos inference."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Use an already-running OpenMythos server instead of starting Docker.",
    )
    parser.add_argument(
        "--smoke",
        help="Send one direct prompt through the guarded OpenMythos client and exit.",
    )
    parser.add_argument(
        "--prompt",
        help="Run one Agentic Network request through OpenMythos and exit.",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL)
    parser.add_argument("--fallback-base-url", default=DEFAULT_FALLBACK_URL)
    parser.add_argument(
        "--router-model",
        default=DEFAULT_ROUTER_MODEL,
        help=(
            "Lightweight model for routing/classification only. "
            "Defaults to qwen2.5:0.5b. Set to the same value as "
            "--model-name to reuse the worker for routing."
        ),
    )
    parser.add_argument(
        "--transport",
        choices=["generate", "chat"],
        default="generate",
        help=(
            "Ollama transport: 'generate' uses /api/generate (prompt-based), "
            "'chat' uses /api/chat (role-aware, supports tool calling)."
        ),
    )
    parser.add_argument(
        "--keep-alive-router",
        default=str(DEFAULT_KEEP_ALIVE_ROUTER),
        help="Ollama keep_alive for the router model. -1 = forever. Default: %(default)s",
    )
    parser.add_argument(
        "--keep-alive-worker",
        default=DEFAULT_KEEP_ALIVE_WORKER,
        help="Ollama keep_alive for the worker model. Default: %(default)s",
    )
    parser.add_argument(
        "--keep-alive-fallback",
        default=DEFAULT_KEEP_ALIVE_FALLBACK,
        help="Ollama keep_alive for the fallback model. Default: %(default)s",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip model warmup on startup (faster launch, slower first request).",
    )
    parser.add_argument(
        "--strict-openmythos",
        action="store_true",
        help="Disable fallback and show raw OpenMythos output even if fragmented.",
    )
    parser.add_argument(
        "--raw-openmythos",
        action="store_true",
        help="Alias for --strict-openmythos, useful for debugging the checkpoint.",
    )
    parser.add_argument(
        "--show-fallback",
        action="store_true",
        help="Print when the coherence guard falls back to the Ollama coder model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = OpenMythosServer(
        checkpoint=args.checkpoint,
        image=args.image,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        loops=args.loops,
        device=args.device,
        start=not args.no_start,
    )
    manager = OpenMythosManager(
        server,
        fallback_model=args.fallback_model,
        fallback_base_url=args.fallback_base_url,
    )
    atexit.register(manager.cleanup)
    if args.raw_openmythos:
        args.strict_openmythos = True

    # Shared session metrics — survives the whole process
    session_metrics = SessionMetrics()

    def _dump_session_metrics():
        summary = session_metrics.summary()
        if summary["total_requests"] > 0:
            print(ci.Color.dim(f"\n  [session] {json.dumps(summary, sort_keys=True)}"))

    atexit.register(_dump_session_metrics)

    patch_claude_integrated(args, manager, session_metrics)

    if args.smoke:
        manager.start_ollama_server()
        client = build_runtime_client(args, manager, session_metrics)
        _warmup_models(args, manager)
        print(client.generate(args.smoke, max_tokens=args.max_tokens, temperature=0.0))
        print(ci.Color.dim(f"metrics: {json.dumps(client.last_metrics(), sort_keys=True)}"))
        return 0

    if args.prompt:
        manager.start_ollama_server()
        _warmup_models(args, manager)
        profile = build_openmythos_profile(args)
        client = ci.AgenticNetworkClient(manager, profile)
        answer, pending = client.process_request(args.prompt, workspace_root=str(ROOT))
        print(answer)
        if pending:
            print(json.dumps([op.__dict__ for op in pending], indent=2, default=str))
        return 0

    print(ci.Color.blue("OpenMythos Test Launcher"))
    print(ci.Color.dim(f"  Model      : {args.model_name}"))
    print(ci.Color.dim(f"  Router     : {args.router_model}"))
    print(ci.Color.dim(f"  Transport  : {args.transport}"))
    print(ci.Color.dim(f"  Checkpoint : {args.checkpoint}"))
    print(ci.Color.dim(f"  Endpoint   : http://{args.host}:{args.port}"))
    if args.strict_openmythos:
        print(ci.Color.yellow("  Quality guard: OFF, raw OpenMythos output enabled."))
    else:
        print(ci.Color.dim(
            f"  Quality guard: ON, fallback={args.fallback_model} at {args.fallback_base_url}"
        ))

    # Warmup models before entering the interactive loop
    _warmup_models(args, manager)

    print(ci.Color.dim("  Entering claude_integrated.py with OpenMythos guarded inference."))
    ci.main()
    return 0


def patch_claude_integrated(
    args: argparse.Namespace,
    manager: OpenMythosManager,
    session_metrics: SessionMetrics | None = None,
) -> None:
    profile = build_openmythos_profile(args)

    def load_or_create_openmythos(force: bool = False) -> dict[str, Any]:
        return profile

    def print_openmythos_summary(profile_payload: dict[str, Any]) -> None:
        print(f"  Runtime worker : {profile_payload['models']['worker']}")
        print(f"  Runtime router : {profile_payload['models']['router']}")
        print(f"  Transport      : {args.transport}")
        print(f"  Endpoint: http://{args.host}:{args.port}")
        print(f"  Recurrent loops: {args.loops}")
        print(f"  Checkpoint: {args.checkpoint}")

    device_profile.load_or_create = load_or_create_openmythos
    device_profile.print_summary = print_openmythos_summary

    class PatchedOpenMythosManager(OpenMythosManager):
        def __init__(self):
            self.__dict__.update(manager.__dict__)

    ci.OllamaManager = PatchedOpenMythosManager
    patch_llm_factory(args, manager, session_metrics)

    active = ci.get_active_config()
    active["llm_provider"] = "ollama"
    active["llm_model"] = args.model_name
    active["llm_base_url"] = f"http://{args.host}:{args.port}"
    active["openmythos_auto_adapt"] = True
    active["max_tokens"] = args.max_tokens
    active["auto_approve_file_ops"] = True
    # Expose router model so the orchestrator can use a separate lightweight model
    active["router_model"] = args.router_model


def patch_llm_factory(
    args: argparse.Namespace,
    manager: OpenMythosManager,
    session_metrics: SessionMetrics | None = None,
) -> None:
    original_factory = llm_client_module.create_llm_client
    openmythos_base_url = f"http://{args.host}:{args.port}"

    def create_openmythos_client(
        provider: str = "openai",
        model: str | None = None,
        base_url: str | None = None,
    ) -> LLMClient:
        if provider == "ollama" and (model == args.model_name or base_url == openmythos_base_url):
            return build_runtime_client(args, manager, session_metrics)
        return original_factory(provider, model, base_url)

    llm_client_module.create_llm_client = create_openmythos_client
    orchestrator_module.create_llm_client = create_openmythos_client
    ci.create_llm_client = create_openmythos_client


def build_guarded_client(
    args: argparse.Namespace,
    session_metrics: SessionMetrics | None = None,
) -> QualityGuardedOpenMythosClient:
    return QualityGuardedOpenMythosClient(
        model=args.model_name,
        openmythos_base_url=f"http://{args.host}:{args.port}",
        fallback_model=args.fallback_model,
        fallback_base_url=args.fallback_base_url,
        strict_openmythos=args.strict_openmythos,
        show_fallback=args.show_fallback,
        transport=args.transport,
        keep_alive_worker=_parse_keep_alive(args.keep_alive_worker),
        keep_alive_fallback=_parse_keep_alive(args.keep_alive_fallback),
        session_metrics=session_metrics,
    )


def build_runtime_client(
    args: argparse.Namespace,
    manager: OpenMythosManager,
    session_metrics: SessionMetrics | None = None,
) -> LLMClient:
    if manager.using_fallback:
        return OllamaClient(
            model=manager.active_model,
            base_url=manager.active_base_url,
            keep_alive=_parse_keep_alive(args.keep_alive_fallback),
        )
    return build_guarded_client(args, session_metrics)


def build_openmythos_profile(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "hardware": {
            "ram_gb": None,
            "cpu_cores": os.cpu_count() or 4,
            "is_laptop": None,
            "platform": sys.platform,
        },
        "models": {
            "worker": args.model_name,
            "router": args.router_model,
            "available": [args.model_name, args.router_model, args.fallback_model],
        },
        "runtime": {
            "enable_parallel": True,
            "max_parallel_agents": 3,
            "budget_mode": "balanced",
            "max_validation_retries": 2,
            "verifier_on_first_pass": True,
            "max_tokens": args.max_tokens,
        },
        "web_search": {
            "enabled": True,
            "trigger_freshness_threshold": 0.45,
            "trigger_ambiguity_threshold": 0.55,
            "max_results": 3,
        },
        "routing": {
            "default_max_agents": 3,
            "mixed_task_max_agents": 3,
        },
        "throughput": {},
        "profile_version": "openmythos-test",
    }


def _parse_keep_alive(value: str | int | None) -> str | int | None:
    """Convert CLI keep_alive string to the right type for Ollama."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except ValueError:
        return value  # e.g. "15m", "1h"


def _warmup_models(args: argparse.Namespace, manager: OpenMythosManager) -> None:
    """Preload router, worker, and fallback models into VRAM/RAM.

    Skipped when --no-warmup is set.  Each model gets a tiny request
    with the configured keep_alive so Ollama pins it in memory.
    """
    if args.no_warmup:
        return

    models_to_warm: list[tuple[str, str, str | int | None]] = []

    # Router (at Ollama, not at the OpenMythos server)
    if args.router_model and args.router_model != args.model_name:
        models_to_warm.append((
            args.router_model,
            args.fallback_base_url,  # router runs on the main Ollama instance
            _parse_keep_alive(args.keep_alive_router),
        ))

    # Primary worker (at the OpenMythos server, or Ollama if fallback)
    if not manager.using_fallback:
        models_to_warm.append((
            args.model_name,
            f"http://{args.host}:{args.port}",
            _parse_keep_alive(args.keep_alive_worker),
        ))

    # Fallback worker
    models_to_warm.append((
        args.fallback_model,
        args.fallback_base_url,
        _parse_keep_alive(args.keep_alive_fallback),
    ))

    for model, base_url, keep_alive in models_to_warm:
        label = model.split(":")[0] if ":" in model else model
        print(ci.Color.dim(f"  [warmup] {label} at {base_url} ..."), end=" ", flush=True)
        client = OllamaClient(model=model, base_url=base_url, keep_alive=keep_alive)
        ok = client.warmup(timeout=60)
        if ok:
            print(ci.Color.green("ready"))
        else:
            print(ci.Color.yellow("skipped (not reachable)"))


if __name__ == "__main__":
    raise SystemExit(main())
