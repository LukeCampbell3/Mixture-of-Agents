from argparse import Namespace

import claude_openmythos as launcher
from app.models.local_llm_client import OllamaClient


class FakeServer:
    def __init__(self):
        self.model_name = "openmythos-scaled-medium"
        self.base_url = "http://localhost:11435"

    def is_running(self):
        return False

    def start(self):
        raise RuntimeError("Docker Desktop is not running")

    def stop(self):
        return None


def test_openmythos_manager_falls_back_when_server_start_fails(monkeypatch):
    monkeypatch.setattr(
        launcher.requests,
        "get",
        lambda url, timeout=2: type("Resp", (), {"status_code": 200})(),
    )

    manager = launcher.OpenMythosManager(
        FakeServer(),
        fallback_model="qwen2.5-coder:1.5b",
        fallback_base_url="http://localhost:11434",
    )

    assert manager.start_ollama_server() is True
    assert manager.using_fallback is True
    assert manager.active_model == "qwen2.5-coder:1.5b"
    assert manager.active_base_url == "http://localhost:11434"


def test_build_runtime_client_uses_fallback_client_when_manager_is_in_fallback_mode():
    args = Namespace(
        model_name="openmythos-scaled-medium",
        host="localhost",
        port=11435,
        fallback_model="qwen2.5-coder:1.5b",
        fallback_base_url="http://localhost:11434",
        strict_openmythos=False,
        show_fallback=False,
    )
    manager = launcher.OpenMythosManager(
        FakeServer(),
        fallback_model=args.fallback_model,
        fallback_base_url=args.fallback_base_url,
    )
    manager.using_fallback = True
    manager.active_model = args.fallback_model
    manager.active_base_url = args.fallback_base_url

    client = launcher.build_runtime_client(args, manager)

    assert isinstance(client, OllamaClient)
    assert client.model == args.fallback_model
    assert client.base_url == args.fallback_base_url
