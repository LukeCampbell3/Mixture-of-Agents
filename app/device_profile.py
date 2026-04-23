"""
Device profiler — runs once at first launch, stores a local hardware profile,
and returns the best deployment configuration for this machine.
"""

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Optional

PROFILE_PATH = Path(os.path.expanduser("~/.claude_device_profile.json"))
OLLAMA_URL   = "http://localhost:11434"

# ── model catalogue ──────────────────────────────────────────────────────────
# (name, size_mb, role)
MODELS = {
    "router":  [("qwen2.5:0.5b",        398),  ("qwen2.5:1.5b",       986)],
    "worker":  [("qwen2.5-coder:1.5b",  986),  ("qwen2.5:1.5b",       986),
                ("qwen2.5:7b",         4700)],
}


# ── hardware detection ────────────────────────────────────────────────────────

def _ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    # Fallback: platform-specific
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                text=True, timeout=5
            )
            for line in out.splitlines():
                line = line.strip()
                if line.isdigit():
                    return int(line) / (1024 ** 3)
        except Exception:
            pass
    return 8.0  # safe default


def _cpu_cores() -> int:
    return os.cpu_count() or 4


def _is_laptop() -> bool:
    """Heuristic: check battery presence."""
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "path", "Win32_Battery", "get", "BatteryStatus"],
                text=True, timeout=5
            )
            return "BatteryStatus" in out and len(out.strip().splitlines()) > 1
        except Exception:
            pass
    elif platform.system() == "Linux":
        return Path("/sys/class/power_supply/BAT0").exists()
    elif platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["pmset", "-g", "batt"], text=True, timeout=5
            )
            return "Battery" in out
        except Exception:
            pass
    return True  # assume laptop when uncertain


def _available_ollama_models() -> list:
    try:
        import requests
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def _measure_throughput(model: str) -> Optional[float]:
    """
    Quick throughput probe: generate 20 tokens, return tok/s.
    Returns None if the model is not available or times out.
    Skipped if the model is not already loaded (avoids cold-load stall).
    """
    try:
        import requests, json as _json
        # Only probe if the model is already running (loaded in memory)
        ps = requests.get(f"{OLLAMA_URL}/api/ps", timeout=3)
        if ps.status_code == 200:
            running = [m.get("name","") for m in ps.json().get("models", [])]
            if not any(model in r for r in running):
                return None   # not loaded — skip probe to avoid stall

        payload = {
            "model": model,
            "prompt": "Hello",
            "stream": True,
            "options": {"num_predict": 20, "temperature": 0.0},
        }
        t0 = time.perf_counter()
        tokens = 0
        with requests.post(f"{OLLAMA_URL}/api/generate",
                           json=payload, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = _json.loads(line)
                    if chunk.get("response"):
                        tokens += 1
                    if chunk.get("done"):
                        tokens = chunk.get("eval_count", tokens)
                        break
        elapsed = time.perf_counter() - t0
        return round(tokens / elapsed, 1) if elapsed > 0 else None
    except Exception:
        return None


# ── profile builder ───────────────────────────────────────────────────────────

def _select_models(ram_gb: float, available: list, laptop: bool = True) -> dict:
    """
    Pick router + worker models based on available RAM and what is pulled.
    Falls back gracefully when preferred models are not yet downloaded.

    On laptops we always cap at 1.5b — the 7b model is too slow on CPU
    regardless of how much RAM is installed.
    """
    def pulled(name):
        return any(name in m for m in available)

    # Worker selection
    if laptop:
        # Laptop: always use the fastest small model available
        if pulled("qwen2.5-coder:1.5b"):
            worker = "qwen2.5-coder:1.5b"
        elif pulled("qwen2.5:1.5b"):
            worker = "qwen2.5:1.5b"
        elif pulled("qwen2.5:0.5b"):
            worker = "qwen2.5:0.5b"
        else:
            worker = "qwen2.5-coder:1.5b"   # will be pulled on first run
    else:
        # Desktop / server: scale with RAM
        if ram_gb >= 12 and pulled("qwen2.5:7b"):
            worker = "qwen2.5:7b"
        elif pulled("qwen2.5-coder:1.5b"):
            worker = "qwen2.5-coder:1.5b"
        elif pulled("qwen2.5:1.5b"):
            worker = "qwen2.5:1.5b"
        else:
            worker = "qwen2.5-coder:1.5b"

    # Router selection — always use the smallest available model
    if pulled("qwen2.5:0.5b"):
        router = "qwen2.5:0.5b"
    elif pulled("qwen2.5:1.5b"):
        router = "qwen2.5:1.5b"
    elif pulled("qwen2.5-coder:1.5b"):
        router = "qwen2.5-coder:1.5b"
    else:
        router = worker   # last resort: reuse worker

    return {"worker": worker, "router": router}


def _build_profile(force: bool = False) -> dict:
    ram    = _ram_gb()
    cores  = _cpu_cores()
    laptop = _is_laptop()
    avail  = _available_ollama_models()
    models = _select_models(ram, avail, laptop=laptop)

    # Parallelism and agent count are no longer capped for laptops.
    # The user controls these via /concurrency or /config set.
    parallel   = cores >= 4
    max_agents = max(2, min(cores // 2, 4))

    # Budget mode scales with RAM only
    if ram < 8:
        budget = "low"
    elif ram < 16:
        budget = "balanced"
    else:
        budget = "thorough"

    profile = {
        "hardware": {
            "ram_gb":       round(ram, 1),
            "cpu_cores":    cores,
            "is_laptop":    laptop,
            "platform":     platform.system(),
        },
        "models": {
            "worker":  models["worker"],
            "router":  models["router"],
            "available": avail,
        },
        "runtime": {
            "enable_parallel":    parallel,
            "max_parallel_agents": max_agents,
            "budget_mode":        budget,
            "max_validation_retries": 2,
            "verifier_on_first_pass": True,
            "max_tokens":         2000,  # per-agent token budget
        },
        "web_search": {
            "enabled":                    True,
            "trigger_freshness_threshold": 0.45,
            "trigger_ambiguity_threshold": 0.55,
            "max_results":                3,
        },
        "routing": {
            "default_max_agents":    max_agents,
            "mixed_task_max_agents": max_agents,
        },
        "throughput": {},
        "profile_version": 2,
    }

    # Quick throughput probe (only if Ollama is reachable and models are warm)
    if avail:
        for role, model in models.items():
            tps = _measure_throughput(model)
            if tps is not None:
                profile["throughput"][model] = tps

    return profile


# ── public API ────────────────────────────────────────────────────────────────

def load_or_create(force: bool = False) -> dict:
    """
    Load the stored device profile, or create one if it doesn't exist.
    Pass force=True to re-profile (e.g. after pulling new models).
    """
    if not force and PROFILE_PATH.exists():
        try:
            with open(PROFILE_PATH) as f:
                return json.load(f)
        except Exception:
            pass

    print("  Profiling device hardware...", end=" ", flush=True)
    profile = _build_profile()
    PROFILE_PATH.write_text(json.dumps(profile, indent=2))
    print("done")
    return profile


def print_summary(profile: dict):
    hw = profile["hardware"]
    rt = profile["runtime"]
    m  = profile["models"]
    tp = profile.get("throughput", {})

    print(f"  RAM: {hw['ram_gb']} GB  |  Cores: {hw['cpu_cores']}  |  "
          f"{'Laptop' if hw['is_laptop'] else 'Desktop'}  |  {hw['platform']}")
    print(f"  Worker : {m['worker']}"
          + (f"  ({tp[m['worker']]} tok/s)" if m['worker'] in tp else ""))
    print(f"  Router : {m['router']}"
          + (f"  ({tp[m['router']]} tok/s)" if m['router'] in tp else ""))
    print(f"  Parallel: {rt['enable_parallel']}  |  "
          f"Max agents: {rt['max_parallel_agents']}  |  "
          f"Budget: {rt['budget_mode']}")
