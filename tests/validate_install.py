"""
Install validation — verifies the environment is correctly set up.

Checks:
  1. Python version >= 3.9
  2. All required packages importable at correct versions
  3. Virtual environment structure (if running inside one)
  4. Ollama binary present and server reachable
  5. At least one model pulled
  6. Launcher scripts exist and are executable
  7. .env file present
  8. data/ directory writable

Run:
    python tests/validate_install.py
    python tests/validate_install.py --strict   # fail on warnings too
"""

import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).parent.parent.resolve()

# ── Result tracking ───────────────────────────────────────────────────────────

class Result:
    def __init__(self):
        self.passed: List[str] = []
        self.warnings: List[str] = []
        self.failures: List[str] = []

    def ok(self, msg: str):
        self.passed.append(msg)
        print(f"  \033[32m✓\033[0m {msg}")

    def warn(self, msg: str):
        self.warnings.append(msg)
        print(f"  \033[33m⚠\033[0m {msg}")

    def fail(self, msg: str):
        self.failures.append(msg)
        print(f"  \033[31m✗\033[0m {msg}")

    def summary(self) -> Tuple[int, int, int]:
        return len(self.passed), len(self.warnings), len(self.failures)


# ── Individual checks ─────────────────────────────────────────────────────────

def check_python(r: Result):
    v = sys.version_info
    if v >= (3, 9):
        r.ok(f"Python {v.major}.{v.minor}.{v.micro} (>= 3.9 required)")
    else:
        r.fail(f"Python {v.major}.{v.minor} — need 3.9+")


def check_packages(r: Result):
    """Check all required packages are importable at correct versions."""
    required = {
        "pydantic":  "2.0",
        "requests":  "2.31",
        "numpy":     "1.26",
        "yaml":      None,   # pyyaml
        "bs4":       None,   # beautifulsoup4
        "sklearn":   None,   # scikit-learn
        "pytest":    "8.0",
    }
    # sentence_transformers downloads models on import — check via importlib.util instead
    slow_packages = {"sentence_transformers"}

    optional = {
        "openai":    "1.30",
        "anthropic": "0.28",
    }

    for mod, min_ver in required.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            if min_ver and ver != "?" and ver < min_ver:
                r.warn(f"{mod} {ver} (want >= {min_ver})")
            else:
                r.ok(f"{mod} {ver}")
        except ImportError:
            r.fail(f"{mod} not installed — run: pip install -r requirements.txt")

    # Check slow packages by spec only (no import)
    import importlib.util as _ilu
    for mod in slow_packages:
        spec = _ilu.find_spec(mod)
        if spec is not None:
            r.ok(f"{mod} (installed, not imported to avoid slow init)")
        else:
            r.warn(f"{mod} not installed — run: pip install sentence-transformers")

    for mod, min_ver in optional.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            r.ok(f"{mod} {ver} (optional)")
        except ImportError:
            r.warn(f"{mod} not installed (optional — needed for cloud API)")


def check_venv(r: Result):
    """Check if running inside a virtual environment."""
    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )
    venv_dir = ROOT / ".venv"

    if in_venv:
        r.ok(f"Running inside virtual environment: {sys.prefix}")
    elif venv_dir.exists():
        r.warn(f".venv exists at {venv_dir} but not activated — run: source .venv/bin/activate")
    else:
        r.warn("No virtual environment found — consider running install.sh first")


def check_ollama(r: Result):
    """Check Ollama binary and server."""
    # Binary
    result = subprocess.run(
        ["ollama", "--version"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        r.ok(f"Ollama binary: {result.stdout.strip()}")
    else:
        r.warn("Ollama not found — local models unavailable (cloud API still works)")
        return

    # Server
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            r.ok(f"Ollama server running ({len(models)} model(s) pulled)")
            for m in models[:5]:
                size_mb = m.get("size", 0) // (1024 * 1024)
                r.ok(f"  Model: {m['name']} ({size_mb} MB)")
            if not models:
                r.warn("No models pulled — run: ollama pull qwen2.5:1.5b")
        else:
            r.warn(f"Ollama server returned {resp.status_code}")
    except Exception as e:
        r.warn(f"Ollama server not reachable: {e} — run: ollama serve")


def check_launchers(r: Result):
    """Check launcher scripts exist."""
    if platform.system() == "Windows":
        for name in ("run.ps1", "run.bat"):
            p = ROOT / name
            if p.exists():
                r.ok(f"Launcher: {name}")
            else:
                r.warn(f"Launcher missing: {name} — run install.ps1 to create it")
    else:
        p = ROOT / "run.sh"
        if p.exists():
            executable = os.access(p, os.X_OK)
            if executable:
                r.ok("Launcher: run.sh (executable)")
            else:
                r.warn("run.sh exists but not executable — run: chmod +x run.sh")
        else:
            r.warn("run.sh missing — run install.sh to create it")


def check_env_file(r: Result):
    """Check .env file exists."""
    env = ROOT / ".env"
    example = ROOT / ".env.example"
    if env.exists():
        r.ok(".env file present")
        # Check for placeholder values
        content = env.read_text(encoding="utf-8", errors="replace")
        if "your-openai-api-key-here" in content or "your-anthropic-api-key-here" in content:
            r.warn(".env contains placeholder API keys — edit .env to add real keys")
    elif example.exists():
        r.warn(".env missing — copy .env.example to .env and fill in API keys")
    else:
        r.warn("Neither .env nor .env.example found")


def check_data_dir(r: Result):
    """Check data directory is writable."""
    data = ROOT / "data"
    data.mkdir(parents=True, exist_ok=True)

    test_file = data / ".write_test"
    try:
        test_file.write_text("ok")
        test_file.unlink()
        r.ok(f"data/ directory writable: {data}")
    except Exception as e:
        r.fail(f"data/ directory not writable: {e}")

    # Check registry
    registry = data / "agent_registry.json"
    if registry.exists():
        try:
            import json
            with open(registry) as f:
                reg = json.load(f)
            agents = list(reg.get("agents", {}).keys())
            r.ok(f"Agent registry: {len(agents)} agent(s) — {', '.join(agents[:5])}")
        except Exception as e:
            r.fail(f"Agent registry corrupt: {e}")
    else:
        r.warn("Agent registry not found — will be created on first run")


def check_app_imports(r: Result):
    """Check core app modules import without errors."""
    import importlib.util as _ilu

    modules = [
        "app.orchestrator",
        "app.router",
        "app.agents.base_agent",
        "app.agents.code_primary",
        "app.agents.web_research",
        "app.agents.knowledge_enricher",
        "app.tools.filesystem",
        "app.tools.web_fetcher",
        "app.tools.code_runner",
        "app.tools.codebase_builder",
        "app.storage.registry_store",
        "app.storage.artifact_store",
        "app.schemas.run_state",
        "app.schemas.registry",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
            r.ok(f"Import: {mod}")
        except Exception as e:
            r.fail(f"Import failed: {mod} — {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate installation")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as failures")
    args = parser.parse_args()

    # Add project root to path
    sys.path.insert(0, str(ROOT))

    print("\n\033[1;34m╔══════════════════════════════════════════════════════════╗\033[0m")
    print("\033[1;34m║  Install Validation                                       ║\033[0m")
    print("\033[1;34m╚══════════════════════════════════════════════════════════╝\033[0m\n")

    r = Result()

    sections = [
        ("Python",          check_python),
        ("Packages",        check_packages),
        ("Virtual Env",     check_venv),
        ("Ollama",          check_ollama),
        ("Launchers",       check_launchers),
        ("Environment",     check_env_file),
        ("Data Directory",  check_data_dir),
        ("App Imports",     check_app_imports),
    ]

    for title, fn in sections:
        print(f"\033[1m{title}\033[0m")
        print("─" * 40)
        fn(r)
        print()

    passed, warnings, failures = r.summary()
    total = passed + warnings + failures

    print("─" * 60)
    print(f"Results: \033[32m{passed} passed\033[0m  "
          f"\033[33m{warnings} warnings\033[0m  "
          f"\033[31m{failures} failures\033[0m  "
          f"({total} checks)")

    if failures:
        print("\n\033[31mInstallation has errors — fix failures before running the CLI.\033[0m")
        sys.exit(1)
    elif warnings and args.strict:
        print("\n\033[33mWarnings present (--strict mode).\033[0m")
        sys.exit(1)
    elif warnings:
        print("\n\033[33mInstallation OK with warnings — CLI should work.\033[0m")
        sys.exit(0)
    else:
        print("\n\033[32mInstallation fully validated — ready to run!\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    main()
