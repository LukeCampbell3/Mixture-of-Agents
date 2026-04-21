#!/usr/bin/env python3
"""
Mixture-of-Agents CLI — cross-platform setup entry point.

Delegates to the platform-specific installer:
  - Windows  → install.ps1  (PowerShell)
  - Linux    → install.sh   (Bash)
  - macOS    → install.sh   (Bash)

Can also be used as a quick dependency check without running the full installer.

Usage:
    python setup_claude_cli.py            # run full installer
    python setup_claude_cli.py --check    # check deps only, no install
    python setup_claude_cli.py --no-ollama
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

HERE = Path(__file__).parent.resolve()


def _color(code: str, text: str) -> str:
    if sys.platform == "win32" and not os.environ.get("TERM"):
        return text
    return f"\033[{code}m{text}\033[0m"

ok   = lambda t: print(_color("32", f"  ✓ {t}"))
warn = lambda t: print(_color("33", f"  ⚠ {t}"))
err  = lambda t: print(_color("31", f"  ✗ {t}"), file=sys.stderr)
info = lambda t: print(_color("36", f"  → {t}"))


# ---------------------------------------------------------------------------
# Dependency checker (no installs)
# ---------------------------------------------------------------------------

def check_python() -> bool:
    v = sys.version_info
    if v >= (3, 9):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    err(f"Python 3.9+ required, found {v.major}.{v.minor}")
    return False


def check_pip() -> bool:
    try:
        import pip  # noqa: F401
        ok(f"pip {pip.__version__}")
        return True
    except ImportError:
        warn("pip not found")
        return False


def check_venv() -> bool:
    venv = HERE / ".venv"
    if venv.exists():
        ok(f".venv found at {venv}")
        return True
    warn(".venv not found — run the installer")
    return False


def check_requirements() -> bool:
    missing = []
    packages = {
        "openai":                 "openai",
        "anthropic":              "anthropic",
        "pydantic":               "pydantic",
        "requests":               "requests",
        "numpy":                  "numpy",
        "sentence_transformers":  "sentence-transformers",
        "sklearn":                "scikit-learn",
        "yaml":                   "pyyaml",
        "bs4":                    "beautifulsoup4",
    }
    for mod, pkg in packages.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        warn(f"Missing packages: {', '.join(missing)}")
        return False
    ok("All Python packages installed")
    return True


def check_ollama() -> bool:
    # Check binary
    result = subprocess.run(
        ["ollama", "--version"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        warn("Ollama not found — install from https://ollama.com/")
        return False
    ok(f"Ollama: {result.stdout.strip()}")

    # Check server
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = r.json().get("models", [])
            ok(f"Ollama server running ({len(models)} model(s) pulled)")
            for m in models[:5]:
                info(f"  {m['name']}")
            return True
    except Exception:
        pass
    warn("Ollama server not running — start with: ollama serve")
    return False


def run_checks() -> bool:
    print(_color("1;34", "\nDependency Check"))
    print("─" * 40)
    results = [
        check_python(),
        check_pip(),
        check_venv(),
        check_requirements(),
        check_ollama(),
    ]
    print()
    passed = sum(results)
    total  = len(results)
    if passed == total:
        ok(f"All {total} checks passed — ready to run!")
        print(f"\n  {_color('32', 'Start the CLI:')}")
        if sys.platform == "win32":
            print("    .\\run.ps1   or   run.bat")
        else:
            print("    ./run.sh")
    else:
        warn(f"{passed}/{total} checks passed — run the installer to fix issues")
        if sys.platform == "win32":
            print("    .\\install.ps1")
        else:
            print("    ./install.sh")
    return passed == total


# ---------------------------------------------------------------------------
# Installer delegation
# ---------------------------------------------------------------------------

def run_installer(args: argparse.Namespace):
    extra = []
    if args.no_ollama:
        extra.append("--no-ollama")
    if args.dev:
        extra.append("--dev")
    if args.model:
        extra += ["--model", args.model]

    if sys.platform == "win32":
        script = HERE / "install.ps1"
        if not script.exists():
            err(f"install.ps1 not found at {script}")
            sys.exit(1)

        ps_args = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)]
        if args.no_ollama:
            ps_args += ["-NoOllama"]
        if args.dev:
            ps_args += ["-Dev"]
        if args.model:
            ps_args += ["-Model", args.model]

        info(f"Running: {' '.join(ps_args)}")
        sys.exit(subprocess.call(ps_args))

    else:
        script = HERE / "install.sh"
        if not script.exists():
            err(f"install.sh not found at {script}")
            sys.exit(1)

        # Ensure executable
        script.chmod(script.stat().st_mode | 0o755)

        cmd = ["bash", str(script)] + extra
        info(f"Running: {' '.join(cmd)}")
        sys.exit(subprocess.call(cmd))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mixture-of-Agents CLI setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_claude_cli.py              Run full installer
  python setup_claude_cli.py --check      Check deps only
  python setup_claude_cli.py --no-ollama  Skip Ollama
  python setup_claude_cli.py --model qwen2.5:1.5b
        """,
    )
    parser.add_argument("--check",     action="store_true", help="Check dependencies only, no install")
    parser.add_argument("--no-ollama", action="store_true", help="Skip Ollama installation")
    parser.add_argument("--dev",       action="store_true", help="Install dev dependencies")
    parser.add_argument("--model",     default="",          help="Ollama model to pull")
    args = parser.parse_args()

    print(_color("1;36", f"""
╔══════════════════════════════════════════════════════════╗
║       Mixture-of-Agents CLI — Setup                      ║
║       Platform: {platform.system()} {platform.machine():<36}║
╚══════════════════════════════════════════════════════════╝"""))

    if args.check:
        ok_all = run_checks()
        sys.exit(0 if ok_all else 1)

    run_installer(args)


if __name__ == "__main__":
    main()
