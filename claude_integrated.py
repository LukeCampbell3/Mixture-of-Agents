#!/usr/bin/env python3
"""
Integrated Claude Code CLI with Agentic Network v2
--------------------------------------------------

This integrates the LLMCode CLI interface with the Agentic Network v2 framework.
It provides an interactive CLI that uses the sparse multi-agent orchestration system
for intelligent coding assistance.

Key features:
- Interactive CLI with file operations
- Uses Agentic Network orchestrator for intelligent responses
- Central model configuration using local models (Ollama/vLLM)
- Agent management commands
- Workspace context awareness
- Automatic Ollama setup and validation
"""

import os
import sys
import json
import time
import argparse
import traceback
import fnmatch
import subprocess
import threading
import shlex
import signal
import select
import re as _re
from typing import Dict, List, Any, Optional

# Import the agentic network
from app.orchestrator import Orchestrator
from app.schemas.run_state import RunState
from app.models.llm_client import create_llm_client
from app.tools.filesystem import FilesystemExecutor, FileOperation


# ---------------------------------------------------------------------------
# Shell executor — runs real commands, tracks cwd, streams output
# ---------------------------------------------------------------------------

# Commands that must be handled natively (they change process state)
_BUILTIN_CMDS = {"cd", "export", "unset", "source", "."}

# Known shell command prefixes — used to decide "is this a shell command?"
_SHELL_CMD_PREFIXES = {
    # navigation
    "ls", "ll", "la", "l", "dir", "pwd", "cd", "pushd", "popd", "dirs",
    # files
    "cat", "less", "more", "head", "tail", "touch", "cp", "mv", "rm",
    "mkdir", "rmdir", "ln", "chmod", "chown", "chgrp", "stat", "file",
    "find", "locate", "which", "whereis", "type",
    # text processing
    "grep", "egrep", "fgrep", "rg", "ag", "sed", "awk", "cut", "sort",
    "uniq", "wc", "tr", "tee", "xargs", "diff", "patch", "strings",
    "column", "paste", "join", "comm", "fold", "fmt", "pr",
    # archives
    "tar", "gzip", "gunzip", "zip", "unzip", "bzip2", "xz", "7z",
    # processes
    "ps", "top", "htop", "kill", "killall", "pkill", "pgrep", "jobs",
    "bg", "fg", "nohup", "nice", "renice", "wait", "sleep",
    # system info
    "uname", "hostname", "whoami", "id", "groups", "uptime", "date",
    "cal", "df", "du", "free", "lscpu", "lsmem", "lsblk", "lsusb",
    "lspci", "dmesg", "journalctl", "systemctl", "service",
    # network
    "ping", "curl", "wget", "ssh", "scp", "rsync", "netstat", "ss",
    "ip", "ifconfig", "nslookup", "dig", "host", "traceroute", "nc",
    "telnet", "ftp", "sftp",
    # package managers
    "apt", "apt-get", "dpkg", "yum", "dnf", "rpm", "pacman", "brew",
    "pip", "pip3", "npm", "yarn", "cargo", "go", "gem", "composer",
    # dev tools
    "git", "make", "cmake", "gcc", "g++", "clang", "python", "python3",
    "node", "deno", "ruby", "java", "javac", "mvn", "gradle",
    "docker", "docker-compose", "kubectl", "helm", "terraform",
    "ansible", "vagrant",
    # editors (launch but don't block)
    "nano", "vim", "vi", "emacs", "code", "subl",
    # shell utilities
    "echo", "printf", "read", "test", "[", "true", "false",
    "env", "printenv", "set", "alias", "history", "clear", "reset",
    "man", "info", "help",
    # misc
    "tree", "watch", "timeout", "time", "strace", "ltrace",
    "ldd", "nm", "objdump", "readelf", "hexdump", "xxd",
    "base64", "md5sum", "sha256sum", "sha1sum",
    "jq", "yq", "xmllint", "csvkit",
    "ollama", "ffmpeg", "convert", "identify",
}

# Patterns that strongly indicate natural language (→ send to AI)
_NL_PATTERNS = _re.compile(
    r"^(how|what|why|when|where|which|who|can you|could you|please|"
    r"explain|describe|tell me|show me|help me|write|create|build|"
    r"implement|generate|make|fix|debug|refactor|review|analyze|"
    r"compare|summarize|translate|convert|i want|i need|i would|"
    r"give me|provide|suggest|recommend)",
    _re.IGNORECASE,
)


def _looks_like_shell_command(text: str) -> bool:
    """Return True if text looks like a shell command rather than natural language."""
    text = text.strip()
    if not text:
        return False

    # Pipes, redirects, semicolons → definitely shell
    if any(c in text for c in ("|", ">", "<", "&&", "||", ";")):
        return True

    # Starts with ./ or ~/ or / → path execution
    if text.startswith(("./", "../", "~/", "/")):
        return True

    # Natural language patterns → not a shell command
    if _NL_PATTERNS.match(text):
        return False

    # Check first word against known commands
    first_word = text.split()[0].lower().rstrip("\\")
    if first_word in _SHELL_CMD_PREFIXES:
        # "help me ..." is natural language even though "help" is a command
        if first_word in ("help", "man", "info") and len(text.split()) > 2:
            return False
        return True

    # Short single-word input with no spaces → probably a command
    if " " not in text and len(text) < 20 and text.isalnum():
        return True

    return False


class ShellExecutor:
    """
    Execute shell commands with proper output streaming, cwd tracking,
    and environment variable management.

    Handles:
    - cd (updates internal cwd, syncs os.getcwd())
    - export VAR=val (updates os.environ)
    - pipes, redirects, &&, ||, ; (via shell=True)
    - streaming output (line-by-line as it arrives)
    - Ctrl+C forwarding to child process
    - Interactive commands (vim, nano, etc.) via os.system()
    """

    # Commands that need a real TTY (interactive)
    _INTERACTIVE = {"vim", "vi", "nano", "emacs", "less", "more", "top",
                    "htop", "man", "info", "python", "python3", "node",
                    "irb", "pry", "ipython", "julia", "R", "ghci"}

    def __init__(self):
        self.env = os.environ.copy()

    def run(self, command: str) -> int:
        """
        Execute a shell command. Returns exit code.
        Output is streamed directly to stdout/stderr.
        """
        command = command.strip()
        if not command:
            return 0

        # ── Built-in: cd ─────────────────────────────────────────────────────
        if _re.match(r"^cd(\s|$)", command):
            return self._builtin_cd(command)

        # ── Built-in: export ─────────────────────────────────────────────────
        if command.startswith("export "):
            return self._builtin_export(command)

        # ── Built-in: unset ──────────────────────────────────────────────────
        if command.startswith("unset "):
            varname = command[6:].strip()
            self.env.pop(varname, None)
            os.environ.pop(varname, None)
            return 0

        # ── Built-in: clear / reset ──────────────────────────────────────────
        if command in ("clear", "reset"):
            os.system("cls" if sys.platform == "win32" else "clear")
            return 0

        # ── Interactive commands — hand off to os.system() for full TTY ──────
        first_word = command.split()[0].lower()
        if first_word in self._INTERACTIVE:
            return os.system(command)

        # ── Everything else — subprocess with streaming output ────────────────
        return self._run_subprocess(command)

    def _builtin_cd(self, command: str) -> int:
        parts = command.split(None, 1)
        if len(parts) == 1 or parts[1] == "~":
            target = os.path.expanduser("~")
        elif parts[1] == "-":
            target = self.env.get("OLDPWD", os.getcwd())
        else:
            target = os.path.expanduser(os.path.expandvars(parts[1]))
            target = os.path.abspath(os.path.join(os.getcwd(), target))

        try:
            old = os.getcwd()
            os.chdir(target)
            self.env["OLDPWD"] = old
            self.env["PWD"] = os.getcwd()
            os.environ["PWD"] = os.getcwd()
            return 0
        except FileNotFoundError:
            print(f"cd: {target}: No such file or directory", file=sys.stderr)
            return 1
        except NotADirectoryError:
            print(f"cd: {target}: Not a directory", file=sys.stderr)
            return 1
        except PermissionError:
            print(f"cd: {target}: Permission denied", file=sys.stderr)
            return 1

    def _builtin_export(self, command: str) -> int:
        # export VAR=value  or  export VAR
        rest = command[7:].strip()
        if "=" in rest:
            key, _, val = rest.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            val = os.path.expandvars(val)
            self.env[key] = val
            os.environ[key] = val
        else:
            # export VAR — mark for export (already in env if set)
            pass
        return 0

    def _run_subprocess(self, command: str) -> int:
        """Run command via subprocess, streaming output line by line."""
        # Use shell=True so pipes, redirects, &&, || all work
        use_shell = True

        # On Windows use cmd.exe; on Unix use bash if available
        if sys.platform == "win32":
            executable = None  # uses cmd.exe
        else:
            executable = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"

        proc = None
        try:
            proc = subprocess.Popen(
                command,
                shell=use_shell,
                executable=executable,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd(),
                env=self.env,
                text=True,
                bufsize=1,
            )

            # Stream stdout and stderr concurrently
            import threading as _threading

            def _stream(pipe, color_fn=None):
                try:
                    for line in pipe:
                        if color_fn:
                            print(color_fn(line), end="")
                        else:
                            print(line, end="")
                except Exception:
                    pass

            t_out = _threading.Thread(target=_stream, args=(proc.stdout,), daemon=True)
            t_err = _threading.Thread(
                target=_stream,
                args=(proc.stderr, lambda l: f"\033[91m{l}\033[0m"),
                daemon=True,
            )
            t_out.start()
            t_err.start()

            try:
                proc.wait()
            except KeyboardInterrupt:
                # Forward Ctrl+C to child
                try:
                    proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
                print()  # newline after ^C
                return 130

            t_out.join(timeout=2)
            t_err.join(timeout=2)
            return proc.returncode

        except FileNotFoundError:
            first = command.split()[0]
            print(f"\033[91m{first}: command not found\033[0m", file=sys.stderr)
            return 127
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m", file=sys.stderr)
            return 1
        finally:
            if proc and proc.poll() is None:
                proc.kill()


# Singleton shell executor — persists cwd and env across commands
_shell = ShellExecutor()


# ---------------------------------------------------------------------------
# Language preference detection + user prompt
# ---------------------------------------------------------------------------

# Signals that a task is a large codebase build (not a one-off snippet)
_CODEBASE_SIGNALS = _re.compile(
    r"\b(build|create|implement|develop|write|generate|scaffold|set up|setup)\b"
    r".{0,60}"
    r"\b(app|application|api|server|service|cli|tool|library|framework|"
    r"project|codebase|system|backend|frontend|full.?stack|microservice)\b",
    _re.IGNORECASE,
)

# Languages the system can target
_SUPPORTED_LANGUAGES = {
    "python":     ["python", "py", "django", "flask", "fastapi"],
    "typescript": ["typescript", "ts", "node", "nodejs", "next", "react", "vue"],
    "javascript": ["javascript", "js"],
    "rust":       ["rust", "rs", "cargo"],
    "go":         ["golang", "go"],
    "java":       ["java", "spring", "maven", "gradle"],
    "kotlin":     ["kotlin", "kt"],
    "swift":      ["swift", "ios", "xcode"],
    "csharp":     ["c#", "csharp", "dotnet", ".net", "asp.net"],
    "cpp":        ["c++", "cpp"],
    "ruby":       ["ruby", "rails"],
    "php":        ["php", "laravel"],
}

# Reverse map: keyword → canonical language name
_LANG_KEYWORDS: dict = {}
for _lang, _kws in _SUPPORTED_LANGUAGES.items():
    for _kw in _kws:
        _LANG_KEYWORDS[_kw] = _lang


def _detect_language_from_text(text: str) -> str:
    """Return canonical language name if explicitly mentioned, else ''."""
    t = text.lower()
    for kw, lang in _LANG_KEYWORDS.items():
        # Word-boundary match
        if _re.search(r"\b" + _re.escape(kw) + r"\b", t):
            return lang
    return ""


def _is_large_codebase_task(text: str) -> bool:
    """Return True if the task looks like building a multi-file codebase."""
    return bool(_CODEBASE_SIGNALS.search(text))


def _detect_or_prompt_language(user_input: str, agentic_client) -> str:
    """
    Detect language from user input or prompt the user to choose.

    Returns the chosen language name, or '' if not applicable.

    Logic:
    - If language is already explicit in the request → use it, no prompt
    - If task is a large codebase build AND no language specified → ask once
    - If task is a simple snippet/question → no prompt
    - If orchestrator already has a language preference set → reuse it
    """
    # Reuse existing preference if set
    if agentic_client.orchestrator:
        existing = getattr(agentic_client.orchestrator, "_language_preference", "")
        if existing:
            return existing

    # Detect from text
    detected = _detect_language_from_text(user_input)
    if detected:
        return detected

    # Only prompt for large codebase tasks
    if not _is_large_codebase_task(user_input):
        return ""

    # Ask the user
    print(Color.yellow(
        "\n  This looks like a multi-file project. What language should I use?"
    ))
    print(Color.dim(
        "  Options: python, typescript, javascript, rust, go, java, kotlin, "
        "swift, csharp, cpp, ruby, php"
    ))
    print(Color.dim("  Press Enter to let the AI decide.\n"))

    try:
        choice = input(Color.yellow("  Language> ")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return ""

    if not choice:
        return ""

    # Normalise
    lang = _LANG_KEYWORDS.get(choice, choice)
    if lang in _SUPPORTED_LANGUAGES:
        print(Color.green(f"  Using: {lang}"))
        return lang

    # Fuzzy: partial match
    for canonical in _SUPPORTED_LANGUAGES:
        if choice in canonical or canonical in choice:
            print(Color.green(f"  Using: {canonical}"))
            return canonical

    print(Color.dim(f"  Unrecognised language '{choice}' — AI will decide."))
    return ""

# ANSI color codes for colored terminal output
class Color:
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def blue(text):
        return f"{Color.BLUE}{text}{Color.RESET}"

    @staticmethod
    def green(text):
        return f"{Color.GREEN}{text}{Color.RESET}"

    @staticmethod
    def red(text):
        return f"{Color.RED}{text}{Color.RESET}"

    @staticmethod
    def yellow(text):
        return f"{Color.YELLOW}{text}{Color.RESET}"

    @staticmethod
    def dim(text):
        return f"{Color.DIM}{text}{Color.RESET}"


# ----------------------------
# Ollama Management
# ----------------------------

class OllamaManager:
    """Manages Ollama installation, server, and model lifecycle."""
    
    def __init__(self):
        self.ollama_installed = False
        self.ollama_running = False
        self.model_available = False
        self.ollama_process = None
        self.setup_complete = False

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def check_ollama_installed(self) -> bool:
        """Check if Ollama binary is on PATH."""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True, text=True, timeout=5
            )
            self.ollama_installed = result.returncode == 0
            return self.ollama_installed
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.ollama_installed = False
            return False

    def check_ollama_running(self) -> bool:
        """Check if Ollama HTTP server is reachable."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            self.ollama_running = response.status_code == 200
            return self.ollama_running
        except Exception:
            self.ollama_running = False
            return False

    def check_model_available(self, model_name: str = "qwen2.5:7b") -> bool:
        """Check if the specified model is already pulled."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    if model_name.lower() in model.get("name", "").lower():
                        self.model_available = True
                        return True
            self.model_available = False
            return False
        except Exception:
            self.model_available = False
            return False

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    def install_ollama(self) -> bool:
        """Automatically install Ollama for the current platform."""
        print(Color.yellow("Installing Ollama automatically..."))

        try:
            if sys.platform == "win32":
                return self._install_ollama_windows()
            elif sys.platform == "darwin":
                return self._install_ollama_macos()
            else:
                return self._install_ollama_linux()
        except Exception as e:
            print(Color.red(f"✗ Ollama installation failed: {e}"))
            return False

    def _install_ollama_windows(self) -> bool:
        """Install Ollama on Windows by downloading the official EXE installer."""
        import urllib.request
        import tempfile

        installer_url = "https://ollama.com/download/OllamaSetup.exe"
        installer_path = os.path.join(tempfile.gettempdir(), "OllamaSetup.exe")

        # --- download ---
        print(Color.dim(f"  Downloading installer from {installer_url} ..."))
        try:
            def _progress(block_num, block_size, total_size):
                if total_size > 0:
                    pct = min(100, block_num * block_size * 100 // total_size)
                    print(f"\r  Downloading... {pct}%", end="", flush=True)

            urllib.request.urlretrieve(installer_url, installer_path, _progress)
            print()  # newline after progress
            print(Color.green(f"✓ Installer downloaded to {installer_path}"))
        except Exception as e:
            print(Color.red(f"✗ Download failed: {e}"))
            return False

        # --- silent install ---
        print(Color.dim("  Running silent install (may take 1-2 minutes)..."))
        try:
            # Run installer and wait up to 5 minutes
            result = subprocess.run(
                [installer_path, "/S"],   # NSIS silent flag
                capture_output=True, text=True, timeout=300,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            # Refresh PATH regardless of return code — installer may exit 0 or 1
            self._refresh_path_windows()
            # Verify the binary actually landed
            if self.check_ollama_installed():
                print(Color.green("✓ Ollama installed successfully"))
                return True
            else:
                print(Color.red(f"✗ Installer finished (code {result.returncode}) but ollama binary not found"))
                print(Color.yellow("  Try opening a new terminal — PATH may need refreshing"))
                return False
        except subprocess.TimeoutExpired:
            # Installer is still running in background; check if binary appeared
            self._refresh_path_windows()
            if self.check_ollama_installed():
                print(Color.green("✓ Ollama installed (installer still finishing in background)"))
                return True
            print(Color.red("✗ Installer timed out and binary not found"))
            return False
        except Exception as e:
            print(Color.red(f"✗ Install failed: {e}"))
            return False

    def _refresh_path_windows(self):
        """Add the default Ollama install location to PATH for this process."""
        ollama_dir = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama")
        if os.path.isdir(ollama_dir) and ollama_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ollama_dir + os.pathsep + os.environ.get("PATH", "")

    def _install_ollama_macos(self) -> bool:
        """Install Ollama on macOS via Homebrew or curl installer."""
        # Try Homebrew first
        try:
            result = subprocess.run(
                ["brew", "install", "--cask", "ollama"],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print(Color.green("✓ Ollama installed via Homebrew"))
                self.ollama_installed = True
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: curl installer
        print(Color.dim("  Trying curl installer..."))
        try:
            result = subprocess.run(
                ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print(Color.green("✓ Ollama installed via curl"))
                self.ollama_installed = True
                return True
            else:
                print(Color.red(f"✗ curl installer failed: {result.stderr[:200]}"))
                return False
        except Exception as e:
            print(Color.red(f"✗ macOS installation failed: {e}"))
            return False

    def _install_ollama_linux(self) -> bool:
        """Install Ollama on Linux via curl installer."""
        try:
            result = subprocess.run(
                ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print(Color.green("✓ Ollama installed via curl"))
                self.ollama_installed = True
                return True
            else:
                print(Color.red(f"✗ Linux installer failed: {result.stderr[:200]}"))
                return False
        except Exception as e:
            print(Color.red(f"✗ Linux installation failed: {e}"))
            return False

    # ------------------------------------------------------------------
    # Server management
    # ------------------------------------------------------------------

    def start_ollama_server(self) -> bool:
        """Start Ollama server in the background."""
        try:
            print(Color.yellow("Starting Ollama server..."))

            kwargs = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            self.ollama_process = subprocess.Popen(["ollama", "serve"], **kwargs)

            import requests
            for i in range(30):
                time.sleep(1)
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code == 200:
                        print(Color.green("✓ Ollama server started"))
                        self.ollama_running = True
                        return True
                except Exception:
                    pass
                if i % 5 == 4:
                    print(Color.dim(f"  Waiting for Ollama... ({i+1}/30s)"))

            print(Color.red("✗ Ollama server did not respond in time"))
            return False

        except Exception as e:
            print(Color.red(f"✗ Error starting Ollama: {e}"))
            return False

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def pull_model(self, model_name: str = "qwen2.5:7b") -> bool:
        """Pull the specified model, streaming progress."""
        try:
            print(Color.yellow(f"Pulling model: {model_name}  (this may take a few minutes)..."))

            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )

            for raw in process.stdout:
                try:
                    line = raw.decode("utf-8", errors="replace").strip()
                except Exception:
                    line = ""
                if line:
                    print(Color.dim(f"  {line}"))

            process.wait()

            if process.returncode == 0:
                print(Color.green(f"✓ Model {model_name} ready"))
                self.model_available = True
                return True
            else:
                print(Color.red(f"✗ Failed to pull model {model_name}"))
                return False

        except Exception as e:
            print(Color.red(f"✗ Error pulling model: {e}"))
            return False

    # ------------------------------------------------------------------
    # Full setup orchestration
    # ------------------------------------------------------------------

    def setup_ollama(self, model_name: str = "qwen2.5:1.5b") -> bool:
        """
        Full auto-setup:
          1. Install Ollama if missing
          2. Start server if not running
          3. Pull model if not available
        """
        print(Color.blue("\nOllama Auto-Setup"))
        print(Color.blue("=" * 40))

        # --- Step 1: ensure Ollama is installed ---
        if not self.check_ollama_installed():
            print(Color.yellow("⚠ Ollama not found — installing automatically..."))
            if not self.install_ollama():
                print(Color.red("✗ Could not install Ollama automatically."))
                print(Color.yellow("  Please install manually from: https://ollama.com/"))
                self.setup_complete = False
                return False
            # Re-check after install
            if not self.check_ollama_installed():
                print(Color.red("✗ Ollama still not found after install attempt."))
                print(Color.yellow("  You may need to open a new terminal or reboot."))
                self.setup_complete = False
                return False
        print(Color.green("✓ Ollama is installed"))

        # --- Step 2: ensure server is running ---
        if not self.check_ollama_running():
            print(Color.yellow("⚠ Ollama server not running — starting..."))
            if not self.start_ollama_server():
                self.setup_complete = False
                return False
        else:
            print(Color.green("✓ Ollama server is running"))

        # --- Step 3: ensure model is available ---
        if not self.check_model_available(model_name):
            print(Color.yellow(f"⚠ Model {model_name} not found — pulling..."))
            if not self.pull_model(model_name):
                self.setup_complete = False
                return False
        else:
            print(Color.green(f"✓ Model {model_name} is available"))

        print(Color.green("\n✅ Ollama setup complete — AI features enabled"))
        self.setup_complete = True
        return True

    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True only when every component is confirmed ready."""
        return (
            self.setup_complete
            and self.ollama_installed
            and self.ollama_running
            and self.model_available
        )

    def cleanup(self):
        """Terminate the Ollama server process if we started it."""
        if self.ollama_process:
            try:
                self.ollama_process.terminate()
                self.ollama_process.wait(timeout=5)
            except Exception:
                try:
                    self.ollama_process.kill()
                except Exception:
                    pass


# ----------------------------
# Configuration Management
# ----------------------------

CONFIG_PATH = os.path.expanduser("~/.claude_agentic_config.json")
DEFAULT_CONFIG = {
    "configs": {
        "default": {
            # "auto" means: pick the fastest available provider at startup
            # Override with a specific provider/model if you want to pin one.
            "llm_provider": "auto",
            "llm_model": "auto",
            "llm_base_url": None,
            "budget_mode": "balanced",
            "enable_parallel": True,
            "max_parallel_agents": 3,
            "max_tokens": 800,
            "auto_approve_file_ops": False,
            "debug": False
        }
    },
    "active": "default"
}

def load_config():
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        else:
            save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
    except Exception as e:
        print(Color.red("Error loading config:"), e)
        return DEFAULT_CONFIG.copy()

def save_config(config):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(Color.red("Error saving config:"), e)

# Load configuration on startup.
config = load_config()

def get_active_config():
    active = config.get("active", "default")
    return config.get("configs", {}).get(active, {})

def set_active_config(key, value):
    active = config.get("active", "default")
    if active in config.get("configs", {}):
        config["configs"][active][key] = value
        save_config(config)
        return True
    return False


# ----------------------------
# File System Operations
# ----------------------------

def get_current_dir():
    return os.getcwd()

def print_tree(path, prefix="", is_last=True):
    """Print directory structure in tree format."""
    try:
        full_path = os.path.abspath(os.path.join(os.getcwd(), path))
        base_name = os.path.basename(full_path)
        
        # Print current node
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{Color.blue(base_name) if os.path.isdir(full_path) else base_name}")
        
        # If it's a directory, process its contents
        if os.path.isdir(full_path):
            entries = os.listdir(full_path)
            entries.sort(key=lambda x: (not os.path.isdir(os.path.join(full_path, x)), x.lower()))
            
            for i, entry in enumerate(entries):
                entry_path = os.path.join(full_path, entry)
                # Skip hidden files and common ignore patterns
                if entry.startswith('.') or entry in ['__pycache__', 'node_modules']:
                    continue
                    
                is_last_entry = i == len(entries) - 1
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(os.path.join(path, entry), new_prefix, is_last_entry)
                
    except Exception as e:
        print(Color.red(f"Error accessing {path}: {str(e)}"))
        
def list_directory(dir_path="."):
    try:
        full_path = os.path.abspath(os.path.join(os.getcwd(), dir_path))
        entries = os.listdir(full_path)
        files = []
        for entry in entries:
            entry_path = os.path.join(full_path, entry)
            try:
                stat = os.stat(entry_path)
                files.append({
                    "name": entry,
                    "isDirectory": os.path.isdir(entry_path),
                    "size": stat.st_size,
                    "modified": time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(stat.st_mtime))
                })
            except Exception as e:
                files.append({"name": entry, "error": str(e)})
        return {"success": True, "path": full_path, "files": files}
    except Exception as e:
        return {"success": False, "error": str(e)}

def change_directory(dir_path):
    try:
        full_path = os.path.abspath(os.path.join(os.getcwd(), dir_path))
        if not os.path.isdir(full_path):
            return {"success": False, "error": f"{full_path} is not a directory"}
        os.chdir(full_path)
        return {"success": True, "path": os.getcwd()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def make_directory(dir_path):
    try:
        full_path = os.path.abspath(os.path.join(os.getcwd(), dir_path))
        os.makedirs(full_path, exist_ok=True)
        return {"success": True, "path": full_path}
    except Exception as e:
        return {"success": False, "error": str(e)}

def read_file(file_path):
    try:
        full_path = os.path.abspath(os.path.join(os.getcwd(), file_path))
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"success": True, "path": full_path, "content": content}
    except Exception as e:
        return {"success": False, "error": str(e)}

def write_file(file_path, content):
    try:
        full_path = os.path.abspath(os.path.join(os.getcwd(), file_path))
        dir_name = os.path.dirname(full_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"success": True, "path": full_path}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_workspace_context(path="."):
    """Get context from files in the workspace — delegates to ContextBuilderAgent."""
    from app.agents.context_agent import ContextBuilderAgent
    try:
        agent = ContextBuilderAgent(workspace_root=os.getcwd())
        result = agent.build(
            explicit_files=[path] if path != "." else [],
            scan_dir=(path == "."),
            include_tree=False,
        )
        context_list = [
            {"path": e.path, "content": e.content}
            for e in result.entries
            if not e.is_binary and not e.error and e.content
        ]
        return {"success": True, "context": context_list}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ----------------------------
# File Edit Mode State
# ----------------------------

class FileEditMode:
    def __init__(self):
        self.active = False
        self.file_path = None
        self.content = []
        self.mode = None  # 'create' or 'append'

    def reset(self):
        self.active = False
        self.file_path = None
        self.content = []
        self.mode = None

file_edit_mode = FileEditMode()


# ----------------------------
# Agentic Network Integration
# ----------------------------

class AgenticNetworkClient:
    def __init__(self, ollama_manager: OllamaManager, device_profile: dict):
        self.orchestrator = None
        self.ollama_manager = ollama_manager
        self.device_profile = device_profile
        self.ai_enabled = False
        self.worker_model = device_profile["models"]["worker"]
        self.router_model = device_profile["models"]["router"]
        # Conversation history: list of {"role": "user"|"assistant", "content": str}
        self.history: list = []
        self.max_history_turns = 6   # keep last 6 exchanges (12 messages)
        self.initialize_orchestrator()

    def initialize_orchestrator(self):
        """Initialize the orchestrator using the device profile + active config overrides."""
        try:
            rt = self.device_profile["runtime"]
            active_config = get_active_config()

            provider = active_config.get("llm_provider", "auto")
            if provider == "auto":
                provider = "ollama"

            # Active config overrides device profile for concurrency settings
            enable_parallel = active_config.get("enable_parallel", rt.get("enable_parallel", True))
            max_agents      = active_config.get("max_parallel_agents", rt.get("max_parallel_agents", 2))
            max_tokens      = active_config.get("max_tokens", rt.get("max_tokens", 800))

            # Coerce types (config values are stored as strings via /config set)
            if isinstance(enable_parallel, str):
                enable_parallel = enable_parallel.lower() not in ("false", "0", "off")
            max_agents = int(max_agents)
            max_tokens = int(max_tokens)

            # Auto-approve file operations if enabled in config
            auto_approve = active_config.get("auto_approve_file_ops", False)
            
            self.orchestrator = Orchestrator(
                llm_provider=provider,
                llm_model=self.worker_model,
                llm_base_url=active_config.get("llm_base_url") or "http://localhost:11434",
                router_model=self.router_model,
                budget_mode=rt.get("budget_mode", "balanced"),
                enable_parallel=enable_parallel,
                max_parallel_agents=max_agents,
                max_tokens=max_tokens,
                auto_approve_file_ops=auto_approve,
            )

            print(Color.green("✅ Agentic Network initialized"))
            print(Color.dim(f"  Worker : {self.worker_model}"))
            if self.router_model != self.worker_model:
                print(Color.dim(f"  Router : {self.router_model}"))
            print(Color.dim(f"  Parallel: {enable_parallel}  "
                            f"Max agents: {max_agents}  "
                            f"Budget: {rt.get('budget_mode', 'balanced')}"))
            self.ai_enabled = True
            return True

        except Exception as e:
            print(Color.red(f"Error initializing Agentic Network: {e}"))
            self.ai_enabled = False
            return False

    def process_request(self, user_input: str, context: str = "",
                        workspace_root: str = ".") -> tuple:
        """Returns (answer_text, pending_tool_calls)."""
        if not self.ai_enabled:
            return ("AI features are disabled. Ollama is not ready.\n"
                    "Run: ollama serve  then restart the CLI.", [])
        if not self.orchestrator:
            if not self.initialize_orchestrator():
                return ("Error: Could not initialize Agentic Network.", [])

        # Detect topic shift — if the new request is off-topic, suppress history
        topic_shifted = self._detect_topic_shift(user_input)
        if topic_shifted:
            print(Color.dim("  [context] Topic shift detected — starting fresh context"))
            history_block = self._build_topic_shift_note()
        else:
            history_block = self._build_history_block()

        # Combine: workspace context + current request
        parts = []
        if context:
            parts.append(f"WORKSPACE CONTEXT:\n{context}")
        parts.append(user_input)
        full_request = "\n\n".join(parts)

        result = self.orchestrator.run_task(
            full_request,
            workspace_root=workspace_root,
            conversation_history=history_block,
        )
        answer = result.final_answer

        # Record this turn in history
        self.history.append({"role": "user",      "content": user_input})
        self.history.append({"role": "assistant",  "content": answer or ""})
        # Trim to max_history_turns exchanges
        max_msgs = self.max_history_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

        return answer, getattr(result, "pending_tool_calls", [])

    # ------------------------------------------------------------------
    # Topic shift detection
    # ------------------------------------------------------------------

    # Keyword clusters used for lightweight topic fingerprinting
    _TOPIC_CLUSTERS = {
        "data_structures": ["linked list", "binary tree", "stack", "queue", "heap",
                            "graph", "trie", "hash map", "array", "node", "pointer"],
        "algorithms":      ["sort", "search", "recursion", "dynamic programming",
                            "big o", "complexity", "algorithm", "traverse"],
        "python":          ["python", "def ", "class ", "import ", "pip", "venv",
                            "django", "flask", "pandas", "numpy"],
        "excel":           ["excel", "spreadsheet", "cell", "formula", "pivot",
                            "vlookup", "worksheet", "column", "row", "sort by"],
        "web":             ["html", "css", "javascript", "react", "api", "http",
                            "endpoint", "frontend", "backend", "rest"],
        "database":        ["sql", "query", "table", "join", "index", "postgres",
                            "mysql", "schema", "migration"],
        "devops":          ["docker", "kubernetes", "ci/cd", "deploy", "pipeline",
                            "terraform", "ansible", "nginx"],
        "ml":              ["pytorch", "tensorflow", "model", "training", "neural",
                            "embedding", "transformer", "dataset", "loss"],
    }

    def _topic_fingerprint(self, text: str) -> set:
        """Return the set of topic cluster names that match the text."""
        t = text.lower()
        return {
            cluster
            for cluster, keywords in self._TOPIC_CLUSTERS.items()
            if any(kw in t for kw in keywords)
        }

    def _detect_topic_shift(self, new_request: str) -> bool:
        """Return True if the new request is clearly off-topic from recent history.

        Uses keyword cluster overlap: if the new request shares no clusters with
        the last 2 user turns, it's a topic shift.
        Falls back to False (no shift) when history is empty or too short.
        """
        if len(self.history) < 2:
            return False

        new_clusters = self._topic_fingerprint(new_request)
        if not new_clusters:
            # No strong topic signal — don't suppress history
            return False

        # Collect clusters from the last 2 user messages
        recent_user_msgs = [
            m["content"] for m in self.history if m["role"] == "user"
        ][-2:]
        recent_clusters: set = set()
        for msg in recent_user_msgs:
            recent_clusters |= self._topic_fingerprint(msg)

        if not recent_clusters:
            return False

        # Shift if zero overlap between new topic and recent topics
        return len(new_clusters & recent_clusters) == 0

    def _build_history_block(self) -> str:
        """Format recent conversation history as a context block.

        Only includes turns that are topically relevant — assistant responses
        are truncated to 300 chars to keep the prompt lean.
        """
        if not self.history:
            return ""
        lines = ["CONVERSATION HISTORY (most recent last):"]
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            if msg["role"] == "assistant" and len(content) > 300:
                content = content[:300] + "... [truncated]"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _build_topic_shift_note(self) -> str:
        """Return a minimal context note when a topic shift is detected.

        Tells the model the previous topic without injecting its content,
        so it doesn't bleed into the new answer.
        """
        recent_user = [m["content"] for m in self.history if m["role"] == "user"]
        if not recent_user:
            return ""
        last_topic = recent_user[-1][:80].strip()
        return (
            f"NOTE: The previous conversation was about a different topic "
            f"(\"{last_topic}...\"). "
            f"Answer the current question independently — do not reference or apply "
            f"concepts from the previous topic unless explicitly asked."
        )

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

    def get_agent_info(self) -> dict:
        rt = self.device_profile["runtime"]
        return {
            "status":              "initialized" if self.ai_enabled else "disabled",
            "worker_model":        self.worker_model,
            "router_model":        self.router_model,
            "budget_mode":         rt.get("budget_mode"),
            "parallel_enabled":    rt.get("enable_parallel"),
            "max_parallel_agents": rt.get("max_parallel_agents"),
        }


# ----------------------------
# Main Interactive Loop
# ----------------------------

def print_banner():
    """Print the application banner."""
    banner = f"""
{Color.blue("╔══════════════════════════════════════════════════════════╗")}
{Color.blue("║")}      {Color.green("Claude Code + Agentic Network v2")}                    {Color.blue("║")}
{Color.blue("║")}      {Color.dim("Adaptive Local Deployment")}                           {Color.blue("║")}
{Color.blue("╚══════════════════════════════════════════════════════════╝")}
"""
    print(banner)

def print_help():
    """Print help information about available commands."""
    commands = {
        "Shell Commands (no prefix needed)": {
            "ls, ll, la":          "List directory contents",
            "cd <path>":           "Change directory",
            "cat <file>":          "Display file contents",
            "grep <pat> <file>":   "Search file contents",
            "find <path> -name":   "Find files",
            "mkdir -p <path>":     "Create directories",
            "cp / mv / rm":        "Copy, move, delete files",
            "git <cmd>":           "Git operations",
            "python <file>":       "Run Python scripts",
            "pip install <pkg>":   "Install packages",
            "ps aux / kill <pid>": "Process management",
            "curl / wget <url>":   "HTTP requests",
            "tar / zip / unzip":   "Archive operations",
            "chmod / chown":       "File permissions",
            "df / du / free":      "Disk and memory usage",
            "echo / printf":       "Print text",
            "env / export VAR=v":  "Environment variables",
            "| > < && ||":         "Pipes and redirects work too",
        },
        "Slash Commands": {
            "/help":               "Show this help message",
            "/exit, /quit":        "Exit the program",
            "/pwd, /cwd":          "Print working directory",
            "/ls [path]":          "List directory (alias for ls -la)",
            "/cd <path>":          "Change directory (alias for cd)",
            "/cat <file>":         "Display file (alias for cat)",
            "/tree [path]":        "Directory tree",
            "/mkdir <path>":       "Create directory",
            "/write <file>":       "Interactive file editor",
            "/append <file>":      "Append to file interactively",
        },
        "AI Commands": {
            "Just ask":            "e.g. 'create a FastAPI app in src/main.py'",
            "/context [path], /#": "Load workspace files into AI context",
            "/fs":                 "Show workspace root",
        },
        "Agentic Network": {
            "/agents":             "Show agent status and full registry",
            "/reload":             "Reload config and clear conversation history",
            "/test":               "Test agentic network with a simple request",
            "/concurrency [n|off]":"Set parallel agents",
            "/build [on|off]":     "Toggle codebase mode (large tokens + build loop)",
            "/history":            "Show conversation history",
            "/new, /clear":        "Clear conversation history",
        },
        "Configuration": {
            "/config set <k> <v>": "Set configuration value",
            "/config list":        "List all configurations",
            "/config show":        "Show active configuration",
        },
        "Ollama Management": {
            "/ollama status":      "Check Ollama status",
            "/ollama start":       "Start Ollama server",
            "/ollama pull <model>":"Pull a model",
        },
    }

    print(Color.blue("\nAvailable Commands:"))
    for section, cmds in commands.items():
        print(Color.yellow(f"\n{section}:"))
        for cmd_name, desc in cmds.items():
            print(f"  {Color.green(cmd_name):<35} {desc}")
    print(Color.dim("\n  Tip: shell commands run directly; natural language goes to the AI."))
    print()

def main():
    """Main interactive loop."""
    print_banner()

    # ── 1. Device profile ────────────────────────────────────────────────────
    from app.device_profile import load_or_create, print_summary
    print(Color.blue("Device Profile"))
    print(Color.blue("=" * 40))
    profile = load_or_create()
    print_summary(profile)
    print()

    worker_model = profile["models"]["worker"]
    router_model = profile["models"]["router"]

    # ── 2. Ollama setup — only pull what is actually needed ──────────────────
    ollama_manager = OllamaManager()

    needed_models = list({worker_model, router_model})  # deduplicate
    print(Color.blue("Ollama Setup"))
    print(Color.blue("=" * 40))

    # Ensure server is running first
    if not ollama_manager.check_ollama_installed():
        print(Color.yellow("⚠ Ollama not found — installing automatically..."))
        if not ollama_manager.install_ollama():
            print(Color.red("❌ Could not install Ollama. File operations still available."))
    else:
        print(Color.green("✓ Ollama is installed"))

    if ollama_manager.ollama_installed and not ollama_manager.check_ollama_running():
        ollama_manager.start_ollama_server()
    elif ollama_manager.ollama_installed:
        print(Color.green("✓ Ollama server is running"))

    # Pull only the models this device profile needs
    if ollama_manager.ollama_running:
        for model in needed_models:
            if not ollama_manager.check_model_available(model):
                print(Color.yellow(f"⚠ Pulling {model}..."))
                ollama_manager.pull_model(model)
            else:
                print(Color.green(f"✓ {model} is available"))

        ollama_manager.setup_complete = True
        ollama_manager.model_available = True

    # ── 3. Agentic network ───────────────────────────────────────────────────
    agentic_client = AgenticNetworkClient(ollama_manager, profile)

    # ── 4. Status summary ────────────────────────────────────────────────────
    print(Color.dim(f"\nWorking directory: {os.getcwd()}"))
    if agentic_client.ai_enabled:
        print(Color.green("✅ AI features: ENABLED"))
    else:
        print(Color.yellow("⚠ AI features: DISABLED — check Ollama"))
    print()
    
    # Store workspace context
    workspace_context = ""
    
    while True:
        try:
            # Update prompt to show more context
            dir_name = os.path.basename(os.getcwd())
            budget_tag = Color.yellow(" [build]") if profile["runtime"].get("budget_mode") == "codebase" else ""
            prompt = (Color.yellow(f"[{dir_name}] edit> ") if file_edit_mode.active and file_edit_mode.mode == "create" else
                     Color.yellow(f"[{dir_name}] append> ") if file_edit_mode.active and file_edit_mode.mode == "append" else
                     Color.green(f"[{dir_name}]{budget_tag}> "))
            line = input(prompt)

        except (EOFError, KeyboardInterrupt):
            print("\n" + Color.blue("Goodbye! Thanks for using Claude Code + Agentic Network."))
            ollama_manager.cleanup()
            break

        # File edit mode handling
        if file_edit_mode.active:
            if line.strip() == "/save":
                content = "\n".join(file_edit_mode.content)
                result = write_file(file_edit_mode.file_path, content)
                if result.get("success"):
                    print(Color.green(f"File saved: {result['path']}"))
                else:
                    print(Color.red(f"Error saving file: {result.get('error')}"))
                file_edit_mode.reset()
                continue
            elif line.strip() == "/cancel":
                print(Color.yellow("File edit cancelled."))
                file_edit_mode.reset()
                continue
            else:
                file_edit_mode.content.append(line)
                continue

        cmd = line.strip()
        
        # Basic commands
        if cmd in ("/exit", "/quit"):
            print(Color.blue("Goodbye!"))
            ollama_manager.cleanup()
            break

        if cmd == "/help":
            print_help()
            continue

        if cmd in ("/pwd", "/cwd"):
            print(Color.green("Current directory:"), get_current_dir())
            continue

        # Ollama management commands
        if cmd.startswith("/ollama"):
            parts = cmd.split()
            if len(parts) < 2:
                print(Color.red("Usage: /ollama status|start|pull <model>"))
                continue
                
            if parts[1] == "status":
                print(Color.green("Ollama Status:"))
                print(f"  Installed: {'Yes' if ollama_manager.ollama_installed else 'No'}")
                print(f"  Running: {'Yes' if ollama_manager.ollama_running else 'No'}")
                print(f"  Model available: {'Yes' if ollama_manager.model_available else 'No'}")
                print(f"  Setup complete: {'Yes' if ollama_manager.setup_complete else 'No'}")
                continue
                
            elif parts[1] == "start":
                if ollama_manager.start_ollama_server():
                    print(Color.green("Ollama server started"))
                    # Try to reinitialize agentic network
                    agentic_client.initialize_orchestrator()
                else:
                    print(Color.red("Failed to start Ollama server"))
                continue
                
            elif parts[1] == "pull" and len(parts) >= 3:
                model = parts[2]
                if ollama_manager.pull_model(model):
                    print(Color.green(f"Model {model} pulled successfully"))
                    # Try to reinitialize agentic network
                    agentic_client.initialize_orchestrator()
                else:
                    print(Color.red(f"Failed to pull model {model}"))
                continue
                
            else:
                print(Color.red("Usage: /ollama status|start|pull <model>"))
                continue

        # Agentic Network commands
        if cmd == "/agents":
            agent_info = agentic_client.get_agent_info()
            print(Color.green("Agentic Network Status:"))
            for key, value in agent_info.items():
                print(f"  {key}: {value}")
            # Also show live registry
            if agentic_client.orchestrator:
                reg = agentic_client.orchestrator.registry
                print(Color.green(f"\nAgent Registry ({len(reg.agents)} agents):"))
                for aid, spec in reg.agents.items():
                    tag = " [base]" if aid in ("code_primary", "web_research", "critic_verifier") else " [spawned]"
                    print(f"  {Color.blue(aid):<35} domain={spec.domain:<18} state={spec.lifecycle_state}{tag}")
            continue

        if cmd == "/reload":
            print(Color.yellow("Reloading Agentic Network configuration..."))
            if agentic_client.initialize_orchestrator():
                agentic_client.clear_history()
                print(Color.green("Configuration reloaded and conversation history cleared."))
            else:
                print(Color.red("Failed to reload configuration"))
            continue

        if cmd.startswith("/concurrency"):
            # /concurrency              — show current settings
            # /concurrency <n>          — set max_parallel_agents to n, enable parallel
            # /concurrency off          — disable parallel, max_agents=1
            parts = cmd.split()
            if len(parts) == 1:
                rt = profile["runtime"]
                print(Color.green("Concurrency settings:"))
                print(f"  enable_parallel:    {rt['enable_parallel']}")
                print(f"  max_parallel_agents: {rt['max_parallel_agents']}")
                print(f"  budget_mode:        {rt['budget_mode']}")
                print(Color.dim("  Use: /concurrency <n>  or  /concurrency off"))
            elif parts[1].lower() == "off":
                profile["runtime"]["enable_parallel"] = False
                profile["runtime"]["max_parallel_agents"] = 1
                set_active_config("enable_parallel", False)
                set_active_config("max_parallel_agents", 1)
                agentic_client.initialize_orchestrator()
                print(Color.yellow("Parallel disabled. Single-agent mode."))
            else:
                try:
                    n = int(parts[1])
                    if n < 1 or n > 16:
                        raise ValueError
                    profile["runtime"]["enable_parallel"] = n > 1
                    profile["runtime"]["max_parallel_agents"] = n
                    set_active_config("enable_parallel", n > 1)
                    set_active_config("max_parallel_agents", n)
                    agentic_client.initialize_orchestrator()
                    print(Color.green(f"Concurrency set to {n} agent(s), parallel={'on' if n > 1 else 'off'}."))
                except ValueError:
                    print(Color.red("Usage: /concurrency <number 1-16>  or  /concurrency off"))
            continue

        if cmd == "/test":
            print(Color.yellow("Testing Agentic Network with simple request..."))
            test_response, _ = agentic_client.process_request("Say hello and confirm you're working")
            print(Color.blue("\nTest Response:"))
            print(test_response)
            continue

        if cmd.startswith("/build"):
            # /build          — toggle codebase mode on/off
            # /build on       — enable codebase mode (large tokens, build loop)
            # /build off      — disable codebase mode
            parts = cmd.split()
            rt = profile["runtime"]
            current = rt.get("budget_mode", "balanced")
            if len(parts) == 1:
                # Toggle
                if current == "codebase":
                    rt["budget_mode"] = "balanced"
                    set_active_config("budget_mode", "balanced")
                    set_active_config("max_tokens", 800)
                    print(Color.yellow("Codebase mode OFF — back to balanced (800 tokens/agent)"))
                else:
                    rt["budget_mode"] = "codebase"
                    set_active_config("budget_mode", "codebase")
                    set_active_config("max_tokens", 4000)
                    print(Color.green("Codebase mode ON — large token budget, build-test-fix loop enabled"))
                    print(Color.dim("  Tokens/agent: 4000  |  Max iterations: 5  |  Tests: auto-generated"))
            elif parts[1].lower() in ("on", "1", "true"):
                rt["budget_mode"] = "codebase"
                set_active_config("budget_mode", "codebase")
                set_active_config("max_tokens", 4000)
                print(Color.green("Codebase mode ON"))
            elif parts[1].lower() in ("off", "0", "false"):
                rt["budget_mode"] = "balanced"
                set_active_config("budget_mode", "balanced")
                set_active_config("max_tokens", 800)
                print(Color.yellow("Codebase mode OFF"))
            else:
                print(Color.red("Usage: /build  or  /build on|off"))
            agentic_client.initialize_orchestrator()
            continue

        if cmd in ("/new", "/clear"):
            agentic_client.clear_history()
            print(Color.green("Conversation history cleared. Starting fresh."))
            continue

        if cmd in ("/history",):
            if not agentic_client.history:
                print(Color.dim("No conversation history yet."))
            else:
                print(Color.green(f"Conversation history ({len(agentic_client.history)//2} turns):"))
                for i, msg in enumerate(agentic_client.history):
                    role_color = Color.blue if msg["role"] == "user" else Color.green
                    label = "You" if msg["role"] == "user" else "AI "
                    content = msg["content"]
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"  {role_color(label)}: {content}")
            continue

        # Configuration commands
        if cmd.startswith("/config"):
            parts = cmd.split()
            if len(parts) == 1:
                print(Color.red("Missing config command. Use: /config set|list|show"))
                continue
                
            if parts[1] == "list":
                print(Color.green("Available configurations:"))
                for name in config.get("configs", {}):
                    if name == config.get("active"):
                        print(f"* {name} (active)")
                    else:
                        print(f"  {name}")
                continue
                
            if parts[1] == "show":
                active = get_active_config()
                print(Color.green("Active configuration:"))
                for key, value in active.items():
                    print(f"  {key}: {value}")
                continue
                
            if parts[1] == "set" and len(parts) >= 4:
                key = parts[2]
                value = " ".join(parts[3:])
                if set_active_config(key, value):
                    print(Color.green(f"Configuration updated: {key} = {value}"))
                    # Reload orchestrator if any runtime setting changed
                    if key in ["llm_provider", "llm_model", "llm_base_url", "budget_mode",
                               "enable_parallel", "max_parallel_agents", "max_tokens",
                               "auto_approve_file_ops"]:
                        print(Color.yellow("Reloading Agentic Network..."))
                        agentic_client.initialize_orchestrator()
                else:
                    print(Color.red("Failed to update configuration"))
                continue
                
            print(Color.red("Invalid config command. Use: /config set|list|show"))
            continue

        # Directory operations
        if cmd.startswith("/cd ") or cmd == "/cd":
            path_arg = cmd[3:].strip() or "~"
            _shell.run(f"cd {path_arg}")
            print(Color.green(f"  {os.getcwd()}"))
            continue

        if cmd.startswith("/mkdir "):
            path_arg = cmd[7:].strip()
            _shell.run(f"mkdir -p {shlex.quote(path_arg)}")
            continue

        # File operations
        if cmd.startswith("/cat "):
            file_arg = cmd[5:].strip()
            _shell.run(f"cat {shlex.quote(file_arg)}")
            continue

        if cmd.startswith("/ls") or cmd == "/ls":
            arg = cmd[3:].strip()
            _shell.run(f"ls -la {arg}" if arg else "ls -la")
            continue

        if cmd.startswith("/tree"):
            arg = cmd[5:].strip()
            # Use tree if available, fall back to find
            rc = _shell.run(f"tree {arg}" if arg else "tree")
            if rc == 127:  # command not found
                _shell.run(f"find {arg or '.'} -not -path '*/.*' | sort | head -100")
            continue

        if cmd.startswith("/write "):
            file_arg = cmd[7:].strip()
            file_edit_mode.active = True
            file_edit_mode.file_path = file_arg
            file_edit_mode.content = []
            file_edit_mode.mode = "create"
            print(Color.green(f"Creating file: {file_arg}"))
            print(Color.dim("Enter file content (type /save to save and exit, or /cancel to cancel):"))
            continue

        if cmd.startswith("/append "):
            file_arg = cmd[8:].strip()
            result = read_file(file_arg)
            if result.get("success"):
                file_edit_mode.active = True
                file_edit_mode.file_path = file_arg
                file_edit_mode.content = result.get("content", "").splitlines()
                file_edit_mode.mode = "append"
                print(Color.green(f"Appending to file: {file_arg}"))
                print(Color.dim("Enter content to append (type /save to save and exit, or /cancel to cancel):"))
            else:
                print(Color.red(f"Error: {result.get('error')}"))
            continue

        # Context commands — /context, /#, @file inline, or bare filename
        if cmd.startswith("/context") or cmd.startswith("/#"):
            from app.agents.context_agent import ContextBuilderAgent
            parts_ctx = cmd.split(maxsplit=1)
            path_arg = parts_ctx[1].strip() if len(parts_ctx) > 1 else "."

            agent_ctx = ContextBuilderAgent(workspace_root=os.getcwd())

            # Multiple space-separated files/paths supported
            explicit = [p for p in path_arg.split() if p] if path_arg != "." else []

            print(Color.yellow(f"  Building context from: {os.path.abspath(path_arg)}"))
            ctx_result = agent_ctx.build(
                explicit_files=explicit,
                scan_dir=(path_arg == "."),
                include_tree=True,
                max_scan_files=30,
            )

            # Print summary
            print(Color.green(f"\n  {ctx_result.summary()}"))

            # Show fuzzy resolutions
            if ctx_result.fuzzy_matches:
                print(Color.dim("  Resolved:"))
                for req, res in ctx_result.fuzzy_matches.items():
                    print(Color.dim(f"    {req!r} → {res}"))

            # Show ambiguous
            if ctx_result.ambiguous:
                print(Color.yellow("  Ambiguous (using first match):"))
                for name, candidates in ctx_result.ambiguous.items():
                    print(Color.yellow(f"    {name!r}: {', '.join(candidates[:4])}"))

            # Show warnings
            for w in ctx_result.warnings:
                print(Color.yellow(f"  ⚠ {w}"))

            # Print file tree
            if ctx_result.file_tree:
                print(Color.blue("\n  File tree:"))
                for line in ctx_result.file_tree.splitlines()[:40]:
                    print(Color.dim(f"    {line}"))
                if ctx_result.file_tree.count("\n") > 40:
                    print(Color.dim("    ... (truncated)"))

            # Print loaded files
            if ctx_result.entries:
                print(Color.green(f"\n  Loaded {len(ctx_result.entries)} file(s):"))
                for e in ctx_result.entries:
                    if e.error:
                        print(Color.red(f"    ✗ {e.path}: {e.error}"))
                    elif e.is_image:
                        print(Color.blue(f"    🖼  {e.path} ({e.size_bytes // 1024} KB image)"))
                    elif e.is_binary:
                        print(Color.dim(f"    ○ {e.path} (binary)"))
                    else:
                        trunc = " [truncated]" if e.truncated else ""
                        lines_count = e.content.count("\n") + 1
                        print(Color.green(f"    ✓ {e.path} ({lines_count} lines{trunc})"))

            workspace_context = ctx_result.context_block
            print(Color.green("\n  Context stored — will be injected into next AI request."))
            continue

        if cmd.startswith("/ls"):
            parts = cmd.split(maxsplit=1)
            dir_path = parts[1] if len(parts) > 1 else "."
            result = list_directory(dir_path)
            if result.get("success"):
                print(Color.green(f"Contents of: {result['path']}"))
                files = result["files"]
                files.sort(key=lambda x: (not x.get("isDirectory", False), x["name"].lower()))
                for f in files:
                    if f.get("isDirectory"):
                        print(Color.blue(f"{f['name']}/"))
                    elif f.get("error"):
                        print(Color.red(f"{f['name']} (error: {f['error']})"))
                    else:
                        print(f["name"])
            else:
                print(Color.red(f"Error: {result.get('error')}"))
            continue

        if cmd.startswith("/tree"):
            parts = cmd.split(maxsplit=1)
            path = parts[1] if len(parts) > 1 else "."
            print(Color.green(f"Directory tree for: {os.path.abspath(path)}"))
            print_tree(path)
            continue

        if cmd == "/fs":
            print(Color.green(f"Workspace root: {os.getcwd()}"))
            print(Color.dim("  The AI will write files relative to this directory."))
            print(Color.dim("  Use /context to load files into AI context."))
            print(Color.dim("  Use /cd to change the workspace root."))
            continue

        # ── Shell command passthrough ─────────────────────────────────────────
        # Any input that looks like a shell command runs directly.
        # Natural language goes to the AI.
        if not cmd.startswith("/") and _looks_like_shell_command(cmd):
            _shell.run(cmd)
            continue

        # ── @file inline resolution + auto-context injection ─────────────────
        # If the user's message contains @filename or bare filenames, resolve
        # them and inject their content into the request automatically.
        if not cmd.startswith("/"):
            from app.agents.context_agent import ContextBuilderAgent
            ctx_agent = ContextBuilderAgent(workspace_root=os.getcwd())
            mentioned = ctx_agent._extract_file_refs(cmd)

            if mentioned:
                print(Color.dim(f"  Resolving {len(mentioned)} file reference(s)..."))
                inline_ctx = ctx_agent.build(
                    user_input=cmd,
                    explicit_files=mentioned,
                    scan_dir=False,
                    include_tree=False,
                )
                if inline_ctx.entries:
                    # Show what was resolved
                    for name, resolved in inline_ctx.fuzzy_matches.items():
                        print(Color.dim(f"  {name!r} → {resolved}"))
                    for w in inline_ctx.warnings:
                        print(Color.yellow(f"  ⚠ {w}"))
                    # Merge inline context with any existing workspace_context
                    if workspace_context:
                        workspace_context = inline_ctx.context_block + "\n\n" + workspace_context
                    else:
                        workspace_context = inline_ctx.context_block

        # ── Language preference prompt for large codebase tasks ───────────────
        # Only ask when the task clearly involves building a multi-file codebase
        # and no language is already specified.
        if not cmd.startswith("/") and agentic_client.ai_enabled:
            lang_pref = _detect_or_prompt_language(cmd, agentic_client)
            if lang_pref and agentic_client.orchestrator:
                agentic_client.orchestrator._language_preference = lang_pref

        # If not a built-in command, send to agentic network
        print(Color.blue("Processing with Agentic Network..."))
        response, pending_ops = agentic_client.process_request(
            cmd, workspace_context, workspace_root=os.getcwd()
        )
        print(Color.blue("\nAssistant:"))
        print(response)
        print()

        # ── File operation approval flow ──────────────────────────────────
        if pending_ops:
            executor = FilesystemExecutor(workspace_root=os.getcwd())
            print(Color.yellow(f"\n  {len(pending_ops)} file operation(s) proposed:\n"))

            approved_ops = []
            for i, op in enumerate(pending_ops, 1):
                print(Color.blue(f"  [{i}/{len(pending_ops)}] {op.tool.upper()}: {op.path}"))
                if op.description:
                    print(Color.dim(f"  {op.description}"))

                # Show diff/preview
                preview = executor.preview(op)
                if preview and preview != "(no changes)":
                    # Colour the diff lines
                    for line in preview.splitlines():
                        if line.startswith("+") and not line.startswith("+++"):
                            print(Color.green(f"  {line}"))
                        elif line.startswith("-") and not line.startswith("---"):
                            print(Color.red(f"  {line}"))
                        else:
                            print(Color.dim(f"  {line}"))
                print()

                # Ask for approval
                try:
                    choice = input(
                        Color.yellow("  Apply? [y]es / [n]o / [a]ll / [q]uit: ")
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = "q"

                if choice in ("a", "all"):
                    approved_ops.extend(pending_ops[i - 1:])
                    print(Color.green(f"  Approved all remaining {len(pending_ops)-i+1} operation(s)."))
                    break
                elif choice in ("y", "yes", ""):
                    approved_ops.append(op)
                elif choice in ("q", "quit"):
                    print(Color.yellow("  Aborted remaining operations."))
                    break
                else:
                    print(Color.dim("  Skipped."))

            # Execute approved operations
            if approved_ops:
                print()
                # Use batch mode for faster execution when auto-approving
                results = executor.execute_all(approved_ops, batch_mode=auto_approve)
                for res in results:
                    if res.success:
                        print(Color.green(f"  ✓ {res.message}"))
                    else:
                        print(Color.red(f"  ✗ {res.message}"))
                print()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(Color.red("Fatal error:"))
        traceback.print_exc()
        sys.exit(1)