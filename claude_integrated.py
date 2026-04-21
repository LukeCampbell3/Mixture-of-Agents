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
from typing import Dict, List, Any, Optional

# Import the agentic network
from app.orchestrator import Orchestrator
from app.schemas.run_state import RunState
from app.models.llm_client import create_llm_client

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
    """Get context from files in the workspace."""
    try:
        ignore_patterns = []
        gitignore_path = os.path.join(os.getcwd(), '.gitignore')
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                ignore_patterns = [
                    line.strip().replace('/', os.sep) for line in f 
                    if line.strip() and not line.startswith('#')
                ]

        context = []
        full_path = os.path.abspath(os.path.join(os.getcwd(), path))

        if os.path.isfile(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                context.append({
                    "path": os.path.basename(full_path),
                    "content": content
                })
                return {"success": True, "context": context}
            except (UnicodeDecodeError, IOError):
                return {"success": False, "error": "Cannot read file: binary or unreadable"}

        for root, dirs, files in os.walk(full_path):
            rel_root = os.path.relpath(root, full_path)
            dirs[:] = [d for d in dirs if not any(
                fnmatch.fnmatch(os.path.join(rel_root, d), p) or
                fnmatch.fnmatch(d, p.rstrip(os.sep))
                for p in ignore_patterns
            )]
            
            for file in files:
                rel_path = os.path.join(rel_root, file)
                if any(fnmatch.fnmatch(rel_path, p) for p in ignore_patterns) or \
                   any(fnmatch.fnmatch(file, p.rstrip(os.sep)) for p in ignore_patterns) or \
                   any(pattern in file.lower() for pattern in ['.git', '.pyc', '.env', '__pycache__']):
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, path)
                
                if any(any(fnmatch.fnmatch(part, p) for p in ignore_patterns) 
                       for part in rel_path.split(os.sep)):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    context.append({"path": rel_path, "content": content})
                except (UnicodeDecodeError, IOError):
                    continue
                    
        return {"success": True, "context": context}
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

            # Coerce types (config values are stored as strings via /config set)
            if isinstance(enable_parallel, str):
                enable_parallel = enable_parallel.lower() not in ("false", "0", "off")
            max_agents = int(max_agents)

            self.orchestrator = Orchestrator(
                llm_provider=provider,
                llm_model=self.worker_model,
                llm_base_url=active_config.get("llm_base_url") or "http://localhost:11434",
                router_model=self.router_model,
                budget_mode=rt.get("budget_mode", "balanced"),
                enable_parallel=enable_parallel,
                max_parallel_agents=max_agents,
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

    def process_request(self, user_input: str, context: str = "") -> str:
        if not self.ai_enabled:
            return ("AI features are disabled. Ollama is not ready.\n"
                    "Run: ollama serve  then restart the CLI.")
        if not self.orchestrator:
            if not self.initialize_orchestrator():
                return "Error: Could not initialize Agentic Network."
        full_request = f"{context}\n\nUser request: {user_input}" if context else user_input
        result = self.orchestrator.run_task(full_request)
        return result.final_answer

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
        "Basic Commands": {
            "/help": "Show this help message",
            "/exit, /quit": "Exit the program",
            "/pwd, /cwd": "Print working directory"
        },
        "File Operations": {
            "/ls [path]": "List directory contents",
            "/tree [path]": "Show directory structure in tree format",
            "/cat <file>": "Display file contents",
            "/write <file>": "Create/overwrite a file",
            "/append <file>": "Append to existing file"
        },
        "Directory Operations": {
            "/cd <path>": "Change directory",
            "/mkdir <path>": "Create directory"
        },
        "Configuration": {
            "/config set <key> <value>": "Set configuration value",
            "/config list": "List all configurations",
            "/config show": "Show active configuration"
        },
        "Context": {
            "/context [path], /#": "Get workspace context from path (default: current directory)"
        },
        "Agentic Network": {
            "/agents": "Show agent status and full registry",
            "/reload": "Reload agentic network configuration",
            "/test": "Test agentic network with a simple request",
            "/concurrency [n|off]": "Set parallel agents (e.g. /concurrency 3) or disable"
        },
        "Ollama Management": {
            "/ollama status": "Check Ollama status",
            "/ollama start": "Start Ollama server",
            "/ollama pull <model>": "Pull a model (e.g., /ollama pull qwen2.5:7b)"
        }
    }

    print(Color.blue("\nAvailable Commands:"))
    for section, cmds in commands.items():
        print(Color.yellow(f"\n{section}:"))
        for cmd, desc in cmds.items():
            print(f"  {Color.green(cmd):<30} {desc}")
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
            prompt = (Color.yellow(f"[{dir_name}] edit> ") if file_edit_mode.active and file_edit_mode.mode == "create" else
                     Color.yellow(f"[{dir_name}] append> ") if file_edit_mode.active and file_edit_mode.mode == "append" else
                     Color.green(f"[{dir_name}]> "))
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
                print(Color.green("Configuration reloaded successfully"))
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
            test_response = agentic_client.process_request("Say hello and confirm you're working")
            print(Color.blue("\nTest Response:"))
            print(test_response)
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
                               "enable_parallel", "max_parallel_agents"]:
                        print(Color.yellow("Reloading Agentic Network..."))
                        agentic_client.initialize_orchestrator()
                else:
                    print(Color.red("Failed to update configuration"))
                continue
                
            print(Color.red("Invalid config command. Use: /config set|list|show"))
            continue

        # Directory operations
        if cmd.startswith("/cd "):
            path_arg = cmd[4:].strip()
            result = change_directory(path_arg)
            if result.get("success"):
                print(Color.green(f"Changed directory to: {result['path']}"))
            else:
                print(Color.red(f"Error: {result.get('error')}"))
            continue

        if cmd.startswith("/mkdir "):
            path_arg = cmd[7:].strip()
            result = make_directory(path_arg)
            if result.get("success"):
                print(Color.green(f"Created directory: {result['path']}"))
            else:
                print(Color.red(f"Error: {result.get('error')}"))
            continue

        # File operations
        if cmd.startswith("/cat "):
            file_arg = cmd[5:].strip()
            result = read_file(file_arg)
            if result.get("success"):
                print(Color.green(f"Contents of: {result['path']}"))
                print("─" * 40)
                print(result.get("content"))
                print("─" * 40)
            else:
                print(Color.red(f"Error: {result.get('error')}"))
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

        # Context commands
        if cmd.startswith("/context") or cmd.startswith("/#"):
            parts = cmd.split(maxsplit=1)
            path = parts[1] if len(parts) > 1 else "."
            print(Color.green(f"Getting context from: {os.path.abspath(path)}"))
            result = get_workspace_context(path)
            if result.get("success"):
                files = result["context"]
                print(Color.green(f"\nFound {len(files)} file(s) in workspace:"))
                
                # Build context message for agentic network
                context_message = f"Workspace context from {path}:\n\n"
                for file in files:
                    print(Color.blue(f"\n[{file['path']}]"))
                    print("─" * 80)
                    content_lines = file['content'].splitlines()
                    for i, line in enumerate(content_lines, 1):
                        print(f"{Color.dim(f'{i:4d} │')} {line}")
                    print("─" * 80)
                    
                    # Add file content to context message
                    context_message += f"File: {file['path']}\n```\n{file['content']}\n```\n\n"
                
                # Store context for future requests
                workspace_context = context_message
                print(Color.green("\nWorkspace context stored for AI assistance."))
            else:
                print(Color.red(f"Error: {result.get('error')}"))
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

        # If not a built-in command, send to agentic network
        print(Color.blue("Processing with Agentic Network..."))
        response = agentic_client.process_request(cmd, workspace_context)
        print(Color.blue("\nAssistant:"))
        print(response)
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(Color.red("Fatal error:"))
        traceback.print_exc()
        sys.exit(1)