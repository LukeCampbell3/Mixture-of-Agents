#!/usr/bin/env bash
# =============================================================================
# Mixture-of-Agents CLI — Linux / macOS installer
# =============================================================================
# Usage:
#   chmod +x install.sh && ./install.sh
#   ./install.sh --model qwen2.5:1.5b   # pick a specific model
#   ./install.sh --no-ollama             # skip Ollama (use OpenAI/Anthropic)
#   ./install.sh --dev                   # also install dev/test dependencies
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "${GREEN}✓${RESET} $*"; }
warn() { echo -e "${YELLOW}⚠${RESET} $*"; }
err()  { echo -e "${RED}✗${RESET} $*" >&2; }
info() { echo -e "${BLUE}→${RESET} $*"; }
hdr()  { echo -e "\n${BOLD}$*${RESET}"; echo "$(printf '─%.0s' {1..60})"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
INSTALL_OLLAMA=true
DEV_DEPS=false
DEFAULT_MODEL="auto"   # auto = pick smallest available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MIN_PYTHON_MINOR=9     # Python 3.9+

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-ollama)   INSTALL_OLLAMA=false ;;
        --dev)         DEV_DEPS=true ;;
        --model)       DEFAULT_MODEL="$2"; shift ;;
        --model=*)     DEFAULT_MODEL="${1#*=}" ;;
        -h|--help)
            echo "Usage: $0 [--no-ollama] [--dev] [--model MODEL]"
            echo "  --no-ollama     Skip Ollama installation (use cloud API instead)"
            echo "  --dev           Install dev/test dependencies"
            echo "  --model MODEL   Default Ollama model (e.g. qwen2.5:1.5b)"
            exit 0 ;;
        *) warn "Unknown option: $1" ;;
    esac
    shift
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       Mixture-of-Agents CLI — Installer                  ║"
echo "║       Linux / macOS                                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo "Platform : $(uname -s) $(uname -m)"
echo "Directory: $SCRIPT_DIR"
echo ""

# ── 1. OS detection ───────────────────────────────────────────────────────────
hdr "1. Detecting OS"
OS="$(uname -s)"
ARCH="$(uname -m)"
PKG_MGR=""

case "$OS" in
    Linux)
        if command -v apt-get &>/dev/null; then
            PKG_MGR="apt"
        elif command -v dnf &>/dev/null; then
            PKG_MGR="dnf"
        elif command -v yum &>/dev/null; then
            PKG_MGR="yum"
        elif command -v pacman &>/dev/null; then
            PKG_MGR="pacman"
        elif command -v zypper &>/dev/null; then
            PKG_MGR="zypper"
        fi
        ok "Linux detected (package manager: ${PKG_MGR:-unknown})"
        ;;
    Darwin)
        PKG_MGR="brew"
        ok "macOS detected"
        ;;
    *)
        err "Unsupported OS: $OS"
        exit 1 ;;
esac

# ── 2. System dependencies ────────────────────────────────────────────────────
hdr "2. System dependencies"

install_pkg() {
    local pkg="$1"
    case "$PKG_MGR" in
        apt)    sudo apt-get install -y -q "$pkg" ;;
        dnf)    sudo dnf install -y "$pkg" ;;
        yum)    sudo yum install -y "$pkg" ;;
        pacman) sudo pacman -S --noconfirm "$pkg" ;;
        zypper) sudo zypper install -y "$pkg" ;;
        brew)   brew install "$pkg" ;;
        *)      warn "Cannot auto-install $pkg — please install manually" ;;
    esac
}

# curl
if ! command -v curl &>/dev/null; then
    info "Installing curl..."
    install_pkg curl
fi
ok "curl"

# git
if ! command -v git &>/dev/null; then
    info "Installing git..."
    install_pkg git
fi
ok "git"

# build tools (needed for some Python packages)
if [[ "$OS" == "Linux" && "$PKG_MGR" == "apt" ]]; then
    if ! dpkg -l build-essential &>/dev/null 2>&1; then
        info "Installing build-essential..."
        sudo apt-get install -y -q build-essential
    fi
    ok "build-essential"
fi

# ── 3. Python ─────────────────────────────────────────────────────────────────
hdr "3. Python"

find_python() {
    for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver="$("$cmd" -c 'import sys; print(sys.version_info.minor)')"
            if [[ "$ver" -ge "$MIN_PYTHON_MINOR" ]]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON=""
if PYTHON="$(find_python)"; then
    PY_VER="$($PYTHON --version 2>&1)"
    ok "Found: $PY_VER ($PYTHON)"
else
    warn "Python 3.$MIN_PYTHON_MINOR+ not found — installing..."
    case "$PKG_MGR" in
        apt)
            sudo apt-get install -y -q python3 python3-pip python3-venv
            ;;
        dnf|yum)
            sudo "$PKG_MGR" install -y python3 python3-pip
            ;;
        pacman)
            sudo pacman -S --noconfirm python python-pip
            ;;
        brew)
            brew install python@3.11
            ;;
        *)
            err "Cannot auto-install Python. Please install Python 3.$MIN_PYTHON_MINOR+ manually."
            err "  https://www.python.org/downloads/"
            exit 1 ;;
    esac
    PYTHON="$(find_python)" || { err "Python install failed"; exit 1; }
    ok "Installed: $($PYTHON --version)"
fi

# pip
if ! "$PYTHON" -m pip --version &>/dev/null; then
    info "Installing pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | "$PYTHON"
fi
ok "pip $($PYTHON -m pip --version | awk '{print $2}')"

# ── 4. Virtual environment ────────────────────────────────────────────────────
hdr "4. Virtual environment"

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment at $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Created .venv"
else
    ok "Existing .venv found"
fi

# Activate
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# Upgrade pip inside venv
"$PIP" install --quiet --upgrade pip wheel setuptools

# ── 5. Python dependencies ────────────────────────────────────────────────────
hdr "5. Python dependencies"

info "Installing from requirements.txt..."
"$PIP" install --quiet -r "$SCRIPT_DIR/requirements.txt"
ok "Core dependencies installed"

if [[ "$DEV_DEPS" == "true" ]]; then
    info "Installing dev dependencies..."
    "$PIP" install --quiet pytest pytest-asyncio black ruff mypy
    ok "Dev dependencies installed"
fi

# ── 6. Ollama ─────────────────────────────────────────────────────────────────
if [[ "$INSTALL_OLLAMA" == "true" ]]; then
    hdr "6. Ollama"

    if command -v ollama &>/dev/null; then
        ok "Ollama already installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
    else
        info "Installing Ollama..."
        if [[ "$OS" == "Darwin" ]]; then
            if command -v brew &>/dev/null; then
                brew install --cask ollama
            else
                curl -fsSL https://ollama.com/install.sh | sh
            fi
        else
            curl -fsSL https://ollama.com/install.sh | sh
        fi

        # Verify
        if command -v ollama &>/dev/null; then
            ok "Ollama installed: $(ollama --version 2>/dev/null || echo 'ok')"
        else
            warn "Ollama install may need a new shell session to be on PATH"
            warn "If 'ollama' is not found, run: export PATH=\$PATH:/usr/local/bin"
        fi
    fi

    # Start Ollama server if not running
    if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
        info "Starting Ollama server..."
        if [[ "$OS" == "Darwin" ]]; then
            # macOS: open the app or start as background process
            if [[ -d "/Applications/Ollama.app" ]]; then
                open -a Ollama &>/dev/null || true
            else
                nohup ollama serve &>/dev/null &
            fi
        else
            # Linux: use systemd if available, else background
            if systemctl is-active --quiet ollama 2>/dev/null; then
                ok "Ollama systemd service already running"
            elif systemctl list-unit-files ollama.service &>/dev/null 2>&1; then
                sudo systemctl enable --now ollama
                ok "Ollama systemd service started"
            else
                nohup ollama serve &>/tmp/ollama.log &
                info "Ollama started in background (log: /tmp/ollama.log)"
            fi
        fi

        # Wait for server
        info "Waiting for Ollama server..."
        for i in $(seq 1 20); do
            if curl -sf http://localhost:11434/api/tags &>/dev/null; then
                ok "Ollama server is ready"
                break
            fi
            sleep 1
            if [[ $i -eq 20 ]]; then
                warn "Ollama server not responding — you may need to run 'ollama serve' manually"
            fi
        done
    else
        ok "Ollama server already running"
    fi

    # Pull model
    hdr "7. Model"
    if [[ "$DEFAULT_MODEL" == "auto" ]]; then
        # Pick smallest model that fits available RAM
        RAM_GB=8
        if command -v free &>/dev/null; then
            RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
        elif [[ "$OS" == "Darwin" ]]; then
            RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
        fi

        if [[ $RAM_GB -ge 12 ]]; then
            DEFAULT_MODEL="qwen2.5:7b"
        elif [[ $RAM_GB -ge 6 ]]; then
            DEFAULT_MODEL="qwen2.5:3b"
        else
            DEFAULT_MODEL="qwen2.5:1.5b"
        fi
        info "Auto-selected model for ${RAM_GB}GB RAM: $DEFAULT_MODEL"
    fi

    # Check if model already pulled
    if curl -sf http://localhost:11434/api/tags 2>/dev/null | grep -q "${DEFAULT_MODEL%%:*}"; then
        ok "Model available: $DEFAULT_MODEL"
    else
        info "Pulling $DEFAULT_MODEL (this may take a few minutes)..."
        ollama pull "$DEFAULT_MODEL"
        ok "Model ready: $DEFAULT_MODEL"
    fi
else
    hdr "6. Ollama"
    warn "Skipped (--no-ollama). Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env"
fi

# ── 7. Environment file ───────────────────────────────────────────────────────
hdr "8. Environment"

ENV_FILE="$SCRIPT_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    cp "$SCRIPT_DIR/.env.example" "$ENV_FILE"
    ok "Created .env from .env.example"
    info "Edit .env to add API keys if using OpenAI/Anthropic"
else
    ok ".env already exists"
fi

# ── 8. Launcher script ────────────────────────────────────────────────────────
hdr "9. Launcher"

LAUNCHER="$SCRIPT_DIR/run.sh"
cat > "$LAUNCHER" << 'LAUNCHER_EOF'
#!/usr/bin/env bash
# Mixture-of-Agents CLI launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# Activate venv
if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
else
    echo "Error: .venv not found. Run ./install.sh first." >&2
    exit 1
fi

# Load .env if present
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

cd "$SCRIPT_DIR"
exec python claude_integrated.py "$@"
LAUNCHER_EOF

chmod +x "$LAUNCHER"
ok "Created run.sh"

# ── 9. Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  Installation complete!                                   ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${GREEN}Start the CLI:${RESET}  ./run.sh"
echo ""
if [[ "$INSTALL_OLLAMA" == "true" ]]; then
    echo -e "  ${BLUE}Model:${RESET}          $DEFAULT_MODEL"
    echo -e "  ${BLUE}Ollama API:${RESET}     http://localhost:11434"
fi
echo -e "  ${BLUE}Config:${RESET}         .env  (add API keys here)"
echo ""
echo -e "  ${YELLOW}Tip:${RESET} Inside the CLI, type /help to see all commands."
echo ""
