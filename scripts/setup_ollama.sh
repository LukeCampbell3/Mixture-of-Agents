#!/usr/bin/env bash
# =============================================================================
# Ollama model management helper
# =============================================================================
# Usage:
#   ./scripts/setup_ollama.sh              # pull auto-selected model
#   ./scripts/setup_ollama.sh 7b           # pull qwen2.5:7b
#   ./scripts/setup_ollama.sh list         # list available models
#   ./scripts/setup_ollama.sh status       # check server status
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}✓${RESET} $*"; }
warn() { echo -e "${YELLOW}⚠${RESET} $*"; }
err()  { echo -e "${RED}✗${RESET} $*" >&2; exit 1; }

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
ACTION="${1:-auto}"

# ── Status check ──────────────────────────────────────────────────────────────
check_server() {
    if curl -sf "$OLLAMA_URL/api/tags" &>/dev/null; then
        ok "Ollama server running at $OLLAMA_URL"
        return 0
    else
        warn "Ollama server not reachable at $OLLAMA_URL"
        return 1
    fi
}

# ── List models ───────────────────────────────────────────────────────────────
list_models() {
    echo "Available models:"
    curl -sf "$OLLAMA_URL/api/tags" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for m in data.get('models', []):
    size = m.get('size', 0)
    print(f\"  {m['name']:<40} {size // 1024 // 1024:>6} MB\")
" 2>/dev/null || echo "  (none pulled yet)"
}

# ── Auto-select model based on RAM ────────────────────────────────────────────
auto_model() {
    local ram_gb=8
    if command -v free &>/dev/null; then
        ram_gb=$(free -g | awk '/^Mem:/{print $2}')
    elif [[ "$(uname)" == "Darwin" ]]; then
        ram_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
    fi

    if [[ $ram_gb -ge 12 ]]; then echo "qwen2.5:7b"
    elif [[ $ram_gb -ge 6 ]]; then echo "qwen2.5:3b"
    else echo "qwen2.5:1.5b"
    fi
}

# ── Pull model ────────────────────────────────────────────────────────────────
pull_model() {
    local model="$1"
    echo "Pulling $model..."
    ollama pull "$model"
    ok "Model ready: $model"
}

# ── Main ──────────────────────────────────────────────────────────────────────
case "$ACTION" in
    status)
        check_server
        list_models
        ;;
    list)
        check_server && list_models
        ;;
    auto)
        check_server || err "Start Ollama first: ollama serve"
        MODEL=$(auto_model)
        echo "Auto-selected: $MODEL"
        pull_model "$MODEL"
        ;;
    [0-9]*b|*:[0-9]*)
        # e.g. "7b" or "qwen2.5:7b"
        check_server || err "Start Ollama first: ollama serve"
        if [[ "$ACTION" =~ ^[0-9] ]]; then
            MODEL="qwen2.5:$ACTION"
        else
            MODEL="$ACTION"
        fi
        pull_model "$MODEL"
        ;;
    *)
        echo "Usage: $0 [auto|status|list|<size>|<model:tag>]"
        echo "  auto          Auto-select based on RAM (default)"
        echo "  status        Check server and list models"
        echo "  list          List pulled models"
        echo "  1.5b          Pull qwen2.5:1.5b"
        echo "  7b            Pull qwen2.5:7b"
        echo "  llama3.2:3b   Pull any model by name"
        ;;
esac
