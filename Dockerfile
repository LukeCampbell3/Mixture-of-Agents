# =============================================================================
# Mixture-of-Agents CLI — Docker image
# =============================================================================
# Build:  docker build -t moa-cli .
# Run:    docker run -it --rm -v $(pwd)/data:/app/data moa-cli
# With Ollama on host:
#   docker run -it --rm \
#     --add-host=host.docker.internal:host-gateway \
#     -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
#     -v $(pwd)/data:/app/data \
#     moa-cli
# =============================================================================

FROM python:3.11-slim AS base

# System deps — minimal set, no build tools unless needed
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies layer (cached unless requirements.txt changes) ───────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel \
 && pip install --no-cache-dir -r requirements.txt

# ── Application layer ─────────────────────────────────────────────────────────
FROM deps AS app

# Copy source
COPY app/           ./app/
COPY configs/       ./configs/
COPY claude_integrated.py .
COPY .env.example   .

# Create persistent data directory
RUN mkdir -p /app/data

# Non-root user for security
RUN useradd -m -u 1000 moa && chown -R moa:moa /app
USER moa

# Environment defaults (override at runtime)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    DATA_DIR=/app/data

# Health check — verifies Python env is intact
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from app.orchestrator import Orchestrator; print('ok')" || exit 1

CMD ["python", "claude_integrated.py"]
