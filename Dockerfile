# =============================================================================
# Saido Agent — Multi-stage Docker build
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install dependencies and build the wheel
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY saido_agent/ saido_agent/

# Build wheel and install into a virtual env so we can copy it cleanly
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2: Runtime — minimal image with only what we need
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Install curl for healthcheck (lightweight addition)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 saido && \
    useradd --uid 1000 --gid saido --create-home --shell /bin/bash saido

# Copy the pre-built virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source (needed for uvicorn module import)
WORKDIR /app
COPY saido_agent/ saido_agent/

# Copy migration files
COPY saido_agent/api/migrations/ saido_agent/api/migrations/

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
ENV SAIDO_KNOWLEDGE_DIR=/data/knowledge \
    SAIDO_JWT_SECRET="" \
    SAIDO_API_PORT=8000 \
    SAIDO_LOG_LEVEL=info \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create data directories with correct ownership
RUN mkdir -p /data/knowledge /home/saido/.saido_agent && \
    chown -R saido:saido /data /home/saido/.saido_agent /app

# Switch to non-root user
USER saido

EXPOSE 8000

# Health check using curl against the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "saido_agent.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
