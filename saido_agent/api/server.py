"""FastAPI application for Saido Agent REST API.

Entry point::

    uvicorn saido_agent.api.server:app --host 0.0.0.0 --port 8000

Or via the SDK::

    from saido_agent import SaidoAgent
    SaidoAgent.serve(port=8000)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from saido_agent.api.auth import get_tenant_knowledge_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tenant agent cache
# ---------------------------------------------------------------------------

_agent_cache: dict[str, Any] = {}


def get_agent_for_tenant(tenant_id: str) -> Any:
    """Return a SaidoAgent scoped to the tenant's knowledge directory.

    Agents are cached by tenant_id so repeated requests reuse the same
    instance (and its in-memory state like conversation history).
    """
    if tenant_id in _agent_cache:
        return _agent_cache[tenant_id]

    from saido_agent import SaidoAgent

    knowledge_dir = get_tenant_knowledge_dir(tenant_id)
    agent = SaidoAgent(knowledge_dir=knowledge_dir)
    _agent_cache[tenant_id] = agent
    logger.info("Created SaidoAgent for tenant %s at %s", tenant_id, knowledge_dir)
    return agent


def clear_agent_cache() -> None:
    """Clear the tenant agent cache (for shutdown / testing)."""
    _agent_cache.clear()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    logger.info("Saido Agent API starting up")
    yield
    logger.info("Saido Agent API shutting down — clearing agent cache")
    clear_agent_cache()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Saido Agent API",
    version="0.1.0",
    description="Knowledge-compounding AI agent REST API by Saido Labs LLC",
    lifespan=lifespan,
)

# CORS middleware — restrict origins via SAIDO_CORS_ORIGINS env var
_cors_origins = os.environ.get("SAIDO_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


# ---------------------------------------------------------------------------
# Health check (no auth required)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "healthy", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# Mount versioned routes
# ---------------------------------------------------------------------------

from saido_agent.api.routes import v1_router  # noqa: E402
from saido_agent.api.websocket import ws_router  # noqa: E402

app.include_router(v1_router)
app.include_router(ws_router)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_server(host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
    """Start the API server via uvicorn.

    Called by ``SaidoAgent.serve()`` and the ``/serve`` CLI command.
    """
    import uvicorn

    uvicorn.run(
        "saido_agent.api.server:app",
        host=host,
        port=port,
        log_level="info",
        **kwargs,
    )
