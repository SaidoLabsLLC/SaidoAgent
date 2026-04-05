"""API layer for Saido Agent (Phase 2+).

Public exports:
    - ``app``        -- FastAPI application instance
    - ``run_server`` -- start the API server via uvicorn
"""

from saido_agent.api.server import app, run_server

__all__ = ["app", "run_server"]
