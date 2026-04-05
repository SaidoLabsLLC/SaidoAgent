"""WebSocket real-time events for Saido Agent.

Provides:
  - Per-team event channels for real-time collaboration
  - JWT-authenticated WebSocket connections
  - Event types: article_created, article_updated, article_deleted, member_joined
  - Connection manager with proper lifecycle handling

Usage::

    # Client connects with JWT in query param:
    ws = websocket.connect("ws://host/v1/ws?token=<jwt>")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

ws_router = APIRouter()


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

EVENT_ARTICLE_CREATED = "article_created"
EVENT_ARTICLE_UPDATED = "article_updated"
EVENT_ARTICLE_DELETED = "article_deleted"
EVENT_MEMBER_JOINED = "member_joined"


@dataclass
class WSEvent:
    """A real-time event to broadcast to team members."""

    event_type: str
    team_id: str
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({
            "event": self.event_type,
            "team_id": self.team_id,
            "data": self.data,
            "timestamp": self.timestamp,
        })


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages WebSocket connections organized by team_id.

    Each team has a set of active connections. Events are broadcast
    only to members of the relevant team.
    """

    def __init__(self):
        # { team_id: { (user_id, websocket), ... } }
        self._connections: dict[str, set[tuple[str, WebSocket]]] = {}

    async def connect(
        self, websocket: WebSocket, team_id: str, user_id: str
    ) -> None:
        """Accept a WebSocket connection and register it for a team."""
        await websocket.accept()
        if team_id not in self._connections:
            self._connections[team_id] = set()
        self._connections[team_id].add((user_id, websocket))
        logger.info(
            "WebSocket connected: user=%s team=%s (total=%d)",
            user_id,
            team_id,
            len(self._connections[team_id]),
        )

    def disconnect(
        self, websocket: WebSocket, team_id: str, user_id: str
    ) -> None:
        """Remove a WebSocket connection from a team's channel."""
        if team_id in self._connections:
            self._connections[team_id].discard((user_id, websocket))
            if not self._connections[team_id]:
                del self._connections[team_id]
        logger.info("WebSocket disconnected: user=%s team=%s", user_id, team_id)

    async def broadcast(self, event: WSEvent) -> None:
        """Broadcast an event to all connections in the event's team."""
        team_id = event.team_id
        if team_id not in self._connections:
            return

        message = event.to_json()
        dead_connections = []

        for user_id, ws in self._connections[team_id]:
            try:
                await ws.send_text(message)
            except Exception:
                dead_connections.append((user_id, ws))

        # Clean up dead connections
        for conn in dead_connections:
            self._connections[team_id].discard(conn)

    def get_team_connection_count(self, team_id: str) -> int:
        """Return the number of active connections for a team."""
        return len(self._connections.get(team_id, set()))

    def clear(self) -> None:
        """Clear all connections (for testing/shutdown)."""
        self._connections.clear()


# Singleton manager instance
manager = ConnectionManager()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@ws_router.websocket("/v1/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
    team_id: Optional[str] = Query(default=None),
):
    """WebSocket endpoint for real-time team events.

    Query parameters:
      - ``token``: JWT token for authentication
      - ``team_id``: Team ID to subscribe to events for

    The connection is rejected if:
      - No token is provided
      - The token is invalid or expired
      - No team_id is provided
      - The user is not a member of the team
    """
    from saido_agent.api.auth import verify_jwt_token
    from saido_agent.api.users import get_member_role

    # Validate token
    if not token:
        await websocket.close(code=4001, reason="Missing authentication token")
        return

    jwt_payload = verify_jwt_token(token)
    if jwt_payload is None:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    # jwt_payload can be a string (tenant_id) or dict with user_id
    # For backward compat, handle both
    if isinstance(jwt_payload, dict):
        user_id = jwt_payload.get("user_id", jwt_payload.get("tenant_id", ""))
    else:
        user_id = jwt_payload

    if not team_id:
        await websocket.close(code=4002, reason="Missing team_id parameter")
        return

    # Verify team membership
    role = get_member_role(team_id, user_id)
    if role is None:
        await websocket.close(code=4003, reason="Not a member of this team")
        return

    # Connection accepted -- enter receive loop
    await manager.connect(websocket, team_id, user_id)
    try:
        while True:
            # Keep connection alive; client can send pings or messages
            data = await websocket.receive_text()
            # Echo back acknowledgment for client-side pings
            if data == "ping":
                await websocket.send_text(json.dumps({"event": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket, team_id, user_id)
    except Exception:
        manager.disconnect(websocket, team_id, user_id)
