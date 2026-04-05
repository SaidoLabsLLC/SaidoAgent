"""Inter-agent messaging system.

Provides a thread-safe, in-memory message bus that allows sub-agents
to communicate with each other during execution.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AgentMessage:
    """A single message between agents."""

    from_agent: str
    to_agent: str
    body: str
    timestamp: float = field(default_factory=time.time)
    broadcast: bool = False


class AgentInbox:
    """Thread-safe message passing between agents.

    Every registered agent gets an ordered queue. Messages are kept in
    memory only -- nothing is persisted to disk.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # agent_id -> list of AgentMessage (append-only per agent)
        self._queues: Dict[str, List[AgentMessage]] = defaultdict(list)
        # agent_id -> index of next unread message
        self._read_cursors: Dict[str, int] = defaultdict(int)
        # set of known agent ids (for broadcast fan-out)
        self._agents: set = set()

    # ── registration ──

    def register(self, agent_id: str) -> None:
        """Register an agent so it can receive messages."""
        with self._lock:
            self._agents.add(agent_id)

    def unregister(self, agent_id: str) -> None:
        """Remove an agent and discard its queue."""
        with self._lock:
            self._agents.discard(agent_id)
            self._queues.pop(agent_id, None)
            self._read_cursors.pop(agent_id, None)

    # ── sending ──

    def send(self, from_agent: str, to_agent: str, message: str) -> bool:
        """Send a message from one agent to another.

        Returns True if the target agent is registered, False otherwise.
        """
        with self._lock:
            if to_agent not in self._agents:
                return False
            msg = AgentMessage(
                from_agent=from_agent,
                to_agent=to_agent,
                body=message,
            )
            self._queues[to_agent].append(msg)
            return True

    def broadcast(self, from_agent: str, message: str) -> int:
        """Broadcast a message to all registered agents except the sender.

        Returns the number of agents that received the message.
        """
        with self._lock:
            targets = [a for a in self._agents if a != from_agent]
            msg_template = AgentMessage(
                from_agent=from_agent,
                to_agent="*",
                body=message,
                broadcast=True,
            )
            for target in targets:
                # Each queue gets its own copy conceptually, but since
                # AgentMessage is frozen we can safely share the object.
                self._queues[target].append(msg_template)
            return len(targets)

    # ── receiving ──

    def receive(self, agent_id: str) -> List[dict]:
        """Return all unread messages for *agent_id* as plain dicts.

        Advances the read cursor so messages are not returned twice.
        """
        with self._lock:
            q = self._queues.get(agent_id, [])
            cursor = self._read_cursors.get(agent_id, 0)
            unread = q[cursor:]
            self._read_cursors[agent_id] = len(q)

        return [
            {
                "from": m.from_agent,
                "to": m.to_agent,
                "body": m.body,
                "timestamp": m.timestamp,
                "broadcast": m.broadcast,
            }
            for m in unread
        ]

    def peek(self, agent_id: str) -> int:
        """Return the count of unread messages without consuming them."""
        with self._lock:
            q = self._queues.get(agent_id, [])
            cursor = self._read_cursors.get(agent_id, 0)
            return max(0, len(q) - cursor)

    # ── introspection ──

    @property
    def registered_agents(self) -> List[str]:
        with self._lock:
            return sorted(self._agents)

    def all_messages(self, agent_id: str) -> List[dict]:
        """Return *all* messages (read and unread) for an agent."""
        with self._lock:
            q = list(self._queues.get(agent_id, []))
        return [
            {
                "from": m.from_agent,
                "to": m.to_agent,
                "body": m.body,
                "timestamp": m.timestamp,
                "broadcast": m.broadcast,
            }
            for m in q
        ]
