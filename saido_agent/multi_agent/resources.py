"""Resource management for sub-agents.

Provides configurable limits (tokens, turns, timeout, tool calls) per agent,
tracks usage in real time, and kills agents that exceed their budgets.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AgentResourceLimits:
    """Configurable resource budget for a single sub-agent."""

    max_tokens: int = 100_000
    max_turns: int = 50
    timeout_seconds: int = 300
    max_tool_calls: int = 100


@dataclass
class AgentResourceUsage:
    """Tracks cumulative resource consumption for a single sub-agent."""

    tokens_used: int = 0
    turns_used: int = 0
    tool_calls_used: int = 0
    start_time: float = field(default_factory=time.monotonic)
    end_time: Optional[float] = None
    exceeded_limit: Optional[str] = None

    # ── mutators (all thread-safe via GIL for simple int increments) ──

    def add_tokens(self, count: int) -> None:
        self.tokens_used += count

    def add_turn(self) -> None:
        self.turns_used += 1

    def add_tool_call(self) -> None:
        self.tool_calls_used += 1

    def mark_finished(self) -> None:
        self.end_time = time.monotonic()

    # ── queries ──

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time if self.end_time is not None else time.monotonic()
        return end - self.start_time

    def check_limits(self, limits: AgentResourceLimits) -> Optional[str]:
        """Return a human-readable reason string if any limit is exceeded,
        otherwise return ``None``."""
        if self.tokens_used >= limits.max_tokens:
            return f"token budget exceeded ({self.tokens_used}/{limits.max_tokens})"
        if self.turns_used >= limits.max_turns:
            return f"turn limit exceeded ({self.turns_used}/{limits.max_turns})"
        if self.tool_calls_used >= limits.max_tool_calls:
            return f"tool-call limit exceeded ({self.tool_calls_used}/{limits.max_tool_calls})"
        if self.elapsed_seconds >= limits.timeout_seconds:
            return f"timeout exceeded ({self.elapsed_seconds:.0f}s/{limits.timeout_seconds}s)"
        return None

    def summary(self, limits: Optional[AgentResourceLimits] = None) -> dict:
        """Return a serialisable summary dict suitable for reporting."""
        d: dict = {
            "tokens_used": self.tokens_used,
            "turns_used": self.turns_used,
            "tool_calls_used": self.tool_calls_used,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }
        if limits is not None:
            d["limits"] = {
                "max_tokens": limits.max_tokens,
                "max_turns": limits.max_turns,
                "max_tool_calls": limits.max_tool_calls,
                "timeout_seconds": limits.timeout_seconds,
            }
        if self.exceeded_limit:
            d["exceeded_limit"] = self.exceeded_limit
        return d


class ResourceTracker:
    """Central registry that maps agent IDs to their limits and usage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._limits: Dict[str, AgentResourceLimits] = {}
        self._usage: Dict[str, AgentResourceUsage] = {}

    def register(
        self,
        agent_id: str,
        limits: Optional[AgentResourceLimits] = None,
    ) -> AgentResourceUsage:
        """Register a new agent and start tracking."""
        lim = limits or AgentResourceLimits()
        usage = AgentResourceUsage()
        with self._lock:
            self._limits[agent_id] = lim
            self._usage[agent_id] = usage
        return usage

    def get_usage(self, agent_id: str) -> Optional[AgentResourceUsage]:
        with self._lock:
            return self._usage.get(agent_id)

    def get_limits(self, agent_id: str) -> Optional[AgentResourceLimits]:
        with self._lock:
            return self._limits.get(agent_id)

    def record_tokens(self, agent_id: str, count: int) -> Optional[str]:
        """Record token usage. Returns exceeded-limit reason or None."""
        with self._lock:
            usage = self._usage.get(agent_id)
            limits = self._limits.get(agent_id)
        if usage is None or limits is None:
            return None
        usage.add_tokens(count)
        return usage.check_limits(limits)

    def record_turn(self, agent_id: str) -> Optional[str]:
        with self._lock:
            usage = self._usage.get(agent_id)
            limits = self._limits.get(agent_id)
        if usage is None or limits is None:
            return None
        usage.add_turn()
        return usage.check_limits(limits)

    def record_tool_call(self, agent_id: str) -> Optional[str]:
        with self._lock:
            usage = self._usage.get(agent_id)
            limits = self._limits.get(agent_id)
        if usage is None or limits is None:
            return None
        usage.add_tool_call()
        return usage.check_limits(limits)

    def check(self, agent_id: str) -> Optional[str]:
        """Check current limits without recording anything."""
        with self._lock:
            usage = self._usage.get(agent_id)
            limits = self._limits.get(agent_id)
        if usage is None or limits is None:
            return None
        return usage.check_limits(limits)

    def finish(self, agent_id: str) -> Optional[dict]:
        """Mark agent finished and return summary."""
        with self._lock:
            usage = self._usage.get(agent_id)
            limits = self._limits.get(agent_id)
        if usage is None:
            return None
        usage.mark_finished()
        return usage.summary(limits)

    def remove(self, agent_id: str) -> None:
        with self._lock:
            self._limits.pop(agent_id, None)
            self._usage.pop(agent_id, None)
