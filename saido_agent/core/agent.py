"""Core agent loop: neutral message format, multi-provider streaming."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Generator

from saido_agent.core.tool_registry import get_tool_schemas
from saido_agent.core.tools import execute_tool
import saido_agent.core.tools as _tools_init  # ensure built-in tools are registered on import
from saido_agent.core.providers import stream, AssistantTurn, TextChunk, ThinkingChunk, detect_provider
from saido_agent.core.compaction import maybe_compact

# -- Re-export event types (used by cli/repl.py) --
__all__ = [
    "AgentState", "run",
    "TextChunk", "ThinkingChunk",
    "ToolStart", "ToolEnd", "TurnDone", "PermissionRequest",
]


@dataclass
class AgentState:
    """Mutable session state. messages use the neutral provider-independent format."""
    messages: list = field(default_factory=list)
    total_input_tokens:  int = 0
    total_output_tokens: int = 0
    turn_count: int = 0


@dataclass
class ToolStart:
    name:   str
    inputs: dict

@dataclass
class ToolEnd:
    name:      str
    result:    str
    permitted: bool = True

@dataclass
class TurnDone:
    input_tokens:  int
    output_tokens: int

@dataclass
class PermissionRequest:
    description: str
    granted: bool = False


# -- Agent loop --

def run(
    user_message: str,
    state: AgentState,
    config: dict,
    system_prompt: str,
    depth: int = 0,
    cancel_check=None,
) -> Generator:
    """
    Multi-turn agent loop (generator).
    Yields: TextChunk | ThinkingChunk | ToolStart | ToolEnd |
            PermissionRequest | TurnDone
    """
    state.messages.append({"role": "user", "content": user_message})

    config = {**config, "_depth": depth, "_system_prompt": system_prompt}

    while True:
        if cancel_check and cancel_check():
            return
        state.turn_count += 1
        assistant_turn: AssistantTurn | None = None

        maybe_compact(state, config)

        for event in stream(
            model=config["model"],
            system=system_prompt,
            messages=state.messages,
            tool_schemas=get_tool_schemas(),
            config=config,
        ):
            if isinstance(event, (TextChunk, ThinkingChunk)):
                yield event
            elif isinstance(event, AssistantTurn):
                assistant_turn = event

        if assistant_turn is None:
            break

        state.messages.append({
            "role":       "assistant",
            "content":    assistant_turn.text,
            "tool_calls": assistant_turn.tool_calls,
        })

        state.total_input_tokens  += assistant_turn.in_tokens
        state.total_output_tokens += assistant_turn.out_tokens
        yield TurnDone(assistant_turn.in_tokens, assistant_turn.out_tokens)

        if not assistant_turn.tool_calls:
            break

        # -- Execute tools --
        for tc in assistant_turn.tool_calls:
            yield ToolStart(tc["name"], tc["input"])

            permitted = _check_permission(tc, config)
            if not permitted:
                req = PermissionRequest(description=_permission_desc(tc))
                yield req
                permitted = req.granted

            if not permitted:
                result = "Denied: user rejected this operation"
            else:
                result = execute_tool(
                    tc["name"], tc["input"],
                    permission_mode="accept-all",
                    config=config,
                )

            yield ToolEnd(tc["name"], result, permitted)

            state.messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "name":         tc["name"],
                "content":      result,
            })


# -- Helpers --

def _check_permission(tc: dict, config: dict) -> bool:
    """Return True if operation is auto-approved (no need to ask user)."""
    perm_mode = config.get("permission_mode", "auto")
    if perm_mode == "accept-all":
        return True
    if perm_mode == "manual":
        return False

    name = tc["name"]
    if name in ("Read", "Glob", "Grep", "WebFetch", "WebSearch", "AstGrep"):
        return True
    if name == "Bash":
        from saido_agent.core.tools import _is_safe_bash
        return _is_safe_bash(tc["input"].get("command", ""))
    return False


def _permission_desc(tc: dict) -> str:
    name = tc["name"]
    inp  = tc["input"]
    if name == "Bash":   return f"Run: {inp.get('command', '')}"
    if name == "Write":  return f"Write to: {inp.get('file_path', '')}"
    if name == "Edit":   return f"Edit: {inp.get('file_path', '')}"
    return f"{name}({list(inp.values())[:1]})"
