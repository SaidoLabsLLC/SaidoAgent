"""Context window management: two-layer compression for long conversations."""
from __future__ import annotations

from saido_agent.core import providers


# -- Token estimation --

def estimate_tokens(messages: list) -> int:
    """Estimate token count by summing content lengths / 3.5."""
    total_chars = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for v in block.values():
                        if isinstance(v, str):
                            total_chars += len(v)
        for tc in m.get("tool_calls", []):
            if isinstance(tc, dict):
                for v in tc.values():
                    if isinstance(v, str):
                        total_chars += len(v)
    return int(total_chars / 3.5)


def get_context_limit(model: str) -> int:
    """Look up context window size for a model."""
    provider_name = providers.detect_provider(model)
    prov = providers.PROVIDERS.get(provider_name, {})
    return prov.get("context_limit", 128000)


# -- Layer 1: Snip old tool results --

def snip_old_tool_results(
    messages: list,
    max_chars: int = 2000,
    preserve_last_n_turns: int = 6,
) -> list:
    """Truncate tool-role messages older than preserve_last_n_turns from end."""
    cutoff = max(0, len(messages) - preserve_last_n_turns)
    for i in range(cutoff):
        m = messages[i]
        if m.get("role") != "tool":
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or len(content) <= max_chars:
            continue
        first_half = content[: max_chars // 2]
        last_quarter = content[-(max_chars // 4):]
        snipped = len(content) - len(first_half) - len(last_quarter)
        m["content"] = f"{first_half}\n[... {snipped} chars snipped ...]\n{last_quarter}"
    return messages


# -- Layer 2: Auto-compact --

def find_split_point(messages: list, keep_ratio: float = 0.3) -> int:
    """Find index that splits messages so ~keep_ratio of tokens are in the recent portion."""
    total = estimate_tokens(messages)
    target = int(total * keep_ratio)
    running = 0
    for i in range(len(messages) - 1, -1, -1):
        running += estimate_tokens([messages[i]])
        if running >= target:
            return i
    return 0


def compact_messages(messages: list, config: dict) -> list:
    """Compress old messages into a summary via LLM call."""
    split = find_split_point(messages)
    if split <= 0:
        return messages

    old = messages[:split]
    recent = messages[split:]

    old_text = ""
    for m in old:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str):
            old_text += f"[{role}]: {content[:500]}\n"
        elif isinstance(content, list):
            old_text += f"[{role}]: (structured content)\n"

    summary_prompt = (
        "Summarize the following conversation history concisely. "
        "Preserve key decisions, file paths, tool results, and context "
        "needed to continue the conversation:\n\n" + old_text
    )

    summary_text = ""
    for event in providers.stream(
        model=config["model"],
        system="You are a concise summarizer.",
        messages=[{"role": "user", "content": summary_prompt}],
        tool_schemas=[],
        config=config,
    ):
        if isinstance(event, providers.TextChunk):
            summary_text += event.text

    summary_msg = {
        "role": "user",
        "content": f"[Previous conversation summary]\n{summary_text}",
    }
    ack_msg = {
        "role": "assistant",
        "content": "Understood. I have the context from the previous conversation. Let's continue.",
    }
    return [summary_msg, ack_msg, *recent]


# -- Main entry --

def maybe_compact(state, config: dict) -> bool:
    """Check if context window is getting full and compress if needed."""
    model = config.get("model", "")
    limit = get_context_limit(model)
    threshold = limit * 0.7

    if estimate_tokens(state.messages) <= threshold:
        return False

    snip_old_tool_results(state.messages)

    if estimate_tokens(state.messages) <= threshold:
        return True

    state.messages = compact_messages(state.messages, config)
    return True
