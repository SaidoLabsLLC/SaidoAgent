"""Conversation context extraction for automatic memory creation.

Uses simple heuristic (keyword-based) extraction to identify:
  - Key decisions made during conversation
  - Facts learned about the codebase/project
  - User preferences expressed

LLM-based extraction is planned for Phase 2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExtractedInsight:
    """A single insight extracted from conversation."""
    category: str          # "decision", "fact", or "preference"
    summary: str           # One-line summary
    context: str           # Supporting text from the conversation
    confidence: float      # 0.0 - 1.0 heuristic confidence


# -- Keyword patterns for heuristic extraction --

_DECISION_PATTERNS = [
    re.compile(r"(?:let's|we(?:'ll| will| should)|I(?:'ll| will)|decided to|going to|the plan is to)\s+(.{10,120})", re.IGNORECASE),
    re.compile(r"(?:we chose|I chose|selected|picking|opting for)\s+(.{10,120})", re.IGNORECASE),
]

_FACT_PATTERNS = [
    re.compile(r"(?:the (?:project|codebase|repo|app|system) (?:uses?|is|has|runs?|depends?))\s+(.{10,120})", re.IGNORECASE),
    re.compile(r"(?:built (?:with|on|using)|powered by|running on|deployed (?:to|on|via))\s+(.{10,120})", re.IGNORECASE),
    re.compile(r"(?:the database is|the api uses?|the frontend (?:is|uses?))\s+(.{10,120})", re.IGNORECASE),
]

_PREFERENCE_PATTERNS = [
    re.compile(r"(?:I (?:prefer|like|want|always|never|hate|love))\s+(.{10,120})", re.IGNORECASE),
    re.compile(r"(?:please (?:always|never|don't|do not))\s+(.{10,80})", re.IGNORECASE),
    re.compile(r"(?:from now on|going forward|in the future),?\s+(.{10,120})", re.IGNORECASE),
]


def _extract_user_messages(messages: list[dict]) -> list[str]:
    """Pull plain text from user messages in the conversation."""
    texts: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
    return texts


def _extract_assistant_messages(messages: list[dict]) -> list[str]:
    """Pull plain text from assistant messages."""
    texts: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
    return texts


def _match_patterns(
    texts: list[str],
    patterns: list[re.Pattern],
    category: str,
    confidence: float = 0.6,
) -> list[ExtractedInsight]:
    """Run regex patterns against texts and return insights."""
    insights: list[ExtractedInsight] = []
    seen_summaries: set[str] = set()

    for text in texts:
        for pattern in patterns:
            for match in pattern.finditer(text):
                summary = match.group(1).strip().rstrip(".")
                # Deduplicate by normalized summary
                norm = summary.lower()
                if norm in seen_summaries:
                    continue
                seen_summaries.add(norm)

                # Extract surrounding context (up to 200 chars around match)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 100)
                context = text[start:end].strip()

                insights.append(ExtractedInsight(
                    category=category,
                    summary=summary,
                    context=context,
                    confidence=confidence,
                ))
    return insights


class ConversationExtractor:
    """Extract persistent insights from a conversation using heuristics.

    Usage:
        extractor = ConversationExtractor(messages)
        insights = extractor.extract()
        entries = extractor.to_memory_entries(insights)
    """

    def __init__(self, messages: list[dict]):
        self.messages = messages
        self._user_texts = _extract_user_messages(messages)
        self._assistant_texts = _extract_assistant_messages(messages)

    def extract_decisions(self) -> list[ExtractedInsight]:
        """Extract key decisions from both user and assistant messages."""
        all_texts = self._user_texts + self._assistant_texts
        return _match_patterns(all_texts, _DECISION_PATTERNS, "decision", 0.6)

    def extract_facts(self) -> list[ExtractedInsight]:
        """Extract codebase/project facts from assistant messages."""
        # Facts are more reliable from assistant (which has read the code)
        return _match_patterns(
            self._assistant_texts + self._user_texts,
            _FACT_PATTERNS,
            "fact",
            0.5,
        )

    def extract_preferences(self) -> list[ExtractedInsight]:
        """Extract user preferences from user messages only."""
        return _match_patterns(self._user_texts, _PREFERENCE_PATTERNS, "preference", 0.7)

    def extract(self, min_confidence: float = 0.0) -> list[ExtractedInsight]:
        """Run all extractors and return combined, deduplicated insights."""
        all_insights = (
            self.extract_decisions()
            + self.extract_facts()
            + self.extract_preferences()
        )
        # Filter by confidence
        filtered = [i for i in all_insights if i.confidence >= min_confidence]
        # Sort by confidence descending
        filtered.sort(key=lambda i: i.confidence, reverse=True)
        return filtered

    def to_memory_entries(self, insights: list[ExtractedInsight] | None = None):
        """Convert extracted insights into MemoryEntry objects ready for saving.

        Returns a list of MemoryEntry (imported from store to avoid circular deps).
        """
        from .store import MemoryEntry, _slugify

        if insights is None:
            insights = self.extract()

        entries: list[MemoryEntry] = []
        today = datetime.now().strftime("%Y-%m-%d")

        category_to_type = {
            "decision": "project",
            "fact": "project",
            "preference": "feedback",
        }

        for insight in insights:
            mem_type = category_to_type.get(insight.category, "project")
            name = f"auto_{insight.category}_{_slugify(insight.summary[:40])}"
            entry = MemoryEntry(
                name=name,
                description=f"[auto-extracted {insight.category}] {insight.summary[:80]}",
                type=mem_type,
                content=(
                    f"{insight.summary}\n\n"
                    f"**Context:** {insight.context}\n\n"
                    f"**Confidence:** {insight.confidence:.0%} (heuristic extraction)"
                ),
                created=today,
            )
            entries.append(entry)
        return entries
