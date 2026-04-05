"""Automatic knowledge grounding for the agent loop.

Intercepts user messages, performs a lightweight knowledge search,
and injects relevant context into the system prompt before the LLM call.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class KnowledgeGrounder:
    """Automatically grounds agent responses in the knowledge store."""

    def __init__(self, bridge: Any, config: dict | None = None) -> None:
        self._bridge = bridge
        self._config = config or {}
        self._enabled = True
        self._top_k = self._config.get("grounding_top_k", 3)
        self._max_context_chars = self._config.get("grounding_max_chars", 4000)
        self._min_relevance_score = self._config.get("grounding_min_score", 0.1)

    def ground(self, user_message: str) -> Optional[str]:
        """Search knowledge store for relevant context.

        Returns a formatted context string to inject into the system prompt,
        or None if no relevant knowledge found.
        """
        if not self._enabled or not self._bridge:
            return None

        # Guard: check knowledge store has articles
        try:
            stats = self._bridge.stats
            if not stats or stats.get("document_count", 0) == 0:
                return None
        except Exception:
            return None

        # Search for relevant articles
        try:
            results = self._bridge.search(user_message, top_k=self._top_k)
        except Exception as e:
            logger.debug("Knowledge search failed: %s", e)
            return None

        if not results:
            return None

        # Filter by minimum relevance score
        relevant = []
        for r in results:
            score = r.get("score", 0) if isinstance(r, dict) else getattr(r, "score", 0)
            if score >= self._min_relevance_score:
                relevant.append(r)

        if not relevant:
            return None

        # Format context for injection into system prompt
        context_parts: list[str] = []
        total_chars = 0

        for idx, r in enumerate(relevant):
            title = r.get("title", "") if isinstance(r, dict) else getattr(r, "title", "")
            slug = r.get("slug", "") if isinstance(r, dict) else getattr(r, "slug", "")
            snippet = (
                (r.get("snippet", "") or r.get("summary", ""))
                if isinstance(r, dict)
                else (getattr(r, "snippet", "") or getattr(r, "summary", ""))
            )

            # Top result gets full article content; rest get snippets
            content = snippet
            if idx == 0 and self._bridge:
                try:
                    article = self._bridge.read_article(slug)
                    if article:
                        body = (
                            article.get("body", "")
                            if isinstance(article, dict)
                            else getattr(article, "body", "")
                        )
                        if body:
                            content = body[:2000]
                except Exception:
                    pass

            entry = f"### [{title}]\n{content}"

            if total_chars + len(entry) > self._max_context_chars:
                break

            context_parts.append(entry)
            total_chars += len(entry)

        if not context_parts:
            return None

        return (
            "\n\n## Relevant Knowledge Base Articles\n"
            "Use the following knowledge to inform your response. "
            "Cite sources using [Article Title] notation when referencing specific information.\n\n"
            + "\n\n".join(context_parts)
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
