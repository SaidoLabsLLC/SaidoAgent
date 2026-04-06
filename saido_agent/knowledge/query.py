"""Knowledge-grounded Q&A with citation support.

Retrieval is fully delegated to SmartRAG via KnowledgeBridge.
This module handles:
  - Mapping SmartRAG results to a grounded Q&A prompt
  - LLM answer generation with citation extraction
  - Citation validation against the knowledge store
  - Confidence assessment (HIGH / MEDIUM / LOW)
  - Conversational context (last N turns)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from saido_agent.knowledge.bridge import KnowledgeBridge
    from saido_agent.core.routing import ModelRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Citation regex — matches [Article Title] in the LLM response
# ---------------------------------------------------------------------------
_CITATION_RE = re.compile(r"\[([^\[\]]{2,})\]")

# Hedging language that indicates lower confidence
_HEDGING_PATTERNS = re.compile(
    r"\b(I'm not sure|i'm not sure|not certain|may |might |possibly|"
    r"could be|unclear|uncertain|I don't have enough|insufficient|"
    r"cannot determine|hard to say|speculative)\b",
    re.IGNORECASE,
)

# Maximum conversation history turns to retain
_MAX_HISTORY = 5

# How many top results get full article content
_FULL_CONTENT_COUNT = 3


@dataclass
class Citation:
    """A single citation extracted from an LLM answer."""

    slug: str
    title: str
    excerpt: str = ""
    verified: bool = True


@dataclass
class SaidoQueryResult:
    """Complete result from a knowledge-grounded Q&A query."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: str = "medium"  # high, medium, low
    retrieval_stats: dict = field(default_factory=dict)
    tokens_used: int = 0
    provider: str = ""


class KnowledgeQA:
    """Knowledge-grounded Q&A with citation support.

    Orchestrates: bridge retrieval -> prompt construction -> LLM generation
    -> citation extraction/validation -> confidence assessment.
    """

    def __init__(
        self,
        bridge: KnowledgeBridge,
        model_router: ModelRouter | None = None,
    ) -> None:
        self._bridge = bridge
        self._router = model_router
        self._conversation_history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        context: dict | None = None,
    ) -> SaidoQueryResult:
        """Answer a question grounded in the knowledge store.

        Workflow:
          1. Check knowledge store is non-empty
          2. Retrieve relevant articles via bridge
          3. Build grounded Q&A prompt with full + snippet articles
          4. Send to LLM via ModelRouter
          5. Extract and validate citations
          6. Assess confidence
          7. Update conversation history
        """
        # -- Guard: empty knowledge store ----------------------------------
        stats = self._bridge.stats
        doc_count = stats.get("document_count", 0)
        if doc_count == 0:
            return SaidoQueryResult(
                answer=(
                    "No knowledge base articles found. "
                    "Ingest documents first with /ingest."
                ),
                confidence="low",
                retrieval_stats={"document_count": 0, "results_found": 0},
            )

        # -- Step 1: Retrieve via bridge -----------------------------------
        search_results = self._bridge.search(question, top_k=5)
        retrieval_stats: dict[str, Any] = {
            "document_count": doc_count,
            "results_found": len(search_results),
        }

        if not search_results:
            return SaidoQueryResult(
                answer=(
                    "I could not find any relevant articles for your question. "
                    "Try rephrasing or ingesting more documents."
                ),
                confidence="low",
                retrieval_stats=retrieval_stats,
            )

        # -- Step 2: Build article context ---------------------------------
        # Top-3 get full article content; remainder use snippet only
        articles_context = self._build_articles_context(search_results)
        retrieval_stats["full_articles"] = min(
            _FULL_CONTENT_COUNT, len(search_results)
        )
        retrieval_stats["snippet_articles"] = max(
            0, len(search_results) - _FULL_CONTENT_COUNT
        )

        # -- Step 3: Build prompt ------------------------------------------
        prompt = self._build_prompt(question, articles_context)

        # -- Step 4: Call LLM ----------------------------------------------
        llm_text, tokens_used, provider = self._call_llm(prompt)

        if llm_text is None:
            return SaidoQueryResult(
                answer="LLM call failed. Please check model availability.",
                confidence="low",
                retrieval_stats=retrieval_stats,
            )

        # -- Step 5: Extract and validate citations ------------------------
        # Build a title -> slug lookup from the search results
        title_to_slug = self._build_title_slug_map(search_results)
        citations = self._extract_citations(llm_text, title_to_slug)

        # -- Step 6: Assess confidence -------------------------------------
        confidence = self._assess_confidence(llm_text, citations)

        # -- Step 7: Update conversation history ---------------------------
        self._append_history(question, llm_text)

        return SaidoQueryResult(
            answer=llm_text,
            citations=citations,
            confidence=confidence,
            retrieval_stats=retrieval_stats,
            tokens_used=tokens_used,
            provider=provider,
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the knowledge store without generating an answer.

        Returns a list of dicts with: slug, title, summary, score, snippet.
        """
        results = self._bridge.search(query, top_k=top_k)
        return [
            {
                "slug": r.slug,
                "title": r.title,
                "summary": r.summary,
                "score": r.score,
                "snippet": r.summary,  # SmartRAG uses summary as snippet
            }
            for r in results
        ]

    def clear_history(self) -> None:
        """Reset conversation context."""
        self._conversation_history.clear()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_articles_context(
        self, search_results: list,
    ) -> list[dict[str, str]]:
        """Build article context blocks from search results.

        Top N results get full article body; the rest get snippet only.
        """
        articles: list[dict[str, str]] = []
        for idx, result in enumerate(search_results):
            entry: dict[str, str] = {
                "title": result.title,
                "slug": result.slug,
            }
            if idx < _FULL_CONTENT_COUNT:
                doc = self._bridge.read_article(result.slug)
                if doc is not None:
                    entry["content"] = doc.body
                    entry["mode"] = "full"
                else:
                    entry["content"] = result.summary
                    entry["mode"] = "snippet"
            else:
                entry["content"] = result.summary
                entry["mode"] = "snippet"
            articles.append(entry)
        return articles

    def _build_prompt(
        self,
        question: str,
        articles: list[dict[str, str]],
    ) -> str:
        """Assemble the grounded Q&A system + user prompt."""
        sections: list[str] = []

        # System preamble
        sections.append(
            "You are Saido Agent, a knowledge-grounded AI assistant. /no_think\n"
            "Answer the question using ONLY the provided knowledge base "
            "articles. Cite your sources using [Article Title] notation.\n\n"
            "If the knowledge base does not contain enough information to "
            "fully answer the question, say so explicitly. Do not make up "
            "information.\n\nBe concise and direct."
        )

        # Knowledge base articles
        sections.append("\n## Knowledge Base Articles\n")
        for art in articles:
            mode_label = "" if art["mode"] == "full" else " (snippet only)"
            sections.append(
                f"### [{art['title']}] (slug: {art['slug']}){mode_label}\n"
                f"{art['content']}\n"
            )

        # Conversation history (if any)
        if self._conversation_history:
            sections.append("\n## Previous Conversation Context\n")
            for turn in self._conversation_history:
                sections.append(f"User: {turn['question']}")
                # Include a brief summary of the previous answer
                prev_answer = turn["answer"]
                if len(prev_answer) > 300:
                    prev_answer = prev_answer[:300] + "..."
                sections.append(f"Assistant: {prev_answer}\n")

        # Current question
        sections.append(f"\n## Question\n{question}")

        # Instructions
        sections.append(
            "\n## Instructions\n"
            "- Answer concisely and accurately\n"
            "- Cite every claim with [Article Title]\n"
            "- If information is insufficient, state what's missing\n"
            "- Confidence: state HIGH if answer is fully supported, "
            "MEDIUM if partially, LOW if speculative"
        )

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> tuple[str | None, int, str]:
        """Send the prompt to the LLM via ModelRouter.

        Returns (response_text, tokens_used, provider_name).
        """
        if self._router is None:
            logger.warning("No ModelRouter configured — cannot call LLM")
            return None, 0, ""

        try:
            from saido_agent.core.providers import stream as llm_stream

            provider, model = self._router.select_model("qa")
            # Append /no_think for Qwen 3 models to skip reasoning and respond faster
            effective_prompt = prompt + "\n/no_think" if "qwen" in model.lower() else prompt
            messages = [{"role": "user", "content": effective_prompt}]
            total_text = ""
            tokens = 0

            thinking_text = ""
            for chunk in llm_stream(
                model=f"{provider}/{model}" if provider not in ("anthropic",) else model,
                system="",
                messages=messages,
                tool_schemas=[],
                config={"max_tokens": 2048, "no_tools": True},
            ):
                from saido_agent.core.providers import (
                    AssistantTurn,
                    TextChunk,
                    ThinkingChunk,
                )

                if isinstance(chunk, TextChunk):
                    total_text += chunk.text
                elif isinstance(chunk, ThinkingChunk):
                    thinking_text += chunk.text
                elif isinstance(chunk, AssistantTurn):
                    if chunk.text:
                        total_text = chunk.text
                    tokens = chunk.in_tokens + chunk.out_tokens

            # If model only produced thinking (no content), use thinking as the answer
            if not total_text.strip() and thinking_text.strip():
                total_text = thinking_text.strip()

            return total_text, tokens, f"{provider}/{model}"

        except Exception as exc:
            try:
                logger.error("LLM call failed: %s", str(exc).encode("ascii", "replace").decode())
            except Exception:
                logger.error("LLM call failed (details not printable)")
            return None, 0, ""

    # ------------------------------------------------------------------
    # Citation extraction & validation
    # ------------------------------------------------------------------

    def _build_title_slug_map(self, search_results: list) -> dict[str, str]:
        """Build a title -> slug lookup from search results."""
        mapping: dict[str, str] = {}
        for result in search_results:
            mapping[result.title] = result.slug
        return mapping

    def _extract_citations(
        self,
        answer: str,
        title_to_slug: dict[str, str],
    ) -> list[Citation]:
        """Extract [Title] citations from the answer and validate them."""
        seen_titles: set[str] = set()
        citations: list[Citation] = []

        for match in _CITATION_RE.finditer(answer):
            title = match.group(1).strip()
            if title in seen_titles:
                continue
            seen_titles.add(title)

            # Skip instruction-like patterns that are not actual citations
            if title.lower() in (
                "article title",
                "article 1 title",
                "article 2 title",
                "article 3 title",
            ):
                continue

            slug = title_to_slug.get(title)
            if slug is not None:
                # Extract a small excerpt around the citation for context
                excerpt = self._extract_excerpt(answer, match.start())
                citations.append(
                    Citation(
                        slug=slug,
                        title=title,
                        excerpt=excerpt,
                        verified=True,
                    )
                )
            else:
                # Citation references an article not in our results
                citations.append(
                    Citation(
                        slug="",
                        title=title,
                        excerpt="",
                        verified=False,
                    )
                )
        return citations

    @staticmethod
    def _extract_excerpt(text: str, citation_pos: int) -> str:
        """Extract a brief excerpt of text surrounding a citation position."""
        start = max(0, citation_pos - 100)
        end = min(len(text), citation_pos + 100)
        excerpt = text[start:end].strip()
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(text):
            excerpt = excerpt + "..."
        return excerpt

    # ------------------------------------------------------------------
    # Confidence assessment
    # ------------------------------------------------------------------

    @staticmethod
    def _assess_confidence(answer: str, citations: list[Citation]) -> str:
        """Determine answer confidence: high, medium, or low.

        Rules:
          HIGH   - 2+ verified citations, no hedging language
          MEDIUM - 1 verified citation, or some hedging
          LOW    - 0 verified citations, or heavy hedging / explicit uncertainty
        """
        # Check if LLM explicitly stated confidence
        explicit = re.search(
            r"(?:confidence|Confidence)\s*:\s*(HIGH|MEDIUM|LOW)",
            answer,
            re.IGNORECASE,
        )
        if explicit:
            return explicit.group(1).lower()

        verified_count = sum(1 for c in citations if c.verified)
        has_hedging = bool(_HEDGING_PATTERNS.search(answer))

        if verified_count >= 2 and not has_hedging:
            return "high"
        elif verified_count >= 1 and not has_hedging:
            return "medium"
        elif verified_count >= 1 and has_hedging:
            return "medium"
        else:
            return "low"

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def _append_history(self, question: str, answer: str) -> None:
        """Add a turn to conversation history, enforcing max length."""
        self._conversation_history.append(
            {"question": question, "answer": answer}
        )
        if len(self._conversation_history) > _MAX_HISTORY:
            self._conversation_history = self._conversation_history[
                -_MAX_HISTORY:
            ]
