"""Tests for saido_agent.knowledge.query — KnowledgeQA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from saido_agent.knowledge.query import (
    Citation,
    KnowledgeQA,
    SaidoQueryResult,
    _CITATION_RE,
    _FULL_CONTENT_COUNT,
    _MAX_HISTORY,
)


# ---------------------------------------------------------------------------
# Fake SmartRAG types (avoid importing real SmartRAG in unit tests)
# ---------------------------------------------------------------------------


@dataclass
class FakeSearchResult:
    slug: str
    title: str
    summary: str
    score: float
    categories: list[str] = field(default_factory=list)


@dataclass
class FakeDocument:
    slug: str
    title: str
    body: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    word_count: int = 100
    has_children: bool = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ARTICLES = {
    "python-basics": FakeDocument(
        slug="python-basics",
        title="Python Basics",
        body="Python is a high-level programming language created by Guido van Rossum. "
        "It supports multiple paradigms including OOP and functional programming.",
    ),
    "python-async": FakeDocument(
        slug="python-async",
        title="Python Async Programming",
        body="Python asyncio provides infrastructure for writing single-threaded "
        "concurrent code using coroutines. Use async/await syntax.",
    ),
    "rust-ownership": FakeDocument(
        slug="rust-ownership",
        title="Rust Ownership Model",
        body="Rust uses an ownership system with borrowing and lifetimes "
        "to guarantee memory safety without a garbage collector.",
    ),
}

SEARCH_RESULTS = [
    FakeSearchResult(
        slug="python-basics",
        title="Python Basics",
        summary="Python is a high-level programming language.",
        score=0.95,
    ),
    FakeSearchResult(
        slug="python-async",
        title="Python Async Programming",
        summary="Python asyncio provides concurrent code infrastructure.",
        score=0.82,
    ),
    FakeSearchResult(
        slug="rust-ownership",
        title="Rust Ownership Model",
        summary="Rust uses ownership for memory safety.",
        score=0.45,
    ),
]


@pytest.fixture()
def mock_bridge() -> MagicMock:
    """Return a mock KnowledgeBridge with canned data."""
    bridge = MagicMock()
    bridge.stats = {"document_count": 3, "index_size_bytes": 1024}
    bridge.search.return_value = SEARCH_RESULTS
    bridge.read_article.side_effect = lambda slug: ARTICLES.get(slug)
    return bridge


@pytest.fixture()
def mock_router() -> MagicMock:
    """Return a mock ModelRouter."""
    router = MagicMock()
    router.select_model.return_value = ("ollama", "qwen3:8b")
    return router


@pytest.fixture()
def qa(mock_bridge: MagicMock, mock_router: MagicMock) -> KnowledgeQA:
    """Return a KnowledgeQA wired to mocked bridge + router."""
    return KnowledgeQA(bridge=mock_bridge, model_router=mock_router)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Verify the Q&A prompt includes retrieved articles correctly."""

    def test_prompt_includes_full_articles(
        self, qa: KnowledgeQA, mock_bridge: MagicMock
    ) -> None:
        articles_ctx = qa._build_articles_context(SEARCH_RESULTS)
        # Top _FULL_CONTENT_COUNT should attempt full reads
        assert mock_bridge.read_article.call_count == _FULL_CONTENT_COUNT

    def test_prompt_includes_snippet_articles(
        self, qa: KnowledgeQA
    ) -> None:
        articles_ctx = qa._build_articles_context(SEARCH_RESULTS)
        full = [a for a in articles_ctx if a["mode"] == "full"]
        snippet = [a for a in articles_ctx if a["mode"] == "snippet"]
        assert len(full) == _FULL_CONTENT_COUNT
        assert len(snippet) == len(SEARCH_RESULTS) - _FULL_CONTENT_COUNT

    def test_prompt_contains_question(self, qa: KnowledgeQA) -> None:
        prompt = qa._build_prompt(
            "What is Python?",
            [
                {
                    "title": "Python Basics",
                    "slug": "python-basics",
                    "content": "Python is great.",
                    "mode": "full",
                }
            ],
        )
        assert "What is Python?" in prompt

    def test_prompt_contains_article_titles(self, qa: KnowledgeQA) -> None:
        prompt = qa._build_prompt(
            "question",
            [
                {
                    "title": "Python Basics",
                    "slug": "python-basics",
                    "content": "content",
                    "mode": "full",
                }
            ],
        )
        assert "[Python Basics]" in prompt
        assert "python-basics" in prompt

    def test_prompt_includes_citation_instructions(
        self, qa: KnowledgeQA
    ) -> None:
        prompt = qa._build_prompt("q", [])
        assert "Cite every claim with [Article Title]" in prompt

    def test_prompt_marks_snippet_only(self, qa: KnowledgeQA) -> None:
        prompt = qa._build_prompt(
            "q",
            [
                {
                    "title": "T",
                    "slug": "s",
                    "content": "c",
                    "mode": "snippet",
                }
            ],
        )
        assert "(snippet only)" in prompt


# ---------------------------------------------------------------------------
# Citation extraction & validation
# ---------------------------------------------------------------------------


class TestCitationExtraction:
    """Verify [Title] citations are extracted and validated."""

    def test_extracts_valid_citations(self, qa: KnowledgeQA) -> None:
        title_map = {"Python Basics": "python-basics"}
        answer = "Python is great [Python Basics]. It supports OOP."
        citations = qa._extract_citations(answer, title_map)
        assert len(citations) == 1
        assert citations[0].title == "Python Basics"
        assert citations[0].slug == "python-basics"
        assert citations[0].verified is True

    def test_extracts_multiple_citations(self, qa: KnowledgeQA) -> None:
        title_map = {
            "Python Basics": "python-basics",
            "Python Async Programming": "python-async",
        }
        answer = (
            "Python [Python Basics] supports async [Python Async Programming]."
        )
        citations = qa._extract_citations(answer, title_map)
        assert len(citations) == 2

    def test_deduplicates_citations(self, qa: KnowledgeQA) -> None:
        title_map = {"Python Basics": "python-basics"}
        answer = "[Python Basics] foo [Python Basics] bar."
        citations = qa._extract_citations(answer, title_map)
        assert len(citations) == 1

    def test_invalid_citation_marked_unverified(
        self, qa: KnowledgeQA
    ) -> None:
        title_map = {"Python Basics": "python-basics"}
        answer = "Some claim [Nonexistent Article]."
        citations = qa._extract_citations(answer, title_map)
        assert len(citations) == 1
        assert citations[0].verified is False
        assert citations[0].slug == ""
        assert citations[0].title == "Nonexistent Article"

    def test_skips_instruction_patterns(self, qa: KnowledgeQA) -> None:
        title_map = {}
        answer = "Use [Article Title] notation."
        citations = qa._extract_citations(answer, title_map)
        assert len(citations) == 0

    def test_citation_has_excerpt(self, qa: KnowledgeQA) -> None:
        title_map = {"Python Basics": "python-basics"}
        answer = "Python is a language. [Python Basics]"
        citations = qa._extract_citations(answer, title_map)
        assert len(citations) == 1
        assert citations[0].excerpt != ""


# ---------------------------------------------------------------------------
# Confidence assessment
# ---------------------------------------------------------------------------


class TestConfidenceAssessment:
    """Verify HIGH/MEDIUM/LOW confidence assignment."""

    def test_high_confidence_with_multiple_citations(self) -> None:
        citations = [
            Citation(slug="a", title="A", verified=True),
            Citation(slug="b", title="B", verified=True),
        ]
        conf = KnowledgeQA._assess_confidence("Solid answer.", citations)
        assert conf == "high"

    def test_medium_confidence_with_one_citation(self) -> None:
        citations = [Citation(slug="a", title="A", verified=True)]
        conf = KnowledgeQA._assess_confidence("Answer here.", citations)
        assert conf == "medium"

    def test_low_confidence_with_no_citations(self) -> None:
        conf = KnowledgeQA._assess_confidence("I'm not sure about this.", [])
        assert conf == "low"

    def test_medium_with_hedging_and_citations(self) -> None:
        citations = [
            Citation(slug="a", title="A", verified=True),
            Citation(slug="b", title="B", verified=True),
        ]
        conf = KnowledgeQA._assess_confidence(
            "This might be the case.", citations
        )
        assert conf == "medium"

    def test_explicit_confidence_from_llm(self) -> None:
        citations = []
        conf = KnowledgeQA._assess_confidence(
            "Answer here.\n\nConfidence: HIGH", citations
        )
        assert conf == "high"

    def test_explicit_low_confidence_from_llm(self) -> None:
        citations = [Citation(slug="a", title="A", verified=True)]
        conf = KnowledgeQA._assess_confidence(
            "Answer.\n\nConfidence: LOW", citations
        )
        assert conf == "low"


# ---------------------------------------------------------------------------
# Full query workflow
# ---------------------------------------------------------------------------


class TestQueryWorkflow:
    """End-to-end query tests with mocked LLM."""

    def _patch_llm(self, qa: KnowledgeQA, response: str, tokens: int = 150):
        """Patch _call_llm to return a canned response."""
        qa._call_llm = MagicMock(  # type: ignore[method-assign]
            return_value=(response, tokens, "ollama/qwen3:8b")
        )

    def test_query_returns_saido_query_result(
        self, qa: KnowledgeQA
    ) -> None:
        self._patch_llm(
            qa,
            "Python is a language [Python Basics]. "
            "It has async [Python Async Programming].\n\nConfidence: HIGH",
        )
        result = qa.query("What is Python?")
        assert isinstance(result, SaidoQueryResult)
        assert "Python" in result.answer
        assert result.confidence == "high"
        assert result.tokens_used == 150
        assert result.provider == "ollama/qwen3:8b"

    def test_query_includes_citations(self, qa: KnowledgeQA) -> None:
        self._patch_llm(
            qa,
            "Python is great [Python Basics]. "
            "Async rocks [Python Async Programming].\n\nConfidence: HIGH",
        )
        result = qa.query("Tell me about Python")
        assert len(result.citations) == 2
        slugs = {c.slug for c in result.citations}
        assert "python-basics" in slugs
        assert "python-async" in slugs

    def test_query_includes_retrieval_stats(
        self, qa: KnowledgeQA
    ) -> None:
        self._patch_llm(qa, "Answer [Python Basics].")
        result = qa.query("question")
        assert result.retrieval_stats["document_count"] == 3
        assert result.retrieval_stats["results_found"] == 3
        assert "full_articles" in result.retrieval_stats
        assert "snippet_articles" in result.retrieval_stats

    def test_query_updates_conversation_history(
        self, qa: KnowledgeQA
    ) -> None:
        self._patch_llm(qa, "Answer [Python Basics].")
        qa.query("First question")
        assert len(qa._conversation_history) == 1
        assert qa._conversation_history[0]["question"] == "First question"

    def test_follow_up_uses_history(self, qa: KnowledgeQA) -> None:
        self._patch_llm(qa, "First answer [Python Basics].")
        qa.query("First question")

        self._patch_llm(qa, "Follow-up answer [Python Async Programming].")
        qa.query("What about async?")

        assert len(qa._conversation_history) == 2
        # The prompt should include previous context
        prompt = qa._build_prompt(
            "Third question?",
            [
                {
                    "title": "T",
                    "slug": "s",
                    "content": "c",
                    "mode": "full",
                }
            ],
        )
        assert "First question" in prompt
        assert "Previous Conversation Context" in prompt


# ---------------------------------------------------------------------------
# Empty knowledge store
# ---------------------------------------------------------------------------


class TestEmptyKnowledgeStore:
    """Verify behavior when no documents are ingested."""

    def test_empty_store_returns_helpful_message(self) -> None:
        bridge = MagicMock()
        bridge.stats = {"document_count": 0}
        qa = KnowledgeQA(bridge=bridge)
        result = qa.query("Any question")
        assert "No knowledge base articles found" in result.answer
        assert "/ingest" in result.answer
        assert result.confidence == "low"
        assert result.retrieval_stats["document_count"] == 0

    def test_no_search_results_returns_helpful_message(self) -> None:
        bridge = MagicMock()
        bridge.stats = {"document_count": 5}
        bridge.search.return_value = []
        qa = KnowledgeQA(bridge=bridge)
        result = qa.query("Obscure question")
        assert "could not find" in result.answer.lower()
        assert result.confidence == "low"


# ---------------------------------------------------------------------------
# LLM failure
# ---------------------------------------------------------------------------


class TestLLMFailure:
    """Verify graceful degradation when LLM is unavailable."""

    def test_no_router_returns_error(self, mock_bridge: MagicMock) -> None:
        qa = KnowledgeQA(bridge=mock_bridge, model_router=None)
        result = qa.query("question")
        assert "LLM call failed" in result.answer
        assert result.confidence == "low"

    def test_llm_exception_returns_error(
        self, qa: KnowledgeQA
    ) -> None:
        qa._call_llm = MagicMock(return_value=(None, 0, ""))  # type: ignore[method-assign]
        result = qa.query("question")
        assert "LLM call failed" in result.answer
        assert result.confidence == "low"


# ---------------------------------------------------------------------------
# search() method
# ---------------------------------------------------------------------------


class TestSearch:
    """Verify the thin search wrapper."""

    def test_search_returns_ranked_results(
        self, qa: KnowledgeQA, mock_bridge: MagicMock
    ) -> None:
        results = qa.search("python")
        assert len(results) == 3
        mock_bridge.search.assert_called_once_with("python", top_k=5)

    def test_search_result_format(
        self, qa: KnowledgeQA
    ) -> None:
        results = qa.search("python")
        for r in results:
            assert "slug" in r
            assert "title" in r
            assert "summary" in r
            assert "score" in r
            assert "snippet" in r

    def test_search_custom_top_k(
        self, qa: KnowledgeQA, mock_bridge: MagicMock
    ) -> None:
        qa.search("python", top_k=10)
        mock_bridge.search.assert_called_once_with("python", top_k=10)

    def test_search_empty_results(self) -> None:
        bridge = MagicMock()
        bridge.search.return_value = []
        qa = KnowledgeQA(bridge=bridge)
        results = qa.search("nothing")
        assert results == []


# ---------------------------------------------------------------------------
# SaidoQueryResult dataclass
# ---------------------------------------------------------------------------


class TestSaidoQueryResult:
    """Verify the result dataclass defaults and structure."""

    def test_defaults(self) -> None:
        r = SaidoQueryResult(answer="test")
        assert r.answer == "test"
        assert r.citations == []
        assert r.confidence == "medium"
        assert r.retrieval_stats == {}
        assert r.tokens_used == 0
        assert r.provider == ""

    def test_populated(self) -> None:
        r = SaidoQueryResult(
            answer="answer",
            citations=[Citation(slug="s", title="T")],
            confidence="high",
            retrieval_stats={"document_count": 5},
            tokens_used=200,
            provider="ollama/qwen3:8b",
        )
        assert len(r.citations) == 1
        assert r.retrieval_stats["document_count"] == 5


# ---------------------------------------------------------------------------
# Conversation history management
# ---------------------------------------------------------------------------


class TestConversationHistory:
    """Verify history is maintained and bounded."""

    def test_clear_history(self, qa: KnowledgeQA) -> None:
        qa._conversation_history.append({"question": "q", "answer": "a"})
        qa.clear_history()
        assert len(qa._conversation_history) == 0

    def test_history_bounded_to_max(self, qa: KnowledgeQA) -> None:
        for i in range(_MAX_HISTORY + 3):
            qa._append_history(f"q{i}", f"a{i}")
        assert len(qa._conversation_history) == _MAX_HISTORY
        # Oldest entries should be dropped
        assert qa._conversation_history[0]["question"] == f"q3"

    def test_history_included_in_prompt(self, qa: KnowledgeQA) -> None:
        qa._append_history("prev question", "prev answer text")
        prompt = qa._build_prompt(
            "current question",
            [
                {
                    "title": "T",
                    "slug": "s",
                    "content": "c",
                    "mode": "full",
                }
            ],
        )
        assert "prev question" in prompt
        assert "Previous Conversation Context" in prompt
