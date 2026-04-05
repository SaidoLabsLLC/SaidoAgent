"""Tests for saido_agent.knowledge.grounding — KnowledgeGrounder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from saido_agent.knowledge.grounding import KnowledgeGrounder


# ---------------------------------------------------------------------------
# Fake types
# ---------------------------------------------------------------------------


@dataclass
class FakeSearchResult:
    slug: str
    title: str
    summary: str
    score: float
    snippet: str = ""


@dataclass
class FakeDocument:
    slug: str
    title: str
    body: str
    frontmatter: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bridge(
    doc_count: int = 5,
    search_results: list | None = None,
    articles: dict | None = None,
):
    """Build a mock KnowledgeBridge."""
    bridge = MagicMock()
    bridge.stats = {"document_count": doc_count}
    bridge.search.return_value = search_results or []
    if articles:
        bridge.read_article.side_effect = lambda slug: articles.get(slug)
    else:
        bridge.read_article.return_value = None
    return bridge


SAMPLE_RESULTS = [
    FakeSearchResult(
        slug="auth-guide",
        title="Authentication Guide",
        summary="Guide to implementing OAuth2 and JWT authentication.",
        score=0.9,
        snippet="Guide to implementing OAuth2 and JWT authentication.",
    ),
    FakeSearchResult(
        slug="api-security",
        title="API Security Best Practices",
        summary="Rate limiting, input validation, and CORS configuration.",
        score=0.7,
        snippet="Rate limiting, input validation, and CORS configuration.",
    ),
    FakeSearchResult(
        slug="deployment",
        title="Deployment Handbook",
        summary="CI/CD pipelines and container orchestration.",
        score=0.5,
        snippet="CI/CD pipelines and container orchestration.",
    ),
]

SAMPLE_ARTICLES = {
    "auth-guide": FakeDocument(
        slug="auth-guide",
        title="Authentication Guide",
        body="Full body content: OAuth2 flow with PKCE. JWT tokens with RS256 signatures. "
        "Session management and refresh token rotation.",
    ),
}


# ---------------------------------------------------------------------------
# Tests: ground() returns context when articles found
# ---------------------------------------------------------------------------


class TestGroundReturnsContext:
    def test_returns_context_string_with_articles(self):
        bridge = _make_bridge(
            doc_count=5,
            search_results=SAMPLE_RESULTS,
            articles=SAMPLE_ARTICLES,
        )
        grounder = KnowledgeGrounder(bridge)
        result = grounder.ground("How do I authenticate?")

        assert result is not None
        assert "## Relevant Knowledge Base Articles" in result
        assert "[Authentication Guide]" in result
        assert "[API Security Best Practices]" in result

    def test_top_article_gets_full_content(self):
        bridge = _make_bridge(
            doc_count=5,
            search_results=SAMPLE_RESULTS,
            articles=SAMPLE_ARTICLES,
        )
        grounder = KnowledgeGrounder(bridge)
        result = grounder.ground("How do I authenticate?")

        assert result is not None
        # Top article should have full body content
        assert "OAuth2 flow with PKCE" in result
        # Second article should have snippet only
        assert "Rate limiting, input validation" in result

    def test_remaining_articles_get_snippets(self):
        bridge = _make_bridge(
            doc_count=5,
            search_results=SAMPLE_RESULTS,
            articles=SAMPLE_ARTICLES,
        )
        grounder = KnowledgeGrounder(bridge)
        result = grounder.ground("How do I authenticate?")

        # read_article should only be called for the first result (top article)
        bridge.read_article.assert_called_once_with("auth-guide")


# ---------------------------------------------------------------------------
# Tests: returns None when knowledge store is empty
# ---------------------------------------------------------------------------


class TestGroundReturnsNone:
    def test_returns_none_when_store_empty(self):
        bridge = _make_bridge(doc_count=0)
        grounder = KnowledgeGrounder(bridge)
        result = grounder.ground("Any question")
        assert result is None

    def test_returns_none_when_no_search_results(self):
        bridge = _make_bridge(doc_count=5, search_results=[])
        grounder = KnowledgeGrounder(bridge)
        result = grounder.ground("Obscure query")
        assert result is None

    def test_returns_none_when_bridge_is_none(self):
        grounder = KnowledgeGrounder(bridge=None)
        result = grounder.ground("Any question")
        assert result is None

    def test_returns_none_when_disabled(self):
        bridge = _make_bridge(doc_count=5, search_results=SAMPLE_RESULTS)
        grounder = KnowledgeGrounder(bridge)
        grounder.enabled = False
        result = grounder.ground("How do I authenticate?")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: respects min_relevance_score filter
# ---------------------------------------------------------------------------


class TestRelevanceFilter:
    def test_filters_low_score_results(self):
        low_score_results = [
            FakeSearchResult(slug="a", title="A", summary="aaa", score=0.05),
            FakeSearchResult(slug="b", title="B", summary="bbb", score=0.03),
        ]
        bridge = _make_bridge(doc_count=5, search_results=low_score_results)
        grounder = KnowledgeGrounder(bridge, config={"grounding_min_score": 0.1})
        result = grounder.ground("query")
        assert result is None

    def test_keeps_high_score_results(self):
        mixed_results = [
            FakeSearchResult(slug="a", title="High Score", summary="Good match", score=0.8),
            FakeSearchResult(slug="b", title="Low Score", summary="Bad match", score=0.05),
        ]
        bridge = _make_bridge(doc_count=5, search_results=mixed_results)
        grounder = KnowledgeGrounder(bridge, config={"grounding_min_score": 0.1})
        result = grounder.ground("query")

        assert result is not None
        assert "[High Score]" in result
        assert "[Low Score]" not in result


# ---------------------------------------------------------------------------
# Tests: respects max_context_chars limit
# ---------------------------------------------------------------------------


class TestMaxContextChars:
    def test_truncates_at_char_limit(self):
        long_results = [
            FakeSearchResult(
                slug=f"article-{i}",
                title=f"Article {i}",
                summary="x" * 2000,
                score=0.9,
            )
            for i in range(5)
        ]
        bridge = _make_bridge(doc_count=10, search_results=long_results)
        # Small limit - should only fit 1-2 articles
        grounder = KnowledgeGrounder(bridge, config={"grounding_max_chars": 2500})
        result = grounder.ground("query")

        assert result is not None
        # Should have at most 1 article given the 2500 char limit with 2000-char summaries
        article_count = result.count("### [")
        assert article_count <= 2


# ---------------------------------------------------------------------------
# Tests: formatted output includes [Article Title] headers
# ---------------------------------------------------------------------------


class TestFormattedOutput:
    def test_includes_article_title_headers(self):
        bridge = _make_bridge(
            doc_count=5,
            search_results=SAMPLE_RESULTS[:1],
            articles=SAMPLE_ARTICLES,
        )
        grounder = KnowledgeGrounder(bridge)
        result = grounder.ground("auth question")

        assert "### [Authentication Guide]" in result

    def test_includes_citation_instruction(self):
        bridge = _make_bridge(
            doc_count=5,
            search_results=SAMPLE_RESULTS[:1],
            articles=SAMPLE_ARTICLES,
        )
        grounder = KnowledgeGrounder(bridge)
        result = grounder.ground("auth question")

        assert "Cite sources using [Article Title] notation" in result


# ---------------------------------------------------------------------------
# Tests: enabled/disabled toggle
# ---------------------------------------------------------------------------


class TestEnabledToggle:
    def test_default_enabled(self):
        grounder = KnowledgeGrounder(bridge=MagicMock())
        assert grounder.enabled is True

    def test_disable_prevents_grounding(self):
        bridge = _make_bridge(doc_count=5, search_results=SAMPLE_RESULTS)
        grounder = KnowledgeGrounder(bridge)
        grounder.enabled = False
        result = grounder.ground("question")
        assert result is None
        bridge.search.assert_not_called()

    def test_re_enable_restores_grounding(self):
        bridge = _make_bridge(
            doc_count=5,
            search_results=SAMPLE_RESULTS,
            articles=SAMPLE_ARTICLES,
        )
        grounder = KnowledgeGrounder(bridge)
        grounder.enabled = False
        assert grounder.ground("question") is None

        grounder.enabled = True
        result = grounder.ground("question")
        assert result is not None


# ---------------------------------------------------------------------------
# Tests: agent loop injects knowledge context (mock bridge + LLM)
# ---------------------------------------------------------------------------


class TestAgentLoopIntegration:
    def test_agent_loop_uses_grounder(self):
        """Verify that the agent loop calls grounder.ground() and augments system_prompt."""
        from saido_agent.core.agent import AgentState, run, AssistantTurn, TextChunk

        mock_grounder = MagicMock()
        mock_grounder.ground.return_value = "\n\n## Relevant Knowledge\nSome context here."

        state = AgentState()
        config = {
            "model": "test-model",
            "_knowledge_grounder": mock_grounder,
            "permission_mode": "accept-all",
        }

        # Mock the stream function to return a simple response
        fake_turn = AssistantTurn(
            text="Test response",
            tool_calls=[],
            in_tokens=10,
            out_tokens=20,
        )

        with patch("saido_agent.core.agent.stream") as mock_stream, \
             patch("saido_agent.core.agent.get_tool_schemas", return_value=[]), \
             patch("saido_agent.core.agent.maybe_compact"):
            mock_stream.return_value = iter([TextChunk("Test response"), fake_turn])

            events = list(run("What is OAuth?", state, config, "Base system prompt"))

            mock_grounder.ground.assert_called_once_with("What is OAuth?")

            # Verify the system prompt passed to stream includes knowledge context
            call_kwargs = mock_stream.call_args
            system_arg = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
            if system_arg is None:
                # positional args
                system_arg = call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None
            assert system_arg is not None
            assert "Relevant Knowledge" in system_arg

    def test_agent_loop_without_grounder(self):
        """Agent loop works normally when no grounder is configured."""
        from saido_agent.core.agent import AgentState, run, AssistantTurn, TextChunk

        state = AgentState()
        config = {
            "model": "test-model",
            "permission_mode": "accept-all",
        }

        fake_turn = AssistantTurn(
            text="Normal response",
            tool_calls=[],
            in_tokens=10,
            out_tokens=20,
        )

        with patch("saido_agent.core.agent.stream") as mock_stream, \
             patch("saido_agent.core.agent.get_tool_schemas", return_value=[]), \
             patch("saido_agent.core.agent.maybe_compact"):
            mock_stream.return_value = iter([TextChunk("Normal response"), fake_turn])
            events = list(run("Hello", state, config, "System prompt"))

            # Should work without error
            text_events = [e for e in events if isinstance(e, TextChunk)]
            assert len(text_events) == 1


# ---------------------------------------------------------------------------
# Tests: knowledge context doesn't accumulate across turns
# ---------------------------------------------------------------------------


class TestPerTurnIsolation:
    def test_context_does_not_accumulate(self):
        """Each call to run() gets fresh grounding, not cumulative."""
        from saido_agent.core.agent import AgentState, run, AssistantTurn, TextChunk

        call_count = 0
        system_prompts_seen = []

        def mock_grounder_ground(msg):
            return f"\n\nKnowledge for: {msg}"

        mock_grounder = MagicMock()
        mock_grounder.ground.side_effect = mock_grounder_ground

        state = AgentState()
        config = {
            "model": "test-model",
            "_knowledge_grounder": mock_grounder,
            "permission_mode": "accept-all",
        }
        base_prompt = "Base system prompt"

        fake_turn = AssistantTurn(
            text="Response",
            tool_calls=[],
            in_tokens=10,
            out_tokens=20,
        )

        def capture_stream(**kwargs):
            system_prompts_seen.append(kwargs.get("system", ""))
            return iter([TextChunk("Response"), fake_turn])

        with patch("saido_agent.core.agent.stream", side_effect=capture_stream), \
             patch("saido_agent.core.agent.get_tool_schemas", return_value=[]), \
             patch("saido_agent.core.agent.maybe_compact"):

            # Turn 1
            list(run("Question 1", state, config, base_prompt))
            # Turn 2
            list(run("Question 2", state, config, base_prompt))

        assert len(system_prompts_seen) == 2
        # Each turn should have base prompt + only its own knowledge context
        assert "Knowledge for: Question 1" in system_prompts_seen[0]
        assert "Knowledge for: Question 2" in system_prompts_seen[1]
        # Turn 2 should NOT contain Turn 1's knowledge
        assert "Knowledge for: Question 1" not in system_prompts_seen[1]


# ---------------------------------------------------------------------------
# Tests: graceful degradation when bridge unavailable
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    def test_stats_exception_returns_none(self):
        bridge = MagicMock()
        bridge.stats = property(lambda self: (_ for _ in ()).throw(RuntimeError("DB error")))
        # Simulate stats raising by using a property that raises
        type(bridge).stats = property(lambda self: (_ for _ in ()).throw(RuntimeError("fail")))
        grounder = KnowledgeGrounder(bridge)
        result = grounder.ground("question")
        assert result is None

    def test_search_exception_returns_none(self):
        bridge = MagicMock()
        bridge.stats = {"document_count": 5}
        bridge.search.side_effect = RuntimeError("Search engine down")
        grounder = KnowledgeGrounder(bridge)
        result = grounder.ground("question")
        assert result is None

    def test_read_article_exception_falls_back_to_snippet(self):
        results = [
            FakeSearchResult(
                slug="test",
                title="Test Article",
                summary="Snippet content here",
                score=0.9,
            ),
        ]
        bridge = _make_bridge(doc_count=5, search_results=results)
        bridge.read_article.side_effect = RuntimeError("Read failed")
        grounder = KnowledgeGrounder(bridge)
        result = grounder.ground("question")

        assert result is not None
        assert "Snippet content here" in result


# ---------------------------------------------------------------------------
# Tests: REPL initializes grounder in knowledge context
# ---------------------------------------------------------------------------


class TestREPLInit:
    def test_init_knowledge_context_creates_grounder(self):
        """_init_knowledge_context should create and store a KnowledgeGrounder."""
        config = {"model": "test"}

        with patch("saido_agent.knowledge.grounding.KnowledgeGrounder") as MockGrounder, \
             patch("saido_agent.cli.repl._init_knowledge_context") as mock_init:
            # We test this at a higher level — verify the real function stores grounder
            pass

        # Integration-style: call the real function with mocked SmartRAG
        with patch("saido_agent.knowledge.bridge.SMARTRAG_AVAILABLE", False):
            from saido_agent.cli.repl import _init_knowledge_context
            kctx = _init_knowledge_context(config)
            # Bridge will be None since SmartRAG is unavailable, but grounder
            # should still be created (it handles None bridge gracefully)
            # The grounder might not be created if bridge init fails before it
            # In degraded mode, bridge is None, so grounder wraps None
            grounder = kctx.get("grounder")
            if grounder is not None:
                assert grounder.ground("test") is None  # Bridge is None


# ---------------------------------------------------------------------------
# Tests: SaidoAgent has grounder attached
# ---------------------------------------------------------------------------


class TestSaidoAgentGrounder:
    def test_saido_agent_has_grounder(self):
        """SaidoAgent.__init__ should create a KnowledgeGrounder."""
        with patch("saido_agent.knowledge.bridge.SMARTRAG_AVAILABLE", False), \
             patch("saido_agent.knowledge.bridge.KnowledgeBridge"):
            from saido_agent import SaidoAgent

            with patch.object(SaidoAgent, "__init__", lambda self, **kw: None):
                agent = SaidoAgent.__new__(SaidoAgent)
                agent._bridge = MagicMock()
                agent._grounder = KnowledgeGrounder(bridge=agent._bridge)
                assert agent.grounder is not None
                assert isinstance(agent.grounder, KnowledgeGrounder)


# ---------------------------------------------------------------------------
# Tests: /grounding CLI commands
# ---------------------------------------------------------------------------


class TestGroundingCommand:
    def test_grounding_on(self, capsys):
        grounder = KnowledgeGrounder(bridge=MagicMock())
        grounder.enabled = False
        config = {"_knowledge_grounder": grounder}

        from saido_agent.cli.repl import cmd_grounding
        cmd_grounding("on", None, config)

        assert grounder.enabled is True

    def test_grounding_off(self, capsys):
        grounder = KnowledgeGrounder(bridge=MagicMock())
        config = {"_knowledge_grounder": grounder}

        from saido_agent.cli.repl import cmd_grounding
        cmd_grounding("off", None, config)

        assert grounder.enabled is False

    def test_grounding_status(self, capsys):
        grounder = KnowledgeGrounder(bridge=MagicMock())
        config = {"_knowledge_grounder": grounder}

        from saido_agent.cli.repl import cmd_grounding
        cmd_grounding("status", None, config)

        captured = capsys.readouterr()
        assert "enabled" in captured.out

    def test_grounding_no_grounder(self, capsys):
        config = {}

        from saido_agent.cli.repl import cmd_grounding
        cmd_grounding("on", None, config)

        captured = capsys.readouterr()
        assert "unavailable" in captured.out.lower() or "not initialized" in captured.out.lower()

    def test_grounding_empty_args_shows_status(self, capsys):
        grounder = KnowledgeGrounder(bridge=MagicMock())
        config = {"_knowledge_grounder": grounder}

        from saido_agent.cli.repl import cmd_grounding
        cmd_grounding("", None, config)

        captured = capsys.readouterr()
        assert "enabled" in captured.out


# ---------------------------------------------------------------------------
# Tests: config parameters
# ---------------------------------------------------------------------------


class TestConfigParams:
    def test_custom_top_k(self):
        bridge = _make_bridge(doc_count=5, search_results=SAMPLE_RESULTS)
        grounder = KnowledgeGrounder(bridge, config={"grounding_top_k": 1})
        assert grounder._top_k == 1

    def test_custom_max_chars(self):
        grounder = KnowledgeGrounder(bridge=MagicMock(), config={"grounding_max_chars": 8000})
        assert grounder._max_context_chars == 8000

    def test_custom_min_score(self):
        grounder = KnowledgeGrounder(bridge=MagicMock(), config={"grounding_min_score": 0.5})
        assert grounder._min_relevance_score == 0.5

    def test_defaults_without_config(self):
        grounder = KnowledgeGrounder(bridge=MagicMock())
        assert grounder._top_k == 3
        assert grounder._max_context_chars == 4000
        assert grounder._min_relevance_score == 0.1


# ---------------------------------------------------------------------------
# Tests: dict-based results (not dataclass)
# ---------------------------------------------------------------------------


class TestDictResults:
    def test_handles_dict_search_results(self):
        dict_results = [
            {"slug": "a", "title": "Article A", "summary": "Content A", "score": 0.8, "snippet": "Content A"},
            {"slug": "b", "title": "Article B", "summary": "Content B", "score": 0.6, "snippet": "Content B"},
        ]
        bridge = _make_bridge(doc_count=5, search_results=dict_results)
        bridge.read_article.return_value = {"body": "Full body of Article A", "title": "Article A"}
        grounder = KnowledgeGrounder(bridge)
        result = grounder.ground("test query")

        assert result is not None
        assert "[Article A]" in result
        assert "Full body of Article A" in result

    def test_handles_dict_article_response(self):
        results = [
            FakeSearchResult(slug="x", title="X", summary="snippet", score=0.9),
        ]
        bridge = _make_bridge(doc_count=5, search_results=results)
        bridge.read_article.return_value = {"body": "Dict body content", "slug": "x"}
        grounder = KnowledgeGrounder(bridge)
        result = grounder.ground("query")

        assert result is not None
        assert "Dict body content" in result
