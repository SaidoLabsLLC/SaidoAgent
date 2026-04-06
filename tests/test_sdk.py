"""Tests for the Saido Agent public SDK API.

Validates the pip-installable API contract:
  - SaidoAgent initializes and exposes all public methods
  - Type dataclasses serialize correctly
  - Public exports are correct (only expected symbols)
  - Integration pattern from PRD Section 8.2 works with mocks
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def tmp_knowledge_dir(tmp_path):
    """Create a temporary knowledge directory."""
    kb = tmp_path / "knowledge"
    kb.mkdir()
    return str(kb)


@pytest.fixture
def mock_bridge():
    """Create a mock KnowledgeBridge that works without SmartRAG."""
    bridge = MagicMock()
    bridge.available = True
    bridge.stats = {"document_count": 0, "index_size_bytes": 0, "categories": []}
    bridge.search.return_value = []
    bridge.list_articles.return_value = []
    return bridge


@pytest.fixture
def agent(tmp_knowledge_dir):
    """Create a SaidoAgent with a temporary knowledge directory.

    Patches out SmartRAG dependency so tests run without it installed.
    """
    with patch("saido_agent.knowledge.bridge.SMARTRAG_AVAILABLE", False):
        from saido_agent import SaidoAgent

        a = SaidoAgent(knowledge_dir=tmp_knowledge_dir)
        return a


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------


class TestSaidoAgentInit:
    """SaidoAgent initialization tests."""

    def test_init_with_defaults(self, tmp_knowledge_dir):
        """SaidoAgent initializes with default parameters."""
        with patch("saido_agent.knowledge.bridge.SMARTRAG_AVAILABLE", False):
            from saido_agent import SaidoAgent

            agent = SaidoAgent(knowledge_dir=tmp_knowledge_dir)
            assert agent is not None
            assert agent._knowledge_dir == str(Path(tmp_knowledge_dir).resolve())

    def test_init_with_custom_knowledge_dir(self, tmp_path):
        """SaidoAgent respects a custom knowledge_dir."""
        custom_dir = tmp_path / "custom_kb"
        custom_dir.mkdir()

        with patch("saido_agent.knowledge.bridge.SMARTRAG_AVAILABLE", False):
            from saido_agent import SaidoAgent

            agent = SaidoAgent(knowledge_dir=str(custom_dir))
            assert str(custom_dir.resolve()) in agent._knowledge_dir

    def test_init_creates_internal_components(self, agent):
        """SaidoAgent creates all internal components on init."""
        assert agent._bridge is not None
        assert agent._router is not None
        assert agent._cost_tracker is not None
        assert agent._ingest_pipeline is not None
        assert agent._compiler is not None
        assert agent._qa is not None

    def test_init_with_routing_config(self, tmp_knowledge_dir, tmp_path):
        """SaidoAgent accepts a routing_config path."""
        routing_file = tmp_path / "routing.json"
        routing_file.write_text(json.dumps({
            "routing": {"qa": {"prefer": "local", "model": "qwen3:8b"}},
            "local_providers": {},
        }))

        with patch("saido_agent.knowledge.bridge.SMARTRAG_AVAILABLE", False):
            from saido_agent import SaidoAgent

            agent = SaidoAgent(
                knowledge_dir=tmp_knowledge_dir,
                routing_config=str(routing_file),
            )
            assert agent._router is not None


# ---------------------------------------------------------------------------
# Test: Ingest
# ---------------------------------------------------------------------------


class TestIngest:
    """SaidoAgent.ingest() tests."""

    def test_ingest_file_returns_ingest_result(self, agent):
        """ingest() returns an IngestResult dataclass."""
        from saido_agent.types import IngestResult

        sample = FIXTURES_DIR / "sample.md"
        result = agent.ingest(str(sample))

        assert isinstance(result, IngestResult)
        assert result.slug is not None
        assert result.status in ("created", "updated", "duplicate", "failed")

    def test_ingest_missing_file_returns_failed(self, agent):
        """ingest() returns failed status for non-existent file."""
        result = agent.ingest("/nonexistent/file.md")
        assert result.status == "failed"

    def test_ingest_directory(self, agent, tmp_path):
        """ingest() handles directory input with children list."""
        # Create a temp directory with a markdown file
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "test.md").write_text("# Test\nSome content.")

        result = agent.ingest(str(doc_dir))
        assert result.slug == "docs"
        assert isinstance(result.children, list)

    def test_ingest_result_serializes(self):
        """IngestResult can be serialized to a dict."""
        from saido_agent.types import IngestResult

        result = IngestResult(
            slug="test-doc",
            title="Test Document",
            status="created",
            children=["child-1", "child-2"],
        )
        d = asdict(result)
        assert d["slug"] == "test-doc"
        assert d["status"] == "created"
        assert len(d["children"]) == 2
        assert d["error"] is None


# ---------------------------------------------------------------------------
# Test: Query
# ---------------------------------------------------------------------------


class TestQuery:
    """SaidoAgent.query() tests."""

    def test_query_returns_saido_query_result(self, agent):
        """query() returns a SaidoQueryResult."""
        from saido_agent.knowledge.query import SaidoQueryResult

        result = agent.query("What is this about?")

        assert isinstance(result, SaidoQueryResult)
        assert result.answer is not None
        assert isinstance(result.answer, str)

    def test_query_empty_store_returns_low_confidence(self, agent):
        """query() with empty knowledge store returns low confidence."""
        result = agent.query("Anything?")
        assert result.confidence == "low"

    def test_query_with_mock_llm(self, tmp_knowledge_dir):
        """query() works end-to-end with a mocked LLM."""
        with patch("saido_agent.knowledge.bridge.SMARTRAG_AVAILABLE", False):
            from saido_agent import SaidoAgent

            agent = SaidoAgent(knowledge_dir=tmp_knowledge_dir)

            # Mock the QA engine's internal LLM call
            with patch.object(agent._qa, "_call_llm") as mock_llm:
                mock_llm.return_value = (
                    "This is about knowledge management. [Sample Doc]",
                    150,
                    "ollama/qwen3:8b",
                )
                # Still returns a result (empty store -> low confidence path)
                result = agent.query("What is this about?")
                assert result.answer is not None


# ---------------------------------------------------------------------------
# Test: Search
# ---------------------------------------------------------------------------


class TestSearch:
    """SaidoAgent.search() tests."""

    def test_search_returns_list_of_search_result(self, agent):
        """search() returns a list of SearchResult."""
        from saido_agent.types import SearchResult

        results = agent.search("test query")
        assert isinstance(results, list)
        # Empty store returns empty list
        for r in results:
            assert isinstance(r, SearchResult)

    def test_search_result_serializes(self):
        """SearchResult can be serialized to a dict."""
        from saido_agent.types import SearchResult

        result = SearchResult(
            slug="doc-1",
            title="Document One",
            summary="A summary.",
            score=0.95,
            snippet="...relevant snippet...",
        )
        d = asdict(result)
        assert d["score"] == 0.95
        assert d["slug"] == "doc-1"


# ---------------------------------------------------------------------------
# Test: Stats
# ---------------------------------------------------------------------------


class TestStats:
    """SaidoAgent.stats property tests."""

    def test_stats_returns_store_stats(self, agent):
        """stats property returns a StoreStats dataclass."""
        from saido_agent.types import StoreStats

        stats = agent.stats
        assert isinstance(stats, StoreStats)
        assert isinstance(stats.document_count, int)

    def test_stats_serializes(self):
        """StoreStats can be serialized to a dict."""
        from saido_agent.types import StoreStats

        stats = StoreStats(
            document_count=42,
            category_count=5,
            concept_count=100,
            total_size_bytes=1024000,
        )
        d = asdict(stats)
        assert d["document_count"] == 42
        assert d["total_size_bytes"] == 1024000


# ---------------------------------------------------------------------------
# Test: Cost
# ---------------------------------------------------------------------------


class TestCost:
    """SaidoAgent.cost property tests."""

    def test_cost_returns_dict(self, agent):
        """cost property returns a dict with expected keys."""
        cost = agent.cost
        assert isinstance(cost, dict)
        assert "total_cost" in cost
        assert "total_tokens" in cost
        assert "estimated_savings" in cost
        assert "report" in cost

    def test_cost_values_are_numeric(self, agent):
        """cost values are numeric types."""
        cost = agent.cost
        assert isinstance(cost["total_cost"], (int, float))
        assert isinstance(cost["total_tokens"], int)
        assert isinstance(cost["estimated_savings"], (int, float))


# ---------------------------------------------------------------------------
# Test: Compile
# ---------------------------------------------------------------------------


class TestCompile:
    """SaidoAgent.compile() tests."""

    def test_compile_returns_compile_result(self, agent):
        """compile() returns a CompileResult."""
        from saido_agent.types import CompileResult

        result = agent.compile()
        assert isinstance(result, CompileResult)

    def test_compile_empty_queue_returns_skipped(self, agent):
        """compile() with no pending articles returns skipped."""
        result = agent.compile()
        assert result.status == "skipped"

    def test_compile_with_slug_triggers_compiler(self, agent):
        """compile(slug=...) delegates to WikiCompiler."""
        with patch.object(agent._compiler, "compile") as mock_compile:
            from saido_agent.knowledge.compile import (
                CompileResult as InternalCR,
            )

            mock_compile.return_value = InternalCR(
                slug="test-article",
                status="compiled",
                summary="A test summary.",
                concepts=["testing", "sdk"],
                categories=["development"],
            )

            result = agent.compile(slug="test-article")
            mock_compile.assert_called_once_with("test-article")
            assert result.status == "compiled"
            assert result.slug == "test-article"

    def test_compile_result_serializes(self):
        """CompileResult can be serialized to a dict."""
        from saido_agent.types import CompileResult

        result = CompileResult(
            slug="doc-1",
            status="compiled",
            summary="Summary text.",
            concepts=["concept-a"],
            categories=["cat-1"],
        )
        d = asdict(result)
        assert d["status"] == "compiled"
        assert "concept-a" in d["concepts"]


# ---------------------------------------------------------------------------
# Test: Agent Run
# ---------------------------------------------------------------------------


class TestRun:
    """SaidoAgent.run() tests."""

    def test_run_returns_agent_result(self, agent):
        """run() returns an AgentResult."""
        from saido_agent.types import AgentResult

        result = agent.run("Do something useful")
        assert isinstance(result, AgentResult)
        assert isinstance(result.output, str)

    def test_agent_result_serializes(self):
        """AgentResult can be serialized to a dict."""
        from saido_agent.types import AgentResult

        result = AgentResult(
            output="Done.",
            tool_calls=[{"name": "search", "args": {"q": "test"}}],
            tokens_used=500,
        )
        d = asdict(result)
        assert d["tokens_used"] == 500
        assert len(d["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# Test: Public API Exports
# ---------------------------------------------------------------------------


class TestExports:
    """Verify the public API surface is correct."""

    def test_saido_agent_exportable(self):
        """SaidoAgent can be imported from saido_agent."""
        from saido_agent import SaidoAgent

        assert SaidoAgent is not None

    def test_saido_config_exportable(self):
        """SaidoConfig can be imported from saido_agent."""
        from saido_agent import SaidoConfig

        assert SaidoConfig is not None

    def test_version_exportable(self):
        """__version__ can be imported from saido_agent."""
        from saido_agent import __version__

        assert __version__ == "0.1.0"

    def test_types_importable(self):
        """All public types can be imported from saido_agent.types."""
        from saido_agent.types import (
            AgentResult,
            Citation,
            CompileResult,
            IngestResult,
            SaidoQueryResult,
            SearchResult,
            StoreStats,
        )

        assert all([
            AgentResult,
            Citation,
            CompileResult,
            IngestResult,
            SaidoQueryResult,
            SearchResult,
            StoreStats,
        ])

    def test_only_expected_public_exports(self):
        """__all__ contains only the expected public symbols."""
        import saido_agent

        expected = {"SaidoAgent", "SaidoConfig", "__version__"}
        assert set(saido_agent.__all__) == expected

    def test_internal_modules_not_in_public_api(self):
        """Internal modules are not exposed via __all__."""
        import saido_agent

        for name in saido_agent.__all__:
            assert name not in (
                "KnowledgeBridge",
                "IngestPipeline",
                "WikiCompiler",
                "KnowledgeQA",
                "ModelRouter",
                "CostTracker",
            )


# ---------------------------------------------------------------------------
# Test: SaidoConfig
# ---------------------------------------------------------------------------


class TestSaidoConfig:
    """SaidoConfig tests."""

    def test_config_init_missing_file(self, tmp_path):
        """SaidoConfig initializes gracefully with missing config file."""
        from saido_agent import SaidoConfig

        config = SaidoConfig(config_path=str(tmp_path / "nonexistent.json"))
        assert config.knowledge_dir == "./knowledge"
        assert config.routing == {}

    def test_config_init_with_file(self, tmp_path):
        """SaidoConfig loads from a JSON file."""
        from saido_agent import SaidoConfig

        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "knowledge_dir": "/data/kb",
            "model": "qwen3:8b",
            "routing": {"qa": {"prefer": "local"}},
        }))

        config = SaidoConfig(config_path=str(cfg_file))
        assert config.knowledge_dir == "/data/kb"
        assert config.model == "qwen3:8b"
        assert config.routing == {"qa": {"prefer": "local"}}

    def test_config_update_persists(self, tmp_path):
        """SaidoConfig.update() writes changes to disk."""
        from saido_agent import SaidoConfig

        cfg_file = tmp_path / "config.json"
        config = SaidoConfig(config_path=str(cfg_file))
        config.update(knowledge_dir="/new/path", model="gpt-4o")

        # Reload and verify
        config2 = SaidoConfig(config_path=str(cfg_file))
        assert config2.knowledge_dir == "/new/path"
        assert config2.model == "gpt-4o"

    def test_config_to_dict(self, tmp_path):
        """SaidoConfig.to_dict() returns a plain dict."""
        from saido_agent import SaidoConfig

        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"model": "test-model"}))

        config = SaidoConfig(config_path=str(cfg_file))
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["model"] == "test-model"


# ---------------------------------------------------------------------------
# Test: PRD Section 8.2 Embedding Pattern
# ---------------------------------------------------------------------------


class TestPRDEmbeddingPattern:
    """Integration test: the canonical SDK usage pattern from PRD 8.2."""

    def test_ingest_then_query_pattern(self, tmp_knowledge_dir):
        """The PRD 8.2 pattern works: init -> ingest -> query."""
        with patch("saido_agent.knowledge.bridge.SMARTRAG_AVAILABLE", False):
            from saido_agent import SaidoAgent

            agent = SaidoAgent(knowledge_dir=tmp_knowledge_dir)

            # Ingest a sample file
            sample = FIXTURES_DIR / "sample.md"
            ingest_result = agent.ingest(str(sample))
            assert ingest_result.slug is not None

            # Query (will hit empty-store path since SmartRAG is mocked out)
            result = agent.query("What is this about?")
            assert result.answer is not None
            assert isinstance(result.answer, str)
            assert len(result.answer) > 0


# ---------------------------------------------------------------------------
# Test: Type Dataclass Edge Cases
# ---------------------------------------------------------------------------


class TestTypeEdgeCases:
    """Edge cases for public type dataclasses."""

    def test_ingest_result_with_error(self):
        from saido_agent.types import IngestResult

        result = IngestResult(
            slug="bad-file",
            title="Bad File",
            status="failed",
            error="Unsupported format",
        )
        assert result.error == "Unsupported format"
        assert result.children == []

    def test_store_stats_defaults(self):
        from saido_agent.types import StoreStats

        stats = StoreStats(document_count=0)
        assert stats.category_count == 0
        assert stats.concept_count == 0
        assert stats.total_size_bytes == 0

    def test_agent_result_defaults(self):
        from saido_agent.types import AgentResult

        result = AgentResult(output="hello")
        assert result.tool_calls == []
        assert result.tokens_used == 0

    def test_citation_dataclass(self):
        from saido_agent.types import Citation

        cit = Citation(slug="doc-1", title="Doc One", excerpt="some text")
        d = asdict(cit)
        assert d["verified"] is True
        assert d["slug"] == "doc-1"
