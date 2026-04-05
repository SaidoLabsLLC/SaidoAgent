"""Tests for saido_agent.knowledge.bridge — KnowledgeBridge ↔ SmartRAG."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from saido_agent.knowledge.bridge import (
    SMARTRAG_AVAILABLE,
    BridgeConfig,
    KnowledgeBridge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_knowledge(tmp_path: Path) -> Path:
    """Return a temporary directory to use as knowledge root."""
    return tmp_path / "knowledge"


@pytest.fixture()
def bridge(tmp_knowledge: Path) -> KnowledgeBridge:
    """Return a KnowledgeBridge backed by a temp directory."""
    cfg = BridgeConfig(knowledge_root=str(tmp_knowledge))
    return KnowledgeBridge(config=cfg)


# ---------------------------------------------------------------------------
# Import / availability checks
# ---------------------------------------------------------------------------

class TestAvailability:
    """Verify import detection and degraded-mode behaviour."""

    def test_smartrag_import_detected(self) -> None:
        assert SMARTRAG_AVAILABLE is True, "SmartRAG should be importable"

    def test_bridge_available_when_smartrag_present(self, bridge: KnowledgeBridge) -> None:
        assert bridge.available is True

    def test_bridge_degrades_when_smartrag_missing(self, tmp_knowledge: Path) -> None:
        """If SmartRAG is None, the bridge should still init but degrade."""
        cfg = BridgeConfig(knowledge_root=str(tmp_knowledge))
        # Pass None explicitly to simulate missing SmartRAG
        kb = KnowledgeBridge.__new__(KnowledgeBridge)
        kb._config = cfg
        kb._root = Path(cfg.knowledge_root).resolve()
        kb._rag = None
        # All operations return empty / None
        assert kb.available is False
        assert kb.create_article("x", "body") is None
        assert kb.read_article("x") is None
        assert kb.read_article_frontmatter("x") is None
        assert kb.update_article("x", body="y") is None
        assert kb.delete_article("x") is False
        assert kb.list_articles() == []
        assert kb.search("q") == []
        assert kb.query("q") is None
        assert kb.reindex() == 0
        assert kb.stats == {"document_count": 0, "index_size_bytes": 0, "categories": []}
        assert kb.get_backlinks("x") == []
        assert kb.add_code_structure("x", {}) is None


# ---------------------------------------------------------------------------
# Saido directory scaffolding
# ---------------------------------------------------------------------------

class TestDirectoryScaffolding:
    """Ensure Saido-specific dirs are created on init."""

    def test_raw_dir_created(self, bridge: KnowledgeBridge, tmp_knowledge: Path) -> None:
        assert (tmp_knowledge / "raw").is_dir()

    def test_outputs_reports_created(self, bridge: KnowledgeBridge, tmp_knowledge: Path) -> None:
        assert (tmp_knowledge / "outputs" / "reports").is_dir()

    def test_outputs_slides_created(self, bridge: KnowledgeBridge, tmp_knowledge: Path) -> None:
        assert (tmp_knowledge / "outputs" / "slides").is_dir()

    def test_outputs_charts_created(self, bridge: KnowledgeBridge, tmp_knowledge: Path) -> None:
        assert (tmp_knowledge / "outputs" / "charts").is_dir()

    def test_saido_dir_created(self, bridge: KnowledgeBridge, tmp_knowledge: Path) -> None:
        assert (tmp_knowledge / "saido").is_dir()

    def test_compile_log_initialized(self, bridge: KnowledgeBridge, tmp_knowledge: Path) -> None:
        log = tmp_knowledge / "saido" / "compile_log.json"
        assert log.exists()
        assert json.loads(log.read_text()) == []


# ---------------------------------------------------------------------------
# Article CRUD
# ---------------------------------------------------------------------------

class TestArticleCRUD:
    """Full create / read / update / delete cycle."""

    def test_create_article(self, bridge: KnowledgeBridge) -> None:
        doc = bridge.create_article("hello-world", "Hello, world!", {"custom": "val"})
        assert doc is not None
        assert doc.slug == "hello-world"
        assert doc.body == "Hello, world!"

    def test_read_article(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("test-read", "Body text")
        doc = bridge.read_article("test-read")
        assert doc is not None
        assert doc.slug == "test-read"
        assert doc.body == "Body text"

    def test_read_article_not_found(self, bridge: KnowledgeBridge) -> None:
        assert bridge.read_article("nonexistent") is None

    def test_update_article_body(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("upd", "Old body")
        doc = bridge.update_article("upd", body="New body")
        assert doc is not None
        assert doc.body == "New body"

    def test_update_article_frontmatter(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("upd-fm", "Body", {"tag": "a"})
        doc = bridge.update_article("upd-fm", frontmatter_updates={"tag": "b"})
        assert doc is not None
        assert doc.frontmatter["tag"] == "b"

    def test_delete_article(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("del-me", "Delete this")
        assert bridge.delete_article("del-me") is True
        assert bridge.read_article("del-me") is None

    def test_delete_nonexistent_returns_true(self, bridge: KnowledgeBridge) -> None:
        # SmartRAG.delete is a no-op for missing slugs
        assert bridge.delete_article("ghost") is True

    def test_list_articles(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("a", "A body")
        bridge.create_article("b", "B body")
        articles = bridge.list_articles()
        slugs = [a[0] for a in articles]
        assert "a" in slugs
        assert "b" in slugs


# ---------------------------------------------------------------------------
# Frontmatter-only fast path
# ---------------------------------------------------------------------------

class TestFrontmatterFastPath:
    """read_article_frontmatter returns metadata without loading body."""

    def test_returns_frontmatter_dict(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("fm-test", "Body", {"custom_key": "custom_val"})
        fm = bridge.read_article_frontmatter("fm-test")
        assert fm is not None
        assert isinstance(fm, dict)
        assert "custom_key" in fm
        assert fm["custom_key"] == "custom_val"

    def test_returns_none_for_missing(self, bridge: KnowledgeBridge) -> None:
        assert bridge.read_article_frontmatter("missing") is None


# ---------------------------------------------------------------------------
# Search & query
# ---------------------------------------------------------------------------

class TestSearchAndQuery:
    """Verify search and query delegate to SmartRAG."""

    def test_search_returns_results(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("searchable", "The quick brown fox")
        bridge.reindex()
        results = bridge.search("fox")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0].slug == "searchable"

    def test_search_empty_store(self, bridge: KnowledgeBridge) -> None:
        results = bridge.search("nothing")
        assert results == []

    def test_query_returns_result(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("queryable", "Python is a programming language")
        bridge.reindex()
        qr = bridge.query("What is Python?")
        assert qr is not None
        assert len(qr.results) >= 1

    def test_query_empty_store(self, bridge: KnowledgeBridge) -> None:
        qr = bridge.query("anything")
        assert qr is not None
        assert len(qr.results) == 0


# ---------------------------------------------------------------------------
# Code structure frontmatter
# ---------------------------------------------------------------------------

class TestCodeStructure:
    """add_code_structure attaches AST/symbol info to frontmatter."""

    def test_add_code_structure(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("code-doc", "def hello(): pass")
        struct = {"functions": ["hello"], "classes": []}
        doc = bridge.add_code_structure("code-doc", struct)
        assert doc is not None
        assert doc.frontmatter["code_structure"] == struct

    def test_code_structure_retrievable(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("code-doc2", "class Foo: pass")
        struct = {"functions": [], "classes": ["Foo"]}
        bridge.add_code_structure("code-doc2", struct)
        fm = bridge.read_article_frontmatter("code-doc2")
        assert fm is not None
        assert fm["code_structure"] == struct


# ---------------------------------------------------------------------------
# Reindex
# ---------------------------------------------------------------------------

class TestReindex:
    """reindex delegates to SmartRAG."""

    def test_reindex_returns_count(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("idx-a", "Alpha")
        bridge.create_article("idx-b", "Beta")
        count = bridge.reindex(incremental=False)
        assert count == 2

    def test_reindex_empty(self, bridge: KnowledgeBridge) -> None:
        count = bridge.reindex()
        assert count == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    """Stats proxy returns document count and index info."""

    def test_stats_document_count(self, bridge: KnowledgeBridge) -> None:
        assert bridge.stats["document_count"] == 0
        bridge.create_article("s1", "One")
        bridge.reindex()
        assert bridge.stats["document_count"] == 1

    def test_stats_has_required_keys(self, bridge: KnowledgeBridge) -> None:
        s = bridge.stats
        assert "document_count" in s
        assert "index_size_bytes" in s


# ---------------------------------------------------------------------------
# Backlinks
# ---------------------------------------------------------------------------

class TestBacklinks:
    """get_backlinks finds articles that wikilink to a given slug."""

    def test_backlinks_found(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("target", "I am the target")
        bridge.create_article("linker", "See [[target]] for info")
        bridge.create_article("other", "No links here")
        bl = bridge.get_backlinks("target")
        assert "linker" in bl
        assert "other" not in bl

    def test_backlinks_empty(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("lonely", "No one links to me")
        assert bridge.get_backlinks("lonely") == []

    def test_backlinks_multiple_sources(self, bridge: KnowledgeBridge) -> None:
        bridge.create_article("hub", "Central doc")
        bridge.create_article("spoke1", "Ref [[hub]] here")
        bridge.create_article("spoke2", "Also [[hub]] here")
        bl = bridge.get_backlinks("hub")
        assert set(bl) == {"spoke1", "spoke2"}


# ---------------------------------------------------------------------------
# Ingest with compile
# ---------------------------------------------------------------------------

class TestIngestWithCompile:
    """ingest_with_compile orchestrates raw copy + ingest + optional compile."""

    def test_basic_ingest(self, bridge: KnowledgeBridge, tmp_path: Path) -> None:
        src = tmp_path / "example.md"
        src.write_text("# Example\n\nSome content here.", encoding="utf-8")
        result = bridge.ingest_with_compile(str(src))
        assert result is not None
        assert result.slug == "example"

    def test_raw_copy_created(self, bridge: KnowledgeBridge, tmp_path: Path, tmp_knowledge: Path) -> None:
        src = tmp_path / "raw_test.md"
        src.write_text("# Raw\nContent", encoding="utf-8")
        bridge.ingest_with_compile(str(src))
        assert (tmp_knowledge / "raw" / "raw_test.md").exists()

    def test_compiler_invoked(self, bridge: KnowledgeBridge, tmp_path: Path, tmp_knowledge: Path) -> None:
        src = tmp_path / "compiled.md"
        src.write_text("# Compiled\nBody", encoding="utf-8")
        compile_output = {"status": "ok", "warnings": 0}
        bridge.ingest_with_compile(
            str(src), compiler=lambda _result: compile_output
        )
        log = json.loads(
            (tmp_knowledge / "saido" / "compile_log.json").read_text()
        )
        assert len(log) == 1
        assert log[0]["compile"] == compile_output

    def test_structural_analyzer_invoked(self, bridge: KnowledgeBridge, tmp_path: Path) -> None:
        src = tmp_path / "analyzed.md"
        src.write_text("# Analyzed\ndef foo(): pass", encoding="utf-8")
        struct = {"functions": ["foo"]}
        bridge.ingest_with_compile(
            str(src), structural_analyzer=lambda _path: struct
        )
        doc = bridge.read_article("analyzed")
        assert doc is not None
        assert doc.frontmatter.get("code_structure") == struct

    def test_missing_file_returns_none(self, bridge: KnowledgeBridge) -> None:
        result = bridge.ingest_with_compile("/nonexistent/path.md")
        assert result is None
