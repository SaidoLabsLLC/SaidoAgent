"""Tests for the WikiIndexer LLM-powered indexing layer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from saido_agent.knowledge.index import (
    IndexResult,
    WikiIndexer,
    _extract_json,
)


# ---------------------------------------------------------------------------
# Helpers: mock Document and bridge
# ---------------------------------------------------------------------------


@dataclass
class FakeDocument:
    """Minimal stand-in for SmartRAG Document."""

    slug: str
    body: str
    frontmatter: dict


def _make_bridge(
    articles: dict[str, dict] | None = None,
    all_articles: list[tuple[str, str, str]] | None = None,
    knowledge_root: str | None = None,
) -> MagicMock:
    """Build a mock KnowledgeBridge.

    Args:
        articles: slug -> {"body": ..., "frontmatter": {...}}
        all_articles: list of (slug, title, summary) for list_articles()
        knowledge_root: path for _root attribute
    """
    articles = articles or {}
    if all_articles is None:
        all_articles = [
            (slug, meta.get("frontmatter", {}).get("title", slug), "")
            for slug, meta in articles.items()
        ]

    bridge = MagicMock()

    def _read_article(slug):
        if slug in articles:
            return FakeDocument(
                slug=slug,
                body=articles[slug].get("body", ""),
                frontmatter=articles[slug].get("frontmatter", {}),
            )
        return None

    def _read_frontmatter(slug):
        if slug in articles:
            return articles[slug].get("frontmatter", {})
        return None

    bridge.read_article.side_effect = _read_article
    bridge.read_article_frontmatter.side_effect = _read_frontmatter
    bridge.list_articles.return_value = all_articles
    bridge.update_article.return_value = FakeDocument(
        slug="updated", body="", frontmatter={}
    )

    if knowledge_root:
        bridge._root = Path(knowledge_root)
    else:
        bridge._root = Path("knowledge")

    return bridge


# ---------------------------------------------------------------------------
# Test: IndexResult dataclass
# ---------------------------------------------------------------------------


class TestIndexResult:
    def test_default_values(self):
        result = IndexResult()
        assert result.articles_processed == 0
        assert result.articles_skipped == 0
        assert result.concept_map_updated is False
        assert result.category_tree_updated is False
        assert result.duration_ms == 0
        assert result.errors == []

    def test_custom_values(self):
        result = IndexResult(
            articles_processed=5,
            articles_skipped=3,
            concept_map_updated=True,
            category_tree_updated=True,
            duration_ms=1500,
        )
        assert result.articles_processed == 5
        assert result.articles_skipped == 3
        assert result.concept_map_updated is True
        assert result.duration_ms == 1500


# ---------------------------------------------------------------------------
# Test: JSON extraction
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_raw_json(self):
        result = _extract_json('{"summary": "test"}')
        assert result == {"summary": "test"}

    def test_fenced_json(self):
        result = _extract_json('```json\n{"summary": "test"}\n```')
        assert result == {"summary": "test"}

    def test_invalid_json(self):
        result = _extract_json("not json at all")
        assert result is None

    def test_json_with_surrounding_text(self):
        result = _extract_json('Here is the result: {"key": "val"} done.')
        assert result == {"key": "val"}


# ---------------------------------------------------------------------------
# Test: Enriched summary generation
# ---------------------------------------------------------------------------


class TestEnrichedSummaries:
    def test_generate_summary_calls_llm(self, tmp_path):
        """Enriched summary generation sends body to LLM and updates frontmatter."""
        articles = {
            "jwt-auth": {
                "body": "JWT authentication is a token-based auth mechanism.",
                "frontmatter": {"title": "JWT Auth", "concepts": ["jwt"]},
            }
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))

        indexer = WikiIndexer(bridge, model_router=None)
        # Patch _call_llm directly on the instance
        indexer._call_llm = MagicMock(
            return_value='{"summary": "JWT auth provides stateless token-based authentication for APIs."}'
        )

        count = indexer.generate_enriched_summaries(slugs=["jwt-auth"])
        assert count == 1
        bridge.update_article.assert_called_once()
        call_kwargs = bridge.update_article.call_args
        fm_updates = call_kwargs[1].get(
            "frontmatter_updates", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {}
        )
        assert "enriched_summary" in fm_updates

    def test_generate_summary_no_router_returns_zero(self, tmp_path):
        """Without a model router, summaries cannot be generated."""
        articles = {
            "test": {
                "body": "Some content here.",
                "frontmatter": {"title": "Test"},
            }
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        # _call_llm returns None without a router, so summary generation fails
        count = indexer.generate_enriched_summaries(slugs=["test"])
        assert count == 0

    def test_generate_all_summaries(self, tmp_path):
        """When slugs=None, all articles are summarized."""
        articles = {
            "a": {"body": "Article A content.", "frontmatter": {"title": "A"}},
            "b": {"body": "Article B content.", "frontmatter": {"title": "B"}},
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)
        indexer._call_llm = MagicMock(
            return_value='{"summary": "A summary."}'
        )

        count = indexer.generate_enriched_summaries()
        assert count == 2
        assert indexer._call_llm.call_count == 2


# ---------------------------------------------------------------------------
# Test: Concept map generation
# ---------------------------------------------------------------------------


class TestConceptMap:
    def test_concept_map_has_valid_structure(self, tmp_path):
        """Concept map must have nodes and edges keys."""
        articles = {
            "jwt-auth": {
                "body": "JWT auth article.",
                "frontmatter": {
                    "title": "JWT Auth",
                    "concepts": ["jwt", "authentication"],
                },
            },
            "oauth-guide": {
                "body": "OAuth guide article.",
                "frontmatter": {
                    "title": "OAuth Guide",
                    "concepts": ["oauth", "authentication"],
                },
            },
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        # Without router, we get nodes but no LLM-generated edges
        result = indexer.generate_concept_map()

        assert "nodes" in result
        assert "edges" in result
        assert isinstance(result["nodes"], list)
        assert isinstance(result["edges"], list)

        # Verify nodes contain expected concepts
        node_ids = {n["id"] for n in result["nodes"]}
        assert "jwt" in node_ids
        assert "oauth" in node_ids
        assert "authentication" in node_ids

    def test_concept_map_with_llm_edges(self, tmp_path):
        """When LLM is available, edges are generated between concepts."""
        articles = {
            "jwt-auth": {
                "body": "JWT auth.",
                "frontmatter": {"concepts": ["jwt", "oauth"]},
            },
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=MagicMock())

        llm_response = json.dumps({
            "edges": [
                {"source": "jwt", "target": "oauth", "relation": "is-a"}
            ]
        })
        indexer._call_llm = MagicMock(return_value=llm_response)

        result = indexer.generate_concept_map()
        assert len(result["edges"]) == 1
        assert result["edges"][0]["source"] == "jwt"
        assert result["edges"][0]["target"] == "oauth"
        assert result["edges"][0]["relation"] == "is-a"

    def test_concept_map_node_article_counts(self, tmp_path):
        """Nodes should track how many articles reference each concept."""
        articles = {
            "a": {"body": "A.", "frontmatter": {"concepts": ["api", "rest"]}},
            "b": {"body": "B.", "frontmatter": {"concepts": ["api", "grpc"]}},
            "c": {"body": "C.", "frontmatter": {"concepts": ["api"]}},
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        result = indexer.generate_concept_map()
        api_node = next(n for n in result["nodes"] if n["id"] == "api")
        assert api_node["articles"] == 3

    def test_concept_map_persisted_to_disk(self, tmp_path):
        """Concept map should be saved to concept_map.json."""
        articles = {
            "x": {"body": "X.", "frontmatter": {"concepts": ["testing"]}},
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        indexer.generate_concept_map()

        path = tmp_path / "saido" / "concept_map.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "nodes" in data
        assert "edges" in data

    def test_concept_map_invalid_edges_filtered(self, tmp_path):
        """Edges referencing non-existent concepts should be filtered out."""
        articles = {
            "a": {"body": "A.", "frontmatter": {"concepts": ["jwt"]}},
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=MagicMock())

        llm_response = json.dumps({
            "edges": [
                {"source": "jwt", "target": "nonexistent", "relation": "uses"},
                {"source": "jwt", "target": "jwt", "relation": "self"},
            ]
        })
        indexer._call_llm = MagicMock(return_value=llm_response)

        result = indexer.generate_concept_map()
        # "nonexistent" is not a valid concept, "jwt" -> "jwt" is valid
        assert len(result["edges"]) == 1
        assert result["edges"][0]["source"] == "jwt"
        assert result["edges"][0]["target"] == "jwt"


# ---------------------------------------------------------------------------
# Test: Category tree generation
# ---------------------------------------------------------------------------


class TestCategoryTree:
    def test_category_tree_valid_hierarchy(self, tmp_path):
        """Category tree must have a root key with a list."""
        articles = {
            "api-guide": {
                "body": "API guide.",
                "frontmatter": {"categories": ["Backend", "API"]},
            },
            "db-design": {
                "body": "DB design.",
                "frontmatter": {"categories": ["Backend", "Database"]},
            },
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        result = indexer.generate_category_tree()
        assert "root" in result
        assert isinstance(result["root"], list)
        assert len(result["root"]) > 0

    def test_category_tree_with_llm(self, tmp_path):
        """LLM should organize categories into a hierarchy."""
        articles = {
            "a": {"body": "A.", "frontmatter": {"categories": ["API", "Database"]}},
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=MagicMock())

        llm_response = json.dumps({
            "root": [
                {
                    "name": "Engineering",
                    "children": [
                        {"name": "API", "children": []},
                        {"name": "Database", "children": []},
                    ],
                }
            ]
        })
        indexer._call_llm = MagicMock(return_value=llm_response)

        result = indexer.generate_category_tree()
        assert result["root"][0]["name"] == "Engineering"
        assert len(result["root"][0]["children"]) == 2

    def test_category_tree_persisted_to_disk(self, tmp_path):
        """Category tree should be saved to category_tree.json."""
        articles = {
            "x": {"body": "X.", "frontmatter": {"categories": ["Testing"]}},
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        indexer.generate_category_tree()

        path = tmp_path / "saido" / "category_tree.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "root" in data


# ---------------------------------------------------------------------------
# Test: Incremental indexing
# ---------------------------------------------------------------------------


class TestIncrementalIndexing:
    def test_incremental_skips_unchanged(self, tmp_path):
        """Articles with unchanged content hashes should be skipped."""
        body = "JWT auth provides token-based authentication."
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

        articles = {
            "jwt-auth": {
                "body": body,
                "frontmatter": {"title": "JWT Auth", "concepts": ["jwt"]},
            },
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))

        # Pre-populate index state with current hash
        state_path = tmp_path / "saido" / "index_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"jwt-auth": content_hash}), encoding="utf-8"
        )

        indexer = WikiIndexer(bridge, model_router=None)
        result = indexer.reindex()

        assert result.articles_skipped == 1
        assert result.articles_processed == 0

    def test_incremental_processes_changed(self, tmp_path):
        """Articles with changed content should be processed."""
        articles = {
            "jwt-auth": {
                "body": "Updated JWT auth content.",
                "frontmatter": {"title": "JWT Auth", "concepts": ["jwt"]},
            },
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))

        # Pre-populate with stale hash
        state_path = tmp_path / "saido" / "index_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"jwt-auth": "stale_hash_value"}), encoding="utf-8"
        )

        indexer = WikiIndexer(bridge, model_router=None)
        indexer._call_llm = MagicMock(
            return_value='{"summary": "Updated summary."}'
        )

        result = indexer.reindex()
        assert result.articles_processed == 1
        assert result.articles_skipped == 0

    def test_full_reindex_processes_all(self, tmp_path):
        """Full reindex should process ALL articles regardless of hash."""
        body = "Some article body."
        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

        articles = {
            "a": {"body": body, "frontmatter": {"title": "A", "concepts": ["x"]}},
            "b": {"body": body, "frontmatter": {"title": "B", "concepts": ["y"]}},
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))

        # Pre-populate with current hashes (both up to date)
        state_path = tmp_path / "saido" / "index_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"a": content_hash, "b": content_hash}), encoding="utf-8"
        )

        indexer = WikiIndexer(bridge, model_router=None)
        indexer._call_llm = MagicMock(
            return_value='{"summary": "A summary."}'
        )

        result = indexer.reindex(full=True)
        assert result.articles_processed == 2
        assert result.articles_skipped == 0

    def test_index_state_tracks_hashes(self, tmp_path):
        """After reindex, index_state.json should contain content hashes."""
        body = "Article body content."
        articles = {
            "test-article": {
                "body": body,
                "frontmatter": {"title": "Test"},
            },
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))

        indexer = WikiIndexer(bridge, model_router=None)
        indexer._call_llm = MagicMock(
            return_value='{"summary": "A test summary."}'
        )

        indexer.reindex()

        state_path = tmp_path / "saido" / "index_state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        expected_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        assert state.get("test-article") == expected_hash

    def test_reindex_returns_timing(self, tmp_path):
        """Reindex result should have non-negative duration_ms."""
        bridge = _make_bridge({}, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        result = indexer.reindex()
        assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# Test: Empty store handling
# ---------------------------------------------------------------------------


class TestEmptyStore:
    def test_empty_store_reindex(self, tmp_path):
        """Reindex on empty store should return zeros gracefully."""
        bridge = _make_bridge({}, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        result = indexer.reindex()
        assert result.articles_processed == 0
        assert result.articles_skipped == 0
        assert result.errors == []

    def test_empty_store_concept_map(self, tmp_path):
        """Concept map on empty store should return empty nodes and edges."""
        bridge = _make_bridge({}, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        result = indexer.generate_concept_map()
        assert result == {"nodes": [], "edges": []}

    def test_empty_store_category_tree(self, tmp_path):
        """Category tree on empty store should return empty root."""
        bridge = _make_bridge({}, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        result = indexer.generate_category_tree()
        assert result == {"root": []}

    def test_empty_store_summaries(self, tmp_path):
        """Generating summaries on empty store should return 0."""
        bridge = _make_bridge({}, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        count = indexer.generate_enriched_summaries()
        assert count == 0

    def test_load_nonexistent_concept_map(self, tmp_path):
        """Loading concept map when file doesn't exist should return default."""
        bridge = _make_bridge({}, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        result = indexer.load_concept_map()
        assert result == {"nodes": [], "edges": []}

    def test_load_nonexistent_category_tree(self, tmp_path):
        """Loading category tree when file doesn't exist should return default."""
        bridge = _make_bridge({}, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        result = indexer.load_category_tree()
        assert result == {"root": []}


# ---------------------------------------------------------------------------
# Test: Concept map and category tree reloading from disk
# ---------------------------------------------------------------------------


class TestPersistenceRoundtrip:
    def test_concept_map_roundtrip(self, tmp_path):
        """Generate then load concept map should return same data."""
        articles = {
            "a": {"body": "A.", "frontmatter": {"concepts": ["testing"]}},
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        generated = indexer.generate_concept_map()
        loaded = indexer.load_concept_map()
        assert generated == loaded

    def test_category_tree_roundtrip(self, tmp_path):
        """Generate then load category tree should return same data."""
        articles = {
            "a": {"body": "A.", "frontmatter": {"categories": ["Backend"]}},
        }
        bridge = _make_bridge(articles, knowledge_root=str(tmp_path))
        indexer = WikiIndexer(bridge, model_router=None)

        generated = indexer.generate_category_tree()
        loaded = indexer.load_category_tree()
        assert generated == loaded
