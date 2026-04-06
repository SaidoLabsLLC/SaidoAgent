"""Tests for the WikiCompiler and LLM compile pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from saido_agent.knowledge.compile import (
    CompileResult,
    WikiCompiler,
    _build_compile_prompt,
    _extract_json,
    _validate_compile_response,
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
) -> MagicMock:
    """Build a mock KnowledgeBridge.

    Args:
        articles: slug -> {"body": ..., "frontmatter": {...}}
        all_articles: list of (slug, title, summary) for list_articles()
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

    return bridge


def _make_router(response_text: str) -> MagicMock:
    """Build a mock ModelRouter that makes _call_llm return a canned response."""
    router = MagicMock()
    router.select_model.return_value = ("ollama", "qwen3:8b")
    # We patch _call_llm directly rather than the full provider chain
    return router


def _good_llm_response(
    summary: str = "A concise summary of the article",
    concepts: list[str] | None = None,
    categories: list[str] | None = None,
    backlinks: list[str] | None = None,
    see_also: list[str] | None = None,
) -> str:
    """Generate a well-formed LLM JSON response."""
    return json.dumps({
        "summary": summary,
        "concepts": concepts or ["machine learning", "neural networks", "backpropagation"],
        "categories": categories or ["AI", "Deep Learning"],
        "backlinks": backlinks or [],
        "see_also": see_also or [],
    })


# ---------------------------------------------------------------------------
# _extract_json tests
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_raw_json(self):
        raw = '{"summary": "test", "concepts": ["a"]}'
        result = _extract_json(raw)
        assert result is not None
        assert result["summary"] == "test"

    def test_json_in_code_fence(self):
        raw = '```json\n{"summary": "test", "concepts": ["a"]}\n```'
        result = _extract_json(raw)
        assert result is not None
        assert result["summary"] == "test"

    def test_json_in_plain_fence(self):
        raw = '```\n{"summary": "test", "concepts": ["a"]}\n```'
        result = _extract_json(raw)
        assert result is not None

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"summary": "test", "concepts": ["a"]}\nDone.'
        result = _extract_json(raw)
        assert result is not None
        assert result["summary"] == "test"

    def test_invalid_json_returns_none(self):
        assert _extract_json("not json at all") is None

    def test_empty_string(self):
        assert _extract_json("") is None

    def test_array_returns_none(self):
        assert _extract_json('[1, 2, 3]') is None


# ---------------------------------------------------------------------------
# _validate_compile_response tests
# ---------------------------------------------------------------------------


class TestValidateResponse:
    def test_valid_response(self):
        data = {
            "summary": "A good summary",
            "concepts": ["concept1", "concept2", "concept3"],
            "categories": ["cat1"],
            "backlinks": ["existing-article"],
            "see_also": ["[[existing-article]]"],
        }
        result = _validate_compile_response(data, {"existing-article", "other"})
        assert result["summary"] == "A good summary"
        assert len(result["concepts"]) == 3
        assert result["backlinks"] == ["existing-article"]

    def test_summary_capped_at_200(self):
        data = {
            "summary": "x" * 300,
            "concepts": ["a"],
            "categories": [],
            "backlinks": [],
            "see_also": [],
        }
        result = _validate_compile_response(data, set())
        assert len(result["summary"]) == 200

    def test_empty_concepts_raises(self):
        data = {
            "summary": "test",
            "concepts": [],
            "categories": [],
            "backlinks": [],
            "see_also": [],
        }
        with pytest.raises(ValueError, match="concepts must be a non-empty list"):
            _validate_compile_response(data, set())

    def test_backlinks_filtered_to_existing(self):
        data = {
            "summary": "test",
            "concepts": ["a"],
            "categories": [],
            "backlinks": ["exists", "does-not-exist"],
            "see_also": [],
        }
        result = _validate_compile_response(data, {"exists"})
        assert result["backlinks"] == ["exists"]

    def test_see_also_filtered_to_existing(self):
        data = {
            "summary": "test",
            "concepts": ["a"],
            "categories": [],
            "backlinks": [],
            "see_also": ["[[exists]]", "[[gone]]"],
        }
        result = _validate_compile_response(data, {"exists"})
        assert result["see_also"] == ["[[exists]]"]


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_includes_existing_synopsis(self):
        prompt = _build_compile_prompt(
            title="Test Article",
            body="Some content here",
            existing_synopsis="Original synopsis",
            existing_fingerprint="keyword1, keyword2",
            existing_categories="cat1, cat2",
            existing_articles=["other-article"],
        )
        assert "Original synopsis" in prompt
        assert "keyword1, keyword2" in prompt
        assert "cat1, cat2" in prompt
        assert "Test Article" in prompt
        assert "other-article" in prompt

    def test_includes_code_structure(self):
        prompt = _build_compile_prompt(
            title="Code Module",
            body="def foo(): pass",
            existing_synopsis="",
            existing_fingerprint="",
            existing_categories="",
            existing_articles=[],
            code_structure={
                "language": "python",
                "functions": ["foo"],
                "classes": [],
                "endpoints": [],
            },
        )
        assert "Language: python" in prompt
        assert "foo" in prompt

    def test_omits_code_structure_when_none(self):
        prompt = _build_compile_prompt(
            title="Doc",
            body="text",
            existing_synopsis="",
            existing_fingerprint="",
            existing_categories="",
            existing_articles=[],
            code_structure=None,
        )
        assert "Code Structure" not in prompt


# ---------------------------------------------------------------------------
# WikiCompiler.compile() tests
# ---------------------------------------------------------------------------


class TestWikiCompilerCompile:
    def test_successful_compile(self):
        articles = {
            "neural-networks": {
                "body": "Neural networks are computational models...",
                "frontmatter": {
                    "title": "Neural Networks",
                    "synopsis": "Old extractive synopsis",
                    "fingerprint": "neural, network",
                    "categories": "AI",
                },
            },
            "backpropagation": {
                "body": "Backprop content",
                "frontmatter": {"title": "Backpropagation"},
            },
        }
        bridge = _make_bridge(articles)
        router = _make_router("")

        compiler = WikiCompiler(bridge, router)
        llm_response = _good_llm_response(
            backlinks=["backpropagation"],
            see_also=["[[backpropagation]]"],
        )

        with patch.object(compiler, "_call_llm", return_value=llm_response):
            result = compiler.compile("neural-networks")

        assert result.status == "compiled"
        assert result.summary == "A concise summary of the article"
        assert len(result.concepts) == 3
        assert "backpropagation" in result.backlinks

        # Verify bridge was updated
        bridge.update_article.assert_called_once()
        call_kwargs = bridge.update_article.call_args
        fm = call_kwargs.kwargs.get(
            "frontmatter_updates",
            call_kwargs[1].get("frontmatter_updates", {}),
        )
        assert fm["synopsis"] == "A concise summary of the article"
        assert fm["compiled"] is True

    def test_document_not_found(self):
        bridge = _make_bridge({})
        compiler = WikiCompiler(bridge)
        result = compiler.compile("nonexistent")
        assert result.status == "failed"
        assert "not found" in result.error.lower()

    def test_llm_returns_none(self):
        articles = {
            "test-article": {
                "body": "content",
                "frontmatter": {"title": "Test"},
            },
        }
        bridge = _make_bridge(articles)
        compiler = WikiCompiler(bridge, MagicMock())

        with patch.object(compiler, "_call_llm", return_value=None):
            result = compiler.compile("test-article")

        assert result.status == "failed"
        assert "no response" in result.error.lower()

    def test_malformed_response_retries_then_fails(self):
        articles = {
            "test-article": {
                "body": "content",
                "frontmatter": {"title": "Test"},
            },
        }
        bridge = _make_bridge(articles)
        compiler = WikiCompiler(bridge, MagicMock())

        # Both attempts return garbage
        with patch.object(
            compiler, "_call_llm", return_value="not valid json at all"
        ):
            result = compiler.compile("test-article")

        assert result.status == "failed"
        assert "parse" in result.error.lower()
        # Bridge should NOT have been updated
        bridge.update_article.assert_not_called()

    def test_malformed_first_response_succeeds_on_retry(self):
        articles = {
            "test-article": {
                "body": "content",
                "frontmatter": {"title": "Test"},
            },
        }
        bridge = _make_bridge(articles)
        compiler = WikiCompiler(bridge, MagicMock())

        good_response = _good_llm_response()
        # First call returns garbage, second returns valid JSON
        with patch.object(
            compiler,
            "_call_llm",
            side_effect=["totally not json", good_response],
        ):
            result = compiler.compile("test-article")

        assert result.status == "compiled"
        assert result.summary == "A concise summary of the article"

    def test_summary_capped_at_200_chars(self):
        articles = {
            "test-article": {
                "body": "content",
                "frontmatter": {"title": "Test"},
            },
        }
        bridge = _make_bridge(articles)
        compiler = WikiCompiler(bridge, MagicMock())

        long_summary = "x" * 300
        llm_response = _good_llm_response(summary=long_summary)

        with patch.object(compiler, "_call_llm", return_value=llm_response):
            result = compiler.compile("test-article")

        assert result.status == "compiled"
        assert len(result.summary) <= 200

    def test_concepts_validated_non_empty(self):
        articles = {
            "test-article": {
                "body": "content",
                "frontmatter": {"title": "Test"},
            },
        }
        bridge = _make_bridge(articles)
        compiler = WikiCompiler(bridge, MagicMock())

        # LLM returns empty concepts
        bad_response = json.dumps({
            "summary": "test",
            "concepts": [],
            "categories": [],
            "backlinks": [],
            "see_also": [],
        })

        with patch.object(compiler, "_call_llm", return_value=bad_response):
            result = compiler.compile("test-article")

        assert result.status == "failed"
        assert "validation" in result.error.lower()

    def test_backlinks_validated_against_existing(self):
        articles = {
            "test-article": {
                "body": "content",
                "frontmatter": {"title": "Test"},
            },
            "real-article": {
                "body": "exists",
                "frontmatter": {"title": "Real"},
            },
        }
        bridge = _make_bridge(articles)
        compiler = WikiCompiler(bridge, MagicMock())

        llm_response = _good_llm_response(
            backlinks=["real-article", "fake-article"]
        )

        with patch.object(compiler, "_call_llm", return_value=llm_response):
            result = compiler.compile("test-article")

        assert result.status == "compiled"
        assert "real-article" in result.backlinks
        assert "fake-article" not in result.backlinks

    def test_article_updated_with_enriched_frontmatter(self):
        articles = {
            "test-article": {
                "body": "content",
                "frontmatter": {"title": "Test"},
            },
        }
        bridge = _make_bridge(articles)
        compiler = WikiCompiler(bridge, MagicMock())

        llm_response = _good_llm_response(
            categories=["Science", "Tech"],
        )

        with patch.object(compiler, "_call_llm", return_value=llm_response):
            compiler.compile("test-article")

        bridge.update_article.assert_called_once()
        call_args = bridge.update_article.call_args
        fm = call_args.kwargs.get(
            "frontmatter_updates",
            call_args[1].get("frontmatter_updates", {}),
        )
        assert "concepts" in fm
        assert fm["categories"] == ["Science", "Tech"]
        assert fm["compiled"] is True


# ---------------------------------------------------------------------------
# WikiCompiler.compile_batch() tests
# ---------------------------------------------------------------------------


class TestWikiCompilerBatch:
    def test_batch_processes_all(self):
        articles = {
            "article-1": {
                "body": "content 1",
                "frontmatter": {"title": "Article 1"},
            },
            "article-2": {
                "body": "content 2",
                "frontmatter": {"title": "Article 2"},
            },
            "article-3": {
                "body": "content 3",
                "frontmatter": {"title": "Article 3"},
            },
        }
        bridge = _make_bridge(articles)
        compiler = WikiCompiler(bridge, MagicMock())

        with patch.object(
            compiler, "_call_llm", return_value=_good_llm_response()
        ):
            results = compiler.compile_batch(
                ["article-1", "article-2", "article-3"]
            )

        assert len(results) == 3
        assert all(r.status == "compiled" for r in results)

    def test_batch_continues_on_failure(self):
        articles = {
            "good-article": {
                "body": "content",
                "frontmatter": {"title": "Good"},
            },
            "bad-article": {
                "body": "content",
                "frontmatter": {"title": "Bad"},
            },
        }
        bridge = _make_bridge(articles)
        compiler = WikiCompiler(bridge, MagicMock())

        def _selective_llm(prompt):
            # Return garbage for bad-article (it will appear in the prompt)
            if "Bad" in prompt:
                return "not json"
            return _good_llm_response()

        with patch.object(compiler, "_call_llm", side_effect=_selective_llm):
            results = compiler.compile_batch(["good-article", "bad-article"])

        assert len(results) == 2
        statuses = {r.slug: r.status for r in results}
        assert statuses["good-article"] == "compiled"
        assert statuses["bad-article"] == "failed"

    def test_batch_handles_exception_in_compile(self):
        articles = {
            "ok-article": {
                "body": "content",
                "frontmatter": {"title": "OK"},
            },
        }
        bridge = _make_bridge(articles)
        compiler = WikiCompiler(bridge, MagicMock())

        with patch.object(
            compiler, "compile", side_effect=RuntimeError("boom")
        ):
            results = compiler.compile_batch(["ok-article"])

        assert len(results) == 1
        assert results[0].status == "failed"
        assert "boom" in results[0].error

    def test_batch_progress_callback(self):
        articles = {
            "a1": {"body": "c", "frontmatter": {"title": "Alpha"}},
            "a2": {"body": "c", "frontmatter": {"title": "Beta"}},
        }
        bridge = _make_bridge(articles)
        compiler = WikiCompiler(bridge, MagicMock())

        progress_calls: list[tuple] = []

        def track_progress(current, total, title):
            progress_calls.append((current, total, title))

        with patch.object(
            compiler, "_call_llm", return_value=_good_llm_response()
        ):
            compiler.compile_batch(["a1", "a2"], progress_callback=track_progress)

        assert len(progress_calls) == 2
        assert progress_calls[0] == (1, 2, "Alpha")
        assert progress_calls[1] == (2, 2, "Beta")


# ---------------------------------------------------------------------------
# CompileResult dataclass tests
# ---------------------------------------------------------------------------


class TestCompileResult:
    def test_success_result(self):
        r = CompileResult(
            slug="test",
            status="compiled",
            summary="A summary",
            concepts=["a", "b"],
            categories=["cat"],
            backlinks=["other"],
        )
        assert r.slug == "test"
        assert r.status == "compiled"
        assert r.error is None
        assert len(r.concepts) == 2

    def test_failure_result(self):
        r = CompileResult(
            slug="test",
            status="failed",
            error="Something went wrong",
        )
        assert r.status == "failed"
        assert r.error == "Something went wrong"
        assert r.summary == ""
        assert r.concepts == []

    def test_skipped_result(self):
        r = CompileResult(slug="test", status="skipped")
        assert r.status == "skipped"
        assert r.error is None


# ---------------------------------------------------------------------------
# Section-split parent compile tests
# ---------------------------------------------------------------------------


class TestSectionSplitCompile:
    def test_parent_gets_updated_section_map(self):
        articles = {
            "parent-article": {
                "body": "Parent overview",
                "frontmatter": {
                    "title": "Parent",
                    "children": ["child-1", "child-2"],
                    "section_map": {
                        "child-1": {"synopsis": "old"},
                        "child-2": {"synopsis": "old"},
                    },
                },
            },
            "child-1": {
                "body": "First child content about databases",
                "frontmatter": {"title": "Child 1"},
            },
            "child-2": {
                "body": "Second child content about caching",
                "frontmatter": {"title": "Child 2"},
            },
        }
        bridge = _make_bridge(articles)
        compiler = WikiCompiler(bridge, MagicMock())

        with patch.object(
            compiler, "_call_llm", return_value=_good_llm_response()
        ):
            result = compiler.compile("parent-article")

        assert result.status == "compiled"
        assert result.slug == "parent-article"

        # Verify parent was updated with section_map
        parent_update_calls = [
            c for c in bridge.update_article.call_args_list
            if c[0][0] == "parent-article" or c.kwargs.get("slug") == "parent-article"
        ]
        # The parent should have been updated (possibly via positional arg)
        found_section_map_update = False
        for call in bridge.update_article.call_args_list:
            args, kwargs = call
            fm = kwargs.get("frontmatter_updates", {})
            if "section_map" in fm:
                found_section_map_update = True
                section_map = fm["section_map"]
                # Children should have updated synopses
                assert "child-1" in section_map
                assert "child-2" in section_map
                assert "synopsis" in section_map["child-1"]
                break

        assert found_section_map_update, "Parent should have received section_map update"

    def test_parent_aggregates_child_metadata(self):
        articles = {
            "parent": {
                "body": "Parent",
                "frontmatter": {
                    "title": "Parent",
                    "children": ["child-a"],
                    "section_map": {},
                },
            },
            "child-a": {
                "body": "Child A content",
                "frontmatter": {"title": "Child A"},
            },
        }
        bridge = _make_bridge(articles)
        compiler = WikiCompiler(bridge, MagicMock())

        with patch.object(
            compiler, "_call_llm", return_value=_good_llm_response(
                concepts=["databases", "indexing"],
                categories=["Infrastructure"],
            )
        ):
            result = compiler.compile("parent")

        assert result.status == "compiled"
        assert "databases" in result.concepts
        assert "Infrastructure" in result.categories


# ---------------------------------------------------------------------------
# IngestPipeline.process_compile_queue() integration
# ---------------------------------------------------------------------------


class TestProcessCompileQueue:
    def test_process_compile_queue_delegates_to_compiler(self):
        from saido_agent.knowledge.ingest import IngestPipeline

        bridge = MagicMock()
        pipeline = IngestPipeline(bridge)

        # Manually populate the compile queue
        pipeline._compile_queue = ["slug-1", "slug-2"]

        mock_compiler = MagicMock()
        mock_compiler.compile_batch.return_value = [
            CompileResult(slug="slug-1", status="compiled"),
            CompileResult(slug="slug-2", status="compiled"),
        ]

        results = pipeline.process_compile_queue(mock_compiler)

        assert len(results) == 2
        mock_compiler.compile_batch.assert_called_once_with(
            ["slug-1", "slug-2"], progress_callback=None
        )
        # Queue should be cleared after processing
        assert pipeline.get_compile_queue() == []

    def test_process_empty_queue_returns_empty(self):
        from saido_agent.knowledge.ingest import IngestPipeline

        bridge = MagicMock()
        pipeline = IngestPipeline(bridge)

        mock_compiler = MagicMock()
        results = pipeline.process_compile_queue(mock_compiler)

        assert results == []
        mock_compiler.compile_batch.assert_not_called()
