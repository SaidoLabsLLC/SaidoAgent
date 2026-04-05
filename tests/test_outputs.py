"""Tests for saido_agent.knowledge.outputs — ReportGenerator."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from saido_agent.knowledge.outputs import (
    ReportGenerator,
    ReportResult,
    _slugify,
    _MAX_REPORT_ARTICLES,
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
# Test data
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
        score=0.80,
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeBridgeConfig:
    knowledge_root: str = "knowledge"


def _make_bridge(
    articles: dict[str, FakeDocument] | None = None,
    search_results: list[FakeSearchResult] | None = None,
    doc_count: int | None = None,
) -> MagicMock:
    """Build a mock KnowledgeBridge."""
    arts = articles or ARTICLES
    sr = search_results if search_results is not None else SEARCH_RESULTS
    bridge = MagicMock()
    bridge._config = FakeBridgeConfig()
    bridge.available = True
    bridge.stats = {"document_count": doc_count if doc_count is not None else len(arts)}
    bridge.search.return_value = sr
    bridge.read_article.side_effect = lambda slug: arts.get(slug)
    bridge.list_articles.return_value = [
        (slug, doc.title, doc.body[:80]) for slug, doc in arts.items()
    ]
    return bridge


def _make_generator(
    bridge: MagicMock | None = None,
    llm_response: str | None = None,
    tmp_path: Path | None = None,
) -> ReportGenerator:
    """Build a ReportGenerator with mocked LLM."""
    if bridge is None:
        bridge = _make_bridge()

    # Override knowledge_root to use tmp_path so files land in a temp dir
    if tmp_path is not None:
        bridge._config.knowledge_root = str(tmp_path / "knowledge")

    gen = ReportGenerator(bridge, model_router=MagicMock())

    # Mock the LLM call
    if llm_response is not None:
        gen._call_llm = MagicMock(return_value=llm_response)
    else:
        gen._call_llm = MagicMock(return_value=None)

    return gen


SAMPLE_REPORT = """\
## Executive Summary

Python is a versatile language supporting multiple paradigms [Python Basics].

## Key Findings

### Language Design
Python was created by Guido van Rossum and supports OOP [Python Basics].

### Async Capabilities
Python asyncio enables concurrent programming [Python Async Programming].

## Recommendations

Use asyncio for I/O-bound workloads.

## Sources

- [Python Basics]
- [Python Async Programming]
"""


# ---------------------------------------------------------------------------
# Tests: generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for ReportGenerator.generate_report."""

    def test_generates_report_with_expected_sections(self, tmp_path: Path):
        gen = _make_generator(llm_response=SAMPLE_REPORT, tmp_path=tmp_path)
        result = gen.generate_report("Python programming")

        assert result.status == "generated"
        assert result.error is None
        assert result.word_count > 0

        # Verify file was written
        path = Path(result.path)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "## Executive Summary" in content
        assert "## Key Findings" in content
        assert "## Recommendations" in content
        assert "## Sources" in content

    def test_report_saved_to_correct_path(self, tmp_path: Path):
        gen = _make_generator(llm_response=SAMPLE_REPORT, tmp_path=tmp_path)
        result = gen.generate_report("Python programming")

        path = Path(result.path)
        assert "outputs" in str(path)
        assert "reports" in str(path)
        assert path.name.startswith("python-programming-")
        assert path.suffix == ".md"

    def test_report_result_populated_correctly(self, tmp_path: Path):
        gen = _make_generator(llm_response=SAMPLE_REPORT, tmp_path=tmp_path)
        result = gen.generate_report("Python programming")

        assert isinstance(result, ReportResult)
        assert result.title == "Python programming"
        assert result.status == "generated"
        assert result.word_count > 0
        assert result.articles_cited == 2  # Both articles cited in SAMPLE_REPORT
        assert result.path != ""
        assert result.error is None

    def test_empty_store_returns_failed(self, tmp_path: Path):
        bridge = _make_bridge(doc_count=0)
        gen = _make_generator(bridge=bridge, llm_response=SAMPLE_REPORT, tmp_path=tmp_path)
        result = gen.generate_report("Python")

        assert result.status == "failed"
        assert "empty" in result.error.lower()
        assert result.word_count == 0

    def test_no_search_results_returns_failed(self, tmp_path: Path):
        bridge = _make_bridge(search_results=[])
        gen = _make_generator(bridge=bridge, llm_response=SAMPLE_REPORT, tmp_path=tmp_path)
        result = gen.generate_report("quantum physics")

        assert result.status == "failed"
        assert "no articles" in result.error.lower()

    def test_llm_failure_returns_failed(self, tmp_path: Path):
        gen = _make_generator(llm_response=None, tmp_path=tmp_path)
        # _call_llm returns None
        result = gen.generate_report("Python")

        assert result.status == "failed"
        assert "llm" in result.error.lower()

    def test_articles_read_from_bridge(self, tmp_path: Path):
        bridge = _make_bridge()
        gen = _make_generator(bridge=bridge, llm_response=SAMPLE_REPORT, tmp_path=tmp_path)
        gen.generate_report("Python")

        bridge.search.assert_called_once()
        # Should have read the articles returned by search
        assert bridge.read_article.call_count == len(SEARCH_RESULTS)

    def test_prompt_includes_article_content(self, tmp_path: Path):
        gen = _make_generator(llm_response=SAMPLE_REPORT, tmp_path=tmp_path)
        gen.generate_report("Python")

        # Inspect the prompt sent to _call_llm
        call_args = gen._call_llm.call_args[0][0]
        assert "Python Basics" in call_args
        assert "Python Async Programming" in call_args
        assert "Guido van Rossum" in call_args


# ---------------------------------------------------------------------------
# Tests: export_docs
# ---------------------------------------------------------------------------


class TestExportDocs:
    """Tests for ReportGenerator.export_docs."""

    def test_creates_zip_with_articles(self, tmp_path: Path):
        gen = _make_generator(tmp_path=tmp_path)
        zip_path = gen.export_docs(output_dir=str(tmp_path))

        assert zip_path != ""
        zp = Path(zip_path)
        assert zp.exists()
        assert zp.suffix == ".zip"

        with zipfile.ZipFile(zp, "r") as zf:
            names = zf.namelist()
            assert len(names) == len(ARTICLES)
            for slug in ARTICLES:
                assert f"{slug}.md" in names

    def test_zip_contains_article_content(self, tmp_path: Path):
        gen = _make_generator(tmp_path=tmp_path)
        zip_path = gen.export_docs(output_dir=str(tmp_path))

        with zipfile.ZipFile(zip_path, "r") as zf:
            content = zf.read("python-basics.md").decode("utf-8")
            assert "# Python Basics" in content
            assert "Guido van Rossum" in content

    def test_empty_store_returns_empty_string(self, tmp_path: Path):
        bridge = _make_bridge()
        bridge.list_articles.return_value = []
        gen = _make_generator(bridge=bridge, tmp_path=tmp_path)

        result = gen.export_docs(output_dir=str(tmp_path))
        assert result == ""

    def test_default_output_dir(self, tmp_path: Path):
        gen = _make_generator(tmp_path=tmp_path)
        zip_path = gen.export_docs()

        assert zip_path != ""
        assert "exports" in zip_path
        assert Path(zip_path).exists()


# ---------------------------------------------------------------------------
# Tests: export_article
# ---------------------------------------------------------------------------


class TestExportArticle:
    """Tests for ReportGenerator.export_article."""

    def test_creates_standalone_markdown(self, tmp_path: Path):
        gen = _make_generator(tmp_path=tmp_path)
        path = gen.export_article("python-basics", output_dir=str(tmp_path))

        assert path != ""
        p = Path(path)
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert "# Python Basics" in content
        assert "Guido van Rossum" in content

    def test_nonexistent_slug_returns_empty(self, tmp_path: Path):
        gen = _make_generator(tmp_path=tmp_path)
        path = gen.export_article("does-not-exist", output_dir=str(tmp_path))
        assert path == ""

    def test_default_output_dir(self, tmp_path: Path):
        gen = _make_generator(tmp_path=tmp_path)
        path = gen.export_article("python-basics")

        assert path != ""
        assert "exports" in path
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Tests: slugify helper
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        assert _slugify("Python 3.12 Features!") == "python-312-features"

    def test_empty(self):
        assert _slugify("") == "untitled"

    def test_truncation(self):
        long = "a" * 200
        assert len(_slugify(long)) <= 80


# ---------------------------------------------------------------------------
# Tests: CLI command registration
# ---------------------------------------------------------------------------


class TestCLICommands:
    """Verify that report/export commands are registered in the REPL."""

    def test_report_command_registered(self):
        from saido_agent.cli.repl import COMMANDS
        assert "report" in COMMANDS

    def test_export_command_registered(self):
        from saido_agent.cli.repl import COMMANDS
        assert "export" in COMMANDS
