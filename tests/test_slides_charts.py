"""Tests for SlideGenerator, ChartGenerator, and chart sandbox."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from saido_agent.knowledge.outputs import (
    ChartGenerator,
    ChartResult,
    SlideGenerator,
    SlideResult,
    _validate_chart_code,
    _slugify,
)


# ---------------------------------------------------------------------------
# Fake SmartRAG types (mirror test_outputs.py)
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
# Shared fixtures
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


# ---------------------------------------------------------------------------
# Sample Marp output
# ---------------------------------------------------------------------------

SAMPLE_MARP = """\
---
marp: true
theme: default
---

# Python Programming Overview

---

## What is Python?
- High-level programming language
- Created by Guido van Rossum
- Supports OOP and functional programming

---

## Async Programming
- asyncio module for concurrency
- async/await syntax
- Single-threaded concurrent code

---

## Summary
- Versatile language
- Great async support
"""

# ---------------------------------------------------------------------------
# Sample matplotlib code from LLM
# ---------------------------------------------------------------------------

SAMPLE_CHART_CODE_RAW = """\
Here is the chart code:

```python
import matplotlib.pyplot as plt
import numpy as np

data = [30, 45, 25]
labels = ['OOP', 'Functional', 'Async']

plt.figure(figsize=(8, 6))
plt.bar(labels, data, color=['#2196F3', '#4CAF50', '#FF9800'])
plt.title('Python Paradigm Usage')
plt.ylabel('Percentage')
plt.savefig(__OUTPUT_PATH__)
plt.close()
```

This creates a bar chart showing paradigm usage.
"""

SAMPLE_CHART_CODE_EXTRACTED = """\
import matplotlib.pyplot as plt
import numpy as np

data = [30, 45, 25]
labels = ['OOP', 'Functional', 'Async']

plt.figure(figsize=(8, 6))
plt.bar(labels, data, color=['#2196F3', '#4CAF50', '#FF9800'])
plt.title('Python Paradigm Usage')
plt.ylabel('Percentage')
plt.savefig(__OUTPUT_PATH__)
plt.close()"""


# =========================================================================
# Slide Tests
# =========================================================================


class TestSlideGenerator:
    """Tests for SlideGenerator.generate_slides."""

    def _make_generator(
        self,
        bridge: MagicMock | None = None,
        llm_response: str | None = None,
        tmp_path: Path | None = None,
    ) -> SlideGenerator:
        if bridge is None:
            bridge = _make_bridge()
        if tmp_path is not None:
            bridge._config.knowledge_root = str(tmp_path / "knowledge")
        gen = SlideGenerator(bridge, model_router=MagicMock())
        if llm_response is not None:
            gen._call_llm = MagicMock(return_value=llm_response)
        else:
            gen._call_llm = MagicMock(return_value=None)
        return gen

    # -- Happy path -------------------------------------------------------

    def test_generates_valid_marp_markdown(self, tmp_path: Path):
        gen = self._make_generator(llm_response=SAMPLE_MARP, tmp_path=tmp_path)
        result = gen.generate_slides("Python programming")

        assert result.status == "generated"
        assert result.error is None

        # Read back and verify Marp frontmatter
        content = Path(result.path).read_text(encoding="utf-8")
        assert "marp: true" in content
        assert "---" in content

    def test_slides_saved_to_correct_path(self, tmp_path: Path):
        gen = self._make_generator(llm_response=SAMPLE_MARP, tmp_path=tmp_path)
        result = gen.generate_slides("Python programming")

        assert result.status == "generated"
        path = Path(result.path)
        assert path.exists()
        assert path.suffix == ".md"
        assert "slides" in str(path)
        assert "python-programming" in path.name

    def test_slide_result_populated_correctly(self, tmp_path: Path):
        gen = self._make_generator(llm_response=SAMPLE_MARP, tmp_path=tmp_path)
        result = gen.generate_slides("Python programming")

        assert isinstance(result, SlideResult)
        assert result.title == "Python programming"
        assert result.slide_count > 1  # Multiple slides
        assert result.status == "generated"
        assert result.error is None
        assert result.path != ""

    def test_slide_count_matches_separators(self, tmp_path: Path):
        gen = self._make_generator(llm_response=SAMPLE_MARP, tmp_path=tmp_path)
        result = gen.generate_slides("Python programming")

        # SAMPLE_MARP has 4 "\n---" separators, so slide_count = 4 + 1 = 5
        assert result.slide_count == 5

    # -- Error paths ------------------------------------------------------

    def test_empty_store_returns_failed(self, tmp_path: Path):
        bridge = _make_bridge(doc_count=0)
        gen = self._make_generator(bridge=bridge, tmp_path=tmp_path)
        result = gen.generate_slides("Python")

        assert result.status == "failed"
        assert "empty" in result.error.lower()
        assert result.slide_count == 0

    def test_no_matching_articles_returns_failed(self, tmp_path: Path):
        bridge = _make_bridge(search_results=[])
        gen = self._make_generator(bridge=bridge, tmp_path=tmp_path)
        result = gen.generate_slides("quantum physics")

        assert result.status == "failed"
        assert "No articles found" in result.error

    def test_unreadable_articles_returns_failed(self, tmp_path: Path):
        bridge = _make_bridge()
        bridge.read_article.side_effect = None
        bridge.read_article.return_value = None
        gen = self._make_generator(
            bridge=bridge, llm_response=SAMPLE_MARP, tmp_path=tmp_path,
        )
        result = gen.generate_slides("Python")

        assert result.status == "failed"
        assert "Could not read" in result.error

    def test_llm_failure_returns_failed(self, tmp_path: Path):
        gen = self._make_generator(llm_response=None, tmp_path=tmp_path)
        result = gen.generate_slides("Python")

        assert result.status == "failed"
        assert "LLM call failed" in result.error

    # -- Prompt construction ----------------------------------------------

    def test_build_slide_prompt_includes_topic_and_articles(self):
        articles = [
            {"slug": "a1", "title": "Article 1", "body": "Body 1"},
            {"slug": "a2", "title": "Article 2", "body": "Body 2"},
        ]
        prompt = SlideGenerator._build_slide_prompt("test topic", articles)

        assert "test topic" in prompt
        assert "Article 1" in prompt
        assert "Article 2" in prompt
        assert "marp: true" in prompt


# =========================================================================
# Chart Sandbox Validation Tests
# =========================================================================


class TestChartCodeValidation:
    """Tests for _validate_chart_code safety checks."""

    def test_safe_code_passes(self):
        code = (
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "plt.bar([1,2], [3,4])\n"
            "plt.savefig(__OUTPUT_PATH__)\n"
        )
        assert _validate_chart_code(code) is None

    def test_allowed_imports_pass(self):
        code = (
            "import matplotlib\n"
            "import numpy\n"
            "import json\n"
            "import math\n"
        )
        assert _validate_chart_code(code) is None

    def test_blocked_import_os(self):
        code = "import os\nos.system('rm -rf /')\n"
        err = _validate_chart_code(code)
        assert err is not None
        assert "os" in err

    def test_blocked_import_subprocess(self):
        code = "import subprocess\n"
        err = _validate_chart_code(code)
        assert err is not None
        assert "subprocess" in err

    def test_blocked_import_socket(self):
        code = "import socket\n"
        err = _validate_chart_code(code)
        assert err is not None
        assert "socket" in err

    def test_blocked_import_shutil(self):
        code = "import shutil\n"
        err = _validate_chart_code(code)
        assert err is not None
        assert "shutil" in err

    def test_blocked_from_import(self):
        code = "from os.path import join\n"
        err = _validate_chart_code(code)
        assert err is not None
        assert "os" in err

    def test_blocked_exec_call(self):
        code = "exec('print(1)')\n"
        err = _validate_chart_code(code)
        assert err is not None
        assert "exec" in err.lower() or "Blocked" in err

    def test_blocked_eval_call(self):
        code = "eval('1+1')\n"
        err = _validate_chart_code(code)
        assert err is not None

    def test_blocked_open_call(self):
        code = "f = open('/etc/passwd')\n"
        err = _validate_chart_code(code)
        assert err is not None
        assert "open" in err.lower()

    def test_blocked_dunder_import(self):
        code = "__import__('os')\n"
        err = _validate_chart_code(code)
        assert err is not None

    def test_disallowed_unknown_module(self):
        code = "import requests\n"
        err = _validate_chart_code(code)
        assert err is not None
        assert "requests" in err

    def test_comments_and_blanks_ignored(self):
        code = (
            "# import os\n"
            "\n"
            "import matplotlib.pyplot as plt\n"
        )
        assert _validate_chart_code(code) is None


# =========================================================================
# Chart Generator Tests
# =========================================================================


class TestChartGenerator:
    """Tests for ChartGenerator.generate_chart."""

    def _make_generator(
        self,
        bridge: MagicMock | None = None,
        llm_response: str | None = None,
        tmp_path: Path | None = None,
    ) -> ChartGenerator:
        if bridge is None:
            bridge = _make_bridge()
        if tmp_path is not None:
            bridge._config.knowledge_root = str(tmp_path / "knowledge")
        gen = ChartGenerator(bridge, model_router=MagicMock())
        if llm_response is not None:
            gen._call_llm = MagicMock(return_value=llm_response)
        else:
            gen._call_llm = MagicMock(return_value=None)
        return gen

    # -- Code extraction --------------------------------------------------

    def test_extract_code_from_fenced_block(self):
        extracted = ChartGenerator._extract_code(SAMPLE_CHART_CODE_RAW)
        assert "import matplotlib" in extracted
        assert "plt.savefig" in extracted
        assert "```" not in extracted

    def test_extract_code_raw_fallback(self):
        raw = "import matplotlib.pyplot as plt\nplt.plot([1,2])\n"
        extracted = ChartGenerator._extract_code(raw)
        assert "import matplotlib" in extracted

    # -- Chart type detection ---------------------------------------------

    def test_detect_bar_chart(self):
        assert ChartGenerator._detect_chart_type("plt.bar(x, y)") == "bar"

    def test_detect_pie_chart(self):
        assert ChartGenerator._detect_chart_type("plt.pie(data)") == "pie"

    def test_detect_scatter_chart(self):
        assert ChartGenerator._detect_chart_type("plt.scatter(x, y)") == "scatter"

    def test_detect_line_chart(self):
        assert ChartGenerator._detect_chart_type("plt.plot(x, y)") == "line"

    def test_detect_histogram(self):
        assert ChartGenerator._detect_chart_type("plt.hist(data)") == "histogram"

    def test_detect_unknown(self):
        assert ChartGenerator._detect_chart_type("something()") == "unknown"

    # -- Happy path (mock subprocess to avoid needing matplotlib) ----------

    def test_chart_generation_produces_result(self, tmp_path: Path, monkeypatch):
        gen = self._make_generator(
            llm_response=SAMPLE_CHART_CODE_RAW, tmp_path=tmp_path,
        )

        # Mock _execute_chart_code to simulate success and create a dummy PNG
        def fake_execute(code: str, output_path: str, timeout: int = 30):
            Path(output_path).write_bytes(b"\x89PNG fake image data")
            return None

        monkeypatch.setattr(
            "saido_agent.knowledge.outputs._execute_chart_code", fake_execute,
        )

        result = gen.generate_chart("Python paradigm usage chart")

        assert result.status == "generated"
        assert result.error is None
        assert result.chart_type == "bar"
        assert Path(result.path).exists()
        assert result.path.endswith(".png")

    def test_chart_result_populated_correctly(self, tmp_path: Path, monkeypatch):
        gen = self._make_generator(
            llm_response=SAMPLE_CHART_CODE_RAW, tmp_path=tmp_path,
        )

        def fake_execute(code: str, output_path: str, timeout: int = 30):
            Path(output_path).write_bytes(b"\x89PNG fake")
            return None

        monkeypatch.setattr(
            "saido_agent.knowledge.outputs._execute_chart_code", fake_execute,
        )

        result = gen.generate_chart("Python paradigm usage chart")

        assert isinstance(result, ChartResult)
        assert result.title == "Python paradigm usage chart"
        assert result.chart_type == "bar"
        assert result.status == "generated"

    def test_chart_saved_to_correct_path(self, tmp_path: Path, monkeypatch):
        gen = self._make_generator(
            llm_response=SAMPLE_CHART_CODE_RAW, tmp_path=tmp_path,
        )

        def fake_execute(code: str, output_path: str, timeout: int = 30):
            Path(output_path).write_bytes(b"\x89PNG fake")
            return None

        monkeypatch.setattr(
            "saido_agent.knowledge.outputs._execute_chart_code", fake_execute,
        )

        result = gen.generate_chart("paradigm chart")
        path = Path(result.path)
        assert "charts" in str(path)
        assert path.suffix == ".png"

    # -- Unsafe code blocked ----------------------------------------------

    def test_unsafe_code_blocked(self, tmp_path: Path):
        unsafe_response = "```python\nimport os\nos.system('rm -rf /')\n```"
        gen = self._make_generator(
            llm_response=unsafe_response, tmp_path=tmp_path,
        )
        result = gen.generate_chart("dangerous chart")

        assert result.status == "failed"
        assert "Unsafe" in result.error or "Blocked" in result.error

    # -- Error paths ------------------------------------------------------

    def test_empty_store_returns_failed(self, tmp_path: Path):
        bridge = _make_bridge(doc_count=0)
        gen = self._make_generator(bridge=bridge, tmp_path=tmp_path)
        result = gen.generate_chart("some chart")

        assert result.status == "failed"
        assert "empty" in result.error.lower()

    def test_no_matching_articles_returns_failed(self, tmp_path: Path):
        bridge = _make_bridge(search_results=[])
        gen = self._make_generator(bridge=bridge, tmp_path=tmp_path)
        result = gen.generate_chart("quantum chart")

        assert result.status == "failed"
        assert "No articles found" in result.error

    def test_unreadable_articles_returns_failed(self, tmp_path: Path):
        bridge = _make_bridge()
        bridge.read_article.side_effect = None
        bridge.read_article.return_value = None
        gen = self._make_generator(
            bridge=bridge, llm_response=SAMPLE_CHART_CODE_RAW, tmp_path=tmp_path,
        )
        result = gen.generate_chart("some chart")

        assert result.status == "failed"
        assert "Could not read" in result.error

    def test_llm_failure_returns_failed(self, tmp_path: Path):
        gen = self._make_generator(llm_response=None, tmp_path=tmp_path)
        result = gen.generate_chart("some chart")

        assert result.status == "failed"
        assert "LLM call failed" in result.error

    def test_execution_failure_returns_failed(self, tmp_path: Path, monkeypatch):
        gen = self._make_generator(
            llm_response=SAMPLE_CHART_CODE_RAW, tmp_path=tmp_path,
        )

        def fake_execute(code: str, output_path: str, timeout: int = 30):
            return "Chart code execution failed:\nSyntaxError: invalid syntax"

        monkeypatch.setattr(
            "saido_agent.knowledge.outputs._execute_chart_code", fake_execute,
        )

        result = gen.generate_chart("broken chart")
        assert result.status == "failed"
        assert "execution failed" in result.error.lower()

    def test_no_output_file_returns_failed(self, tmp_path: Path, monkeypatch):
        gen = self._make_generator(
            llm_response=SAMPLE_CHART_CODE_RAW, tmp_path=tmp_path,
        )

        # Execute succeeds but does not create a file
        def fake_execute(code: str, output_path: str, timeout: int = 30):
            return None

        monkeypatch.setattr(
            "saido_agent.knowledge.outputs._execute_chart_code", fake_execute,
        )

        result = gen.generate_chart("missing output chart")
        assert result.status == "failed"
        assert "no output file" in result.error.lower()

    # -- Prompt construction ----------------------------------------------

    def test_build_chart_prompt_includes_description_and_articles(self):
        articles = [
            {"slug": "a1", "title": "Article 1", "body": "Body 1"},
        ]
        prompt = ChartGenerator._build_chart_prompt("test chart", articles)

        assert "test chart" in prompt
        assert "Article 1" in prompt
        assert "__OUTPUT_PATH__" in prompt
        assert "matplotlib" in prompt
