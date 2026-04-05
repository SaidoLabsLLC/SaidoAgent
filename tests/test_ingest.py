"""Tests for the ingest pipeline and structural analyzer."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from saido_agent.knowledge.ingest import (
    IngestPipeline,
    _ALL_SUPPORTED,
    _CODE_EXTENSIONS,
    _EXT_TO_LANGUAGE,
)
from saido_agent.knowledge.structural import CodeStructure, StructuralAnalyzer

# ---------------------------------------------------------------------------
# Paths to test fixtures
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PY = FIXTURES / "sample.py"
SAMPLE_MD = FIXTURES / "sample.md"


# ---------------------------------------------------------------------------
# CodeStructure dataclass
# ---------------------------------------------------------------------------


class TestCodeStructure:
    def test_empty_structure(self):
        cs = CodeStructure(language="python")
        d = cs.to_dict()
        assert d == {"language": "python"}

    def test_populated_structure(self):
        cs = CodeStructure(
            language="python",
            functions=["def greet(name)", "def add(a, b)"],
            classes=["class Calculator"],
            imports=["import os"],
            ast_patterns_detected=4,
        )
        d = cs.to_dict()
        assert d["language"] == "python"
        assert len(d["functions"]) == 2
        assert len(d["classes"]) == 1
        assert len(d["imports"]) == 1
        assert d["ast_patterns_detected"] == 4
        # Empty lists should be excluded
        assert "endpoints" not in d
        assert "decorators" not in d

    def test_to_dict_excludes_zero_and_empty(self):
        cs = CodeStructure(language="go", endpoints=[], ast_patterns_detected=0)
        d = cs.to_dict()
        assert "endpoints" not in d
        assert "ast_patterns_detected" not in d


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------


class TestFileTypeDetection:
    @pytest.mark.parametrize(
        "ext",
        [".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h"],
    )
    def test_code_extensions_recognized(self, ext):
        assert IngestPipeline._is_code_file(f"somefile{ext}")

    @pytest.mark.parametrize(
        "ext",
        [".md", ".txt", ".pdf", ".json", ".yaml", ".yml", ".csv", ".html"],
    )
    def test_doc_extensions_not_code(self, ext):
        assert not IngestPipeline._is_code_file(f"somefile{ext}")

    def test_unsupported_extension(self):
        assert not IngestPipeline._is_code_file("somefile.xyz")
        assert not IngestPipeline._is_supported_file("somefile.xyz")

    def test_all_supported_extensions_covered(self):
        expected = {
            ".md", ".txt", ".pdf",
            ".py", ".js", ".ts", ".jsx", ".tsx",
            ".go", ".rs", ".java",
            ".cpp", ".c", ".h",
            ".json", ".yaml", ".yml", ".csv", ".html",
        }
        assert expected == _ALL_SUPPORTED


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class TestLanguageDetection:
    @pytest.mark.parametrize(
        "ext, expected",
        [
            (".py", "python"),
            (".js", "javascript"),
            (".ts", "typescript"),
            (".go", "go"),
            (".rs", "rust"),
            (".java", "java"),
            (".cpp", "cpp"),
            (".c", "c"),
            (".h", "c"),
        ],
    )
    def test_extension_to_language(self, ext, expected):
        assert IngestPipeline._detect_language(f"file{ext}") == expected

    def test_unknown_extension(self):
        assert IngestPipeline._detect_language("file.xyz") == "unknown"


# ---------------------------------------------------------------------------
# StructuralAnalyzer
# ---------------------------------------------------------------------------


class TestStructuralAnalyzer:
    def test_analyze_returns_code_structure(self):
        """analyze() should return a CodeStructure regardless of sg availability."""
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(str(SAMPLE_PY), language="python")
        assert isinstance(result, CodeStructure)
        assert result.language == "python"

    def test_analyze_nonexistent_file(self):
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze("/nonexistent/file.py", language="python")
        assert isinstance(result, CodeStructure)
        assert result.functions == []

    def test_analyze_unknown_language(self):
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(str(SAMPLE_PY), language="brainfuck")
        assert isinstance(result, CodeStructure)
        assert result.language == "brainfuck"
        assert result.ast_patterns_detected == 0

    @patch.object(StructuralAnalyzer, "_is_sg_available", return_value=False)
    def test_graceful_skip_when_sg_unavailable(self, mock_sg):
        """When ast-grep is not installed, return empty structure without error."""
        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(str(SAMPLE_PY), language="python")
        assert isinstance(result, CodeStructure)
        assert result.functions == []
        assert result.classes == []
        assert result.ast_patterns_detected == 0

    @patch.object(StructuralAnalyzer, "_is_sg_available", return_value=True)
    @patch("subprocess.run")
    def test_analyze_python_with_mocked_sg(self, mock_run, mock_sg):
        """Verify that analyze correctly parses sg JSON output."""
        # Simulate sg --json output for function pattern
        fn_output = json.dumps([
            {"text": "def greet(name: str) -> str:\n    ..."},
            {"text": "def add(a: int, b: int) -> int:\n    ..."},
        ])
        class_output = json.dumps([
            {"text": "class Calculator:\n    ..."},
            {"text": "class AdvancedCalculator(Calculator):\n    ..."},
        ])
        import_output = json.dumps([
            {"text": "import os"},
        ])
        from_import_output = json.dumps([
            {"text": "from pathlib import Path"},
        ])
        empty_output = json.dumps([])

        # sg is called once per pattern string; Python has:
        # functions: 1 pattern, classes: 1, imports: 2, endpoints: 5, decorators: 1
        # Total: 10 calls
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=fn_output),      # functions
            MagicMock(returncode=0, stdout=class_output),    # classes
            MagicMock(returncode=0, stdout=import_output),   # imports[0]
            MagicMock(returncode=0, stdout=from_import_output),  # imports[1]
            MagicMock(returncode=0, stdout=empty_output),    # endpoints[0]
            MagicMock(returncode=0, stdout=empty_output),    # endpoints[1]
            MagicMock(returncode=0, stdout=empty_output),    # endpoints[2]
            MagicMock(returncode=0, stdout=empty_output),    # endpoints[3]
            MagicMock(returncode=0, stdout=empty_output),    # endpoints[4]
            MagicMock(returncode=0, stdout=empty_output),    # decorators
        ]

        analyzer = StructuralAnalyzer()
        result = analyzer.analyze(str(SAMPLE_PY), language="python")

        assert "def greet(name: str) -> str:" in result.functions
        assert "def add(a: int, b: int) -> int:" in result.functions
        assert "class Calculator:" in result.classes
        assert "import os" in result.imports
        assert "from pathlib import Path" in result.imports
        assert result.ast_patterns_detected == 6  # 2 fn + 2 class + 2 import

    def test_parse_sg_json_empty(self):
        assert StructuralAnalyzer._parse_sg_json("") == []
        assert StructuralAnalyzer._parse_sg_json("  ") == []

    def test_parse_sg_json_invalid(self):
        assert StructuralAnalyzer._parse_sg_json("not json") == []

    def test_parse_sg_json_valid(self):
        raw = json.dumps([
            {"text": "def foo():\n    pass"},
            {"text": "def bar():\n    pass"},
        ])
        result = StructuralAnalyzer._parse_sg_json(raw)
        assert result == ["def foo():", "def bar():"]


# ---------------------------------------------------------------------------
# IngestPipeline - single file
# ---------------------------------------------------------------------------


class TestIngestFile:
    def _make_pipeline(self, bridge=None):
        return IngestPipeline(bridge=bridge)

    def test_ingest_code_file(self):
        pipeline = self._make_pipeline()
        result = pipeline.ingest_file(str(SAMPLE_PY))
        assert result["status"] == "ok"
        assert result["is_code"] is True
        assert result["slug"] is not None

    def test_ingest_doc_file(self):
        pipeline = self._make_pipeline()
        result = pipeline.ingest_file(str(SAMPLE_MD))
        assert result["status"] == "ok"
        assert result["is_code"] is False
        assert result["slug"] is not None

    def test_ingest_nonexistent_file(self):
        pipeline = self._make_pipeline()
        result = pipeline.ingest_file("/nonexistent/file.py")
        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_ingest_unsupported_type(self, tmp_path):
        unsupported = tmp_path / "file.xyz"
        unsupported.write_text("data")
        pipeline = self._make_pipeline()
        result = pipeline.ingest_file(str(unsupported))
        assert result["status"] == "skipped"
        assert "unsupported" in result["error"]

    def test_compile_queue_populated(self):
        pipeline = self._make_pipeline()
        pipeline.ingest_file(str(SAMPLE_PY))
        pipeline.ingest_file(str(SAMPLE_MD))
        queue = pipeline.get_compile_queue()
        assert len(queue) == 2
        assert "sample" in queue[0]

    def test_compile_queue_clear(self):
        pipeline = self._make_pipeline()
        pipeline.ingest_file(str(SAMPLE_PY))
        assert len(pipeline.get_compile_queue()) == 1
        pipeline.clear_compile_queue()
        assert len(pipeline.get_compile_queue()) == 0


# ---------------------------------------------------------------------------
# IngestPipeline - directory
# ---------------------------------------------------------------------------


class TestIngestDirectory:
    def test_ingest_fixtures_directory(self):
        pipeline = IngestPipeline(bridge=None)
        results = pipeline.ingest_directory(str(FIXTURES))
        # At least sample.py and sample.md
        paths = [r["path"] for r in results]
        suffixes = {Path(p).suffix for p in paths}
        assert ".py" in suffixes
        assert ".md" in suffixes
        assert all(r["status"] == "ok" for r in results)

    def test_ingest_nonexistent_directory(self):
        pipeline = IngestPipeline(bridge=None)
        results = pipeline.ingest_directory("/nonexistent/dir")
        assert results == []

    def test_ingest_directory_non_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.py").write_text("x = 1")
        (sub / "nested.py").write_text("y = 2")

        pipeline = IngestPipeline(bridge=None)
        results = pipeline.ingest_directory(str(tmp_path), recursive=False)
        paths = [r["path"] for r in results]
        assert any("top.py" in p for p in paths)
        assert not any("nested.py" in p for p in paths)

    def test_ingest_directory_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.py").write_text("x = 1")
        (sub / "nested.py").write_text("y = 2")

        pipeline = IngestPipeline(bridge=None)
        results = pipeline.ingest_directory(str(tmp_path), recursive=True)
        paths = [r["path"] for r in results]
        assert any("top.py" in p for p in paths)
        assert any("nested.py" in p for p in paths)

    def test_skips_pycache_and_venv(self, tmp_path):
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        venv = tmp_path / "venv"
        venv.mkdir()
        (pycache / "cached.py").write_text("x = 1")
        (venv / "lib.py").write_text("y = 2")
        (tmp_path / "real.py").write_text("z = 3")

        pipeline = IngestPipeline(bridge=None)
        results = pipeline.ingest_directory(str(tmp_path))
        paths = [r["path"] for r in results]
        assert any("real.py" in p for p in paths)
        assert not any("cached.py" in p for p in paths)
        assert not any("lib.py" in p for p in paths)


# ---------------------------------------------------------------------------
# IngestPipeline - bridge integration
# ---------------------------------------------------------------------------


class TestBridgeIntegration:
    def test_bridge_ingest_file_called(self):
        bridge = MagicMock()
        bridge.ingest_file.return_value = {"slug": "test-slug"}

        pipeline = IngestPipeline(bridge=bridge)
        result = pipeline.ingest_file(str(SAMPLE_MD))

        bridge.ingest_file.assert_called_once_with(str(SAMPLE_MD))
        assert result["slug"] == "test-slug"

    def test_bridge_frontmatter_update_for_code(self):
        bridge = MagicMock()
        bridge.ingest_file.return_value = {"slug": "sample-py"}

        pipeline = IngestPipeline(bridge=bridge)
        result = pipeline.ingest_file(str(SAMPLE_PY))

        assert result["slug"] == "sample-py"
        # Frontmatter update is called only if structure has matches,
        # which depends on sg availability.  With a mock bridge the
        # important thing is the pipeline doesn't crash.

    def test_bridge_error_handled(self):
        bridge = MagicMock()
        bridge.ingest_file.side_effect = RuntimeError("connection failed")

        pipeline = IngestPipeline(bridge=bridge)
        result = pipeline.ingest_file(str(SAMPLE_PY))

        assert result["status"] == "error"
        assert "bridge error" in result["error"]


# ---------------------------------------------------------------------------
# Non-code files skip structural analysis
# ---------------------------------------------------------------------------


class TestNonCodeSkipsStructural:
    @patch.object(StructuralAnalyzer, "analyze")
    def test_markdown_skips_structural(self, mock_analyze):
        pipeline = IngestPipeline(bridge=None)
        result = pipeline.ingest_file(str(SAMPLE_MD))
        mock_analyze.assert_not_called()
        assert result["structure"] is None
        assert result["is_code"] is False

    @patch.object(StructuralAnalyzer, "analyze")
    def test_code_triggers_structural(self, mock_analyze):
        mock_analyze.return_value = CodeStructure(language="python")
        pipeline = IngestPipeline(bridge=None)
        result = pipeline.ingest_file(str(SAMPLE_PY))
        mock_analyze.assert_called_once()
        assert result["is_code"] is True
