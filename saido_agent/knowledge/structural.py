"""Structural code analysis via ast-grep CLI.

Extracts functions, classes, imports, endpoints, and decorators from source
files using AST pattern matching.  Falls back gracefully when the ``sg`` binary
is not available.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CodeStructure:
    """Aggregated structural information extracted from a single source file."""

    language: str
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    endpoints: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    ast_patterns_detected: int = 0

    def to_dict(self) -> dict:
        """Return a dict with only non-empty / non-zero fields."""
        return {k: v for k, v in self.__dict__.items() if v}


# ---------------------------------------------------------------------------
# Per-language ast-grep patterns
# ---------------------------------------------------------------------------

# Each value is either a single pattern string or a list of pattern strings.
# Meta-variables follow ast-grep conventions: $NAME for single node,
# $$$NAMES for variadic.

PATTERNS: dict[str, dict[str, Union[str, list[str]]]] = {
    "python": {
        "functions": "def $FUNC($$$ARGS)",
        "classes": "class $NAME($$$BASES):",
        "imports": ["import $MODULE", "from $MODULE import $$$NAMES"],
        "endpoints": [
            "@app.route($$$ARGS)",
            "@app.get($$$ARGS)",
            "@app.post($$$ARGS)",
            "@app.put($$$ARGS)",
            "@app.delete($$$ARGS)",
        ],
        "decorators": "@$DECORATOR($$$ARGS)",
    },
    "javascript": {
        "functions": [
            "function $FUNC($$$ARGS) { $$$BODY }",
            "const $FUNC = ($$$ARGS) => { $$$BODY }",
            "const $FUNC = ($$$ARGS) => $EXPR",
        ],
        "classes": "class $NAME { $$$BODY }",
        "imports": [
            "import $$$NAMES from '$MODULE'",
            "const $NAME = require('$MODULE')",
        ],
        "endpoints": [
            "app.get($$$ARGS)",
            "app.post($$$ARGS)",
            "app.put($$$ARGS)",
            "app.delete($$$ARGS)",
            "router.get($$$ARGS)",
            "router.post($$$ARGS)",
        ],
    },
    "typescript": {
        "functions": [
            "function $FUNC($$$ARGS) { $$$BODY }",
            "const $FUNC = ($$$ARGS) => { $$$BODY }",
            "const $FUNC = ($$$ARGS) => $EXPR",
        ],
        "classes": "class $NAME { $$$BODY }",
        "imports": [
            "import $$$NAMES from '$MODULE'",
            "import { $$$NAMES } from '$MODULE'",
        ],
        "endpoints": [
            "app.get($$$ARGS)",
            "app.post($$$ARGS)",
            "router.get($$$ARGS)",
            "router.post($$$ARGS)",
        ],
        "decorators": "@$DECORATOR($$$ARGS)",
    },
    "go": {
        "functions": "func $FUNC($$$ARGS) $$$RET { $$$BODY }",
        "imports": 'import "$MODULE"',
        "endpoints": [
            'http.HandleFunc($$$ARGS)',
            'r.HandleFunc($$$ARGS)',
        ],
    },
    "rust": {
        "functions": "fn $FUNC($$$ARGS) $$$RET { $$$BODY }",
        "imports": "use $$$PATH;",
        "decorators": "#[$ATTR($$$ARGS)]",
    },
    "java": {
        "functions": "$$$MODS $RET $FUNC($$$ARGS) { $$$BODY }",
        "classes": "$$$MODS class $NAME { $$$BODY }",
        "imports": "import $$$PATH;",
        "endpoints": [
            "@GetMapping($$$ARGS)",
            "@PostMapping($$$ARGS)",
            "@RequestMapping($$$ARGS)",
        ],
        "decorators": "@$ANNOTATION($$$ARGS)",
    },
}

# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

_SG_TIMEOUT = 15  # seconds per invocation


class StructuralAnalyzer:
    """Extracts code structure via the ``sg`` (ast-grep) CLI.

    If the CLI is not installed or a pattern fails, the analyzer degrades
    gracefully --- it never raises on sg errors.
    """

    PATTERNS = PATTERNS  # expose for external use / testing

    def __init__(self) -> None:
        self._sg_available: bool | None = None  # lazy probe

    # -- public API ---------------------------------------------------------

    def analyze(self, file_path: str, language: str | None = None) -> CodeStructure:
        """Run ast-grep patterns against *file_path* and return structure."""
        path = Path(file_path)
        if not path.is_file():
            log.warning("structural: file not found: %s", file_path)
            return CodeStructure(language=language or "unknown")

        lang = language or _detect_language_from_ext(path.suffix)
        structure = CodeStructure(language=lang)

        if not self._is_sg_available():
            log.info("structural: sg CLI not available, returning empty structure")
            return structure

        lang_patterns = self.PATTERNS.get(lang)
        if not lang_patterns:
            log.debug("structural: no patterns defined for language '%s'", lang)
            return structure

        total_matches = 0
        for category, patterns in lang_patterns.items():
            if isinstance(patterns, str):
                patterns = [patterns]
            names: list[str] = []
            for pat in patterns:
                matches = self._run_sg(pat, lang, str(path))
                names.extend(matches)
                total_matches += len(matches)
            # Deduplicate while preserving order
            seen: set[str] = set()
            deduped: list[str] = []
            for n in names:
                if n not in seen:
                    seen.add(n)
                    deduped.append(n)
            if hasattr(structure, category):
                setattr(structure, category, deduped)

        structure.ast_patterns_detected = total_matches
        return structure

    # -- internals ----------------------------------------------------------

    def _is_sg_available(self) -> bool:
        """Probe once whether the ``sg`` binary is on PATH."""
        if self._sg_available is None:
            try:
                subprocess.run(
                    ["sg", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self._sg_available = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._sg_available = False
        return self._sg_available

    def _run_sg(self, pattern: str, language: str, path: str) -> list[str]:
        """Execute a single ``sg`` search and return matched text fragments."""
        cmd = ["sg", "--pattern", pattern, "--lang", language, "--json", path]
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_SG_TIMEOUT,
            )
            if r.returncode != 0 and not r.stdout.strip():
                return []
            return self._parse_sg_json(r.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            log.debug("structural: sg invocation failed: %s", exc)
            return []
        except Exception as exc:  # noqa: BLE001
            log.warning("structural: unexpected sg error: %s", exc)
            return []

    @staticmethod
    def _parse_sg_json(raw: str) -> list[str]:
        """Extract matched text from ``sg --json`` output.

        The JSON output is a list of match objects.  Each has a ``text``
        field with the matched source fragment.  We return the first line
        of each match (often the signature) trimmed of whitespace.
        """
        if not raw or not raw.strip():
            return []
        try:
            matches = json.loads(raw)
        except json.JSONDecodeError:
            return []
        results: list[str] = []
        for m in matches:
            text = m.get("text", "").strip()
            if text:
                # Use first line as the representative identifier
                first_line = text.split("\n")[0].strip()
                if first_line:
                    results.append(first_line)
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
}


def _detect_language_from_ext(ext: str) -> str:
    """Map a file extension to ast-grep language identifier."""
    return _EXT_TO_LANG.get(ext.lower(), "unknown")
