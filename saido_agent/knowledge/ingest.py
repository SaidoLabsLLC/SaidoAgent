"""Knowledge ingestion pipeline.

Orchestrates: SmartRAG storage (via bridge) + ast-grep structural analysis
+ compile queue population for downstream LLM enrichment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from saido_agent.knowledge.structural import CodeStructure, StructuralAnalyzer

if TYPE_CHECKING:
    pass  # KnowledgeBridge type will be available when bridge is built

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported file types
# ---------------------------------------------------------------------------

_CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".go", ".rs", ".java",
    ".cpp", ".c", ".h",
})

_DOC_EXTENSIONS: frozenset[str] = frozenset({
    ".md", ".txt", ".pdf",
    ".json", ".yaml", ".yml",
    ".csv", ".html",
})

_ALL_SUPPORTED: frozenset[str] = _CODE_EXTENSIONS | _DOC_EXTENSIONS

# Extension -> ast-grep language identifier
_EXT_TO_LANGUAGE: dict[str, str] = {
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

# ---------------------------------------------------------------------------
# Directories to skip during recursive traversal
# ---------------------------------------------------------------------------

_SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".hg", ".svn",
    "node_modules", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache",
    "dist", "build", ".eggs", "*.egg-info",
})


# ---------------------------------------------------------------------------
# IngestPipeline
# ---------------------------------------------------------------------------


class IngestPipeline:
    """Orchestrates document ingestion.

    Flow per file:
    1. Detect file type (code vs doc).
    2. Delegate to ``KnowledgeBridge`` for SmartRAG storage (article creation,
       section splitting, deduplication).
    3. If the file is source code, run ``StructuralAnalyzer`` to extract
       functions / classes / endpoints, then update article frontmatter via
       the bridge.
    4. Add the resulting slug to the compile queue for downstream LLM
       enrichment.
    """

    def __init__(self, bridge, model_router=None):
        self._bridge = bridge  # KnowledgeBridge instance
        self._router = model_router
        self._structural = StructuralAnalyzer()
        self._compile_queue: list[str] = []

    # -- public API ---------------------------------------------------------

    def ingest_file(self, path: str) -> dict:
        """Ingest a single file.

        Returns a result dict::

            {
                "path": str,
                "slug": str | None,
                "status": "ok" | "skipped" | "error",
                "is_code": bool,
                "structure": dict | None,
                "error": str | None,
            }
        """
        result: dict = {
            "path": path,
            "slug": None,
            "status": "skipped",
            "is_code": False,
            "structure": None,
            "error": None,
        }

        p = Path(path)
        if not p.is_file():
            result["status"] = "error"
            result["error"] = "file not found"
            return result

        if not self._is_code_file(path) and not self._is_supported_file(path):
            result["error"] = "unsupported file type"
            return result

        is_code = self._is_code_file(path)
        result["is_code"] = is_code

        # Step 1: Delegate to bridge for SmartRAG storage
        try:
            bridge_result = self._store_via_bridge(path)
            slug = bridge_result.get("slug") if isinstance(bridge_result, dict) else None
            result["slug"] = slug
        except Exception as exc:  # noqa: BLE001
            log.error("ingest: bridge storage failed for %s: %s", path, exc)
            result["status"] = "error"
            result["error"] = f"bridge error: {exc}"
            return result

        # Step 2: Structural analysis for code files
        structure: CodeStructure | None = None
        if is_code:
            lang = self._detect_language(path)
            try:
                structure = self._structural.analyze(path, language=lang)
                result["structure"] = structure.to_dict()
            except Exception as exc:  # noqa: BLE001
                log.warning("ingest: structural analysis failed for %s: %s", path, exc)
                # Non-blocking: continue without structure

            # Update article frontmatter with structural data if we got both
            if slug and structure and structure.ast_patterns_detected > 0:
                try:
                    self._update_frontmatter(slug, structure)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "ingest: frontmatter update failed for %s: %s", slug, exc
                    )

        # Step 3: Queue for compile
        if slug:
            self._compile_queue.append(slug)

        result["status"] = "ok"
        return result

    def ingest_directory(self, path: str, recursive: bool = True) -> list[dict]:
        """Ingest all supported files from a directory.

        Returns a list of per-file result dicts (same shape as
        :meth:`ingest_file`).
        """
        root = Path(path)
        if not root.is_dir():
            log.error("ingest_directory: not a directory: %s", path)
            return []

        results: list[dict] = []
        for file_path in self._walk(root, recursive=recursive):
            result = self.ingest_file(str(file_path))
            results.append(result)

        return results

    def get_compile_queue(self) -> list[str]:
        """Return slugs pending LLM compile enrichment."""
        return list(self._compile_queue)

    def clear_compile_queue(self) -> None:
        """Clear the compile queue (after handing off to compile agent)."""
        self._compile_queue.clear()

    def process_compile_queue(self, compiler, progress_callback=None) -> list:
        """Run the WikiCompiler over all queued slugs, then clear the queue.

        Args:
            compiler: A ``WikiCompiler`` instance (or any object with a
                ``compile_batch(slugs, progress_callback)`` method).
            progress_callback: Optional callback(current, total, title).

        Returns:
            List of ``CompileResult`` objects from the compiler.
        """
        if not self._compile_queue:
            log.info("Compile queue is empty -- nothing to process")
            return []

        slugs = list(self._compile_queue)
        log.info("Processing compile queue: %d slugs", len(slugs))

        results = compiler.compile_batch(slugs, progress_callback=progress_callback)
        self._compile_queue.clear()
        return results

    # -- static helpers -----------------------------------------------------

    @staticmethod
    def _is_code_file(path: str) -> bool:
        """Check if file is source code based on extension."""
        return Path(path).suffix.lower() in _CODE_EXTENSIONS

    @staticmethod
    def _is_supported_file(path: str) -> bool:
        """Check if file has any supported extension."""
        return Path(path).suffix.lower() in _ALL_SUPPORTED

    @staticmethod
    def _detect_language(path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(path).suffix.lower()
        return _EXT_TO_LANGUAGE.get(ext, "unknown")

    # -- private helpers ----------------------------------------------------

    def _store_via_bridge(self, path: str) -> dict:
        """Delegate to the knowledge bridge for SmartRAG article creation.

        The bridge is expected to expose an ``ingest_file(path) -> dict``
        method that returns at minimum ``{"slug": "<article-slug>"}``.
        """
        if self._bridge is None:
            # No bridge configured -- return a synthetic slug
            slug = Path(path).stem.replace(" ", "-").lower()
            return {"slug": slug}

        if hasattr(self._bridge, "ingest_file"):
            return self._bridge.ingest_file(path)

        # Fallback for minimal / placeholder bridges
        slug = Path(path).stem.replace(" ", "-").lower()
        log.debug(
            "ingest: bridge does not implement ingest_file, using slug '%s'", slug
        )
        return {"slug": slug}

    def _update_frontmatter(self, slug: str, structure: CodeStructure) -> None:
        """Push structural metadata into the article's frontmatter via bridge."""
        if self._bridge is None:
            return
        if hasattr(self._bridge, "update_frontmatter"):
            self._bridge.update_frontmatter(slug, structure.to_dict())

    @staticmethod
    def _walk(root: Path, recursive: bool = True) -> list[Path]:
        """Collect supported files under *root*, respecting skip-dirs."""
        files: list[Path] = []
        if recursive:
            for dirpath, dirnames, filenames in os.walk(root):
                # Prune skipped directories in-place
                dirnames[:] = [
                    d for d in dirnames if d not in _SKIP_DIRS
                ]
                for fname in sorted(filenames):
                    fp = Path(dirpath) / fname
                    if fp.suffix.lower() in _ALL_SUPPORTED:
                        files.append(fp)
        else:
            for fp in sorted(root.iterdir()):
                if fp.is_file() and fp.suffix.lower() in _ALL_SUPPORTED:
                    files.append(fp)
        return files
