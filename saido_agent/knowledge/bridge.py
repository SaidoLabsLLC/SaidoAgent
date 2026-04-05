"""Knowledge bridge — connects Saido Agent to SmartRAG retrieval engine.

SmartRAG owns ALL retrieval, storage, indexing, and splitting logic.
This bridge wraps SmartRAG for Saido Agent's domain-specific needs:
- Thin CRUD wrappers with consistent error handling
- Saido-specific directory structure (raw/, outputs/, saido/)
- Code structure frontmatter extension
- Compile-aware ingestion orchestration
- Backlink extraction (SmartRAG stores wikilinks but bridge queries them)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SmartRAG import — fail fast with clear error if not installed
# ---------------------------------------------------------------------------
try:
    from smartrag import (
        Document,
        IngestResult,
        QueryResult,
        SearchResult,
        SmartRAG,
        SmartRAGConfig,
    )

    SMARTRAG_AVAILABLE = True
except ImportError:
    SMARTRAG_AVAILABLE = False
    logger.warning("SmartRAG not installed. Run: pip install smartrag")

    # Sentinel types so the module can still be imported
    SmartRAG = None  # type: ignore[assignment,misc]
    SmartRAGConfig = None  # type: ignore[assignment,misc]
    Document = None  # type: ignore[assignment,misc]
    IngestResult = None  # type: ignore[assignment,misc]
    QueryResult = None  # type: ignore[assignment,misc]
    SearchResult = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Embeddings extra availability check
# ---------------------------------------------------------------------------
try:
    from smartrag.embeddings import EmbeddingIndex  # noqa: F401

    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

# ---------------------------------------------------------------------------
# Wikilink regex for backlink extraction
# ---------------------------------------------------------------------------
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:#[^\]|]*)?(?:\|[^\]]+)?\]\]")

# ---------------------------------------------------------------------------
# Saido directory layout (created alongside SmartRAG's managed dirs)
# ---------------------------------------------------------------------------
_SAIDO_DIRS = [
    "raw",
    os.path.join("outputs", "reports"),
    os.path.join("outputs", "slides"),
    os.path.join("outputs", "charts"),
    "saido",
]


@dataclass
class BridgeConfig:
    """Saido-specific overrides layered on top of SmartRAGConfig defaults."""

    knowledge_root: str = "knowledge"
    split_threshold: int = 2000
    synopsis_mode: str = "extractive"
    fts5: bool = True
    embeddings: bool = False
    obsidian_compat: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_smartrag_config(self) -> "SmartRAGConfig":
        """Build a SmartRAGConfig from Saido settings + user overrides."""
        if not SMARTRAG_AVAILABLE:
            raise RuntimeError("SmartRAG is not installed")
        kwargs: dict[str, Any] = {
            "split_threshold": self.split_threshold,
            "synopsis_mode": self.synopsis_mode,
            "fts5": self.fts5,
            "embeddings": self.embeddings,
            "obsidian_compat": self.obsidian_compat,
        }
        kwargs.update(self.extra)
        return SmartRAGConfig(**kwargs)


class KnowledgeBridge:
    """Thin integration layer between Saido Agent and SmartRAG.

    All retrieval, storage, indexing, and splitting logic is delegated to
    SmartRAG.  This class adds:
    - Consistent CRUD wrappers with Saido error handling
    - Saido directory scaffolding (raw/, outputs/, saido/)
    - Code-structure frontmatter extension
    - Compile-aware ingestion orchestration
    - Backlink queries across the knowledge store
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: BridgeConfig | None = None,
        smartrag_instance: "SmartRAG | None" = None,
    ) -> None:
        self._config = config or BridgeConfig()
        self._root = Path(self._config.knowledge_root).resolve()
        self._rag: SmartRAG | None = None

        # Scaffold Saido-owned directories
        self._ensure_saido_dirs()

        if smartrag_instance is not None:
            self._rag = smartrag_instance
        elif SMARTRAG_AVAILABLE:
            sr_config = self._config.to_smartrag_config()
            self._rag = SmartRAG(str(self._root), sr_config)
            logger.info("SmartRAG initialized at %s", self._root)
        else:
            logger.warning(
                "SmartRAG not available — bridge running in degraded mode. "
                "Install with: pip install smartrag"
            )

    @property
    def available(self) -> bool:
        """True when SmartRAG is loaded and operational."""
        return self._rag is not None

    # ------------------------------------------------------------------
    # CRUD wrappers (delegate to SmartRAG._store)
    # ------------------------------------------------------------------

    def create_article(
        self,
        slug: str,
        body: str,
        frontmatter: dict[str, Any] | None = None,
    ) -> Document | None:
        """Create a new article in the knowledge store."""
        if not self._require_rag("create_article"):
            return None
        assert self._rag is not None
        doc = self._rag._store.create(slug, body, frontmatter)
        logger.debug("Created article: %s", slug)
        return doc

    def read_article(self, slug: str) -> Document | None:
        """Return a full document by slug, or None if not found."""
        if not self._require_rag("read_article"):
            return None
        assert self._rag is not None
        return self._rag.get(slug)

    def read_article_frontmatter(self, slug: str) -> dict[str, Any] | None:
        """Tier-0 fast path: return only frontmatter metadata (no body)."""
        if not self._require_rag("read_article_frontmatter"):
            return None
        assert self._rag is not None
        try:
            return self._rag._store.read_frontmatter(slug)
        except Exception:
            logger.debug("Frontmatter not found for slug: %s", slug)
            return None

    def update_article(
        self,
        slug: str,
        body: str | None = None,
        frontmatter_updates: dict[str, Any] | None = None,
    ) -> Document | None:
        """Update an existing article's body and/or frontmatter."""
        if not self._require_rag("update_article"):
            return None
        assert self._rag is not None
        doc = self._rag._store.update(
            slug, body=body, frontmatter_updates=frontmatter_updates
        )
        logger.debug("Updated article: %s", slug)
        return doc

    def delete_article(self, slug: str) -> bool:
        """Delete an article. Returns True if SmartRAG was available."""
        if not self._require_rag("delete_article"):
            return False
        assert self._rag is not None
        self._rag.delete(slug)
        logger.debug("Deleted article: %s", slug)
        return True

    def list_articles(self) -> list[tuple[str, str, str]]:
        """List all articles as (slug, title, summary) tuples."""
        if not self._require_rag("list_articles"):
            return []
        assert self._rag is not None
        return self._rag._store.list_all()

    # ------------------------------------------------------------------
    # Search & query (delegate to SmartRAG top-level API)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Full-text search with optional filters."""
        if not self._require_rag("search"):
            return []
        assert self._rag is not None
        return self._rag.search(query, top_k=top_k, filters=filters)

    def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> QueryResult | None:
        """Tiered retrieval query — SmartRAG's highest-level API."""
        if not self._require_rag("query"):
            return None
        assert self._rag is not None
        return self._rag.query(question, top_k=top_k)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def reindex(self, incremental: bool = True) -> int:
        """Trigger SmartRAG reindex. Returns count of processed docs."""
        if not self._require_rag("reindex"):
            return 0
        assert self._rag is not None
        count = self._rag.reindex(incremental=incremental)
        logger.info("Reindex complete: %d documents processed", count)
        return count

    @property
    def stats(self) -> dict[str, Any]:
        """Proxy to SmartRAG stats: document_count, index_size, etc."""
        if not self._require_rag("stats"):
            return {"document_count": 0, "index_size_bytes": 0, "categories": []}
        assert self._rag is not None
        return self._rag.stats

    @property
    def embeddings_enabled(self) -> bool:
        """True when embeddings are enabled in the bridge config."""
        return self._config.embeddings

    def embeddings_status(self) -> dict[str, Any]:
        """Return embedding coverage statistics.

        Returns dict with keys:
        - enabled: whether embeddings are configured on
        - available: whether smartrag[embeddings] extra is installed
        - total_articles: total articles in the store
        - articles_with_embeddings: articles that have embedding vectors
        """
        total = 0
        with_embeddings = 0
        if self._rag is not None:
            st = self._rag.stats
            total = st.get("document_count", 0)
            with_embeddings = st.get("embedded_count", 0)
        return {
            "enabled": self._config.embeddings,
            "available": HAS_EMBEDDINGS,
            "total_articles": total,
            "articles_with_embeddings": with_embeddings,
        }

    def enable_embeddings(self) -> None:
        """Enable embeddings and reinitialize SmartRAG with the new config."""
        if not HAS_EMBEDDINGS:
            raise RuntimeError(
                "Embeddings extra not installed. "
                "Run: pip install smartrag[embeddings]"
            )
        self._config.embeddings = True
        if SMARTRAG_AVAILABLE:
            sr_config = self._config.to_smartrag_config()
            self._rag = SmartRAG(str(self._root), sr_config)
            logger.info("SmartRAG reinitialized with embeddings enabled")

    def disable_embeddings(self) -> None:
        """Disable embeddings and reinitialize SmartRAG without them."""
        self._config.embeddings = False
        if SMARTRAG_AVAILABLE:
            sr_config = self._config.to_smartrag_config()
            self._rag = SmartRAG(str(self._root), sr_config)
            logger.info("SmartRAG reinitialized with embeddings disabled")

    # ------------------------------------------------------------------
    # Saido-specific extensions (NOT in SmartRAG)
    # ------------------------------------------------------------------

    def add_code_structure(
        self, slug: str, code_structure: dict[str, Any]
    ) -> Document | None:
        """Attach a code_structure map to an article's frontmatter.

        This is a Saido-specific extension for storing parsed AST / symbol
        information alongside knowledge articles.
        """
        if not self._require_rag("add_code_structure"):
            return None
        return self.update_article(
            slug, frontmatter_updates={"code_structure": code_structure}
        )

    def ingest_with_compile(
        self,
        file_path: str,
        compiler: Any | None = None,
        structural_analyzer: Any | None = None,
    ) -> IngestResult | None:
        """Orchestrate full ingest: raw copy -> SmartRAG ingest -> optional compile.

        1. Copy the source file into ``raw/`` for provenance.
        2. Delegate to SmartRAG ``ingest()`` for parsing + indexing.
        3. If a *compiler* callable is provided, run it on the ingest result
           and log to ``saido/compile_log.json``.
        4. If a *structural_analyzer* callable is provided, attach its output
           as ``code_structure`` frontmatter.

        Args:
            file_path: Path to the file to ingest.
            compiler: Optional callable(IngestResult) -> dict with compile
                metadata.
            structural_analyzer: Optional callable(str) -> dict that returns
                code structure from a file path.

        Returns:
            The SmartRAG IngestResult, or None if SmartRAG is unavailable.
        """
        if not self._require_rag("ingest_with_compile"):
            return None
        assert self._rag is not None

        src = Path(file_path)
        if not src.exists():
            logger.error("File not found: %s", file_path)
            return None

        # 1. Copy to raw/
        raw_dir = self._root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / src.name
        dest.write_bytes(src.read_bytes())

        # 2. SmartRAG ingest
        result = self._rag.ingest(str(src))
        # ingest may return a list for directories; normalise to single result
        if isinstance(result, list):
            result = result[0] if result else None
        if result is None:
            return None

        # 3. Optional compile step
        if compiler is not None:
            compile_meta = compiler(result)
            self._append_compile_log(result.slug, compile_meta)

        # 4. Optional structural analysis
        if structural_analyzer is not None:
            code_struct = structural_analyzer(str(src))
            if code_struct:
                self.add_code_structure(result.slug, code_struct)

        return result

    def get_backlinks(self, slug: str) -> list[str]:
        """Return slugs of articles that contain a ``[[slug]]`` wikilink.

        SmartRAG stores wikilinks in document bodies.  This method scans
        all documents for ``[[<slug>]]`` references and returns the list
        of linking slugs.  This is a Saido-layer feature because SmartRAG
        v0.1.0 does not yet expose a backlink query API.
        """
        if not self._require_rag("get_backlinks"):
            return []
        assert self._rag is not None

        # First, try backlinks.json (SmartRAG creates it but may be empty)
        bl_path = self._root / "backlinks.json"
        if bl_path.exists():
            try:
                data = json.loads(bl_path.read_text(encoding="utf-8"))
                if slug in data and data[slug]:
                    return data[slug]
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: scan all documents for wikilinks to this slug
        backlinks: list[str] = []
        articles = self.list_articles()
        for art_slug, _title, _summary in articles:
            if art_slug == slug:
                continue
            doc = self.read_article(art_slug)
            if doc is None:
                continue
            linked_slugs = {m.group(1) for m in _WIKILINK_RE.finditer(doc.body)}
            if slug in linked_slugs:
                backlinks.append(art_slug)
        return backlinks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_rag(self, operation: str) -> bool:
        """Guard: log warning and return False if SmartRAG unavailable."""
        if self._rag is None:
            logger.warning(
                "SmartRAG unavailable — %s returning empty result", operation
            )
            return False
        return True

    def _ensure_saido_dirs(self) -> None:
        """Create Saido-owned directories alongside the knowledge root."""
        for rel in _SAIDO_DIRS:
            d = self._root / rel
            d.mkdir(parents=True, exist_ok=True)
        # Ensure compile log exists
        log_path = self._root / "saido" / "compile_log.json"
        if not log_path.exists():
            log_path.write_text("[]", encoding="utf-8")

    def _append_compile_log(
        self, slug: str, compile_meta: dict[str, Any]
    ) -> None:
        """Append an entry to the Saido compile log."""
        log_path = self._root / "saido" / "compile_log.json"
        try:
            entries = json.loads(log_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            entries = []
        entries.append({"slug": slug, "compile": compile_meta})
        log_path.write_text(
            json.dumps(entries, indent=2), encoding="utf-8"
        )
