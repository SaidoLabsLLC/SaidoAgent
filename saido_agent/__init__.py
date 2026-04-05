"""Saido Agent -- Knowledge-compounding AI agent platform by Saido Labs LLC.

Public API surface:
    - ``SaidoAgent``  -- main facade class
    - ``SaidoConfig`` -- public configuration interface
    - ``types``       -- all public dataclasses (IngestResult, SearchResult, etc.)

Internal modules (core, knowledge, memory) should NOT be imported directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from saido_agent.config import SaidoConfig
from saido_agent.types import (
    AgentResult,
    CompileResult,
    IngestResult,
    SearchResult,
    StoreStats,
)

# Lazy imports are used for heavy internal modules to keep import time fast
# and to allow graceful degradation when optional deps (SmartRAG) are missing.

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


class SaidoAgent:
    """Knowledge-compounding AI agent by Saido Labs LLC.

    This is the primary entry point for the ``saido-agent`` SDK.  It wires
    together the knowledge bridge, ingest pipeline, wiki compiler, Q&A
    engine, model router, and cost tracker behind a clean public API.

    Example::

        from saido_agent import SaidoAgent

        agent = SaidoAgent(knowledge_dir="./my_kb")
        agent.ingest("docs/")
        result = agent.query("What authentication methods are supported?")
        print(result.answer)
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        knowledge_dir: str = "./knowledge",
        llm_provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt_extension: str = "",
        routing_config: Optional[str] = None,
    ) -> None:
        self._knowledge_dir = str(Path(knowledge_dir).resolve())
        self._llm_provider = llm_provider
        self._model = model
        self._api_key = api_key
        self._system_prompt_extension = system_prompt_extension

        # -- Internal components (lazy-initialized where possible) --

        # 1. KnowledgeBridge
        from saido_agent.knowledge.bridge import BridgeConfig, KnowledgeBridge

        bridge_cfg = BridgeConfig(knowledge_root=self._knowledge_dir)
        self._bridge = KnowledgeBridge(config=bridge_cfg)

        # 2. ModelRouter
        from saido_agent.core.routing import ModelRouter

        routing_path = Path(routing_config) if routing_config else None
        self._router = ModelRouter(config_path=routing_path)

        # 3. CostTracker
        from saido_agent.core.cost_tracker import CostTracker

        self._cost_tracker = CostTracker()

        # 4. IngestPipeline
        from saido_agent.knowledge.ingest import IngestPipeline

        self._ingest_pipeline = IngestPipeline(
            bridge=self._bridge,
            model_router=self._router,
        )

        # 5. WikiCompiler
        from saido_agent.knowledge.compile import WikiCompiler

        self._compiler = WikiCompiler(
            bridge=self._bridge,
            model_router=self._router,
        )

        # 6. KnowledgeQA
        from saido_agent.knowledge.query import KnowledgeQA

        self._qa = KnowledgeQA(
            bridge=self._bridge,
            model_router=self._router,
        )

        logger.info(
            "SaidoAgent initialized (knowledge_dir=%s, bridge_available=%s)",
            self._knowledge_dir,
            self._bridge.available,
        )

    # ------------------------------------------------------------------
    # Public API -- Ingestion
    # ------------------------------------------------------------------

    def ingest(self, path: str) -> IngestResult:
        """Ingest a file or directory into the knowledge store.

        For directories, all supported files are ingested recursively.
        The returned ``IngestResult`` reflects the top-level operation;
        for directories the ``children`` field lists per-file slugs.

        Args:
            path: Path to a file or directory to ingest.

        Returns:
            An ``IngestResult`` with slug, title, status, and optional
            children/error fields.
        """
        p = Path(path)

        if p.is_dir():
            results = self._ingest_pipeline.ingest_directory(str(p))
            slugs = [r["slug"] for r in results if r.get("slug")]
            failed = [r for r in results if r["status"] == "error"]
            status = "failed" if len(failed) == len(results) and results else "created"
            if not results:
                status = "failed"
            return IngestResult(
                slug=p.name,
                title=p.name,
                status=status,
                children=slugs,
                error=(
                    f"{len(failed)} file(s) failed"
                    if failed
                    else None
                ),
            )

        # Single file
        result = self._ingest_pipeline.ingest_file(str(p))
        slug = result.get("slug") or p.stem
        status_map = {"ok": "created", "skipped": "duplicate", "error": "failed"}
        return IngestResult(
            slug=slug,
            title=slug,
            status=status_map.get(result["status"], result["status"]),
            error=result.get("error"),
        )

    # ------------------------------------------------------------------
    # Public API -- Query & Search
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        context: Optional[dict] = None,
    ) -> "SaidoQueryResult":
        """Ask a knowledge-grounded question.

        Retrieves relevant articles from the knowledge store, constructs
        a grounded prompt, and generates an answer with citations.

        Args:
            question: The question to answer.
            context: Optional additional context dict (reserved for future use).

        Returns:
            A ``SaidoQueryResult`` with answer, citations, confidence, and
            retrieval statistics.
        """
        from saido_agent.knowledge.query import SaidoQueryResult

        return self._qa.query(question, context=context)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search the knowledge store without generating an LLM answer.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``SearchResult`` dataclasses.
        """
        raw_results = self._qa.search(query, top_k=top_k)
        return [
            SearchResult(
                slug=r["slug"],
                title=r["title"],
                summary=r["summary"],
                score=r["score"],
                snippet=r["snippet"],
            )
            for r in raw_results
        ]

    # ------------------------------------------------------------------
    # Public API -- Agent Loop
    # ------------------------------------------------------------------

    def run(self, instruction: str) -> AgentResult:
        """Run a full agent loop with tool use.

        This is a higher-level API that combines knowledge retrieval,
        reasoning, and tool execution into a single call.

        Args:
            instruction: The task instruction to execute.

        Returns:
            An ``AgentResult`` with output text, tool calls, and token usage.
        """
        # Phase 1: delegate to query for knowledge-grounded response
        # Full agent loop with tool use will be implemented in Phase 2+
        qa_result = self._qa.query(instruction)
        return AgentResult(
            output=qa_result.answer,
            tokens_used=qa_result.tokens_used,
        )

    # ------------------------------------------------------------------
    # Public API -- Compilation
    # ------------------------------------------------------------------

    def compile(self, slug: Optional[str] = None) -> CompileResult:
        """Compile/enrich a specific article or all pending articles.

        Compilation upgrades SmartRAG's extractive metadata with
        LLM-generated summaries, concepts, categories, and backlinks.

        Args:
            slug: Specific article slug to compile.  If ``None``, compiles
                all articles in the ingest pipeline's compile queue.

        Returns:
            A ``CompileResult`` with status and enrichment data.
        """
        from saido_agent.knowledge.compile import (
            CompileResult as InternalCompileResult,
        )

        if slug is not None:
            internal = self._compiler.compile(slug)
            return CompileResult(
                slug=internal.slug,
                status=internal.status,
                summary=internal.summary,
                concepts=internal.concepts,
                categories=internal.categories,
                error=internal.error,
            )

        # Compile all pending
        queue = self._ingest_pipeline.get_compile_queue()
        if not queue:
            return CompileResult(
                slug="",
                status="skipped",
                summary="No articles pending compilation.",
            )

        results = self._ingest_pipeline.process_compile_queue(self._compiler)
        if not results:
            return CompileResult(
                slug="",
                status="skipped",
                summary="Compile queue was empty.",
            )

        # Aggregate into a single CompileResult
        compiled = [r for r in results if r.status == "compiled"]
        failed = [r for r in results if r.status == "failed"]
        all_concepts: list[str] = []
        all_categories: list[str] = []
        for r in compiled:
            all_concepts.extend(r.concepts)
            all_categories.extend(r.categories)

        return CompileResult(
            slug=f"batch:{len(results)}",
            status="compiled" if compiled else "failed",
            summary=f"Compiled {len(compiled)}/{len(results)} articles.",
            concepts=list(dict.fromkeys(all_concepts)),
            categories=list(dict.fromkeys(all_categories)),
            error=(
                f"{len(failed)} article(s) failed"
                if failed
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Public API -- Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> StoreStats:
        """Knowledge store statistics."""
        raw = self._bridge.stats
        return StoreStats(
            document_count=raw.get("document_count", 0),
            category_count=len(raw.get("categories", [])),
            total_size_bytes=raw.get("index_size_bytes", 0),
        )

    @property
    def cost(self) -> dict[str, Any]:
        """Current session cost summary.

        Returns a dict with keys: total_cost, total_tokens,
        estimated_savings, and report (formatted string).
        """
        return {
            "total_cost": self._cost_tracker.total_cost,
            "total_tokens": self._cost_tracker.total_tokens,
            "estimated_savings": self._cost_tracker.estimated_savings,
            "report": self._cost_tracker.format_report(),
        }

    @property
    def bridge(self) -> Any:
        """Access the underlying KnowledgeBridge (advanced use)."""
        return self._bridge

    @property
    def router(self) -> Any:
        """Access the underlying ModelRouter (advanced use)."""
        return self._router

    @property
    def cost_tracker(self) -> Any:
        """Access the underlying CostTracker (advanced use)."""
        return self._cost_tracker


__all__ = ["SaidoAgent", "SaidoConfig", "__version__"]
