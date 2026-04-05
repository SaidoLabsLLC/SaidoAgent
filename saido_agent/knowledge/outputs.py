"""Output generation — structured reports and exports from knowledge store.

Generates:
  - Structured markdown reports synthesising articles on a topic
  - Full knowledge store export as zip archive
  - Single-article export as standalone markdown
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saido_agent.knowledge.bridge import KnowledgeBridge
    from saido_agent.core.routing import ModelRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug[:80] or "untitled"


# Maximum articles to include in a report prompt
_MAX_REPORT_ARTICLES = 10


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReportResult:
    """Outcome of a report generation attempt."""

    title: str
    path: str
    word_count: int
    articles_cited: int
    status: str  # "generated", "failed"
    error: str | None = None


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generates structured reports from knowledge store content."""

    def __init__(
        self,
        bridge: KnowledgeBridge,
        model_router: ModelRouter | None = None,
    ) -> None:
        self._bridge = bridge
        self._router = model_router

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(
        self,
        topic: str,
        format: str = "markdown",
    ) -> ReportResult:
        """Generate a report on *topic* from knowledge articles.

        Workflow:
          1. Search knowledge store for related articles
          2. Read top articles (up to ``_MAX_REPORT_ARTICLES``)
          3. Build a report-generation prompt and send to LLM
          4. Save the resulting markdown to the outputs directory
          5. Return a ``ReportResult``
        """
        # -- Guard: empty knowledge store ---------------------------------
        stats = self._bridge.stats
        doc_count = stats.get("document_count", 0)
        if doc_count == 0:
            return ReportResult(
                title=topic,
                path="",
                word_count=0,
                articles_cited=0,
                status="failed",
                error="Knowledge store is empty. Ingest documents first.",
            )

        # -- Step 1: search -----------------------------------------------
        search_results = self._bridge.search(topic, top_k=_MAX_REPORT_ARTICLES)
        if not search_results:
            return ReportResult(
                title=topic,
                path="",
                word_count=0,
                articles_cited=0,
                status="failed",
                error=f"No articles found matching topic: {topic}",
            )

        # -- Step 2: read full articles -----------------------------------
        articles: list[dict[str, str]] = []
        for result in search_results:
            doc = self._bridge.read_article(result.slug)
            if doc is not None:
                articles.append({
                    "slug": result.slug,
                    "title": result.title,
                    "body": doc.body,
                })

        if not articles:
            return ReportResult(
                title=topic,
                path="",
                word_count=0,
                articles_cited=0,
                status="failed",
                error="Could not read any matching articles.",
            )

        # -- Step 3: build prompt and call LLM ----------------------------
        prompt = self._build_report_prompt(topic, articles)
        report_text = self._call_llm(prompt)

        if report_text is None:
            return ReportResult(
                title=topic,
                path="",
                word_count=0,
                articles_cited=0,
                status="failed",
                error="LLM call failed. Check model availability.",
            )

        # -- Step 4: save to disk -----------------------------------------
        date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        slug = _slugify(topic)
        filename = f"{slug}-{date_str}.md"

        output_dir = self._reports_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        output_path.write_text(report_text, encoding="utf-8")

        word_count = len(report_text.split())
        cited = self._count_citations(report_text, articles)

        logger.info(
            "Report generated: %s (%d words, %d sources)",
            output_path,
            word_count,
            cited,
        )

        return ReportResult(
            title=topic,
            path=str(output_path),
            word_count=word_count,
            articles_cited=cited,
            status="generated",
        )

    def export_docs(self, output_dir: str | None = None) -> str:
        """Export entire knowledge store as a zip file.

        Returns the path to the generated zip archive, or an empty string
        on failure.
        """
        articles = self._bridge.list_articles()
        if not articles:
            logger.warning("No articles to export.")
            return ""

        date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        dest_dir = Path(output_dir) if output_dir else self._exports_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dest_dir / f"knowledge-export-{date_str}.zip"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for slug, title, _summary in articles:
                doc = self._bridge.read_article(slug)
                if doc is None:
                    continue
                # Build a simple markdown file with frontmatter-style header
                md = f"# {title}\n\n{doc.body}\n"
                zf.writestr(f"{slug}.md", md)

        zip_path.write_bytes(buf.getvalue())
        logger.info("Exported %d articles to %s", len(articles), zip_path)
        return str(zip_path)

    def export_article(self, slug: str, output_dir: str | None = None) -> str:
        """Export a single article as standalone markdown.

        Returns the path to the exported file, or an empty string on
        failure.
        """
        doc = self._bridge.read_article(slug)
        if doc is None:
            logger.warning("Article not found: %s", slug)
            return ""

        dest_dir = Path(output_dir) if output_dir else self._exports_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)

        title = getattr(doc, "title", slug)
        md = f"# {title}\n\n{doc.body}\n"
        out_path = dest_dir / f"{slug}.md"
        out_path.write_text(md, encoding="utf-8")
        logger.info("Exported article %s to %s", slug, out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_report_prompt(
        topic: str,
        articles: list[dict[str, str]],
    ) -> str:
        """Assemble the report-generation prompt."""
        sections: list[str] = []
        sections.append(
            "You are Saido Agent, a knowledge synthesis assistant. "
            "Generate a structured report on the given topic using ONLY "
            "the provided knowledge base articles.\n"
        )
        sections.append("## Knowledge Base Articles\n")
        for art in articles:
            sections.append(
                f"### [{art['title']}] (slug: {art['slug']})\n"
                f"{art['body']}\n"
            )
        sections.append(f"## Topic\n{topic}\n")
        sections.append(
            "## Output Structure\n"
            "Generate the report in markdown with the following structure:\n\n"
            "## Executive Summary\n"
            "## Key Findings\n"
            "### [Theme 1]\n"
            "### [Theme 2]\n"
            "## Recommendations\n"
            "## Sources\n\n"
            "Cite sources using [Article Title] notation.\n"
            "Only include information from the provided articles."
        )
        return "\n".join(sections)

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str | None:
        """Send the prompt to the LLM via ModelRouter.

        Returns the generated text or None on failure.
        """
        if self._router is None:
            logger.warning("No ModelRouter configured — cannot call LLM")
            return None

        try:
            from saido_agent.core.providers import stream as llm_stream

            provider, model = self._router.select_model("report")
            messages = [{"role": "user", "content": prompt}]
            total_text = ""

            for chunk in llm_stream(
                model=(
                    f"{provider}/{model}"
                    if provider not in ("anthropic",)
                    else model
                ),
                system="",
                messages=messages,
                tool_schemas=[],
                config={},
            ):
                from saido_agent.core.providers import AssistantTurn, TextChunk

                if isinstance(chunk, TextChunk):
                    total_text += chunk.text
                elif isinstance(chunk, AssistantTurn):
                    total_text = chunk.text

            return total_text or None

        except Exception:
            logger.exception("LLM call failed during report generation")
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_citations(
        report_text: str,
        articles: list[dict[str, str]],
    ) -> int:
        """Count how many of the provided articles are cited in the report."""
        titles = {art["title"] for art in articles}
        cited = 0
        for title in titles:
            if f"[{title}]" in report_text:
                cited += 1
        return cited

    def _reports_dir(self) -> Path:
        """Return the reports output directory under the knowledge root."""
        root = Path(self._bridge._config.knowledge_root).resolve()
        return root / "outputs" / "reports"

    def _exports_dir(self) -> Path:
        """Return the exports output directory under the knowledge root."""
        root = Path(self._bridge._config.knowledge_root).resolve()
        return root / "outputs" / "exports"
