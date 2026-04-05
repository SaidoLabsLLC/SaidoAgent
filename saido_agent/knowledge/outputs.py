"""Output generation — structured reports and exports from knowledge store.

Generates:
  - Structured markdown reports synthesising articles on a topic
  - Full knowledge store export as zip archive
  - Single-article export as standalone markdown
  - Marp presentation slides from knowledge articles
  - Matplotlib charts from natural language descriptions
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
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


# ---------------------------------------------------------------------------
# SlideResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class SlideResult:
    """Outcome of a slide generation attempt."""

    title: str
    path: str
    slide_count: int
    status: str  # "generated", "failed"
    error: str | None = None


# Maximum articles to include in a slide prompt
_MAX_SLIDE_ARTICLES = 8


# ---------------------------------------------------------------------------
# SlideGenerator
# ---------------------------------------------------------------------------

class SlideGenerator:
    """Generates Marp presentation slides from knowledge store content."""

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

    def generate_slides(self, topic: str) -> SlideResult:
        """Generate Marp presentation slides on *topic* from knowledge articles.

        Workflow:
          1. Search knowledge store for related articles
          2. Read top articles
          3. Build a slide-generation prompt and send to LLM
          4. Save the resulting Marp markdown to the outputs directory
          5. Optionally convert to HTML/PDF via marp-cli
          6. Return a ``SlideResult``
        """
        # -- Guard: empty knowledge store ---------------------------------
        stats = self._bridge.stats
        doc_count = stats.get("document_count", 0)
        if doc_count == 0:
            return SlideResult(
                title=topic,
                path="",
                slide_count=0,
                status="failed",
                error="Knowledge store is empty. Ingest documents first.",
            )

        # -- Step 1: search -----------------------------------------------
        search_results = self._bridge.search(topic, top_k=_MAX_SLIDE_ARTICLES)
        if not search_results:
            return SlideResult(
                title=topic,
                path="",
                slide_count=0,
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
            return SlideResult(
                title=topic,
                path="",
                slide_count=0,
                status="failed",
                error="Could not read any matching articles.",
            )

        # -- Step 3: build prompt and call LLM ----------------------------
        prompt = self._build_slide_prompt(topic, articles)
        slide_text = self._call_llm(prompt)

        if slide_text is None:
            return SlideResult(
                title=topic,
                path="",
                slide_count=0,
                status="failed",
                error="LLM call failed. Check model availability.",
            )

        # -- Step 4: save to disk -----------------------------------------
        date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        slug = _slugify(topic)
        filename = f"{slug}-{date_str}.md"

        output_dir = self._slides_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        output_path.write_text(slide_text, encoding="utf-8")

        slide_count = slide_text.count("\n---")
        # The first slide (before any ---) counts as 1
        slide_count = max(slide_count, 0) + 1

        # -- Step 5: optional marp-cli conversion -------------------------
        self._try_marp_convert(output_path)

        logger.info(
            "Slides generated: %s (%d slides)",
            output_path,
            slide_count,
        )

        return SlideResult(
            title=topic,
            path=str(output_path),
            slide_count=slide_count,
            status="generated",
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_slide_prompt(
        topic: str,
        articles: list[dict[str, str]],
    ) -> str:
        """Assemble the slide-generation prompt."""
        sections: list[str] = []
        sections.append(
            "You are Saido Agent, a knowledge synthesis assistant. "
            "Create a Marp presentation on the given topic using ONLY "
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
            '## Output Format\n'
            'Create a Marp presentation. Format:\n\n'
            '---\n'
            'marp: true\n'
            'theme: default\n'
            '---\n\n'
            '# Title Slide\n\n'
            '---\n\n'
            '## Slide 2 Title\n'
            '- Bullet point\n'
            '- Bullet point\n\n'
            '---\n\n'
            '(continue for 8-12 slides)\n\n'
            'Only include information from the provided articles.'
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

            provider, model = self._router.select_model("slides")
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
            logger.exception("LLM call failed during slide generation")
            return None

    # ------------------------------------------------------------------
    # Marp CLI conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _try_marp_convert(md_path: Path) -> Path | None:
        """Attempt to convert Marp markdown to HTML via marp-cli.

        Returns the HTML path on success, None if marp-cli is unavailable.
        """
        if shutil.which("npx") is None:
            logger.debug("npx not found — skipping marp conversion")
            return None

        html_path = md_path.with_suffix(".html")
        try:
            subprocess.run(
                ["npx", "@marp-team/marp-cli", str(md_path), "-o", str(html_path)],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            logger.info("Marp conversion successful: %s", html_path)
            return html_path
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("Marp conversion skipped: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _slides_dir(self) -> Path:
        """Return the slides output directory under the knowledge root."""
        root = Path(self._bridge._config.knowledge_root).resolve()
        return root / "outputs" / "slides"


# ---------------------------------------------------------------------------
# ChartResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChartResult:
    """Outcome of a chart generation attempt."""

    title: str
    path: str
    chart_type: str
    status: str  # "generated", "failed"
    error: str | None = None


# Maximum articles to include in a chart prompt
_MAX_CHART_ARTICLES = 6

# Allowlist of imports permitted inside chart sandbox
_CHART_ALLOWED_MODULES = frozenset({"matplotlib", "numpy", "json", "math"})

# Blocklist of dangerous modules that must never appear in chart code
_CHART_BLOCKED_MODULES = frozenset({
    "os", "subprocess", "shutil", "socket", "sys", "importlib",
    "ctypes", "multiprocessing", "threading", "signal",
    "webbrowser", "http", "urllib", "requests", "pathlib",
    "pickle", "shelve", "tempfile", "glob", "fnmatch",
    "code", "codeop", "compile", "compileall", "exec",
    "builtins", "__builtin__",
})


# ---------------------------------------------------------------------------
# Chart sandbox execution
# ---------------------------------------------------------------------------

def _validate_chart_code(code: str) -> str | None:
    """Validate that chart code only uses allowed imports.

    Returns an error message if the code is unsafe, or None if valid.
    """
    # Check each line for import statements
    for line in code.splitlines():
        stripped = line.strip()

        # Skip comments and empty lines
        if stripped.startswith("#") or not stripped:
            continue

        # Match 'import X' and 'from X import ...'
        import_match = re.match(r"^import\s+([\w.]+)", stripped)
        from_match = re.match(r"^from\s+([\w.]+)", stripped)

        module_name = None
        if import_match:
            module_name = import_match.group(1).split(".")[0]
        elif from_match:
            module_name = from_match.group(1).split(".")[0]

        if module_name is not None:
            if module_name in _CHART_BLOCKED_MODULES:
                return f"Blocked import: {module_name}"
            if module_name not in _CHART_ALLOWED_MODULES:
                return f"Disallowed import: {module_name}"

    # Check for exec/eval/compile calls
    if re.search(r"\b(exec|eval|compile|__import__)\s*\(", code):
        return "Blocked function call: exec/eval/compile/__import__"

    # Check for open() calls
    if re.search(r"\bopen\s*\(", code):
        return "Blocked function call: open()"

    return None


def _execute_chart_code(code: str, output_path: str, timeout: int = 30) -> str | None:
    """Execute matplotlib code in a subprocess sandbox.

    Returns an error message on failure, or None on success.
    The code must save its figure to the path provided via the
    ``__OUTPUT_PATH__`` placeholder.
    """
    # Inject the output path into the code
    full_code = f"__OUTPUT_PATH__ = {output_path!r}\n{code}"

    # Write to a temp file and execute in a subprocess
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    ) as tmp:
        tmp.write(full_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            return f"Chart code execution failed:\n{stderr[:500]}"
        return None
    except subprocess.TimeoutExpired:
        return f"Chart code execution timed out after {timeout}s"
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# ChartGenerator
# ---------------------------------------------------------------------------

class ChartGenerator:
    """Generates Matplotlib charts from natural language descriptions."""

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

    def generate_chart(self, description: str) -> ChartResult:
        """Generate a chart from a natural language description + knowledge data.

        Workflow:
          1. Search knowledge store for relevant data
          2. Send description + articles to LLM for matplotlib code generation
          3. Validate the generated code for safety
          4. Execute in a sandboxed subprocess
          5. Return a ``ChartResult``
        """
        # -- Guard: empty knowledge store ---------------------------------
        stats = self._bridge.stats
        doc_count = stats.get("document_count", 0)
        if doc_count == 0:
            return ChartResult(
                title=description,
                path="",
                chart_type="",
                status="failed",
                error="Knowledge store is empty. Ingest documents first.",
            )

        # -- Step 1: search -----------------------------------------------
        search_results = self._bridge.search(description, top_k=_MAX_CHART_ARTICLES)
        if not search_results:
            return ChartResult(
                title=description,
                path="",
                chart_type="",
                status="failed",
                error=f"No articles found matching description: {description}",
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
            return ChartResult(
                title=description,
                path="",
                chart_type="",
                status="failed",
                error="Could not read any matching articles.",
            )

        # -- Step 3: build prompt and call LLM ----------------------------
        prompt = self._build_chart_prompt(description, articles)
        llm_response = self._call_llm(prompt)

        if llm_response is None:
            return ChartResult(
                title=description,
                path="",
                chart_type="",
                status="failed",
                error="LLM call failed. Check model availability.",
            )

        # Extract code from the LLM response
        chart_code = self._extract_code(llm_response)
        chart_type = self._detect_chart_type(chart_code)

        # -- Step 4: validate code safety ---------------------------------
        validation_error = _validate_chart_code(chart_code)
        if validation_error is not None:
            return ChartResult(
                title=description,
                path="",
                chart_type=chart_type,
                status="failed",
                error=f"Unsafe chart code: {validation_error}",
            )

        # -- Step 5: prepare output path and execute ----------------------
        date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        slug = _slugify(description)
        filename = f"{slug}-{date_str}.png"

        output_dir = self._charts_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        exec_error = _execute_chart_code(chart_code, str(output_path))
        if exec_error is not None:
            return ChartResult(
                title=description,
                path="",
                chart_type=chart_type,
                status="failed",
                error=exec_error,
            )

        # Verify the file was actually created
        if not output_path.exists():
            return ChartResult(
                title=description,
                path="",
                chart_type=chart_type,
                status="failed",
                error="Chart code ran but no output file was produced.",
            )

        logger.info("Chart generated: %s (type: %s)", output_path, chart_type)

        return ChartResult(
            title=description,
            path=str(output_path),
            chart_type=chart_type,
            status="generated",
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chart_prompt(
        description: str,
        articles: list[dict[str, str]],
    ) -> str:
        """Assemble the chart-generation prompt."""
        sections: list[str] = []
        sections.append(
            "You are Saido Agent, a knowledge synthesis assistant. "
            "Generate Python matplotlib code to create a chart based on "
            "the description and knowledge base articles below.\n"
        )
        sections.append("## Knowledge Base Articles\n")
        for art in articles:
            sections.append(
                f"### [{art['title']}] (slug: {art['slug']})\n"
                f"{art['body']}\n"
            )
        sections.append(f"## Chart Description\n{description}\n")
        sections.append(
            "## Instructions\n"
            "Generate ONLY Python code (no explanation). The code must:\n"
            "1. Only import from: matplotlib, numpy, json, math\n"
            "2. Save the chart to the path in the variable __OUTPUT_PATH__\n"
            "3. Use plt.savefig(__OUTPUT_PATH__) and plt.close()\n"
            "4. NOT use plt.show()\n\n"
            "Wrap the code in a ```python code fence."
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

            provider, model = self._router.select_model("chart")
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
            logger.exception("LLM call failed during chart generation")
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code(llm_response: str) -> str:
        """Extract Python code from LLM response (handles code fences)."""
        # Try to find fenced code block
        match = re.search(
            r"```(?:python)?\s*\n(.*?)```",
            llm_response,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        # Fallback: return the entire response (assume it's raw code)
        return llm_response.strip()

    @staticmethod
    def _detect_chart_type(code: str) -> str:
        """Heuristically detect the chart type from matplotlib code."""
        code_lower = code.lower()
        if "bar(" in code_lower or "barh(" in code_lower:
            return "bar"
        if "pie(" in code_lower:
            return "pie"
        if "scatter(" in code_lower:
            return "scatter"
        if "hist(" in code_lower:
            return "histogram"
        if "plot(" in code_lower:
            return "line"
        return "unknown"

    def _charts_dir(self) -> Path:
        """Return the charts output directory under the knowledge root."""
        root = Path(self._bridge._config.knowledge_root).resolve()
        return root / "outputs" / "charts"
