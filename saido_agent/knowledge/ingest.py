"""Knowledge ingestion pipeline.

Orchestrates: SmartRAG storage (via bridge) + ast-grep structural analysis
+ compile queue population for downstream LLM enrichment.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Comment

from saido_agent.core.ssrf import validate_url
from saido_agent.knowledge.structural import CodeStructure, StructuralAnalyzer

if TYPE_CHECKING:
    pass  # KnowledgeBridge type will be available when bridge is built

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML extraction settings
# ---------------------------------------------------------------------------

# Tags to strip entirely (including their content)
_STRIP_TAGS: frozenset[str] = frozenset({
    "nav", "footer", "script", "style", "aside", "header",
    "noscript", "iframe", "svg",
})

# Default fetch timeout in seconds
_FETCH_TIMEOUT: float = 30.0

# User-Agent for outgoing requests
_USER_AGENT: str = "SaidoAgent/0.1 (knowledge-ingest)"

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

    # -- web ingest API -----------------------------------------------------

    def ingest_url(self, url: str) -> dict:
        """Fetch URL, extract text, ingest into knowledge store.

        Returns a result dict::

            {
                "url": str,
                "slug": str | None,
                "status": "ok" | "error",
                "title": str | None,
                "metadata": dict | None,
                "error": str | None,
            }
        """
        result: dict[str, Any] = {
            "url": url,
            "slug": None,
            "status": "error",
            "title": None,
            "metadata": None,
            "error": None,
        }

        # 1. SSRF validation
        is_safe, reason = validate_url(url)
        if not is_safe:
            result["error"] = f"SSRF blocked: {reason}"
            log.warning("ingest_url: SSRF blocked %s — %s", url, reason)
            return result

        # 2. Fetch with httpx
        try:
            html = self._fetch_url(url)
        except Exception as exc:
            result["error"] = f"Fetch failed: {exc}"
            log.error("ingest_url: fetch failed for %s: %s", url, exc)
            return result

        # 3. Extract content from HTML
        extracted = extract_html_content(html, source_url=url)
        title = extracted.get("title") or _slug_from_url(url)
        text = extracted.get("text", "")

        if not text.strip():
            result["error"] = "No text content extracted"
            return result

        result["title"] = title
        result["metadata"] = {
            "source_url": url,
            "canonical_url": extracted.get("canonical_url"),
            "description": extracted.get("description"),
            "publish_date": extracted.get("publish_date"),
        }

        # 4. Generate slug from title
        slug = _slugify(title)

        # 5. Store via bridge
        try:
            if self._bridge is not None and hasattr(self._bridge, "create_article"):
                frontmatter = {
                    "source_url": url,
                    "source_type": "web",
                    **(result["metadata"] or {}),
                }
                self._bridge.create_article(slug, text, frontmatter=frontmatter)
            result["slug"] = slug
        except Exception as exc:
            log.error("ingest_url: bridge storage failed for %s: %s", url, exc)
            result["error"] = f"Storage failed: {exc}"
            return result

        # 6. Queue for compile
        self._compile_queue.append(slug)
        result["status"] = "ok"
        return result

    def ingest_html(self, html: str, url: str = "", title: str = "") -> dict:
        """Ingest pre-fetched HTML content (e.g. from browser clipper).

        Returns the same result dict shape as :meth:`ingest_url`.
        """
        result: dict[str, Any] = {
            "url": url,
            "slug": None,
            "status": "error",
            "title": None,
            "metadata": None,
            "error": None,
        }

        extracted = extract_html_content(html, source_url=url)
        title = title or extracted.get("title") or _slug_from_url(url) or "untitled-clip"
        text = extracted.get("text", "")

        if not text.strip():
            result["error"] = "No text content extracted"
            return result

        result["title"] = title
        slug = _slugify(title)

        try:
            if self._bridge is not None and hasattr(self._bridge, "create_article"):
                frontmatter = {
                    "source_url": url,
                    "source_type": "web-clip",
                    "description": extracted.get("description"),
                }
                self._bridge.create_article(slug, text, frontmatter=frontmatter)
            result["slug"] = slug
        except Exception as exc:
            result["error"] = f"Storage failed: {exc}"
            return result

        self._compile_queue.append(slug)
        result["status"] = "ok"
        return result

    def ingest_selection(self, text: str, url: str = "", title: str = "") -> dict:
        """Ingest a user-selected text fragment (e.g. from browser clipper).

        Returns the same result dict shape as :meth:`ingest_url`.
        """
        result: dict[str, Any] = {
            "url": url,
            "slug": None,
            "status": "error",
            "title": None,
            "metadata": None,
            "error": None,
        }

        if not text.strip():
            result["error"] = "Empty selection"
            return result

        title = title or _slug_from_url(url) or "untitled-selection"
        result["title"] = title
        slug = _slugify(title)

        try:
            if self._bridge is not None and hasattr(self._bridge, "create_article"):
                frontmatter = {
                    "source_url": url,
                    "source_type": "web-selection",
                }
                self._bridge.create_article(slug, text, frontmatter=frontmatter)
            result["slug"] = slug
        except Exception as exc:
            result["error"] = f"Storage failed: {exc}"
            return result

        self._compile_queue.append(slug)
        result["status"] = "ok"
        return result

    def ingest_search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search the web, fetch top results, and ingest them.

        Uses a simple web search approach. Returns a list of ingest result
        dicts (same shape as :meth:`ingest_url`).
        """
        results: list[dict] = []

        # Use DuckDuckGo HTML search (no API key required)
        try:
            urls = self._search_web(query, max_results=max_results)
        except Exception as exc:
            log.error("ingest_search: web search failed: %s", exc)
            return [{"url": "", "slug": None, "status": "error",
                     "title": None, "metadata": None,
                     "error": f"Search failed: {exc}"}]

        for url in urls:
            result = self.ingest_url(url)
            results.append(result)

        return results

    # -- web fetch helpers --------------------------------------------------

    @staticmethod
    def _fetch_url(url: str) -> str:
        """Fetch URL content via httpx. Returns HTML string."""
        with httpx.Client(
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text

    @staticmethod
    def _search_web(query: str, max_results: int = 5) -> list[str]:
        """Search the web via DuckDuckGo HTML and return result URLs.

        This is a lightweight approach that does not require an API key.
        For production use, consider integrating a dedicated search API.
        """
        search_url = "https://html.duckduckgo.com/html/"
        with httpx.Client(
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            resp = client.post(search_url, data={"q": query})
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        urls: list[str] = []
        for link in soup.select("a.result__a"):
            href = link.get("href", "")
            if href and href.startswith("http"):
                urls.append(href)
                if len(urls) >= max_results:
                    break
        return urls

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


# ---------------------------------------------------------------------------
# HTML content extraction (module-level helpers)
# ---------------------------------------------------------------------------


def extract_html_content(html: str, source_url: str = "") -> dict[str, Any]:
    """Extract clean text and metadata from an HTML document.

    Returns::

        {
            "title": str | None,
            "description": str | None,
            "canonical_url": str | None,
            "publish_date": str | None,
            "text": str,
        }
    """
    soup = BeautifulSoup(html, "lxml")

    # -- Extract metadata before stripping tags ----------------------------
    title = _extract_title(soup)
    description = _extract_meta(soup, "description")
    canonical_url = _extract_canonical(soup) or source_url
    publish_date = _extract_publish_date(soup)

    # -- Strip unwanted tags -----------------------------------------------
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Strip HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # -- Extract clean text ------------------------------------------------
    text = soup.get_text(separator="\n", strip=True)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return {
        "title": title,
        "description": description,
        "canonical_url": canonical_url,
        "publish_date": publish_date,
        "text": text.strip(),
    }


def _extract_title(soup: BeautifulSoup) -> str | None:
    """Extract page title from <title> or og:title."""
    # Try <title> first
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        return title_tag.string.strip()

    # Fallback: og:title
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return og["content"].strip()

    # Fallback: first <h1>
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    return None


def _extract_meta(soup: BeautifulSoup, name: str) -> str | None:
    """Extract a <meta name=...> content value."""
    tag = soup.find("meta", attrs={"name": name})
    if tag and tag.get("content"):
        return tag["content"].strip()
    # Also check property (og:description)
    tag = soup.find("meta", attrs={"property": f"og:{name}"})
    if tag and tag.get("content"):
        return tag["content"].strip()
    return None


def _extract_canonical(soup: BeautifulSoup) -> str | None:
    """Extract canonical URL from <link rel=canonical>."""
    link = soup.find("link", attrs={"rel": "canonical"})
    if link and link.get("href"):
        return link["href"].strip()
    return None


def _extract_publish_date(soup: BeautifulSoup) -> str | None:
    """Extract publish date from meta tags or JSON-LD."""
    # Try meta tags first
    for attr in ("article:published_time", "datePublished", "date"):
        tag = soup.find("meta", attrs={"property": attr}) or soup.find(
            "meta", attrs={"name": attr}
        )
        if tag and tag.get("content"):
            return tag["content"].strip()

    # Try JSON-LD
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
            if isinstance(data, dict):
                for key in ("datePublished", "dateCreated", "dateModified"):
                    if key in data:
                        return str(data[key])
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for key in ("datePublished", "dateCreated"):
                            if key in item:
                                return str(item[key])
        except (json.JSONDecodeError, TypeError):
            continue

    return None


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug[:80] or "untitled"


def _slug_from_url(url: str) -> str:
    """Generate a readable slug from a URL path."""
    if not url:
        return ""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if path:
        # Use the last path segment
        segment = path.split("/")[-1]
        # Remove file extensions
        segment = re.sub(r"\.\w+$", "", segment)
        if segment:
            return _slugify(segment)
    # Fall back to hostname
    hostname = parsed.hostname or ""
    return _slugify(hostname) or "web-clip"
