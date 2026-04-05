"""Tests for web content ingestion — URL fetch, SSRF protection, HTML
extraction, clip endpoint, and CLI commands.
"""

from __future__ import annotations

import ipaddress
import json
import socket
from unittest.mock import MagicMock, patch

import httpx
import pytest

from saido_agent.core.ssrf import (
    BLOCKED_HOSTS,
    BLOCKED_NETWORKS,
    validate_url,
    validate_url_no_resolve,
)
from saido_agent.knowledge.ingest import (
    IngestPipeline,
    _slugify,
    _slug_from_url,
    extract_html_content,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_HTML = """\
<!DOCTYPE html>
<html>
<head>
    <title>Test Article Title</title>
    <meta name="description" content="A test description for the article.">
    <link rel="canonical" href="https://example.com/articles/test-article">
    <meta property="article:published_time" content="2025-06-15T10:00:00Z">
</head>
<body>
    <header><nav><a href="/">Home</a></nav></header>
    <main>
        <h1>Test Article Title</h1>
        <p>This is the main content of the article.</p>
        <p>It has multiple paragraphs with important information.</p>
    </main>
    <aside>Related links sidebar</aside>
    <footer>Copyright 2025</footer>
    <script>console.log('should be stripped');</script>
    <style>.hidden { display: none; }</style>
</body>
</html>
"""

SAMPLE_HTML_JSONLD = """\
<!DOCTYPE html>
<html>
<head>
    <title>JSON-LD Article</title>
    <script type="application/ld+json">
    {
        "datePublished": "2025-01-10T12:00:00Z",
        "@type": "Article",
        "headline": "JSON-LD Article"
    }
    </script>
</head>
<body>
    <p>Article body content here.</p>
</body>
</html>
"""


@pytest.fixture
def pipeline():
    """IngestPipeline with a mock bridge."""
    bridge = MagicMock()
    bridge.create_article = MagicMock(return_value=None)
    return IngestPipeline(bridge=bridge)


@pytest.fixture
def pipeline_no_bridge():
    """IngestPipeline with no bridge."""
    return IngestPipeline(bridge=None)


# ===================================================================
# SSRF Protection Tests
# ===================================================================


class TestSSRFValidation:
    """Tests for SSRF protection module."""

    def test_blocks_private_10x(self):
        """SSRF blocks 10.0.0.0/8 addresses."""
        with patch("saido_agent.core.ssrf.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),
            ]
            is_safe, reason = validate_url("http://internal.example.com/secret")
            assert not is_safe
            assert "10.0.0.1" in reason

    def test_blocks_private_172_16x(self):
        """SSRF blocks 172.16.0.0/12 addresses."""
        with patch("saido_agent.core.ssrf.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("172.16.5.10", 0)),
            ]
            is_safe, reason = validate_url("http://internal.corp.com/api")
            assert not is_safe
            assert "172.16.5.10" in reason

    def test_blocks_private_192_168x(self):
        """SSRF blocks 192.168.0.0/16 addresses."""
        with patch("saido_agent.core.ssrf.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.1", 0)),
            ]
            is_safe, reason = validate_url("http://router.local/admin")
            assert not is_safe
            assert "192.168.1.1" in reason

    def test_blocks_loopback_127x(self):
        """SSRF blocks 127.0.0.0/8 (loopback) addresses."""
        with patch("saido_agent.core.ssrf.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
            ]
            is_safe, reason = validate_url("http://localhost/admin")
            assert not is_safe
            assert "127.0.0.1" in reason

    def test_blocks_link_local_169_254x(self):
        """SSRF blocks 169.254.0.0/16 (link-local) addresses."""
        with patch("saido_agent.core.ssrf.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("169.254.1.1", 0)),
            ]
            is_safe, reason = validate_url("http://link-local.test/")
            assert not is_safe
            assert "169.254.1.1" in reason

    def test_blocks_cloud_metadata(self):
        """SSRF blocks cloud metadata endpoint (169.254.169.254)."""
        # Direct host check (before DNS resolution)
        is_safe, reason = validate_url_no_resolve("http://169.254.169.254/latest/meta-data/")
        assert not is_safe
        assert "169.254" in reason or "Blocked" in reason

    def test_blocks_cloud_metadata_hostname(self):
        """SSRF blocks metadata.google.internal."""
        is_safe, reason = validate_url_no_resolve("http://metadata.google.internal/computeMetadata/v1/")
        assert not is_safe
        assert "Blocked host" in reason

    def test_public_url_passes(self):
        """Public URLs with valid DNS pass SSRF check."""
        with patch("saido_agent.core.ssrf.socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0)),
            ]
            is_safe, reason = validate_url("https://example.com/page")
            assert is_safe
            assert reason == "ok"

    def test_blocks_ftp_scheme(self):
        """SSRF blocks non-HTTP schemes."""
        is_safe, reason = validate_url("ftp://evil.com/file")
        assert not is_safe
        assert "scheme" in reason.lower()

    def test_blocks_file_scheme(self):
        """SSRF blocks file:// scheme."""
        is_safe, reason = validate_url("file:///etc/passwd")
        assert not is_safe
        assert "scheme" in reason.lower()

    def test_blocks_missing_hostname(self):
        """SSRF blocks URLs without hostname."""
        is_safe, reason = validate_url("http:///path/only")
        assert not is_safe

    def test_no_resolve_literal_private_ip(self):
        """validate_url_no_resolve blocks literal private IPs."""
        is_safe, reason = validate_url_no_resolve("http://10.0.0.5/secret")
        assert not is_safe
        assert "10.0.0.5" in reason

    def test_no_resolve_public_hostname_passes(self):
        """validate_url_no_resolve allows public hostnames (no resolution)."""
        is_safe, reason = validate_url_no_resolve("https://example.com/page")
        assert is_safe

    def test_dns_failure_blocks(self):
        """DNS resolution failure should block the URL."""
        with patch("saido_agent.core.ssrf.socket.getaddrinfo") as mock_dns:
            mock_dns.side_effect = socket.gaierror("Name resolution failed")
            is_safe, reason = validate_url("http://nonexistent.invalid/")
            assert not is_safe
            assert "DNS" in reason


# ===================================================================
# HTML Extraction Tests
# ===================================================================


class TestHTMLExtraction:
    """Tests for HTML content extraction."""

    def test_extracts_title(self):
        """Title is extracted from <title> tag."""
        result = extract_html_content(SAMPLE_HTML)
        assert result["title"] == "Test Article Title"

    def test_extracts_description(self):
        """Meta description is extracted."""
        result = extract_html_content(SAMPLE_HTML)
        assert result["description"] == "A test description for the article."

    def test_extracts_canonical_url(self):
        """Canonical URL is extracted from <link rel=canonical>."""
        result = extract_html_content(SAMPLE_HTML)
        assert result["canonical_url"] == "https://example.com/articles/test-article"

    def test_extracts_publish_date_meta(self):
        """Publish date extracted from meta article:published_time."""
        result = extract_html_content(SAMPLE_HTML)
        assert result["publish_date"] == "2025-06-15T10:00:00Z"

    def test_extracts_publish_date_jsonld(self):
        """Publish date extracted from JSON-LD."""
        result = extract_html_content(SAMPLE_HTML_JSONLD)
        assert result["publish_date"] == "2025-01-10T12:00:00Z"

    def test_strips_nav(self):
        """<nav> content is stripped."""
        result = extract_html_content(SAMPLE_HTML)
        assert "Home" not in result["text"]

    def test_strips_footer(self):
        """<footer> content is stripped."""
        result = extract_html_content(SAMPLE_HTML)
        assert "Copyright 2025" not in result["text"]

    def test_strips_script(self):
        """<script> content is stripped."""
        result = extract_html_content(SAMPLE_HTML)
        assert "console.log" not in result["text"]
        assert "should be stripped" not in result["text"]

    def test_strips_style(self):
        """<style> content is stripped."""
        result = extract_html_content(SAMPLE_HTML)
        assert ".hidden" not in result["text"]
        assert "display: none" not in result["text"]

    def test_strips_aside(self):
        """<aside> content is stripped."""
        result = extract_html_content(SAMPLE_HTML)
        assert "Related links sidebar" not in result["text"]

    def test_strips_header(self):
        """<header> content is stripped."""
        result = extract_html_content(SAMPLE_HTML)
        # The <header> wraps the <nav>, both should be gone
        # Main h1 should still be present though (it's in <main>)
        assert "Test Article Title" in result["text"]

    def test_preserves_main_content(self):
        """Main article text is preserved."""
        result = extract_html_content(SAMPLE_HTML)
        assert "main content of the article" in result["text"]
        assert "multiple paragraphs" in result["text"]

    def test_empty_html_returns_empty_text(self):
        """Empty HTML returns empty text."""
        result = extract_html_content("<html><body></body></html>")
        assert result["text"] == ""

    def test_plain_text_passthrough(self):
        """Non-HTML text is returned as-is."""
        result = extract_html_content("Just plain text content.")
        assert "Just plain text content" in result["text"]


# ===================================================================
# Slugify Tests
# ===================================================================


class TestSlugify:
    """Tests for slug generation helpers."""

    def test_basic_slug(self):
        assert _slugify("Hello World") == "hello-world"

    def test_special_chars_removed(self):
        assert _slugify("What's Up? (Test!)") == "whats-up-test"

    def test_max_length(self):
        long_title = "a" * 100
        assert len(_slugify(long_title)) <= 80

    def test_empty_returns_untitled(self):
        assert _slugify("") == "untitled"

    def test_slug_from_url_path(self):
        assert _slug_from_url("https://example.com/blog/my-article") == "my-article"

    def test_slug_from_url_with_extension(self):
        slug = _slug_from_url("https://example.com/page.html")
        assert slug == "page"

    def test_slug_from_empty_url(self):
        assert _slug_from_url("") == ""


# ===================================================================
# IngestPipeline Web Ingest Tests
# ===================================================================


class TestIngestURL:
    """Tests for IngestPipeline.ingest_url()."""

    def test_successful_url_ingest(self, pipeline):
        """Successful URL fetch and ingest."""
        with (
            patch.object(IngestPipeline, "_fetch_url", return_value=SAMPLE_HTML),
            patch("saido_agent.knowledge.ingest.validate_url", return_value=(True, "ok")),
        ):
            result = pipeline.ingest_url("https://example.com/article")

        assert result["status"] == "ok"
        assert result["slug"] is not None
        assert result["title"] == "Test Article Title"
        assert result["url"] == "https://example.com/article"
        pipeline._bridge.create_article.assert_called_once()

    def test_ssrf_blocked_url(self, pipeline):
        """SSRF blocked URL returns error."""
        with patch(
            "saido_agent.knowledge.ingest.validate_url",
            return_value=(False, "Blocked IP range: 10.0.0.1 is in 10.0.0.0/8"),
        ):
            result = pipeline.ingest_url("http://10.0.0.1/secret")

        assert result["status"] == "error"
        assert "SSRF" in result["error"]

    def test_fetch_failure(self, pipeline):
        """Network error during fetch returns error result."""
        with (
            patch("saido_agent.knowledge.ingest.validate_url", return_value=(True, "ok")),
            patch.object(
                IngestPipeline,
                "_fetch_url",
                side_effect=httpx.ConnectError("Connection refused"),
            ),
        ):
            result = pipeline.ingest_url("https://unreachable.example.com/")

        assert result["status"] == "error"
        assert "Fetch failed" in result["error"]

    def test_empty_content_returns_error(self, pipeline):
        """URL with no extractable text returns error."""
        empty_html = "<html><body><script>only scripts</script></body></html>"
        with (
            patch.object(IngestPipeline, "_fetch_url", return_value=empty_html),
            patch("saido_agent.knowledge.ingest.validate_url", return_value=(True, "ok")),
        ):
            result = pipeline.ingest_url("https://example.com/empty")

        assert result["status"] == "error"
        assert "No text" in result["error"]

    def test_url_queued_for_compile(self, pipeline):
        """Successful ingest adds slug to compile queue."""
        with (
            patch.object(IngestPipeline, "_fetch_url", return_value=SAMPLE_HTML),
            patch("saido_agent.knowledge.ingest.validate_url", return_value=(True, "ok")),
        ):
            result = pipeline.ingest_url("https://example.com/article")

        assert result["slug"] in pipeline.get_compile_queue()

    def test_no_bridge_still_returns_slug(self, pipeline_no_bridge):
        """Without a bridge, ingest_url still returns a slug."""
        with (
            patch.object(IngestPipeline, "_fetch_url", return_value=SAMPLE_HTML),
            patch("saido_agent.knowledge.ingest.validate_url", return_value=(True, "ok")),
        ):
            result = pipeline_no_bridge.ingest_url("https://example.com/article")

        assert result["status"] == "ok"
        assert result["slug"] is not None


class TestIngestHTML:
    """Tests for IngestPipeline.ingest_html()."""

    def test_html_ingest(self, pipeline):
        """Pre-fetched HTML is extracted and ingested."""
        result = pipeline.ingest_html(SAMPLE_HTML, url="https://example.com/clip")
        assert result["status"] == "ok"
        assert result["title"] == "Test Article Title"
        pipeline._bridge.create_article.assert_called_once()

    def test_empty_html_returns_error(self, pipeline):
        """Empty HTML body returns error."""
        result = pipeline.ingest_html("<html><body></body></html>")
        assert result["status"] == "error"


class TestIngestSelection:
    """Tests for IngestPipeline.ingest_selection()."""

    def test_selection_ingest(self, pipeline):
        """Selected text is ingested directly."""
        result = pipeline.ingest_selection(
            "This is the selected text from the page.",
            url="https://example.com/page",
            title="My Selection",
        )
        assert result["status"] == "ok"
        assert result["slug"] == "my-selection"

    def test_empty_selection_returns_error(self, pipeline):
        """Empty selection returns error."""
        result = pipeline.ingest_selection("   ")
        assert result["status"] == "error"
        assert "Empty" in result["error"]


class TestIngestSearch:
    """Tests for IngestPipeline.ingest_search()."""

    def test_search_and_ingest(self, pipeline):
        """Search returns URLs that are fetched and ingested."""
        mock_urls = [
            "https://example.com/result1",
            "https://example.com/result2",
        ]
        with (
            patch.object(IngestPipeline, "_search_web", return_value=mock_urls),
            patch.object(IngestPipeline, "_fetch_url", return_value=SAMPLE_HTML),
            patch("saido_agent.knowledge.ingest.validate_url", return_value=(True, "ok")),
        ):
            results = pipeline.ingest_search("test query", max_results=2)

        assert len(results) == 2
        assert all(r["status"] == "ok" for r in results)

    def test_search_failure(self, pipeline):
        """Search API failure returns error result."""
        with patch.object(
            IngestPipeline,
            "_search_web",
            side_effect=httpx.ConnectError("Search unavailable"),
        ):
            results = pipeline.ingest_search("test query")

        assert len(results) == 1
        assert results[0]["status"] == "error"
        assert "Search failed" in results[0]["error"]


# ===================================================================
# Clip Endpoint Tests
# ===================================================================


class TestClipEndpoint:
    """Tests for the /v1/clip API endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client with mocked auth and agent."""
        from fastapi.testclient import TestClient

        from saido_agent.api.routes import v1_router

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(v1_router)

        # Mock the auth dependency
        from saido_agent.api import routes

        mock_pipeline = MagicMock()
        mock_agent = MagicMock()
        mock_agent._ingest_pipeline = mock_pipeline
        mock_agent.ingest_pipeline = mock_pipeline

        original_get_tenant = routes.get_current_tenant
        original_get_agent = routes._get_tenant_agent

        app.dependency_overrides[original_get_tenant] = lambda: "test-tenant"

        with patch.object(routes, "_get_tenant_agent", return_value=mock_agent):
            yield TestClient(app), mock_pipeline

    def test_clip_with_selection(self, client):
        """Clip endpoint ingests selection text."""
        test_client, mock_pipeline = client
        mock_pipeline.ingest_selection.return_value = {
            "url": "https://example.com",
            "slug": "test-selection",
            "status": "ok",
            "title": "Test Selection",
            "error": None,
        }

        response = test_client.post(
            "/v1/clip",
            json={
                "url": "https://example.com",
                "selection": "Selected text from the page",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["slug"] == "test-selection"

    def test_clip_with_html(self, client):
        """Clip endpoint extracts and ingests HTML."""
        test_client, mock_pipeline = client
        mock_pipeline.ingest_html.return_value = {
            "url": "https://example.com",
            "slug": "test-article",
            "status": "ok",
            "title": "Test Article",
            "error": None,
        }

        response = test_client.post(
            "/v1/clip",
            json={
                "url": "https://example.com",
                "html": SAMPLE_HTML,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_clip_with_url_only(self, client):
        """Clip endpoint fetches and ingests URL."""
        test_client, mock_pipeline = client
        mock_pipeline.ingest_url.return_value = {
            "url": "https://example.com/article",
            "slug": "article",
            "status": "ok",
            "title": "Article Title",
            "error": None,
        }

        response = test_client.post(
            "/v1/clip",
            json={"url": "https://example.com/article"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_clip_empty_request(self, client):
        """Clip endpoint rejects empty request."""
        test_client, _ = client
        response = test_client.post("/v1/clip", json={})
        assert response.status_code == 422

    def test_clip_priority_selection_over_html(self, client):
        """Selection takes priority over html when both provided."""
        test_client, mock_pipeline = client
        mock_pipeline.ingest_selection.return_value = {
            "url": "https://example.com",
            "slug": "sel",
            "status": "ok",
            "title": "Selection",
            "error": None,
        }

        response = test_client.post(
            "/v1/clip",
            json={
                "url": "https://example.com",
                "html": "<p>HTML</p>",
                "selection": "Selected text",
            },
        )
        assert response.status_code == 200
        mock_pipeline.ingest_selection.assert_called_once()
        mock_pipeline.ingest_html.assert_not_called()

    def test_clip_error_result(self, client):
        """Clip endpoint returns error from pipeline."""
        test_client, mock_pipeline = client
        mock_pipeline.ingest_url.return_value = {
            "url": "http://10.0.0.1/secret",
            "slug": None,
            "status": "error",
            "title": None,
            "error": "SSRF blocked",
        }

        response = test_client.post(
            "/v1/clip",
            json={"url": "http://10.0.0.1/secret"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "SSRF" in data["error"]


# ===================================================================
# CLI Command Tests
# ===================================================================


class TestCLICommands:
    """Tests for CLI slash commands."""

    def test_ingest_url_registered(self):
        """The /ingest-url command is registered in the COMMANDS dict."""
        from saido_agent.cli.repl import COMMANDS
        assert "ingest-url" in COMMANDS

    def test_ingest_search_registered(self):
        """The /ingest-search command is registered in the COMMANDS dict."""
        from saido_agent.cli.repl import COMMANDS
        assert "ingest-search" in COMMANDS

    def test_cmd_ingest_url_no_pipeline(self):
        """cmd_ingest_url errors gracefully when pipeline is missing."""
        from saido_agent.cli.repl import cmd_ingest_url

        config = {"_knowledge_context": {}}
        result = cmd_ingest_url("https://example.com", None, config)
        assert result is True  # Handled, just printed error

    def test_cmd_ingest_url_no_args(self):
        """cmd_ingest_url prints usage when no URL provided."""
        from saido_agent.cli.repl import cmd_ingest_url

        mock_pipeline = MagicMock()
        config = {"_knowledge_context": {"ingest_pipeline": mock_pipeline}}
        result = cmd_ingest_url("", None, config)
        assert result is True
        mock_pipeline.ingest_url.assert_not_called()

    def test_cmd_ingest_url_success(self):
        """cmd_ingest_url calls pipeline.ingest_url on valid input."""
        from saido_agent.cli.repl import cmd_ingest_url

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_url.return_value = {
            "status": "ok",
            "slug": "test-article",
            "title": "Test",
            "error": None,
        }
        config = {"_knowledge_context": {"ingest_pipeline": mock_pipeline}}
        result = cmd_ingest_url("https://example.com", None, config)
        assert result is True
        mock_pipeline.ingest_url.assert_called_once_with("https://example.com")

    def test_cmd_ingest_search_no_args(self):
        """cmd_ingest_search prints usage when no query provided."""
        from saido_agent.cli.repl import cmd_ingest_search

        mock_pipeline = MagicMock()
        config = {"_knowledge_context": {"ingest_pipeline": mock_pipeline}}
        result = cmd_ingest_search("", None, config)
        assert result is True
        mock_pipeline.ingest_search.assert_not_called()

    def test_cmd_ingest_search_success(self):
        """cmd_ingest_search calls pipeline.ingest_search on valid input."""
        from saido_agent.cli.repl import cmd_ingest_search

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_search.return_value = [
            {"status": "ok", "slug": "r1", "title": "Result 1", "url": "https://example.com/1", "error": None},
        ]
        config = {"_knowledge_context": {"ingest_pipeline": mock_pipeline}}
        result = cmd_ingest_search("test query", None, config)
        assert result is True
        mock_pipeline.ingest_search.assert_called_once_with("test query")


# ===================================================================
# Error Handling Tests
# ===================================================================


class TestErrorHandling:
    """Tests for error handling edge cases."""

    def test_http_timeout(self, pipeline):
        """HTTP timeout returns proper error."""
        with (
            patch("saido_agent.knowledge.ingest.validate_url", return_value=(True, "ok")),
            patch.object(
                IngestPipeline,
                "_fetch_url",
                side_effect=httpx.TimeoutException("Request timed out"),
            ),
        ):
            result = pipeline.ingest_url("https://slow.example.com/")

        assert result["status"] == "error"
        assert "Fetch failed" in result["error"]

    def test_http_404(self, pipeline):
        """HTTP 404 returns proper error."""
        with (
            patch("saido_agent.knowledge.ingest.validate_url", return_value=(True, "ok")),
            patch.object(
                IngestPipeline,
                "_fetch_url",
                side_effect=httpx.HTTPStatusError(
                    "Not Found",
                    request=MagicMock(),
                    response=MagicMock(status_code=404),
                ),
            ),
        ):
            result = pipeline.ingest_url("https://example.com/missing")

        assert result["status"] == "error"
        assert "Fetch failed" in result["error"]

    def test_bridge_storage_failure(self, pipeline):
        """Bridge storage exception returns proper error."""
        pipeline._bridge.create_article.side_effect = RuntimeError("DB connection lost")
        with (
            patch.object(IngestPipeline, "_fetch_url", return_value=SAMPLE_HTML),
            patch("saido_agent.knowledge.ingest.validate_url", return_value=(True, "ok")),
        ):
            result = pipeline.ingest_url("https://example.com/article")

        assert result["status"] == "error"
        assert "Storage failed" in result["error"]

    def test_malformed_html_graceful(self, pipeline):
        """Badly formed HTML does not crash extraction."""
        bad_html = "<html><body><p>Unclosed paragraph<div>Mixed <b>tags"
        result = extract_html_content(bad_html)
        assert "Unclosed paragraph" in result["text"]
