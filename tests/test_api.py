"""Tests for the Saido Agent REST API.

Validates:
  - Health check returns 200
  - All endpoints return correct response shapes
  - Auth rejects missing/invalid API keys
  - Rate limiting blocks excessive requests
  - Tenant isolation: two tenants cannot access each other's docs
  - File upload ingest works
  - Query returns SaidoQueryResult shape
  - Search returns results
  - SSE streaming produces events
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Mock objects — we need to mock SaidoAgent and its dependencies so tests
# run without SmartRAG or LLM providers installed.
# ---------------------------------------------------------------------------

@dataclass
class _MockIngestResult:
    slug: str = "test-doc"
    title: str = "Test Document"
    status: str = "created"
    children: list = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class _MockCitation:
    slug: str = "ref-article"
    title: str = "Reference Article"
    excerpt: str = "some excerpt"
    verified: bool = True


@dataclass
class _MockQueryResult:
    answer: str = "This is the answer."
    citations: list = field(default_factory=lambda: [_MockCitation()])
    confidence: str = "high"
    retrieval_stats: dict = field(default_factory=lambda: {"document_count": 5})
    tokens_used: int = 150
    provider: str = "mock/model"


@dataclass
class _MockSearchResult:
    slug: str = "found-doc"
    title: str = "Found Document"
    summary: str = "A summary"
    score: float = 0.95
    snippet: str = "A relevant snippet"


@dataclass
class _MockStoreStats:
    document_count: int = 10
    category_count: int = 3
    concept_count: int = 5
    total_size_bytes: int = 1024


@dataclass
class _MockAgentResult:
    output: str = "Agent completed the task."
    tool_calls: list = field(default_factory=list)
    tokens_used: int = 200


@dataclass
class _MockDocument:
    slug: str = "test-doc"
    title: str = "Test Document"
    body: str = "Full body content of the document."


class _MockBridge:
    """Mock KnowledgeBridge with all methods used by routes."""

    def __init__(self):
        self._articles = {}

    @property
    def available(self):
        return True

    def list_articles(self):
        return [
            (slug, doc.title, "Summary of " + slug)
            for slug, doc in self._articles.items()
        ]

    def read_article(self, slug):
        return self._articles.get(slug)

    def read_article_frontmatter(self, slug):
        if slug in self._articles:
            return {"categories": ["test"], "updated": "2026-01-01"}
        return None

    def add_article(self, slug, title, body):
        self._articles[slug] = _MockDocument(slug=slug, title=title, body=body)


class _MockAgent:
    """Mock SaidoAgent with all public methods used by routes."""

    def __init__(self, knowledge_dir="./knowledge"):
        self._bridge = _MockBridge()
        self._stats = _MockStoreStats()

    @property
    def bridge(self):
        return self._bridge

    @property
    def stats(self):
        return self._stats

    def ingest(self, path):
        return _MockIngestResult()

    def query(self, question, context=None):
        return _MockQueryResult()

    def search(self, query, top_k=5):
        return [_MockSearchResult()]

    def run(self, instruction):
        return _MockAgentResult()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_auth_state(tmp_path, monkeypatch):
    """Isolate auth state to tmp_path for every test."""
    saido_dir = tmp_path / ".saido_agent"
    saido_dir.mkdir()

    import saido_agent.api.auth as auth_mod
    monkeypatch.setattr(auth_mod, "_SAIDO_DIR", saido_dir)
    monkeypatch.setattr(auth_mod, "_API_KEYS_FILE", saido_dir / "api_keys.json")
    monkeypatch.setattr(auth_mod, "_JWT_SECRET_FILE", saido_dir / "jwt_secret")
    auth_mod.reset_rate_limits()
    yield


@pytest.fixture(autouse=True)
def _mock_agent_factory(monkeypatch):
    """Replace get_agent_for_tenant with a mock factory."""
    _agents = {}

    def _factory(tenant_id):
        if tenant_id not in _agents:
            _agents[tenant_id] = _MockAgent()
        return _agents[tenant_id]

    import saido_agent.api.server as server_mod
    monkeypatch.setattr(server_mod, "get_agent_for_tenant", _factory)
    yield _agents


@pytest.fixture()
def api_key():
    """Create and return a valid API key for tenant 'test-tenant'."""
    from saido_agent.api.auth import create_api_key
    return create_api_key("test-tenant")


@pytest.fixture()
def client():
    """FastAPI test client."""
    from saido_agent.api.server import app
    return TestClient(app)


@pytest.fixture()
def authed_client(client, api_key):
    """A wrapper that injects the API key header on every request."""
    return _AuthedClient(client, api_key)


class _AuthedClient:
    """Thin wrapper that injects X-API-Key on every request."""

    def __init__(self, client: TestClient, api_key: str):
        self._client = client
        self._headers = {"X-API-Key": api_key}

    def get(self, url, **kwargs):
        headers = {**self._headers, **kwargs.pop("headers", {})}
        return self._client.get(url, headers=headers, **kwargs)

    def post(self, url, **kwargs):
        headers = {**self._headers, **kwargs.pop("headers", {})}
        return self._client.post(url, headers=headers, **kwargs)


# ===========================================================================
# Tests
# ===========================================================================


class TestHealthCheck:
    """Health check endpoint."""

    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestAuth:
    """Authentication and authorization."""

    def test_missing_api_key_returns_401(self, client):
        resp = client.post("/v1/query", json={"question": "hello"})
        assert resp.status_code == 401

    def test_invalid_api_key_returns_401(self, client):
        resp = client.post(
            "/v1/query",
            json={"question": "hello"},
            headers={"X-API-Key": "sk-saido-bogus"},
        )
        assert resp.status_code == 401

    def test_valid_api_key_succeeds(self, authed_client):
        resp = authed_client.post("/v1/query", json={"question": "hello"})
        assert resp.status_code == 200

    def test_jwt_token_exchange(self, client, api_key):
        # Exchange API key for JWT
        resp = client.post("/v1/auth/token", json={"api_key": api_key})
        assert resp.status_code == 200
        token = resp.json()["token"]
        assert resp.json()["tenant_id"] == "test-tenant"

        # Use JWT for a request
        resp = client.post(
            "/v1/query",
            json={"question": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_invalid_jwt_returns_401(self, client):
        resp = client.post(
            "/v1/query",
            json={"question": "hello"},
            headers={"Authorization": "Bearer invalid.jwt.token"},
        )
        assert resp.status_code == 401

    def test_create_api_key_endpoint(self, client, api_key):
        resp = client.post(
            "/v1/auth/keys",
            json={"tenant_id": "new-tenant"},
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key"].startswith("sk-saido-")
        assert data["tenant_id"] == "new-tenant"

    def test_create_api_key_requires_auth(self, client):
        resp = client.post(
            "/v1/auth/keys",
            json={"tenant_id": "new-tenant"},
        )
        assert resp.status_code == 401


class TestRateLimiting:
    """Per-tenant rate limiting."""

    def test_rate_limit_blocks_excess(self, client):
        from saido_agent.api.auth import create_api_key

        # Create a key with a very low rate limit
        key = create_api_key("rate-test", rate_limit=3)
        headers = {"X-API-Key": key}

        # First 3 should succeed
        for _ in range(3):
            resp = client.post(
                "/v1/query",
                json={"question": "hello"},
                headers=headers,
            )
            assert resp.status_code == 200

        # 4th should be rate-limited
        resp = client.post(
            "/v1/query",
            json={"question": "hello"},
            headers=headers,
        )
        assert resp.status_code == 429


class TestTenantIsolation:
    """Verify tenants cannot access each other's data."""

    def test_tenants_have_separate_stores(self, client, _mock_agent_factory):
        from saido_agent.api.auth import create_api_key

        key_a = create_api_key("tenant-a")
        key_b = create_api_key("tenant-b")

        # Trigger lazy agent creation by making a request for each tenant
        client.get("/v1/stats", headers={"X-API-Key": key_a})
        client.get("/v1/stats", headers={"X-API-Key": key_b})

        # Tenant A adds a document to their bridge
        agent_a = _mock_agent_factory["tenant-a"]
        agent_a.bridge.add_article("doc-a", "Tenant A Doc", "Secret A content")

        # Tenant B adds a different document
        agent_b = _mock_agent_factory["tenant-b"]
        agent_b.bridge.add_article("doc-b", "Tenant B Doc", "Secret B content")

        # Tenant A sees only their doc
        resp = client.get("/v1/documents", headers={"X-API-Key": key_a})
        assert resp.status_code == 200
        slugs = [d["slug"] for d in resp.json()]
        assert "doc-a" in slugs
        assert "doc-b" not in slugs

        # Tenant B sees only their doc
        resp = client.get("/v1/documents", headers={"X-API-Key": key_b})
        assert resp.status_code == 200
        slugs = [d["slug"] for d in resp.json()]
        assert "doc-b" in slugs
        assert "doc-a" not in slugs

    def test_tenant_cannot_read_other_tenant_doc(self, client, _mock_agent_factory):
        from saido_agent.api.auth import create_api_key

        key_a = create_api_key("tenant-x")
        key_b = create_api_key("tenant-y")

        # Trigger lazy agent creation
        client.get("/v1/stats", headers={"X-API-Key": key_a})
        client.get("/v1/stats", headers={"X-API-Key": key_b})

        # Tenant X has a doc
        agent_x = _mock_agent_factory["tenant-x"]
        agent_x.bridge.add_article("secret-doc", "Secret", "Top secret")

        # Tenant Y should get 404 for that doc
        resp = client.get(
            "/v1/documents/secret-doc",
            headers={"X-API-Key": key_b},
        )
        assert resp.status_code == 404


class TestIngest:
    """Ingest endpoints."""

    def test_ingest_json(self, authed_client):
        resp = authed_client.post(
            "/v1/ingest",
            json={
                "content": "# Hello World\nSome content here.",
                "filename": "hello.md",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "slug" in data
        assert "status" in data
        assert data["status"] in ("created", "updated", "duplicate", "failed")

    def test_ingest_file_upload(self, client, api_key):
        resp = client.post(
            "/v1/ingest/upload",
            files={"file": ("test.md", b"# Test Upload\nContent here.", "text/markdown")},
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "slug" in data
        assert "status" in data


class TestQuery:
    """Query endpoint."""

    def test_query_returns_correct_shape(self, authed_client):
        resp = authed_client.post(
            "/v1/query",
            json={"question": "What is Saido Agent?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "citations" in data
        assert "confidence" in data
        assert "retrieval_stats" in data
        assert "tokens_used" in data

    def test_query_with_top_k(self, authed_client):
        resp = authed_client.post(
            "/v1/query",
            json={"question": "What is Saido Agent?", "top_k": 3},
        )
        assert resp.status_code == 200

    def test_query_sse_streaming(self, client, api_key):
        """SSE streaming produces events with correct structure."""
        resp = client.post(
            "/v1/query",
            json={"question": "What is Saido Agent?"},
            headers={
                "X-API-Key": api_key,
                "Accept": "text/event-stream",
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE events
        events = []
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        assert len(events) >= 2  # At least one token + done
        # Last event should be "done"
        assert events[-1]["type"] == "done"
        assert "result" in events[-1]
        assert "answer" in events[-1]["result"]

        # Earlier events should be "token" type
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) >= 1


class TestDocuments:
    """Document listing and retrieval."""

    def test_list_documents(self, authed_client, _mock_agent_factory):
        # Trigger lazy agent creation
        authed_client.get("/v1/stats")

        # Add a doc to the tenant's bridge
        agent = _mock_agent_factory["test-tenant"]
        agent.bridge.add_article("my-doc", "My Document", "Body content")

        resp = authed_client.get("/v1/documents")
        assert resp.status_code == 200
        docs = resp.json()
        assert isinstance(docs, list)
        assert len(docs) >= 1
        doc = docs[0]
        assert "slug" in doc
        assert "title" in doc
        assert "summary" in doc

    def test_get_document_by_slug(self, authed_client, _mock_agent_factory):
        # Trigger lazy agent creation
        authed_client.get("/v1/stats")

        agent = _mock_agent_factory["test-tenant"]
        agent.bridge.add_article("detail-doc", "Detail Doc", "Full body here")

        resp = authed_client.get("/v1/documents/detail-doc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "detail-doc"
        assert data["body"] == "Full body here"

    def test_get_missing_document_returns_404(self, authed_client):
        resp = authed_client.get("/v1/documents/nonexistent")
        assert resp.status_code == 404


class TestSearch:
    """Search endpoint."""

    def test_search_returns_results(self, authed_client):
        resp = authed_client.get("/v1/search", params={"q": "test query"})
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)
        assert len(results) >= 1
        r = results[0]
        assert "slug" in r
        assert "title" in r
        assert "score" in r
        assert "snippet" in r

    def test_search_requires_query(self, authed_client):
        resp = authed_client.get("/v1/search")
        assert resp.status_code == 422  # Validation error


class TestStats:
    """Stats endpoint."""

    def test_stats_returns_correct_shape(self, authed_client):
        resp = authed_client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "document_count" in data
        assert "category_count" in data
        assert "total_size_bytes" in data


class TestAgent:
    """Agent endpoint."""

    def test_agent_returns_correct_shape(self, authed_client):
        resp = authed_client.post(
            "/v1/agent",
            json={"instruction": "Summarize the knowledge base"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "output" in data
        assert "tool_calls" in data
        assert "tokens_used" in data
