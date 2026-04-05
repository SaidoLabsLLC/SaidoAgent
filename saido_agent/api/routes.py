"""API routes for Saido Agent REST API.

All routes are mounted under ``/v1/`` via the ``v1_router`` APIRouter.
Each route receives a ``tenant_id`` from the ``get_current_tenant``
dependency, which scopes all operations to the tenant's knowledge store.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from saido_agent.api.auth import (
    create_api_key,
    create_jwt_token,
    get_current_tenant,
    get_tenant_knowledge_dir,
    verify_api_key,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for POST /v1/query."""

    question: str
    context: Optional[dict] = None
    top_k: int = Field(default=5, ge=1, le=50)


class IngestRequest(BaseModel):
    """Request body for POST /v1/ingest (JSON mode)."""

    content: str
    filename: str
    metadata: Optional[dict] = None


class AgentRequest(BaseModel):
    """Request body for POST /v1/agent."""

    instruction: str


class TokenRequest(BaseModel):
    """Request body for POST /v1/auth/token."""

    api_key: str


class CreateKeyRequest(BaseModel):
    """Request body for POST /v1/auth/keys."""

    tenant_id: str
    rate_limit: int = 60


# Response models

class IngestResponse(BaseModel):
    slug: str
    title: str
    status: str
    children: list[str] = []
    error: Optional[str] = None


class CitationResponse(BaseModel):
    slug: str
    title: str
    excerpt: str = ""
    verified: bool = True


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse] = []
    confidence: str = "medium"
    retrieval_stats: dict = {}
    tokens_used: int = 0
    provider: str = ""


class DocumentSummary(BaseModel):
    slug: str
    title: str
    summary: str
    categories: list[str] = []
    updated: Optional[str] = None


class DocumentDetail(BaseModel):
    slug: str
    title: str
    body: str
    summary: str = ""
    categories: list[str] = []
    frontmatter: dict = {}


class SearchResultResponse(BaseModel):
    slug: str
    title: str
    summary: str
    score: float
    snippet: str


class StatsResponse(BaseModel):
    document_count: int
    category_count: int = 0
    concept_count: int = 0
    total_size_bytes: int = 0


class AgentResponse(BaseModel):
    output: str
    tool_calls: list[dict] = []
    tokens_used: int = 0


class TokenResponse(BaseModel):
    token: str
    tenant_id: str
    expires_in: int = 3600


class KeyCreatedResponse(BaseModel):
    api_key: str
    tenant_id: str
    message: str = "Store this key securely. It will not be shown again."


# ---------------------------------------------------------------------------
# Helper: build a SaidoAgent scoped to a tenant
# ---------------------------------------------------------------------------

def _get_tenant_agent(tenant_id: str) -> Any:
    """Build a SaidoAgent scoped to the tenant's knowledge directory.

    Uses a module-level cache to avoid re-initializing agents on every
    request. The cache is bounded and will be cleared on shutdown.
    """
    from saido_agent.api.server import get_agent_for_tenant
    return get_agent_for_tenant(tenant_id)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

v1_router = APIRouter(prefix="/v1", tags=["v1"])


# -- Auth routes (no tenant dependency) ------------------------------------

@v1_router.post("/auth/token", response_model=TokenResponse, tags=["auth"])
async def create_token(body: TokenRequest):
    """Exchange an API key for a JWT session token."""
    entry = verify_api_key(body.api_key)
    if entry is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    tenant_id = entry["tenant_id"]
    token = create_jwt_token(tenant_id)
    return TokenResponse(token=token, tenant_id=tenant_id)


@v1_router.post("/auth/keys", response_model=KeyCreatedResponse, tags=["auth"])
async def create_key(body: CreateKeyRequest):
    """Create a new API key for a tenant.

    In production this endpoint should be admin-only.
    """
    key = create_api_key(body.tenant_id, rate_limit=body.rate_limit)
    return KeyCreatedResponse(api_key=key, tenant_id=body.tenant_id)


# -- Ingest ----------------------------------------------------------------

@v1_router.post("/ingest", response_model=IngestResponse)
async def ingest_json(
    body: IngestRequest,
    tenant_id: str = Depends(get_current_tenant),
):
    """Ingest content from a JSON payload."""
    agent = _get_tenant_agent(tenant_id)

    # Write content to a temp file and ingest it
    suffix = Path(body.filename).suffix or ".md"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        prefix="saido_ingest_",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(body.content)
        tmp_path = f.name

    try:
        result = await asyncio.to_thread(agent.ingest, tmp_path)
        return IngestResponse(
            slug=result.slug,
            title=result.title,
            status=result.status,
            children=result.children,
            error=result.error,
        )
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@v1_router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile,
    tenant_id: str = Depends(get_current_tenant),
):
    """Ingest content from a file upload."""
    agent = _get_tenant_agent(tenant_id)

    suffix = Path(file.filename or "upload.md").suffix or ".md"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        prefix="saido_upload_",
        delete=False,
    ) as f:
        content = await file.read()
        f.write(content)
        tmp_path = f.name

    try:
        result = await asyncio.to_thread(agent.ingest, tmp_path)
        return IngestResponse(
            slug=result.slug,
            title=result.title,
            status=result.status,
            children=result.children,
            error=result.error,
        )
    except Exception as exc:
        logger.exception("Upload ingest failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# -- Query -----------------------------------------------------------------

@v1_router.post("/query", response_model=QueryResponse)
async def query_knowledge(
    body: QueryRequest,
    request: Request,
    tenant_id: str = Depends(get_current_tenant),
):
    """Query the knowledge store. Supports SSE streaming via Accept header."""
    accept = request.headers.get("Accept", "")

    if "text/event-stream" in accept:
        return _stream_query(body, tenant_id)

    agent = _get_tenant_agent(tenant_id)
    result = await asyncio.to_thread(
        agent.query, body.question, body.context
    )

    citations = [
        CitationResponse(
            slug=c.slug,
            title=c.title,
            excerpt=c.excerpt,
            verified=c.verified,
        )
        for c in result.citations
    ]

    return QueryResponse(
        answer=result.answer,
        citations=citations,
        confidence=result.confidence,
        retrieval_stats=result.retrieval_stats,
        tokens_used=result.tokens_used,
        provider=result.provider,
    )


def _stream_query(body: QueryRequest, tenant_id: str) -> StreamingResponse:
    """Return an SSE streaming response for a query."""

    async def event_generator():
        agent = _get_tenant_agent(tenant_id)

        # Run the query in a thread (it's synchronous)
        result = await asyncio.to_thread(
            agent.query, body.question, body.context
        )

        # Stream the answer token-by-token (simulated since the SDK
        # returns a complete answer; a future version will support
        # true streaming from the LLM provider)
        words = result.answer.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            event_data = json.dumps({"type": "token", "content": token})
            yield f"data: {event_data}\n\n"

        # Final event with full result
        citations = [
            {
                "slug": c.slug,
                "title": c.title,
                "excerpt": c.excerpt,
                "verified": c.verified,
            }
            for c in result.citations
        ]
        done_data = json.dumps({
            "type": "done",
            "result": {
                "answer": result.answer,
                "citations": citations,
                "confidence": result.confidence,
                "retrieval_stats": result.retrieval_stats,
                "tokens_used": result.tokens_used,
                "provider": result.provider,
            },
        })
        yield f"data: {done_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# -- Documents -------------------------------------------------------------

@v1_router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    tenant_id: str = Depends(get_current_tenant),
):
    """List all articles in the knowledge store."""
    agent = _get_tenant_agent(tenant_id)
    articles = await asyncio.to_thread(agent.bridge.list_articles)

    results = []
    for slug, title, summary in articles:
        # Try to get frontmatter for categories and updated date
        fm = agent.bridge.read_article_frontmatter(slug)
        categories = fm.get("categories", []) if fm else []
        updated = fm.get("updated", None) if fm else None
        results.append(DocumentSummary(
            slug=slug,
            title=title,
            summary=summary,
            categories=categories,
            updated=updated,
        ))
    return results


@v1_router.get("/documents/{slug}", response_model=DocumentDetail)
async def get_document(
    slug: str,
    tenant_id: str = Depends(get_current_tenant),
):
    """Read a specific article with full body and metadata."""
    agent = _get_tenant_agent(tenant_id)
    doc = await asyncio.to_thread(agent.bridge.read_article, slug)

    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{slug}' not found")

    fm = agent.bridge.read_article_frontmatter(slug) or {}
    return DocumentDetail(
        slug=slug,
        title=getattr(doc, "title", slug),
        body=doc.body,
        summary=fm.get("summary", ""),
        categories=fm.get("categories", []),
        frontmatter=fm,
    )


# -- Search ----------------------------------------------------------------

@v1_router.get("/search", response_model=list[SearchResultResponse])
async def search_knowledge(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=100),
    tenant_id: str = Depends(get_current_tenant),
):
    """Search the knowledge store without generating an LLM answer."""
    agent = _get_tenant_agent(tenant_id)
    results = await asyncio.to_thread(agent.search, q, top_k)
    return [
        SearchResultResponse(
            slug=r.slug,
            title=r.title,
            summary=r.summary,
            score=r.score,
            snippet=r.snippet,
        )
        for r in results
    ]


# -- Stats -----------------------------------------------------------------

@v1_router.get("/stats", response_model=StatsResponse)
async def get_stats(
    tenant_id: str = Depends(get_current_tenant),
):
    """Knowledge store statistics."""
    agent = _get_tenant_agent(tenant_id)
    stats = agent.stats
    return StatsResponse(
        document_count=stats.document_count,
        category_count=stats.category_count,
        concept_count=stats.concept_count,
        total_size_bytes=stats.total_size_bytes,
    )


# -- Agent -----------------------------------------------------------------

@v1_router.post("/agent", response_model=AgentResponse)
async def run_agent(
    body: AgentRequest,
    tenant_id: str = Depends(get_current_tenant),
):
    """Run a full agent loop with tool use."""
    agent = _get_tenant_agent(tenant_id)
    result = await asyncio.to_thread(agent.run, body.instruction)
    return AgentResponse(
        output=result.output,
        tool_calls=result.tool_calls,
        tokens_used=result.tokens_used,
    )
