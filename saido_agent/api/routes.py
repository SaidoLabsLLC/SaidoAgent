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
    AuthContext,
    create_api_key,
    create_jwt_token,
    create_user_jwt_token,
    get_auth_context,
    get_current_tenant,
    get_tenant_knowledge_dir,
    verify_api_key,
)
from saido_agent.api.rbac import Role, require_permission

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


class ClipRequest(BaseModel):
    """Request body for POST /v1/clip (web clipper)."""

    url: Optional[str] = None
    html: Optional[str] = None
    selection: Optional[str] = None
    title: Optional[str] = None


class VoiceRequest(BaseModel):
    """Request body for POST /v1/voice (JSON mode with base64 audio)."""

    audio_base64: Optional[str] = None
    context: Optional[dict] = None
    stream: bool = False


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


class VoiceResponse(BaseModel):
    transcript: str
    answer: str
    audio_base64: str = ""
    citations: list[dict] = []
    latency_ms: dict = {}


class TokenResponse(BaseModel):
    token: str
    tenant_id: str
    expires_in: int = 3600


class ClipResponse(BaseModel):
    url: Optional[str] = None
    slug: Optional[str] = None
    title: Optional[str] = None
    status: str
    error: Optional[str] = None


class KeyCreatedResponse(BaseModel):
    api_key: str
    tenant_id: str
    message: str = "Store this key securely. It will not be shown again."


# -- RBAC models (Phase 3) --------------------------------------------------

class RegisterRequest(BaseModel):
    """Request body for POST /v1/auth/register."""
    email: str = Field(..., min_length=3)
    name: str = Field(..., min_length=1)
    password: str = Field(..., min_length=8, description="Minimum 8 characters")


class LoginRequest(BaseModel):
    """Request body for POST /v1/auth/login."""
    email: str
    password: str
    team_id: Optional[str] = None


class RegisterResponse(BaseModel):
    user_id: str
    email: str
    name: str


class LoginResponse(BaseModel):
    token: str
    user_id: str
    email: str
    team_id: str = ""
    role: str = ""
    expires_in: int = 3600


class CreateTeamRequest(BaseModel):
    name: str


class TeamResponse(BaseModel):
    id: str
    name: str
    owner_id: str
    role: Optional[str] = None


class AddMemberRequest(BaseModel):
    user_id: str
    role: str = "viewer"


class UpdateRoleRequest(BaseModel):
    role: str


class MemberResponse(BaseModel):
    user_id: str
    email: str = ""
    name: str = ""
    role: str
    team_id: str = ""


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
async def create_key(body: CreateKeyRequest, tenant: str = Depends(get_current_tenant)):
    """Create a new API key for a tenant. Requires authentication (admin-only)."""
    key = create_api_key(body.tenant_id, rate_limit=body.rate_limit)
    return KeyCreatedResponse(api_key=key, tenant_id=body.tenant_id)


# -- User auth (Phase 3) ---------------------------------------------------

@v1_router.post("/auth/register", response_model=RegisterResponse, tags=["auth"])
async def register_user(body: RegisterRequest):
    """Register a new user account."""
    from saido_agent.api.users import create_user

    try:
        user = create_user(body.email, body.name, body.password)
        return RegisterResponse(
            user_id=user["id"], email=user["email"], name=user["name"]
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@v1_router.post("/auth/login", response_model=LoginResponse, tags=["auth"])
async def login_user(body: LoginRequest):
    """Authenticate and receive a JWT token.

    If ``team_id`` is provided, the token is scoped to that team and
    includes the user's role. Otherwise, a token with no team scope is
    returned (the user must switch team context later).
    """
    from saido_agent.api.users import authenticate_user, get_member_role, list_user_teams

    user = authenticate_user(body.email, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    team_id = body.team_id or ""
    role = ""

    if team_id:
        member_role = get_member_role(team_id, user["id"])
        if member_role is None:
            raise HTTPException(
                status_code=403, detail="You are not a member of this team"
            )
        role = member_role
    else:
        # Auto-select first team if user has one
        teams = list_user_teams(user["id"])
        if teams:
            team_id = teams[0]["id"]
            role = teams[0]["role"]

    token = create_user_jwt_token(user["id"], team_id, role)
    return LoginResponse(
        token=token,
        user_id=user["id"],
        email=user["email"],
        team_id=team_id,
        role=role,
    )


# -- Teams (Phase 3) -------------------------------------------------------

@v1_router.post("/teams", response_model=TeamResponse, tags=["teams"])
async def create_team_endpoint(
    body: CreateTeamRequest,
    ctx: AuthContext = Depends(get_auth_context),
):
    """Create a new team. The authenticated user becomes the owner/admin."""
    from saido_agent.api.users import create_team

    if not ctx.user_id:
        raise HTTPException(
            status_code=400,
            detail="Team creation requires user-level authentication (login first)",
        )

    team = create_team(body.name, ctx.user_id)
    return TeamResponse(id=team["id"], name=team["name"], owner_id=team["owner_id"], role="admin")


@v1_router.get("/teams", response_model=list[TeamResponse], tags=["teams"])
async def list_teams_endpoint(
    ctx: AuthContext = Depends(get_auth_context),
):
    """List all teams the authenticated user belongs to."""
    from saido_agent.api.users import list_user_teams

    if not ctx.user_id:
        raise HTTPException(
            status_code=400,
            detail="Listing teams requires user-level authentication",
        )

    teams = list_user_teams(ctx.user_id)
    return [
        TeamResponse(id=t["id"], name=t["name"], owner_id=t["owner_id"], role=t["role"])
        for t in teams
    ]


@v1_router.post(
    "/teams/{team_id}/members",
    response_model=MemberResponse,
    tags=["teams"],
)
async def add_team_member(
    team_id: str,
    body: AddMemberRequest,
    ctx: AuthContext = Depends(get_auth_context),
):
    """Add a member to a team. Requires admin role in the team."""
    from saido_agent.api.users import add_member, get_member_role, get_user

    # Verify caller is admin in this team
    if ctx.user_id:
        caller_role = get_member_role(team_id, ctx.user_id)
        if caller_role != "admin":
            raise HTTPException(
                status_code=403, detail="Only team admins can add members"
            )

    # Validate role
    try:
        Role(body.role)
    except ValueError:
        raise HTTPException(
            status_code=422, detail=f"Invalid role '{body.role}'. Must be admin, editor, or viewer."
        )

    try:
        result = add_member(team_id, body.user_id, body.role)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    user = get_user(body.user_id)
    return MemberResponse(
        user_id=body.user_id,
        email=user["email"] if user else "",
        name=user["name"] if user else "",
        role=body.role,
        team_id=team_id,
    )


@v1_router.patch(
    "/teams/{team_id}/members/{user_id}",
    response_model=MemberResponse,
    tags=["teams"],
)
async def update_team_member_role(
    team_id: str,
    user_id: str,
    body: UpdateRoleRequest,
    ctx: AuthContext = Depends(get_auth_context),
):
    """Update a team member's role. Requires admin role."""
    from saido_agent.api.users import get_member_role, get_user, update_member_role

    # Verify caller is admin
    if ctx.user_id:
        caller_role = get_member_role(team_id, ctx.user_id)
        if caller_role != "admin":
            raise HTTPException(
                status_code=403, detail="Only team admins can update roles"
            )

    try:
        Role(body.role)
    except ValueError:
        raise HTTPException(
            status_code=422, detail=f"Invalid role '{body.role}'. Must be admin, editor, or viewer."
        )

    updated = update_member_role(team_id, user_id, body.role)
    if not updated:
        raise HTTPException(status_code=404, detail="Member not found in team")

    user = get_user(user_id)
    return MemberResponse(
        user_id=user_id,
        email=user["email"] if user else "",
        name=user["name"] if user else "",
        role=body.role,
        team_id=team_id,
    )


@v1_router.delete(
    "/teams/{team_id}/members/{user_id}",
    tags=["teams"],
)
async def remove_team_member(
    team_id: str,
    user_id: str,
    ctx: AuthContext = Depends(get_auth_context),
):
    """Remove a member from a team. Requires admin role."""
    from saido_agent.api.users import get_member_role, remove_member

    # Verify caller is admin
    if ctx.user_id:
        caller_role = get_member_role(team_id, ctx.user_id)
        if caller_role != "admin":
            raise HTTPException(
                status_code=403, detail="Only team admins can remove members"
            )

    removed = remove_member(team_id, user_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Member not found in team")

    return {"status": "removed", "user_id": user_id, "team_id": team_id}


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
    """Ingest content from a file upload. Max 50MB."""
    agent = _get_tenant_agent(tenant_id)

    # P2-MED-1: Enforce upload size limit to prevent memory exhaustion
    MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is 50MB, got {len(content)} bytes.")

    suffix = Path(file.filename or "upload.md").suffix or ".md"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        prefix="saido_upload_",
        delete=False,
    ) as f:
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


# -- Web clipper -----------------------------------------------------------

@v1_router.post("/clip", response_model=ClipResponse, tags=["ingest"])
async def clip_web_content(
    body: ClipRequest,
    tenant_id: str = Depends(get_current_tenant),
):
    """Accept browser-clipped content and ingest it.

    Accepts three modes (checked in priority order):

    1. **selection** — ingest just the user-selected text.
    2. **html** — extract text from the provided HTML and ingest.
    3. **url** — fetch the page, extract text, and ingest.

    At least one of ``url``, ``html``, or ``selection`` must be provided.

    **Bookmarklet** (paste into browser bookmark URL field)::

        javascript:void(
          fetch('YOUR_SERVER/v1/clip', {
            method:'POST',
            headers:{'Content-Type':'application/json','Authorization':'Bearer TOKEN'},
            body:JSON.stringify({
              url:location.href,
              html:document.documentElement.outerHTML,
              selection:window.getSelection().toString()
            })
          }).then(r=>r.json()).then(d=>alert(d.status==='ok'?'Clipped!':d.error))
        )
    """
    agent = _get_tenant_agent(tenant_id)
    pipeline = getattr(agent, "_ingest_pipeline", None)
    if pipeline is None:
        # Try to get pipeline from the knowledge context
        pipeline = getattr(agent, "ingest_pipeline", None)

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Ingest pipeline not available",
        )

    # Determine ingest mode
    if body.selection and body.selection.strip():
        result = await asyncio.to_thread(
            pipeline.ingest_selection,
            body.selection,
            url=body.url or "",
            title=body.title or "",
        )
    elif body.html and body.html.strip():
        result = await asyncio.to_thread(
            pipeline.ingest_html,
            body.html,
            url=body.url or "",
            title=body.title or "",
        )
    elif body.url and body.url.strip():
        result = await asyncio.to_thread(pipeline.ingest_url, body.url)
    else:
        raise HTTPException(
            status_code=422,
            detail="At least one of url, html, or selection must be provided",
        )

    if result.get("status") == "error":
        return ClipResponse(
            url=body.url,
            slug=result.get("slug"),
            title=result.get("title"),
            status="error",
            error=result.get("error"),
        )

    return ClipResponse(
        url=body.url,
        slug=result.get("slug"),
        title=result.get("title"),
        status="ok",
    )


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

        # Send a "thinking" event so the UI knows we're working
        yield f"data: {json.dumps({'type': 'thinking', 'content': 'Searching knowledge store...'})}\n\n"

        # Run the query in a thread (it's synchronous)
        result = await asyncio.to_thread(
            agent.query, body.question, body.context
        )

        # If answer is empty, provide a helpful fallback
        answer = result.answer
        if not answer or not answer.strip():
            answer = "I don't have information about that in the knowledge store. Try ingesting relevant documents first, or ask about topics already in the knowledge base."

        # Stream the answer token-by-token
        words = answer.split(" ")
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


# -- Voice -----------------------------------------------------------------

@v1_router.post("/voice", response_model=VoiceResponse, tags=["voice"])
async def voice_process(
    request: Request,
    body: VoiceRequest = None,
    tenant_id: str = Depends(get_current_tenant),
):
    """Process voice input and return voice output.

    Accepts audio as:
      - **JSON body**: base64-encoded audio in ``audio_base64`` field
      - **Binary body**: raw audio bytes (Content-Type: application/octet-stream)

    Returns the transcript, agent answer, synthesized audio (base64), and
    latency measurements for each pipeline stage.
    """
    import base64

    agent = _get_tenant_agent(tenant_id)

    # Resolve audio bytes from request
    # P3-HIGH-2: Enforce audio size limit (25MB max, ~10min at 16kHz mono)
    MAX_AUDIO_BYTES = 25 * 1024 * 1024
    content_type = request.headers.get("content-type", "")

    if "application/octet-stream" in content_type:
        audio_bytes = await request.body()
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail=f"Audio too large. Max 25MB, got {len(audio_bytes)} bytes.")
        context = None
    elif body is not None and body.audio_base64:
        try:
            audio_bytes = base64.b64decode(body.audio_base64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 audio data")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail=f"Audio too large. Max 25MB, got {len(audio_bytes)} bytes.")
        context = body.context
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide audio as base64 in JSON body or raw bytes with "
                   "Content-Type: application/octet-stream",
        )

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio data")

    # Process through voice pipeline
    try:
        pipeline = agent.voice_pipeline
        result = await pipeline.process(audio_bytes, context=context)

        audio_b64 = ""
        if result.audio_out:
            audio_b64 = base64.b64encode(result.audio_out).decode("ascii")

        return VoiceResponse(
            transcript=result.transcript,
            answer=result.answer,
            audio_base64=audio_b64,
            citations=result.citations,
            latency_ms=result.latency_ms,
        )
    except Exception as exc:
        logger.exception("Voice pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))


# -- Billing (Phase 3) ----------------------------------------------------

class CheckoutRequest(BaseModel):
    """Request body for POST /v1/billing/checkout."""
    tier: str
    success_url: str = "https://app.saido.ai/billing/success"
    cancel_url: str = "https://app.saido.ai/billing/cancel"


@v1_router.get("/billing/subscription", tags=["billing"])
async def get_subscription(
    tenant_id: str = Depends(get_current_tenant),
):
    """Return the current subscription details for the authenticated tenant."""
    from saido_agent.api.billing import get_billing_manager

    bm = get_billing_manager()
    sub = bm.get_or_create_subscription(tenant_id)
    return sub


@v1_router.get("/billing/usage", tags=["billing"])
async def get_usage(
    tenant_id: str = Depends(get_current_tenant),
):
    """Return usage summary for the current billing period."""
    from saido_agent.api.billing import get_billing_manager

    bm = get_billing_manager()
    return bm.get_usage_summary(tenant_id)


@v1_router.post("/billing/checkout", tags=["billing"])
async def create_checkout(
    body: CheckoutRequest,
    tenant_id: str = Depends(get_current_tenant),
):
    """Create a Stripe Checkout session for upgrading the subscription."""
    from saido_agent.api.billing import get_billing_manager

    bm = get_billing_manager()
    try:
        result = bm.create_checkout_session(
            tenant_id, body.tier,
            success_url=body.success_url,
            cancel_url=body.cancel_url,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@v1_router.post("/webhooks/stripe", tags=["billing"])
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events.

    This endpoint does NOT require authentication -- events are
    verified by the Stripe webhook signature.
    """
    from saido_agent.api.billing import get_billing_manager

    bm = get_billing_manager()
    payload = await request.body()
    signature = request.headers.get("Stripe-Signature", "")

    try:
        result = bm.handle_webhook(payload, signature)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
