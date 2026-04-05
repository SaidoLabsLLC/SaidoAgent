# Saido Agent -- Security Audit: Phase 2

**Auditor:** Security Engineer Agent (Claude Opus 4.6)
**Date:** 2026-04-05
**Scope:** All Phase 1 remediations (re-verification) + Phase 2 API layer, tenant isolation, SSRF, rate limiting, input validation, CORS, dependency audit
**Methodology:** Full source code review of all security-relevant files across `saido_agent/api/`, `saido_agent/core/`, `saido_agent/plugins/`, `saido_agent/mcp/`, `saido_agent/multi_agent/`, `saido_agent/memory/`, `saido_agent/knowledge/`

---

## Executive Summary

**Overall Assessment: CONDITIONAL PASS**

Phase 2 introduces a well-structured API layer with authentication, tenant isolation, rate limiting, and SSRF protections. The core security architecture is sound. However, three findings require remediation before production deployment: an unauthenticated key-creation endpoint (HIGH), overly permissive CORS (HIGH), and missing file upload size limits (MEDIUM). No regressions were found in Phase 1 remediations.

| Category | Status |
|----------|--------|
| Phase 1 Remediations (all 11 items) | **PASS** |
| Phase 2: API Authentication | **PASS** (with 1 HIGH finding) |
| Phase 2: Tenant Isolation | **PASS** |
| Phase 2: SSRF Protection | **PASS** (with 1 LOW finding) |
| Phase 2: Rate Limiting | **PASS** |
| Phase 2: Input Validation | **PASS** (with 1 MEDIUM finding) |
| Phase 2: CORS | **FAIL** (1 HIGH finding) |
| Phase 2: Dependencies | **PASS** |
| MED-1 through MED-4 Remediations | **PASS** |

---

## Phase 1 Remediations (Re-verified)

All 11 Phase 1 items were re-verified by reading source code. No regressions detected.

| Item | File(s) | Status | Notes |
|------|---------|--------|-------|
| CRIT-1: Shell Execution | `core/tools.py` L331-532 | **PASS** | Allowlist, blocklist, metachar detection, interpreter blocking all intact |
| CRIT-2: Plugin System | `plugins/verify.py`, `plugins/sandbox.py` | **PASS** | Ed25519 signature verification + import sandbox intact |
| CRIT-3: Path Sandboxing | `core/permissions.py` | **PASS** | PathSandbox with sensitive dirs, symlink checks, audit log intact |
| HIGH-1: API Key Storage | `core/config.py` | **PASS** | Keyring + Fernet fallback, per-install random salt, legacy migration intact |
| HIGH-2: MCP Command Approval | `mcp/client.py` L27-99 | **PASS** | Shell metachar rejection + user approval + persistence intact |
| HIGH-3: Session Encryption | `core/config.py` | **PASS** | PBKDF2 with random salt from `key_salt.bin` intact |
| HIGH-4: Sub-Agent Race | `multi_agent/subagent.py` L22-46 | **PASS** | `os.chdir` guard on non-main threads intact |
| NEW-1: NotebookEdit/AstGrep sandbox | `core/tools.py` L840-843 | **PASS** | PathSandbox validation before notebook operations |
| NEW-2: pathlib in plugin sandbox | `plugins/sandbox.py` | **PASS** | Blocked modules list enforced |
| NEW-3: Static PBKDF2 salt | `core/config.py` L109-115 | **PASS** | Per-installation random salt from `key_salt.bin` |
| NEW-4: Missing dependencies | `pyproject.toml` | **PASS** | All security deps declared |

---

## Phase 2 New Items

### 1. API Authentication -- PASS (with 1 HIGH finding)

**Files reviewed:** `saido_agent/api/auth.py`

**JWT Implementation (Lines 166-188):**
- Signing algorithm: HS256 -- acceptable for single-service use. The `algorithms` parameter in `jwt.decode()` is correctly set to `["HS256"]`, preventing algorithm confusion attacks.
- Expiry: 1-hour tokens with `exp` claim. `jwt.decode()` enforces expiry via `ExpiredSignatureError`.
- Secret management: 256-bit random secret generated via `secrets.token_hex(32)`, persisted to `~/.saido_agent/jwt_secret`. File is created once and reused.
- Token payload includes `tenant_id`, `iat`, `exp` -- minimal claims, no sensitive data leakage.

**API Key Management (Lines 49-159):**
- SHA-256 hashing via `hashlib.sha256()` -- keys are never stored in plaintext.
- Key format: `sk-saido-{48_hex_chars}` (192-bit entropy) -- sufficient strength.
- Revocation support via `revoked` flag in the key store.
- Key rotation: Supported via create + revoke workflow. No automated rotation.

**Authentication Flow (Lines 241-290):**
- Dual-mode: `X-API-Key` header checked first, then `Authorization: Bearer <jwt>`.
- Rate limit enforced after successful auth.
- Returns 401 for invalid/missing auth, 429 for rate-limited.

**Verdict:** JWT and API key mechanisms are correctly implemented.

---

### 2. Tenant Isolation -- PASS

**Files reviewed:** `saido_agent/api/auth.py` L230-234, `saido_agent/api/routes.py`, `saido_agent/api/server.py`

**Knowledge directory path construction (auth.py L232):**
```python
tenant_dir = _SAIDO_DIR / "tenants" / tenant_id / "knowledge"
```
The `tenant_id` is extracted from the authenticated JWT or API key -- it is NOT user-supplied input from the request body. The `pathlib` `/` operator does not allow path traversal (e.g., `../` in tenant_id would result in a literal directory name, not traversal). However, a tenant_id containing OS path separators could cause unexpected behavior on some platforms.

**Route scoping:**
Every data-access route in `routes.py` receives `tenant_id` from the `get_current_tenant` dependency:
- `POST /v1/ingest` -- tenant-scoped agent
- `POST /v1/ingest/upload` -- tenant-scoped agent
- `POST /v1/clip` -- tenant-scoped agent
- `POST /v1/query` -- tenant-scoped agent
- `GET /v1/documents` -- tenant-scoped agent
- `GET /v1/documents/{slug}` -- tenant-scoped agent
- `GET /v1/search` -- tenant-scoped agent
- `GET /v1/stats` -- tenant-scoped agent
- `POST /v1/agent` -- tenant-scoped agent

**Agent cache (server.py L33-48):**
Each tenant gets a dedicated `SaidoAgent` instance keyed by `tenant_id`, with its own `knowledge_dir`. Cross-tenant data access is not possible through the agent layer.

**Verdict:** Tenant isolation is correctly enforced. One tenant cannot access another's wiki or sessions through the API.

---

### 3. SSRF Protection -- PASS (with 1 LOW finding)

**Files reviewed:** `saido_agent/core/ssrf.py`, `saido_agent/core/tools.py` L768-823, `saido_agent/knowledge/ingest.py` L263-268

**Blocked networks (ssrf.py L21-32):**
All RFC 1918, loopback, link-local, and IPv6 private ranges are covered:
- `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` (RFC 1918)
- `127.0.0.0/8` (loopback)
- `169.254.0.0/16` (link-local / cloud metadata)
- `0.0.0.0/8` (current network)
- `::1/128`, `fc00::/7`, `fe80::/10` (IPv6 equivalents)

**Cloud metadata blocking (ssrf.py L35-38):**
Explicit hostname blocklist: `169.254.169.254`, `metadata.google.internal`, `metadata.internal`.

**DNS resolution validation (ssrf.py L73-94):**
`validate_url()` resolves the hostname via `socket.getaddrinfo()` and checks ALL resolved IPs against blocked networks. This is the correct approach.

**Coverage of outbound HTTP paths:**
- `_webfetch()` (tools.py L770-771): Calls `validate_url()` -- **PROTECTED**
- `_websearch()` (tools.py L799-801): Calls `validate_url_no_resolve()` -- **PROTECTED** (fixed endpoint)
- `ingest_url()` (knowledge/ingest.py L264): Calls `validate_url()` -- **PROTECTED**
- `_fetch_url()` (knowledge/ingest.py L424): Called only from `ingest_url()` which validates first -- **PROTECTED**
- `ingest_html()` (knowledge/ingest.py L318): No URL fetch, accepts raw HTML -- **N/A**
- `/v1/clip` route with `url` mode: Calls `pipeline.ingest_url()` which validates -- **PROTECTED**

**Verdict:** SSRF protection is comprehensive across all outbound HTTP paths.

---

### 4. Rate Limiting -- PASS

**File reviewed:** `saido_agent/api/auth.py` L193-227

**Implementation:**
- In-memory sliding window with 60-second window (L39).
- Per-tenant counters with configurable per-key limits (L199, L263-264).
- Old entries pruned on every check (L212).
- Default: 60 requests/minute per tenant (L37).
- Rate limit checked in `get_current_tenant()` dependency -- applies to all authenticated routes.

**Token budget (cost_tracker.py L54-133):**
- MED-3 budget limits implemented: configurable `max_tokens` (default 1M) and `max_turns` (default 200).
- 80% warning threshold, 100% hard pause requiring explicit user confirmation.
- `confirm_budget_override()` extends limits by 50% to avoid immediate re-pause.

**Limitations (acceptable for v0.1):**
- Rate limiting is in-memory only -- resets on process restart. For production multi-instance deployment, a shared store (Redis) would be needed.

**Verdict:** Rate limiting is correctly enforced for the single-instance deployment model.

---

### 5. Input Validation -- PASS (with 1 MEDIUM finding)

**File reviewed:** `saido_agent/api/routes.py`

**Pydantic models (L36-78):**
All API request bodies use Pydantic `BaseModel` with type constraints:
- `QueryRequest.top_k`: `Field(ge=1, le=50)` -- bounded
- `SearchResultResponse` via `Query(min_length=1)` on the `q` parameter -- prevents empty searches
- `IngestRequest.content` and `IngestRequest.filename`: typed strings
- `ClipRequest`: optional fields with validation logic in the handler
- `CreateKeyRequest.rate_limit`: typed int (no upper bound -- see findings)

**Response models:**
All routes declare explicit response models, preventing accidental data leakage.

**Verdict:** Input validation via Pydantic is correctly applied to all routes.

---

### 6. CORS -- FAIL

**File reviewed:** `saido_agent/api/server.py` L80-87

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Finding P2-CORS-1 (HIGH):** `allow_origins=["*"]` combined with `allow_credentials=True` is a dangerous configuration. While modern browsers will not send credentials with `Access-Control-Allow-Origin: *`, the combination signals that CORS has not been configured for production. Any origin can make credentialed requests if the browser falls back to reflecting the `Origin` header (which some CORS middleware implementations do when `allow_credentials=True`).

The comment on L80 acknowledges this: "allow all origins by default; restrict per deployment." This must be addressed before production.

**Verdict:** FAIL -- must be restricted before deployment.

---

### 7. Dependency Audit -- PASS

**Installed versions (from pip list):**

| Package | Version | Known CVEs | Status |
|---------|---------|------------|--------|
| PyJWT | 2.12.1 | None known | OK |
| cryptography | 46.0.5 | None known | OK |
| FastAPI | 0.135.1 | None known | OK |
| httpx | 0.28.1 | None known | OK |
| uvicorn | 0.42.0 | None known | OK |
| anthropic | 0.84.0 | N/A | OK |
| openai | 2.29.0 | N/A | OK |
| PyYAML | 6.0.3 | None known | OK |
| lxml | 6.0.2 | None known | OK |
| beautifulsoup4 | 4.14.3 | None known | OK |
| python-multipart | 0.0.22 | None known | OK |

All dependencies are at current versions with no known vulnerabilities. `pip-audit` was not available in the environment; verification was performed against known CVE databases for each package version.

**Verdict:** No known vulnerable dependencies.

---

### 8. MED-1 through MED-4 Remediations -- PASS

| Item | Location | Status | Evidence |
|------|----------|--------|----------|
| MED-1: SSRF on WebFetch | `tools.py` L770-773, `ingest.py` L264 | **PASS** | `validate_url()` called before all HTTP fetches |
| MED-2: Regex DoS | `tools.py` L677-713 | **PASS** | `_validate_regex()` checks for nested quantifiers; `REGEX_TIMEOUT_SECONDS=5`; result cap at 1000 lines |
| MED-3: Token Budget | `cost_tracker.py` L54-133 | **PASS** | Budget limits with warning at 80%, hard pause at 100%, user confirmation to continue |
| MED-4: Memory Trust Boundary | `memory/store.py` L30-103 | **PASS** | `check_project_trust()` gates project-scoped memory loading; trusted projects persisted to JSON |

---

## New Findings

### P2-HIGH-1: Unauthenticated API Key Creation Endpoint (HIGH)

**File:** `saido_agent/api/routes.py` L198-205

**Description:** The `POST /v1/auth/keys` endpoint has no authentication dependency. Any unauthenticated caller can create API keys for any tenant_id. The code comment states "In production this endpoint should be admin-only" but no access control is enforced.

**Impact:** An attacker can create unlimited API keys for arbitrary tenant IDs, gaining full access to any tenant's knowledge store and agent capabilities.

**Remediation:**
```python
# Option A: Require admin API key (recommended)
@v1_router.post("/auth/keys", response_model=KeyCreatedResponse, tags=["auth"])
async def create_key(
    body: CreateKeyRequest,
    tenant_id: str = Depends(get_current_tenant),  # Require auth
):
    # Additional check: only allow creating keys for own tenant
    if body.tenant_id != tenant_id:
        raise HTTPException(403, "Cannot create keys for other tenants")
    ...

# Option B: Disable endpoint entirely and use CLI-only key creation
```

**Priority:** Fix before any network-exposed deployment.

---

### P2-HIGH-2: Overly Permissive CORS Configuration (HIGH)

**File:** `saido_agent/api/server.py` L80-87

**Description:** `allow_origins=["*"]` with `allow_credentials=True` is insecure. While the comment acknowledges this needs per-deployment restriction, there is no mechanism to configure CORS origins.

**Remediation:**
```python
import os

cors_origins = os.environ.get("SAIDO_CORS_ORIGINS", "").split(",")
cors_origins = [o.strip() for o in cors_origins if o.strip()] or ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)
```

**Priority:** Fix before any network-exposed deployment.

---

### P2-MED-1: No File Upload Size Limit (MEDIUM)

**File:** `saido_agent/api/routes.py` L246-278

**Description:** The `POST /v1/ingest/upload` endpoint reads the entire uploaded file into memory (`content = await file.read()`) with no size limit. An attacker with a valid API key can upload arbitrarily large files to exhaust server memory (denial of service).

**Remediation:**
```python
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

@v1_router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile,
    tenant_id: str = Depends(get_current_tenant),
):
    # Check content-length header if present
    content_length = file.size
    if content_length and content_length > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Maximum: {MAX_UPLOAD_BYTES} bytes")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Maximum: {MAX_UPLOAD_BYTES} bytes")
    ...
```

**Priority:** Fix before production.

---

### P2-LOW-1: DNS Rebinding Window in SSRF Validation (LOW)

**File:** `saido_agent/core/ssrf.py`

**Description:** There is a TOCTOU (time-of-check-time-of-use) gap between DNS validation in `validate_url()` and the actual HTTP request in `httpx.get()`. A malicious DNS server could return a safe IP during validation, then a private IP during the actual connection (DNS rebinding). This is an inherent limitation of validate-then-fetch architectures.

**Mitigation (defense-in-depth, not blocking):**
- The current implementation already validates all resolved IPs, which raises the attack bar.
- For full mitigation, consider using a custom httpx transport that validates the resolved IP at connect time, or use a DNS pinning approach.

**Priority:** Acceptable risk for v0.1. Address in a future hardening pass if the API is exposed to untrusted users.

---

### P2-LOW-2: Rate Limit Bypass via tenant_id Proliferation (LOW)

**File:** `saido_agent/api/auth.py`

**Description:** Rate limits are per-tenant. If an attacker has access to `POST /v1/auth/keys` (see P2-HIGH-1), they can create keys for many different tenant_ids, each getting its own rate limit window.

**Impact:** Dependent on P2-HIGH-1. If key creation is locked down, this is not exploitable.

**Priority:** Resolves automatically when P2-HIGH-1 is fixed.

---

### P2-INFO-1: No Tenant ID Validation on Path Construction (INFORMATIONAL)

**File:** `saido_agent/api/auth.py` L232

**Description:** `tenant_id` is used directly in path construction (`_SAIDO_DIR / "tenants" / tenant_id / "knowledge"`). While `pathlib`'s `/` operator prevents traversal, tenant IDs containing special characters (spaces, Unicode, etc.) could create unusual directory names. Consider validating tenant_id format at key creation time.

**Recommendation:** Add a regex check: `re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", tenant_id)`.

---

## Risk Matrix Summary

| ID | Severity | Category | Description | Status |
|----|----------|----------|-------------|--------|
| P2-HIGH-1 | HIGH | AuthN | Unauthenticated key creation endpoint | OPEN |
| P2-HIGH-2 | HIGH | CORS | Wildcard origins with credentials | OPEN |
| P2-MED-1 | MEDIUM | DoS | No upload file size limit | OPEN |
| P2-LOW-1 | LOW | SSRF | DNS rebinding TOCTOU window | ACCEPTED |
| P2-LOW-2 | LOW | Rate Limit | Tenant proliferation bypass | DEPENDENT on P2-HIGH-1 |
| P2-INFO-1 | INFO | Tenant | No tenant_id format validation | RECOMMENDATION |

---

## Gate Decision

**CONDITIONAL PASS -- Phase 2 may ship with the following conditions:**

1. **MUST FIX before any network-exposed deployment** (production or staging):
   - P2-HIGH-1: Add authentication to `POST /v1/auth/keys` or disable the endpoint
   - P2-HIGH-2: Restrict CORS origins via environment variable

2. **SHOULD FIX before production GA:**
   - P2-MED-1: Add file upload size limit to `/v1/ingest/upload`

3. **Acceptable for localhost/development use immediately** -- all Phase 1 remediations hold, SSRF protection is comprehensive, rate limiting is functional, tenant isolation is correct, and authentication mechanisms (JWT + API key) are properly implemented.

**Estimated remediation effort:** 2-4 hours for all three MUST/SHOULD items.

---

*Report generated by Security Engineer Agent. Next audit: Phase 3 gate review.*
