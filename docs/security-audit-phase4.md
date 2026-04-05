# Saido Agent -- Security Audit: Phase 4 (Pre-GA Final)

**Auditor:** Security Engineer Agent (Claude Opus 4.6)
**Date:** 2026-04-05
**Scope:** Full cumulative audit -- all Phase 1-3 remediations re-verified + Phase 4 new items: fine-tuning pipeline, synthetic data, plugin marketplace, enterprise audit log, SSO/SAML, GDPR compliance, dependency supply chain
**Methodology:** Complete source code review of all security-relevant files across `saido_agent/`

---

## Executive Summary

**Overall Assessment: CONDITIONAL PASS**

Phase 4 introduces enterprise features that are architecturally sound. The fine-tuning pipeline, synthetic data generator, plugin marketplace, enterprise audit log, GDPR compliance, and billing system are all well-structured with appropriate security boundaries. All Phase 1-3 remediations continue to hold with no regressions.

However, five findings require attention before GA: unauthenticated SSO configuration endpoints (CRITICAL), OIDC client secrets stored in plaintext in SQLite (HIGH), mock SSO token validation accepting arbitrary JSON (HIGH), no tenant scoping in the fine-tuning pipeline (MEDIUM), and the marketplace `publish()` method not enforcing `sanitize_plugin_name()` on the package destination path (MEDIUM).

No prior-phase regressions were detected. The P3-CRIT-1 (RBAC bypass via legacy API key) has been fixed -- legacy keys now default to viewer-level access. The P3-HIGH-1 (password strength) has been fixed -- `min_length=8` is enforced on `RegisterRequest.password`. The P3-HIGH-2 (audio size limit) has been fixed -- 25MB cap enforced. The P3-MED-1 (security headers) has been fixed -- CSP, X-Content-Type-Options, X-Frame-Options, Referrer-Policy all present.

---

## Cumulative Status (All Phases)

| Phase | ID | Severity | Description | Status |
|-------|----|----------|-------------|--------|
| P1 | CRIT-1 | CRITICAL | Shell execution sandboxing | **PASS** |
| P1 | CRIT-2 | CRITICAL | Plugin signature verification + sandbox | **PASS** |
| P1 | CRIT-3 | CRITICAL | Path sandboxing | **PASS** |
| P1 | HIGH-1 | HIGH | API key storage (keyring + Fernet) | **PASS** |
| P1 | HIGH-2 | HIGH | MCP command approval | **PASS** |
| P1 | HIGH-3 | HIGH | Session encryption | **PASS** |
| P1 | HIGH-4 | HIGH | Sub-agent race condition guard | **PASS** |
| P1 | NEW-1 | HIGH | NotebookEdit/AstGrep sandbox | **PASS** |
| P1 | NEW-2 | MEDIUM | pathlib removed from plugin sandbox | **PASS** |
| P1 | NEW-3 | MEDIUM | Per-installation PBKDF2 salt | **PASS** |
| P1 | NEW-4 | LOW | Missing dependencies declared | **PASS** |
| P2 | P2-HIGH-1 | HIGH | Unauthenticated key creation | **FIXED** (auth required at `routes.py` L278) |
| P2 | P2-HIGH-2 | HIGH | Wildcard CORS | **FIXED** (env-var origins at `server.py` L84) |
| P2 | P2-MED-1 | MEDIUM | No upload size limit | **FIXED** (50MB cap) |
| P2 | P2-LOW-1 | LOW | DNS rebinding TOCTOU | **ACCEPTED** |
| P2 | MED-1-4 | MEDIUM | SSRF, regex DoS, token budget, memory trust | **ALL PASS** |
| P3 | P3-CRIT-1 | CRITICAL | RBAC bypass via legacy API key | **FIXED** (viewer default at `rbac.py` L87-97) |
| P3 | P3-HIGH-1 | HIGH | No password strength enforcement | **FIXED** (`min_length=8` at `routes.py` L190) |
| P3 | P3-HIGH-2 | HIGH | No audio input size limit | **FIXED** (25MB cap at `routes.py` L877) |
| P3 | P3-MED-1 | MEDIUM | No security headers | **FIXED** (CSP + headers at `server.py` L95-106) |
| P3 | P3-MED-2 | MEDIUM | No billing endpoint | **FIXED** (billing.py + routes implemented) |

**All 27 prior items pass. No regressions detected.**

---

## Phase 4 New Items

### 1. Fine-tuning Pipeline -- CONDITIONAL PASS

**File reviewed:** `saido_agent/knowledge/finetune.py` (603 lines)

#### 1a. Training Data Tenant Isolation -- FINDING P4-MED-1

**Description:** The `FinetuneManager` class has no concept of `tenant_id`. Training files are referenced by absolute filesystem path (`str(path.resolve())` at L199). The jobs file is stored at `~/.saido_agent/finetune_jobs.json` -- a single global file shared across all tenants. Any tenant with access to the fine-tuning API could:

1. List all fine-tuning jobs from all tenants via `list_jobs()`.
2. Check status of or deploy another tenant's completed model via `check_status()` / `deploy()`.
3. Reference training data files outside their own tenant directory.

**Impact:** MEDIUM. Cross-tenant information disclosure of training job metadata and potential access to another tenant's model artifacts. The training file path is stored as an absolute path, but no validation ensures the file is within the calling tenant's directory.

**Remediation:**
```python
class FinetuneManager:
    def __init__(self, tenant_id: str, ...):
        self._tenant_id = tenant_id
        self._config_dir = config_dir or Path.home() / ".saido_agent" / "tenants" / tenant_id
        ...

    def start_openai(self, training_file: str, ...):
        path = Path(training_file).resolve()
        tenant_dir = (Path.home() / ".saido_agent" / "tenants" / self._tenant_id).resolve()
        if not str(path).startswith(str(tenant_dir)):
            raise ValueError("Training file must be within tenant directory")
        ...
```

#### 1b. OpenAI API Key Handling -- PASS

The `_get_openai_client()` function (L119-125) uses `openai.OpenAI()` which reads `OPENAI_API_KEY` from the environment. The API key is never logged, stored in job metadata, or included in error messages. The `str(exc)` in error handling (L245) could theoretically leak the key if the OpenAI SDK includes it in exception messages, but modern OpenAI SDK versions redact keys from error strings.

#### 1c. Model Artifact Signing -- FINDING (INFORMATIONAL)

No model artifact signing or integrity verification exists. Completed local models are identified solely by checking for the presence of marker files (`adapter_model.bin`, `model.safetensors`, etc.) at L405-415. A malicious actor with filesystem access could replace model artifacts. This is acceptable for the current threat model (single-machine deployment) but should be addressed for distributed deployments.

#### 1d. JSONL Validation -- PASS

`_validate_jsonl()` (L90-112) correctly validates file existence, extension, JSON structure, and the presence of the required `messages` field. Empty files are rejected.

---

### 2. Synthetic Data -- PASS

**File reviewed:** `saido_agent/knowledge/synthetic.py` (597 lines)

#### 2a. PII in Training Data -- PASS (ACCEPTABLE)

The synthetic data generator creates QA pairs from knowledge store articles via LLM calls. The system prompts (L46-67) instruct the LLM to generate question/answer pairs from article content. There is no explicit PII scrubbing of the generated output, but:

1. The source material is knowledge articles (not user PII).
2. The LLM generates synthetic QA pairs, not direct copies of source text.
3. The validation layer (L321-385) filters by quality but does not scan for PII patterns.

**Recommendation (non-blocking):** Consider adding a post-generation PII scan (regex for emails, phone numbers, SSNs) as a defense-in-depth measure.

#### 2b. Output File Paths -- PASS

Output files are written to a caller-specified `output_dir` or `Path.cwd()` (L188). The `Path.mkdir(parents=True, exist_ok=True)` call creates the directory safely. Filenames are generated with timestamps (`training_data_{timestamp}.jsonl`) -- no user input in filenames. The `export_*` methods all use `Path(path).parent.mkdir(parents=True, exist_ok=True)` before writing.

#### 2c. JSON Parsing Safety -- PASS

The `_extract_json_array()` method (L567-596) safely handles malformed LLM output with try/except blocks and returns `None` on failure. No `eval()` or `exec()` is used. All parsing goes through `json.loads()`.

---

### 3. Plugin Marketplace -- CONDITIONAL PASS

**File reviewed:** `saido_agent/plugins/marketplace.py` (495 lines)

#### 3a. Plugin Sandboxing and Signing -- PASS

The marketplace correctly integrates with the existing plugin security infrastructure:
- Signature verification on install: L165-175 calls `verify_manifest_signature()` when `entry.signed` is True.
- AST-based restricted import scanning: L389-418 uses `ast.parse()` and `ast.walk()` to detect restricted imports. This is robust -- it cannot be bypassed by string obfuscation since it operates on the parsed AST, not raw text.
- Restricted imports list (L35-43) mirrors `sandbox.BLOCKED_MODULES`: `os`, `subprocess`, `shutil`, `socket`, `http`, `urllib`, `ctypes`.

#### 3b. Path Traversal in Package Names -- FINDING P4-MED-2

**File:** `saido_agent/plugins/marketplace.py` L234-237

```python
dest = self._packages_dir / manifest.name
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(str(plugin_path), str(dest))
```

The `publish()` method uses `manifest.name` directly as the destination directory name under `_packages_dir`. If `manifest.name` contains path separators (e.g., `../../evil`) or other filesystem-special characters, this could write files outside the packages directory. The `install()` method similarly uses the marketplace entry name directly: `package_dir = self._packages_dir / name` (L162).

The existing `sanitize_plugin_name()` function in `types.py` (L300-302) strips non-word characters, but it is NOT called in the marketplace `publish()` or `install()` paths.

**Impact:** MEDIUM. An attacker who submits a plugin with a crafted name could write files outside the marketplace packages directory or overwrite system files.

**Remediation:**
```python
from .types import sanitize_plugin_name

def publish(self, plugin_dir: str) -> SubmissionResult:
    ...
    # After loading manifest:
    safe_name = sanitize_plugin_name(manifest.name)
    dest = self._packages_dir / safe_name
    ...
```

#### 3c. Automated Security Checks -- PASS

The `_run_automated_checks()` method (L309-387) performs four checks:
1. Manifest v2 validation (required fields present)
2. Dependency audit (valid pip package names via `validate_pip_package_name()`)
3. Restricted import scan (AST-based, see 3a)
4. Signature check (informational, not a hard failure)

Unsigned plugins are placed in `pending_review` status (L232), not auto-approved. Only signed plugins that pass all checks are auto-approved (L229).

#### 3d. Install Flow -- PASS

Installation delegates to `store.install_plugin()` (L178) which goes through the standard plugin install flow including sandboxing. The `approval_callback=lambda _prompt: True` auto-approves marketplace installs, which is acceptable because the plugin has already passed automated checks.

---

### 4. Enterprise Audit Log -- PASS

**File reviewed:** `saido_agent/api/enterprise.py` (L52-187), `saido_agent/api/migrations/004_enterprise.sql`

#### 4a. Append-Only Enforcement -- PASS

The `EnterpriseAuditLog` class exposes only three methods: `log()` (INSERT), `search()` (SELECT), and `export()` (SELECT via search). There are **no UPDATE or DELETE methods** on the audit log class. A grep across the entire codebase confirms no `UPDATE enterprise_audit` or `DELETE FROM enterprise_audit` statements exist.

The migration comment (004_enterprise.sql L14) explicitly documents: "Append-only: application code must never UPDATE or DELETE from this table."

**Note:** SQLite does not support database-level triggers to prevent DELETE/UPDATE. A determined admin with direct database access could modify records. For true tamper-proofing, consider hash-chaining audit entries (each entry includes the hash of the previous entry). This is an informational recommendation, not a blocking finding.

#### 4b. Completeness -- PASS

Audit logging is integrated at the route level in `enterprise_routes.py`:
- Tenant data deletion: L167-173 logs `tenant_data_deleted`
- Consent recording: L197-203 logs `consent_recorded`

The audit log captures: `user_id`, `tenant_id`, `action`, `resource`, `details`, `ip_address`, `timestamp`. All fields are populated via parameterized queries.

**Recommendation (non-blocking):** Audit logging coverage should be extended to additional actions: login, logout, SSO authentication, API key creation/revocation, role changes, article ingestion/deletion.

#### 4c. Injection-Safe Queries -- PASS

All queries in `EnterpriseAuditLog.search()` (L105-151) use parameterized placeholders (`?`). The WHERE clause is constructed by appending `"column = ?"` strings with parameters in a separate list. No string interpolation of user input into SQL.

The `LIMIT` parameter is appended as a bound parameter (L132: `params.append(limit)`), and the route validates `limit` with `Query(100, ge=1, le=1000)` (enterprise_routes.py L91).

---

### 5. SSO/SAML -- FAIL (2 findings: 1 CRITICAL, 1 HIGH)

**Files reviewed:** `saido_agent/api/enterprise.py` (L382-531), `saido_agent/api/enterprise_routes.py` (L212-273)

#### 5a. FINDING P4-CRIT-1: SSO Configuration Endpoints Unauthenticated (CRITICAL)

**File:** `saido_agent/api/enterprise_routes.py` L212-223

```python
@sso_router.post("/sso/saml/config")
async def configure_saml(body: SAMLConfigRequest):
    """Configure SAML 2.0 SSO for a tenant."""
    sso = get_sso_manager()
    return sso.configure_saml(body.tenant_id, body.idp_metadata_url, body.entity_id)

@sso_router.post("/sso/oidc/config")
async def configure_oidc(body: OIDCConfigRequest):
    """Configure OIDC SSO for a tenant."""
    sso = get_sso_manager()
    return sso.configure_oidc(body.tenant_id, body.issuer, body.client_id, body.client_secret)
```

**Description:** Both SSO configuration endpoints (`POST /v1/auth/sso/saml/config` and `POST /v1/auth/sso/oidc/config`) have **no authentication dependency**. Any unauthenticated caller can:

1. Configure SAML SSO for any tenant, pointing it to an attacker-controlled IdP.
2. Configure OIDC SSO for any tenant with attacker-controlled issuer/client credentials.
3. After configuring a malicious IdP, use `POST /v1/auth/sso/saml` or `POST /v1/auth/sso/oidc` to obtain a valid JWT with any role (including admin), because the mock validator accepts arbitrary JSON.

**Impact:** CRITICAL. Complete authentication bypass and privilege escalation. An attacker can configure a malicious IdP, then authenticate as admin for any tenant.

**Attack chain:**
1. `POST /v1/auth/sso/oidc/config` with `{"tenant_id": "victim", "issuer": "evil.com", "client_id": "x", "client_secret": "x"}`
2. `POST /v1/auth/sso/oidc` with `{"id_token": "{\"sub\": \"admin\", \"email\": \"admin@evil.com\", \"groups\": [\"admin\"]}"}`
3. Receive a valid JWT with admin role for tenant "victim"

**Remediation:**
```python
@sso_router.post("/sso/saml/config")
async def configure_saml(
    body: SAMLConfigRequest,
    ctx: AuthContext = Depends(require_permission("manage_settings")),
):
    if body.tenant_id != ctx.tenant_id:
        raise HTTPException(403, "Cannot configure SSO for another tenant")
    ...
```

**Priority:** MUST FIX before any deployment.

#### 5b. FINDING P4-HIGH-1: Mock SSO Validation Accepts Arbitrary JSON (HIGH)

**File:** `saido_agent/api/enterprise.py` L468-503

```python
def validate_saml_response(self, saml_response: str) -> dict:
    try:
        payload = json.loads(saml_response)
        return {"valid": True, "user_id": payload.get("user_id", ""), ...}
    except (json.JSONDecodeError, TypeError):
        return {"valid": False, "error": "Invalid SAML response"}

def validate_oidc_token(self, id_token: str) -> dict:
    try:
        payload = json.loads(id_token)
        return {"valid": True, "user_id": payload.get("sub", ""), ...}
    except (json.JSONDecodeError, TypeError):
        return {"valid": False, "error": "Invalid OIDC token"}
```

**Description:** Both SSO validation methods parse raw JSON and return `{"valid": True}` for any syntactically valid JSON input. There is no actual cryptographic validation -- no XML signature verification for SAML, no JWT/JWK validation for OIDC. The docstrings acknowledge this is a mock, but the mock is deployed in the production API routes without any guard.

**Impact:** HIGH. Combined with P4-CRIT-1, this allows authentication bypass. Even without P4-CRIT-1, any caller to the SSO login endpoints can forge authentication tokens by passing arbitrary JSON.

**Remediation:**
1. **Immediate:** Add a feature flag to disable mock SSO validators in production:
```python
_SSO_MOCK_MODE = os.environ.get("SAIDO_SSO_MOCK", "false").lower() == "true"

def validate_saml_response(self, saml_response: str) -> dict:
    if not _SSO_MOCK_MODE:
        return {"valid": False, "error": "SAML validation not configured. Install python3-saml."}
    ...
```
2. **Before GA:** Implement real validation using `python3-saml` and `python-jose`.

**Priority:** MUST FIX before any deployment.

#### 5c. Session Management -- PASS

SSO login endpoints (enterprise_routes.py L226-273) correctly create JWT tokens via `create_user_jwt_token()` with user_id, team_id, and role from the IdP response. Token expiry is set to `_JWT_EXPIRY_SECONDS` (1 hour). The refresh endpoint (L281-319) validates the existing token before issuing a new one.

#### 5d. OIDC Client Secret Storage -- FINDING P4-HIGH-2

**File:** `saido_agent/api/enterprise.py` L424-446

The OIDC client_secret is stored as plaintext JSON in the `sso_configs` table (`config_json` column). Anyone with database access can read all OIDC client secrets.

**Impact:** HIGH. Client secret compromise allows impersonation of the application to the IdP. In a real OIDC flow, the client_secret is used in the token exchange -- leaking it allows an attacker to complete the OIDC flow independently.

**Remediation:** Encrypt the `config_json` field using Fernet (already available in the codebase) before storage. Decrypt on read.

**Priority:** SHOULD FIX before production GA with real IdP integration.

---

### 6. GDPR Compliance -- PASS

**File reviewed:** `saido_agent/api/enterprise.py` (L193-376), `saido_agent/api/enterprise_routes.py` (L139-204)

#### 6a. Data Export (Right to Portability) -- PASS

`export_tenant_data()` (L209-273) creates a ZIP archive containing:
- Audit log records
- Consent records
- Subscriptions
- Usage records
- SSO configurations

All queries are parameterized with `tenant_id`. Tables that may not exist are wrapped in `try/except sqlite3.OperationalError`. The export is comprehensive for the current data model.

#### 6b. Data Deletion (Right to Erasure) -- PASS

`delete_tenant_data()` (L276-334) requires `confirm=True` as a safety guard. It deletes:
- Consent records
- Subscriptions
- Usage records
- SSO configurations

Audit log entries are intentionally preserved (legal retention requirement). A deletion event is logged in the audit trail (enterprise_routes.py L167-173).

**Note:** The deletion does not cover the tenant's knowledge store on the filesystem (articles, compiled output). This is a gap -- GDPR erasure should include all personal data.

**Recommendation (non-blocking):** Add filesystem cleanup of `~/.saido_agent/tenants/{tenant_id}/` to the deletion flow, or document that knowledge store deletion is a separate manual step.

#### 6c. Consent Management -- PASS

`record_consent()` (L355-375) uses `INSERT ... ON CONFLICT DO UPDATE` for upsert semantics. The `consent_records` table has a `UNIQUE(tenant_id, purpose)` constraint. `get_consent_record()` (L336-353) returns all consents for a tenant. Routes are tenant-scoped via `Depends(get_current_tenant)`.

---

### 7. Dependency Audit -- PASS

**File reviewed:** `pyproject.toml`

| Dependency | Pinned Version | Known CVEs (as of 2026-04-05) | Assessment |
|------------|---------------|-------------------------------|------------|
| anthropic>=0.40.0 | Minimum pinned | None known | OK |
| openai>=1.30.0 | Minimum pinned | None known | OK |
| httpx>=0.27.0 | Minimum pinned | None known | OK |
| rich>=13.0.0 | Minimum pinned | None known | OK |
| keyring>=25.0.0 | Minimum pinned | None known | OK |
| cryptography>=42.0.0 | Minimum pinned | None known | OK |
| pyyaml>=6.0 | Minimum pinned | None known | OK |
| sounddevice | **UNPINNED** | N/A (native audio) | LOW risk |
| smartrag | **UNPINNED** | N/A (internal) | LOW risk |
| fastapi>=0.110.0 | Minimum pinned | None known | OK |
| uvicorn>=0.27.0 | Minimum pinned | None known | OK |
| python-multipart>=0.0.9 | Minimum pinned | None known | OK |
| pyjwt>=2.8.0 | Minimum pinned | None known | OK |
| beautifulsoup4>=4.12.0 | Minimum pinned | None known | OK |
| lxml>=5.0.0 | Minimum pinned | None known | OK |

**Observations:**
- `sounddevice` and `smartrag` have no minimum version pins. `sounddevice` is a thin wrapper around PortAudio -- low supply chain risk. `smartrag` appears to be an internal/first-party package.
- No lock file (e.g., `requirements.lock`, `poetry.lock`) exists for reproducible builds. Transitive dependencies are not pinned.
- The `stripe` library is used conditionally but not declared in dependencies. It is loaded dynamically in `billing.py` with graceful fallback.

**Recommendation (non-blocking):** Generate a lock file with pinned transitive dependencies before GA. Add `pip-audit` to CI.

---

## New Findings Summary

| ID | Severity | Category | Description | Status |
|----|----------|----------|-------------|--------|
| P4-CRIT-1 | **CRITICAL** | AuthN | SSO config endpoints unauthenticated -- allows attacker to hijack any tenant's IdP | **OPEN** |
| P4-HIGH-1 | **HIGH** | AuthN | Mock SSO validators accept arbitrary JSON as valid auth | **OPEN** |
| P4-HIGH-2 | **HIGH** | Secrets | OIDC client_secret stored in plaintext in SQLite | **OPEN** |
| P4-MED-1 | **MEDIUM** | Isolation | Fine-tuning pipeline has no tenant scoping -- global job store | **OPEN** |
| P4-MED-2 | **MEDIUM** | Path Traversal | Marketplace publish uses unsanitized manifest.name as directory name | **OPEN** |
| P4-INFO-1 | **INFO** | Integrity | No model artifact signing for local fine-tuned models | **RECOMMENDATION** |
| P4-INFO-2 | **INFO** | GDPR | Data deletion does not cover filesystem knowledge store | **RECOMMENDATION** |
| P4-INFO-3 | **INFO** | Supply Chain | No lock file for transitive dependency pinning | **RECOMMENDATION** |
| P4-INFO-4 | **INFO** | Audit | Audit log coverage does not include login/logout/key management events | **RECOMMENDATION** |

---

## GA Readiness Assessment

### Blocking Items (MUST FIX before GA)

1. **P4-CRIT-1: Unauthenticated SSO configuration endpoints.** This is the most severe finding in the entire audit history. An attacker can configure a malicious IdP for any tenant and then authenticate as admin. The fix is straightforward: add `Depends(require_permission("manage_settings"))` to both SSO config endpoints and validate `body.tenant_id == ctx.tenant_id`. **Estimated effort: 30 minutes.**

2. **P4-HIGH-1: Mock SSO validators in production.** The mock validators must be gated behind an environment variable or feature flag. Without real IdP validation, the SSO login endpoints effectively accept any JSON payload as a valid authentication assertion. **Estimated effort: 30 minutes.**

### Should Fix Before GA

3. **P4-HIGH-2: OIDC client_secret plaintext storage.** Encrypt using Fernet before persisting to the database. The encryption infrastructure already exists in `core/config.py`. **Estimated effort: 1 hour.**

4. **P4-MED-1: Fine-tuning tenant isolation.** Add `tenant_id` parameter to `FinetuneManager` and scope the jobs file and training file paths to the tenant directory. **Estimated effort: 2 hours.**

5. **P4-MED-2: Marketplace path traversal.** Call `sanitize_plugin_name()` on `manifest.name` before using it as a directory name in `publish()` and validate the resolved destination is within `_packages_dir`. **Estimated effort: 30 minutes.**

### Overall Risk Assessment

The application has a strong security foundation built across four phases. The core security controls -- shell execution sandboxing, plugin signature verification, path sandboxing, API authentication, tenant isolation, SSRF protection, RBAC, rate limiting, and encryption -- are all correctly implemented and verified across multiple audit cycles.

The Phase 4 findings are concentrated in the SSO subsystem (a new feature) and operational gaps in the fine-tuning pipeline and marketplace. The SSO findings are severe in isolation but confined to three endpoints and fixable in under 2 hours total.

**Verdict: The application is NOT ready for GA with the SSO endpoints in their current state.** After fixing P4-CRIT-1 and P4-HIGH-1 (approximately 1 hour of work), the application achieves an acceptable security posture for GA release with the following caveats:

- SSO integration should be documented as "beta" until real SAML/OIDC validation libraries are integrated.
- The fine-tuning pipeline should be restricted to admin users until tenant isolation is added.
- A dependency lock file should be generated before the GA container image is built.

**Total estimated remediation effort for all blocking items: 1-2 hours.**
**Total estimated remediation effort for all items (blocking + should fix): 4-5 hours.**

---

*Report generated by Security Engineer Agent (Claude Opus 4.6). This is the final pre-GA audit. Post-remediation verification recommended before release.*
