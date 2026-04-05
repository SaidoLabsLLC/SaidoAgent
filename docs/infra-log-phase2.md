# Saido Agent — Infrastructure Log: Phase 2

**Product:** Saido Agent
**Phase:** 2 — Cloud + Intelligence
**Date Started:** 2026-04-05
**Status:** Complete — All 10 build prompts executed, 789+ tests passing

---

## 1. New Infrastructure Decisions

| Decision | Choice | Alternatives Considered | Rationale | Limitations |
|----------|--------|------------------------|-----------|-------------|
| REST API framework | FastAPI | Flask, Django REST | Async-ready, Pydantic validation, OpenAPI auto-docs | Adds uvicorn dependency |
| Authentication | API key + JWT (HS256) | OAuth2, session cookies | Simple for API consumers, JWT for session state | No key rotation mechanism yet |
| Rate limiting | In-memory sliding window | Redis, token bucket | Zero new dependencies for single-process | Not distributed; resets on restart |
| Tenant isolation | Per-tenant knowledge_dir | Database-level isolation | Maps cleanly to SmartRAG's directory model | File system based, not database |
| Container runtime | Docker (multi-stage) | Podman, direct deploy | Universal, CI/CD integration | Requires Docker on dev machines |
| Database | SQLite (local/single-node) | PostgreSQL, Supabase | Zero new infrastructure for Phase 2 | Single-writer, not distributed |
| SSRF protection | ipaddress module validation | proxy-based, DNS pinning | No external dependencies, covers RFC1918 + metadata | No DNS rebinding protection |
| HTML extraction | BeautifulSoup + lxml | readability, trafilatura | Mature, well-tested, fine-grained control | Heavier than alternatives |
| Embeddings | SmartRAG[embeddings] (opt-in) | Standalone Chroma, Pinecone | Zero new infrastructure, SmartRAG handles everything | Requires extra install |

## 2. API Server Architecture

```
Client → FastAPI (uvicorn)
  ├── GET /health (no auth)
  ├── POST /v1/auth/token (API key → JWT)
  ├── POST /v1/auth/keys (create key)
  ├── POST /v1/ingest (file/content)
  ├── POST /v1/ingest/upload (multipart)
  ├── POST /v1/query (SSE streaming)
  ├── POST /v1/agent (full loop)
  ├── POST /v1/clip (web clipper)
  ├── GET /v1/documents (list)
  ├── GET /v1/documents/{slug} (detail)
  ├── GET /v1/search?q= (search)
  └── GET /v1/stats (statistics)
```

Auth flow: `X-API-Key` header or `Bearer` JWT token → `get_current_tenant` dependency → scoped SaidoAgent instance.

## 3. New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | >=0.110.0 | REST API framework |
| uvicorn | >=0.27.0 | ASGI server |
| python-multipart | >=0.0.9 | File upload support |
| pyjwt | >=2.8.0 | JWT token handling |
| beautifulsoup4 | >=4.12.0 | HTML extraction |
| lxml | >=5.0.0 | HTML parser backend |

## 4. Knowledge Intelligence Features

| Feature | Module | Storage | LLM Required |
|---------|--------|---------|-------------|
| Wiki linting | knowledge/lint.py | saido/lint_history.json | Yes (duplicates, contradictions) |
| Concept maps | knowledge/index.py | saido/concept_map.json | Yes |
| Category trees | knowledge/index.py | saido/category_tree.json | Yes |
| Enriched summaries | knowledge/index.py | Article frontmatter | Yes |
| Report generation | knowledge/outputs.py | outputs/reports/ | Yes |
| Knowledge export | knowledge/outputs.py | outputs/exports/ | No |

## 5. Security Patches (MED-1 through MED-4)

| Finding | Remediation | Status |
|---------|-------------|--------|
| MED-1: SSRF on WebFetch | validate_url() wired into _webfetch and _websearch | Implemented |
| MED-2: Regex DoS | Pattern complexity validator + 5s timeout + 1000 result limit | Implemented |
| MED-3: Token budget | Per-session limits (1M tokens, 200 turns) with 80%/100% thresholds | Implemented |
| MED-4: Memory trust | Untrusted project dir warning + trusted_projects.json | Implemented |

## 6. Plugin System v2 Enhancements

- Manifest v2 with required fields (version, author, license)
- Plugin update mechanism with version comparison
- Dependency resolution with circular detection
- Test framework with shadow detection
- CLI commands: /plugin update, update-all, test, list, info

## 7. MCP Integration

- MCPIngestBridge: pipe MCP tool results into knowledge store
- 4 recipe templates: ast-grep, Gmail, Google Drive, Slack
- Auto-ingest configuration per tool
- CLI: /ingest-mcp, /mcp-setup, /mcp-auto

## 8. Test Coverage

| File | Tests | Coverage |
|------|-------|----------|
| test_lint.py | 39 | Wiki health checks |
| test_api.py | 22 | REST API endpoints + auth |
| test_plugin_v2.py | 46 | Plugin manifest v2 + updates |
| test_index.py | 30 | Concept maps + indexing |
| test_web_ingest.py | 64 | URL ingest + SSRF |
| test_mcp_ingest.py | 23 | MCP-to-ingest bridge |
| test_embeddings_integration.py | 29 | SmartRAG embeddings |
| test_outputs.py | 21 | Report generation |
| test_med_security.py | 38 | Medium security patches |
| **Phase 2 total** | **312** | |
| **Cumulative total** | **789+** | |

---

*Last updated: 2026-04-05 — Phase 2 build complete, gate reviews in progress*
