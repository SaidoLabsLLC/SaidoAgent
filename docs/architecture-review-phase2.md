# Saido Agent — Architecture Review: Phase 2

**Product:** Saido Agent
**Phase:** 2 — Cloud + Intelligence
**Date:** 2026-04-05
**Status:** Complete

---

## Executive Summary

Phase 2 successfully adds cloud-ready infrastructure (FastAPI REST API, tenant isolation, authentication), knowledge intelligence features (linting, indexing, concept maps, embeddings), web content ingestion (URL fetch, SSRF protection, web clipper), and medium security hardening. The architecture cleanly extends Phase 1's foundation without requiring rework.

## Subsystem Scores

| Subsystem | Score | Key Findings |
|-----------|-------|-------------|
| REST API | **Green** | Clean route design, Pydantic validation, SSE streaming, tenant isolation |
| Authentication | **Yellow** | API key + JWT works, but no key rotation, in-memory rate limits |
| Tenant Isolation | **Green** | Per-tenant knowledge dirs, scoped agent instances |
| LLM Linting | **Green** | 7 health checks, mix of deterministic and LLM-powered |
| LLM Indexing | **Green** | Concept maps, category trees, incremental with content hashing |
| SmartRAG Embeddings | **Green** | Clean pass-through, graceful degradation, opt-in |
| Web Ingestion | **Green** | SSRF protection, HTML extraction, web clipper endpoint |
| Plugin System v2 | **Green** | Manifest v2, updates, dep resolution, test framework |
| MCP Integration | **Green** | Ingest bridge, recipes, auto-ingest config |
| Report Generation | **Green** | Structured reports, zip export |
| Medium Security | **Green** | SSRF, regex DoS, token budget, memory trust all solid |

## Phase 3 Blockers

1. **Auth needs key rotation** — current API keys are permanent, no revocation beyond deleting from JSON
2. **Rate limiting is in-memory** — resets on restart, not distributed for multi-process
3. **No WebSocket infrastructure** — needed for real-time collaboration
4. **CORS is wildcard** — needs per-tenant/per-deployment configuration
5. **SQLite single-writer** — will need PostgreSQL for production multi-tenancy

## Recommendations

1. Add Redis-backed rate limiting before scaling beyond single process
2. Implement API key rotation and revocation
3. Add WebSocket support for real-time features (Phase 3)
4. Lock down CORS origins per deployment
5. Migration path from SQLite to PostgreSQL for cloud production

## Gate Decision

**PROCEED to Phase 3.** Architecture is clean and well-tested. Yellow items are expected Phase 3 work, not Phase 2 defects.
