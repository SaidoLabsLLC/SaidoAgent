# Saido Agent — Architecture Review: Phase 1

**Product:** Saido Agent
**Phase:** 1 — Foundation
**Date:** 2026-04-05
**Status:** In Progress — Updated as build progresses

---

## Subsystem Scores

| Subsystem | Score | Notes |
|-----------|-------|-------|
| Core Agent Loop | TBD | Refactored from nano-claude-code |
| Security Hardening | TBD | CRIT-1,2,3 + HIGH-1,2,3,4 |
| Model Routing | TBD | Local-first with cloud escalation |
| SmartRAG Integration | TBD | Bridge pattern |
| Knowledge Pipeline | TBD | Ingest → Compile → Q&A |
| Memory System | TBD | Session persistence + extraction |
| SDK Public API | TBD | SaidoAgent class |
| CLI/REPL | TBD | Knowledge commands |

Scoring: Green (solid) / Yellow (needs attention) / Red (rework required)

---

## 1. Module Boundaries & Separation of Concerns

### Package Structure
The codebase follows a clean layered architecture:
- `core/` — Engine layer (agent loop, providers, tools, permissions, routing)
- `knowledge/` — Knowledge layer (bridge, ingest, compile, query, structural)
- `memory/` — Persistence layer (store, context, extraction)
- `cli/` — Interface layer (REPL, commands)
- `plugins/`, `mcp/`, `multi_agent/`, `tasks/`, `skills/` ��� Extension layers

### Key Boundary: Saido Agent ↔ SmartRAG
SmartRAG handles ALL retrieval, storage, indexing, and splitting. Saido Agent adds LLM intelligence, code analysis, and agent orchestration on top. The boundary is enforced through KnowledgeBridge — no direct SmartRAG imports outside knowledge/bridge.py.

*To be updated after Wave 3-5 complete.*

---

## 2. Data Flow: Ingest → Compile → Wiki Store → Q&A

```
File/URL → IngestPipeline
  ├── SmartRAG.ingest() → extract, split, dedup, store, index
  ├── StructuralAnalyzer.analyze() → CodeStructure (code files only)
  └��─ compile_queue.append(slug)
       │
       ▼
WikiCompiler.compile(slug)
  ├── Read article from SmartRAG
  ├── LLM enrichment (synopsis, concepts, categories, backlinks)
  └��─ SmartRAG.update() → propagate to master index, FTS5
       │
       ▼
KnowledgeQA.query(question)
  ├── SmartRAG.query() → tiered retrieval (Tier 0→1→2→3)
  ├── Read top articles via bridge
  ├── LLM answer generation with citations
  └── Return SaidoQueryResult
```

*To be evaluated after implementation complete.*

---

## 3. Security Architecture

### Defense-in-Depth Layers
1. **Shell execution:** Command parser → blocklist → sensitive path check → audit log
2. **File operations:** PathSandbox → sensitive dir deny → symlink check → audit log
3. **Plugin system:** Signature verification → import sandbox → hash-pinned deps → permission display
4. **Credentials:** OS keyring → Fernet fallback → secret redaction in sessions
5. **MCP:** Command approval → metacharacter rejection → allowlist

*Full security audit to be conducted at Phase 1 gate.*

---

## 4. Architectural Debt & Phase 2 Blockers

*To be populated after full Phase 1 build is complete.*

---

*Last updated: 2026-04-05 — Wave 2 complete, Wave 3 in progress*
