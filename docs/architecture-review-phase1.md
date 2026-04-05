# Saido Agent — Architecture Review: Phase 1

**Product:** Saido Agent
**Phase:** 1 — Foundation
**Date:** 2026-04-05
**Reviewer:** Architecture Review Agent (automated)

---

## Executive Summary

Saido Agent Phase 1 delivers a well-structured, multi-provider AI agent platform with strong separation of concerns between the core agent loop, knowledge pipeline, model routing, security, and memory systems. The codebase demonstrates consistent patterns, good error handling, and thoughtful extensibility points. However, there are Yellow-grade concerns around knowledge grounding integration in the main agent loop, some tight coupling in the REPL's `_knowledge_context` dict, and the absence of an async execution model that will become a blocker for Phase 2 REST API/multi-tenancy.

## Subsystem Scores

| Subsystem | Score | Key Findings |
|-----------|-------|-------------|
| Core Agent Loop | **Green** | Clean generator-based loop, neutral message format, good provider abstraction |
| Security Hardening | **Green** | PathSandbox, shell command validation, plugin sandboxing, session encryption all solid |
| Model Routing | **Green** | Local-first routing, offline detection, escalation logic, cost tracking well-designed |
| SmartRAG Integration | **Green** | Clean delegation boundary, no duplicate retrieval logic, graceful degradation |
| Knowledge Pipeline | **Yellow** | Ingest/compile/query flow coherent, but knowledge not injected into main agent loop system prompt |
| Memory System | **Yellow** | File-based persistence works, extraction heuristics sound, but linear search won't scale |
| SDK Public API | **Green** | Clean facade, proper type exports, lazy initialization, good separation |
| CLI/REPL | **Yellow** | All commands wired, but knowledge context passed via mutable dict; REPL is synchronous |

---

## Detailed Findings

### 1. Core Agent Loop
**Score: Green**

The agent loop in `saido_agent/core/agent.py` is well-designed:
- **Generator-based streaming**: The `run()` function yields typed event objects (`TextChunk`, `ThinkingChunk`, `ToolStart`, `ToolEnd`, `TurnDone`, `PermissionRequest`), making it easy for different frontends (CLI, future REST API) to consume.
- **Neutral message format**: Messages use a provider-independent dict format, with conversion to Anthropic/OpenAI formats handled in `providers.py`.
- **Model routing integration**: The router is consulted via `state.router.select_model(task_type)`, which is clean. However, `task_type` is passed via `config["_task_type"]` which is somewhat ad-hoc.
- **No tight coupling**: The loop delegates tool execution through `tool_registry.py` which uses a plugin-style registry pattern. Provider selection in `providers.py` uses a prefix-detection strategy that supports 10+ providers cleanly.

**Concern**: The `config` dict is used as a catch-all for runtime state (`_depth`, `_system_prompt`, `_task_type`, `_knowledge_context`). A dedicated `RunContext` dataclass would be cleaner.

### 2. Security Hardening
**Score: Green**

Security implementation is thorough across multiple layers:

- **CRIT-3 PathSandbox** (`permissions.py`): Blocks `..` traversal before path resolution, validates against hardcoded sensitive dirs, checks symlink resolution, writes audit log. Sensitive dirs are non-configurable (security invariant).
- **Shell command validation** (`tools.py`): Multi-layer defense: blocklist file check, sensitive path check, shell metacharacter analysis, pipeline segment validation, safe binary allowlist, blocked interpreter list. Private IP blocking for curl/wget.
- **Plugin sandboxing** (`plugins/sandbox.py`): Import restriction via `builtins.__import__` replacement during plugin module execution. Blocked modules include `os`, `subprocess`, `shutil`, `socket`, `http`, `urllib`, `ctypes`.
- **Plugin verification** (`plugins/verify.py`): Ed25519 signature verification for plugin manifests, trusted registry classification, source classification.
- **Session encryption** (REPL): Fernet encryption with key stored in OS keyring, secret redaction before save, 30-day session expiry cleanup.
- **Dependency pinning**: `DependencyPin` requires sha256 hash per package in `plugins/types.py`.

**No significant bypass vectors identified.** One minor note: the plugin sandbox's `builtins.__import__` replacement is not thread-safe, but Phase 1 is single-threaded so this is acceptable.

### 3. Model Routing
**Score: Green**

`routing.py` implements a complete local-first routing engine:

- **Provider probing**: Ollama at `/api/tags`, LM Studio at `/v1/models`, with 3-second timeouts.
- **Internet check**: Tests Anthropic API reachability, correctly handles HTTP errors.
- **Selection logic**: Per-task-type routing config, with local/cloud preference. Falls back gracefully.
- **Offline mode**: Detected automatically, routes everything local. Escalation correctly blocked.
- **Cost tracking**: Tracks per-provider/model token usage, computes actual cost vs. cloud-equivalent for savings estimation.

### 4. SmartRAG Integration
**Score: Green**

`knowledge/bridge.py` properly delegates ALL retrieval/storage to SmartRAG:

- **No duplicate retrieval logic**: No chunk/split logic exists in `saido_agent/knowledge/`.
- **Clean CRUD wrappers**: All delegate to SmartRAG's store.
- **Saido-specific extensions**: Code structure frontmatter, compile orchestration, and backlink extraction are clearly Saido-layer features.
- **Graceful degradation**: `SMARTRAG_AVAILABLE` flag with sentinel types. Every method guards with `_require_rag()`.

**One concern**: The bridge accesses `self._rag._store` (private member). If SmartRAG refactors `_store`, it will break. Upstream public API methods recommended.

### 5. Knowledge Pipeline
**Score: Yellow**

The ingest -> compile -> query data flow is coherent:

- **Ingest**: Detects file type, delegates storage to bridge, runs structural analysis for code files, queues for compile.
- **Compile**: Reads document + frontmatter, builds prompt from template, calls LLM, parses/validates JSON with retry, updates article via bridge.
- **Query**: Retrieves via bridge, builds grounded prompt, calls LLM, extracts/validates citations, assesses confidence.

**Yellow because:**
1. Knowledge not injected into main agent loop — grounding only works via explicit `/search` or SDK `query()`, not in the default chat mode.
2. LLM call duplication between `compile.py._call_llm()` and `query.py._call_llm()` — should be a shared utility.

### 6. Memory System
**Score: Yellow**

- **Storage**: File-based with frontmatter metadata, user + project scopes, index file generation.
- **Extraction**: Heuristic regex-based extraction of decisions, facts, preferences. Phase 1 appropriate.
- **Integration**: `to_knowledge_article()` prepares data but isn't wired yet.

**Yellow because:**
1. Linear search — substring matching won't scale beyond ~100 entries.
2. Memory-knowledge bridge not wired yet.
3. File-based storage needs database backend for Phase 2 multi-tenancy.

### 7. SDK Public API
**Score: Green**

- **Complete surface**: `ingest()`, `query()`, `search()`, `run()`, `compile()`, `stats`, `cost`.
- **Lazy initialization**: Heavy imports deferred to `__init__` body.
- **Type definitions**: All use dataclasses with proper Optional typing and default factories.
- **Integration contract**: `__all__` exports explicit. Internal modules documented.

### 8. CLI/REPL
**Score: Yellow**

- **Knowledge commands**: All registered and implemented.
- **Startup banner**: Correct status display.

**Yellow because:**
1. Mutable dict as context bus — fragile, should use frozen dataclass.
2. Synchronous execution — Phase 2 REST API needs async.
3. REPL file is ~1700+ lines — should split into modules.

---

## Architectural Debt

1. **Config dict as catch-all**: Runtime state mixed in single dict. Refactor to typed context objects.
2. **Duplicate LLM call patterns**: compile, query, and compaction all have similar provider-dispatch logic.
3. **SmartRAG private API access**: Bridge uses `self._rag._store` directly.
4. **Memory search is O(n)**: Substring search over all memory files.
5. **REPL monolith**: ~1700+ lines in single file.
6. **No async**: Entire codebase is synchronous.
7. **`_EXT_TO_LANGUAGE` defined twice**: Both ingest.py and structural.py define extension-to-language mappings.

## Phase 2 Blockers

1. **CRITICAL — No async execution model**: Agent loop, streaming, knowledge pipeline all synchronous. REST API needs async.
2. **CRITICAL — No tenant isolation**: AgentState, PathSandbox, ModelRouter, CostTracker, KnowledgeBridge are all single-tenant. Module-level singletons need conversion to instance-scoped.
3. **HIGH — Knowledge grounding gap**: Main agent loop doesn't auto-retrieve knowledge context.
4. **HIGH — Memory backend**: File-based needs database for multi-tenancy.
5. **MEDIUM — Plugin sandbox thread safety**: `builtins.__import__` replacement not thread-safe.

## Recommendations

1. **Before Phase 2**: Add knowledge auto-grounding to agent loop — validate core value proposition end-to-end.
2. **Extract shared LLM utility**: `saido_agent/core/llm_call.py` used by compile, query, compaction.
3. **Replace config dict with typed context**: `RunContext` and `KnowledgeContext` dataclasses.
4. **Plan async migration**: Generators -> async generators before starting REST API.
5. **Convert singletons to instance-scoped**: PathSandbox, tool registry, routing per-session/per-tenant.
6. **Split REPL**: Break into submodules (commands, rendering, initialization).
7. **Upstream SmartRAG public API**: Request public methods for `_store` operations.

---

## Gate Decision

**PROCEED to Phase 2 with conditions.** The architecture is sound and well-executed for Phase 1 scope. The two critical blockers (async migration and tenant isolation) are expected Phase 2 work. The knowledge auto-grounding gap should be addressed in a Phase 1.1 patch before Phase 2 begins.
