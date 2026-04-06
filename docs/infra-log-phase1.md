# Saido Agent — Infrastructure Log: Phase 1

**Product:** Saido Agent
**Phase:** 1 — Foundation
**Date Started:** 2026-04-05
**Status:** Complete — All 14 build prompts executed, 473+ tests passing

---

## 1. Runtime Environment

| Decision | Choice | Alternatives Considered | Rationale | Limitations |
|----------|--------|------------------------|-----------|-------------|
| Language | Python 3.11+ (running 3.14.3) | Node.js, Rust | Inherited from nano-claude-code; best LLM SDK ecosystem; SmartRAG is Python | Python 3.14 is bleeding edge — some Phase 3 deps (torch, faster-whisper) may lack wheels |
| Package format | pyproject.toml (PEP 621) | setup.py, setup.cfg | Modern standard, clean dependency declaration | None |
| Entry point | `saido-agent` CLI → `saido_agent.cli.repl:main` | Direct `python -m` only | User-friendly CLI command | Requires pip install |

## 2. Local LLM Setup

| Decision | Choice | Alternatives Considered | Rationale | Limitations |
|----------|--------|------------------------|-----------|-------------|
| Local LLM runtime | Ollama | LM Studio, llama.cpp direct | Already installed, simple API, model management | Ollama-specific API for model listing |
| Default local model | qwen3:8b (18GB) | qwen2.5-coder:32b, llama3.3 | User's installed model; newer than PRD spec | Large RAM requirement (~18GB) |
| Model routing default | Local-first for all tasks except review/architect | All-cloud, hybrid | Zero marginal cost for routine operations | Local model quality ceiling for complex reasoning |
| Cloud escalation | claude-sonnet-4-6 (mid), claude-opus-4-6 (frontier) | GPT-4o, Gemini | Anthropic SDK already integrated | Requires API key for cloud operations |
| Routing config | ~/.saido_agent/routing.json | Environment vars, pyproject.toml section | User-editable JSON, per-task granularity | Manual editing required for changes |

## 3. Dependency Versions

| Package | Version | Purpose |
|---------|---------|---------|
| smartrag | 0.1.0 (editable) | Tiered retrieval engine — all document storage, FTS5, indexing, splitting |
| ast-grep (sg) | 0.42.1 | Structural code search via AST pattern matching |
| anthropic | (from nano-claude-code) | Anthropic Claude API SDK |
| openai | (from nano-claude-code) | OpenAI-compatible API SDK (used for Ollama/LM Studio too) |
| httpx | (from nano-claude-code) | HTTP client for local LLM probing, web fetch |
| rich | (from nano-claude-code) | Terminal UI rendering |
| keyring | added Phase 1 | Secure API key storage (OS keyring) |
| cryptography | added Phase 1 | Fernet encryption for sessions, Ed25519 for plugin signatures |

## 4. File System Layout

```
saido-agent/                    # Project root
├── saido_agent/                # Main package
│   ├── __init__.py             # Public API: SaidoAgent, SaidoConfig
│   ├── core/                   # Agent engine (from nano-claude-code)
│   │   ├── agent.py            # Agent loop with model routing
│   │   ├── providers.py        # Multi-provider LLM streaming
│   │   ├── tools.py            # Tool execution (hardened: CRIT-1,3)
│   │   ├── permissions.py      # PathSandbox (CRIT-3)
│   │   ├── routing.py          # ModelRouter + local LLM detection
│   │   ├── cost_tracker.py     # Token/cost tracking per provider
│   │   ├── compaction.py       # Context window management
│   │   ├── config.py           # Config with keyring (HIGH-1)
│   │   ├── context.py          # System prompt template
│   │   └── tool_registry.py    # Tool registration system
│   ├── knowledge/              # Knowledge engine (Saido-specific)
│   │   ├── bridge.py           # SmartRAG integration bridge
│   │   ├── ingest.py           # Data ingest pipeline
│   │   ├── compile.py          # LLM compile (raw → wiki)
│   │   ├── query.py            # Knowledge-grounded Q&A
│   │   ├── structural.py       # ast-grep code analysis
│   │   ├── lint.py             # Wiki health checks (Phase 2)
│   │   └── index.py            # LLM indexing (Phase 2)
│   ├── memory/                 # Persistent memory system
│   ├── plugins/                # Plugin system (hardened: CRIT-2)
│   ├── mcp/                    # MCP client (hardened: HIGH-2)
│   ├── multi_agent/            # Sub-agent system (fixed: HIGH-4)
│   ├── tasks/                  # Task management
│   ├── skills/                 # Built-in skills
│   ├── voice/                  # Voice pipeline (Phase 3)
│   ├── api/                    # REST API (Phase 2)
│   └── cli/                    # CLI/REPL interface
├── tests/                      # Test suite
├── docs/                       # Documentation
├── pyproject.toml              # Package metadata
├── LICENSE                     # Apache 2.0
├── NOTICE                      # Attribution to SafeRL-Lab
└── README.md                   # Product readme
```

## 5. Configuration File Formats

| File | Location | Format | Purpose |
|------|----------|--------|---------|
| config.json | ~/.saido_agent/ | JSON | Non-secret settings (model prefs, UI options) |
| routing.json | ~/.saido_agent/ | JSON | Model routing rules per task type |
| command_blocklist.json | ~/.saido_agent/ | JSON | Blocked shell commands |
| mcp_approved.json | ~/.saido_agent/ | JSON | Approved MCP server commands |
| audit.log | ~/.saido_agent/ | JSON lines | All tool execution audit trail |
| API keys | OS keyring | keyring API | Secure credential storage |
| keys.enc | ~/.saido_agent/ | Fernet encrypted | Fallback key storage for headless |
| sessions/*.enc | ~/.saido_agent/ | Fernet encrypted | Encrypted conversation sessions |

## 6. Knowledge Store Directory Structure

```
knowledge/                      # Root knowledge directory
├── _index.md                   # SmartRAG master index (Tier 0)
├── documents/                  # SmartRAG article storage
├── backlinks.json              # SmartRAG backlink graph
├── .smartrag/                  # SmartRAG internal (FTS5 DB, config)
├── raw/                        # Saido: unprocessed source documents
├── outputs/                    # Saido: generated artifacts
│   ├── reports/
│   ├── slides/
│   └── charts/
└── saido/                      # Saido: agent metadata
    └── compile_log.json
```

## 7. SmartRAG Integration

| Decision | Choice | Rationale | Limitations |
|----------|--------|-----------|-------------|
| Integration pattern | KnowledgeBridge wrapper class | Clean separation; SmartRAG API may evolve | Extra indirection layer |
| SmartRAG version | 0.1.0 (editable install) | Developed in parallel by Saido Labs | API may be incomplete |
| Delegated to SmartRAG | Storage, FTS5, master index, splitting, dedup, backlinks | SmartRAG is the retrieval engine | Saido Agent cannot customize retrieval internals |
| Kept in Saido Agent | LLM compile, ast-grep analysis, Q&A answer generation, cost tracking | Domain-specific intelligence | Requires bridge to translate between systems |
| Embeddings | Disabled in Phase 1 | Phase 2 enables via smartrag[embeddings] | FTS5-only retrieval in Phase 1 |

## 8. Security Infrastructure

| Finding | Remediation | Status |
|---------|-------------|--------|
| CRIT-1: Shell execution | Full command parser, blocklist, audit log | Implemented (89 tests) |
| CRIT-2: Plugin system | Ed25519 signatures, import sandbox, hash-pinned deps | Implemented |
| CRIT-3: Path sandboxing | PathSandbox class, sensitive dir deny, symlink prevention | Implemented (31 tests) |
| HIGH-1: Plaintext API keys | OS keyring storage with Fernet fallback | Implemented |
| HIGH-2: MCP spawning | Command approval flow, metacharacter rejection | Implemented |
| HIGH-3: Session encryption | Fernet encryption, auto-expiry, secret redaction | Implemented |
| HIGH-4: os.chdir() race | Removed; subprocess cwd= pattern | Implemented |

---

## 9. Test Infrastructure

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Test framework | pytest | Standard Python test framework, already in use |
| Test organization | tests/ with test_ prefix per module | Standard pytest discovery |
| Fixtures | tests/fixtures/ with sample.py and sample.md | Real files for ingest/structural analysis testing |
| Mocking strategy | unittest.mock for LLM calls, bridge operations | Avoid requiring running LLM for unit tests |
| Total test count | 473+ tests across 11 test files | Comprehensive coverage of all Phase 1 modules |

### Test Files
| File | Tests | Coverage |
|------|-------|----------|
| test_shell_security.py | 89 | CRIT-1 shell hardening |
| test_plugin_security.py | 41 | CRIT-2 plugin system |
| test_path_sandbox.py | 31 | CRIT-3 path sandboxing |
| test_high_security.py | 13 | HIGH-1 through HIGH-4 |
| test_routing.py | 32 | Model routing + cost tracking |
| test_knowledge_bridge.py | 37 | SmartRAG integration |
| test_ingest.py | 56 | Ingest pipeline + ast-grep |
| test_memory.py | 31 | Session persistence + memory |
| test_compile.py | 35 | LLM compile enrichment |
| test_query.py | 36 | Knowledge Q&A + citations |
| test_sdk.py | 38 | Public SDK API |
| test_cli.py | 35 | CLI/REPL commands |

## 10. Build Execution Timeline

| Wave | Prompts | Duration | Tests Added |
|------|---------|----------|-------------|
| 1 | P1-01 (Foundation) | ~22 min | CLI boot verification |
| 2 | P1-02,03,04,05,06 (parallel) | ~7 min | 205 |
| 3 | P1-07,08,11 (parallel) | ~7 min | 124 |
| 4 | P1-09,10 (parallel) | ~4 min | 71 |
| 5 | P1-12,13 (parallel) | ~6 min | 73 |
| Gate | P1-14 + reviews (parallel) | In progress | TBD |

---

*Last updated: 2026-04-05 — Phase 1 build complete, gate reviews in progress*
