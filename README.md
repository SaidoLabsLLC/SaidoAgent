# Saido Agent

**Knowledge-compounding AI agent platform by [Saido Labs LLC](https://saidolabs.com)**

Saido Agent ingests your documents, compiles them into a searchable knowledge wiki, and answers questions grounded in that knowledge — with citations. It runs locally by default (zero API cost), embeds into any Python app as an SDK, and scales to a multi-tenant cloud service with a web UI.

> Every interaction makes it smarter. Every document compounds your knowledge. No throwaway conversations.

---

## What It Does

- **Ingest anything** — Markdown, PDF, code files, URLs, web clips, MCP tool outputs
- **Compile into a wiki** — LLM-enriched articles with backlinks, concepts, and categories
- **Answer with citations** — Knowledge-grounded Q&A that cites specific articles
- **Run locally for free** — Default to Ollama (Qwen 3, Llama, DeepSeek) at $0.00/query
- **Embed in your apps** — `pip install saido-agent` and 5 lines of Python
- **Deploy as a service** — FastAPI REST API with auth, tenants, and a React web UI

## Quick Start

```bash
# Install
pip install -e .

# Make sure Ollama is running with a model
ollama pull qwen3:8b

# Use as SDK
python -c "
from saido_agent import SaidoAgent

agent = SaidoAgent(knowledge_dir='./my_knowledge')
agent.ingest('./docs/')
result = agent.query('What authentication method does the API use?')
print(result.answer)
print('Sources:', [c.title for c in result.citations])
"
```

### Web UI

```bash
python -m uvicorn saido_agent.api.server:app --port 8000
# Open http://localhost:8000
```

### CLI

```bash
saido-agent
# Then: /ingest ./docs/
#        /search authentication
#        /docs
#        /cost
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        SAIDO AGENT                           │
│                                                              │
│  Documents ──▶ SmartRAG ──▶ Knowledge Wiki ──▶ Grounded Q&A  │
│  (md/pdf/code/url)  (tiered retrieval)   (citations + LLM)  │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    AGENT CORE                            │ │
│  │  Local-first LLM routing · Cost tracking · Security     │ │
│  │  Plugin marketplace · Sub-agents · Voice SDK · Memory    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  CLI/REPL  ·  REST API  ·  Python SDK  ·  React Web UI      │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

### Knowledge Engine
- **SmartRAG integration** — Tiered retrieval (master index → frontmatter → sections → full content), FTS5 acceleration, optional semantic embeddings
- **Auto-grounding** — Every conversation automatically searches the knowledge store and injects relevant context
- **LLM compile** — Raw documents enriched with summaries, concepts, categories, and backlinks
- **Wiki linting** — Dead link detection, orphan articles, stale content, contradiction checking

### LLM & Model Routing
- **Local-first** — Ollama auto-detection, routes to local models by default
- **Multi-provider** — Anthropic, OpenAI, Gemini, DeepSeek, Ollama, LM Studio, any OpenAI-compatible endpoint
- **Per-task routing** — Configure which model handles ingest vs. Q&A vs. code review
- **Cost tracking** — Token usage per provider, estimated savings vs. all-cloud

### Security (23 findings, 23 fixed)
- Path sandboxing on all file operations
- Shell command parsing with injection prevention
- Plugin Ed25519 signature verification + import sandbox
- API key storage via OS keyring (no plaintext)
- Session encryption (Fernet), SSRF protection, regex DoS prevention
- RBAC (admin/editor/viewer), JWT auth, rate limiting

### Interfaces
- **Python SDK** — `SaidoAgent` class with `ingest()`, `query()`, `search()`, `run()`
- **REST API** — FastAPI with 20+ endpoints, SSE streaming, tenant isolation
- **CLI/REPL** — Terminal agent with 25+ slash commands
- **React Web UI** — Chat, document browser, ingest, cost dashboard, settings

### Voice SDK
- STT: FasterWhisper (local), Deepgram (cloud)
- TTS: Kokoro, Piper (local), Edge-TTS (free cloud), ElevenLabs (premium)
- VAD: Silero + energy-based fallback
- Pipeline orchestrator with latency instrumentation

### Enterprise
- Multi-user teams with RBAC
- Stripe billing (Free/Pro/Team/Enterprise tiers)
- Append-only audit logging
- GDPR compliance (data export, deletion, consent)
- SSO framework (SAML/OIDC)
- Plugin marketplace with automated security scanning

## Embedding in Your Apps

```python
from saido_agent import SaidoAgent

# Feastrio — meal planning with food knowledge
food_brain = SaidoAgent(
    knowledge_dir="./feastrio_knowledge",
    system_prompt_extension="You are Feastrio's food intelligence engine."
)
food_brain.ingest("./data/recipes/")
plan = food_brain.query("High-protein meals under 500 calories for someone lactose intolerant")

# Footura — sports analytics
sports_brain = SaidoAgent(
    knowledge_dir="./footura_knowledge",
    system_prompt_extension="You are Footura's sports analytics engine."
)
sports_brain.ingest("./data/scouting_reports/")
analysis = sports_brain.query("Compare Player A's last 5 matches against our midfielder criteria")
```

## Project Stats

| Metric | Value |
|--------|-------|
| Python source | 25,580 lines across 85 files |
| Test suite | 1,276 tests (17,238 lines) |
| Frontend | 19 React/TypeScript files |
| Security audits | 4 phases, 23 findings, all resolved |
| Build prompts | 34 executed + 12 gate reviews |

## Tech Stack

- **Runtime:** Python 3.11+
- **Retrieval:** [SmartRAG](https://github.com/SaidoLabsLLC/SmartRAG) (tiered retrieval, FTS5, optional embeddings)
- **Local LLM:** Ollama (Qwen 3, Llama, DeepSeek, Mistral)
- **Cloud LLM:** Anthropic Claude, OpenAI GPT, Google Gemini
- **API:** FastAPI + uvicorn
- **Frontend:** React + Vite + Tailwind
- **Auth:** JWT + API keys + RBAC
- **Security:** Ed25519 signatures, Fernet encryption, OS keyring
- **Code analysis:** ast-grep (structural AST search, 6 language patterns)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

This product includes software originally developed by SafeRL-Lab (UC Berkeley). See [NOTICE](NOTICE) for attribution.

---

Built by [Saido Labs LLC](https://saidolabs.com)
