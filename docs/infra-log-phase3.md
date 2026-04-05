# Saido Agent — Infrastructure Log: Phase 3

**Product:** Saido Agent
**Phase:** 3 — Collaboration + Voice + Monetization
**Date Started:** 2026-04-05
**Status:** In Progress

---

## 1. New Infrastructure Decisions

| Decision | Choice | Alternatives | Rationale | Limitations |
|----------|--------|-------------|-----------|-------------|
| RBAC | Custom role system (admin/editor/viewer) | Django auth, Casbin | Lightweight, fits existing JWT flow | No fine-grained resource-level permissions |
| Password hashing | hashlib.scrypt | bcrypt, argon2 | No new dependency, constant-time comparison | Less community review than bcrypt |
| Real-time | WebSocket via FastAPI | Socket.io, SSE-only | Native FastAPI support, bidirectional | No message persistence |
| Voice STT | faster-whisper (local default) | Deepgram, AssemblyAI | Zero API cost, runs on CPU | Requires model download |
| Voice VAD | Silero VAD + energy fallback | WebRTC VAD, Picovoice | Works without torch via energy fallback | Energy-based less accurate |
| Voice TTS | Kokoro (default), Voxtral (future) | ElevenLabs, OpenAI TTS | Lightweight, local-first | Voxtral placeholder until model available |
| Slides | Marp markdown format | reveal.js, PPTX | Text-based, version-controllable | Requires marp-cli for PDF export |
| Charts | Matplotlib in subprocess sandbox | Plotly, D3 | Proven, sandboxable | Subprocess overhead |
| Sub-agent isolation | Git worktree | Docker containers, chroot | No new dependencies, git-native | Requires git repo |
| Agent messaging | In-memory queues | Redis, ZMQ | Zero infrastructure | Not persistent across restarts |
| Frontend | React + Vite + Tailwind | Next.js, SvelteKit | Fast dev experience, wide ecosystem | SPA (no SSR) |
| Billing | Stripe | Paddle, LemonSqueezy | Market leader, best webhook system | Requires Stripe account |

## 2. RBAC Architecture

```
User → JWT Token (user_id, team_id, role)
  │
  ├── Admin: full access (manage members, billing, delete)
  ├── Editor: read/write (ingest, compile, edit, run agent)
  └── Viewer: read-only (query, search, view stats)
```

Team-scoped tenancy: team_id = tenant_id for knowledge store isolation.

## 3. Voice Pipeline Architecture

```
Audio → SileroVAD (speech detect) → FasterWhisperSTT (transcribe)
  → SaidoAgent.query() (knowledge-grounded)
    → KokoroTTS (synthesize, sentence-level streaming)
      → Audio output
```

All stages lazy-load models. Energy-based VAD fallback when torch unavailable.

## 4. Phase 3 Test Coverage

| File | Tests | Coverage |
|------|-------|----------|
| test_rbac.py | 33 | Users, teams, roles, WebSocket |
| test_voice.py | 61 | STT, VAD, TTS, pipeline, streaming |
| test_slides_charts.py | 41 | Marp slides, Matplotlib charts |
| test_subagent.py | 46 | Isolation, resources, messaging |
| test_billing.py | TBD | Stripe, tiers, metering |
| test_frontend.py | TBD | React scaffold validation |
| **Phase 3 total** | **181+** | |
| **Cumulative** | **1,000+** | |

---

*Last updated: 2026-04-05 — Wave A complete (1,000 tests), Wave B in progress*
