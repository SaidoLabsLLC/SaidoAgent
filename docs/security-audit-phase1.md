# Saido Agent -- Security Audit: Phase 1 (Final)

**Auditor:** Security Engineer Agent (Claude Opus 4.6)
**Date:** 2026-04-05
**Scope:** All security-relevant code in `saido_agent/` post-Phase 1 hardening
**Methodology:** Manual source code review of every file referenced in the remediation plan, plus re-verification of all 4 re-audit findings

---

## Executive Summary

**Overall Assessment: PASS -- All Critical and High findings remediated. All re-audit findings fixed.**

Phase 1 security remediations are complete. All seven original tracked findings (CRIT-1 through HIGH-4) remain correctly implemented. All 4 re-audit findings (NEW-1 through NEW-4) have been addressed. One residual low-severity bug was discovered during this verification pass (see Residual Risks).

| Category | Status |
|----------|--------|
| CRIT-1: Shell Execution | **PASS** |
| CRIT-2: Plugin System | **PASS** |
| CRIT-3: Path Sandboxing | **PASS** |
| HIGH-1: API Key Storage | **PASS** |
| HIGH-2: MCP Server Spawning | **PASS** |
| HIGH-3: Session Encryption | **PASS** |
| HIGH-4: Sub-Agent Race | **PASS** |
| NEW-1: notebook_edit/ast_grep sandbox | **FIXED** |
| NEW-2: pathlib in plugin sandbox | **FIXED** |
| NEW-3: Static PBKDF2 salt | **FIXED** |
| NEW-4: Missing dependencies | **FIXED** |

---

## Original Remediations

### CRIT-1: Shell Execution -- PASS

**File:** `saido_agent/core/tools.py` (line 465)

`_parse_and_validate_command()` is implemented and enforces:
- Blocklist check for dangerous commands
- Sensitive path detection
- Shell metacharacter analysis with per-segment validation of pipelines
- Command substitution (`$()` and backticks) blocked
- Safe binary allowlist via `_validate_single_command()`
- Blocked interpreter list (`_BLOCKED_INTERPRETERS`)

**Verdict:** Correctly implemented. Defense-in-depth with multiple validation layers.

### CRIT-2: Plugin System -- PASS

**Files:** `saido_agent/plugins/verify.py`, `saido_agent/plugins/sandbox.py`

- Ed25519 signature verification via `verify_manifest_signature()` loads a bundled Saido Labs public key from `plugins/keys/saido_labs_public.pem`
- Signatures are base64-decoded, verified against canonical manifest bytes
- Plugin sandbox (`sandbox.py`) enforces import restrictions with `BLOCKED_MODULES` (os, subprocess, shutil, socket, http, urllib, ctypes)
- `DEFAULT_ALLOWED_MODULES` contains only safe stdlib modules (json, re, datetime, math, etc.)
- Restricted `__import__` hook intercepts all top-level and sub-module imports

**Verdict:** Correctly implemented. Signature-then-sandbox design is sound.

### CRIT-3: Path Sandboxing -- PASS

**File:** `saido_agent/core/permissions.py`

- `PathSandbox` class with hardcoded `SENSITIVE_DIRS` deny list (`.ssh`, `.aws`, `.gnupg`, `gcloud`, `/etc`, `/var`, `/proc`, `/sys`, `System32`)
- Symlink resolution to prevent escape via symbolic links
- Path traversal (`..`) blocking
- Audit logging of all file access decisions
- `PathSandboxError` exception for denied access
- `get_sandbox()` and `validate()` API used by all tool implementations

**Verdict:** Correctly implemented. Deny list is hardcoded and not user-configurable (security invariant).

### HIGH-1: API Key Storage -- PASS

**File:** `saido_agent/core/config.py`

- API keys stored via OS keyring (`keyring.set_password` / `keyring.get_password`)
- Fernet-encrypted fallback (`keys.enc`) for headless/CI environments
- Legacy plaintext keys in `config.json` are auto-migrated and removed on first load (`_migrate_legacy_keys`)
- `_API_KEY_FIELDS` list covers all provider key fields

**Verdict:** Correctly implemented. Dual storage strategy handles all deployment scenarios.

### HIGH-2: MCP Server Spawning -- PASS

**File:** `saido_agent/mcp/client.py` (line 52)

- `validate_mcp_command()` rejects shell metacharacters (`;`, `&&`, `||`, `|`, `$()`, backticks) via regex
- `check_mcp_approval()` calls validation unconditionally before any command execution
- Interactive user approval prompt for first-run commands
- Approved commands persisted for future sessions
- `SecurityError` raised for rejected commands

**Verdict:** Correctly implemented. Shell injection via MCP configs is blocked.

### HIGH-3: Session Encryption -- PASS

**File:** `saido_agent/cli/repl.py` (line 111)

- Session data encrypted with Fernet (`_encrypt_session` / `_decrypt_session`)
- Encryption key stored in OS keyring (`_get_session_fernet`)
- Secret redaction via `_redact_session_data` before encryption (defense-in-depth)
- Expired session cleanup (`_cleanup_expired_sessions`)

**Verdict:** Correctly implemented. Session files at rest are encrypted and secrets are redacted.

### HIGH-4: Sub-Agent Race Condition -- PASS

**File:** `saido_agent/multi_agent/subagent.py` (line 22)

- `os.chdir` monkey-patched to `_safe_chdir` at module import
- `_safe_chdir` raises `RuntimeError` when called from any non-main thread
- Guard installed automatically via `install_chdir_guard()` on import
- Sub-agents use `subprocess.run(cwd=target_dir)` pattern instead

**Verdict:** Correctly implemented. Thread-safety race condition is structurally prevented.

---

## Re-Audit Findings (All Fixed)

### NEW-1: notebook_edit/ast_grep sandbox -- FIXED

**Severity:** HIGH
**File:** `saido_agent/core/tools.py`

**Verification:**
- `_notebook_edit()` (line 768): Calls `get_sandbox().validate(notebook_path, "edit")` before any filesystem access. `PathSandboxError` is caught and returns error string.
- `_ast_grep()` (line 974): Calls `get_sandbox().validate(search_path, "glob")` before constructing the `sg` command. `PathSandboxError` is caught and returns error string.

**Verdict:** FIXED. Both functions now enforce path sandbox validation with proper error handling.

### NEW-2: pathlib in plugin sandbox -- FIXED

**Severity:** MEDIUM
**File:** `saido_agent/plugins/sandbox.py` (line 30)

**Verification:** `DEFAULT_ALLOWED_MODULES` contains: `json`, `re`, `datetime`, `math`, `collections`, `typing`, `dataclasses`, `enum`, `functools`, `itertools`, `abc`, `copy`, `hashlib`, `base64`, `textwrap`, `string`, `logging`. `pathlib` is NOT present.

**Verdict:** FIXED. `pathlib` removed from the allowed module list, preventing plugins from performing arbitrary filesystem operations.

### NEW-3: Static PBKDF2 salt -- FIXED

**Severity:** MEDIUM
**File:** `saido_agent/core/config.py` (line 108)

**Verification:**
- Comment on line 108 explicitly references "NEW-3 fix"
- Salt loaded from `CONFIG_DIR / "key_salt.bin"` if file exists
- On first use, salt generated with `os.urandom(32)` and persisted to `key_salt.bin`
- PBKDF2 derivation uses this per-installation random salt with 100,000 iterations

**Verdict:** FIXED. Each installation generates a unique 32-byte random salt, eliminating rainbow table attacks against the passphrase-derived key.

### NEW-4: Missing dependencies in pyproject.toml -- FIXED

**Severity:** LOW
**File:** `pyproject.toml` (lines 18-19)

**Verification:**
- `keyring>=25.0.0` is listed in `dependencies`
- `cryptography>=42.0.0` is listed in `dependencies`

**Verdict:** FIXED. Both security-critical dependencies are pinned with minimum versions.

---

## SmartRAG Integration Security

**File:** `saido_agent/knowledge/bridge.py`

Assessment of the SmartRAG integration layer:

- **Import safety:** SmartRAG is an optional dependency with graceful fallback (`SMARTRAG_AVAILABLE` flag). Missing dependency does not crash the agent.
- **Data flow:** The bridge is a thin CRUD wrapper. All retrieval, storage, indexing, and splitting logic is delegated to SmartRAG. No raw user input is passed to shell commands through this path.
- **Directory structure:** Uses a fixed layout (`raw/`, `outputs/reports/`, `saido/`) under the knowledge base root. No dynamic path construction from user input.
- **Wikilink parsing:** Regex-based (`_WIKILINK_RE`) for backlink extraction. Input is document content already stored in the knowledge base, not direct user input. No injection risk.

**Verdict:** No security concerns identified in the SmartRAG integration layer. The bridge delegates all heavy lifting to SmartRAG and does not introduce new attack surface.

---

## Residual Risks

### RES-1 (LOW): Undefined variable in `_ast_grep` -- `resolved` vs `search_path`

**File:** `saido_agent/core/tools.py`, line 988
**Severity:** LOW (functional bug, not exploitable)

The `_ast_grep` function stores the sandbox-validated path in `search_path` (line 978), but line 988 references `str(resolved)` which is undefined. This will raise a `NameError` at runtime when `_ast_grep` is called, making the tool non-functional but not exploitable. The sandbox validation itself executes correctly before this line.

**Remediation:** Change `cmd.append(str(resolved))` to `cmd.append(str(search_path))` on line 988.

### RES-2 (INFORMATIONAL): Fernet key fallback when keyring unavailable

**Files:** `saido_agent/core/config.py` (line 123), `saido_agent/cli/repl.py` (line 123)

When the OS keyring is unavailable and no `SAIDO_AGENT_KEY_PASSPHRASE` env var is set, a Fernet key is generated in memory and lost when the process exits. This means encrypted data (keys.enc, session files) cannot be decrypted across process restarts in keyring-less environments. This is a graceful degradation, not a security vulnerability -- data is still encrypted at rest, just not recoverable.

### RES-3 (INFORMATIONAL): Plugin key distribution

The Ed25519 public key for plugin verification is bundled at `saido_agent/plugins/keys/saido_labs_public.pem`. Key rotation would require a new agent release. Consider a key-pinning + rotation mechanism for future phases.

---

## Gate Decision

**PASS -- Phase 1 security requirements met.**

All 7 original findings (CRIT-1 through HIGH-4) are correctly implemented and verified. All 4 re-audit findings (NEW-1 through NEW-4) have been fixed and verified. The residual items are low-severity or informational and do not block release.

**Recommended before release:**
- Fix RES-1 (`resolved` -> `search_path` in `_ast_grep`) -- 1-line fix, no security impact but breaks tool functionality.

**Recommended for Phase 2:**
- Plugin key rotation mechanism (RES-3)
- Runtime security monitoring and alerting
- Automated security regression test suite
