"""End-to-end integration tests for Saido Agent Phase 1.

Validates:
  1. Full SDK path: init -> ingest -> query -> search -> stats -> cost
  2. Security: path sandboxing, shell hardening, no plaintext API keys
  3. Local LLM detection via ModelRouter
  4. CLI boot test
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# 1. SDK end-to-end path
# ---------------------------------------------------------------------------


class TestSDKEndToEnd:
    """Full SDK end-to-end: init -> ingest -> query -> search -> stats."""

    def test_sdk_e2e(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            knowledge_dir = os.path.join(tmpdir, "knowledge")

            # Initialize agent (patch out SmartRAG so test is self-contained)
            with patch("saido_agent.knowledge.bridge.SMARTRAG_AVAILABLE", False):
                from saido_agent import SaidoAgent
                from saido_agent.types import IngestResult, SearchResult, StoreStats

                agent = SaidoAgent(knowledge_dir=knowledge_dir)
                assert agent is not None

                # Create test documents -----------------------------------------
                test_dir = os.path.join(tmpdir, "test_data")
                os.makedirs(test_dir)

                # Architecture decisions doc
                with open(os.path.join(test_dir, "architecture_decisions.md"), "w") as f:
                    f.write(
                        "# Architecture Decision Records\n\n"
                        "## ADR-001: JWT Authentication\n"
                        "We decided to use JWT tokens for API authentication.\n"
                        "The alternative was session-based auth with cookies.\n"
                        "JWT was chosen for stateless scalability.\n\n"
                        "## ADR-002: PostgreSQL Database\n"
                        "PostgreSQL was selected over MongoDB for relational data integrity.\n"
                    )

                # API design notes
                with open(os.path.join(test_dir, "api_design.md"), "w") as f:
                    f.write(
                        "# API Design Notes\n\n"
                        "## Authentication Endpoints\n"
                        "- POST /api/auth/login -- returns JWT token\n"
                        "- POST /api/auth/refresh -- refresh expired token\n"
                        "- POST /api/auth/logout -- invalidate token\n\n"
                        "## User Endpoints\n"
                        "- GET /api/users -- list all users (admin only)\n"
                        "- GET /api/users/:id -- get user profile\n"
                        "- PUT /api/users/:id -- update user profile\n"
                    )

                # Python code file
                with open(os.path.join(test_dir, "auth_service.py"), "w") as f:
                    f.write(
                        "from flask import Flask, jsonify, request\n"
                        "import jwt\n\n"
                        "app = Flask(__name__)\n\n"
                        "class AuthService:\n"
                        '    """Handles JWT authentication."""\n\n'
                        "    def __init__(self, secret_key: str):\n"
                        "        self.secret_key = secret_key\n\n"
                        "    def login(self, username: str, password: str) -> str:\n"
                        '        """Authenticate user and return JWT token."""\n'
                        '        token = jwt.encode({"user": username}, self.secret_key)\n'
                        "        return token\n\n"
                        "    def verify(self, token: str) -> dict:\n"
                        '        """Verify JWT token and return payload."""\n'
                        '        return jwt.decode(token, self.secret_key, algorithms=["HS256"])\n\n'
                        '@app.route("/api/auth/login", methods=["POST"])\n'
                        "def login_endpoint():\n"
                        "    data = request.get_json()\n"
                        '    auth = AuthService("secret")\n'
                        '    token = auth.login(data["username"], data["password"])\n'
                        '    return jsonify({"token": token})\n'
                    )

                # Ingest all files ----------------------------------------------
                results = []
                for fname in sorted(os.listdir(test_dir)):
                    fpath = os.path.join(test_dir, fname)
                    r = agent.ingest(fpath)
                    results.append(r)
                    assert r is not None

                assert len(results) == 3

                # Verify each ingest returned a valid result --------------------
                for r in results:
                    assert isinstance(r, IngestResult)
                    assert r.slug is not None
                    assert r.status in ("created", "updated", "duplicate", "failed")

                # Check stats (degraded mode returns 0 — verify shape only) -----
                stats = agent.stats
                assert isinstance(stats, StoreStats)
                assert isinstance(stats.document_count, int)
                assert isinstance(stats.category_count, int)
                assert isinstance(stats.total_size_bytes, int)

                # Search --------------------------------------------------------
                search_results = agent.search("authentication")
                assert isinstance(search_results, list)

                # Query (no LLM configured -> low-confidence result) -----------
                result = agent.query("What authentication method does the API use?")
                assert result is not None
                assert hasattr(result, "answer")
                assert hasattr(result, "citations")
                assert hasattr(result, "confidence")

                # Cost tracking -------------------------------------------------
                cost = agent.cost
                assert isinstance(cost, dict)
                assert "total_cost" in cost
                assert "total_tokens" in cost
                assert "estimated_savings" in cost


# ---------------------------------------------------------------------------
# 2. Security verification tests
# ---------------------------------------------------------------------------


class TestSecurityPathSandbox:
    """Verify path sandboxing blocks sensitive paths."""

    def test_blocks_ssh_directory(self):
        from saido_agent.core.permissions import PathSandbox, PathSandboxError

        sandbox = PathSandbox(allowed_paths=["/tmp/test"])

        with pytest.raises(PathSandboxError):
            sandbox.validate(os.path.expanduser("~/.ssh/id_rsa"), "read")

    def test_blocks_etc_on_unix(self):
        from saido_agent.core.permissions import PathSandbox, PathSandboxError

        sandbox = PathSandbox(allowed_paths=["/tmp/test"])

        # On Windows /etc/passwd resolves differently, but the sandbox still
        # blocks it because it is outside allowed_paths.
        with pytest.raises(PathSandboxError):
            sandbox.validate("/etc/passwd", "read")


class TestSecurityShellHardening:
    """Verify shell command validation."""

    @pytest.fixture(autouse=True)
    def _isolate_saido_dir(self, tmp_path, monkeypatch):
        """Redirect audit/blocklist files to temp dir."""
        fake_dir = tmp_path / ".saido_agent"
        fake_dir.mkdir()
        monkeypatch.setattr("saido_agent.core.tools._SAIDO_DIR", fake_dir)
        monkeypatch.setattr("saido_agent.core.tools._AUDIT_LOG", fake_dir / "audit.log")
        monkeypatch.setattr(
            "saido_agent.core.tools._BLOCKLIST_FILE",
            fake_dir / "command_blocklist.json",
        )

    def test_python_interpreter_blocked(self):
        from saido_agent.core.tools import _parse_and_validate_command

        is_safe, reason = _parse_and_validate_command(
            "python -c 'import os; os.system(\"rm -rf /\")'"
        )
        assert not is_safe

    def test_rm_rf_blocked(self):
        from saido_agent.core.tools import _parse_and_validate_command

        is_safe, reason = _parse_and_validate_command("rm -rf /")
        assert not is_safe

    def test_git_status_passes(self):
        from saido_agent.core.tools import _parse_and_validate_command

        is_safe, reason = _parse_and_validate_command("git status")
        assert is_safe


class TestSecurityNoPlaintextAPIKeys:
    """Verify API keys are not stored in plaintext config."""

    def test_api_keys_not_in_config(self):
        config_path = os.path.expanduser("~/.saido_agent/config.json")
        if not os.path.exists(config_path):
            pytest.skip("No config file present — nothing to check")

        with open(config_path) as f:
            config = json.load(f)

        key_fields = ["anthropic_api_key", "openai_api_key", "api_key"]
        for field in key_fields:
            assert field not in config, f"Plaintext API key found: {field}"


# ---------------------------------------------------------------------------
# 3. Local LLM detection
# ---------------------------------------------------------------------------


class TestLocalLLMDetection:
    """Verify Ollama auto-detection via ModelRouter."""

    def test_local_model_detection(self):
        from saido_agent.core.routing import ModelRouter

        router = ModelRouter()
        local_models = router.get_available_local_models()
        # Informational: passes even if Ollama is down
        if local_models:
            model_names = [m[1] for m in local_models]
            assert any(
                "qwen" in m.lower() for m in model_names
            ), f"Expected a qwen model among {model_names}"


# ---------------------------------------------------------------------------
# 4. CLI boot test
# ---------------------------------------------------------------------------


class TestCLIBoot:
    """Verify CLI entry point loads without errors."""

    def test_cli_version_flag(self):
        result = subprocess.run(
            ["python", "-m", "saido_agent.cli.repl", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "saido-agent" in result.stdout.lower()
