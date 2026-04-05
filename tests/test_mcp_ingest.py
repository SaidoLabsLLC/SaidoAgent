"""Tests for MCP-to-knowledge ingest bridge (Phase 2).

Covers:
- ingest_tool_result creates a knowledge article
- call_and_ingest calls tool then ingests (mock MCP)
- Auto-ingest config saves and loads
- Recipe files are valid JSON with required fields
- CLI commands registered
- MCP approval flow still enforced
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_auto_ingest(tmp_path, monkeypatch):
    """Redirect auto-ingest config to a temp file."""
    cfg_file = tmp_path / "mcp_auto_ingest.json"
    monkeypatch.setattr(
        "saido_agent.mcp.ingest_bridge._AUTO_INGEST_FILE", cfg_file
    )
    return cfg_file


class FakeBridge:
    """Minimal stand-in for KnowledgeBridge."""

    def __init__(self):
        self.articles: dict[str, dict] = {}

    def create_article(self, slug: str, body: str, frontmatter: dict = None):
        self.articles[slug] = {
            "slug": slug,
            "body": body,
            "frontmatter": frontmatter or {},
        }
        # Return a truthy object (simulates Document)
        return self.articles[slug]


class FakeMCPManager:
    """Minimal stand-in for MCPManager."""

    def __init__(self, results: dict[str, str] | None = None):
        self._results = results or {}
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, qualified_name: str, arguments: dict) -> str:
        self.calls.append((qualified_name, arguments))
        if qualified_name in self._results:
            return self._results[qualified_name]
        raise RuntimeError(f"Tool not found: {qualified_name}")


# ---------------------------------------------------------------------------
# MCPIngestBridge.ingest_tool_result
# ---------------------------------------------------------------------------

class TestIngestToolResult:
    """ingest_tool_result creates a knowledge article."""

    def test_creates_article_with_metadata(self):
        from saido_agent.mcp.ingest_bridge import MCPIngestBridge

        bridge = FakeBridge()
        ib = MCPIngestBridge(mcp_manager=None, knowledge_bridge=bridge)

        result = ib.ingest_tool_result(
            server_name="gmail",
            tool_name="read_message",
            result="Subject: Hello\n\nBody text here.",
            metadata={"thread_id": "abc123"},
        )

        assert result["status"] == "ok"
        assert result["slug"] is not None

        # Verify the article was stored in the bridge
        slug = result["slug"]
        assert slug in bridge.articles
        article = bridge.articles[slug]
        assert "Hello" in article["body"]
        assert article["frontmatter"]["source"] == "mcp"
        assert article["frontmatter"]["source_server"] == "gmail"
        assert article["frontmatter"]["source_tool"] == "read_message"
        assert article["frontmatter"]["thread_id"] == "abc123"
        assert "ingested_at" in article["frontmatter"]

    def test_returns_error_when_bridge_returns_none(self):
        from saido_agent.mcp.ingest_bridge import MCPIngestBridge

        bridge = mock.MagicMock()
        bridge.create_article.return_value = None
        ib = MCPIngestBridge(mcp_manager=None, knowledge_bridge=bridge)

        result = ib.ingest_tool_result("s", "t", "text")
        assert result["status"] == "error"
        assert "bridge returned None" in result["error"]

    def test_returns_error_on_exception(self):
        from saido_agent.mcp.ingest_bridge import MCPIngestBridge

        bridge = mock.MagicMock()
        bridge.create_article.side_effect = RuntimeError("db fail")
        ib = MCPIngestBridge(mcp_manager=None, knowledge_bridge=bridge)

        result = ib.ingest_tool_result("s", "t", "text")
        assert result["status"] == "error"
        assert "db fail" in result["error"]


# ---------------------------------------------------------------------------
# MCPIngestBridge.call_and_ingest
# ---------------------------------------------------------------------------

class TestCallAndIngest:
    """call_and_ingest calls tool then ingests (mock MCP)."""

    def test_calls_mcp_then_ingests(self):
        from saido_agent.mcp.ingest_bridge import MCPIngestBridge

        bridge = FakeBridge()
        mgr = FakeMCPManager(results={
            "mcp__slack__read_thread": "Thread contents here..."
        })

        ib = MCPIngestBridge(mcp_manager=mgr, knowledge_bridge=bridge)
        result = ib.call_and_ingest("slack", "read_thread", {"channel": "C01"})

        # MCP was called with the qualified name
        assert len(mgr.calls) == 1
        assert mgr.calls[0] == ("mcp__slack__read_thread", {"channel": "C01"})

        # Result was ingested
        assert result["status"] == "ok"
        assert result["result"] == "Thread contents here..."
        assert result["slug"] is not None
        assert len(bridge.articles) == 1

    def test_returns_error_on_mcp_failure(self):
        from saido_agent.mcp.ingest_bridge import MCPIngestBridge

        bridge = FakeBridge()
        mgr = FakeMCPManager()  # no results configured -> raises

        ib = MCPIngestBridge(mcp_manager=mgr, knowledge_bridge=bridge)
        result = ib.call_and_ingest("bad", "tool", {})

        assert result["status"] == "error"
        assert result["slug"] is None
        assert "MCP call failed" in result["error"]

    def test_returns_error_on_mcp_tool_error(self):
        from saido_agent.mcp.ingest_bridge import MCPIngestBridge

        bridge = FakeBridge()
        mgr = FakeMCPManager(results={
            "mcp__s__t": "[MCP tool error]\nSomething broke"
        })

        ib = MCPIngestBridge(mcp_manager=mgr, knowledge_bridge=bridge)
        result = ib.call_and_ingest("s", "t", {})

        assert result["status"] == "error"
        assert "MCP tool returned an error" in result["error"]

    def test_passes_metadata_to_ingest(self):
        from saido_agent.mcp.ingest_bridge import MCPIngestBridge

        bridge = FakeBridge()
        mgr = FakeMCPManager(results={"mcp__g__r": "file content"})

        ib = MCPIngestBridge(mcp_manager=mgr, knowledge_bridge=bridge)
        result = ib.call_and_ingest("g", "r", {}, metadata={"drive_id": "xyz"})

        assert result["status"] == "ok"
        slug = result["slug"]
        assert bridge.articles[slug]["frontmatter"]["drive_id"] == "xyz"


# ---------------------------------------------------------------------------
# Auto-ingest config saves and loads
# ---------------------------------------------------------------------------

class TestAutoIngestConfig:
    """Auto-ingest config saves and loads correctly."""

    def test_configure_and_query(self, tmp_auto_ingest):
        from saido_agent.mcp.ingest_bridge import MCPIngestBridge

        ib = MCPIngestBridge(mcp_manager=None, knowledge_bridge=None)

        # Initially not configured
        assert not ib.is_auto_ingest("gmail", "read_message")

        # Enable
        ib.configure_auto_ingest("gmail", "read_message", enabled=True)
        assert ib.is_auto_ingest("gmail", "read_message")

        # Persisted to disk
        assert tmp_auto_ingest.exists()
        data = json.loads(tmp_auto_ingest.read_text())
        assert data["gmail"]["read_message"] is True

        # Disable
        ib.configure_auto_ingest("gmail", "read_message", enabled=False)
        assert not ib.is_auto_ingest("gmail", "read_message")

    def test_get_full_config(self, tmp_auto_ingest):
        from saido_agent.mcp.ingest_bridge import MCPIngestBridge

        ib = MCPIngestBridge(mcp_manager=None, knowledge_bridge=None)
        ib.configure_auto_ingest("slack", "read_thread", True)
        ib.configure_auto_ingest("gmail", "read_message", True)

        config = ib.get_auto_ingest_config()
        assert config["slack"]["read_thread"] is True
        assert config["gmail"]["read_message"] is True

    def test_handles_missing_file_gracefully(self, tmp_auto_ingest):
        from saido_agent.mcp.ingest_bridge import MCPIngestBridge

        # File doesn't exist yet
        assert not tmp_auto_ingest.exists()

        ib = MCPIngestBridge(mcp_manager=None, knowledge_bridge=None)
        assert ib.get_auto_ingest_config() == {}
        assert not ib.is_auto_ingest("any", "tool")


# ---------------------------------------------------------------------------
# Recipe files are valid JSON with required fields
# ---------------------------------------------------------------------------

class TestRecipes:
    """Recipe files are valid JSON with required fields."""

    def test_all_recipes_are_valid_json(self):
        from saido_agent.mcp.ingest_bridge import _RECIPES_DIR, _RECIPE_REQUIRED_FIELDS

        recipe_files = list(_RECIPES_DIR.glob("*.json"))
        assert len(recipe_files) >= 4, f"Expected at least 4 recipes, found {len(recipe_files)}"

        for recipe_file in recipe_files:
            data = json.loads(recipe_file.read_text(encoding="utf-8"))
            assert isinstance(data, dict), f"{recipe_file.name} is not a JSON object"

            missing = _RECIPE_REQUIRED_FIELDS - set(data.keys())
            assert not missing, (
                f"{recipe_file.name} missing required fields: {missing}"
            )

    def test_load_recipe_by_name(self):
        from saido_agent.mcp.ingest_bridge import load_recipe

        recipe = load_recipe("gmail")
        assert recipe["name"] == "gmail"
        assert "tools" in recipe
        assert isinstance(recipe["tools"], list)

    def test_load_recipe_not_found(self):
        from saido_agent.mcp.ingest_bridge import load_recipe

        with pytest.raises(FileNotFoundError):
            load_recipe("nonexistent_recipe_xyz")

    def test_list_recipes(self):
        from saido_agent.mcp.ingest_bridge import list_recipes

        names = list_recipes()
        assert "gmail" in names
        assert "slack" in names
        assert "google_drive" in names
        assert "ast_grep" in names

    def test_ast_grep_recipe_has_command(self):
        from saido_agent.mcp.ingest_bridge import load_recipe

        recipe = load_recipe("ast_grep")
        assert "command" in recipe
        assert isinstance(recipe["command"], list)

    def test_gmail_recipe_has_ingest_tools(self):
        from saido_agent.mcp.ingest_bridge import load_recipe

        recipe = load_recipe("gmail")
        assert recipe["auto_ingest"] is True
        assert "ingest_tools" in recipe
        assert "read_message" in recipe["ingest_tools"]


# ---------------------------------------------------------------------------
# CLI commands registered
# ---------------------------------------------------------------------------

class TestCLICommands:
    """CLI commands are registered in the COMMANDS dict."""

    def test_ingest_mcp_command_registered(self):
        from saido_agent.cli.repl import COMMANDS
        assert "ingest-mcp" in COMMANDS

    def test_mcp_setup_command_registered(self):
        from saido_agent.cli.repl import COMMANDS
        assert "mcp-setup" in COMMANDS

    def test_mcp_auto_command_registered(self):
        from saido_agent.cli.repl import COMMANDS
        assert "mcp-auto" in COMMANDS


# ---------------------------------------------------------------------------
# MCP approval flow still enforced
# ---------------------------------------------------------------------------

class TestMCPApprovalEnforced:
    """MCP approval flow (HIGH-2) is still enforced through the ingest bridge."""

    def test_call_and_ingest_delegates_to_manager(self):
        """The bridge delegates to MCPManager.call_tool, which enforces approval."""
        from saido_agent.mcp.ingest_bridge import MCPIngestBridge

        bridge = FakeBridge()
        mgr = mock.MagicMock()
        mgr.call_tool.return_value = "result text"

        ib = MCPIngestBridge(mcp_manager=mgr, knowledge_bridge=bridge)
        ib.call_and_ingest("server", "tool", {"key": "val"})

        # Verify the call went through MCPManager (which enforces HIGH-2)
        mgr.call_tool.assert_called_once_with(
            "mcp__server__tool", {"key": "val"}
        )

    def test_security_error_propagates(self):
        """SecurityError from MCP approval check propagates as error result."""
        from saido_agent.mcp.ingest_bridge import MCPIngestBridge
        from saido_agent.mcp.client import SecurityError

        bridge = FakeBridge()
        mgr = mock.MagicMock()
        mgr.call_tool.side_effect = SecurityError("Command not approved")

        ib = MCPIngestBridge(mcp_manager=mgr, knowledge_bridge=bridge)
        result = ib.call_and_ingest("evil", "tool", {})

        assert result["status"] == "error"
        assert "Command not approved" in result["error"]


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """MCPIngestBridge is exported from the mcp package."""

    def test_import_from_package(self):
        from saido_agent.mcp import MCPIngestBridge
        assert MCPIngestBridge is not None

    def test_import_recipe_helpers(self):
        from saido_agent.mcp.ingest_bridge import load_recipe, list_recipes
        assert callable(load_recipe)
        assert callable(list_recipes)
