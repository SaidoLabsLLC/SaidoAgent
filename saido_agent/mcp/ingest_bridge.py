"""MCP-to-knowledge ingest bridge.

Pipes MCP tool results into the knowledge ingest pipeline, enabling
automatic knowledge capture from external MCP servers (Gmail, Slack,
Google Drive, etc.).

All MCP calls go through the existing approval flow (HIGH-2).
Auto-ingest only fires after the user has approved the MCP server.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Auto-ingest config persistence ──────────────────────────────────────────

_AUTO_INGEST_FILE = Path.home() / ".saido_agent" / "mcp_auto_ingest.json"


def _load_auto_ingest_config() -> Dict[str, Dict[str, bool]]:
    """Load auto-ingest configuration.

    Returns a dict of ``{server_name: {tool_name: enabled}}``.
    """
    if _AUTO_INGEST_FILE.exists():
        try:
            return json.loads(_AUTO_INGEST_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_auto_ingest_config(config: Dict[str, Dict[str, bool]]) -> None:
    """Persist auto-ingest configuration to disk."""
    _AUTO_INGEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    _AUTO_INGEST_FILE.write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


# ── Recipe loader ───────────────────────────────────────────────────────────

_RECIPES_DIR = Path(__file__).parent / "recipes"

# Required top-level fields in a recipe JSON file
_RECIPE_REQUIRED_FIELDS = frozenset({"name", "description", "tools"})


def load_recipe(name: str) -> Dict[str, Any]:
    """Load a recipe JSON by name (without .json extension).

    Raises ``FileNotFoundError`` if the recipe does not exist and
    ``ValueError`` if required fields are missing.
    """
    path = _RECIPES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Recipe not found: {name}")

    data = json.loads(path.read_text(encoding="utf-8"))

    missing = _RECIPE_REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(
            f"Recipe '{name}' is missing required fields: {', '.join(sorted(missing))}"
        )
    return data


def list_recipes() -> list[str]:
    """Return the names of all available recipes (without .json)."""
    if not _RECIPES_DIR.is_dir():
        return []
    return sorted(p.stem for p in _RECIPES_DIR.glob("*.json"))


# ── MCPIngestBridge ─────────────────────────────────────────────────────────


class MCPIngestBridge:
    """Pipes MCP tool results into the knowledge ingest pipeline.

    Responsibilities:
    - ``ingest_tool_result``: wrap raw MCP output as a knowledge article
    - ``call_and_ingest``: call an MCP tool then ingest the result
    - ``configure_auto_ingest``: toggle per-tool auto-ingest behaviour

    Security: all MCP calls are delegated to the existing ``MCPManager``
    which enforces the HIGH-2 approval flow.
    """

    def __init__(
        self,
        mcp_manager,
        knowledge_bridge,
        ingest_pipeline=None,
    ) -> None:
        self._mcp = mcp_manager
        self._bridge = knowledge_bridge
        self._pipeline = ingest_pipeline

    # ── Core API ─────────────────────────────────────────────────────────

    def ingest_tool_result(
        self,
        server_name: str,
        tool_name: str,
        result: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest a single MCP tool result as a knowledge article.

        The result text is wrapped in a markdown document with metadata
        frontmatter (source_server, source_tool, timestamp).  The article
        is created via ``KnowledgeBridge.create_article``.

        Returns a dict with ``slug``, ``status``, and optional ``error``.
        """
        ts = datetime.now(timezone.utc).isoformat()
        slug = f"mcp-{server_name}-{tool_name}-{ts[:19].replace(':', '-')}"

        frontmatter: Dict[str, Any] = {
            "source": "mcp",
            "source_server": server_name,
            "source_tool": tool_name,
            "ingested_at": ts,
        }
        if metadata:
            frontmatter.update(metadata)

        body = (
            f"# MCP Result: {server_name}/{tool_name}\n\n"
            f"**Server:** {server_name}  \n"
            f"**Tool:** {tool_name}  \n"
            f"**Timestamp:** {ts}\n\n"
            f"---\n\n"
            f"{result}"
        )

        try:
            doc = self._bridge.create_article(slug, body, frontmatter)
            if doc is None:
                return {"slug": slug, "status": "error", "error": "bridge returned None"}
            logger.info("Ingested MCP result as article: %s", slug)
            return {"slug": slug, "status": "ok", "error": None}
        except Exception as exc:
            logger.error("Failed to ingest MCP result: %s", exc)
            return {"slug": slug, "status": "error", "error": str(exc)}

    def call_and_ingest(
        self,
        server_name: str,
        tool_name: str,
        params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call an MCP tool and ingest the result.

        Delegates to ``MCPManager.call_tool`` for the actual MCP call
        (which enforces HIGH-2 approval), then pipes the text result
        into :meth:`ingest_tool_result`.

        Returns a dict with ``slug``, ``status``, ``result`` (raw text),
        and optional ``error``.
        """
        qualified_name = f"mcp__{server_name}__{tool_name}"

        try:
            result_text = self._mcp.call_tool(qualified_name, params)
        except Exception as exc:
            logger.error("MCP call failed: %s/%s: %s", server_name, tool_name, exc)
            return {
                "slug": None,
                "status": "error",
                "result": None,
                "error": f"MCP call failed: {exc}",
            }

        # Check for MCP tool error responses
        if result_text.startswith("[MCP tool error]"):
            return {
                "slug": None,
                "status": "error",
                "result": result_text,
                "error": "MCP tool returned an error",
            }

        ingest_result = self.ingest_tool_result(
            server_name, tool_name, result_text, metadata
        )
        ingest_result["result"] = result_text
        return ingest_result

    def configure_auto_ingest(
        self,
        server_name: str,
        tool_name: str,
        enabled: bool = True,
    ) -> None:
        """Configure a tool to auto-ingest its results.

        Persists configuration to ``~/.saido_agent/mcp_auto_ingest.json``.
        """
        config = _load_auto_ingest_config()
        if server_name not in config:
            config[server_name] = {}
        config[server_name][tool_name] = enabled
        _save_auto_ingest_config(config)
        logger.info(
            "Auto-ingest %s for %s/%s",
            "enabled" if enabled else "disabled",
            server_name,
            tool_name,
        )

    def is_auto_ingest(self, server_name: str, tool_name: str) -> bool:
        """Check whether a tool has auto-ingest enabled."""
        config = _load_auto_ingest_config()
        return config.get(server_name, {}).get(tool_name, False)

    def get_auto_ingest_config(self) -> Dict[str, Dict[str, bool]]:
        """Return the full auto-ingest configuration."""
        return _load_auto_ingest_config()
