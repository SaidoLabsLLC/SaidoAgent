"""Tests for the SmartRAG semantic embedding integration layer.

Covers:
- SaidoConfig embeddings properties (toggle, model, quantize)
- BridgeConfig -> SmartRAGConfig embeddings passthrough
- KnowledgeBridge embedding lifecycle (enable, disable, status)
- CLI /embeddings commands (enable, disable, status)
- CLI /reindex --embeddings flag
- Graceful degradation when smartrag[embeddings] extra is not installed
- Empty store safety (no crash on enable/reindex)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from saido_agent.config import SaidoConfig
from saido_agent.knowledge.bridge import (
    SMARTRAG_AVAILABLE,
    BridgeConfig,
    HAS_EMBEDDINGS,
    KnowledgeBridge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Path:
    """Return path to a temporary config.json."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text("{}", encoding="utf-8")
    return cfg_path


@pytest.fixture()
def saido_config(tmp_config: Path) -> SaidoConfig:
    """Return a SaidoConfig backed by a temp file."""
    return SaidoConfig(config_path=str(tmp_config))


@pytest.fixture()
def tmp_knowledge(tmp_path: Path) -> Path:
    """Return a temporary directory to use as knowledge root."""
    return tmp_path / "knowledge"


@pytest.fixture()
def bridge(tmp_knowledge: Path) -> KnowledgeBridge:
    """Return a KnowledgeBridge backed by a temp directory."""
    cfg = BridgeConfig(knowledge_root=str(tmp_knowledge))
    return KnowledgeBridge(config=cfg)


@pytest.fixture()
def bridge_with_embeddings(tmp_knowledge: Path) -> KnowledgeBridge:
    """Return a KnowledgeBridge with embeddings enabled in config."""
    cfg = BridgeConfig(knowledge_root=str(tmp_knowledge), embeddings=True)
    return KnowledgeBridge(config=cfg)


# ---------------------------------------------------------------------------
# SaidoConfig embeddings properties
# ---------------------------------------------------------------------------


class TestSaidoConfigEmbeddings:
    """Verify SaidoConfig exposes embeddings settings with correct defaults."""

    def test_defaults(self, saido_config: SaidoConfig) -> None:
        assert saido_config.embeddings_enabled is False
        assert saido_config.embeddings_model == "all-MiniLM-L6-v2"
        assert saido_config.embeddings_quantize is False

    def test_toggle_on(self, saido_config: SaidoConfig) -> None:
        saido_config.update(embeddings_enabled=True)
        assert saido_config.embeddings_enabled is True

    def test_toggle_off(self, saido_config: SaidoConfig) -> None:
        saido_config.update(embeddings_enabled=True)
        saido_config.update(embeddings_enabled=False)
        assert saido_config.embeddings_enabled is False

    def test_custom_model(self, saido_config: SaidoConfig) -> None:
        saido_config.update(embeddings_model="all-mpnet-base-v2")
        assert saido_config.embeddings_model == "all-mpnet-base-v2"

    def test_quantize(self, saido_config: SaidoConfig) -> None:
        saido_config.update(embeddings_quantize=True)
        assert saido_config.embeddings_quantize is True

    def test_persists_to_disk(self, tmp_config: Path) -> None:
        cfg = SaidoConfig(config_path=str(tmp_config))
        cfg.update(embeddings_enabled=True, embeddings_model="test-model")
        # Reload from disk
        cfg2 = SaidoConfig(config_path=str(tmp_config))
        assert cfg2.embeddings_enabled is True
        assert cfg2.embeddings_model == "test-model"


# ---------------------------------------------------------------------------
# BridgeConfig -> SmartRAGConfig passthrough
# ---------------------------------------------------------------------------


class TestBridgeConfigPassthrough:
    """BridgeConfig.to_smartrag_config() passes embeddings flag correctly."""

    @pytest.mark.skipif(not SMARTRAG_AVAILABLE, reason="SmartRAG not installed")
    def test_embeddings_false_by_default(self) -> None:
        cfg = BridgeConfig()
        sr_cfg = cfg.to_smartrag_config()
        assert sr_cfg.embeddings is False

    @pytest.mark.skipif(not SMARTRAG_AVAILABLE, reason="SmartRAG not installed")
    def test_embeddings_true_when_set(self) -> None:
        cfg = BridgeConfig(embeddings=True)
        sr_cfg = cfg.to_smartrag_config()
        assert sr_cfg.embeddings is True


# ---------------------------------------------------------------------------
# KnowledgeBridge embedding methods
# ---------------------------------------------------------------------------


class TestBridgeEmbeddings:
    """KnowledgeBridge embedding lifecycle methods."""

    def test_embeddings_enabled_property_default(self, bridge: KnowledgeBridge) -> None:
        assert bridge.embeddings_enabled is False

    def test_embeddings_enabled_property_true(
        self, bridge_with_embeddings: KnowledgeBridge
    ) -> None:
        assert bridge_with_embeddings.embeddings_enabled is True

    def test_embeddings_status_default(self, bridge: KnowledgeBridge) -> None:
        status = bridge.embeddings_status()
        assert status["enabled"] is False
        assert isinstance(status["available"], bool)
        assert status["total_articles"] >= 0
        assert status["articles_with_embeddings"] >= 0

    def test_embeddings_status_keys(self, bridge: KnowledgeBridge) -> None:
        status = bridge.embeddings_status()
        expected_keys = {"enabled", "available", "total_articles", "articles_with_embeddings"}
        assert set(status.keys()) == expected_keys

    @pytest.mark.skipif(not HAS_EMBEDDINGS, reason="Embeddings extra not installed")
    def test_enable_embeddings(self, bridge: KnowledgeBridge) -> None:
        assert bridge.embeddings_enabled is False
        bridge.enable_embeddings()
        assert bridge.embeddings_enabled is True

    @pytest.mark.skipif(not HAS_EMBEDDINGS, reason="Embeddings extra not installed")
    def test_disable_embeddings(self, bridge_with_embeddings: KnowledgeBridge) -> None:
        assert bridge_with_embeddings.embeddings_enabled is True
        bridge_with_embeddings.disable_embeddings()
        assert bridge_with_embeddings.embeddings_enabled is False

    def test_enable_embeddings_without_extra(self, bridge: KnowledgeBridge) -> None:
        """Enabling embeddings without the extra installed raises RuntimeError."""
        with patch("saido_agent.knowledge.bridge.HAS_EMBEDDINGS", False):
            with pytest.raises(RuntimeError, match="Embeddings extra not installed"):
                bridge.enable_embeddings()

    def test_empty_store_status(self, bridge: KnowledgeBridge) -> None:
        """Empty store should not crash on status check."""
        status = bridge.embeddings_status()
        assert status["total_articles"] == 0
        assert status["articles_with_embeddings"] == 0

    @pytest.mark.skipif(not HAS_EMBEDDINGS, reason="Embeddings extra not installed")
    def test_empty_store_enable_no_crash(self, bridge: KnowledgeBridge) -> None:
        """Enabling embeddings on an empty store should not crash."""
        bridge.enable_embeddings()
        count = bridge.reindex(incremental=False)
        assert count == 0


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Verify correct behavior when smartrag[embeddings] is not installed."""

    def test_has_embeddings_is_bool(self) -> None:
        assert isinstance(HAS_EMBEDDINGS, bool)

    def test_bridge_status_reports_unavailable(self, bridge: KnowledgeBridge) -> None:
        with patch("saido_agent.knowledge.bridge.HAS_EMBEDDINGS", False):
            status = bridge.embeddings_status()
            assert status["available"] is False

    def test_degraded_bridge_embeddings_status(self, tmp_knowledge: Path) -> None:
        """A bridge with no SmartRAG still returns a safe status dict."""
        kb = KnowledgeBridge.__new__(KnowledgeBridge)
        kb._config = BridgeConfig(knowledge_root=str(tmp_knowledge))
        kb._root = Path(kb._config.knowledge_root).resolve()
        kb._rag = None
        status = kb.embeddings_status()
        assert status["enabled"] is False
        assert status["total_articles"] == 0
        assert status["articles_with_embeddings"] == 0


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


class TestCLIEmbeddings:
    """Test /embeddings CLI command dispatch."""

    def _make_config_with_bridge(self, bridge: KnowledgeBridge) -> dict:
        return {"_knowledge_context": {"bridge": bridge}}

    def test_cmd_embeddings_status(self, bridge: KnowledgeBridge, capsys) -> None:
        from saido_agent.cli.repl import cmd_embeddings

        config = self._make_config_with_bridge(bridge)
        result = cmd_embeddings("status", {}, config)
        assert result is True
        captured = capsys.readouterr()
        assert "Embeddings enabled:" in captured.out

    def test_cmd_embeddings_usage(self, bridge: KnowledgeBridge, capsys) -> None:
        from saido_agent.cli.repl import cmd_embeddings

        config = self._make_config_with_bridge(bridge)
        result = cmd_embeddings("", {}, config)
        assert result is True
        captured = capsys.readouterr()
        assert "/embeddings enable" in captured.out

    def test_cmd_embeddings_enable_no_extra(
        self, bridge: KnowledgeBridge, capsys
    ) -> None:
        from saido_agent.cli.repl import cmd_embeddings

        config = self._make_config_with_bridge(bridge)
        # HAS_EMBEDDINGS is imported locally from bridge module
        with patch("saido_agent.knowledge.bridge.HAS_EMBEDDINGS", False):
            result = cmd_embeddings("enable", {}, config)
        assert result is True
        captured = capsys.readouterr()
        assert "not installed" in captured.err

    def test_cmd_embeddings_disable(
        self, bridge: KnowledgeBridge, tmp_config: Path, capsys
    ) -> None:
        from saido_agent.cli.repl import cmd_embeddings

        config = self._make_config_with_bridge(bridge)
        # SaidoConfig is imported locally inside cmd_embeddings, patch at source
        with patch("saido_agent.config.SaidoConfig") as MockCfg:
            mock_instance = MagicMock()
            MockCfg.return_value = mock_instance
            result = cmd_embeddings("disable", {}, config)
        assert result is True
        mock_instance.update.assert_called_once_with(embeddings_enabled=False)

    def test_cmd_embeddings_status_no_bridge(self, capsys) -> None:
        from saido_agent.cli.repl import cmd_embeddings

        config = {"_knowledge_context": {"bridge": None}}
        result = cmd_embeddings("status", {}, config)
        assert result is True
        captured = capsys.readouterr()
        assert "not available" in captured.err


class TestCLIReindexEmbeddings:
    """Test /reindex --embeddings flag."""

    def test_reindex_embeddings_no_extra(self, bridge: KnowledgeBridge, capsys) -> None:
        from saido_agent.cli.repl import cmd_reindex

        config = {"_knowledge_context": {"bridge": bridge}}
        with patch("saido_agent.knowledge.bridge.HAS_EMBEDDINGS", False):
            result = cmd_reindex("--embeddings", {}, config)
        assert result is True
        captured = capsys.readouterr()
        assert "not installed" in captured.err

    @pytest.mark.skipif(not HAS_EMBEDDINGS, reason="Embeddings extra not installed")
    def test_reindex_embeddings_with_extra(
        self, bridge: KnowledgeBridge, capsys
    ) -> None:
        from saido_agent.cli.repl import cmd_reindex

        config = {"_knowledge_context": {"bridge": bridge}}
        result = cmd_reindex("--embeddings", {}, config)
        assert result is True
        captured = capsys.readouterr()
        assert "complete" in captured.out.lower() or "Embedding reindex" in captured.out

    def test_reindex_embeddings_no_bridge(self, capsys) -> None:
        from saido_agent.cli.repl import cmd_reindex

        config = {"_knowledge_context": {"bridge": None}}
        result = cmd_reindex("--embeddings", {}, config)
        assert result is True
        captured = capsys.readouterr()
        assert "not available" in captured.err


# ---------------------------------------------------------------------------
# COMMANDS dict registration
# ---------------------------------------------------------------------------


class TestCommandRegistration:
    """Verify /embeddings is registered in the COMMANDS dict."""

    def test_embeddings_in_commands(self) -> None:
        from saido_agent.cli.repl import COMMANDS
        assert "embeddings" in COMMANDS
