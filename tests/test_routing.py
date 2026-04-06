"""
Tests for the Local LLM & Model Routing Engine.

Covers:
  - Auto-detection parses Ollama/LM Studio responses (mock HTTP)
  - Routing selects local when available
  - Falls back to cloud when local unavailable
  - /cloud forces cloud
  - Cost tracking accumulates correctly
  - Offline mode activates and forces local routing
  - Default routing.json created on first run
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from saido_agent.core.routing import (
    ModelRouter,
    LocalProviderInfo,
    _extract_model_size,
    _http_get_json,
    DEFAULT_ROUTING_CONFIG,
)
from saido_agent.core.cost_tracker import CostTracker, ModelUsage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OLLAMA_TAGS_RESPONSE = {
    "models": [
        {"name": "qwen3:8b", "size": 30_000_000_000},
        {"name": "llama3.3:8b", "size": 8_000_000_000},
        {"name": "phi4:3.8b", "size": 3_800_000_000},
    ]
}

LMSTUDIO_MODELS_RESPONSE = {
    "data": [
        {"id": "deepseek-r1-14b", "object": "model"},
        {"id": "mistral-7b", "object": "model"},
    ]
}


@pytest.fixture
def tmp_routing_config(tmp_path: Path) -> Path:
    """Return path to a temporary routing config file (does not exist yet)."""
    return tmp_path / "routing.json"


@pytest.fixture
def router_with_local(tmp_routing_config: Path) -> ModelRouter:
    """Create a ModelRouter with mocked local providers available."""
    with patch("saido_agent.core.routing._http_get_json") as mock_get:
        def side_effect(url, timeout=3):
            if "11434" in url and "tags" in url:
                return OLLAMA_TAGS_RESPONSE
            if "1234" in url and "models" in url:
                return LMSTUDIO_MODELS_RESPONSE
            return None

        mock_get.side_effect = side_effect
        router = ModelRouter(config_path=tmp_routing_config)
        router.probe_local_providers()
        return router


@pytest.fixture
def router_no_local(tmp_routing_config: Path) -> ModelRouter:
    """Create a ModelRouter with no local providers available."""
    with patch("saido_agent.core.routing._http_get_json", return_value=None):
        router = ModelRouter(config_path=tmp_routing_config)
        router.probe_local_providers()
        return router


# ---------------------------------------------------------------------------
# Auto-detection: Ollama / LM Studio
# ---------------------------------------------------------------------------

class TestAutoDetection:
    def test_probe_ollama_parses_models(self, router_with_local: ModelRouter):
        info = router_with_local._local_providers["ollama"]
        assert info.available is True
        assert "qwen3:8b" in info.models
        assert "llama3.3:8b" in info.models
        assert len(info.models) == 3

    def test_probe_lmstudio_parses_models(self, router_with_local: ModelRouter):
        info = router_with_local._local_providers["lmstudio"]
        assert info.available is True
        assert "deepseek-r1-14b" in info.models
        assert "mistral-7b" in info.models

    def test_probe_failure_marks_unavailable(self, router_no_local: ModelRouter):
        for info in router_no_local._local_providers.values():
            assert info.available is False
            assert info.models == []

    def test_refresh_reprobes(self, tmp_routing_config: Path):
        call_count = 0

        def counting_get(url, timeout=3):
            nonlocal call_count
            call_count += 1
            if "11434" in url and "tags" in url:
                return OLLAMA_TAGS_RESPONSE
            return None

        with patch("saido_agent.core.routing._http_get_json", side_effect=counting_get):
            router = ModelRouter(config_path=tmp_routing_config)
            router.probe_local_providers()
            first_count = call_count
            router.refresh()
            assert call_count > first_count


# ---------------------------------------------------------------------------
# Model size extraction
# ---------------------------------------------------------------------------

class TestModelSizeExtraction:
    def test_standard_format(self):
        assert _extract_model_size("qwen3:8b") == 8.0

    def test_decimal_format(self):
        assert _extract_model_size("phi4:3.8b") == 3.8

    def test_no_size(self):
        assert _extract_model_size("phi4") == 0.0

    def test_large_model(self):
        assert _extract_model_size("llama3.3:70b") == 70.0


# ---------------------------------------------------------------------------
# Routing: local selection
# ---------------------------------------------------------------------------

class TestRoutingLocalSelection:
    def test_local_preferred_selects_local(self, router_with_local: ModelRouter):
        provider, model = router_with_local.select_model("ingest")
        assert provider == "ollama"
        assert model == "qwen3:8b"

    def test_local_preferred_for_all_local_tasks(self, router_with_local: ModelRouter):
        for task in ("ingest", "compile", "index", "lint", "qa", "code_gen"):
            provider, model = router_with_local.select_model(task)
            assert provider in ("ollama", "lmstudio"), f"Task {task} not routed locally"

    def test_cloud_preferred_selects_cloud(self, router_with_local: ModelRouter):
        provider, model = router_with_local.select_model("review")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-6"

    def test_architect_selects_cloud(self, router_with_local: ModelRouter):
        provider, model = router_with_local.select_model("architect")
        assert provider == "anthropic"
        assert model == "claude-opus-4-6"

    def test_auto_select_best_by_size(self, router_with_local: ModelRouter):
        best = router_with_local.auto_select_best_local()
        assert best is not None
        provider, model = best
        # deepseek-r1-14b (14B) is the largest model from our test data
        assert model == "deepseek-r1-14b"


# ---------------------------------------------------------------------------
# Routing: fallback to cloud
# ---------------------------------------------------------------------------

class TestRoutingFallback:
    def test_no_local_falls_back_to_cloud(self, router_no_local: ModelRouter):
        provider, model = router_no_local.select_model("ingest")
        # Should fall back to cloud mid_tier
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-6"

    def test_escalation_returns_cloud(self, router_with_local: ModelRouter):
        result = router_with_local.escalate("qa")
        assert result is not None
        provider, model = result
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-6"

    def test_no_escalation_for_non_configured_task(self, router_with_local: ModelRouter):
        result = router_with_local.escalate("ingest")
        assert result is None


# ---------------------------------------------------------------------------
# /cloud force override
# ---------------------------------------------------------------------------

class TestForceCloud:
    def test_force_cloud_overrides_local(self, router_with_local: ModelRouter):
        router_with_local.set_force_cloud(True)
        provider, model = router_with_local.select_model("ingest")
        assert provider == "anthropic"
        assert "claude" in model

    def test_force_cloud_resets_after_one_call(self, router_with_local: ModelRouter):
        router_with_local.set_force_cloud(True)
        router_with_local.select_model("ingest")  # consumes the flag
        provider, model = router_with_local.select_model("ingest")
        assert provider == "ollama"


# ---------------------------------------------------------------------------
# Offline mode
# ---------------------------------------------------------------------------

class TestOfflineMode:
    def test_offline_detected_when_no_internet(self, router_with_local: ModelRouter):
        with patch("saido_agent.core.routing._http_get_json", return_value=None):
            with patch("urllib.request.urlopen", side_effect=OSError("no network")):
                online = router_with_local.check_internet()
                assert online is False
                assert router_with_local.offline_mode is True

    def test_offline_routes_everything_local(self, router_with_local: ModelRouter):
        router_with_local._offline_mode = True
        # Even cloud-preferred tasks should go local
        provider, model = router_with_local.select_model("review")
        assert provider in ("ollama", "lmstudio")

    def test_offline_blocks_escalation(self, router_with_local: ModelRouter):
        router_with_local._offline_mode = True
        result = router_with_local.escalate("qa")
        assert result is None

    def test_online_detected(self, router_with_local: ModelRouter):
        with patch("saido_agent.core.routing._http_get_json", return_value={"type": "error"}):
            online = router_with_local.check_internet()
            assert online is True
            assert router_with_local.offline_mode is False


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

class TestCostTracker:
    def test_local_tokens_are_free(self):
        tracker = CostTracker()
        tracker.record("ollama", "qwen3:8b", 500_000, 500_000)
        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 1_000_000

    def test_cloud_tokens_have_cost(self):
        tracker = CostTracker()
        tracker.record("anthropic", "claude-sonnet-4-6", 10_000, 2_000)
        cost = tracker.total_cost
        assert cost > 0.0

    def test_accumulates_across_calls(self):
        tracker = CostTracker()
        tracker.record("ollama", "qwen3:8b", 100_000, 100_000)
        tracker.record("ollama", "qwen3:8b", 200_000, 200_000)
        assert tracker.total_tokens == 600_000

    def test_mixed_usage_savings(self):
        tracker = CostTracker()
        tracker.record("ollama", "qwen3:8b", 1_000_000, 247_000)
        tracker.record("anthropic", "claude-sonnet-4-6", 10_000, 2_400)
        savings = tracker.estimated_savings
        assert savings > 0.0

    def test_format_report_contains_key_info(self):
        tracker = CostTracker()
        tracker.record("ollama", "qwen3:8b", 1_247_000, 0)
        tracker.record("anthropic", "claude-sonnet-4-6", 12_400, 0)
        report = tracker.format_report()
        assert "qwen3:8b" in report
        assert "claude-sonnet-4-6" in report
        assert "$0.00" in report
        assert "savings" in report.lower()

    def test_reset_clears_all(self):
        tracker = CostTracker()
        tracker.record("ollama", "qwen3:8b", 100_000, 100_000)
        tracker.reset()
        assert tracker.total_tokens == 0
        assert tracker.total_cost == 0.0


# ---------------------------------------------------------------------------
# Default routing.json creation
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    def test_creates_default_on_first_run(self, tmp_routing_config: Path):
        assert not tmp_routing_config.exists()
        with patch("saido_agent.core.routing._http_get_json", return_value=None):
            router = ModelRouter(config_path=tmp_routing_config)
        assert tmp_routing_config.exists()
        data = json.loads(tmp_routing_config.read_text())
        assert "routing" in data
        assert data["routing"]["ingest"]["model"] == "qwen3:8b"

    def test_loads_existing_config(self, tmp_routing_config: Path):
        custom = {
            "routing": {"ingest": {"prefer": "cloud", "model": "gpt-4o"}},
            "escalation": {"mid_tier": "gpt-4o", "frontier": "gpt-4o"},
            "local_providers": {},
            "cost_tracking": {"show_savings": True, "log_cloud_usage": True},
        }
        tmp_routing_config.write_text(json.dumps(custom))
        router = ModelRouter(config_path=tmp_routing_config)
        assert router.routing_config["routing"]["ingest"]["model"] == "gpt-4o"

    def test_status_summary_not_empty(self, router_with_local: ModelRouter):
        summary = router_with_local.status_summary()
        assert "ollama" in summary.lower()
        assert "qwen3:8b" in summary


# ---------------------------------------------------------------------------
# LocalProviderInfo
# ---------------------------------------------------------------------------

class TestLocalProviderInfo:
    def test_to_dict(self):
        info = LocalProviderInfo("ollama", "http://localhost:11434/v1")
        info.available = True
        info.models = ["qwen3:8b"]
        d = info.to_dict()
        assert d["name"] == "ollama"
        assert d["available"] is True
        assert "qwen3:8b" in d["models"]
