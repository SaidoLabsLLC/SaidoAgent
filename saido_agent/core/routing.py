"""
Local LLM & Model Routing Engine for Saido Agent.

Probes Ollama and LM Studio on startup, caches available models,
and routes LLM calls to local or cloud providers based on task type,
availability, and user configuration.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

ROUTING_CONFIG_DIR = Path.home() / ".saido_agent"
ROUTING_CONFIG_FILE = ROUTING_CONFIG_DIR / "routing.json"

DEFAULT_ROUTING_CONFIG: dict[str, Any] = {
    "routing": {
        "ingest":    {"prefer": "local", "model": "qwen3:30b"},
        "compile":   {"prefer": "local", "model": "qwen3:30b"},
        "index":     {"prefer": "local", "model": "qwen3:30b"},
        "lint":      {"prefer": "local", "model": "qwen3:30b"},
        "qa":        {"prefer": "local", "model": "qwen3:30b", "escalate_on_failure": True},
        "code_gen":  {"prefer": "local", "model": "qwen3:30b", "escalate_on_failure": True},
        "review":    {"prefer": "cloud", "model": "claude-sonnet-4-6"},
        "architect": {"prefer": "cloud", "model": "claude-opus-4-6"},
    },
    "escalation": {
        "mid_tier": "claude-sonnet-4-6",
        "frontier": "claude-opus-4-6",
    },
    "local_providers": {
        "ollama":   {"base_url": "http://localhost:11434/v1", "enabled": True},
        "lmstudio": {"base_url": "http://localhost:1234/v1", "enabled": True},
    },
    "cost_tracking": {
        "show_savings": True,
        "log_cloud_usage": True,
    },
}

# Timeout in seconds for local provider probes
_PROBE_TIMEOUT = 3

# Timeout in seconds for internet connectivity check
_INTERNET_TIMEOUT = 2


def _http_get_json(url: str, timeout: float = _PROBE_TIMEOUT) -> Any:
    """Fetch JSON from a URL using only stdlib. Returns parsed JSON or None on failure."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None


def _extract_model_size(name: str) -> float:
    """Extract approximate parameter size in billions from model name string.

    Examples: 'qwen3:30b' -> 30.0, 'llama3.3:70b' -> 70.0, 'phi4' -> 0.0
    """
    import re
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", name)
    if match:
        return float(match.group(1))
    return 0.0


class LocalProviderInfo:
    """Cached information about a local LLM provider."""

    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
        self.available = False
        self.models: list[str] = []
        self.last_probed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "available": self.available,
            "models": self.models,
        }


class ModelRouter:
    """Routes LLM calls to local or cloud models based on task type and availability."""

    def __init__(self, config_path: Path | None = None):
        self._config_path = config_path or ROUTING_CONFIG_FILE
        self._routing_config: dict[str, Any] = {}
        self._local_providers: dict[str, LocalProviderInfo] = {}
        self._offline_mode = False
        self._force_cloud = False

        self._load_routing_config()
        self._init_local_providers()

    # -- Configuration -----------------------------------------------------------

    def _load_routing_config(self) -> None:
        """Load routing config from disk. Create default if missing."""
        if self._config_path.exists():
            try:
                self._routing_config = json.loads(self._config_path.read_text("utf-8"))
                return
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read routing config: %s — using defaults", exc)

        # Create default config
        self._routing_config = json.loads(json.dumps(DEFAULT_ROUTING_CONFIG))
        self._save_routing_config()

    def _save_routing_config(self) -> None:
        """Persist routing config to disk."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(
                json.dumps(self._routing_config, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("Failed to write routing config: %s", exc)

    @property
    def routing_config(self) -> dict[str, Any]:
        return self._routing_config

    @property
    def offline_mode(self) -> bool:
        return self._offline_mode

    # -- Local Provider Detection ------------------------------------------------

    def _init_local_providers(self) -> None:
        """Initialize LocalProviderInfo objects from config."""
        lp_cfg = self._routing_config.get("local_providers", {})
        for name, settings in lp_cfg.items():
            if not settings.get("enabled", True):
                continue
            info = LocalProviderInfo(name, settings.get("base_url", ""))
            self._local_providers[name] = info

    def probe_local_providers(self) -> dict[str, LocalProviderInfo]:
        """Probe all enabled local providers and cache results."""
        for name, info in self._local_providers.items():
            if name == "ollama":
                self._probe_ollama(info)
            elif name == "lmstudio":
                self._probe_lmstudio(info)
            info.last_probed = time.time()

        available = {n: p for n, p in self._local_providers.items() if p.available}
        if not available:
            logger.warning("No local LLM detected. All tasks will route to cloud providers.")
        return self._local_providers

    def _probe_ollama(self, info: LocalProviderInfo) -> None:
        """Probe Ollama at /api/tags endpoint."""
        # Ollama's native endpoint is on the base host, not /v1
        base = info.base_url.replace("/v1", "")
        data = _http_get_json(f"{base}/api/tags")
        if data and "models" in data:
            info.available = True
            info.models = [m.get("name", "") for m in data["models"] if m.get("name")]
        else:
            info.available = False
            info.models = []

    def _probe_lmstudio(self, info: LocalProviderInfo) -> None:
        """Probe LM Studio at /v1/models endpoint."""
        data = _http_get_json(f"{info.base_url}/models")
        if data and "data" in data:
            info.available = True
            info.models = [m.get("id", "") for m in data["data"] if m.get("id")]
        else:
            info.available = False
            info.models = []

    def refresh(self) -> dict[str, LocalProviderInfo]:
        """Re-probe local providers (called by /refresh command)."""
        return self.probe_local_providers()

    def get_available_local_models(self) -> list[tuple[str, str]]:
        """Return list of (provider_name, model_name) for all available local models."""
        results: list[tuple[str, str]] = []
        for name, info in self._local_providers.items():
            if info.available:
                for model in info.models:
                    results.append((name, model))
        return results

    def auto_select_best_local(self) -> tuple[str, str] | None:
        """Select the best available local model by parameter size (largest wins)."""
        candidates = self.get_available_local_models()
        if not candidates:
            return None
        best = max(candidates, key=lambda x: _extract_model_size(x[1]))
        return best

    # -- Internet / Offline Detection --------------------------------------------

    def check_internet(self) -> bool:
        """Check if Anthropic API is reachable. Returns True if online."""
        data = _http_get_json("https://api.anthropic.com/v1", timeout=_INTERNET_TIMEOUT)
        # Even an auth error means the server is reachable
        online = data is not None
        if not online:
            # Try a simple connection check — the API may return non-JSON on GET
            try:
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1",
                    headers={"Accept": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=_INTERNET_TIMEOUT) as resp:
                    online = resp.status < 500
            except urllib.error.HTTPError:
                # HTTP errors (401, 403, 404) mean the server IS reachable
                online = True
            except (urllib.error.URLError, OSError, TimeoutError):
                online = False

        self._offline_mode = not online
        if self._offline_mode:
            logger.warning(
                "No internet connection detected. Running in OFFLINE mode — "
                "all tasks will route to local models."
            )
        return online

    # -- Model Selection ---------------------------------------------------------

    def set_force_cloud(self, enabled: bool) -> None:
        """Toggle /cloud prefix behavior for a single query."""
        self._force_cloud = enabled

    def select_model(self, task_type: str) -> tuple[str, str]:
        """Select provider and model for a given task type.

        Returns:
            (provider_name, model_name) — e.g. ("ollama", "qwen3:30b")
            or ("anthropic", "claude-sonnet-4-6")
        """
        routing = self._routing_config.get("routing", {})
        escalation = self._routing_config.get("escalation", {})

        task_cfg = routing.get(task_type, {})
        prefer = task_cfg.get("prefer", "local")
        model = task_cfg.get("model", "qwen3:30b")

        # /cloud prefix forces cloud for this call
        if self._force_cloud:
            self._force_cloud = False
            cloud_model = escalation.get("mid_tier", "claude-sonnet-4-6")
            from saido_agent.core.providers import detect_provider
            return (detect_provider(cloud_model), cloud_model)

        # Offline mode forces everything local
        if self._offline_mode:
            return self._resolve_local(model)

        # Normal routing
        if prefer == "local":
            return self._resolve_local(model)
        else:
            from saido_agent.core.providers import detect_provider
            return (detect_provider(model), model)

    def select_model_with_escalation(
        self, task_type: str
    ) -> tuple[str, str, bool]:
        """Select model, returning escalation info.

        Returns:
            (provider_name, model_name, is_escalated)
        """
        provider, model = self.select_model(task_type)
        return (provider, model, False)

    def escalate(self, task_type: str) -> tuple[str, str] | None:
        """Escalate a failed local task to cloud if escalate_on_failure is set.

        Returns (provider, model) for escalated call, or None if escalation
        is not configured for this task type.
        """
        routing = self._routing_config.get("routing", {})
        escalation = self._routing_config.get("escalation", {})
        task_cfg = routing.get(task_type, {})

        if not task_cfg.get("escalate_on_failure", False):
            return None

        if self._offline_mode:
            logger.warning("Cannot escalate to cloud in offline mode.")
            return None

        mid_tier = escalation.get("mid_tier", "claude-sonnet-4-6")
        from saido_agent.core.providers import detect_provider
        return (detect_provider(mid_tier), mid_tier)

    def _resolve_local(self, requested_model: str) -> tuple[str, str]:
        """Resolve a local model request to an available provider.

        Checks if the requested model is available on any local provider.
        Falls back to the best available local model, then to the first
        available cloud escalation target.
        """
        # Check if requested model is available locally
        for name, info in self._local_providers.items():
            if info.available and requested_model in info.models:
                return (name, requested_model)

        # Try best available local model
        best = self.auto_select_best_local()
        if best:
            logger.info(
                "Requested model %r not found locally. Using %s/%s instead.",
                requested_model, best[0], best[1],
            )
            return best

        # No local models at all — fall back to cloud if not offline
        if not self._offline_mode:
            escalation = self._routing_config.get("escalation", {})
            fallback = escalation.get("mid_tier", "claude-sonnet-4-6")
            from saido_agent.core.providers import detect_provider
            logger.warning(
                "No local models available. Falling back to cloud: %s", fallback
            )
            return (detect_provider(fallback), fallback)

        # Offline with no local models — return the requested model anyway
        # and let the downstream call fail with a clear error
        return ("ollama", requested_model)

    # -- Status / Display --------------------------------------------------------

    def status_summary(self) -> str:
        """Return a human-readable status summary."""
        lines: list[str] = []

        if self._offline_mode:
            lines.append("** OFFLINE MODE — all tasks routed to local models **")
            lines.append("")

        lines.append("Local Providers:")
        for name, info in self._local_providers.items():
            status = "available" if info.available else "not detected"
            lines.append(f"  {name}: {status}")
            if info.available:
                for m in info.models:
                    lines.append(f"    - {m}")

        lines.append("")
        lines.append("Routing:")
        routing = self._routing_config.get("routing", {})
        for task, cfg in routing.items():
            pref = cfg.get("prefer", "local")
            model = cfg.get("model", "?")
            esc = " [escalate]" if cfg.get("escalate_on_failure") else ""
            lines.append(f"  {task}: {pref} -> {model}{esc}")

        return "\n".join(lines)
