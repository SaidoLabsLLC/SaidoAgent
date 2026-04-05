"""Public configuration interface for Saido Agent SDK.

Internal module ``saido_agent.core.config`` handles secure key storage and
provider-level settings.  This module provides the user-facing configuration
API exposed via ``from saido_agent import SaidoConfig``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path.home() / ".saido_agent"
_DEFAULT_CONFIG_FILE = _DEFAULT_CONFIG_DIR / "config.json"


class SaidoConfig:
    """Public configuration interface for Saido Agent SDK.

    Loads settings from ``~/.saido_agent/config.json`` (or a custom path)
    and exposes them as properties.  Non-secret settings only; API keys are
    managed separately via ``saido_agent.core.config``.
    """

    def __init__(self, config_path: str | None = None) -> None:
        self._path = Path(config_path) if config_path else _DEFAULT_CONFIG_FILE
        self._data: dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # Loading / persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load configuration from disk.  Missing file is not an error."""
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load config from %s: %s", self._path, exc)
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        """Persist current configuration to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("Failed to save config to %s: %s", self._path, exc)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def routing(self) -> dict[str, Any]:
        """Routing configuration (task -> provider/model mappings)."""
        return self._data.get("routing", {})

    @property
    def knowledge_dir(self) -> str:
        """Path to the knowledge store directory."""
        return self._data.get("knowledge_dir", "./knowledge")

    @property
    def model(self) -> str:
        """Default model name."""
        return self._data.get("model", "")

    @property
    def provider(self) -> str:
        """Default LLM provider."""
        return self._data.get("provider", "")

    @property
    def embeddings_enabled(self) -> bool:
        """Whether semantic embeddings are enabled (default: False, FTS5-only)."""
        return bool(self._data.get("embeddings_enabled", False))

    @property
    def embeddings_model(self) -> str:
        """Sentence-transformer model for embeddings."""
        return self._data.get("embeddings_model", "all-MiniLM-L6-v2")

    @property
    def embeddings_quantize(self) -> bool:
        """Whether to quantize embedding vectors for reduced storage."""
        return bool(self._data.get("embeddings_quantize", False))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def update(self, **kwargs: Any) -> None:
        """Update configuration values and persist to disk.

        Example::

            config.update(knowledge_dir="/data/kb", model="qwen3:30b")
        """
        self._data.update(kwargs)
        self._save()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the full configuration dictionary."""
        return dict(self._data)

    def __repr__(self) -> str:
        return f"SaidoConfig(path={self._path!r}, keys={list(self._data.keys())})"
