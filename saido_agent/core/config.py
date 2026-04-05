"""Configuration management for Saido Agent (multi-provider).

HIGH-1 Security Fix: API keys are stored in OS keyring (or Fernet-encrypted
fallback for headless/CI). The JSON config file stores non-secret settings only.
Legacy plaintext keys are migrated on first load and removed from JSON.
"""
import os
import json
import logging
import warnings
from pathlib import Path

import keyring
from cryptography.fernet import Fernet

CONFIG_DIR   = Path.home() / ".saido_agent"
CONFIG_FILE  = CONFIG_DIR  / "config.json"
HISTORY_FILE = CONFIG_DIR  / "input_history.txt"
SESSIONS_DIR = CONFIG_DIR  / "sessions"
ENCRYPTED_KEYS_FILE = CONFIG_DIR / "keys.enc"

MR_SESSION_DIR = SESSIONS_DIR / "mr_sessions"

# Keys that are considered secrets and must never be stored in config.json
_API_KEY_FIELDS = [
    "anthropic_api_key",
    "openai_api_key",
    "gemini_api_key",
    "kimi_api_key",
    "qwen_api_key",
    "zhipu_api_key",
    "deepseek_api_key",
    "api_key",  # legacy field
]

DEFAULTS = {
    "model":            "claude-opus-4-6",
    "max_tokens":       8192,
    "permission_mode":  "auto",   # auto | accept-all | manual
    "verbose":          False,
    "thinking":         False,
    "thinking_budget":  10000,
    "custom_base_url":  "",       # for "custom" provider
    "max_tool_output":  32000,
    "max_agent_depth":  3,
    "max_concurrent_agents": 3,
}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyring helpers
# ---------------------------------------------------------------------------

_KEYRING_SERVICE = "saido_agent"


def _keyring_available() -> bool:
    """Check if keyring backend is functional (not just importable)."""
    try:
        # Attempt a no-op read; some CI backends raise on any call
        keyring.get_password(_KEYRING_SERVICE, "__probe__")
        return True
    except Exception:
        return False


def store_api_key(provider_name: str, api_key: str) -> None:
    """Store an API key securely.

    Tries OS keyring first, falls back to Fernet-encrypted file.
    """
    if _keyring_available():
        keyring.set_password(_KEYRING_SERVICE, provider_name, api_key)
    else:
        _store_key_encrypted(provider_name, api_key)


def retrieve_api_key(provider_name: str) -> str:
    """Retrieve an API key from secure storage.

    Checks keyring first, then encrypted fallback, then env vars.
    """
    # 1. Keyring
    if _keyring_available():
        val = keyring.get_password(_KEYRING_SERVICE, provider_name)
        if val:
            return val

    # 2. Encrypted file fallback
    val = _load_key_encrypted(provider_name)
    if val:
        return val

    return ""


def _get_fernet() -> Fernet:
    """Get or create a Fernet instance for the encrypted keys file.

    The encryption passphrase is derived from an env var
    (SAIDO_AGENT_KEY_PASSPHRASE) for headless/CI use.  If absent, a
    machine-specific key is generated and stored in keyring.
    """
    passphrase = os.environ.get("SAIDO_AGENT_KEY_PASSPHRASE")
    if passphrase:
        import hashlib, base64
        dk = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), b"saido_agent_salt", 100_000)
        return Fernet(base64.urlsafe_b64encode(dk))

    # Try to retrieve or generate a machine key stored in keyring
    machine_key = None
    if _keyring_available():
        machine_key = keyring.get_password(_KEYRING_SERVICE, "__fernet_machine_key__")
    if not machine_key:
        machine_key = Fernet.generate_key().decode()
        if _keyring_available():
            keyring.set_password(_KEYRING_SERVICE, "__fernet_machine_key__", machine_key)
    return Fernet(machine_key.encode() if isinstance(machine_key, str) else machine_key)


def _store_key_encrypted(provider_name: str, api_key: str) -> None:
    """Store a key in the Fernet-encrypted keys.enc file."""
    data = _load_all_encrypted_keys()
    data[provider_name] = api_key
    fernet = _get_fernet()
    CONFIG_DIR.mkdir(exist_ok=True)
    ENCRYPTED_KEYS_FILE.write_bytes(fernet.encrypt(json.dumps(data).encode()))


def _load_all_encrypted_keys() -> dict:
    if not ENCRYPTED_KEYS_FILE.exists():
        return {}
    try:
        fernet = _get_fernet()
        return json.loads(fernet.decrypt(ENCRYPTED_KEYS_FILE.read_bytes()))
    except Exception:
        return {}


def _load_key_encrypted(provider_name: str) -> str:
    return _load_all_encrypted_keys().get(provider_name, "")


# ---------------------------------------------------------------------------
# Legacy migration
# ---------------------------------------------------------------------------

def _migrate_legacy_keys(cfg: dict) -> dict:
    """Migrate plaintext API keys from config dict to secure storage.

    Returns the config dict with secret fields removed.
    """
    migrated_any = False
    for key_field in _API_KEY_FIELDS:
        value = cfg.get(key_field)
        if value and isinstance(value, str) and value.strip():
            # Derive provider name from field (e.g. "anthropic_api_key" -> "anthropic")
            provider = key_field.replace("_api_key", "") if key_field != "api_key" else "anthropic"
            store_api_key(provider, value.strip())
            del cfg[key_field]
            migrated_any = True

    if migrated_any:
        warnings.warn(
            "Plaintext API keys found in config.json have been migrated to "
            "secure storage (OS keyring / encrypted file) and removed from "
            "the config file. This is a one-time migration.",
            UserWarning,
            stacklevel=3,
        )
        # Re-save config without the secret fields
        _save_config_raw(cfg)

    return cfg


# ---------------------------------------------------------------------------
# Public config API
# ---------------------------------------------------------------------------

def load_config() -> dict:
    CONFIG_DIR.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(exist_ok=True)
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text()))
        except Exception:
            pass

    # Backward-compat: legacy single api_key -> anthropic_api_key
    if cfg.get("api_key") and not cfg.get("anthropic_api_key"):
        cfg["anthropic_api_key"] = cfg.pop("api_key")

    # Migrate any plaintext keys to secure storage
    cfg = _migrate_legacy_keys(cfg)

    # Resolve anthropic key from secure storage / env if not already set
    if not cfg.get("anthropic_api_key"):
        secure_key = retrieve_api_key("anthropic")
        if secure_key:
            cfg["anthropic_api_key"] = secure_key
        else:
            cfg["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")

    return cfg


def _save_config_raw(cfg: dict) -> None:
    """Write config dict to JSON, stripping any secret fields."""
    CONFIG_DIR.mkdir(exist_ok=True)
    data = {k: v for k, v in cfg.items() if k not in _API_KEY_FIELDS}
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


def save_config(cfg: dict):
    """Save config.  API key fields are routed to secure storage, not JSON."""
    CONFIG_DIR.mkdir(exist_ok=True)
    # Extract and store any API keys that were set in the dict
    for key_field in _API_KEY_FIELDS:
        value = cfg.get(key_field)
        if value and isinstance(value, str) and value.strip():
            provider = key_field.replace("_api_key", "") if key_field != "api_key" else "anthropic"
            store_api_key(provider, value.strip())

    # Write only non-secret settings to JSON
    _save_config_raw(cfg)


def current_provider(cfg: dict) -> str:
    from saido_agent.core.providers import detect_provider
    return detect_provider(cfg.get("model", "claude-opus-4-6"))


def has_api_key(cfg: dict) -> bool:
    """Check whether the active provider has an API key configured."""
    from saido_agent.core.providers import get_api_key
    pname = current_provider(cfg)
    key = get_api_key(pname, cfg)
    if key:
        return True
    # Also check secure storage
    return bool(retrieve_api_key(pname))


def calc_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    from saido_agent.core.providers import calc_cost as _cc
    return _cc(model, in_tokens, out_tokens)
