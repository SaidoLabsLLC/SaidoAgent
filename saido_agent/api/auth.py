"""Authentication and tenant isolation for Saido Agent API.

Provides:
  - API key management (create, verify, revoke) with SHA-256 hashing
  - JWT session tokens for stateless auth (tenant-level and user-level)
  - Per-key rate limiting with in-memory sliding window
  - FastAPI dependency ``get_current_tenant`` for route injection
  - ``AuthContext`` dataclass carrying user_id, team_id, and role
  - ``get_auth_context`` dependency for RBAC-aware routes
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jwt as pyjwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAIDO_DIR = Path.home() / ".saido_agent"
_API_KEYS_FILE = _SAIDO_DIR / "api_keys.json"
_JWT_SECRET_FILE = _SAIDO_DIR / "jwt_secret"
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRY_SECONDS = 3600  # 1 hour

# Default rate limit: 60 requests per minute
DEFAULT_RATE_LIMIT = 60
_RATE_WINDOW_SECONDS = 60

# FastAPI security scheme
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_key(key: str) -> str:
    """SHA-256 hash of an API key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _ensure_dir() -> None:
    """Ensure the ~/.saido_agent directory exists."""
    _SAIDO_DIR.mkdir(parents=True, exist_ok=True)


def _load_keys() -> dict:
    """Load the API keys store from disk.

    Structure::

        {
            "<sha256_hash>": {
                "tenant_id": "...",
                "created_at": <unix_ts>,
                "rate_limit": 60,
                "revoked": false
            }
        }
    """
    if not _API_KEYS_FILE.exists():
        return {}
    try:
        return json.loads(_API_KEYS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupted api_keys.json — returning empty store")
        return {}


def _save_keys(data: dict) -> None:
    """Persist the API keys store to disk."""
    _ensure_dir()
    _API_KEYS_FILE.write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def _get_jwt_secret() -> str:
    """Load or create the JWT signing secret."""
    _ensure_dir()
    if _JWT_SECRET_FILE.exists():
        return _JWT_SECRET_FILE.read_text(encoding="utf-8").strip()
    secret = secrets.token_hex(32)
    _JWT_SECRET_FILE.write_text(secret, encoding="utf-8")
    return secret


# ---------------------------------------------------------------------------
# Public API — Key management
# ---------------------------------------------------------------------------

def create_api_key(
    tenant_id: str,
    rate_limit: int = DEFAULT_RATE_LIMIT,
) -> str:
    """Create a new API key for a tenant.

    Returns the plaintext key (shown once — never stored).
    """
    plaintext = f"sk-saido-{secrets.token_hex(24)}"
    hashed = _hash_key(plaintext)

    store = _load_keys()
    store[hashed] = {
        "tenant_id": tenant_id,
        "created_at": time.time(),
        "rate_limit": rate_limit,
        "revoked": False,
    }
    _save_keys(store)
    logger.info("Created API key for tenant %s", tenant_id)
    return plaintext


def verify_api_key(key: str) -> Optional[dict]:
    """Verify an API key.

    Returns the key metadata dict (with ``tenant_id``) or ``None``.
    """
    hashed = _hash_key(key)
    store = _load_keys()
    entry = store.get(hashed)
    if entry is None:
        return None
    if entry.get("revoked", False):
        return None
    return entry


def revoke_api_key(key: str) -> bool:
    """Revoke an API key. Returns True if found and revoked."""
    hashed = _hash_key(key)
    store = _load_keys()
    if hashed not in store:
        return False
    store[hashed]["revoked"] = True
    _save_keys(store)
    return True


def list_api_keys() -> list[dict]:
    """List all API keys (metadata only, no plaintext)."""
    store = _load_keys()
    return [
        {"hash_prefix": h[:12], **v}
        for h, v in store.items()
    ]


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------

def create_jwt_token(tenant_id: str) -> str:
    """Create a JWT token for a tenant."""
    secret = _get_jwt_secret()
    payload = {
        "tenant_id": tenant_id,
        "iat": time.time(),
        "exp": time.time() + _JWT_EXPIRY_SECONDS,
    }
    return pyjwt.encode(payload, secret, algorithm=_JWT_ALGORITHM)


def create_user_jwt_token(
    user_id: str,
    team_id: str,
    role: str,
) -> str:
    """Create a JWT token for an authenticated user with team context.

    The token includes user_id, team_id, and role so downstream
    dependencies can enforce RBAC without a database round-trip.
    """
    secret = _get_jwt_secret()
    payload = {
        "user_id": user_id,
        "team_id": team_id,
        "role": role,
        "iat": time.time(),
        "exp": time.time() + _JWT_EXPIRY_SECONDS,
    }
    return pyjwt.encode(payload, secret, algorithm=_JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[dict | str]:
    """Verify a JWT token.

    Returns:
      - A dict with ``user_id``, ``team_id``, ``role`` for user tokens
      - A string ``tenant_id`` for legacy tenant tokens
      - ``None`` on failure
    """
    secret = _get_jwt_secret()
    try:
        payload = pyjwt.decode(token, secret, algorithms=[_JWT_ALGORITHM])
        # User-level token (Phase 3)
        if "user_id" in payload:
            return {
                "user_id": payload["user_id"],
                "team_id": payload.get("team_id", ""),
                "role": payload.get("role", "viewer"),
            }
        # Legacy tenant token (Phase 2)
        return payload.get("tenant_id")
    except pyjwt.ExpiredSignatureError:
        logger.debug("JWT token expired")
        return None
    except pyjwt.InvalidTokenError:
        logger.debug("Invalid JWT token")
        return None


# ---------------------------------------------------------------------------
# Rate limiting — in-memory sliding window
# ---------------------------------------------------------------------------

# { tenant_id: [timestamp, ...] }
_rate_counters: dict[str, list[float]] = {}


def check_rate_limit(tenant_id: str, limit: int = DEFAULT_RATE_LIMIT) -> bool:
    """Check if a tenant is within rate limits.

    Returns ``True`` if allowed, ``False`` if rate-limited.
    """
    now = time.time()
    window_start = now - _RATE_WINDOW_SECONDS

    if tenant_id not in _rate_counters:
        _rate_counters[tenant_id] = []

    # Prune old entries
    timestamps = _rate_counters[tenant_id]
    _rate_counters[tenant_id] = [t for t in timestamps if t > window_start]

    if len(_rate_counters[tenant_id]) >= limit:
        return False

    _rate_counters[tenant_id].append(now)
    return True


def reset_rate_limits() -> None:
    """Clear all rate limit counters (for testing)."""
    _rate_counters.clear()


# ---------------------------------------------------------------------------
# Tenant directory management
# ---------------------------------------------------------------------------

def get_tenant_knowledge_dir(tenant_id: str) -> str:
    """Return the knowledge directory for a tenant, creating it if needed."""
    tenant_dir = _SAIDO_DIR / "tenants" / tenant_id / "knowledge"
    tenant_dir.mkdir(parents=True, exist_ok=True)
    return str(tenant_dir)


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_current_tenant(
    request: Request,
    api_key: Optional[str] = Depends(_api_key_header),
) -> str:
    """FastAPI dependency: extract and validate tenant from API key or JWT.

    Checks in order:
      1. ``X-API-Key`` header
      2. ``Authorization: Bearer <jwt>`` header

    Returns the ``tenant_id`` string.
    Raises ``HTTPException(401)`` on failure.
    Raises ``HTTPException(429)`` if rate-limited.
    """
    tenant_id: Optional[str] = None
    rate_limit = DEFAULT_RATE_LIMIT

    # 1. Try API key
    if api_key:
        entry = verify_api_key(api_key)
        if entry is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
        tenant_id = entry["tenant_id"]
        rate_limit = entry.get("rate_limit", DEFAULT_RATE_LIMIT)

    # 2. Try JWT bearer token
    if tenant_id is None:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = verify_jwt_token(token)
            if payload is None:
                raise HTTPException(
                    status_code=401, detail="Invalid or expired JWT token"
                )
            # Handle both user-level dict and legacy string payloads
            if isinstance(payload, dict):
                tenant_id = payload.get("team_id") or payload.get("user_id", "")
            else:
                tenant_id = payload

    if tenant_id is None:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication. Provide X-API-Key header or Bearer token.",
        )

    # Rate limit check
    if not check_rate_limit(tenant_id, rate_limit):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )

    return tenant_id


# ---------------------------------------------------------------------------
# AuthContext — RBAC-aware authentication context (Phase 3)
# ---------------------------------------------------------------------------

@dataclass
class AuthContext:
    """Authentication context returned by ``get_auth_context``.

    For user-level tokens: all fields populated.
    For legacy API-key/tenant tokens: ``tenant_id`` set, others may be None.
    """

    tenant_id: str
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    role: Optional["_Role"] = None

    @property
    def is_user_auth(self) -> bool:
        """True if this context was created from a user JWT."""
        return self.user_id is not None


# Import Role lazily to avoid circular imports
_Role = None


def _get_role_enum():
    global _Role
    if _Role is None:
        from saido_agent.api.rbac import Role
        _Role = Role
    return _Role


async def get_auth_context(
    request: Request,
    api_key: Optional[str] = Depends(_api_key_header),
) -> AuthContext:
    """FastAPI dependency: extract authentication context with RBAC info.

    Works with both legacy API keys (Phase 2) and user JWT tokens (Phase 3).
    Returns an ``AuthContext`` with tenant_id, user_id, team_id, and role.

    Raises ``HTTPException(401)`` on failure.
    Raises ``HTTPException(429)`` if rate-limited.
    """
    Role = _get_role_enum()
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    role: Optional[Role] = None
    rate_limit = DEFAULT_RATE_LIMIT

    # 1. Try API key
    if api_key:
        entry = verify_api_key(api_key)
        if entry is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
        tenant_id = entry["tenant_id"]
        rate_limit = entry.get("rate_limit", DEFAULT_RATE_LIMIT)

    # 2. Try JWT bearer token
    if tenant_id is None:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = verify_jwt_token(token)
            if payload is None:
                raise HTTPException(
                    status_code=401, detail="Invalid or expired JWT token"
                )
            if isinstance(payload, dict):
                # User-level token
                user_id = payload["user_id"]
                team_id = payload.get("team_id", "")
                role_str = payload.get("role", "viewer")
                try:
                    role = Role(role_str)
                except ValueError:
                    role = Role.VIEWER
                tenant_id = team_id  # Use team_id as tenant scope
            else:
                # Legacy tenant token
                tenant_id = payload

    if tenant_id is None:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication. Provide X-API-Key header or Bearer token.",
        )

    # Rate limit check
    if not check_rate_limit(tenant_id, rate_limit):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )

    return AuthContext(
        tenant_id=tenant_id,
        user_id=user_id,
        team_id=team_id,
        role=role,
    )
