"""Role-Based Access Control (RBAC) for Saido Agent API.

Provides:
  - Role enum with three tiers: admin, editor, viewer
  - Permission mappings per role
  - FastAPI dependencies for permission enforcement
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException

if TYPE_CHECKING:
    from saido_agent.api.auth import AuthContext


class Role(str, Enum):
    """Team membership roles, ordered by privilege level."""

    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


# Each role has an explicit set of allowed permissions.
# Higher roles are supersets of lower roles.
ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.VIEWER: {
        "read_articles",
        "query",
        "search",
        "view_stats",
    },
    Role.EDITOR: {
        "read_articles",
        "query",
        "search",
        "view_stats",
        "ingest",
        "compile",
        "edit_articles",
        "run_agent",
    },
    Role.ADMIN: {
        "read_articles",
        "query",
        "search",
        "view_stats",
        "ingest",
        "compile",
        "edit_articles",
        "run_agent",
        "manage_members",
        "manage_keys",
        "manage_settings",
        "delete_articles",
        "manage_billing",
    },
}


def check_permission(role: Role, permission: str) -> bool:
    """Return True if the given role has the specified permission."""
    return permission in ROLE_PERMISSIONS.get(role, set())


def require_permission(permission: str):
    """FastAPI dependency factory that checks the authenticated user has
    the required permission.

    Usage::

        @router.post("/ingest")
        async def ingest(ctx: AuthContext = Depends(require_permission("ingest"))):
            ...

    Returns the AuthContext if the check passes, so downstream code
    can access ``ctx.user_id``, ``ctx.team_id``, ``ctx.role``.
    """

    async def _check(
        ctx=Depends(_get_auth_context_dep()),
    ):
        if ctx.role is None:
            # Legacy API-key auth with no user/role -- treat as admin
            # for backward compatibility with Phase 2 keys.
            return ctx
        if not check_permission(ctx.role, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: '{permission}' requires role "
                f"'{_min_role_for(permission)}' or higher, "
                f"but your role is '{ctx.role.value}'.",
            )
        return ctx

    return _check


def require_viewer():
    """Dependency: require at least viewer-level access."""
    return require_permission("read_articles")


def require_editor():
    """Dependency: require at least editor-level access."""
    return require_permission("ingest")


def require_admin():
    """Dependency: require admin-level access."""
    return require_permission("manage_members")


def _min_role_for(permission: str) -> str:
    """Return the minimum role name that grants a permission."""
    for role in (Role.VIEWER, Role.EDITOR, Role.ADMIN):
        if permission in ROLE_PERMISSIONS[role]:
            return role.value
    return "admin"


def _get_auth_context_dep():
    """Lazy import to avoid circular dependency with auth module.

    Returns the ``get_auth_context`` FastAPI dependency function.
    """
    from saido_agent.api.auth import get_auth_context

    return get_auth_context
