"""User and team management for Saido Agent.

Provides:
  - User CRUD with secure password hashing (hashlib.scrypt)
  - Authentication by email + password
  - Team CRUD with membership management
  - All operations use the SQLite database from ``db.py``

Password hashing uses ``hashlib.scrypt`` from the standard library with
a 32-byte random salt, N=2^14, r=8, p=1.  The stored format is::

    scrypt:<hex_salt>:<hex_derived_key>
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sqlite3
from typing import Optional

from saido_agent.api.db import get_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Password hashing (stdlib scrypt -- no external dependency)
# ---------------------------------------------------------------------------

_SCRYPT_N = 2**14  # CPU/memory cost parameter
_SCRYPT_R = 8  # Block size
_SCRYPT_P = 1  # Parallelization
_SCRYPT_DKLEN = 64  # Derived key length


def _hash_password(password: str) -> str:
    """Hash a password using scrypt. Returns ``scrypt:<salt>:<key>``."""
    salt = os.urandom(32)
    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return f"scrypt:{salt.hex()}:{dk.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored scrypt hash."""
    try:
        prefix, salt_hex, key_hex = stored_hash.split(":")
        if prefix != "scrypt":
            return False
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        dk = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
            dklen=_SCRYPT_DKLEN,
        )
        return secrets.compare_digest(dk, expected_key)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------


def create_user(
    email: str,
    name: str,
    password: str,
    db_path: str | None = None,
) -> dict:
    """Create a new user. Returns the user dict (without password_hash).

    Raises ``ValueError`` if the email is already registered.
    """
    user_id = secrets.token_hex(16)
    password_hash = _hash_password(password)

    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO users (id, email, name, password_hash) VALUES (?, ?, ?, ?)",
            (user_id, email.lower().strip(), name, password_hash),
        )
        conn.commit()
        logger.info("Created user %s (%s)", user_id, email)
        return {"id": user_id, "email": email.lower().strip(), "name": name}
    except sqlite3.IntegrityError:
        raise ValueError(f"Email '{email}' is already registered")
    finally:
        conn.close()


def authenticate_user(
    email: str,
    password: str,
    db_path: str | None = None,
) -> Optional[dict]:
    """Authenticate a user by email and password.

    Returns the user dict (without password_hash) or None.
    """
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT id, email, name, password_hash FROM users WHERE email = ?",
            (email.lower().strip(),),
        ).fetchone()
        if row is None:
            return None
        if not _verify_password(password, row["password_hash"]):
            return None
        return {"id": row["id"], "email": row["email"], "name": row["name"]}
    finally:
        conn.close()


def get_user(
    user_id: str,
    db_path: str | None = None,
) -> Optional[dict]:
    """Get a user by ID. Returns user dict or None."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT id, email, name, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Team CRUD
# ---------------------------------------------------------------------------


def create_team(
    name: str,
    owner_id: str,
    db_path: str | None = None,
) -> dict:
    """Create a new team and add the owner as an admin member.

    Returns the team dict.
    """
    team_id = secrets.token_hex(16)
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO teams (id, name, owner_id) VALUES (?, ?, ?)",
            (team_id, name, owner_id),
        )
        # Owner is automatically an admin member
        conn.execute(
            "INSERT INTO team_members (team_id, user_id, role) VALUES (?, ?, ?)",
            (team_id, owner_id, "admin"),
        )
        conn.commit()
        logger.info("Created team %s (%s) owned by %s", team_id, name, owner_id)
        return {"id": team_id, "name": name, "owner_id": owner_id}
    finally:
        conn.close()


def get_team(
    team_id: str,
    db_path: str | None = None,
) -> Optional[dict]:
    """Get a team by ID."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT id, name, owner_id, created_at FROM teams WHERE id = ?",
            (team_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_user_teams(
    user_id: str,
    db_path: str | None = None,
) -> list[dict]:
    """List all teams a user belongs to, including their role."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT t.id, t.name, t.owner_id, tm.role
            FROM teams t
            JOIN team_members tm ON t.id = tm.team_id
            WHERE tm.user_id = ?
            ORDER BY t.name
            """,
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def add_member(
    team_id: str,
    user_id: str,
    role: str = "viewer",
    db_path: str | None = None,
) -> dict:
    """Add a user to a team with the given role.

    Raises ``ValueError`` if the user is already a member.
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO team_members (team_id, user_id, role) VALUES (?, ?, ?)",
            (team_id, user_id, role),
        )
        conn.commit()
        logger.info("Added user %s to team %s as %s", user_id, team_id, role)
        return {"team_id": team_id, "user_id": user_id, "role": role}
    except sqlite3.IntegrityError:
        raise ValueError(f"User '{user_id}' is already a member of team '{team_id}'")
    finally:
        conn.close()


def remove_member(
    team_id: str,
    user_id: str,
    db_path: str | None = None,
) -> bool:
    """Remove a user from a team. Returns True if removed, False if not found."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "DELETE FROM team_members WHERE team_id = ? AND user_id = ?",
            (team_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def update_member_role(
    team_id: str,
    user_id: str,
    new_role: str,
    db_path: str | None = None,
) -> bool:
    """Update a member's role. Returns True if updated, False if not found."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "UPDATE team_members SET role = ? WHERE team_id = ? AND user_id = ?",
            (new_role, team_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_member_role(
    team_id: str,
    user_id: str,
    db_path: str | None = None,
) -> Optional[str]:
    """Get a user's role in a team, or None if not a member."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT role FROM team_members WHERE team_id = ? AND user_id = ?",
            (team_id, user_id),
        ).fetchone()
        return row["role"] if row else None
    finally:
        conn.close()


def list_team_members(
    team_id: str,
    db_path: str | None = None,
) -> list[dict]:
    """List all members of a team."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT u.id, u.email, u.name, tm.role, tm.joined_at
            FROM team_members tm
            JOIN users u ON tm.user_id = u.id
            WHERE tm.team_id = ?
            ORDER BY tm.joined_at
            """,
            (team_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
