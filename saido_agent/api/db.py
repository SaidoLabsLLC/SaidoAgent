"""SQLite database management and migration runner for Saido Agent.

Provides:
  - Automatic database initialization at ``~/.saido_agent/saido.db``
  - Sequential SQL migration runner that reads ``.sql`` files from
    ``saido_agent/api/migrations/`` and applies them in lexicographic order
  - Connection helper for application code

Usage::

    from saido_agent.api.db import get_connection, run_migrations

    run_migrations()          # apply any pending migrations
    conn = get_connection()   # get a connection to the database
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SAIDO_DIR = Path.home() / ".saido_agent"
_DB_PATH = _SAIDO_DIR / "saido.db"
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def get_db_path() -> Path:
    """Return the path to the SQLite database file."""
    return _DB_PATH


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Return a new SQLite connection.

    Parameters
    ----------
    db_path:
        Override the default database path (useful for testing).
        Pass ``":memory:"`` for an in-memory database.
    """
    if db_path is None:
        db_path = _DB_PATH
    db_path = Path(db_path) if db_path != ":memory:" else db_path

    # Ensure parent directory exists for file-based databases
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------

_MIGRATION_TRACKING_SQL = """
CREATE TABLE IF NOT EXISTS _migrations (
    filename TEXT PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _get_applied_migrations(conn: sqlite3.Connection) -> set[str]:
    """Return the set of already-applied migration filenames."""
    conn.execute(_MIGRATION_TRACKING_SQL)
    cursor = conn.execute("SELECT filename FROM _migrations")
    return {row[0] for row in cursor.fetchall()}


def get_migration_files(migrations_dir: Path | None = None) -> list[Path]:
    """Return migration SQL files sorted in lexicographic order."""
    directory = migrations_dir or _MIGRATIONS_DIR
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.sql"))
    return files


def run_migrations(
    db_path: Path | str | None = None,
    migrations_dir: Path | None = None,
) -> list[str]:
    """Apply all pending SQL migrations in order.

    Parameters
    ----------
    db_path:
        Override the database path (pass ``":memory:"`` for testing).
    migrations_dir:
        Override the migrations directory.

    Returns
    -------
    list[str]
        Filenames of newly applied migrations.
    """
    conn = get_connection(db_path)
    applied = _get_applied_migrations(conn)
    migration_files = get_migration_files(migrations_dir)

    newly_applied: list[str] = []

    for migration_file in migration_files:
        if migration_file.name in applied:
            logger.debug("Skipping already-applied migration: %s", migration_file.name)
            continue

        logger.info("Applying migration: %s", migration_file.name)
        sql = migration_file.read_text(encoding="utf-8")

        try:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO _migrations (filename) VALUES (?)",
                (migration_file.name,),
            )
            conn.commit()
            newly_applied.append(migration_file.name)
            logger.info("Successfully applied migration: %s", migration_file.name)
        except sqlite3.Error as exc:
            conn.rollback()
            logger.error(
                "Failed to apply migration %s: %s", migration_file.name, exc
            )
            raise

    conn.close()
    return newly_applied
