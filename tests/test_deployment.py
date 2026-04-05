"""Tests for deployment configuration files and infrastructure.

Validates Dockerfile, docker-compose.yml, migration SQL, the migration
runner, .env.example, and CI workflow YAML.
"""

from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Dockerfile
# ---------------------------------------------------------------------------

class TestDockerfile:
    """Validate Dockerfile structure."""

    def test_dockerfile_exists(self):
        assert (ROOT / "Dockerfile").exists(), "Dockerfile not found at project root"

    def test_dockerfile_has_from(self):
        content = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "FROM" in content, "Dockerfile must contain a FROM instruction"

    def test_dockerfile_multi_stage(self):
        content = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        from_count = sum(1 for line in content.splitlines() if line.strip().startswith("FROM "))
        assert from_count >= 2, "Dockerfile should use multi-stage build (>= 2 FROM instructions)"

    def test_dockerfile_non_root_user(self):
        content = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "USER" in content, "Dockerfile should switch to a non-root USER"

    def test_dockerfile_healthcheck(self):
        content = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "HEALTHCHECK" in content, "Dockerfile should contain a HEALTHCHECK instruction"

    def test_dockerfile_expose(self):
        content = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "EXPOSE 8000" in content, "Dockerfile should EXPOSE port 8000"


# ---------------------------------------------------------------------------
# 2. docker-compose.yml
# ---------------------------------------------------------------------------

class TestDockerCompose:
    """Validate docker-compose.yml structure."""

    def test_compose_file_exists(self):
        assert (ROOT / "docker-compose.yml").exists(), "docker-compose.yml not found"

    def test_compose_valid_yaml(self):
        content = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert isinstance(data, dict), "docker-compose.yml should parse to a dict"

    def test_compose_has_services(self):
        content = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert "services" in data, "docker-compose.yml must define services"
        assert "saido-agent" in data["services"], "Must define saido-agent service"

    def test_compose_has_volumes(self):
        content = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert "volumes" in data, "docker-compose.yml must define volumes"

    def test_compose_healthcheck(self):
        content = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        service = data["services"]["saido-agent"]
        assert "healthcheck" in service, "saido-agent service must have a healthcheck"

    def test_compose_port_mapping(self):
        content = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        service = data["services"]["saido-agent"]
        assert "ports" in service, "saido-agent service must expose ports"


# ---------------------------------------------------------------------------
# 3. Migration SQL files
# ---------------------------------------------------------------------------

class TestMigrations:
    """Validate SQL migration files and the migration runner."""

    MIGRATIONS_DIR = ROOT / "saido_agent" / "api" / "migrations"

    def test_migrations_dir_exists(self):
        assert self.MIGRATIONS_DIR.exists(), "migrations/ directory not found"

    def test_initial_migration_exists(self):
        assert (self.MIGRATIONS_DIR / "001_initial.sql").exists(), "001_initial.sql not found"

    def test_initial_migration_valid_sql(self):
        """Verify the SQL is syntactically valid by executing against in-memory SQLite."""
        sql = (self.MIGRATIONS_DIR / "001_initial.sql").read_text(encoding="utf-8")
        conn = sqlite3.connect(":memory:")
        conn.executescript(sql)
        # Verify tables were created
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "tenants" in tables
        assert "articles_meta" in tables
        assert "sessions" in tables
        assert "audit_log" in tables
        conn.close()

    def test_migration_files_sorted(self):
        """Migration files should be named with numeric prefixes for ordering."""
        files = sorted(self.MIGRATIONS_DIR.glob("*.sql"))
        assert len(files) >= 1, "At least one migration file required"
        assert files[0].name.startswith("001"), "First migration should be 001_"


# ---------------------------------------------------------------------------
# 4. Migration runner (db.py)
# ---------------------------------------------------------------------------

class TestMigrationRunner:
    """Validate the migration runner creates tables correctly."""

    def test_db_module_importable(self):
        from saido_agent.api.db import run_migrations, get_connection  # noqa: F401

    def test_run_migrations_in_memory(self):
        from saido_agent.api.db import run_migrations, get_connection

        applied = run_migrations(db_path=":memory:")
        assert "001_initial.sql" in applied, "001_initial.sql should be applied"

    def test_run_migrations_idempotent(self, tmp_path):
        from saido_agent.api.db import run_migrations

        db_file = tmp_path / "test.db"
        first_run = run_migrations(db_path=db_file)
        second_run = run_migrations(db_path=db_file)
        assert len(first_run) >= 1, "First run should apply migrations"
        assert len(second_run) == 0, "Second run should apply no migrations (idempotent)"

    def test_tables_created_correctly(self, tmp_path):
        from saido_agent.api.db import run_migrations, get_connection

        db_file = tmp_path / "test.db"
        run_migrations(db_path=db_file)

        conn = get_connection(db_file)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        expected = {"tenants", "articles_meta", "sessions", "audit_log", "_migrations"}
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    def test_foreign_keys_enabled(self, tmp_path):
        from saido_agent.api.db import get_connection

        db_file = tmp_path / "test.db"
        conn = get_connection(db_file)
        cursor = conn.execute("PRAGMA foreign_keys")
        assert cursor.fetchone()[0] == 1, "Foreign keys should be enabled"
        conn.close()


# ---------------------------------------------------------------------------
# 5. .env.example
# ---------------------------------------------------------------------------

class TestEnvExample:
    """Validate .env.example contains all required variables."""

    REQUIRED_VARS = [
        "SAIDO_JWT_SECRET",
        "SAIDO_KNOWLEDGE_DIR",
        "SAIDO_API_PORT",
        "SAIDO_LOG_LEVEL",
    ]

    def test_env_example_exists(self):
        assert (ROOT / ".env.example").exists(), ".env.example not found"

    def test_env_example_has_required_vars(self):
        content = (ROOT / ".env.example").read_text(encoding="utf-8")
        for var in self.REQUIRED_VARS:
            assert var in content, f".env.example missing required variable: {var}"


# ---------------------------------------------------------------------------
# 6. CI workflow
# ---------------------------------------------------------------------------

class TestCIWorkflow:
    """Validate GitHub Actions CI workflow."""

    CI_PATH = ROOT / ".github" / "workflows" / "ci.yml"

    def test_ci_workflow_exists(self):
        assert self.CI_PATH.exists(), ".github/workflows/ci.yml not found"

    def test_ci_workflow_valid_yaml(self):
        content = self.CI_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert isinstance(data, dict), "ci.yml should parse to a dict"

    def test_ci_workflow_has_jobs(self):
        content = self.CI_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert "jobs" in data, "ci.yml must define jobs"

    def test_ci_workflow_has_test_job(self):
        content = self.CI_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert "test" in data["jobs"], "ci.yml must have a 'test' job"

    def test_ci_workflow_has_docker_job(self):
        content = self.CI_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert "docker" in data["jobs"], "ci.yml must have a 'docker' job"

    def test_ci_workflow_triggers(self):
        content = self.CI_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        assert "on" in data or True in data, "ci.yml must define triggers"
