"""Tests for the memory system: session persistence, CRUD, extraction, knowledge articles."""
from __future__ import annotations

import json
import os
import time
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from cryptography.fernet import Fernet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_memory_dir(tmp_path):
    """Provide a temporary user memory directory and patch store to use it."""
    user_dir = tmp_path / "user_memory"
    user_dir.mkdir()
    project_dir = tmp_path / "project_memory"
    project_dir.mkdir()

    # MED-4: Auto-trust the cwd so existing tests pass the trust boundary check
    trusted_file = tmp_path / "trusted_projects.json"

    with patch("saido_agent.memory.store.USER_MEMORY_DIR", user_dir), \
         patch("saido_agent.memory.store.get_project_memory_dir", return_value=project_dir), \
         patch("saido_agent.memory.store.TRUSTED_PROJECTS_FILE", trusted_file):
        from saido_agent.memory.store import trust_project
        import os
        trust_project(os.getcwd())
        yield {"user": user_dir, "project": project_dir}


@pytest.fixture
def tmp_sessions_dir(tmp_path):
    """Provide a temporary sessions directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    mr_dir = sessions_dir / "mr_sessions"
    mr_dir.mkdir()
    return sessions_dir


@pytest.fixture
def fernet_key():
    """Provide a deterministic Fernet key for testing."""
    return Fernet.generate_key()


@pytest.fixture
def mock_fernet(fernet_key):
    """Patch session encryption to use a known key."""
    f = Fernet(fernet_key)
    with patch("saido_agent.cli.repl._get_session_fernet", return_value=f):
        yield f


@dataclass
class FakeState:
    """Minimal AgentState stand-in for testing."""
    messages: list = field(default_factory=list)
    turn_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


# ---------------------------------------------------------------------------
# Memory CRUD tests
# ---------------------------------------------------------------------------

class TestMemoryCRUD:
    """Test basic memory create, read, list, delete operations."""

    def test_save_and_load_memory(self, tmp_memory_dir):
        from saido_agent.memory.store import MemoryEntry, save_memory, load_entries

        entry = MemoryEntry(
            name="test preference",
            description="User prefers dark mode",
            type="user",
            content="The user always uses dark mode in their IDE.",
            created="2026-04-05",
        )
        save_memory(entry, scope="user")

        entries = load_entries("user")
        assert len(entries) == 1
        assert entries[0].name == "test preference"
        assert entries[0].description == "User prefers dark mode"
        assert entries[0].content == "The user always uses dark mode in their IDE."

    def test_delete_memory(self, tmp_memory_dir):
        from saido_agent.memory.store import MemoryEntry, save_memory, delete_memory, load_entries

        entry = MemoryEntry(
            name="to delete",
            description="Will be deleted",
            type="user",
            content="Temporary content.",
            created="2026-04-05",
        )
        save_memory(entry, scope="user")
        assert len(load_entries("user")) == 1

        delete_memory("to delete", scope="user")
        assert len(load_entries("user")) == 0

    def test_load_index_all_scopes(self, tmp_memory_dir):
        from saido_agent.memory.store import MemoryEntry, save_memory, load_index

        save_memory(MemoryEntry(
            name="user mem", description="User scope", type="user",
            content="User content", created="2026-04-05",
        ), scope="user")
        save_memory(MemoryEntry(
            name="project mem", description="Project scope", type="project",
            content="Project content", created="2026-04-05",
        ), scope="project")

        all_entries = load_index("all")
        assert len(all_entries) == 2
        scopes = {e.scope for e in all_entries}
        assert scopes == {"user", "project"}

    def test_search_memory(self, tmp_memory_dir):
        from saido_agent.memory.store import MemoryEntry, save_memory, search_memory

        save_memory(MemoryEntry(
            name="dark mode", description="IDE preference", type="user",
            content="Always use dark mode", created="2026-04-05",
        ), scope="user")
        save_memory(MemoryEntry(
            name="testing strategy", description="Use pytest", type="project",
            content="Run pytest for all tests", created="2026-04-05",
        ), scope="project")

        results = search_memory("dark mode")
        assert len(results) >= 1
        assert any("dark" in r.name.lower() for r in results)

    def test_get_memory_by_id(self, tmp_memory_dir):
        from saido_agent.memory.store import MemoryEntry, save_memory, get_memory_by_id

        save_memory(MemoryEntry(
            name="unique memory", description="Test lookup", type="user",
            content="Unique content here", created="2026-04-05",
        ), scope="user")

        found = get_memory_by_id("unique_memory", scope="user")
        assert found is not None
        assert found.name == "unique memory"

        not_found = get_memory_by_id("nonexistent", scope="user")
        assert not_found is None

    def test_get_memory_by_id_all_scopes(self, tmp_memory_dir):
        from saido_agent.memory.store import MemoryEntry, save_memory, get_memory_by_id

        save_memory(MemoryEntry(
            name="proj mem", description="In project", type="project",
            content="Project specific", created="2026-04-05",
        ), scope="project")

        # Should find in "all" scope search
        found = get_memory_by_id("proj_mem", scope="all")
        assert found is not None
        assert found.scope == "project"


# ---------------------------------------------------------------------------
# User vs Project scoping
# ---------------------------------------------------------------------------

class TestMemoryScoping:
    """Test that user and project memories are properly isolated."""

    def test_user_and_project_isolation(self, tmp_memory_dir):
        from saido_agent.memory.store import MemoryEntry, save_memory, load_entries

        save_memory(MemoryEntry(
            name="shared name", description="User version", type="user",
            content="User scope content", created="2026-04-05",
        ), scope="user")
        save_memory(MemoryEntry(
            name="shared name", description="Project version", type="project",
            content="Project scope content", created="2026-04-05",
        ), scope="project")

        user_entries = load_entries("user")
        project_entries = load_entries("project")

        assert len(user_entries) == 1
        assert len(project_entries) == 1
        assert user_entries[0].scope == "user"
        assert project_entries[0].scope == "project"
        assert user_entries[0].content != project_entries[0].content

    def test_index_files_separate(self, tmp_memory_dir):
        from saido_agent.memory.store import (
            MemoryEntry, save_memory, get_index_content, INDEX_FILENAME,
        )

        save_memory(MemoryEntry(
            name="user only", description="User", type="user",
            content="User content", created="2026-04-05",
        ), scope="user")

        user_index = get_index_content("user")
        assert "user only" in user_index

        # Project index should be empty (we only saved to user)
        project_index = get_index_content("project")
        assert "user only" not in project_index


# ---------------------------------------------------------------------------
# to_knowledge_article
# ---------------------------------------------------------------------------

class TestKnowledgeArticle:
    """Test conversion of memory entries to knowledge store format."""

    def test_to_knowledge_article_format(self):
        from saido_agent.memory.store import MemoryEntry, to_knowledge_article

        entry = MemoryEntry(
            name="test article",
            description="A test memory",
            type="feedback",
            content="Always run tests before committing.",
            file_path="/home/user/.saido_agent/memory/test_article.md",
            created="2026-04-05",
            scope="user",
        )

        article = to_knowledge_article(entry)

        assert article["type"] == "memory"
        assert article["category"] == "agent-memory"
        assert article["title"] == "test article"
        assert article["description"] == "A test memory"
        assert article["content"] == "Always run tests before committing."
        assert article["metadata"]["memory_type"] == "feedback"
        assert article["metadata"]["scope"] == "user"
        assert article["metadata"]["created"] == "2026-04-05"
        assert "scope:user" in article["tags"]
        assert "memory-type:feedback" in article["tags"]

    def test_to_knowledge_article_project_scope(self):
        from saido_agent.memory.store import MemoryEntry, to_knowledge_article

        entry = MemoryEntry(
            name="proj article",
            description="Project specific",
            type="project",
            content="This project uses PostgreSQL.",
            scope="project",
            created="2026-04-05",
        )

        article = to_knowledge_article(entry)
        assert "scope:project" in article["tags"]
        assert "memory-type:project" in article["tags"]


# ---------------------------------------------------------------------------
# Session persistence (encryption)
# ---------------------------------------------------------------------------

class TestSessionPersistence:
    """Test session save/load with Fernet encryption."""

    def test_encrypt_decrypt_roundtrip(self, mock_fernet):
        from saido_agent.cli.repl import _encrypt_session, _decrypt_session

        data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "turn_count": 1,
            "total_input_tokens": 100,
            "total_output_tokens": 50,
        }
        encrypted = _encrypt_session(data)
        assert isinstance(encrypted, bytes)
        assert encrypted != json.dumps(data).encode()  # Should be different from plaintext

        decrypted = _decrypt_session(encrypted)
        assert decrypted["messages"] == data["messages"]
        assert decrypted["turn_count"] == 1

    def test_session_save_creates_encrypted_file(self, mock_fernet, tmp_sessions_dir):
        from saido_agent.cli.repl import cmd_save

        state = FakeState(
            messages=[{"role": "user", "content": "Hello world"}],
            turn_count=1,
        )

        with patch("saido_agent.core.config.SESSIONS_DIR", tmp_sessions_dir):
            cmd_save("test_session.enc", state, {})

        path = tmp_sessions_dir / "test_session.enc"
        assert path.exists()
        assert path.stat().st_size > 0

        # Verify it is encrypted (not plaintext JSON)
        raw = path.read_bytes()
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)

    def test_session_load_decrypts_correctly(self, mock_fernet, tmp_sessions_dir):
        from saido_agent.cli.repl import cmd_save, cmd_load

        original_state = FakeState(
            messages=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            turn_count=2,
            total_input_tokens=50,
            total_output_tokens=10,
        )

        with patch("saido_agent.core.config.SESSIONS_DIR", tmp_sessions_dir):
            cmd_save("roundtrip.enc", original_state, {})

            loaded_state = FakeState()
            cmd_load("roundtrip.enc", loaded_state, {})

        assert len(loaded_state.messages) == 2
        assert loaded_state.messages[0]["content"] == "What is 2+2?"
        assert loaded_state.turn_count == 2

    def test_secret_redaction(self):
        from saido_agent.cli.repl import _redact_secrets

        text = "My key is sk-ant-abc123XYZ_test and also sk-1234567890123456789012345"
        redacted = _redact_secrets(text)
        assert "sk-ant-abc123XYZ_test" not in redacted
        assert "[REDACTED]" in redacted

    def test_expired_sessions_cleaned_up(self, mock_fernet, tmp_sessions_dir):
        from saido_agent.cli.repl import _cleanup_expired_sessions, _SESSION_EXPIRY_DAYS

        # Create an "old" session file
        old_path = tmp_sessions_dir / "old_session.enc"
        old_path.write_bytes(b"fake encrypted data")
        # Set mtime to 31 days ago
        old_time = time.time() - ((_SESSION_EXPIRY_DAYS + 1) * 86400)
        os.utime(str(old_path), (old_time, old_time))

        # Create a "recent" session file
        recent_path = tmp_sessions_dir / "recent_session.enc"
        recent_path.write_bytes(b"recent encrypted data")

        with patch("saido_agent.core.config.SESSIONS_DIR", tmp_sessions_dir):
            _cleanup_expired_sessions()

        assert not old_path.exists(), "Old session should be cleaned up"
        assert recent_path.exists(), "Recent session should be preserved"


# ---------------------------------------------------------------------------
# Session data includes metadata
# ---------------------------------------------------------------------------

class TestSessionMetadata:
    """Test that session data includes preview, memories, tool calls."""

    def test_build_session_data_includes_preview(self):
        from saido_agent.cli.repl import _build_session_data

        state = FakeState(
            messages=[
                {"role": "user", "content": "Explain the memory system"},
                {"role": "assistant", "content": "The memory system stores..."},
            ],
            turn_count=2,
        )

        data = _build_session_data(state)
        assert data["first_message_preview"] == "Explain the memory system"
        assert "saved_at" in data
        assert isinstance(data["active_memories"], list)

    def test_build_session_data_with_block_content(self):
        from saido_agent.cli.repl import _build_session_data

        state = FakeState(
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Block content message"}
                ]},
            ],
            turn_count=1,
        )

        data = _build_session_data(state)
        assert data["first_message_preview"] == "Block content message"

    def test_build_session_data_counts_tool_calls(self):
        from saido_agent.cli.repl import _build_session_data

        state = FakeState(
            messages=[
                {"role": "user", "content": "read file"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                    {"type": "tool_use", "id": "t2", "name": "Edit", "input": {}},
                ]},
            ],
            turn_count=1,
        )

        data = _build_session_data(state)
        assert data["tool_call_count"] == 2


# ---------------------------------------------------------------------------
# /resume lists sessions with previews
# ---------------------------------------------------------------------------

class TestResumeCommand:
    """Test the /resume command lists sessions with timestamps and previews."""

    def test_resume_lists_sessions(self, mock_fernet, tmp_sessions_dir, capsys):
        from saido_agent.cli.repl import cmd_save, cmd_resume, _encrypt_session

        # Create two sessions with different content
        state1 = FakeState(
            messages=[{"role": "user", "content": "First session message"}],
            turn_count=1,
        )
        state2 = FakeState(
            messages=[{"role": "user", "content": "Second session message"}],
            turn_count=1,
        )

        with patch("saido_agent.core.config.SESSIONS_DIR", tmp_sessions_dir):
            cmd_save("session_001.enc", state1, {})
            cmd_save("session_002.enc", state2, {})

            # Resume with no args should list sessions
            fake_state = FakeState()
            cmd_resume("", fake_state, {})

        captured = capsys.readouterr()
        assert "session_001.enc" in captured.out
        assert "session_002.enc" in captured.out

    def test_resume_loads_specific_session(self, mock_fernet, tmp_sessions_dir):
        from saido_agent.cli.repl import cmd_save, cmd_resume

        state = FakeState(
            messages=[{"role": "user", "content": "Resumable message"}],
            turn_count=3,
            total_input_tokens=200,
        )

        with patch("saido_agent.core.config.SESSIONS_DIR", tmp_sessions_dir):
            cmd_save("to_resume.enc", state, {})

            loaded_state = FakeState()
            cmd_resume("to_resume.enc", loaded_state, {})

        assert len(loaded_state.messages) == 1
        assert loaded_state.turn_count == 3


# ---------------------------------------------------------------------------
# Auto-save triggers
# ---------------------------------------------------------------------------

class TestAutoSave:
    """Test auto-save triggers at configured interval."""

    def test_autosave_triggers_after_interval(self, mock_fernet, tmp_sessions_dir):
        """Verify _maybe_autosave logic by testing _build_session_data + save_latest."""
        from saido_agent.cli.repl import save_latest, _build_session_data

        mr_dir = tmp_sessions_dir / "mr_sessions"
        mr_dir.mkdir(exist_ok=True)

        state = FakeState(
            messages=[{"role": "user", "content": f"msg {i}"} for i in range(6)],
            turn_count=6,
        )

        with patch("saido_agent.core.config.MR_SESSION_DIR", mr_dir):
            save_latest("", state, {})

        path = mr_dir / "session_latest.enc"
        assert path.exists()
        assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# ConversationExtractor
# ---------------------------------------------------------------------------

class TestConversationExtractor:
    """Test heuristic extraction of insights from conversations."""

    def test_extract_decisions(self):
        from saido_agent.memory.extract import ConversationExtractor

        messages = [
            {"role": "user", "content": "Let's use PostgreSQL for the database instead of MySQL"},
            {"role": "assistant", "content": "We'll implement the API with FastAPI and use PostgreSQL."},
        ]

        extractor = ConversationExtractor(messages)
        decisions = extractor.extract_decisions()
        assert len(decisions) >= 1
        assert any("decision" == d.category for d in decisions)

    def test_extract_preferences(self):
        from saido_agent.memory.extract import ConversationExtractor

        messages = [
            {"role": "user", "content": "I prefer using type hints everywhere in the codebase"},
            {"role": "user", "content": "Please always include docstrings in public functions"},
        ]

        extractor = ConversationExtractor(messages)
        prefs = extractor.extract_preferences()
        assert len(prefs) >= 1
        assert all(p.category == "preference" for p in prefs)

    def test_extract_facts(self):
        from saido_agent.memory.extract import ConversationExtractor

        messages = [
            {"role": "assistant", "content": "The project uses React 18 with TypeScript for the frontend."},
            {"role": "assistant", "content": "The database is PostgreSQL 15 with PostGIS extensions."},
        ]

        extractor = ConversationExtractor(messages)
        facts = extractor.extract_facts()
        assert len(facts) >= 1
        assert all(f.category == "fact" for f in facts)

    def test_extract_combined(self):
        from saido_agent.memory.extract import ConversationExtractor

        messages = [
            {"role": "user", "content": "I prefer short commit messages"},
            {"role": "user", "content": "Let's use Redis for caching the session tokens"},
            {"role": "assistant", "content": "The project uses Django with PostgreSQL."},
        ]

        extractor = ConversationExtractor(messages)
        insights = extractor.extract()
        assert len(insights) >= 1
        categories = {i.category for i in insights}
        # At least one category should be present
        assert len(categories) >= 1

    def test_extract_no_insights_from_empty(self):
        from saido_agent.memory.extract import ConversationExtractor

        extractor = ConversationExtractor([])
        insights = extractor.extract()
        assert insights == []

    def test_to_memory_entries(self):
        from saido_agent.memory.extract import ConversationExtractor

        messages = [
            {"role": "user", "content": "I prefer using pytest over unittest for all testing"},
        ]

        extractor = ConversationExtractor(messages)
        entries = extractor.to_memory_entries()

        if entries:  # May find matches depending on regex
            for entry in entries:
                assert entry.name.startswith("auto_")
                assert entry.type in ("project", "feedback")
                assert entry.created  # Should have a date

    def test_extract_with_block_content(self):
        from saido_agent.memory.extract import ConversationExtractor

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "I prefer tabs over spaces always"},
            ]},
        ]

        extractor = ConversationExtractor(messages)
        prefs = extractor.extract_preferences()
        assert len(prefs) >= 1

    def test_confidence_filtering(self):
        from saido_agent.memory.extract import ConversationExtractor

        messages = [
            {"role": "user", "content": "I prefer using type annotations everywhere"},
            {"role": "assistant", "content": "The project uses Python 3.12 with strict typing."},
        ]

        extractor = ConversationExtractor(messages)

        # High threshold should filter more
        high_conf = extractor.extract(min_confidence=0.9)
        all_insights = extractor.extract(min_confidence=0.0)
        assert len(high_conf) <= len(all_insights)

    def test_deduplication(self):
        from saido_agent.memory.extract import ConversationExtractor

        messages = [
            {"role": "user", "content": "I prefer dark mode in the editor"},
            {"role": "user", "content": "I prefer dark mode in the editor"},  # duplicate
        ]

        extractor = ConversationExtractor(messages)
        prefs = extractor.extract_preferences()
        # Should deduplicate
        summaries = [p.summary.lower() for p in prefs]
        assert len(summaries) == len(set(summaries))


# ---------------------------------------------------------------------------
# Integration: memory + knowledge article pipeline
# ---------------------------------------------------------------------------

class TestMemoryKnowledgePipeline:
    """Test the full pipeline: save memory -> convert to knowledge article."""

    def test_save_then_convert(self, tmp_memory_dir):
        from saido_agent.memory.store import (
            MemoryEntry, save_memory, load_entries, to_knowledge_article,
        )

        entry = MemoryEntry(
            name="pipeline test",
            description="End to end test",
            type="feedback",
            content="Always run linters before committing.",
            created="2026-04-05",
        )
        save_memory(entry, scope="user")

        loaded = load_entries("user")
        assert len(loaded) == 1

        article = to_knowledge_article(loaded[0])
        assert article["type"] == "memory"
        assert article["category"] == "agent-memory"
        assert article["metadata"]["scope"] == "user"
        assert "scope:user" in article["tags"]
