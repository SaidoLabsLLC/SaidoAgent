"""Tests for saido_agent.knowledge.finetune — fine-tuning pipeline integration."""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from saido_agent.knowledge.finetune import (
    FinetuneJob,
    FinetuneManager,
    FinetuneResult,
    FinetuneStatus,
    _validate_jsonl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_config(tmp_path):
    """Provide a temporary config directory."""
    return tmp_path / "config"


@pytest.fixture()
def valid_jsonl(tmp_path):
    """Create a valid JSONL training file."""
    path = tmp_path / "train.jsonl"
    lines = [
        json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}),
        json.dumps({"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "goodbye"}]}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def invalid_jsonl_no_messages(tmp_path):
    """JSONL file missing the 'messages' field."""
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def not_jsonl(tmp_path):
    """A file with the wrong extension."""
    path = tmp_path / "train.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    return path


@pytest.fixture()
def mock_openai_client():
    """Build a mock OpenAI client that simulates file upload + fine-tune creation."""
    client = MagicMock()
    # files.create returns an object with .id
    client.files.create.return_value = SimpleNamespace(id="file-abc123")
    # fine_tuning.jobs.create returns an object with .id
    client.fine_tuning.jobs.create.return_value = SimpleNamespace(id="ftjob-xyz789")
    return client


@pytest.fixture()
def manager(tmp_config, mock_openai_client):
    """FinetuneManager wired to a temp dir and mock OpenAI client."""
    return FinetuneManager(
        config_dir=tmp_config,
        openai_client=mock_openai_client,
    )


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_jsonl_passes(self, valid_jsonl):
        _validate_jsonl(valid_jsonl)  # should not raise

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _validate_jsonl(tmp_path / "nope.jsonl")

    def test_wrong_extension_raises(self, not_jsonl):
        with pytest.raises(ValueError, match="must be .jsonl"):
            _validate_jsonl(not_jsonl)

    def test_missing_messages_field_raises(self, invalid_jsonl_no_messages):
        with pytest.raises(ValueError, match="missing required 'messages'"):
            _validate_jsonl(invalid_jsonl_no_messages)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            _validate_jsonl(path)

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text("{bad json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            _validate_jsonl(path)


# ---------------------------------------------------------------------------
# Job persistence
# ---------------------------------------------------------------------------

class TestJobPersistence:
    def test_save_and_load_roundtrip(self, tmp_config):
        mgr = FinetuneManager(config_dir=tmp_config)
        job = FinetuneJob(
            id="ft-test1",
            provider="openai",
            model_base="gpt-4o-mini",
            training_file="/tmp/train.jsonl",
            status=FinetuneStatus.TRAINING,
            created_at="2026-01-01T00:00:00+00:00",
        )
        mgr._jobs = [job]
        mgr._save_jobs()

        # Load into a fresh manager
        mgr2 = FinetuneManager(config_dir=tmp_config)
        assert len(mgr2._jobs) == 1
        loaded = mgr2._jobs[0]
        assert loaded.id == "ft-test1"
        assert loaded.status == FinetuneStatus.TRAINING
        assert loaded.provider == "openai"

    def test_empty_persistence(self, tmp_config):
        mgr = FinetuneManager(config_dir=tmp_config)
        assert mgr._jobs == []

    def test_corrupt_json_recovers(self, tmp_config):
        tmp_config.mkdir(parents=True, exist_ok=True)
        (tmp_config / "finetune_jobs.json").write_text("NOT JSON", encoding="utf-8")
        mgr = FinetuneManager(config_dir=tmp_config)
        assert mgr._jobs == []


# ---------------------------------------------------------------------------
# OpenAI flow (mocked)
# ---------------------------------------------------------------------------

class TestOpenAIFlow:
    def test_upload_and_start(self, manager, valid_jsonl, mock_openai_client):
        result = manager.start_openai(str(valid_jsonl))

        assert isinstance(result, FinetuneResult)
        assert result.job.status == FinetuneStatus.TRAINING
        assert "ftjob-xyz789" in result.message
        mock_openai_client.files.create.assert_called_once()
        mock_openai_client.fine_tuning.jobs.create.assert_called_once()

    def test_default_model_and_epochs(self, manager, valid_jsonl, mock_openai_client):
        manager.start_openai(str(valid_jsonl))
        call_kwargs = mock_openai_client.fine_tuning.jobs.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini-2024-07-18"
        assert call_kwargs.kwargs["hyperparameters"]["n_epochs"] == 3

    def test_custom_suffix(self, manager, valid_jsonl, mock_openai_client):
        manager.start_openai(str(valid_jsonl), suffix="saido-v1")
        call_kwargs = mock_openai_client.fine_tuning.jobs.create.call_args
        assert call_kwargs.kwargs["suffix"] == "saido-v1"

    def test_api_failure_marks_failed(self, tmp_config, valid_jsonl):
        bad_client = MagicMock()
        bad_client.files.create.side_effect = RuntimeError("API down")
        mgr = FinetuneManager(config_dir=tmp_config, openai_client=bad_client)

        result = mgr.start_openai(str(valid_jsonl))
        assert result.job.status == FinetuneStatus.FAILED
        assert "API down" in result.job.error

    def test_no_client_fails_gracefully(self, tmp_config, valid_jsonl):
        mgr = FinetuneManager(config_dir=tmp_config, openai_client=None)
        with patch("saido_agent.knowledge.finetune._get_openai_client", return_value=None):
            result = mgr.start_openai(str(valid_jsonl))
        assert result.job.status == FinetuneStatus.FAILED
        assert "not available" in result.job.error.lower() or "not configured" in result.job.error.lower()

    def test_status_check_polls_openai(self, manager, valid_jsonl, mock_openai_client):
        # Start a job
        result = manager.start_openai(str(valid_jsonl))
        job_id = result.job.id

        # Simulate API returning "succeeded"
        mock_openai_client.fine_tuning.jobs.retrieve.return_value = SimpleNamespace(
            status="succeeded",
            fine_tuned_model="ft:gpt-4o-mini:saido:abc123",
        )
        updated = manager.check_status(job_id)
        assert updated.status == FinetuneStatus.COMPLETED
        assert updated.fine_tuned_model == "ft:gpt-4o-mini:saido:abc123"

    def test_status_check_failed(self, manager, valid_jsonl, mock_openai_client):
        result = manager.start_openai(str(valid_jsonl))
        job_id = result.job.id

        mock_openai_client.fine_tuning.jobs.retrieve.return_value = SimpleNamespace(
            status="failed",
            fine_tuned_model=None,
            error="Data quality issue",
        )
        updated = manager.check_status(job_id)
        assert updated.status == FinetuneStatus.FAILED

    def test_status_check_unknown_job_raises(self, manager):
        with pytest.raises(ValueError, match="Job not found"):
            manager.check_status("ft-nonexistent")


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------

class TestStatusTransitions:
    def test_pending_to_training_to_completed(self, manager, valid_jsonl, mock_openai_client):
        result = manager.start_openai(str(valid_jsonl))
        job_id = result.job.id
        # After start_openai the status should be TRAINING
        assert result.job.status == FinetuneStatus.TRAINING

        # Simulate completion
        mock_openai_client.fine_tuning.jobs.retrieve.return_value = SimpleNamespace(
            status="succeeded",
            fine_tuned_model="ft:gpt-4o-mini:saido:done",
        )
        final = manager.check_status(job_id)
        assert final.status == FinetuneStatus.COMPLETED

    def test_completed_job_not_repolled(self, manager, valid_jsonl, mock_openai_client):
        result = manager.start_openai(str(valid_jsonl))
        job_id = result.job.id
        # Force completed
        result.job.status = FinetuneStatus.COMPLETED
        result.job.fine_tuned_model = "ft:done"
        manager._save_jobs()

        # check_status should return immediately without polling
        updated = manager.check_status(job_id)
        assert updated.status == FinetuneStatus.COMPLETED
        mock_openai_client.fine_tuning.jobs.retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# Local / Axolotl config generation
# ---------------------------------------------------------------------------

class TestLocalFinetune:
    def test_generates_axolotl_config(self, manager, valid_jsonl, tmp_config):
        result = manager.start_local(str(valid_jsonl))

        assert result.job.provider == "local"
        assert result.job.status == FinetuneStatus.PENDING
        config_path = Path(result.job.metadata["config_path"])
        assert config_path.exists()

        # Read the config and verify structure
        content = config_path.read_text(encoding="utf-8")
        # Could be YAML or JSON depending on pyyaml availability
        if config_path.suffix == ".yml":
            import yaml
            cfg = yaml.safe_load(content)
        else:
            cfg = json.loads(content)

        assert cfg["base_model"] == "mistralai/Mistral-7B-v0.1"
        assert cfg["adapter"] == "lora"
        assert cfg["lora_r"] == 16
        assert cfg["lora_alpha"] == 32
        assert cfg["num_epochs"] == 3
        assert len(cfg["datasets"]) == 1

    def test_custom_config_params(self, manager, valid_jsonl):
        result = manager.start_local(str(valid_jsonl), config={
            "base_model": "meta-llama/Llama-2-13b",
            "lora_r": 32,
            "num_epochs": 5,
        })
        cfg = result.job.metadata["axolotl_config"]
        assert cfg["base_model"] == "meta-llama/Llama-2-13b"
        assert cfg["lora_r"] == 32
        assert cfg["num_epochs"] == 5

    def test_local_status_check_pending(self, manager, valid_jsonl):
        result = manager.start_local(str(valid_jsonl))
        updated = manager.check_status(result.job.id)
        # No output artifacts yet, should remain pending
        assert updated.status == FinetuneStatus.PENDING

    def test_local_status_check_completed(self, manager, valid_jsonl):
        result = manager.start_local(str(valid_jsonl))
        # Simulate training completion by placing a marker file
        output_dir = Path(result.job.metadata["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_model.safetensors").write_text("fake", encoding="utf-8")

        updated = manager.check_status(result.job.id)
        assert updated.status == FinetuneStatus.COMPLETED
        assert updated.fine_tuned_model == str(output_dir)


# ---------------------------------------------------------------------------
# Job listing
# ---------------------------------------------------------------------------

class TestListJobs:
    def test_list_empty(self, manager):
        assert manager.list_jobs() == []

    def test_list_after_creation(self, manager, valid_jsonl):
        manager.start_openai(str(valid_jsonl))
        manager.start_local(str(valid_jsonl))
        jobs = manager.list_jobs()
        assert len(jobs) == 2
        providers = {j.provider for j in jobs}
        assert providers == {"openai", "local"}


# ---------------------------------------------------------------------------
# Model deployment
# ---------------------------------------------------------------------------

class TestDeploy:
    def test_deploy_registers_model(self, tmp_config, valid_jsonl, mock_openai_client):
        mgr = FinetuneManager(config_dir=tmp_config, openai_client=mock_openai_client)
        result = mgr.start_openai(str(valid_jsonl))
        job = result.job
        # Simulate completion
        job.status = FinetuneStatus.COMPLETED
        job.fine_tuned_model = "ft:gpt-4o-mini:saido:abc"
        mgr._save_jobs()

        model_id = mgr.deploy(job.id)
        assert model_id == "ft:gpt-4o-mini:saido:abc"

        # Verify models.json
        models = json.loads((tmp_config / "models.json").read_text(encoding="utf-8"))
        assert "ft:gpt-4o-mini:saido:abc" in models
        entry = models["ft:gpt-4o-mini:saido:abc"]
        assert entry["base_model"] == "gpt-4o-mini-2024-07-18"
        assert entry["job_id"] == job.id

    def test_deploy_updates_routing(self, tmp_config, valid_jsonl, mock_openai_client):
        # Set up a mock router
        mock_router = MagicMock()
        mock_router.routing_config = {"routing": {}}

        mgr = FinetuneManager(
            config_dir=tmp_config,
            model_router=mock_router,
            openai_client=mock_openai_client,
        )
        result = mgr.start_openai(str(valid_jsonl))
        job = result.job
        job.status = FinetuneStatus.COMPLETED
        job.fine_tuned_model = "ft:gpt-4o-mini:saido:abc"
        mgr._save_jobs()

        mgr.deploy(job.id)

        # Routing config should have a 'finetune' entry
        routing = mock_router.routing_config["routing"]
        assert "finetune" in routing
        assert routing["finetune"]["model"] == "ft:gpt-4o-mini:saido:abc"
        mock_router._save_routing_config.assert_called_once()

    def test_deploy_with_domain_metadata(self, tmp_config, valid_jsonl, mock_openai_client):
        mock_router = MagicMock()
        mock_router.routing_config = {"routing": {}}

        mgr = FinetuneManager(
            config_dir=tmp_config,
            model_router=mock_router,
            openai_client=mock_openai_client,
        )
        result = mgr.start_openai(str(valid_jsonl))
        job = result.job
        job.status = FinetuneStatus.COMPLETED
        job.fine_tuned_model = "ft:custom"
        job.metadata["domain"] = "code_gen"
        mgr._save_jobs()

        mgr.deploy(job.id)
        routing = mock_router.routing_config["routing"]
        assert "code_gen" in routing
        assert routing["code_gen"]["model"] == "ft:custom"

    def test_deploy_incomplete_raises(self, manager, valid_jsonl):
        result = manager.start_openai(str(valid_jsonl))
        with pytest.raises(ValueError, match="status is training"):
            manager.deploy(result.job.id)

    def test_deploy_unknown_job_raises(self, manager):
        with pytest.raises(ValueError, match="Job not found"):
            manager.deploy("ft-nonexistent")

    def test_deploy_no_model_raises(self, tmp_config, valid_jsonl, mock_openai_client):
        mgr = FinetuneManager(config_dir=tmp_config, openai_client=mock_openai_client)
        result = mgr.start_openai(str(valid_jsonl))
        result.job.status = FinetuneStatus.COMPLETED
        result.job.fine_tuned_model = None
        mgr._save_jobs()

        with pytest.raises(ValueError, match="no fine_tuned_model"):
            mgr.deploy(result.job.id)


# ---------------------------------------------------------------------------
# A/B comparison
# ---------------------------------------------------------------------------

class TestCompare:
    def test_compare_runs_both_models(self, tmp_config):
        mock_client = MagicMock()
        # Return different responses for different models
        def fake_create(**kwargs):
            model = kwargs["model"]
            prompt = kwargs["messages"][0]["content"]
            content = f"Response from {model} for: {prompt}"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )
        mock_client.chat.completions.create.side_effect = fake_create

        mgr = FinetuneManager(config_dir=tmp_config, openai_client=mock_client)
        questions = ["What is Python?", "Explain REST APIs."]
        report = mgr.compare("model-a", "model-b", questions)

        assert report["model_a"] == "model-a"
        assert report["model_b"] == "model-b"
        assert report["num_questions"] == 2
        assert len(report["comparisons"]) == 2

        # Verify responses contain model names
        for comp in report["comparisons"]:
            assert "model-a" in comp["model_a_response"]
            assert "model-b" in comp["model_b_response"]

    def test_compare_empty_questions_raises(self, tmp_config):
        mgr = FinetuneManager(config_dir=tmp_config)
        with pytest.raises(ValueError, match="non-empty"):
            mgr.compare("a", "b", [])

    def test_compare_summary_metrics(self, tmp_config):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="This is a detailed response with multiple sentences. It covers the topic well. Here is more detail."
                )
            )]
        )
        mgr = FinetuneManager(config_dir=tmp_config, openai_client=mock_client)
        report = mgr.compare("a", "b", ["test question"])

        summary = report["summary"]
        assert "avg_length_a" in summary
        assert "avg_length_b" in summary
        assert "avg_coherence_a" in summary
        assert "avg_coherence_b" in summary
        assert summary["avg_coherence_a"] > 0.0

    def test_compare_without_client(self, tmp_config):
        """Without a client, responses are placeholder strings."""
        mgr = FinetuneManager(config_dir=tmp_config, openai_client=None)
        with patch("saido_agent.knowledge.finetune._get_openai_client", return_value=None):
            report = mgr.compare("a", "b", ["question"])
        assert len(report["comparisons"]) == 1
        # Should get unavailable placeholder responses
        assert "[unavailable]" in report["comparisons"][0]["model_a_response"]


# ---------------------------------------------------------------------------
# Failed job handling
# ---------------------------------------------------------------------------

class TestFailedJobs:
    def test_failed_job_persists_error(self, tmp_config, valid_jsonl):
        bad_client = MagicMock()
        bad_client.files.create.side_effect = RuntimeError("quota exceeded")
        mgr = FinetuneManager(config_dir=tmp_config, openai_client=bad_client)

        result = mgr.start_openai(str(valid_jsonl))
        assert result.job.status == FinetuneStatus.FAILED
        assert "quota exceeded" in result.job.error

        # Reload and verify persistence
        mgr2 = FinetuneManager(config_dir=tmp_config)
        loaded = mgr2._jobs[0]
        assert loaded.status == FinetuneStatus.FAILED
        assert "quota exceeded" in loaded.error

    def test_failed_job_not_deployable(self, tmp_config, valid_jsonl):
        bad_client = MagicMock()
        bad_client.files.create.side_effect = RuntimeError("boom")
        mgr = FinetuneManager(config_dir=tmp_config, openai_client=bad_client)

        result = mgr.start_openai(str(valid_jsonl))
        with pytest.raises(ValueError, match="status is failed"):
            mgr.deploy(result.job.id)


# ---------------------------------------------------------------------------
# FinetuneJob serialization
# ---------------------------------------------------------------------------

class TestJobSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        job = FinetuneJob(
            id="ft-123",
            provider="openai",
            model_base="gpt-4o-mini",
            training_file="/data/train.jsonl",
            status=FinetuneStatus.COMPLETED,
            fine_tuned_model="ft:gpt-4o-mini:org:suffix",
            created_at="2026-01-01T00:00:00+00:00",
            error=None,
            metadata={"n_epochs": 3},
        )
        d = job.to_dict()
        restored = FinetuneJob.from_dict(d)

        assert restored.id == job.id
        assert restored.status == job.status
        assert restored.fine_tuned_model == job.fine_tuned_model
        assert restored.metadata == job.metadata
