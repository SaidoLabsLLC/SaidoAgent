"""Tests for billing, subscriptions, Stripe integration, and usage metering.

Validates:
  - Subscription CRUD (create, read, update)
  - Tier limits enforced (free can't use cloud, pro has token limit)
  - Usage recording and aggregation
  - Grace period enforcement (24 hours after limit hit)
  - 80%/100% threshold warnings
  - Usage summary aggregation by provider/model
  - Stripe checkout session creation (mock)
  - Webhook handling for all event types
  - Billing API endpoints
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Isolate all auth and DB state to tmp_path for every test."""
    saido_dir = tmp_path / ".saido_agent"
    saido_dir.mkdir()

    # Isolate auth module paths
    import saido_agent.api.auth as auth_mod

    monkeypatch.setattr(auth_mod, "_SAIDO_DIR", saido_dir)
    monkeypatch.setattr(auth_mod, "_API_KEYS_FILE", saido_dir / "api_keys.json")
    monkeypatch.setattr(auth_mod, "_JWT_SECRET_FILE", saido_dir / "jwt_secret")
    auth_mod.reset_rate_limits()

    # Use a file-based DB scoped to this test
    db_file = str(tmp_path / "test_billing.db")

    # Patch get_connection to use our test DB in ALL modules that import it
    import saido_agent.api.db as db_mod
    import saido_agent.api.billing as billing_mod

    original_get_connection = db_mod.get_connection

    def _patched_get_connection(db_path=None):
        return original_get_connection(db_path or db_file)

    monkeypatch.setattr(db_mod, "get_connection", _patched_get_connection)
    monkeypatch.setattr(db_mod, "_DB_PATH", Path(db_file))
    monkeypatch.setattr(billing_mod, "get_connection", _patched_get_connection)

    # Also patch users module if it's imported
    try:
        import saido_agent.api.users as users_mod

        monkeypatch.setattr(users_mod, "get_connection", _patched_get_connection)
    except (ImportError, AttributeError):
        pass

    # Reset billing manager state
    billing_mod.reset_billing_manager()

    # Run migrations to set up schema
    from saido_agent.api.db import run_migrations

    run_migrations(db_path=db_file)

    yield db_file


@pytest.fixture()
def billing_manager(_isolated_env):
    """Return a BillingManager configured for the test database."""
    from saido_agent.api.billing import BillingManager

    return BillingManager(db_path=_isolated_env)


@pytest.fixture()
def client(_isolated_env):
    """Return a FastAPI TestClient with mocked agent factory."""
    from dataclasses import dataclass, field

    @dataclass
    class _MockStoreStats:
        document_count: int = 10
        category_count: int = 3
        concept_count: int = 5
        total_size_bytes: int = 1024

    mock_agent = MagicMock()
    mock_agent.stats = _MockStoreStats()

    import saido_agent.api.server as server_mod

    with patch.object(
        server_mod,
        "get_agent_for_tenant",
        return_value=mock_agent,
    ):
        from saido_agent.api.server import app

        yield TestClient(app)


@pytest.fixture()
def auth_header(_isolated_env) -> dict[str, str]:
    """Create an API key and return an auth header dict."""
    from saido_agent.api.auth import create_api_key

    key = create_api_key("test-tenant")
    return {"X-API-Key": key}


# ---------------------------------------------------------------------------
# Subscription CRUD
# ---------------------------------------------------------------------------


class TestSubscriptionCRUD:
    """Test subscription creation, reading, and updating."""

    def test_get_subscription_nonexistent(self, billing_manager):
        """get_subscription returns None for unknown tenant."""
        result = billing_manager.get_subscription("nonexistent")
        assert result is None

    def test_get_or_create_subscription_creates_free(self, billing_manager):
        """get_or_create_subscription creates a free-tier subscription."""
        sub = billing_manager.get_or_create_subscription("tenant-1")
        assert sub is not None
        assert sub["tenant_id"] == "tenant-1"
        assert sub["tier"] == "free"
        assert sub["status"] == "active"
        assert sub["cloud_tokens_used"] == 0
        assert sub["cloud_tokens_limit"] == 0

    def test_get_or_create_subscription_idempotent(self, billing_manager):
        """Calling get_or_create twice returns the same subscription."""
        sub1 = billing_manager.get_or_create_subscription("tenant-1")
        sub2 = billing_manager.get_or_create_subscription("tenant-1")
        assert sub1["id"] == sub2["id"]

    def test_update_subscription_tier(self, billing_manager):
        """Updating tier also updates cloud_tokens_limit."""
        billing_manager.get_or_create_subscription("tenant-1")
        updated = billing_manager.update_subscription("tenant-1", tier="pro")
        assert updated["tier"] == "pro"
        assert updated["cloud_tokens_limit"] == 100_000

    def test_update_subscription_status(self, billing_manager):
        """Can update subscription status."""
        billing_manager.get_or_create_subscription("tenant-1")
        updated = billing_manager.update_subscription(
            "tenant-1", status="past_due"
        )
        assert updated["status"] == "past_due"

    def test_update_subscription_stripe_ids(self, billing_manager):
        """Can set Stripe customer and subscription IDs."""
        billing_manager.get_or_create_subscription("tenant-1")
        updated = billing_manager.update_subscription(
            "tenant-1",
            stripe_customer_id="cus_123",
            stripe_subscription_id="sub_456",
        )
        assert updated["stripe_customer_id"] == "cus_123"
        assert updated["stripe_subscription_id"] == "sub_456"


# ---------------------------------------------------------------------------
# Tier limit enforcement
# ---------------------------------------------------------------------------


class TestTierLimits:
    """Test tier limit checking and enforcement."""

    def test_free_tier_blocks_cloud_tokens(self, billing_manager):
        """Free tier cannot use cloud tokens."""
        billing_manager.get_or_create_subscription("free-tenant")
        allowed, msg = billing_manager.check_tier_limit(
            "free-tenant", "cloud_tokens"
        )
        assert allowed is False
        assert "Pro plan" in msg

    def test_pro_tier_allows_cloud_tokens(self, billing_manager):
        """Pro tier can use cloud tokens within limits."""
        billing_manager.get_or_create_subscription("pro-tenant")
        billing_manager.update_subscription("pro-tenant", tier="pro")
        allowed, msg = billing_manager.check_tier_limit(
            "pro-tenant", "cloud_tokens"
        )
        assert allowed is True
        assert msg == ""

    def test_enterprise_tier_unlimited(self, billing_manager):
        """Enterprise tier has unlimited cloud tokens."""
        billing_manager.get_or_create_subscription("ent-tenant")
        billing_manager.update_subscription("ent-tenant", tier="enterprise")
        allowed, msg = billing_manager.check_tier_limit(
            "ent-tenant", "cloud_tokens"
        )
        assert allowed is True

    def test_80_percent_warning(self, billing_manager):
        """80% usage triggers a warning but still allows access."""
        billing_manager.get_or_create_subscription("pro-tenant")
        billing_manager.update_subscription("pro-tenant", tier="pro")
        # Set usage to 80% of 100k = 80,000
        billing_manager.update_subscription(
            "pro-tenant", cloud_tokens_used=80_000
        )
        allowed, msg = billing_manager.check_tier_limit(
            "pro-tenant", "cloud_tokens"
        )
        assert allowed is True
        assert "Warning" in msg
        assert "80%" in msg

    def test_100_percent_triggers_grace(self, billing_manager):
        """100% usage triggers grace period on first hit."""
        billing_manager.get_or_create_subscription("pro-tenant")
        billing_manager.update_subscription("pro-tenant", tier="pro")
        billing_manager.update_subscription(
            "pro-tenant", cloud_tokens_used=100_000
        )
        allowed, msg = billing_manager.check_tier_limit(
            "pro-tenant", "cloud_tokens"
        )
        # First hit: grace period starts, still allowed
        assert allowed is True
        assert "GRACE PERIOD" in msg
        assert "24 hours" in msg

    def test_free_tier_article_limit(self, billing_manager):
        """Free tier has article limit (not enforced in placeholder)."""
        billing_manager.get_or_create_subscription("free-tenant")
        allowed, msg = billing_manager.check_tier_limit(
            "free-tenant", "articles"
        )
        # Currently returns True since _count_articles returns 0
        assert allowed is True

    def test_team_tier_limits(self, billing_manager):
        """Team tier has higher cloud token limit."""
        billing_manager.get_or_create_subscription("team-tenant")
        billing_manager.update_subscription("team-tenant", tier="team")
        sub = billing_manager.get_subscription("team-tenant")
        assert sub["cloud_tokens_limit"] == 500_000


# ---------------------------------------------------------------------------
# Grace period enforcement
# ---------------------------------------------------------------------------


class TestGracePeriod:
    """Test the 24-hour grace period after hitting token limits."""

    def test_grace_period_allows_first_hit(self, billing_manager):
        """First time hitting limit starts grace period."""
        billing_manager.get_or_create_subscription("grace-tenant")
        billing_manager.update_subscription("grace-tenant", tier="pro")
        billing_manager.update_subscription(
            "grace-tenant", cloud_tokens_used=100_000
        )

        allowed, msg = billing_manager.check_tier_limit(
            "grace-tenant", "cloud_tokens"
        )
        assert allowed is True
        assert "GRACE PERIOD" in msg

    def test_grace_period_allows_within_24h(self, billing_manager, monkeypatch):
        """Access is allowed during the 24-hour grace period."""
        import saido_agent.api.billing as billing_mod

        billing_manager.get_or_create_subscription("grace-tenant")
        billing_manager.update_subscription("grace-tenant", tier="pro")
        billing_manager.update_subscription(
            "grace-tenant", cloud_tokens_used=100_000
        )

        # First check: starts grace period
        billing_manager.check_tier_limit("grace-tenant", "cloud_tokens")

        # Second check still within grace
        allowed, msg = billing_manager.check_tier_limit(
            "grace-tenant", "cloud_tokens"
        )
        assert allowed is True
        assert "GRACE PERIOD" in msg

    def test_grace_period_blocks_after_24h(self, billing_manager, monkeypatch):
        """Access is blocked after the 24-hour grace period expires."""
        import saido_agent.api.billing as billing_mod

        billing_manager.get_or_create_subscription("grace-tenant")
        billing_manager.update_subscription("grace-tenant", tier="pro")
        billing_manager.update_subscription(
            "grace-tenant", cloud_tokens_used=100_000
        )

        # First check: starts grace period
        billing_manager.check_tier_limit("grace-tenant", "cloud_tokens")

        # Simulate 25 hours passing by backdating the timestamp
        billing_mod._limit_hit_timestamps["grace-tenant"] = (
            time.time() - 25 * 3600
        )

        allowed, msg = billing_manager.check_tier_limit(
            "grace-tenant", "cloud_tokens"
        )
        assert allowed is False
        assert "Token limit reached" in msg

    def test_grace_period_reset_on_new_period(self, billing_manager):
        """Resetting the billing period clears the grace period."""
        import saido_agent.api.billing as billing_mod

        billing_manager.get_or_create_subscription("grace-tenant")
        billing_manager.update_subscription("grace-tenant", tier="pro")
        billing_manager.update_subscription(
            "grace-tenant", cloud_tokens_used=100_000
        )

        # Start grace period
        billing_manager.check_tier_limit("grace-tenant", "cloud_tokens")
        assert "grace-tenant" in billing_mod._limit_hit_timestamps

        # Reset period
        billing_manager.reset_period_usage("grace-tenant")
        assert "grace-tenant" not in billing_mod._limit_hit_timestamps


# ---------------------------------------------------------------------------
# Usage recording and metering
# ---------------------------------------------------------------------------


class TestUsageMetering:
    """Test usage recording and summary aggregation."""

    def test_record_usage(self, billing_manager):
        """Recording usage increments cloud_tokens_used."""
        billing_manager.get_or_create_subscription("meter-tenant")
        billing_manager.update_subscription("meter-tenant", tier="pro")

        billing_manager.record_usage("meter-tenant", 1000, "openai", "gpt-4")
        billing_manager.record_usage("meter-tenant", 500, "anthropic", "claude-3")

        sub = billing_manager.get_subscription("meter-tenant")
        assert sub["cloud_tokens_used"] == 1500

    def test_usage_summary_aggregation(self, billing_manager):
        """get_usage_summary aggregates by provider/model."""
        billing_manager.get_or_create_subscription("meter-tenant")
        billing_manager.update_subscription("meter-tenant", tier="pro")

        billing_manager.record_usage("meter-tenant", 1000, "openai", "gpt-4")
        billing_manager.record_usage("meter-tenant", 2000, "openai", "gpt-4")
        billing_manager.record_usage("meter-tenant", 500, "anthropic", "claude-3")

        summary = billing_manager.get_usage_summary("meter-tenant")
        assert summary["tenant_id"] == "meter-tenant"
        assert summary["tier"] == "pro"
        assert summary["total_tokens_this_period"] == 3500
        assert summary["request_count"] == 3
        assert len(summary["breakdown"]) == 2

        # Find the openai breakdown entry
        openai_entry = next(
            (b for b in summary["breakdown"] if b["provider"] == "openai"), None
        )
        assert openai_entry is not None
        assert openai_entry["tokens"] == 3000
        assert openai_entry["requests"] == 2

    def test_usage_summary_includes_limits(self, billing_manager):
        """Summary includes tier limit information."""
        billing_manager.get_or_create_subscription("meter-tenant")
        billing_manager.update_subscription("meter-tenant", tier="team")

        summary = billing_manager.get_usage_summary("meter-tenant")
        assert summary["cloud_tokens_limit"] == 500_000

    def test_reset_period_usage(self, billing_manager):
        """Resetting period clears cloud_tokens_used counter."""
        billing_manager.get_or_create_subscription("meter-tenant")
        billing_manager.update_subscription("meter-tenant", tier="pro")
        billing_manager.record_usage("meter-tenant", 5000, "openai", "gpt-4")

        sub = billing_manager.get_subscription("meter-tenant")
        assert sub["cloud_tokens_used"] == 5000

        billing_manager.reset_period_usage("meter-tenant")
        sub = billing_manager.get_subscription("meter-tenant")
        assert sub["cloud_tokens_used"] == 0


# ---------------------------------------------------------------------------
# Stripe checkout (mock)
# ---------------------------------------------------------------------------


class TestCheckoutSession:
    """Test Stripe checkout session creation."""

    def test_create_mock_checkout(self, billing_manager):
        """Creates a mock checkout session when Stripe is not installed."""
        result = billing_manager.create_checkout_session("tenant-1", "pro")
        assert "url" in result
        assert "session_id" in result
        assert result["session_id"].startswith("cs_mock_")

    def test_create_checkout_invalid_tier(self, billing_manager):
        """Raises ValueError for invalid tier."""
        with pytest.raises(ValueError, match="Invalid tier"):
            billing_manager.create_checkout_session("tenant-1", "invalid")

    def test_create_checkout_for_each_tier(self, billing_manager):
        """Can create checkout for pro, team, and enterprise tiers."""
        for tier in ("pro", "team", "enterprise"):
            result = billing_manager.create_checkout_session("tenant-1", tier)
            assert result["tier"] == tier
            assert "url" in result

    def test_create_checkout_ensures_subscription(self, billing_manager):
        """Checkout creation auto-creates subscription if needed."""
        # No subscription exists yet
        assert billing_manager.get_subscription("new-tenant") is None

        billing_manager.create_checkout_session("new-tenant", "pro")

        # Now subscription should exist
        sub = billing_manager.get_subscription("new-tenant")
        assert sub is not None
        assert sub["tier"] == "free"  # Still free until checkout completes


# ---------------------------------------------------------------------------
# Webhook handling
# ---------------------------------------------------------------------------


class TestWebhookHandling:
    """Test Stripe webhook event processing."""

    def test_checkout_completed(self, billing_manager):
        """checkout.session.completed upgrades the subscription."""
        billing_manager.get_or_create_subscription("webhook-tenant")

        event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "customer": "cus_abc123",
                    "subscription": "sub_xyz789",
                    "metadata": {
                        "tenant_id": "webhook-tenant",
                        "tier": "pro",
                    },
                }
            },
        }

        result = billing_manager.handle_webhook(
            json.dumps(event).encode(), ""
        )
        assert result["status"] == "ok"
        assert result["tier"] == "pro"

        sub = billing_manager.get_subscription("webhook-tenant")
        assert sub["tier"] == "pro"
        assert sub["stripe_customer_id"] == "cus_abc123"
        assert sub["stripe_subscription_id"] == "sub_xyz789"

    def test_subscription_updated(self, billing_manager):
        """customer.subscription.updated changes subscription status."""
        billing_manager.get_or_create_subscription("webhook-tenant")

        event = {
            "type": "customer.subscription.updated",
            "data": {
                "object": {
                    "status": "past_due",
                    "metadata": {"tenant_id": "webhook-tenant"},
                }
            },
        }

        result = billing_manager.handle_webhook(
            json.dumps(event).encode(), ""
        )
        assert result["status"] == "ok"
        assert result["new_status"] == "past_due"

    def test_subscription_deleted(self, billing_manager):
        """customer.subscription.deleted downgrades to free."""
        billing_manager.get_or_create_subscription("webhook-tenant")
        billing_manager.update_subscription("webhook-tenant", tier="pro")

        event = {
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "metadata": {"tenant_id": "webhook-tenant"},
                }
            },
        }

        result = billing_manager.handle_webhook(
            json.dumps(event).encode(), ""
        )
        assert result["status"] == "ok"
        assert result["tier"] == "free"

        sub = billing_manager.get_subscription("webhook-tenant")
        assert sub["tier"] == "free"
        assert sub["status"] == "cancelled"

    def test_payment_failed(self, billing_manager):
        """invoice.payment_failed marks subscription as past_due."""
        billing_manager.get_or_create_subscription("webhook-tenant")
        billing_manager.update_subscription(
            "webhook-tenant", stripe_subscription_id="sub_fail_123"
        )

        event = {
            "type": "invoice.payment_failed",
            "data": {
                "object": {
                    "subscription": "sub_fail_123",
                }
            },
        }

        result = billing_manager.handle_webhook(
            json.dumps(event).encode(), ""
        )
        assert result["status"] == "ok"
        assert result["new_status"] == "past_due"

    def test_unhandled_event_ignored(self, billing_manager):
        """Unknown event types are ignored gracefully."""
        event = {"type": "unknown.event.type", "data": {}}
        result = billing_manager.handle_webhook(
            json.dumps(event).encode(), ""
        )
        assert result["status"] == "ignored"

    def test_invalid_payload_raises(self, billing_manager):
        """Invalid JSON payload raises ValueError."""
        with pytest.raises(ValueError, match="Invalid webhook payload"):
            billing_manager.handle_webhook(b"not json", "")

    def test_checkout_completed_no_tenant_id(self, billing_manager):
        """checkout.session.completed without tenant_id returns error."""
        event = {
            "type": "checkout.session.completed",
            "data": {"object": {"metadata": {}}},
        }
        result = billing_manager.handle_webhook(
            json.dumps(event).encode(), ""
        )
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestBillingEndpoints:
    """Test the billing API endpoints."""

    def test_get_subscription_endpoint(self, client, auth_header):
        """GET /v1/billing/subscription returns subscription data."""
        resp = client.get("/v1/billing/subscription", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "free"
        assert data["status"] == "active"

    def test_get_usage_endpoint(self, client, auth_header):
        """GET /v1/billing/usage returns usage summary."""
        resp = client.get("/v1/billing/usage", headers=auth_header)
        assert resp.status_code == 200
        data = resp.json()
        assert "tier" in data
        assert "total_tokens_this_period" in data
        assert "breakdown" in data

    def test_create_checkout_endpoint(self, client, auth_header):
        """POST /v1/billing/checkout creates a checkout session."""
        resp = client.post(
            "/v1/billing/checkout",
            headers=auth_header,
            json={"tier": "pro"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "url" in data
        assert "session_id" in data

    def test_create_checkout_invalid_tier(self, client, auth_header):
        """POST /v1/billing/checkout with invalid tier returns 400."""
        resp = client.post(
            "/v1/billing/checkout",
            headers=auth_header,
            json={"tier": "invalid"},
        )
        assert resp.status_code == 400

    def test_webhook_endpoint(self, client):
        """POST /v1/webhooks/stripe processes webhook events."""
        event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "customer": "cus_test",
                    "subscription": "sub_test",
                    "metadata": {
                        "tenant_id": "webhook-test-tenant",
                        "tier": "pro",
                    },
                }
            },
        }

        # Webhook endpoint should NOT require auth
        resp = client.post(
            "/v1/webhooks/stripe",
            content=json.dumps(event).encode(),
            headers={"Stripe-Signature": "mock_sig", "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_webhook_no_auth_required(self, client):
        """Webhook endpoint works without any auth header."""
        event = {"type": "unknown.event", "data": {}}
        resp = client.post(
            "/v1/webhooks/stripe",
            content=json.dumps(event).encode(),
            headers={"Content-Type": "application/json"},
        )
        # Should not return 401
        assert resp.status_code == 200

    def test_billing_endpoints_require_auth(self, client):
        """Billing subscription and usage endpoints require auth."""
        resp = client.get("/v1/billing/subscription")
        assert resp.status_code == 401

        resp = client.get("/v1/billing/usage")
        assert resp.status_code == 401

        resp = client.post("/v1/billing/checkout", json={"tier": "pro"})
        assert resp.status_code == 401
