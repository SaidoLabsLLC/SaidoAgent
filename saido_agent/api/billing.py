"""Billing management for Saido Agent.

Provides:
  - Subscription tier management (free, pro, team, enterprise)
  - Token usage metering and limit enforcement
  - Stripe Checkout and webhook integration (mock if stripe not installed)
  - Grace period enforcement (24 hours after limit hit)
  - Tier limit checking with 80%/100% thresholds

All database operations use the SQLite database from ``db.py``.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta, timezone


def _utcnow_str() -> str:
    """Return current UTC time in SQLite-compatible format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
from enum import Enum
from typing import Any, Optional

from saido_agent.api.db import get_connection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import stripe; fall back to mock if not installed
# ---------------------------------------------------------------------------

try:
    import stripe as _stripe_lib

    # Stripe is installed but only "available" if a real API key is configured
    _STRIPE_AVAILABLE = bool(os.environ.get("STRIPE_SECRET_KEY", ""))
except ImportError:
    _stripe_lib = None  # type: ignore[assignment]
    _STRIPE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------


class Tier(str, Enum):
    """Subscription tier levels."""

    FREE = "free"
    PRO = "pro"
    TEAM = "team"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, Enum):
    """Subscription lifecycle states."""

    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    TRIALING = "trialing"


# Limits per tier.  -1 means unlimited.
TIER_LIMITS: dict[Tier, dict[str, int]] = {
    Tier.FREE: {"cloud_tokens": 0, "articles": 100, "users": 1},
    Tier.PRO: {"cloud_tokens": 100_000, "articles": -1, "users": 1},
    Tier.TEAM: {"cloud_tokens": 500_000, "articles": -1, "users": -1},
    Tier.ENTERPRISE: {"cloud_tokens": -1, "articles": -1, "users": -1},
}

# Stripe price IDs (configure via environment variables)
_STRIPE_PRICE_MAP: dict[str, str] = {
    "pro": os.environ.get("STRIPE_PRICE_PRO", "price_pro_placeholder"),
    "team": os.environ.get("STRIPE_PRICE_TEAM", "price_team_placeholder"),
    "enterprise": os.environ.get(
        "STRIPE_PRICE_ENTERPRISE", "price_enterprise_placeholder"
    ),
}

_STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# Grace period: 24 hours after hitting token limit
_GRACE_PERIOD_SECONDS = 24 * 60 * 60

# In-memory cache of when a tenant first hit their limit
# { tenant_id: unix_timestamp_of_first_hit }
_limit_hit_timestamps: dict[str, float] = {}


# ---------------------------------------------------------------------------
# BillingManager
# ---------------------------------------------------------------------------


class BillingManager:
    """Manages subscriptions, usage metering, and Stripe integration."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path

    def _conn(self):
        """Get a database connection."""
        return get_connection(self._db_path)

    # -- Subscription CRUD -------------------------------------------------

    def get_subscription(self, tenant_id: str) -> dict | None:
        """Return the subscription record for a tenant, or None."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM subscriptions WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()

    def get_or_create_subscription(self, tenant_id: str) -> dict:
        """Return existing subscription or create a free-tier one."""
        sub = self.get_subscription(tenant_id)
        if sub is not None:
            return sub

        now = datetime.now(timezone.utc)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        period_end_str = (now + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        sub_id = f"sub_{secrets.token_hex(12)}"
        limits = TIER_LIMITS[Tier.FREE]

        conn = self._conn()
        try:
            conn.execute(
                """INSERT INTO subscriptions
                   (id, tenant_id, tier, status, cloud_tokens_limit,
                    current_period_start, current_period_end)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    sub_id,
                    tenant_id,
                    Tier.FREE.value,
                    SubscriptionStatus.ACTIVE.value,
                    limits["cloud_tokens"],
                    now_str,
                    period_end_str,
                ),
            )
            conn.commit()
            return self.get_subscription(tenant_id)  # type: ignore[return-value]
        finally:
            conn.close()

    def update_subscription(
        self,
        tenant_id: str,
        *,
        tier: str | None = None,
        status: str | None = None,
        stripe_customer_id: str | None = None,
        stripe_subscription_id: str | None = None,
        cloud_tokens_used: int | None = None,
        current_period_start: str | None = None,
        current_period_end: str | None = None,
    ) -> dict | None:
        """Update subscription fields. Returns updated record."""
        sets: list[str] = []
        params: list[Any] = []

        if tier is not None:
            sets.append("tier = ?")
            params.append(tier)
            # Update cloud_tokens_limit when tier changes
            try:
                tier_enum = Tier(tier)
                sets.append("cloud_tokens_limit = ?")
                params.append(TIER_LIMITS[tier_enum]["cloud_tokens"])
            except ValueError:
                pass

        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if stripe_customer_id is not None:
            sets.append("stripe_customer_id = ?")
            params.append(stripe_customer_id)
        if stripe_subscription_id is not None:
            sets.append("stripe_subscription_id = ?")
            params.append(stripe_subscription_id)
        if cloud_tokens_used is not None:
            sets.append("cloud_tokens_used = ?")
            params.append(cloud_tokens_used)
        if current_period_start is not None:
            sets.append("current_period_start = ?")
            params.append(current_period_start)
        if current_period_end is not None:
            sets.append("current_period_end = ?")
            params.append(current_period_end)

        if not sets:
            return self.get_subscription(tenant_id)

        params.append(tenant_id)
        conn = self._conn()
        try:
            conn.execute(
                f"UPDATE subscriptions SET {', '.join(sets)} WHERE tenant_id = ?",
                params,
            )
            conn.commit()
            return self.get_subscription(tenant_id)
        finally:
            conn.close()

    # -- Tier limit checking -----------------------------------------------

    def check_tier_limit(
        self, tenant_id: str, resource: str
    ) -> tuple[bool, str]:
        """Check if a tenant can use a resource under their tier.

        Returns ``(allowed, message)``.
        - ``allowed=True``: proceed, message may contain a warning.
        - ``allowed=False``: blocked, message explains why.
        """
        sub = self.get_or_create_subscription(tenant_id)
        tier = Tier(sub["tier"])
        limits = TIER_LIMITS[tier]

        if resource == "cloud_tokens":
            limit = limits["cloud_tokens"]

            # Free tier: no cloud tokens at all
            if limit == 0:
                return (
                    False,
                    "Cloud model access requires a Pro plan or higher. "
                    "Upgrade at /v1/billing/checkout.",
                )

            # Unlimited
            if limit == -1:
                return (True, "")

            used = sub.get("cloud_tokens_used", 0) or 0

            # 100% threshold
            if used >= limit:
                return self._check_grace_period(
                    tenant_id,
                    f"Token limit reached: {used:,}/{limit:,}. "
                    f"Upgrade your plan for more tokens.",
                )

            # 80% warning threshold
            if used >= limit * 0.8:
                pct = (used / limit) * 100
                return (
                    True,
                    f"Warning: Token usage at {pct:.0f}% "
                    f"({used:,}/{limit:,}). Consider upgrading.",
                )

            return (True, "")

        if resource == "articles":
            limit = limits["articles"]
            if limit == -1:
                return (True, "")
            # Count articles for this tenant
            count = self._count_articles(tenant_id)
            if count >= limit:
                return (
                    False,
                    f"Article limit reached: {count}/{limit}. "
                    f"Upgrade your plan for unlimited articles.",
                )
            return (True, "")

        if resource == "users":
            limit = limits["users"]
            if limit == -1:
                return (True, "")
            return (True, "")

        return (True, "")

    def _check_grace_period(
        self, tenant_id: str, block_message: str
    ) -> tuple[bool, str]:
        """Enforce 24-hour grace period after limit hit.

        First time hitting the limit: record timestamp, allow with warning.
        Within 24 hours of first hit: allow with warning.
        After 24 hours: block.
        """
        now = time.time()

        if tenant_id not in _limit_hit_timestamps:
            _limit_hit_timestamps[tenant_id] = now
            return (
                True,
                f"GRACE PERIOD: {block_message} "
                f"You have 24 hours before access is restricted.",
            )

        first_hit = _limit_hit_timestamps[tenant_id]
        elapsed = now - first_hit
        remaining_hours = max(0, (_GRACE_PERIOD_SECONDS - elapsed) / 3600)

        if elapsed < _GRACE_PERIOD_SECONDS:
            return (
                True,
                f"GRACE PERIOD: {block_message} "
                f"{remaining_hours:.1f} hours remaining in grace period.",
            )

        # Grace period expired
        return (False, block_message)

    def _count_articles(self, tenant_id: str) -> int:
        """Count articles for a tenant (placeholder -- uses usage_records)."""
        # In production this would query the knowledge store.
        # For now return 0 to avoid blocking free-tier users unnecessarily.
        return 0

    # -- Usage metering ----------------------------------------------------

    def record_usage(
        self,
        tenant_id: str,
        tokens: int,
        provider: str,
        model: str,
    ) -> None:
        """Record token usage for a tenant and update running total."""
        conn = self._conn()
        try:
            conn.execute(
                """INSERT INTO usage_records (tenant_id, tokens_used, provider, model)
                   VALUES (?, ?, ?, ?)""",
                (tenant_id, tokens, provider, model),
            )
            # Increment cloud_tokens_used on the subscription
            conn.execute(
                """UPDATE subscriptions
                   SET cloud_tokens_used = cloud_tokens_used + ?
                   WHERE tenant_id = ?""",
                (tokens, tenant_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_usage_summary(self, tenant_id: str) -> dict:
        """Return aggregated usage stats for the current billing period."""
        sub = self.get_or_create_subscription(tenant_id)
        period_start = sub.get("current_period_start", "")

        conn = self._conn()
        try:
            # Total tokens this period
            row = conn.execute(
                """SELECT COALESCE(SUM(tokens_used), 0) as total_tokens,
                          COUNT(*) as request_count
                   FROM usage_records
                   WHERE tenant_id = ?
                     AND timestamp >= COALESCE(?, '1970-01-01')""",
                (tenant_id, period_start),
            ).fetchone()

            total_tokens = row["total_tokens"] if row else 0
            request_count = row["request_count"] if row else 0

            # Breakdown by provider/model
            rows = conn.execute(
                """SELECT provider, model,
                          SUM(tokens_used) as tokens,
                          COUNT(*) as requests
                   FROM usage_records
                   WHERE tenant_id = ?
                     AND timestamp >= COALESCE(?, '1970-01-01')
                   GROUP BY provider, model
                   ORDER BY tokens DESC""",
                (tenant_id, period_start),
            ).fetchall()

            breakdown = [
                {
                    "provider": r["provider"],
                    "model": r["model"],
                    "tokens": r["tokens"],
                    "requests": r["requests"],
                }
                for r in rows
            ]

            tier = Tier(sub["tier"])
            limit = TIER_LIMITS[tier]["cloud_tokens"]

            return {
                "tenant_id": tenant_id,
                "tier": sub["tier"],
                "status": sub["status"],
                "period_start": sub.get("current_period_start"),
                "period_end": sub.get("current_period_end"),
                "cloud_tokens_used": sub.get("cloud_tokens_used", 0),
                "cloud_tokens_limit": limit,
                "total_tokens_this_period": total_tokens,
                "request_count": request_count,
                "breakdown": breakdown,
            }
        finally:
            conn.close()

    def reset_period_usage(self, tenant_id: str) -> None:
        """Reset usage counters for a new billing period."""
        now = datetime.now(timezone.utc)
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        period_end_str = (now + timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        self.update_subscription(
            tenant_id,
            cloud_tokens_used=0,
            current_period_start=now_str,
            current_period_end=period_end_str,
        )
        # Clear grace period tracking
        _limit_hit_timestamps.pop(tenant_id, None)

    # -- Stripe integration ------------------------------------------------

    def create_checkout_session(
        self,
        tenant_id: str,
        tier: str,
        success_url: str = "https://app.saido.ai/billing/success",
        cancel_url: str = "https://app.saido.ai/billing/cancel",
    ) -> dict:
        """Create a Stripe Checkout session for upgrading to a paid tier.

        Returns a dict with ``url`` (the Checkout page URL) and
        ``session_id``.  If ``stripe`` is not installed, returns a
        mock session.
        """
        if tier not in _STRIPE_PRICE_MAP:
            raise ValueError(
                f"Invalid tier '{tier}'. Must be one of: {list(_STRIPE_PRICE_MAP.keys())}"
            )

        price_id = _STRIPE_PRICE_MAP[tier]

        # Ensure subscription record exists
        sub = self.get_or_create_subscription(tenant_id)

        if _STRIPE_AVAILABLE:
            return self._create_real_checkout(
                tenant_id, price_id, success_url, cancel_url, sub
            )

        return self._create_mock_checkout(tenant_id, tier, price_id)

    def _create_real_checkout(
        self,
        tenant_id: str,
        price_id: str,
        success_url: str,
        cancel_url: str,
        sub: dict,
    ) -> dict:
        """Create a real Stripe Checkout session."""
        customer_id = sub.get("stripe_customer_id")

        session_params: dict[str, Any] = {
            "payment_method_types": ["card"],
            "line_items": [{"price": price_id, "quantity": 1}],
            "mode": "subscription",
            "success_url": success_url + "?session_id={CHECKOUT_SESSION_ID}",
            "cancel_url": cancel_url,
            "metadata": {"tenant_id": tenant_id},
        }

        if customer_id:
            session_params["customer"] = customer_id

        session = _stripe_lib.checkout.Session.create(**session_params)

        return {
            "url": session.url,
            "session_id": session.id,
        }

    def _create_mock_checkout(
        self, tenant_id: str, tier: str, price_id: str
    ) -> dict:
        """Return a mock checkout session when Stripe is not installed."""
        mock_session_id = f"cs_mock_{secrets.token_hex(12)}"
        return {
            "url": f"https://checkout.stripe.com/mock/{mock_session_id}",
            "session_id": mock_session_id,
            "mock": True,
            "tenant_id": tenant_id,
            "tier": tier,
            "price_id": price_id,
        }

    def handle_webhook(self, payload: bytes, signature: str) -> dict:
        """Process a Stripe webhook event.

        Verifies the signature, then dispatches to the appropriate
        handler based on event type.

        Returns a dict with ``status`` and optional ``message``.
        """
        event = self._verify_webhook_event(payload, signature)

        event_type = event.get("type", "")
        handlers = {
            "checkout.session.completed": self._handle_checkout_completed,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
            "invoice.payment_failed": self._handle_payment_failed,
        }

        handler = handlers.get(event_type)
        if handler is None:
            logger.info("Ignoring unhandled webhook event: %s", event_type)
            return {"status": "ignored", "event_type": event_type}

        return handler(event)

    def _verify_webhook_event(
        self, payload: bytes, signature: str
    ) -> dict:
        """Verify and parse a Stripe webhook event."""
        if _STRIPE_AVAILABLE and _STRIPE_WEBHOOK_SECRET:
            try:
                event = _stripe_lib.Webhook.construct_event(
                    payload, signature, _STRIPE_WEBHOOK_SECRET
                )
                return dict(event)
            except _stripe_lib.error.SignatureVerificationError:
                raise ValueError("Invalid webhook signature")
        else:
            # Mock mode: parse payload directly (no signature verification)
            try:
                return json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                raise ValueError("Invalid webhook payload")

    def _handle_checkout_completed(self, event: dict) -> dict:
        """Handle checkout.session.completed: activate subscription."""
        session = event.get("data", {}).get("object", {})
        tenant_id = session.get("metadata", {}).get("tenant_id", "")
        customer_id = session.get("customer", "")
        subscription_id = session.get("subscription", "")

        if not tenant_id:
            return {"status": "error", "message": "No tenant_id in session metadata"}

        # Determine tier from the session (in production, look up the price)
        tier = session.get("metadata", {}).get("tier", "pro")

        self.update_subscription(
            tenant_id,
            tier=tier,
            status=SubscriptionStatus.ACTIVE.value,
            stripe_customer_id=customer_id,
            stripe_subscription_id=subscription_id,
        )

        # Reset usage for the new subscription period
        self.reset_period_usage(tenant_id)

        logger.info(
            "Checkout completed for tenant %s: tier=%s", tenant_id, tier
        )
        return {"status": "ok", "tenant_id": tenant_id, "tier": tier}

    def _handle_subscription_updated(self, event: dict) -> dict:
        """Handle customer.subscription.updated."""
        sub_obj = event.get("data", {}).get("object", {})
        tenant_id = sub_obj.get("metadata", {}).get("tenant_id", "")
        status = sub_obj.get("status", "active")

        if not tenant_id:
            return {"status": "error", "message": "No tenant_id in subscription metadata"}

        # Map Stripe statuses to our statuses
        status_map = {
            "active": SubscriptionStatus.ACTIVE.value,
            "past_due": SubscriptionStatus.PAST_DUE.value,
            "canceled": SubscriptionStatus.CANCELLED.value,
            "trialing": SubscriptionStatus.TRIALING.value,
        }
        mapped_status = status_map.get(status, SubscriptionStatus.ACTIVE.value)

        self.update_subscription(tenant_id, status=mapped_status)

        logger.info(
            "Subscription updated for tenant %s: status=%s",
            tenant_id,
            mapped_status,
        )
        return {"status": "ok", "tenant_id": tenant_id, "new_status": mapped_status}

    def _handle_subscription_deleted(self, event: dict) -> dict:
        """Handle customer.subscription.deleted: downgrade to free."""
        sub_obj = event.get("data", {}).get("object", {})
        tenant_id = sub_obj.get("metadata", {}).get("tenant_id", "")

        if not tenant_id:
            return {"status": "error", "message": "No tenant_id in subscription metadata"}

        self.update_subscription(
            tenant_id,
            tier=Tier.FREE.value,
            status=SubscriptionStatus.CANCELLED.value,
        )

        logger.info("Subscription deleted for tenant %s: downgraded to free", tenant_id)
        return {"status": "ok", "tenant_id": tenant_id, "tier": "free"}

    def _handle_payment_failed(self, event: dict) -> dict:
        """Handle invoice.payment_failed: mark subscription as past_due."""
        invoice = event.get("data", {}).get("object", {})
        subscription_id = invoice.get("subscription", "")

        # Look up tenant by stripe_subscription_id
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT tenant_id FROM subscriptions WHERE stripe_subscription_id = ?",
                (subscription_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return {
                "status": "error",
                "message": f"No subscription found for stripe_subscription_id={subscription_id}",
            }

        tenant_id = row["tenant_id"]
        self.update_subscription(
            tenant_id, status=SubscriptionStatus.PAST_DUE.value
        )

        logger.warning("Payment failed for tenant %s", tenant_id)
        return {"status": "ok", "tenant_id": tenant_id, "new_status": "past_due"}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Global billing manager instance (lazily initialized)
_billing_manager: BillingManager | None = None


def get_billing_manager(db_path: str | None = None) -> BillingManager:
    """Return the global BillingManager instance."""
    global _billing_manager
    if _billing_manager is None or db_path is not None:
        _billing_manager = BillingManager(db_path=db_path)
    return _billing_manager


def reset_billing_manager() -> None:
    """Reset the global billing manager (for testing)."""
    global _billing_manager
    _billing_manager = None
    _limit_hit_timestamps.clear()
