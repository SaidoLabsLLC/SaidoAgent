"""
Cost tracking for Saido Agent sessions.

Tracks token usage per provider/model, computes dollar costs,
and estimates savings vs. all-cloud execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from saido_agent.core.providers import COSTS, bare_model


# Fallback cloud pricing for savings estimation (per million tokens)
# Uses Claude Sonnet pricing as the baseline for "what it would cost on cloud"
_CLOUD_BASELINE_INPUT = 3.0   # $/M tokens
_CLOUD_BASELINE_OUTPUT = 15.0  # $/M tokens


@dataclass
class ModelUsage:
    """Token usage for a single model within a session."""
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> float:
        """Actual dollar cost. Local models are always $0."""
        if self.provider in ("ollama", "lmstudio"):
            return 0.0
        bm = bare_model(self.model)
        ic, oc = COSTS.get(bm, (0.0, 0.0))
        return (self.input_tokens * ic + self.output_tokens * oc) / 1_000_000

    @property
    def cloud_equivalent_cost(self) -> float:
        """What these tokens would have cost on cloud (for savings calc)."""
        bm = bare_model(self.model)
        ic, oc = COSTS.get(bm, (_CLOUD_BASELINE_INPUT, _CLOUD_BASELINE_OUTPUT))
        return (self.input_tokens * ic + self.output_tokens * oc) / 1_000_000


@dataclass
class CostTracker:
    """Tracks token usage and costs across an entire session."""
    _usage: dict[str, ModelUsage] = field(default_factory=dict)

    # MED-3: Budget limits
    _max_tokens: int = 1_000_000
    _max_turns: int = 200
    _current_turns: int = 0
    _budget_warned: bool = False
    _budget_paused: bool = False

    def set_budget(self, max_tokens: int = 1_000_000, max_turns: int = 200) -> None:
        """Configure budget limits."""
        self._max_tokens = max_tokens
        self._max_turns = max_turns

    def record(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for a provider/model combination."""
        key = f"{provider}/{model}"
        if key not in self._usage:
            self._usage[key] = ModelUsage(provider=provider, model=model)
        entry = self._usage[key]
        entry.input_tokens += input_tokens
        entry.output_tokens += output_tokens

    def record_turn(self) -> None:
        """Record a conversation turn."""
        self._current_turns += 1

    def check_budget(self) -> tuple[str, bool]:
        """Check budget status.

        Returns:
            (status_message, can_continue) — can_continue is False when
            budget is exceeded and requires user confirmation.
        """
        total = self.total_tokens
        token_pct = (total / self._max_tokens * 100) if self._max_tokens > 0 else 0
        turn_pct = (self._current_turns / self._max_turns * 100) if self._max_turns > 0 else 0

        # Check hard limits (100%)
        if total >= self._max_tokens:
            self._budget_paused = True
            return (
                f"TOKEN BUDGET EXCEEDED: {total:,} / {self._max_tokens:,} tokens "
                f"({token_pct:.0f}%). Session paused — confirm to continue.",
                False,
            )

        if self._current_turns >= self._max_turns:
            self._budget_paused = True
            return (
                f"TURN LIMIT REACHED: {self._current_turns} / {self._max_turns} turns "
                f"({turn_pct:.0f}%). Session paused — confirm to continue.",
                False,
            )

        # Check warning threshold (80%)
        warnings: list[str] = []
        if token_pct >= 80 and not self._budget_warned:
            warnings.append(
                f"Token budget at {token_pct:.0f}%: {total:,} / {self._max_tokens:,}"
            )
        if turn_pct >= 80 and not self._budget_warned:
            warnings.append(
                f"Turn limit at {turn_pct:.0f}%: {self._current_turns} / {self._max_turns}"
            )

        if warnings:
            self._budget_warned = True
            return (
                "WARNING: " + "; ".join(warnings) + ". Consider wrapping up.",
                True,
            )

        return ("", True)

    def confirm_budget_override(self) -> None:
        """User has confirmed they want to continue past budget limits."""
        self._budget_paused = False
        # Double the limits so we don't immediately re-pause
        self._max_tokens = int(self._max_tokens * 1.5)
        self._max_turns = int(self._max_turns * 1.5)
        self._budget_warned = False

    def format_budget(self) -> str:
        """Format budget status for the /budget command."""
        total = self.total_tokens
        token_pct = (total / self._max_tokens * 100) if self._max_tokens > 0 else 0
        turn_pct = (self._current_turns / self._max_turns * 100) if self._max_turns > 0 else 0

        remaining_tokens = max(0, self._max_tokens - total)
        remaining_turns = max(0, self._max_turns - self._current_turns)

        lines = [
            "Session budget:",
            f"  Tokens: {total:,} / {self._max_tokens:,} ({token_pct:.1f}%) — {remaining_tokens:,} remaining",
            f"  Turns:  {self._current_turns} / {self._max_turns} ({turn_pct:.1f}%) — {remaining_turns} remaining",
            f"  Cost:   ${self.total_cost:.4f}",
        ]
        if self._budget_paused:
            lines.append("  Status: PAUSED (budget exceeded — awaiting confirmation)")
        elif self._budget_warned:
            lines.append("  Status: WARNING (approaching limits)")
        else:
            lines.append("  Status: OK")
        return "\n".join(lines)

    @property
    def total_cost(self) -> float:
        return sum(u.cost for u in self._usage.values())

    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self._usage.values())

    @property
    def estimated_savings(self) -> float:
        """Estimated savings vs. routing all tokens through cloud."""
        cloud_total = sum(u.cloud_equivalent_cost for u in self._usage.values())
        actual_total = self.total_cost
        return max(0.0, cloud_total - actual_total)

    def format_report(self) -> str:
        """Format a cost report for the /cost command."""
        lines: list[str] = ["Session cost:"]

        # Group by local vs cloud
        local_entries = []
        cloud_entries = []
        for usage in self._usage.values():
            if usage.provider in ("ollama", "lmstudio"):
                local_entries.append(usage)
            else:
                cloud_entries.append(usage)

        for entry in local_entries:
            tokens_str = f"{entry.total_tokens:,}"
            lines.append(
                f"  Local ({entry.model}):{' ' * max(1, 24 - len(entry.model))}"
                f"{tokens_str:>12} tokens — ${entry.cost:.2f}"
            )

        for entry in cloud_entries:
            label = f"Cloud ({entry.model})"
            tokens_str = f"{entry.total_tokens:,}"
            lines.append(
                f"  {label}:{' ' * max(1, 28 - len(label))}"
                f"{tokens_str:>12} tokens — ${entry.cost:.2f}"
            )

        lines.append(f"  {'Total:':<34}${self.total_cost:>8.2f}")
        lines.append(
            f"  {'Estimated savings vs all-cloud:':<34}${self.estimated_savings:>8.2f}"
        )

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all tracked usage."""
        self._usage.clear()
