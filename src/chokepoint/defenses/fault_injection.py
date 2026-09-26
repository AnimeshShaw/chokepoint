"""Availability faults against a defense, and the fail-open/fail-closed axis.

The defense literature studies whether a guardrail reaches the *right verdict*.
It does not study what happens when the guardrail does not answer at all — a
timeout, a rate-limit, a crashed sidecar — which in production is routine.

That gap hides an attack. If an attacker can induce the judge to fail (an
oversized argument, an adversarially expensive payload), the outcome depends
entirely on a deployment choice nobody measures:

* **fail-open** — an unreachable judge lets the call through. The attacker gets
  a clean bypass by attacking the judge's *availability* rather than its
  judgment.
* **fail-closed** — an unreachable judge blocks the call. The same move becomes
  a denial of service on the agent: no attack lands, but no legitimate work
  completes either.

Either way the attacker gains something, and the deployment cannot escape the
trade-off by being more accurate. This module injects controllable judge faults
so both outcomes are measurable.
"""

from __future__ import annotations

from collections.abc import Callable


class JudgeUnavailable(RuntimeError):
    """Raised by a fault-injecting judge to simulate an availability failure."""


class FaultInjector:
    """Decides, per call, whether the judge should fail.

    Kept as an object rather than a flag so faults can be deterministic in
    tests (fail on a predicate) and configurable in experiments (fail when an
    argument exceeds a size an attacker could deliberately inflate).
    """

    def __init__(
        self,
        predicate: Callable[[str, dict], bool] | None = None,
        oversize_threshold: int | None = None,
    ) -> None:
        """
        Args:
            predicate: ``(tool_name, args) -> bool``; when it returns True the
                judge fails for that call. Overrides ``oversize_threshold``.
            oversize_threshold: If set, the judge fails whenever any string
                argument is at least this long — modelling an attacker who
                inflates an argument to blow the judge's context or latency
                budget.
        """
        self.predicate = predicate
        self.oversize_threshold = oversize_threshold

    def should_fail(self, tool_name: str, args: dict) -> bool:
        if self.predicate is not None:
            return self.predicate(tool_name, args)
        if self.oversize_threshold is not None:
            return any(
                isinstance(v, str) and len(v) >= self.oversize_threshold
                for v in args.values()
            )
        return False


def oversize_argument_injector(threshold: int = 8000) -> FaultInjector:
    """An attacker inflating a tool argument past the judge's budget."""
    return FaultInjector(oversize_threshold=threshold)


def always_fail_injector() -> FaultInjector:
    """Model a judge outage: every call fails."""
    return FaultInjector(predicate=lambda tool, args: True)
