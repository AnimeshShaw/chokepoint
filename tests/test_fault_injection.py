"""Tests for availability faults and the fail-open/fail-closed axis.

The attack the defense literature does not measure: making the judge unreachable
rather than fooling it. Whether that helps the attacker or denies service to the
user depends entirely on the fail policy — which is the point.
"""

from __future__ import annotations

from chokepoint.defenses.fault_injection import (
    FaultInjector,
    always_fail_injector,
    oversize_argument_injector,
)
from chokepoint.defenses.llm_judge import LLMJudgeDefense, Verdict


class TestFaultInjector:
    def test_oversize_argument_triggers_failure(self):
        injector = oversize_argument_injector(threshold=100)
        assert injector.should_fail("send_email", {"body": "x" * 200}) is True

    def test_small_argument_does_not_trigger(self):
        injector = oversize_argument_injector(threshold=100)
        assert injector.should_fail("send_email", {"body": "short"}) is False

    def test_predicate_overrides_size(self):
        injector = FaultInjector(predicate=lambda tool, args: tool == "execute_sql_query")
        assert injector.should_fail("execute_sql_query", {}) is True
        assert injector.should_fail("read_email", {"body": "x" * 9999}) is False

    def test_always_fail(self):
        assert always_fail_injector().should_fail("anything", {}) is True

    def test_no_config_never_fails(self):
        assert FaultInjector().should_fail("send_email", {"body": "x" * 99999}) is False


class TestJudgeUnderFault:
    def _judge(self, fail_closed):
        # No network: _ask_judge returns before touching the model when a fault
        # fires, so _build_judge_llm is the only call that could reach out. We
        # avoid it by constructing the defense and overriding the llm.
        defense = LLMJudgeDefense.__new__(LLMJudgeDefense)
        defense.user_prompt = "do a benign task"
        defense.recorder = None
        defense.fail_closed = fail_closed
        defense.fault_injector = always_fail_injector()
        defense._judge_llm = None  # must never be used when the fault fires
        return defense

    def test_fail_closed_blocks_when_judge_is_down(self):
        verdict = self._judge(fail_closed=True)._ask_judge("send_email", {"body": "x"})
        assert verdict.verdict is Verdict.MALICIOUS
        assert "unavailable" in verdict.reasoning

    def test_fail_open_allows_when_judge_is_down(self):
        verdict = self._judge(fail_closed=False)._ask_judge("send_email", {"body": "x"})
        assert verdict.verdict is Verdict.SAFE
        assert "unavailable" in verdict.reasoning

    def test_fault_confidence_is_zero_so_roc_can_exclude_it(self):
        verdict = self._judge(fail_closed=True)._ask_judge("send_email", {"body": "x"})
        assert verdict.confidence == 0.0

    def test_oversize_argument_is_what_triggers_a_real_bypass(self):
        """An attacker inflates an argument; under fail-open the call goes through."""
        defense = LLMJudgeDefense.__new__(LLMJudgeDefense)
        defense.user_prompt = "p"
        defense.recorder = None
        defense.fail_closed = False
        defense.fault_injector = oversize_argument_injector(threshold=8000)
        defense._judge_llm = None

        bypass = defense._ask_judge("send_email", {"body": "A" * 9000})
        assert bypass.verdict is Verdict.SAFE  # judge skipped, call allowed
