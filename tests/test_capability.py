"""Tests for capability/security disentanglement.

A model that cannot emit a tool call fails every benign task, producing the same
100% false-rejection rate a maximally over-blocking defense would. These pin the
separation of the two.
"""

from __future__ import annotations

import json

from chokepoint.analysis.capability import (
    CapabilityResult,
    annotate,
    frr_decomposition,
    load_capability_results,
)


def write_report(root, name, model, defense, suite, counts, **extra):
    run = root / name
    (run / "traces").mkdir(parents=True)
    (run / "report.json").write_text(
        json.dumps(
            {
                "model_id": model,
                "defense_type": defense,
                "counts": counts,
                "manifest": {"defense": defense, "suite_path": suite},
                **extra,
            }
        ),
        encoding="utf-8",
    )


class TestCapabilityResult:
    def test_pass_rate(self):
        result = CapabilityResult("m", probe_total=10, probe_passed=9, zero_call_episodes=1)
        assert result.pass_rate == 0.9
        assert result.meets_floor(0.8) is True

    def test_below_floor(self):
        result = CapabilityResult("m", probe_total=10, probe_passed=3, zero_call_episodes=7)
        assert result.meets_floor(0.8) is False

    def test_empty_probe_is_not_capable(self):
        assert CapabilityResult("m", 0, 0, 0).meets_floor() is False


class TestLoading:
    def test_reads_undefended_probe_runs(self, tmp_path):
        write_report(
            tmp_path, "r1", "llama3.1:8b", "none",
            "data/scenarios/capability_probe.jsonl",
            {"total_scenarios": 10, "benign_utility_preserved": 4,
             "zero_tool_call_benign": 6, "zero_tool_call_attack": 0},
        )
        results = load_capability_results(tmp_path)
        assert results["llama3.1:8b"].pass_rate == 0.4

    def test_ignores_defended_probe_runs(self, tmp_path):
        """Measuring capability through a defense folds blocking into the estimate."""
        write_report(
            tmp_path, "r1", "m", "llm_judge",
            "data/scenarios/capability_probe.jsonl",
            {"total_scenarios": 10, "benign_utility_preserved": 2},
        )
        assert load_capability_results(tmp_path) == {}

    def test_ignores_non_probe_suites(self, tmp_path):
        write_report(
            tmp_path, "r1", "m", "none", "data/scenarios/core_v2.jsonl",
            {"total_scenarios": 95, "benign_utility_preserved": 40},
        )
        assert load_capability_results(tmp_path) == {}


class TestAnnotation:
    def test_capable_model_is_interpretable(self):
        capability = {"m": CapabilityResult("m", 10, 10, 0)}
        rows = annotate([{"model_id": "m", "defense_type": "none", "asr_percentage": 40.0}],
                        capability)
        assert rows[0]["interpretable"] is True

    def test_incapable_model_is_flagged_not_reported_as_secure(self):
        """Zero ASR from a model that cannot act says nothing about the defense."""
        capability = {"m": CapabilityResult("m", 10, 2, 8)}
        rows = annotate([{"model_id": "m", "defense_type": "type_checker",
                          "asr_percentage": 0.0}], capability)
        assert rows[0]["interpretable"] is False
        assert rows[0]["asr"] == 0.0

    def test_unmeasured_model_is_unknown_not_assumed_capable(self):
        rows = annotate([{"model_id": "unseen", "defense_type": "none"}], {})
        assert rows[0]["interpretable"] is None


class TestFrrDecomposition:
    def test_splits_capability_from_blocking(self):
        cell = {
            "counts": {
                "benign_scenarios": 50,
                "benign_utility_preserved": 10,
                "zero_tool_call_benign": 30,
            }
        }
        result = frr_decomposition(cell)
        assert result["frr_reported"] == 80.0
        assert result["attributable_to_capability"] == 30
        assert result["attributable_to_defense"] == 10
        assert result["frr_capability_adjusted"] == 20.0

    def test_a_fully_capable_model_attributes_everything_to_the_defense(self):
        cell = {
            "counts": {
                "benign_scenarios": 50,
                "benign_utility_preserved": 40,
                "zero_tool_call_benign": 0,
            }
        }
        result = frr_decomposition(cell)
        assert result["attributable_to_defense"] == 10
        assert result["frr_capability_adjusted"] == result["frr_reported"]

    def test_total_capability_failure_leaves_nothing_for_the_defense(self):
        cell = {
            "counts": {
                "benign_scenarios": 50,
                "benign_utility_preserved": 0,
                "zero_tool_call_benign": 50,
            }
        }
        result = frr_decomposition(cell)
        assert result["frr_reported"] == 100.0
        assert result["frr_capability_adjusted"] == 0.0

    def test_no_benign_scenarios_returns_none(self):
        assert frr_decomposition({"counts": {"benign_scenarios": 0}}) is None

    def test_never_reports_negative_attribution(self):
        cell = {
            "counts": {
                "benign_scenarios": 10,
                "benign_utility_preserved": 8,
                "zero_tool_call_benign": 5,
            }
        }
        assert frr_decomposition(cell)["attributable_to_defense"] == 0
