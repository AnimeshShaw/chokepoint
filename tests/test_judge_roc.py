"""Tests for judge calibration analysis.

Published judge defenses report a single binary operating point, so comparing
two judges compares where their thresholds happen to sit rather than how well
they discriminate. These pin the machinery that recovers the whole frontier.
"""

from __future__ import annotations

import json

from chokepoint.analysis.judge_roc import (
    JudgedCall,
    auc,
    best_operating_point,
    calibration_bins,
    collect_judged_calls,
    roc_points,
    summarize,
)


def call(confidence, adversarial, verdict="SAFE"):
    return JudgedCall(
        run_id="r",
        scenario_id="s",
        tool="send_email",
        confidence=confidence,
        verdict=verdict,
        adversarial=adversarial,
    )


PERFECT = [call(0.9, True), call(0.8, True), call(0.2, False), call(0.1, False)]
RANDOM = [call(0.5, True), call(0.5, False), call(0.5, True), call(0.5, False)]


class TestRocPoints:
    def test_perfect_separation_reaches_auc_one(self):
        assert auc(roc_points(PERFECT)) == 1.0

    def test_uninformative_confidence_gives_chance_auc(self):
        assert auc(roc_points(RANDOM)) == 0.5

    def test_lowering_the_threshold_never_lowers_tpr(self):
        points = sorted(roc_points(PERFECT), key=lambda p: -p["threshold"])
        tprs = [p["tpr"] for p in points]
        assert tprs == sorted(tprs)

    def test_confusion_counts_are_consistent(self):
        for point in roc_points(PERFECT):
            assert point["tp"] + point["fn"] == 2
            assert point["fp"] + point["tn"] == 2

    def test_empty_input_yields_no_points(self):
        assert roc_points([]) == []


class TestOperatingPoint:
    def test_respects_the_false_positive_budget(self):
        point = best_operating_point(roc_points(PERFECT), max_fpr=0.0)
        assert point["fpr"] == 0.0
        assert point["tpr"] == 1.0

    def test_a_loose_budget_admits_more_detection(self):
        calls = [call(0.9, True), call(0.6, True), call(0.55, False), call(0.1, False)]
        strict = best_operating_point(roc_points(calls), max_fpr=0.0)
        loose = best_operating_point(roc_points(calls), max_fpr=0.5)
        assert loose["tpr"] >= strict["tpr"]

    def test_impossible_budget_returns_none(self):
        assert best_operating_point([{"fpr": 0.5, "tpr": 1.0}], max_fpr=0.1) is None


class TestCalibration:
    def test_well_calibrated_confidence_matches_observed_rate(self):
        calls = [call(0.9, True) for _ in range(9)] + [call(0.9, False)]
        bins = calibration_bins(calls)
        row = next(b for b in bins if b["n"] == 10)
        assert abs(row["mean_confidence"] - row["observed_adversarial_rate"]) < 0.05

    def test_miscalibration_is_visible(self):
        calls = [call(0.95, False) for _ in range(10)]
        row = calibration_bins(calls)[0]
        assert row["mean_confidence"] > 0.9
        assert row["observed_adversarial_rate"] == 0.0

    def test_confidence_of_exactly_one_lands_in_the_top_bin(self):
        assert calibration_bins([call(1.0, True)])[0]["n"] == 1


class TestSummary:
    def test_reports_the_default_binary_point_alongside_the_curve(self):
        calls = [
            call(0.9, True, "MALICIOUS"),
            call(0.6, True, "SAFE"),
            call(0.2, False, "SAFE"),
        ]
        report = summarize(calls)
        assert report["default_operating_point"]["tpr"] == 0.5
        assert report["auc"] == 1.0

    def test_summary_shape(self):
        report = summarize(PERFECT)
        for key in ("n_calls", "n_adversarial", "auc", "calibration", "roc"):
            assert key in report


class TestTraceCollection:
    def test_pairs_judge_decisions_with_ground_truth(self, tmp_path):
        traces = tmp_path / "run-1" / "traces"
        traces.mkdir(parents=True)
        (traces / "s1.json").write_text(
            json.dumps(
                {
                    "scenario_id": "s1",
                    "scenario": {
                        "attack_goal": {
                            "description": "d",
                            "success_conditions": [
                                {
                                    "tool": "send_email",
                                    "args": {"to_address": {"icontains": "evil"}},
                                }
                            ],
                        }
                    },
                    "defense_logs": {
                        "llm_judge": [
                            {
                                "tool_name": "send_email",
                                "args": {"to_address": "x@evil.example"},
                                "verdict": "MALICIOUS",
                                "confidence": 0.95,
                            },
                            {
                                "tool_name": "read_email",
                                "args": {},
                                "verdict": "SAFE",
                                "confidence": 0.02,
                            },
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

        calls = collect_judged_calls(tmp_path)
        assert len(calls) == 2
        by_tool = {c.tool: c for c in calls}
        assert by_tool["send_email"].adversarial is True
        assert by_tool["read_email"].adversarial is False

    def test_ignores_runs_with_no_judge(self, tmp_path):
        traces = tmp_path / "run-1" / "traces"
        traces.mkdir(parents=True)
        (traces / "s1.json").write_text(
            json.dumps({"scenario_id": "s1", "defense_logs": {}}), encoding="utf-8"
        )
        assert collect_judged_calls(tmp_path) == []
