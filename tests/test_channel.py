"""Tests for channel-asymmetry pairing.

Verifies the analysis pairs matched-payload triplets correctly and only reports
within-payload comparisons over complete triplets.
"""

from __future__ import annotations

import json

from chokepoint.analysis.channel import collect, summarize


def write_trace(root, run, sid, model, base, channel, succeeded):
    report = root / run
    (report / "traces").mkdir(parents=True, exist_ok=True)
    if not (report / "report.json").exists():
        (report / "report.json").write_text(
            json.dumps({"model_id": model}), encoding="utf-8"
        )
    (report / "traces" / f"{sid}.json").write_text(
        json.dumps(
            {
                "scenario_id": sid,
                "scenario": {"tags": ["channel_asymmetry", f"base:{base}", f"channel:{channel}"]},
                "outcome": {"attack_succeeded": succeeded},
            }
        ),
        encoding="utf-8",
    )


class TestCollect:
    def test_extracts_base_and_channel_from_tags(self, tmp_path):
        write_trace(tmp_path, "r1", "s1", "gpt", "exfil", "email", True)
        rows = collect(tmp_path)
        assert rows == [{"model": "gpt", "base": "exfil", "channel": "email", "succeeded": True}]

    def test_ignores_traces_without_channel_tags(self, tmp_path):
        report = tmp_path / "r1"
        (report / "traces").mkdir(parents=True)
        (report / "report.json").write_text(json.dumps({"model_id": "gpt"}), encoding="utf-8")
        (report / "traces" / "x.json").write_text(
            json.dumps({"scenario": {"tags": ["other"]}, "outcome": {}}), encoding="utf-8"
        )
        assert collect(tmp_path) == []


class TestSummarize:
    def test_per_channel_rate(self, tmp_path):
        write_trace(tmp_path, "r1", "s1", "gpt", "b1", "email", True)
        write_trace(tmp_path, "r1", "s2", "gpt", "b2", "email", False)
        write_trace(tmp_path, "r1", "s3", "gpt", "b1", "sql", False)
        report = summarize(collect(tmp_path))
        assert report["channel_success_rate"]["email"] == 50.0
        assert report["channel_success_rate"]["sql"] == 0.0

    def test_only_complete_triplets_count_in_the_paired_analysis(self, tmp_path):
        # b1 complete across all three channels; b2 only has email.
        for ch, ok in [("email", True), ("file", True), ("sql", False)]:
            write_trace(tmp_path, "r1", f"b1-{ch}", "gpt", "b1", ch, ok)
        write_trace(tmp_path, "r1", "b2-email", "gpt", "b2", "email", True)

        report = summarize(collect(tmp_path))
        assert report["n_complete_triplets"] == 1
        assert report["paired_successes"]["email"] == 1
        assert report["paired_successes"]["sql"] == 0

    def test_provenance_gap_is_visible(self, tmp_path):
        """The finding shape: same payload, different success by channel."""
        for base in ("b1", "b2", "b3"):
            write_trace(tmp_path, "r1", f"{base}-email", "gpt", base, "email", True)
            write_trace(tmp_path, "r1", f"{base}-file", "gpt", base, "file", True)
            write_trace(tmp_path, "r1", f"{base}-sql", "gpt", base, "sql", False)
        report = summarize(collect(tmp_path))
        assert report["paired_successes"]["email"] == 3
        assert report["paired_successes"]["sql"] == 0
