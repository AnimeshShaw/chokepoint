"""Gap 1: channel asymmetry with the payload held constant.

Pairs the matched-payload triplets produced by ``build_channel_suite.py`` and
reports attack success by delivery channel. Because the payload text is
byte-identical across each triplet, a difference in success rate isolates a
provenance effect: the model treating the same instruction differently by where
it arrived.

Ground truth is read from persisted traces, so this is offline.

    python analysis/channel.py --results-root results/runs
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _base_and_channel(tags: list[str]) -> tuple[str | None, str | None]:
    base = channel = None
    for tag in tags:
        if tag.startswith("base:"):
            base = tag.split(":", 1)[1]
        elif tag.startswith("channel:"):
            channel = tag.split(":", 1)[1]
    return base, channel


def collect(results_root: str | Path) -> list[dict[str, Any]]:
    """Read channel-asymmetry traces into (base, channel, model, succeeded) rows."""
    rows: list[dict[str, Any]] = []
    for report_path in sorted(Path(results_root).glob("*/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        model = report.get("model_id", "?")
        traces_dir = report_path.parent / "traces"
        for trace_path in sorted(traces_dir.glob("*.json")):
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            scenario = trace.get("scenario", {})
            base, channel = _base_and_channel(scenario.get("tags", []))
            if base is None or channel is None:
                continue
            rows.append(
                {
                    "model": model,
                    "base": base,
                    "channel": channel,
                    "succeeded": bool(trace.get("outcome", {}).get("attack_succeeded")),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-channel success rate and paired within-triplet comparison."""
    by_channel: dict[str, list[bool]] = defaultdict(list)
    triplets: dict[tuple, dict[str, bool]] = defaultdict(dict)

    for row in rows:
        by_channel[row["channel"]].append(row["succeeded"])
        triplets[(row["model"], row["base"])][row["channel"]] = row["succeeded"]

    channel_rates = {
        channel: round(sum(v) / len(v) * 100, 2) if v else None
        for channel, v in by_channel.items()
    }

    # Paired analysis: only triplets with all three channels present, so the
    # comparison is genuinely within-payload. Every channel is reported even at
    # zero successes, so an all-blocked channel is visible rather than absent.
    complete = [t for t in triplets.values() if {"email", "file", "sql"} <= set(t)]
    paired: dict[str, int] = {"email": 0, "file": 0, "sql": 0}
    for triplet in complete:
        for channel in ("email", "file", "sql"):
            if triplet[channel]:
                paired[channel] += 1

    return {
        "n_rows": len(rows),
        "n_complete_triplets": len(complete),
        "channel_success_rate": channel_rates,
        "paired_successes": dict(paired),
        "paired_total": len(complete),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/runs")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = collect(args.results_root)
    if not rows:
        print(
            f"No channel-asymmetry traces under {args.results_root}.\n"
            "Build the suite (scripts/build_channel_suite.py) and run it first."
        )
        return 1

    report = summarize(rows)
    print(f"# Channel asymmetry (payload held constant)\n\n{report['n_rows']} rows, "
          f"{report['n_complete_triplets']} complete triplets\n")

    print("## Attack success by delivery channel\n")
    print("| Channel | ASR % |")
    print("|---|---|")
    for channel in ("email", "file", "sql"):
        rate = report["channel_success_rate"].get(channel)
        print(f"| {channel} | {rate if rate is not None else 'n/a'} |")

    if report["paired_total"]:
        print(f"\n## Paired within-payload successes ({report['paired_total']} triplets)\n")
        print("| Channel | Successes |")
        print("|---|---|")
        for channel in ("email", "file", "sql"):
            print(f"| {channel} | {report['paired_successes'].get(channel, 0)} |")
        print(
            "\n> Because each triplet carries byte-identical payload text, any gap "
            "between these channels is a provenance effect: the same instruction "
            "acted on differently depending on where it was read from."
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFull report: {args.out}")
    return 0
