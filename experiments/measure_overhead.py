"""Latency overhead of each defense layer.

Rewritten after the audit. The pre-audit version timed 3 trials of a single
prompt and reported no variance; it measured the type checker as 0.92x baseline
— i.e. *faster* than doing nothing, which is not physically meaningful — and
that figure was carried into the paper as a finding.

This version runs many trials across several prompts, reports median and IQR,
and states explicitly when a bootstrap interval on the ratio spans 1.0, meaning
no detectable difference.

    python experiments/measure_overhead.py --model gpt-5.6-terra --trials 20
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chokepoint.agents.base_agent import BaseAgent  # noqa: E402
from chokepoint.agents.defended_agent import DefendedAgent, DefenseType  # noqa: E402
from chokepoint.environment.environment import Environment  # noqa: E402
from chokepoint.tools.registry import build_tools  # noqa: E402

PROMPTS = [
    "Read the latest email and summarize its content.",
    "List the open support tickets sorted by priority.",
    "Read notes/standup.md and summarize the blockers.",
    "Query the inventory table for items with fewer than 10 units.",
    "Summarize my unread emails in three bullet points.",
]

ENV_SPEC = {
    "emails": [
        {"id": "1", "sender": "ops@company.example", "subject": "Status",
         "body": "All systems nominal. Deploy window is Thursday 14:00."},
    ],
    "files": {"notes/standup.md": "- API latency fixed\n- Billing migration blocked"},
    "db": {
        "tickets": [
            {"id": 91, "priority": "high", "subject": "VPN drops", "status": "open"},
            {"id": 92, "priority": "low", "subject": "Printer offline", "status": "open"},
        ],
        "inventory": [
            {"sku": "AX-100", "product": "Desk Lamp", "quantity": 4},
            {"sku": "AX-220", "product": "Monitor Arm", "quantity": 31},
        ],
    },
}

CONFIGS = [
    ("baseline", DefenseType.NONE),
    ("type_checker", DefenseType.TYPE_CHECKER),
    ("capability_router", DefenseType.CAPABILITY_ROUTER),
    ("llm_judge", DefenseType.LLM_JUDGE),
]


def time_one(model_id: str, defense: DefenseType, prompt: str) -> float:
    env = Environment.from_spec(ENV_SPEC)
    tools = build_tools(env)
    agent = (
        BaseAgent(model_id=model_id, tools=tools)
        if defense is DefenseType.NONE
        else DefendedAgent(
            model_id=model_id, tools=tools, defense_type=defense, recorder=env.record
        )
    )
    start = time.perf_counter()
    agent.run(prompt)
    return time.perf_counter() - start


def bootstrap_ratio_ci(
    treatment: list[float], baseline: list[float], iterations: int = 5000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the ratio of medians."""
    rng = random.Random(seed)
    ratios = []
    for _ in range(iterations):
        t = statistics.median(rng.choices(treatment, k=len(treatment)))
        b = statistics.median(rng.choices(baseline, k=len(baseline)))
        if b > 0:
            ratios.append(t / b)
    ratios.sort()
    lo = ratios[int(0.025 * len(ratios))]
    hi = ratios[int(0.975 * len(ratios)) - 1]
    return round(lo, 3), round(hi, 3)


def summarize(samples: list[float]) -> dict[str, float]:
    quantiles = statistics.quantiles(samples, n=4) if len(samples) >= 4 else [float("nan")] * 3
    return {
        "n": len(samples),
        "median_s": round(statistics.median(samples), 3),
        "mean_s": round(statistics.fmean(samples), 3),
        "q1_s": round(quantiles[0], 3),
        "q3_s": round(quantiles[2], 3),
        "min_s": round(min(samples), 3),
        "max_s": round(max(samples), 3),
    }


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--trials", type=int, default=20, help="Trials per configuration")
    parser.add_argument("--out", default="results/overhead.json")
    args = parser.parse_args()

    if args.trials < 10:
        print(
            f"Warning: {args.trials} trials is too few to separate defense cost "
            "from API jitter. 20 or more is recommended.",
            file=sys.stderr,
        )

    print(f"Warmup on {args.model}...")
    try:
        time_one(args.model, DefenseType.NONE, PROMPTS[0])
    except Exception as exc:  # noqa: BLE001
        print(f"Warmup failed: {exc}", file=sys.stderr)
        return 1

    samples: dict[str, list[float]] = {}
    for name, defense in CONFIGS:
        print(f"\nMeasuring {name} ({args.trials} trials across {len(PROMPTS)} prompts)")
        durations: list[float] = []
        for i in range(args.trials):
            prompt = PROMPTS[i % len(PROMPTS)]
            try:
                durations.append(time_one(args.model, defense, prompt))
            except Exception as exc:  # noqa: BLE001
                print(f"  trial {i} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if durations:
            samples[name] = durations
            print(f"  median {statistics.median(durations):.2f}s over {len(durations)} trials")

    if "baseline" not in samples:
        print("No baseline samples; cannot compute overhead.", file=sys.stderr)
        return 1

    baseline = samples["baseline"]
    report = {"model": args.model, "prompts": PROMPTS, "configurations": {}}

    print("\n| Defense | Median (s) | IQR (s) | Ratio vs baseline | 95% CI | Detectable? |")
    print("|---|---|---|---|---|---|")
    for name, durations in samples.items():
        stats = summarize(durations)
        ratio = round(stats["median_s"] / statistics.median(baseline), 3)
        if name == "baseline":
            lo = hi = 1.0
            verdict = "reference"
        else:
            lo, hi = bootstrap_ratio_ci(durations, baseline)
            verdict = "no" if lo <= 1.0 <= hi else "yes"

        report["configurations"][name] = {**stats, "ratio": ratio, "ci95": [lo, hi],
                                          "detectable": verdict}
        print(
            f"| {name} | {stats['median_s']:.2f} | "
            f"{stats['q1_s']:.2f}–{stats['q3_s']:.2f} | {ratio:.2f}x "
            f"| [{lo:.2f}, {hi:.2f}] | {verdict} |"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")
    print(
        "\n> A confidence interval spanning 1.00x means the measurement cannot "
        "distinguish that defense's cost from run-to-run variance. Report it as "
        "such rather than as a speedup or slowdown."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
