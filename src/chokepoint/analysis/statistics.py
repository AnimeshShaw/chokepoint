"""Statistical analysis over persisted run reports.

Rewritten after the audit. The pre-audit version reconstructed success counts
by multiplying rounded percentages back out, computed no p-value at all (it
bucketed an uncorrected chi-square against critical values), and the paper then
quoted ``p = 0.005`` to three digits.

This version reads integer counts straight from run reports and reports exact
tests appropriate to the sample size.

Invoked through the thin wrapper at ``analysis/statistics.py``:

    python analysis/statistics.py --results-root results/runs
    python analysis/statistics.py --results-root results/runs --compare llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

Z_95 = 1.959964


# ── Interval estimation ──────────────────────────────────────────────────────

def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion, as percentages."""
    if total == 0:
        return (float("nan"),) * 3
    p = successes / total
    denom = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    return (
        round(p * 100, 2),
        round(max(0.0, (centre - spread) / denom) * 100, 2),
        round(min(1.0, (centre + spread) / denom) * 100, 2),
    )


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for the difference between two proportions."""
    phi = lambda p: 2 * math.asin(math.sqrt(max(0.0, min(1.0, p))))  # noqa: E731
    return round(abs(phi(p1) - phi(p2)), 4)


# ── Exact hypothesis testing ─────────────────────────────────────────────────

def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on the 2x2 table [[a, b], [c, d]].

    Preferred over chi-square here: attack strata run to a few dozen scenarios,
    where expected cell counts fall below the threshold at which the chi-square
    approximation is trustworthy. Uses SciPy when available and falls back to a
    self-contained implementation otherwise.
    """
    try:
        from scipy.stats import fisher_exact  # type: ignore

        return float(fisher_exact([[a, b], [c, d]])[1])
    except ImportError:
        pass

    def hypergeom_pmf(k: int, n1: int, n2: int, t: int) -> float:
        return (
            math.comb(n1, k) * math.comb(n2, t - k) / math.comb(n1 + n2, t)
            if 0 <= k <= n1 and 0 <= t - k <= n2
            else 0.0
        )

    row1, row2, col1 = a + b, c + d, a + c
    observed = hypergeom_pmf(a, row1, row2, col1)
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    # Sum every table at least as extreme as the observed one.
    return min(
        1.0,
        sum(
            p
            for k in range(lo, hi + 1)
            if (p := hypergeom_pmf(k, row1, row2, col1)) <= observed * (1 + 1e-9)
        ),
    )


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction across a family of comparisons.

    A benchmark comparing several defenses against a baseline runs a family of
    tests; reporting uncorrected p-values inflates the family-wise error rate.
    """
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, dict] = {}
    prev = 0.0
    for i, (key, p) in enumerate(ordered):
        adjusted = min(1.0, max(prev, (m - i) * p))
        prev = adjusted
        out[key] = {"p_raw": p, "p_adjusted": adjusted, "significant": adjusted < alpha}
    return out


# ── Report loading ───────────────────────────────────────────────────────────

@dataclass
class Cell:
    """One (model, defense) configuration with integer outcome counts."""

    model: str
    defense: str
    attack_n: int
    attack_successes: int
    benign_n: int
    benign_preserved: int
    run_id: str

    @property
    def asr(self) -> float:
        return self.attack_successes / self.attack_n if self.attack_n else float("nan")

    @property
    def frr(self) -> float:
        if not self.benign_n:
            return float("nan")
        return 1 - self.benign_preserved / self.benign_n


def load_cells(results_root: Path) -> list[Cell]:
    """Read every run report under ``results_root``, newest run per cell wins."""
    latest: dict[tuple[str, str], Cell] = {}
    for report_path in sorted(results_root.glob("*/report.json")):
        data = json.loads(report_path.read_text(encoding="utf-8"))
        counts = data.get("counts", {})
        cell = Cell(
            model=data.get("model_id", "?"),
            defense=data.get("defense_type", "?"),
            attack_n=counts.get("attack_scenarios", 0),
            attack_successes=counts.get("attack_successes", 0),
            benign_n=counts.get("benign_scenarios", 0),
            benign_preserved=counts.get("benign_utility_preserved", 0),
            run_id=data.get("run_id", report_path.parent.name),
        )
        latest[(cell.model, cell.defense)] = cell
    return list(latest.values())


# ── Reporting ────────────────────────────────────────────────────────────────

def print_interval_table(cells: list[Cell]) -> None:
    print("\n## Point estimates with 95% Wilson intervals\n")
    print("| Model | Defense | ASR % [95% CI] | FRR % [95% CI] | n(attack) | n(benign) |")
    print("|---|---|---|---|---|---|")
    for c in sorted(cells, key=lambda x: (x.model, x.defense)):
        a, alo, ahi = wilson_interval(c.attack_successes, c.attack_n)
        f, flo, fhi = wilson_interval(c.benign_n - c.benign_preserved, c.benign_n)
        print(
            f"| `{c.model}` | {c.defense} | {a:.1f} [{alo:.1f}, {ahi:.1f}] "
            f"| {f:.1f} [{flo:.1f}, {fhi:.1f}] | {c.attack_n} | {c.benign_n} |"
        )


def compare_defenses(cells: list[Cell], model: str | None = None) -> None:
    """Test every defense against its model's undefended baseline."""
    print("\n## Defense vs. baseline (Fisher exact, Holm-Bonferroni corrected)\n")

    by_model: dict[str, list[Cell]] = {}
    for c in cells:
        if model and c.model != model:
            continue
        by_model.setdefault(c.model, []).append(c)

    for model_name, model_cells in sorted(by_model.items()):
        baseline = next((c for c in model_cells if c.defense == "none"), None)
        if baseline is None or baseline.attack_n == 0:
            print(f"### `{model_name}` — no usable baseline, skipped\n")
            continue

        raw: dict[str, float] = {}
        details: dict[str, dict] = {}
        for c in model_cells:
            if c.defense == "none":
                continue
            p = fisher_exact_two_sided(
                baseline.attack_successes,
                baseline.attack_n - baseline.attack_successes,
                c.attack_successes,
                c.attack_n - c.attack_successes,
            )
            raw[c.defense] = p
            details[c.defense] = {
                "asr_from": baseline.asr,
                "asr_to": c.asr,
                "h": cohens_h(baseline.asr, c.asr),
            }

        if not raw:
            continue

        corrected = holm_bonferroni(raw)
        print(f"### `{model_name}` (baseline ASR = {baseline.asr * 100:.1f}%, "
              f"n = {baseline.attack_n})\n")
        print("| Defense | ASR % | Δ pp | Cohen's h | p (raw) | p (Holm) | Significant |")
        print("|---|---|---|---|---|---|---|")
        for defense, stats in sorted(corrected.items(), key=lambda kv: kv[1]["p_adjusted"]):
            d = details[defense]
            delta = (d["asr_to"] - d["asr_from"]) * 100
            print(
                f"| {defense} | {d['asr_to'] * 100:.1f} | {delta:+.1f} | {d['h']:.3f} "
                f"| {stats['p_raw']:.4f} | {stats['p_adjusted']:.4f} "
                f"| {'yes' if stats['significant'] else 'no'} |"
            )
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/runs")
    parser.add_argument("--compare", default=None, help="Restrict comparison to one model")
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        print(f"No results at {root}. Run experiments/run_benchmark.py first.", file=sys.stderr)
        return 1

    cells = load_cells(root)
    if not cells:
        print(f"No run reports found under {root}.", file=sys.stderr)
        return 1

    print(f"# Chokepoint statistical analysis\n\n{len(cells)} configuration(s) from {root}")
    print_interval_table(cells)
    compare_defenses(cells, args.compare)

    print(
        "\n> Intervals are Wilson score intervals. Between-condition tests are "
        "two-sided Fisher exact tests, appropriate at these stratum sizes where "
        "the chi-square approximation is not. p-values are Holm-Bonferroni "
        "corrected across each model's family of defense comparisons."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
