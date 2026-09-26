"""Statistical power for the comparisons this benchmark wants to make.

At n=43 attack scenarios, a 95% Wilson interval near 50% is roughly +/-15
percentage points — wider than most between-defense differences of interest. So
before spending effort generating scenarios, this answers: how many are needed
to detect an effect of a given size, and what can the current suite actually
resolve?

Uses a two-proportion test (Fisher-consistent normal approximation for the power
calculation, which is standard; the reported analysis itself uses exact Fisher).

    python analysis/power.py --baseline 0.60 --target 0.35
    python analysis/power.py --scan
"""

from __future__ import annotations

import math

# Standard-normal quantiles for common alpha/power without SciPy.
_Z = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}


def _z(p: float) -> float:
    if p in _Z:
        return _Z[p]
    # Acklam's inverse-normal approximation for arbitrary quantiles.
    a = [-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def required_n(
    p1: float, p2: float, alpha: float = 0.05, power: float = 0.80
) -> int | None:
    """Per-group n to detect a p1-vs-p2 difference at the given alpha and power.

    Two-sided, equal group sizes. Returns None when the effect is zero.
    """
    if p1 == p2:
        return None
    z_alpha = _z(1 - alpha / 2)
    z_beta = _z(power)
    pbar = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * pbar * (1 - pbar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(numerator / (p1 - p2) ** 2)


def achieved_power(
    p1: float, p2: float, n: int, alpha: float = 0.05
) -> float:
    """Power to detect a p1-vs-p2 difference at group size n."""
    if p1 == p2 or n <= 0:
        return 0.0
    z_alpha = _z(1 - alpha / 2)
    pbar = (p1 + p2) / 2
    se_null = math.sqrt(2 * pbar * (1 - pbar) / n)
    se_alt = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    z = (abs(p1 - p2) - z_alpha * se_null) / se_alt
    return round(_normal_cdf(z), 4)


def minimum_detectable_effect(
    baseline: float, n: int, alpha: float = 0.05, power: float = 0.80
) -> float | None:
    """Smallest absolute ASR reduction detectable at n, by search."""
    for delta_pp in range(1, int(baseline * 100) + 1):
        target = baseline - delta_pp / 100
        if target < 0:
            break
        if achieved_power(baseline, target, n, alpha) >= power:
            return round(delta_pp / 100, 2)
    return None


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def scan(
    baseline: float, n_values: list[int], alpha: float = 0.05, power: float = 0.80
) -> list[dict]:
    """Minimum detectable ASR reduction across candidate suite sizes."""
    rows = []
    for n in n_values:
        mde = minimum_detectable_effect(baseline, n, alpha, power)
        rows.append(
            {
                "n_per_group": n,
                "minimum_detectable_reduction_pp": None if mde is None else round(mde * 100, 1),
            }
        )
    return rows


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=float, default=0.60,
                        help="Baseline (undefended) ASR as a proportion.")
    parser.add_argument("--target", type=float, default=None,
                        help="Defended ASR to detect; if set, reports required n.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--scan", action="store_true",
                        help="Report minimum detectable effect across suite sizes.")
    args = parser.parse_args()

    print("# Statistical power\n")
    print(f"Baseline ASR: {args.baseline:.0%}   alpha: {args.alpha}   power: {args.power}\n")

    if args.target is not None:
        n = required_n(args.baseline, args.target, args.alpha, args.power)
        delta = (args.baseline - args.target) * 100
        print(f"To detect {args.baseline:.0%} -> {args.target:.0%} "
              f"(a {delta:.0f} pp reduction):")
        print(f"  required attack scenarios per condition: {n}")
        print(f"  power of the current 43-scenario suite: "
              f"{achieved_power(args.baseline, args.target, 43, args.alpha):.0%}")

    if args.scan or args.target is None:
        print("\n## Minimum detectable ASR reduction by suite size\n")
        print("| n (attack scenarios) | smallest detectable reduction |")
        print("|---|---|")
        for row in scan(args.baseline, [25, 43, 60, 90, 120, 150, 200], args.alpha, args.power):
            mde = row["minimum_detectable_reduction_pp"]
            print(f"| {row['n_per_group']} | "
                  f"{'not detectable' if mde is None else f'{mde} pp'} |")
        print(
            "\n> Read this before sizing the suite. If the defenses you compare "
            "differ by less than the row for your chosen n, the suite cannot "
            "resolve them and a null result would be uninformative."
        )
    return 0
