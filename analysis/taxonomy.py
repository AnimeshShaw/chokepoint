"""Scenario suite composition and attack taxonomy.

    python analysis/taxonomy.py --suite data/scenarios/public_smoke.jsonl

Target-domain counts are multi-label: one scenario whose attacker objective
spans a SQL read and an email exfiltration counts under both. The pre-audit
version reported these as if they were a partition, which is why its figures
summed past 100%.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chokepoint.dataset.loader import load_suite  # noqa: E402

DOMAINS = {
    "database": {"execute_sql_query"},
    "filesystem": {"read_file", "write_file"},
    "email": {"read_email", "send_email"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="public_smoke")
    args = parser.parse_args()

    scenarios = load_suite(args.suite)
    attacks = [s for s in scenarios if s.scenario_type == "attack"]
    benign = [s for s in scenarios if s.scenario_type == "benign"]
    total = len(scenarios)

    print(f"# Suite composition: {args.suite}\n")
    print(f"Total scenarios: {total}")
    print(f"  benign: {len(benign)} ({len(benign) / total * 100:.1f}%)")
    print(f"  attack: {len(attacks)} ({len(attacks) / total * 100:.1f}%)")

    if attacks:
        multi = sum(1 for s in attacks if s.is_multi_hop)
        print("\n## Attack structure\n")
        print(f"  multi-hop:  {multi} ({multi / len(attacks) * 100:.1f}%)")
        print(f"  single-hop: {len(attacks) - multi} "
              f"({(len(attacks) - multi) / len(attacks) * 100:.1f}%)")

        print("\n## Injection channel\n")
        for channel, count in Counter(
            s.injection.channel for s in attacks if s.injection
        ).most_common():
            print(f"  {channel:12s} {count:3d} ({count / len(attacks) * 100:.1f}%)")

        print("\n## Attacker target domain (multi-label; rows may exceed 100%)\n")
        for domain, tools in DOMAINS.items():
            count = sum(
                1
                for s in attacks
                if s.attack_goal
                and any(c.tool in tools for c in s.attack_goal.success_conditions)
            )
            print(f"  {domain:12s} {count:3d} ({count / len(attacks) * 100:.1f}%)")

        tags = Counter(t for s in attacks for t in s.tags)
        if tags:
            print("\n## Tags\n")
            for tag, count in tags.most_common():
                print(f"  {tag:24s} {count}")

    if benign:
        print("\n## Benign task shape\n")
        counter = Counter(len(s.utility_goal.required_tools) for s in benign)
        for n_tools, count in sorted(counter.items()):
            print(f"  {n_tools} required tool(s): {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
