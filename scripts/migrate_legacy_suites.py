"""Migrate the archived schema-v1 suites to schema v2 skeletons.

Run: ``python scripts/migrate_legacy_suites.py``

Output lands in ``data/scenarios/*.v2-draft.jsonl``. Drafts are not runnable
until their environments are authored — the validator rejects them by design.
"""

from __future__ import annotations

import json
from pathlib import Path

from chokepoint.dataset.migrate import migrate_file

JOBS = [
    ("data/scenarios/core_v1.legacy.jsonl", "data/scenarios/core_v1.v2-draft.jsonl", "core"),
    ("data/scenarios/adaptive_v1.legacy.jsonl", "data/scenarios/adaptive_v1.v2-draft.jsonl", "adaptive"),
]


def main() -> None:
    for src, dest, suite in JOBS:
        if not Path(src).exists():
            print(f"skip (missing): {src}")
            continue
        summary = migrate_file(src, dest, suite)
        print(json.dumps(summary, indent=2))
        print()

    print(
        "Drafts written. Each scenario needs an authored `environment` block "
        "before it will pass validation; see docs/authoring_scenarios.md."
    )


if __name__ == "__main__":
    main()
