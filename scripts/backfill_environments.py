"""Derive schema-v2 environments for migrated scenario drafts.

    # dry-run on 5 scenarios first — always do this before a full pass
    python scripts/backfill_environments.py \
        --src data/scenarios/core_v1.v2-draft.jsonl \
        --dest data/scenarios/core_v2.jsonl --limit 5

    # full suite
    python scripts/backfill_environments.py \
        --src data/scenarios/core_v1.v2-draft.jsonl \
        --dest data/scenarios/core_v2.jsonl

Only scenarios that pass schema validation, injection resolution, and the
feasibility checks are written. Failures are reported with their rejection
reasons so they can be authored by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from chokepoint.agents.model_config import get_llm  # noqa: E402
from chokepoint.dataset.backfill import (  # noqa: E402
    BACKFILL_SYSTEM_PROMPT,
    BACKFILL_USER_TEMPLATE,
    EnvironmentProposal,
    backfill_file,
)


def build_chain(model: str, temperature: float):
    llm = get_llm(model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [("system", BACKFILL_SYSTEM_PROMPT), ("human", BACKFILL_USER_TEMPLATE)]
    )
    return prompt | llm.with_structured_output(EnvironmentProposal)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/scenarios/core_v1.v2-draft.jsonl")
    parser.add_argument("--dest", default="data/scenarios/core_v2.jsonl")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-retries", dest="max_retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Backfill only the first N drafts")
    parser.add_argument("--report", default=None, help="Write the full report to this JSON path")
    args = parser.parse_args()

    if not Path(args.src).exists():
        print(
            f"Draft suite not found: {args.src}\n"
            "Run scripts/migrate_legacy_suites.py first.",
            file=sys.stderr,
        )
        return 1

    summary = backfill_file(
        src=args.src,
        dest=args.dest,
        chain=build_chain(args.model, args.temperature),
        max_retries=args.max_retries,
        limit=args.limit,
    )

    print("\n== Backfill summary ==")
    for key in ("drafts", "accepted", "failed", "with_warnings"):
        print(f"  {key:16s} {summary[key]}")

    if summary["failures"]:
        print(f"\n== Rejected ({len(summary['failures'])}) — author these by hand ==")
        for failure in summary["failures"][:15]:
            print(f"  {failure['id']}")
            for attempt in failure["attempts"][-1:]:
                print(f"      {attempt}")

    if summary["warnings"]:
        print(f"\n== Accepted with warnings ({len(summary['warnings'])}) ==")
        for warning in summary["warnings"][:15]:
            print(f"  {warning['id']}: {warning['warnings'][0]}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nFull report: {args.report}")

    print(f"\nValidate before use:\n  chokepoint validate --suite {args.dest}")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
