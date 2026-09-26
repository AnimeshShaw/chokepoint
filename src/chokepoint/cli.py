"""Chokepoint command-line interface.

    chokepoint validate  --suite data/scenarios/public_smoke.jsonl
    chokepoint run       --suite ... --model gpt-5.6-terra --defense type_checker
    chokepoint models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


def _cmd_validate(args: argparse.Namespace) -> int:
    from chokepoint.dataset.feasibility import check_suite
    from chokepoint.dataset.loader import SuiteValidationError, load_suite, suite_summary

    try:
        scenarios = load_suite(args.suite)
    except SuiteValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    print(f"VALID: {args.suite}")
    print(json.dumps(suite_summary(scenarios), indent=2))

    # Schema validity only means a scenario runs. Feasibility asks whether it
    # measures anything — a scenario can execute cleanly and still be inert.
    reports = [r for r in check_suite(scenarios) if r.errors or r.warnings]
    if not reports:
        print("\nFeasibility: no issues.")
        return 0

    errored = [r for r in reports if r.errors]
    warned = [r for r in reports if r.warnings and not r.errors]

    if errored:
        print(f"\nFeasibility ERRORS ({len(errored)}) — these scenarios measure nothing:")
        for report in errored:
            for error in report.errors:
                print(f"  [{report.scenario_id}] {error}")

    if warned:
        print(f"\nFeasibility warnings ({len(warned)}):")
        for report in warned:
            for warning in report.warnings:
                print(f"  [{report.scenario_id}] {warning}")

    if errored and args.strict:
        return 1
    return 0


def _build_disclosure_assessor(mode: str, model: str | None, agent_model: str):
    """Construct the disclosure assessor named by ``--disclosure``."""
    if mode == "off":
        return None
    if mode == "heuristic":
        from chokepoint.eval.disclosure import HeuristicDisclosureAssessor

        return HeuristicDisclosureAssessor()

    from chokepoint.eval.disclosure import LLMDisclosureAssessor

    chosen = model or agent_model
    if chosen == agent_model:
        print(
            "Warning: the disclosure assessor is the same model as the agent. "
            "A model reading its own transcript tends to judge its own phrasing "
            "clearer than it is. Pass --disclosure-model to use a different one.",
            file=sys.stderr,
        )
    return LLMDisclosureAssessor(model=chosen)


def _cmd_run(args: argparse.Namespace) -> int:
    from chokepoint.agents.defended_agent import DefenseType
    from chokepoint.eval.runner import EvaluationRunner

    runner = EvaluationRunner(
        suite_path=args.suite,
        results_root=args.results_root,
        timeout_s=args.timeout,
        disclosure_assessor=_build_disclosure_assessor(
            args.disclosure, args.disclosure_model, args.model
        ),
    )
    result = runner.run(
        model_id=args.model,
        defense_type=DefenseType(args.defense),
        limit=args.limit,
        offset=args.offset,
        judge_model=args.judge_model,
    )

    print("\n=== REPORT ===")
    print(json.dumps(result["report"], indent=2))
    print(f"\nRun id:  {result['run_id']}")
    print(f"Traces:  {Path(result['report_path']).parent / 'traces'}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    from chokepoint.agents.model_config import list_available_models

    registry = list_available_models()
    for tier in ("frontier", "local"):
        print(f"\n{tier.title()} models:")
        for key, meta in registry[tier].items():
            print(f"  {key:24s} -> {meta['provider']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chokepoint", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Check that a suite is executable and measurable")
    p_val.add_argument("--suite", required=True)
    p_val.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on feasibility errors, not just schema failures",
    )
    p_val.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="Evaluate one model/defense configuration")
    p_run.add_argument("--suite", required=True)
    p_run.add_argument("--model", required=True)
    p_run.add_argument(
        "--defense",
        default="none",
        choices=["none", "type_checker", "capability_router", "llm_judge",
                 "llm_judge+type_checker", "all"],
    )
    p_run.add_argument("--judge-model", dest="judge_model", default=None)
    p_run.add_argument(
        "--disclosure",
        default="heuristic",
        choices=["off", "heuristic", "llm"],
        help=(
            "Measure whether the agent told the user about anomalous actions. "
            "'heuristic' is free and offline but under-reports disclosure on "
            "paraphrase; 'llm' is the one to use for reported figures."
        ),
    )
    p_run.add_argument(
        "--disclosure-model",
        dest="disclosure_model",
        default=None,
        help="Assessor model for --disclosure llm. Should differ from --model.",
    )
    p_run.add_argument("--limit", type=int, default=0)
    p_run.add_argument("--offset", type=int, default=0)
    p_run.add_argument("--timeout", type=int, default=120)
    p_run.add_argument("--results-root", dest="results_root", default="results/runs")
    p_run.set_defaults(func=_cmd_run)

    p_models = sub.add_parser("models", help="List the model registry")
    p_models.set_defaults(func=_cmd_models)

    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
