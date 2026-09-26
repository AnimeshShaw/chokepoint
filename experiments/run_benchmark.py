"""Full benchmark sweep: every model x every defense over one suite.

    python experiments/run_benchmark.py --suite data/scenarios/public_smoke.jsonl
    python experiments/run_benchmark.py --config configs/experiment.yaml

Each (model, defense) cell is an independent run with its own manifest, traces,
and report under ``results/runs/``. Completed cells are skipped on re-invocation
so an interrupted sweep resumes rather than restarting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chokepoint.agents.defended_agent import DefenseType  # noqa: E402
from chokepoint.dataset.loader import SuiteValidationError  # noqa: E402
from chokepoint.eval.runner import EvaluationRunner  # noqa: E402

DEFAULT_MODELS = ["gpt-5.6-terra", "llama3.1:8b"]
DEFAULT_DEFENSES = ["none", "type_checker", "capability_router", "llm_judge"]

#: A completed cell with more than this fraction of errored scenarios is treated
#: as invalid (an infrastructure outage, not a result) and re-run.
ERROR_FRACTION_INVALID = 0.25


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def already_done(
    results_root: Path, model: str, defense: str, suite_digest: str, expected_total: int
) -> Path | None:
    """Find a completed run for this cell against this exact suite revision.

    A run only counts as done if it covered the whole suite. A partial run
    (from a --limit probe) must not satisfy the cache, or it silently poisons
    the full sweep with a truncated result.
    """
    safe = model.replace(":", "_").replace("/", "_")
    for run_dir in sorted(results_root.glob(f"{safe}__{defense}__*")):
        report = run_dir / "report.json"
        if not report.exists():
            continue
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        manifest = data.get("manifest", {})
        counts = data.get("counts", {})
        covered = counts.get("total_scenarios", 0)
        errored = counts.get("errored_scenarios", 0)
        # A cell poisoned by an infrastructure outage (e.g. the local model
        # server dropping mid-run) reaches n=95 but is all errors. Do not treat
        # a run as done if a large fraction errored — re-run it instead.
        if covered and errored / covered > ERROR_FRACTION_INVALID:
            continue
        if manifest.get("suite_sha256") == suite_digest and covered >= expected_total:
            return run_dir
    return None


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="public_smoke")
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--defenses", nargs="*", default=None)
    parser.add_argument("--judge-model", dest="judge_model", default=None)
    parser.add_argument("--results-root", dest="results_root", default="results/runs")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-run completed cells")
    args = parser.parse_args()

    config = load_config(args.config)
    models = args.models or config.get("models") or DEFAULT_MODELS
    defenses = args.defenses or config.get("defenses") or DEFAULT_DEFENSES
    results_root = Path(args.results_root)

    try:
        runner = EvaluationRunner(args.suite, results_root=results_root, timeout_s=args.timeout)
    except SuiteValidationError as exc:
        print(f"Suite is not runnable:\n{exc}", file=sys.stderr)
        return 1

    from chokepoint.dataset.loader import suite_digest as digest_of

    suite_digest = digest_of(args.suite)
    print(f"Suite {args.suite} ({len(runner.scenarios)} scenarios, sha256={suite_digest[:12]})")
    print(f"Grid: {len(models)} model(s) x {len(defenses)} defense(s)\n")

    rows = []
    for model in models:
        for defense in defenses:
            expected_total = len(runner.scenarios) if args.limit == 0 else args.limit
            existing = (
                None
                if args.force
                else already_done(results_root, model, defense, suite_digest, expected_total)
            )
            if existing:
                print(f"[skip] {model} / {defense} -> {existing.name}")
                rows.append(json.loads((existing / "report.json").read_text(encoding="utf-8")))
                continue

            print(f"[run ] {model} / {defense}")
            try:
                result = runner.run(
                    model_id=model,
                    defense_type=DefenseType(defense),
                    judge_model=args.judge_model,
                    limit=args.limit,
                )
                rows.append(result["report"])
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] {model} / {defense}: {type(exc).__name__}: {exc}", file=sys.stderr)

    _print_table(rows)
    return 0


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("\nNo results.")
        return
    print("\n| Model | Defense | ASR % | FRR % | UPR(benign) % | UPR(attack) % | Balanced Acc |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        def fmt(v):
            return "n/a" if v is None else f"{v:.1f}"

        print(
            f"| {r.get('model_id')} | {r.get('defense_type')} | {fmt(r.get('asr_percentage'))} "
            f"| {fmt(r.get('frr_percentage'))} | {fmt(r.get('upr_benign_percentage'))} "
            f"| {fmt(r.get('upr_attack_percentage'))} | {fmt(r.get('balanced_accuracy'))} |"
        )


if __name__ == "__main__":
    raise SystemExit(main())
