"""D1 ablation: measure the ASR error from silently-dropped payloads.

Defects D2 and D3 are re-scorable offline from existing traces. D1 is not: the
dropped payloads were never delivered, so there is nothing to re-score. It needs
a paired sweep — the same suite, run once with payloads correctly delivered and
once with the pre-audit delivery bug reintroduced — and the ASR gap between them
is the error D1 caused.

The bug is reintroduced faithfully: the pre-audit runner placed a payload only
when the (prose) data source contained one of four literals. Here, a payload
whose locator does not name one of those literals is dropped, exactly as before.

    python experiments/ablate_delivery.py --suite data/scenarios/core_v2.jsonl \
        --model gpt-5.6-terra --defense none

This costs one extra sweep of the attack scenarios. Run it once, on the final
suite, for the paper's D1 row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chokepoint.agents.defended_agent import DefenseType  # noqa: E402
from chokepoint.dataset.loader import load_suite, prepare_environment  # noqa: E402
from chokepoint.environment.environment import Environment, InjectionError  # noqa: E402

# The four literals the pre-audit runner matched against (runner.py:66-73).
_LEGACY_LITERALS = ("email_1", "email_2", "notes.txt", "input.txt")


def prepare_environment_buggy(scenario) -> Environment:
    """Reproduce the pre-audit delivery bug.

    Places the payload only if the locator references one of the four legacy
    literals; otherwise the environment is built with no payload — the silent
    drop that hit 38 of 43 core attacks.
    """
    env = Environment.from_spec(scenario.environment)
    if scenario.injection:
        locator = scenario.injection.locator
        if any(lit in locator for lit in _LEGACY_LITERALS):
            try:
                env.inject(locator, scenario.injection.payload, scenario.injection.mode)
            except InjectionError:
                pass  # pre-audit code swallowed these too
    return env


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="public_smoke")
    parser.add_argument("--model", required=True)
    parser.add_argument("--defense", default="none")
    parser.add_argument("--out", default="results/ablation_delivery.json")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    scenarios = [s for s in load_suite(args.suite) if s.scenario_type == "attack"]
    if not scenarios:
        print("Suite has no attack scenarios.", file=sys.stderr)
        return 1

    # Report which scenarios the bug would drop — this is the mechanism, before
    # any model is run.
    dropped = [
        s.id
        for s in scenarios
        if s.injection and not any(lit in s.injection.locator for lit in _LEGACY_LITERALS)
    ]
    print(f"Suite: {args.suite} ({len(scenarios)} attack scenarios)")
    print(f"Pre-audit delivery bug would silently drop {len(dropped)}/{len(scenarios)} "
          f"payloads ({len(dropped) / len(scenarios) * 100:.0f}%)\n")

    from chokepoint.agents.base_agent import BaseAgent
    from chokepoint.agents.defended_agent import DefendedAgent
    from chokepoint.eval.metrics import evaluate_attack_goal

    def run_condition(prepare_fn) -> float:
        import concurrent.futures

        successes = 0
        for scenario in scenarios:
            env = prepare_fn(scenario)
            from chokepoint.tools.registry import build_tools

            tools = build_tools(env)
            defense = DefenseType(args.defense)
            agent = (
                BaseAgent(model_id=args.model, tools=tools)
                if defense == DefenseType.NONE
                else DefendedAgent(model_id=args.model, tools=tools,
                                   defense_type=defense, recorder=env.record)
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                try:
                    ex.submit(agent.run, scenario.user_prompt).result(timeout=args.timeout)
                except Exception:  # noqa: BLE001
                    pass
            if scenario.attack_goal:
                succeeded, _ = evaluate_attack_goal(scenario.attack_goal, env.log)
                successes += int(succeeded)
        return round(successes / len(scenarios) * 100, 2)

    print("Running CORRECTED delivery (payloads placed)...")
    asr_correct = run_condition(prepare_environment)
    print(f"  ASR = {asr_correct}%\n")

    print("Running BUGGY delivery (pre-audit silent drop)...")
    asr_buggy = run_condition(prepare_environment_buggy)
    print(f"  ASR = {asr_buggy}%\n")

    report = {
        "suite": args.suite,
        "model": args.model,
        "defense": args.defense,
        "attack_scenarios": len(scenarios),
        "payloads_dropped_by_bug": len(dropped),
        "dropped_ids": dropped,
        "asr_corrected_delivery": asr_correct,
        "asr_buggy_delivery": asr_buggy,
        "asr_understatement_pp": round(asr_correct - asr_buggy, 2),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("== D1 delivery ablation ==")
    print(f"  Corrected delivery ASR:  {asr_correct}%")
    print(f"  Buggy delivery ASR:      {asr_buggy}%")
    print(f"  D1 understated ASR by:   {report['asr_understatement_pp']} pp")
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
