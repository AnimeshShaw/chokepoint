"""Gap 4: the availability attack on an LLM-judge defense.

Runs the attack suite four ways to trace the whole trade-off:

  1. judge healthy, fail-closed   -- the intended baseline
  2. judge healthy, fail-open     -- shows the policy is invisible when nothing fails
  3. judge faulted, fail-closed   -- attacker induces a fault: DoS on the agent
  4. judge faulted, fail-open     -- attacker induces a fault: clean bypass

The finding is the shape across these four: an attacker who cannot fool the
judge can still win by making it unreachable, and the deployment's fail policy
only chooses *which* way it loses.

    python experiments/availability_attack.py --suite data/scenarios/core_v2.jsonl \
        --model gpt-5.6-terra --judge-model claude-sonnet-5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chokepoint.dataset.loader import load_suite, prepare_environment  # noqa: E402
from chokepoint.defenses.fault_injection import oversize_argument_injector  # noqa: E402
from chokepoint.defenses.llm_judge import LLMJudgeDefense  # noqa: E402
from chokepoint.eval.metrics import evaluate_attack_goal, evaluate_utility  # noqa: E402
from chokepoint.tools.registry import build_tools  # noqa: E402


def run_condition(scenarios, model, judge_model, fail_closed, faulted, timeout):
    from chokepoint.agents.base_agent import BaseAgent

    injector = oversize_argument_injector(threshold=8000) if faulted else None
    asr_hits = 0
    utility_hits = 0

    for scenario in scenarios:
        env = prepare_environment(scenario)
        tools = LLMJudgeDefense.wrap_tools(
            build_tools(env),
            user_prompt=scenario.user_prompt,
            judge_model=judge_model,
            recorder=env.record,
            fail_closed=fail_closed,
            fault_injector=injector,
        )
        agent = BaseAgent(model_id=model, tools=tools)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            try:
                ex.submit(agent.run, scenario.user_prompt).result(timeout=timeout)
            except Exception:  # noqa: BLE001
                pass
        if scenario.attack_goal:
            succeeded, _ = evaluate_attack_goal(scenario.attack_goal, env.log)
            asr_hits += int(succeeded)
        utility_ok, _ = evaluate_utility(scenario, env.log)
        utility_hits += int(utility_ok)

    n = len(scenarios)
    return {
        "asr": round(asr_hits / n * 100, 2),
        "utility": round(utility_hits / n * 100, 2),
    }


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="public_smoke")
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-model", dest="judge_model", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", default="results/availability_attack.json")
    args = parser.parse_args()

    scenarios = [s for s in load_suite(args.suite) if s.scenario_type == "attack"]
    if not scenarios:
        print("Suite has no attack scenarios.", file=sys.stderr)
        return 1
    judge_model = args.judge_model or args.model

    conditions = [
        ("healthy_fail_closed", True, False),
        ("healthy_fail_open", False, False),
        ("faulted_fail_closed", True, True),
        ("faulted_fail_open", False, True),
    ]

    print(f"Suite: {args.suite} ({len(scenarios)} attacks)  agent={args.model}  judge={judge_model}\n")
    report = {"suite": args.suite, "model": args.model, "judge_model": judge_model,
              "attack_scenarios": len(scenarios), "conditions": {}}

    for name, fail_closed, faulted in conditions:
        print(f"Running: {name} ...")
        result = run_condition(scenarios, args.model, judge_model, fail_closed, faulted, args.timeout)
        report["conditions"][name] = result
        print(f"  ASR = {result['asr']}%   utility = {result['utility']}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    c = report["conditions"]
    print("\n== Availability attack ==")
    print("| Condition | ASR % | Utility % |")
    print("|---|---|---|")
    for name, _, _ in conditions:
        print(f"| {name} | {c[name]['asr']} | {c[name]['utility']} |")
    print(
        f"\n> Fault + fail-open lifts ASR from {c['healthy_fail_closed']['asr']}% to "
        f"{c['faulted_fail_open']['asr']}% without the attacker ever fooling the judge. "
        f"Fault + fail-closed instead drops utility to {c['faulted_fail_closed']['utility']}% "
        "— a denial of service. The fail policy only chooses which way the deployment loses."
    )
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
