"""Run the iterative adaptive attack against each defense and report the curve.

Replaces the one-shot adaptive generation that produced the indefensible
0%-ASR / "complete mitigation" claim. Here the attacker optimizes against the
live defense round by round, and the result is an ASR-versus-round curve. A
defense whose curve rises across rounds was untested, not robust.

    python experiments/run_adaptive_iterative.py \
        --suite data/scenarios/adaptive_v2.jsonl \
        --model gpt-5.6-terra --judge-model claude-sonnet-5 \
        --attacker-model claude-sonnet-5 --defenses type_checker llm_judge \
        --max-rounds 6

The attacker model should differ from both the agent and the judge: an attacker
sharing the judge's weights has information a real adversary would not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from chokepoint.agents.defended_agent import DefenseType  # noqa: E402
from chokepoint.agents.model_config import get_llm  # noqa: E402
from chokepoint.attacks.adaptive import (  # noqa: E402
    ATTACKER_SYSTEM_PROMPT,
    ATTACKER_USER_TEMPLATE,
    PayloadProposal,
    asr_by_round,
    run_adaptive_attack,
)
from chokepoint.dataset.loader import load_suite  # noqa: E402


def build_attacker(model: str, temperature: float):
    llm = get_llm(model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [("system", ATTACKER_SYSTEM_PROMPT), ("human", ATTACKER_USER_TEMPLATE)]
    )
    return prompt | llm.with_structured_output(PayloadProposal)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="public_smoke")
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-model", dest="judge_model", default=None)
    parser.add_argument("--attacker-model", dest="attacker_model", default=None)
    parser.add_argument("--attacker-temperature", dest="attacker_temp", type=float, default=0.9)
    parser.add_argument("--defenses", nargs="*", default=["type_checker", "llm_judge"])
    parser.add_argument("--max-rounds", dest="max_rounds", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--out", default="results/adaptive_iterative.json")
    args = parser.parse_args()

    scenarios = [s for s in load_suite(args.suite) if s.scenario_type == "attack"]
    if not scenarios:
        print("Suite has no attack scenarios.", file=sys.stderr)
        return 1

    attacker_model = args.attacker_model or args.model
    if attacker_model in (args.model, args.judge_model):
        print(
            "Warning: the attacker model matches the agent or judge. A real "
            "adversary would not share their weights; results will understate "
            "the achievable ASR.",
            file=sys.stderr,
        )
    attacker = build_attacker(attacker_model, args.attacker_temp)

    report = {
        "suite": args.suite,
        "model": args.model,
        "judge_model": args.judge_model or args.model,
        "attacker_model": attacker_model,
        "max_rounds": args.max_rounds,
        "defenses": {},
    }

    for defense_name in args.defenses:
        defense = DefenseType(defense_name)
        print(f"\n=== Adaptive attack vs {defense_name} ===")
        trajectories = []
        for scenario in scenarios:
            traj = run_adaptive_attack(
                scenario, args.model, defense, attacker,
                max_rounds=args.max_rounds, judge_model=args.judge_model,
                timeout_s=args.timeout,
            )
            trajectories.append(traj)
            status = (
                f"broken at round {traj.rounds_to_success}"
                if traj.succeeded
                else f"held {len(traj.rounds)} rounds"
            )
            print(f"  {scenario.id}: {status}")

        curve = asr_by_round(trajectories, args.max_rounds)
        report["defenses"][defense_name] = {
            "curve": curve,
            "final_asr": curve[-1]["cumulative_asr"] if curve else None,
            "trajectories": [t.to_dict() for t in trajectories],
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n== Adaptive ASR by round ==")
    header = "| Defense | " + " | ".join(f"r{k}" for k in range(args.max_rounds)) + " |"
    print(header)
    print("|" + "---|" * (args.max_rounds + 1))
    for name, data in report["defenses"].items():
        cells = " | ".join(f"{p['cumulative_asr']:.0f}%" for p in data["curve"])
        print(f"| {name} | {cells} |")
    print(
        "\n> A curve that climbs across rounds is the finding: the defense was "
        "untested, not robust. A flat low curve is a real robustness claim — "
        "provided the attacker model is competent and independent of the defense."
    )
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
