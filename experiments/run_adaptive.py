"""Adaptive-attack sweep.

Evaluates the adaptive suite, whose payloads are written with knowledge of the
defenses they face — the "attacker moves second" standard for defense
evaluation. A defense that only holds against attacks written before it existed
has not been evaluated.

    python experiments/run_adaptive.py                       # pulls "adaptive" from HF
    python experiments/run_adaptive.py --suite my_local.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.run_benchmark import main as benchmark_main  # noqa: E402

ADAPTIVE_DEFAULTS = ["none", "type_checker", "llm_judge", "llm_judge+type_checker"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--suite", default="adaptive")
    known, rest = parser.parse_known_args()

    argv = ["--suite", known.suite]
    if "--defenses" not in rest:
        argv += ["--defenses", *ADAPTIVE_DEFAULTS]
    sys.argv = [sys.argv[0]] + argv + rest
    return benchmark_main()


if __name__ == "__main__":
    raise SystemExit(main())
