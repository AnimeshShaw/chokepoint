"""Resumable overnight orchestration of the full remaining experimental program.

Runs everything not yet complete, in priority order, skipping finished work so it
can be killed and relaunched freely (the run cache keys on the suite digest and
rejects outage-poisoned cells). Safe to run either as a Claude background task or
directly in a terminal:

    python experiments/run_overnight.py            # run all remaining
    python experiments/run_overnight.py --only grid   # just the model grid
    python experiments/run_overnight.py --dry-run     # print the plan

Design choices baked in:
  * Local models are evaluated on the 55-scenario reduced suite (feasible
    overnight at ~30 s/scenario).
  * The LLM-judge for a LOCAL agent is the FRONTIER model (gpt-5.6-terra), not
    the local model judging itself: faster, and methodologically correct (the
    judge must be independent of the agent -- our own Sec. on judge design).
  * The adaptive attacker and the availability judge are claude-sonnet-5, again
    for independence from the gpt agent.
  * Disclosure is scored offline afterward (heuristic), so grids stay cheap.

Progress is logged to results/overnight.log and echoed to stdout.
"""

from __future__ import annotations

import argparse
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REDUCED = "data/scenarios/core_reduced.jsonl"
CORE = "data/scenarios/core_v2.jsonl"
ADAPTIVE = "data/scenarios/adaptive_v2.jsonl"
CHANNEL = "data/scenarios/channel_asymmetry.jsonl"
LOG = ROOT / "results" / "overnight.log"

FRONTIER = "gpt-5.6-terra"
INDEP = "claude-sonnet-5"   # independent judge / attacker
LOCALS = ["llama3.1:8b", "gemma4:12b"]
DEFENSES = ["none", "type_checker", "capability_router", "llm_judge"]


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        return True
    except Exception:
        return False


def run(cmd: list[str], label: str, dry: bool) -> bool:
    log(f"START {label}")
    if dry:
        log(f"  (dry-run) {' '.join(cmd)}")
        return True
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=6 * 3600)
        ok = proc.returncode == 0
        tail = (proc.stdout or "")[-400:].replace("\r", "")
        log(f"  {'OK' if ok else 'FAIL'} {label} in {time.time()-t0:.0f}s")
        if tail.strip():
            log("  tail: " + tail.strip().replace("\n", " | ")[-300:])
        if not ok:
            log("  stderr: " + (proc.stderr or "")[-300:].replace("\n", " | "))
        return ok
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT {label}")
        return False


def phase_grid(dry: bool) -> None:
    """Complete the local-model grid on the reduced suite."""
    if not ollama_up():
        log("Ollama is DOWN -- skipping local grid. Start `ollama serve` and re-run.")
        return
    for model in LOCALS:
        # Fast deterministic defenses first, then the judge cell (frontier judge).
        run(["python", "experiments/run_benchmark.py",
             "--suite", REDUCED, "--models", model,
             "--defenses", "none", "type_checker", "capability_router",
             "--timeout", "90"],
            f"grid {model} (deterministic defenses)", dry)
        run(["python", "experiments/run_benchmark.py",
             "--suite", REDUCED, "--models", model,
             "--defenses", "llm_judge",
             "--judge-model", FRONTIER, "--timeout", "120"],
            f"grid {model} (llm_judge, frontier judge)", dry)


def phase_channel(dry: bool) -> None:
    """Channel-asymmetry sweep (matched payloads across email/file/sql)."""
    if not Path(ROOT / CHANNEL).exists():
        run(["python", "scripts/build_channel_suite.py"], "build channel suite", dry)
    run(["python", "experiments/run_benchmark.py",
         "--suite", CHANNEL, "--models", FRONTIER, "--defenses", "none",
         "--timeout", "90"],
        "channel sweep (frontier)", dry)
    if ollama_up():
        run(["python", "experiments/run_benchmark.py",
             "--suite", CHANNEL, "--models", "llama3.1:8b", "--defenses", "none",
             "--timeout", "90"],
            "channel sweep (llama)", dry)


def phase_availability(dry: bool) -> None:
    """Availability attack on the LLM-judge (fail-open vs fail-closed)."""
    run(["python", "experiments/availability_attack.py",
         "--suite", REDUCED, "--model", FRONTIER, "--judge-model", INDEP,
         "--timeout", "90", "--out", "results/availability_attack.json"],
        "availability attack (frontier agent, claude judge)", dry)


def phase_adaptive(dry: bool) -> None:
    """Iterative adaptive attacks -> ASR-by-round curves."""
    suite = ADAPTIVE if Path(ROOT / ADAPTIVE).exists() else REDUCED
    run(["python", "experiments/run_adaptive_iterative.py",
         "--suite", suite, "--model", FRONTIER,
         "--judge-model", FRONTIER, "--attacker-model", INDEP,
         "--defenses", "type_checker", "llm_judge",
         "--max-rounds", "5", "--timeout", "120",
         "--out", "results/adaptive_iterative.json"],
        "iterative adaptive (frontier agent, claude attacker)", dry)


def phase_analyze(dry: bool) -> None:
    """Recompute every analysis over the combined results."""
    run(["python", "analysis/disclosure_backfill.py", "--results-root", "results/runs"],
        "disclosure (offline)", dry)
    for a in ["ablation", "statistics", "capability", "judge_roc", "channel"]:
        run(["python", f"analysis/{a}.py", "--results-root", "results/runs"],
            f"analysis: {a}", dry)


PHASES = {
    "grid": phase_grid,
    "channel": phase_channel,
    "availability": phase_availability,
    "adaptive": phase_adaptive,
    "analyze": phase_analyze,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=list(PHASES), default=None,
                        help="Run only these phases (default: all, in order).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    order = args.only or ["grid", "channel", "availability", "adaptive", "analyze"]
    log("=" * 60)
    log(f"Overnight run start. Phases: {order}. Ollama: {'up' if ollama_up() else 'DOWN'}")
    for name in order:
        log(f"--- phase: {name} ---")
        PHASES[name](args.dry_run)
    log("Overnight run complete.")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
