"""Run manifests and per-scenario trace persistence.

The pre-audit harness wrote only aggregate percentages and discarded every
execution trace, so no published number could be audited or reproduced. Every
run now emits a manifest plus one trace file per scenario.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chokepoint.environment.environment import Environment
from chokepoint.eval.metrics import ScenarioOutcome


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_manifest(
    model_id: str,
    defense: str,
    suite_path: str,
    suite_digest: str,
    suite_summary: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance record for a run."""
    manifest = {
        "model_id": model_id,
        "defense": defense,
        "suite_path": str(suite_path),
        "suite_sha256": suite_digest,
        "suite_summary": suite_summary,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def run_id_for(manifest: dict[str, Any]) -> str:
    """Stable identifier derived from the run's defining configuration.

    Timestamp and git commit are excluded so that two runs of the same
    configuration are recognisable as such, and the timestamp suffix keeps them
    from overwriting one another.
    """
    key = {
        "model_id": manifest["model_id"],
        "defense": manifest["defense"],
        "suite_sha256": manifest["suite_sha256"],
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    stamp = manifest["timestamp_utc"].replace(":", "").replace("-", "")[:15]
    safe_model = manifest["model_id"].replace(":", "_").replace("/", "_")
    return f"{safe_model}__{manifest['defense']}__{digest}__{stamp}"


class TraceWriter:
    """Writes a run's manifest, per-scenario traces, and final report."""

    def __init__(self, root: str | Path, manifest: dict[str, Any]) -> None:
        self.run_id = run_id_for(manifest)
        self.dir = Path(root) / self.run_id
        self.traces_dir = self.dir / "traces"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self._write(self.dir / "manifest.json", manifest)

    def write_trace(
        self,
        scenario_id: str,
        user_prompt: str,
        env: Environment,
        outcome: ScenarioOutcome,
        agent_output: str,
        defense_logs: dict[str, Any],
        injection: dict[str, Any] | None = None,
        scenario: dict[str, Any] | None = None,
    ) -> None:
        """Persist the full evidence record for one scenario.

        ``scenario`` carries the attacker predicate and utility goal so that
        offline analyses — judge calibration in particular — can recover ground
        truth from the trace alone, without re-reading the suite that produced
        it. Suites get edited; traces must stay self-describing.
        """
        self._write(
            self.traces_dir / f"{scenario_id}.json",
            {
                "scenario_id": scenario_id,
                "user_prompt": user_prompt,
                "injection": injection,
                "scenario": scenario or {},
                "tool_calls": [c.to_dict() for c in env.log],
                "final_environment": {
                    "files": env.files,
                    "db": env.db,
                    "emails": env.emails,
                },
                "agent_output": agent_output,
                "defense_logs": defense_logs,
                "outcome": outcome.to_dict(),
            },
        )

    def write_report(self, report: dict[str, Any]) -> Path:
        """Write the aggregated report and return its path."""
        path = self.dir / "report.json"
        self._write(path, {**report, **{"run_id": self.run_id, "manifest": self.manifest}})
        return path

    @staticmethod
    def _write(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
