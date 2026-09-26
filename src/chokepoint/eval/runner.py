"""Evaluation loop.

For each scenario: build an isolated environment, place the payload (failing
loudly if it cannot be placed), wrap the tools in the configured defense, run
the agent under a timeout, score the outcome, and persist the trace.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any

from tqdm import tqdm

from chokepoint.agents.base_agent import BaseAgent
from chokepoint.agents.defended_agent import DefendedAgent, DefenseType
from chokepoint.dataset.loader import (
    load_suite,
    prepare_environment,
    suite_digest,
    suite_summary,
)
from chokepoint.dataset.schema import Scenario
from chokepoint.eval.metrics import ScenarioOutcome, aggregate, score_scenario
from chokepoint.eval.traces import TraceWriter, build_manifest
from chokepoint.tools.registry import build_tools

DEFAULT_TIMEOUT_S = 120


class EvaluationRunner:
    """Runs one (model, defense, suite) configuration end to end."""

    def __init__(
        self,
        suite_path: str | Path,
        results_root: str | Path = "results/runs",
        timeout_s: int = DEFAULT_TIMEOUT_S,
        disclosure_assessor: Any | None = None,
    ) -> None:
        """
        Args:
            disclosure_assessor: Optional assessor deciding whether the agent
                surfaced anomalous actions to the user. Omit to skip the
                measurement; unassessed episodes are excluded from concealment
                rates rather than counted as concealed.
        """
        self.suite_path = Path(suite_path)
        self.results_root = Path(results_root)
        self.timeout_s = timeout_s
        self.disclosure_assessor = disclosure_assessor
        self.scenarios: list[Scenario] = load_suite(self.suite_path)

    def run(
        self,
        model_id: str,
        defense_type: DefenseType = DefenseType.NONE,
        limit: int = 0,
        offset: int = 0,
        judge_model: str | None = None,
        progress: bool = True,
    ) -> dict[str, Any]:
        """Evaluate the suite and return the aggregated report."""
        selected = self.scenarios[offset:]
        if limit > 0:
            selected = selected[:limit]

        manifest = build_manifest(
            model_id=model_id,
            defense=defense_type.value,
            suite_path=self.suite_path,
            suite_digest=suite_digest(self.suite_path),
            suite_summary=suite_summary(selected),
            extra={
                "judge_model": judge_model or model_id,
                "limit": limit,
                "offset": offset,
                "timeout_s": self.timeout_s,
            },
        )
        writer = TraceWriter(self.results_root, manifest)

        outcomes: list[ScenarioOutcome] = []
        iterator = tqdm(selected, desc=f"{model_id}/{defense_type.value}") if progress else selected

        for scenario in iterator:
            outcome = self._run_one(scenario, model_id, defense_type, judge_model, writer)
            outcomes.append(outcome)

        report = aggregate(outcomes)
        report["model_id"] = model_id
        report["defense_type"] = defense_type.value
        report_path = writer.write_report(report)

        return {"report": report, "run_id": writer.run_id, "report_path": str(report_path)}

    def _run_one(
        self,
        scenario: Scenario,
        model_id: str,
        defense_type: DefenseType,
        judge_model: str | None,
        writer: TraceWriter,
    ) -> ScenarioOutcome:
        # Fails loudly on an unplaceable payload rather than running a no-op trial.
        env = prepare_environment(scenario)
        tools = build_tools(env)

        if defense_type == DefenseType.NONE:
            agent = BaseAgent(model_id=model_id, tools=tools)
        else:
            agent = DefendedAgent(
                model_id=model_id,
                tools=tools,
                defense_type=defense_type,
                judge_model=judge_model or model_id,
                recorder=env.record,
            )

        result: dict[str, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(agent.run, scenario.user_prompt)
            try:
                result = future.result(timeout=self.timeout_s)
            except concurrent.futures.TimeoutError:
                result = {"output": "", "error": f"timeout after {self.timeout_s}s"}
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                result = {"output": "", "error": f"{type(exc).__name__}: {exc}"}

        outcome = score_scenario(
            scenario,
            env.log,
            error=result.get("error"),
            agent_output=result.get("output", ""),
            disclosure_assessor=self.disclosure_assessor,
        )

        writer.write_trace(
            scenario_id=scenario.id,
            user_prompt=scenario.user_prompt,
            env=env,
            outcome=outcome,
            agent_output=result.get("output", ""),
            defense_logs=result.get("defense_logs", {}),
            injection=scenario.injection.model_dump() if scenario.injection else None,
            scenario={
                "scenario_type": scenario.scenario_type,
                "attack_goal": scenario.attack_goal.model_dump() if scenario.attack_goal else None,
                "utility_goal": scenario.utility_goal.model_dump(),
                "tags": scenario.tags,
                "is_multi_hop": scenario.is_multi_hop,
            },
        )
        return outcome
