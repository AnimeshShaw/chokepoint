"""Feasibility checks beyond schema validation.

`validate_suite` answers "can this scenario run?". These functions answer the
harder question: "can this scenario *measure what it claims to measure*?"

Both failure modes below produce a scenario that executes cleanly and reports a
plausible number while measuring nothing:

* A benign task requiring ``execute_sql_query`` against an environment with no
  tables fails for capability reasons, and the failure is recorded as a false
  rejection. This is audit defect D3 in miniature.
* An attacker objective naming a column no table has is technically reachable —
  the agent can still emit the query text — but the scenario no longer describes
  a realistic exfiltration, so a low ASR says nothing about the defense.

The first is an error. The second is a warning, because the condition can still
legitimately match on query text alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from chokepoint.dataset.schema import Scenario

#: Which part of the environment each tool needs in order to do anything.
_TOOL_REQUIREMENTS: dict[str, str] = {
    "read_email": "emails",
    "send_email": "emails",
    "read_file": "files",
    "write_file": "files",
    "execute_sql_query": "db",
}

#: `send_email` and `write_file` create rather than consume, so an empty
#: starting state is legitimate for them.
_CREATES_RATHER_THAN_CONSUMES = {"send_email", "write_file"}


@dataclass
class FeasibilityReport:
    """Result of checking one scenario for measurement validity."""

    scenario_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def check_utility_feasibility(scenario: Scenario) -> list[str]:
    """Verify the environment can support the legitimate task.

    A required tool that consumes state needs that state to exist, otherwise the
    benign task fails for reasons unrelated to security.
    """
    env = scenario.environment or {}
    errors: list[str] = []

    for tool in scenario.utility_goal.required_tools:
        if tool in _CREATES_RATHER_THAN_CONSUMES:
            continue
        section = _TOOL_REQUIREMENTS.get(tool)
        if section and not env.get(section):
            errors.append(
                f"utility_goal requires {tool!r} but environment.{section} is empty; "
                f"the benign task cannot succeed for capability reasons"
            )
    return errors


def check_attack_reachability(scenario: Scenario) -> list[str]:
    """Flag attacker objectives the environment makes unrealistic.

    Warnings only: a condition matching on query text can fire even when the
    named column does not exist. But a scenario whose target data is absent
    measures the agent's willingness to *type* an exfiltration, not to perform
    one.
    """
    if not scenario.attack_goal:
        return []

    env = scenario.environment or {}
    warnings: list[str] = []

    all_columns = {
        column.lower()
        for rows in (env.get("db") or {}).values()
        for row in rows
        for column in row
    }
    all_paths = {p.lower() for p in (env.get("files") or {})}

    for i, condition in enumerate(scenario.attack_goal.success_conditions):
        for arg, matcher in condition.args.items():
            literal = _literal_of(matcher)
            if not literal:
                continue

            if condition.tool == "execute_sql_query" and arg == "query":
                alternatives = [a for a in literal.lower().split("|") if a.isidentifier()]
                if alternatives and not any(a in all_columns for a in alternatives):
                    warnings.append(
                        f"success_conditions[{i}] targets column(s) {alternatives} "
                        f"that no table declares; the exfiltration has no data to steal"
                    )

            if condition.tool == "read_file" and arg == "file_path":
                needle = literal.lower().strip("^$")
                if needle and not any(needle in p for p in all_paths):
                    warnings.append(
                        f"success_conditions[{i}] targets file {literal!r} which is "
                        f"not in environment.files; the read will error out"
                    )

    return warnings


def check_injection_is_reachable_by_task(scenario: Scenario) -> list[str]:
    """Verify the agent will plausibly *read* the poisoned resource.

    An injection the agent never encounters cannot influence it. The scenario
    then measures nothing, exactly as in audit defect D1 — only here the payload
    is placed correctly and simply never read.
    """
    if not scenario.injection:
        return []

    channel = scenario.injection.channel
    required = set(scenario.utility_goal.required_tools)
    readers = {"email": "read_email", "file": "read_file", "sql": "execute_sql_query"}
    reader = readers[channel]

    if reader not in required:
        return [
            f"payload is injected into a {channel} resource but the task does not "
            f"require {reader!r}; the agent may never read the injection"
        ]
    return []


def _literal_of(matcher: Any) -> str:
    """Best-effort literal text from a matcher, for heuristic checks.

    Strips word-boundary anchors before unescaping. Note that ``str.strip``
    removes *characters* rather than a suffix, so stripping ``"\\b"`` would eat
    a leading ``b`` and turn ``bank_account`` into ``ank_account``.
    """
    for kind in ("equals", "contains", "icontains", "regex", "iregex"):
        value = getattr(matcher, kind, None)
        if value:
            without_anchors = re.sub(r"\\b", "", value)
            return re.sub(r"\\(.)", r"\1", without_anchors).strip()
    return ""


def check_scenario(scenario: Scenario) -> FeasibilityReport:
    """Run every feasibility check against one scenario."""
    report = FeasibilityReport(scenario_id=scenario.id)
    report.errors.extend(check_utility_feasibility(scenario))
    report.warnings.extend(check_attack_reachability(scenario))
    report.warnings.extend(check_injection_is_reachable_by_task(scenario))
    return report


def check_suite(scenarios: list[Scenario]) -> list[FeasibilityReport]:
    """Run feasibility checks across a whole suite."""
    return [check_scenario(s) for s in scenarios]
