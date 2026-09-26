"""Suite loading and validation.

Every evaluation entry point loads suites through here. Validation is fatal by
design: a scenario that cannot be executed faithfully must stop the run rather
than silently degrade into a no-op trial, which is exactly how the pre-audit
harness produced 38 attack trials with no payload present.

The benchmark suites themselves are not part of this repository; they are
published on the Hugging Face Hub (see ``HF_DATASET_REPO``) so the code and the
data can version independently and the data can carry its own license/citation.
A suite reference is either an explicit local path (for tests and the bundled
fixture) or an HF config name / ``hf://config`` reference, resolved and cached
via :mod:`huggingface_hub`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from chokepoint.dataset.schema import Scenario
from chokepoint.environment.environment import Environment, InjectionError
from chokepoint.tools.registry import tool_names

#: Hub dataset backing every suite except the tests/fixtures/ bundled sample.
HF_DATASET_REPO = os.environ.get("CHOKEPOINT_HF_DATASET", "AnimeshShaw/chokepoint-bench")

#: Config name -> file within the dataset repo. Mirrors the HF dataset card.
HF_CONFIGS = {
    "core": "scenarios/core_v2.jsonl",
    "core_reduced": "scenarios/core_reduced.jsonl",
    "adaptive": "scenarios/adaptive_v2.jsonl",
    "capability_probe": "scenarios/capability_probe.jsonl",
    "channel_asymmetry": "scenarios/channel_asymmetry.jsonl",
    "public_smoke": "scenarios/public_smoke.jsonl",
}


class SuiteValidationError(RuntimeError):
    """Raised when a scenario suite contains unexecutable scenarios."""


def resolve_suite_path(ref: str | Path) -> Path:
    """Resolve a suite reference to a local file path.

    Accepts, in order:

    1. An existing local path (tests, the bundled fixture, ad-hoc files) —
       returned unchanged.
    2. ``hf://<config>`` or a bare config name matching :data:`HF_CONFIGS` —
       downloaded (and cached) from :data:`HF_DATASET_REPO` via
       ``huggingface_hub.hf_hub_download``.

    Raises:
        SuiteValidationError: If neither resolves, or the ``huggingface_hub``
            download fails (network, auth, or an unknown config name).
    """
    ref = str(ref)
    if Path(ref).exists():
        return Path(ref)

    config = ref.removeprefix("hf://")
    if config not in HF_CONFIGS:
        raise SuiteValidationError(
            f"Suite reference {ref!r} is neither an existing local path nor a "
            f"known HF config. Known configs: {sorted(HF_CONFIGS)}"
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SuiteValidationError(
            "huggingface_hub is required to load suites by config name. "
            "Install it with: pip install 'chokepoint[data]'"
        ) from exc

    try:
        return Path(
            hf_hub_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                filename=HF_CONFIGS[config],
            )
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear, fatal error
        raise SuiteValidationError(
            f"Could not download config {config!r} from {HF_DATASET_REPO}: {exc}"
        ) from exc


def load_suite(path: str | Path, validate: bool = True) -> list[Scenario]:
    """Read a JSONL suite, parse it against schema v2, and validate it.

    Args:
        path: A local ``.jsonl`` path, or an HF config name / ``hf://config``
            reference resolved via :func:`resolve_suite_path`.
        validate: Run the full executability check. Disable only for tooling
            that deliberately inspects malformed suites (e.g. migration).

    Returns:
        The parsed scenarios, in file order.

    Raises:
        SuiteValidationError: If the suite cannot be resolved, or any scenario
            fails to parse or validate.
    """
    path = resolve_suite_path(path)
    if not path.exists():
        raise SuiteValidationError(f"Suite not found: {path}")

    scenarios: list[Scenario] = []
    errors: list[str] = []

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            scenarios.append(Scenario(**json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"  line {lineno}: {exc}")

    if errors:
        raise SuiteValidationError(
            f"{len(errors)} scenario(s) in {path.name} failed to parse:\n" + "\n".join(errors)
        )

    if validate:
        problems = validate_suite(scenarios)
        if problems:
            raise SuiteValidationError(
                f"{len(problems)} scenario(s) in {path.name} are not executable:\n"
                + "\n".join(f"  {p}" for p in problems)
            )

    return scenarios


def validate_suite(scenarios: list[Scenario]) -> list[str]:
    """Return a human-readable problem list; empty means the suite is runnable."""
    problems: list[str] = []
    known_tools = set(tool_names())
    seen_ids: dict[str, int] = {}

    for scenario in scenarios:
        sid = scenario.id
        if sid in seen_ids:
            problems.append(f"[{sid}] duplicate scenario id.")
        seen_ids[sid] = seen_ids.get(sid, 0) + 1

        for tool in scenario.utility_goal.required_tools:
            if tool not in known_tools:
                problems.append(
                    f"[{sid}] utility_goal names unknown tool {tool!r}; known: {sorted(known_tools)}"
                )

        if scenario.attack_goal:
            for i, condition in enumerate(scenario.attack_goal.success_conditions):
                if condition.tool not in known_tools:
                    problems.append(
                        f"[{sid}] success_conditions[{i}] names unknown tool {condition.tool!r}."
                    )

        # The decisive check: can the payload actually be placed?
        if scenario.injection:
            try:
                env = Environment.from_spec(scenario.environment)
                env.inject(
                    scenario.injection.locator,
                    scenario.injection.payload,
                    scenario.injection.mode,
                )
            except InjectionError as exc:
                problems.append(f"[{sid}] injection is unresolvable: {exc}")

    return problems


def suite_digest(path: str | Path) -> str:
    """SHA-256 of the resolved suite file, recorded in every run manifest."""
    return hashlib.sha256(resolve_suite_path(path).read_bytes()).hexdigest()


def prepare_environment(scenario: Scenario) -> Environment:
    """Build the scenario's environment with its payload already placed.

    Raises:
        InjectionError: If the payload cannot be placed. Callers must not
            catch this and continue — the trial is invalid.
    """
    env = Environment.from_spec(scenario.environment)
    if scenario.injection:
        env.inject(
            scenario.injection.locator,
            scenario.injection.payload,
            scenario.injection.mode,
        )
    return env


def suite_summary(scenarios: list[Scenario]) -> dict[str, Any]:
    """Counts used in run manifests and taxonomy reporting."""
    attacks = [s for s in scenarios if s.scenario_type == "attack"]
    benign = [s for s in scenarios if s.scenario_type == "benign"]
    return {
        "total": len(scenarios),
        "attack": len(attacks),
        "benign": len(benign),
        "multi_hop_attacks": sum(1 for s in attacks if s.is_multi_hop),
    }
