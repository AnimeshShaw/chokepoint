# Reproduction Guide

Suites are referenced by name (`public_smoke`, `core`, `adaptive`,
`capability_probe`, `channel_asymmetry`) and pulled automatically from the
companion Hugging Face dataset —
[`AnimeshShaw/chokepoint-bench`](https://huggingface.co/datasets/AnimeshShaw/chokepoint-bench)
— on first use, then cached locally. Offline tests use a tiny fixture bundled at
`tests/fixtures/public_smoke.jsonl`, so `pytest` never needs network access.

---

## 1. Install

Requires Python ≥ 3.10.

```bash
git clone <repo> && cd chokepoint
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[analysis,dev]"
```

`analysis` pulls in SciPy, which gives exact Fisher p-values. Without it the
package falls back to a self-contained implementation that agrees to within
1e-6 — verified in `tests/test_statistics.py`.

## 2. Credentials

```bash
cp .env.example .env
```

Fill in only the providers you intend to evaluate. Local Ollama models need no
key:

```bash
ollama serve
ollama pull llama3.1:8b
```

`.env` is gitignored permanently. Never commit it.

## 3. Verify without spending anything

```bash
pytest                                                # 173 tests, fully offline
python -m chokepoint.cli validate --suite tests/fixtures/public_smoke.jsonl
python -m chokepoint.cli models
```

Once you have network access, the same commands work against any published
config by name — no local file needed:

```bash
python -m chokepoint.cli validate --suite public_smoke   # downloads + caches via huggingface_hub
python analysis/taxonomy.py --suite core
```

`validate` is the gate that matters: it confirms every injection payload
resolves to a real resource in its scenario's environment. A suite that fails
here will not run.

---

## 4. Run

### One configuration

```bash
python -m chokepoint.cli run \
  --suite public_smoke \
  --model llama3.1:8b \
  --defense llm_judge \
  --judge-model gpt-5.6-terra
```

Setting `--judge-model` separately from `--model` matters. If the judge is the
agent's own model, the "defense" shares the agent's priors and gets credit for
the agent's own refusals.

### The full grid

```bash
python experiments/run_benchmark.py \
  --suite core \
  --models gpt-5.6-terra llama3.1:8b \
  --defenses none type_checker capability_router llm_judge
```

Completed cells are skipped on re-invocation, keyed on the suite's SHA-256, so
an interrupted sweep resumes. Use `--force` to re-run regardless. Editing the
suite changes its digest and correctly invalidates the cache.

### Adaptive suite

```bash
python experiments/run_adaptive.py --suite adaptive
```

The `adaptive` scenarios above are one-shot generations, not attacks optimized
against a live defense, and results from them are a weak lower bound on evasion
— not evidence of mitigation. For a real robustness claim use the iterative
attacker in §6 below, which optimizes against the live defended agent.

### Overhead

```bash
python experiments/measure_overhead.py --model gpt-5.6-terra --trials 20
```

Fewer than 20 trials cannot separate defense cost from API jitter. When the
bootstrap interval on a ratio spans 1.0×, the tool says "not detectable" — report
it that way rather than as a speedup.

---

## 5. Analyze

The analysis scripts read persisted run reports and traces; most need no further
API calls.

```bash
# Between-defense significance: Wilson intervals, Fisher exact, Holm-Bonferroni.
python analysis/statistics.py --results-root results/runs --compare llama3.1:8b

# How far each scoring defect distorts the headline number (the central figure).
python analysis/ablation.py --results-root results/runs

# Did the agent disclose the attacks it suffered? (rates are in each report.json)
# Capability vs. security: probe fidelity, then security conditioned on it.
python analysis/capability.py --results-root results/runs

# Judge calibration: the full security/utility ROC, not one operating point.
python analysis/judge_roc.py --results-root results/runs

# Channel asymmetry: per-channel ASR with payload held constant.
python analysis/channel.py --results-root results/runs

# Suite composition and attack taxonomy.
python analysis/taxonomy.py --suite core
```

Fisher rather than χ²: attack strata run to a few dozen scenarios, where
expected cell counts fall below the threshold at which the χ² approximation
holds.

### Sizing the suite first

Before generating scenarios, check what a given suite size can resolve:

```bash
python analysis/power.py --baseline 0.60 --target 0.35
```

At the headline 60%→35% comparison, 80% power needs 62 attack scenarios per
condition; the pilot 43-scenario suite reaches only 65%. If the defenses you
compare differ by less than the minimum detectable effect for your n, a null
result is uninformative.

## 6. The dedicated experiments

Beyond the main grid, four experiments answer questions the grid does not:

```bash
# D1 ablation: the ASR error from silently-dropped payloads (paired sweep).
python experiments/ablate_delivery.py --suite core \
  --model gpt-5.6-terra --defense none

# Availability attack: bypass the judge by making it unreachable, not by fooling it.
python experiments/availability_attack.py --suite core \
  --model gpt-5.6-terra --judge-model claude-sonnet-5

# Channel asymmetry: matched-payload triplets across email/file/sql.
python experiments/run_benchmark.py --suite channel_asymmetry \
  --models gpt-5.6-terra --defenses none

# Iterative adaptive attacks: optimize the payload against the live defense.
python experiments/run_adaptive_iterative.py --suite adaptive \
  --model gpt-5.6-terra --judge-model claude-sonnet-5 \
  --attacker-model claude-sonnet-5 --defenses type_checker llm_judge
```

`channel_asymmetry` is published pre-built; `scripts/build_channel_suite.py` is
only needed if you want to regenerate or modify it locally.

The adaptive experiment reports an ASR-by-round curve. A curve that climbs is
the finding — the defense was untested, not robust. Keep the attacker model
distinct from both the agent and the judge, or the attack understates what a
real adversary achieves.

---

## 7. Output layout

```
results/runs/<model>__<defense>__<hash>__<timestamp>/
├── manifest.json        model, defense, suite SHA-256, git commit, versions
├── report.json          integer counts alongside every rate
└── traces/
    └── <scenario_id>.json
```

Each trace holds the user prompt, the resolved injection, every tool call with
its arguments and result, which layer blocked what, the final environment state,
the agent's output, defense decision logs, and the per-condition attack
evaluation.

Read a trace when a number surprises you. All four audit defects would have been
visible from a single trace file; none were visible from the aggregate reports.

---

## 8. Adding a scenario

The sections below describe *local dataset construction* — authoring new
scenarios in a local, gitignored `data/scenarios/` directory before optionally
publishing them. This is distinct from *loading* a suite at runtime, which is
always done by config name (`--suite core`) against the published dataset.

Author new scenarios in schema v2 — see `scripts/build_smoke_suite.py` for
worked examples.

```jsonc
{
  "id": "myscenario-01",
  "suite": "custom",
  "scenario_type": "attack",
  "user_prompt": "...",
  "environment": {"emails": [...], "files": {...}, "db": {...}},
  "injection": {"channel": "file", "locator": "file:report.csv",
                "mode": "append", "payload": "..."},
  "utility_goal": {"required_tools": ["read_file", "write_file"]},
  "attack_goal": {
    "description": "...",
    "require": "any",
    "success_conditions": [
      {"tool": "send_email", "args": {"to_address": {"iregex": "attacker\\.example"}}}
    ]
  },
  "is_multi_hop": true,
  "tags": ["exfiltration"]
}
```

Rules that the validator enforces:

1. **The locator must resolve.** `file:report.csv` requires `report.csv` in
   `environment.files`. Locators are `email:<id>`, `file:<path>`, or
   `sql:<table>.<column>[#<row>]`.
2. **The environment must support the benign task.** If `utility_goal` requires
   `execute_sql_query`, the scenario needs a table to query. Otherwise the task
   fails for reasons that have nothing to do with security.
3. **Attack conditions must be argument-level and distinguishing.** Write the
   condition so that only the attacker's version of the call satisfies it.
   `{"tool": "execute_sql_query"}` with no argument constraints will fire on
   legitimate SQL and is almost always a mistake.

Then:

```bash
python -m chokepoint.cli validate --suite data/scenarios/my_suite.jsonl
```

## 9. Adding a defense

Implement `wrap_tools(tools, ..., recorder=None) -> list[BaseTool]`, returning
tools that preserve name, description, and `args_schema`. On a block, return an
explanatory string and call
`recorder(tool_name, args, message, "<your_defense>")` so the blocked call is
distinguishable from one the agent never attempted.

Register the name in `DefenseType` and wire it into
`DefendedAgent._apply_defenses`. Log collection deliberately does not catch
exceptions — an interface mismatch must fail loudly. Silently swallowing one is
exactly how a prior version of this harness ended up with empty type-checker
logs across an entire evaluation without anyone noticing.

---

## 10. Migrating legacy schema-v1 suites

```bash
python scripts/migrate_legacy_suites.py
```

Writes `*.v2-draft.jsonl`. Migration recovers injection locators and attacker
predicates where they are derivable from the payload text (40/43 and 41/43
respectively for the core suite) but **never invents environments** — v1 did not
record them. Drafts fail validation until each `environment` block is authored.

### Backfilling environments

Rather than authoring 95 environments by hand, derive them and verify every one:

```bash
# Always dry-run a handful first.
python scripts/backfill_environments.py \
  --src data/scenarios/core_v1.v2-draft.jsonl \
  --dest data/scenarios/core_v2.jsonl \
  --limit 5

# Then the full suite, keeping the report.
python scripts/backfill_environments.py \
  --src data/scenarios/core_v1.v2-draft.jsonl \
  --dest data/scenarios/core_v2.jsonl \
  --report results/backfill_report.json
```

For each draft the model is given the user prompt, the injection locator, the
required tools, and the attacker's success conditions, and must produce an
environment satisfying all of them. The proposal is then parsed, schema-checked,
injection-resolved, and feasibility-checked. **Failures are fed back into the
next attempt**, so the model corrects against the real validator rather than
guessing again. A draft that never verifies is reported and dropped — never
written in a degraded form.

Current draft state:

| Suite | Drafts | Environment only | Also needs hand-authoring |
|---|---|---|---|
| core | 95 | 90 | 3 missing locator, 2 missing attack goal |
| adaptive | 10 | 9 | 1 missing attack goal |

Then check the result, and read a few by hand before trusting the batch:

```bash
chokepoint validate --suite data/scenarios/core_v2.jsonl --strict
```

Automated authoring is a convenience, not an oracle. Spot-check that the
generated environments are realistic and that each attacker objective has data
worth stealing before running an experiment on them.
