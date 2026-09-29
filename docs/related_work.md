# Related Work and Novelty Assessment

**Last updated:** 2026-08-23
**Purpose:** Position Chokepoint against the published literature and state
honestly which contributions survive that comparison.

This document is deliberately unflattering. A benchmark paper that misjudges
its own novelty gets rejected at review; better to find the overlap here.

---

## 1. The field in one paragraph

Indirect prompt injection (IPI) exploits the absence of a code/data boundary in
the LLM context window: an agent that retrieves attacker-controlled text may
execute it as instruction. Between 2024 and 2026 the research community
converged on a defensive strategy — **do not rely on training the model to
refuse; enforce policy outside the model, at a deterministic mediation point
between the agent's proposed action and its execution.** Chokepoint's framing is
this consensus position. That is worth stating plainly: the framing is correct,
but it is not itself a contribution.

---

## 2. Benchmarks

| Work | Venue | Scale | Environment | Defenses evaluated |
|---|---|---|---|---|
| **InjecAgent** (Zhan et al.) | ACL Findings 2024 | 1,054 cases, 17 user tools, 62 attacker tools, 30 agents | static prompts | no (attack-only) |
| **AgentDojo** (Debenedetti et al.) | NeurIPS 2024 D&B | 97 tasks / 629 cases, 4 domains, ≤18 tool calls | stateful, dynamic | yes |
| **WASP** | 2025 | web-agent focused | live browser | yes |
| **ToolShield / MT-AgentRisk** (CHATS-lab) | ICML 2026 | 365 harmful tasks, 5 MCP tools | multi-turn | yes |
| **PIArena**, **Open-Prompt-Injection** | 2025–26 | platform / leaderboard | varied | yes |
| **Chokepoint** (this work) | — | 95 core + 10 adaptive | stateful, per-scenario | 3 paradigms |

### 2.1 AgentDojo is the closest prior work

AgentDojo (Debenedetti et al., NeurIPS 2024 Datasets & Benchmarks) is the direct
competitor and overlaps on nearly every design axis: tool-calling agents,
stateful simulated environments (email, banking, Slack, travel), indirect
injection through retrieved data, and joint evaluation of attacks *and*
defenses.

Its metric triple is:

- **Benign Utility (BU)** — user tasks solved with no attack present
- **Utility under Attack (UA)** — user tasks still solved when attacked
- **Targeted ASR** — fraction of cases achieving the attacker's specific goal

The corrected Chokepoint metrics (`UPR_benign`, `UPR_attack`, argument-level
`ASR`) are a re-derivation of exactly this triple. That convergence is
reassuring about correctness and fatal to novelty: the metric design cannot be
claimed as a contribution, and the paper must cite AgentDojo as the source of
the design rather than presenting it as new.

AgentDojo is also larger (629 cases vs 95), broader (4 domains vs 3 tool
families), and deeper (up to 18 tool calls per task).

### 2.2 Scale

At 95 core scenarios with 43 attacks, Chokepoint's attack stratum is an order of
magnitude smaller than InjecAgent's and roughly a seventh of AgentDojo's. The
practical consequence is statistical: a 43-item stratum gives 95% Wilson
intervals roughly ±15 percentage points wide near 50%. Most between-defense
differences of interest are smaller than that interval. **The current suite is
underpowered for the comparisons the paper wants to make**, independent of the
harness defects. A power analysis should precede any re-run.

---

## 3. Defenses

Chokepoint implements three paradigms. Each has a stronger published
counterpart:

| Chokepoint layer | Published counterpart | Gap |
|---|---|---|
| `TypeChecker` — regex/keyword parameter validation | **CommandSans** (surgical prompt sanitization), tool-result parsing defenses | Denylist regex is the weakest form of this family; trivially evaded by rephrasing |
| `CapabilityRouter` — intent→tool RBAC | **CaMeL** (DeepMind 2025): dual-LLM, per-value capability metadata, data-flow-enforcing Python interpreter | CaMeL tracks provenance per *value*; the router gates per *prompt*. Categorically weaker |
| `LLMJudge` — secondary model reviews each call | **Task Shield** (task-alignment enforcement), **MELON**, **IPIGuard** (tool dependency graphs), **AgentSentry** | Zero-shot binary safety classification is the baseline form; all counterparts are more structured |

CaMeL reports ~67% of AgentDojo injections mitigated with provable data-flow
guarantees. The `CapabilityRouter` here does prompt-level intent classification
with a keyword map. These are not comparable in strength, and the paper should
not imply they are.

---

## 4. The adaptive-attack problem

This is the most serious threat to the current draft's conclusions.

**Zhan et al., "Adaptive Attacks Break Defenses Against Indirect Prompt
Injection Attacks on LLM Agents" (NAACL Findings 2025)** evaluated **eight**
IPI defenses and broke **all eight**, consistently achieving **>50% ASR** with
adaptive attacks. **"The Attacker Moves Second" (arXiv 2510.09023)** reaches
the same conclusion across jailbreaks and injections. **AutoDojo (arXiv
2606.15057)** extends it with adaptive black-box attacks on AgentDojo.

The current draft (§4.3) reports that `LLMJudge` achieved **0.0% adaptive ASR —
"Complete Mitigation"** on both models tested.

That claim is not credible, and it should be treated as a red flag rather than
a result. The most likely explanations, in order:

1. **The adaptive suite is too weak.** It contains 10 scenarios generated in
   one-shot LLM calls from a description of the defenses. Published adaptive
   attacks use iterative optimization against the deployed defense, not a
   single generation pass. An attack that was never optimized against the judge
   is not an adaptive attack; it is a themed static attack.
2. **The judge and the agent were the same model family.** The judge defaults
   to the agent's own model, so a "defense" that shares the agent's priors is
   being credited for the agent's own refusals.
3. **The tool-name-level ASR metric (defect D2)** applies here too.

**Recommendation:** either implement genuine iterative adaptive attacks — a
generate/test/refine loop against the live defense — or drop the "complete
mitigation" claim entirely and report the adaptive suite as a weak lower bound
on evasion. Publishing a 0% adaptive ASR against a literature that has broken
every comparable defense will not survive review.

---

## 5. Findings that are already established

Two of the draft's headline findings restate known results:

- **"Open SLMs are far more vulnerable than frontier models."** Documented
  repeatedly: open-weight models average ~33% attack success against
  single-digit rates for current proprietary models. Chokepoint's
  62.8% vs 13.9% is directionally consistent with published work but is not new
  information.
- **"Small models fail to emit valid tool frames."** The tool-use capability
  ceiling of small open-weight models is the subject of dedicated work
  (**AgentFloor**, **ToolFailBench**, **BFCL**, **τ-bench**). The draft's §5.3
  observation that `gemma4:12b` and `qwen2.5-coder:7b` hit 100% FRR is a
  known capability limit, and — importantly — is *confounded* with harness
  defect D3, so the current evidence cannot separate the two causes.

---

## 6. Honest novelty assessment

**As currently framed, Chokepoint has no defensible novelty claim.** Every axis
— the benchmark design, the metric triple, each of the three defense paradigms,
the frontier-vs-open finding, the tool-binding observation — is covered, and
covered better, by existing work.

That is a framing problem, not a dead end. Four repositioning options, ranked by
how much defensible contribution each carries.

### Option A — Methodology / reproducibility paper *(strongest)*

**Claim:** agentic-security benchmarks can silently produce plausible,
publishable, entirely invalid numbers, and here is a validated construction that
prevents it.

The audit in [harness_defects.md](harness_defects.md) documents four defects
that each produce *believable* results:

- payload delivery failing silently for 88% of attack cases,
- attack success scored by tool name, marking correct behaviour as compromise,
- an environment that cannot satisfy its own benign tasks, so utility loss is
  misattributed to the defense,
- no persisted traces, making every number unauditable.

None of these produce an obvious error. Each produces a number that looks
reasonable and would pass review. The contribution is the **failure taxonomy
plus the executable countermeasures**: a scenario schema whose payload
placement is machine-verifiable, a validator that refuses to run a degraded
scenario, argument-level attacker predicates, and mandatory trace persistence.

This is genuinely novel, immediately useful to anyone building an agent-security
benchmark, and it is the one contribution that is *strengthened* rather than
weakened by what the audit found. Reproducibility and evaluation-validity work
has a real venue path.

### Option B — Defense economics for locally-deployed agents *(strong, underexplored)*

**Claim:** the defense literature assumes a frontier-grade judge is available.
For locally-deployed small models — the case that motivates on-prem agents in
the first place — that assumption fails.

Open questions with no good published answer:

- If the judge must *also* be a local SLM, does semantic mediation still work,
  or does the judge inherit the same injection susceptibility it is meant to
  catch? (Related: LLM-as-a-Judge is itself injectable — arXiv 2505.13348.)
- What is the security-per-dollar and security-per-millisecond frontier across
  paradigms? Deterministic validation is free; a judge doubles latency and adds
  token cost. Most defense papers report ASR reduction with no cost axis at all.
- Does a cheap deterministic layer plus a small judge Pareto-dominate a single
  expensive judge?

The codebase already supports this: multi-provider registry, local Ollama
models, composite defenses (`llm_judge+type_checker`, `all`), and an overhead
harness. The `judge_model` is independently configurable from the agent model,
which is exactly the knob this study needs.

### Option C — Defense composition *(moderate)*

**Claim:** layered mediation composes non-additively, and here is where.

The literature notes that different defenses catch different attacks and that
combining them helps, but systematic composition analysis is thin. `JUDGE_AND_TYPE`
and `ALL` are implemented and were never evaluated. A clean 2×2×2 factorial over
the three layers, with interaction effects, is a modest but real contribution —
and cheap, since it reuses the existing harness.

### Option D — Benchmark contribution *(weakest — not recommended alone)*

Competing head-on with AgentDojo requires exceeding it on scale, realism, or
domain coverage. At 95 scenarios and 3 tool families, Chokepoint does not. This
framing should be abandoned unless the suite grows by an order of magnitude and
adds domains AgentDojo lacks.

### Recommended framing

**Lead with A, support with B.** A methodology paper that establishes how
agent-security evaluation silently fails, demonstrates the failure modes
concretely, ships a validated harness that prevents them, and then *uses* that
harness to answer the underexplored local-SLM defense-economics question. The
audit becomes the paper's foundation instead of its embarrassment, and the
empirical work has a question that is actually open.

Working title under this framing:

> **Silent Failures in Agentic Security Evaluation: A Validated Harness for
> Tool-Call Mediation Under Indirect Prompt Injection**

*This framing was adopted; the published paper carries this exact title
([arXiv:2609.32691](https://arxiv.org/abs/2609.32691)).*

---

## 7. What must change before submission

*Update: this checklist predates submission. Items 1–7 were addressed in the
published paper; item 8's positioning is carried in the current related-work
section above.*

| # | Item | Why |
|---|---|---|
| 1 | Re-run everything on the corrected harness | Current numbers measure harness defects |
| 2 | Power analysis; grow the attack stratum | 43 cases cannot resolve the differences claimed |
| 3 | Replace one-shot adaptive generation with iterative optimization | Otherwise the adaptive section contradicts NAACL 2025 |
| 4 | Separate judge model from agent model | A self-judging agent is not an independent defense |
| 5 | Cite AgentDojo as the source of the metric design | Convergent re-derivation must be attributed |
| 6 | Reframe §5.1/§5.3 as confirmatory, not novel | Both restate established results |
| 7 | Add cost/latency axis to every defense comparison | The differentiator under framing B |
| 8 | Position against CaMeL, Task Shield, MELON, IPIGuard | Currently uncited; all are stronger counterparts |

---

## Bibliography

- Zhan et al. *InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents.* ACL Findings 2024. [arXiv:2403.02691](https://arxiv.org/abs/2403.02691)
- Debenedetti et al. *AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.* NeurIPS 2024 D&B. [emergentmind summary](https://www.emergentmind.com/topics/agentdojo-benchmark)
- Zhan et al. *Adaptive Attacks Break Defenses Against Indirect Prompt Injection Attacks on LLM Agents.* NAACL Findings 2025. [arXiv:2503.00061](https://arxiv.org/abs/2503.00061)
- *The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against LLM Jailbreaks and Prompt Injections.* [arXiv:2510.09023](https://arxiv.org/html/2510.09023v1)
- *AutoDojo: Adaptive Black-Box Attacks Reveal the Limits of IPI Defenses.* [arXiv:2606.15057](https://arxiv.org/pdf/2606.15057)
- Debenedetti et al. *CaMeL: Defeating Prompt Injections by Design.* DeepMind 2025. [commentary](https://simonwillison.net/2025/Apr/11/camel/)
- *The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents.* [arXiv:2412.16682](https://arxiv.org/pdf/2412.16682)
- *MELON: Provable Defense Against Indirect Prompt Injection Attacks in AI Agents.* [arXiv:2502.05174](https://arxiv.org/pdf/2502.05174)
- *IPIGuard: A Tool Dependency Graph-Based Defense Against Indirect Prompt Injection.* [arXiv:2508.15310](https://arxiv.org/pdf/2508.15310)
- *CommandSans: Securing AI Agents with Surgical Precision Prompt Sanitization.* [arXiv:2510.08829](https://arxiv.org/pdf/2510.08829)
- *AgentSentry: Mitigating Indirect Prompt Injection via Temporal Causal Diagnostics.* [arXiv:2602.22724](https://arxiv.org/pdf/2602.22724)
- *Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks.* [arXiv:2505.13348](https://arxiv.org/pdf/2505.13348)
- *Security in LLM-as-a-Judge: A Comprehensive SoK.* [arXiv:2603.29403](https://arxiv.org/abs/2603.29403)
- *WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks.* [arXiv:2504.18575](https://arxiv.org/pdf/2504.18575)
- *AgentFloor: How Far Up the Tool Use Ladder Can Small Open-Weight Models Go?* [arXiv:2605.00334](https://arxiv.org/pdf/2605.00334)
- *ToolFailBench: Diagnosing Tool-Use Failures in LLM Agents.* [arXiv:2607.04686](https://arxiv.org/pdf/2607.04686)
- *A Survey on Agentic Security: Applications, Threats and Defenses.* [arXiv:2510.06445](https://arxiv.org/pdf/2510.06445)
- *Prompt Injection Attacks in LLMs and AI Agent Systems: A Comprehensive Review.* MDPI Information 17(1):54, 2026. [link](https://www.mdpi.com/2078-2489/17/1/54)
- CHATS-lab. *ToolShield / MT-AgentRisk: Unsafer in Many Turns.* ICML 2026. [GitHub](https://github.com/CHATS-lab/ToolShield) — **name collision; motivated this project's rename**
