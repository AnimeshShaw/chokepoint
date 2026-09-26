# Use of AI Assistance

This document records how generative-AI tools were used in producing Chokepoint,
in the spirit of the disclosure policies now adopted by most venues (ICML,
NeurIPS, ACL, IEEE, USENIX) and of ordinary scientific transparency. It
complements the "Use of AI Assistance" statement in the paper.

## Summary

AI coding assistance was used as a tool, under author direction and review, for
implementation and drafting. It was not an author, and the author takes full
responsibility for every claim, number, and design decision.

## Where AI assistance was used

- **Implementation and refactoring** of the harness, defenses, metrics,
  feasibility checks, analysis scripts, and regression tests.
- **Drafting prose** for the paper and documentation, subsequently edited and
  verified by the author.
- **Scenario tooling**: an LLM was used to migrate and back-fill scenario
  environments from a prior suite; every generated scenario was passed through
  the harness's feasibility checks, and scenarios the tooling could not complete
  validly were **hand-authored by the author**.

## What was decided and verified by the author, not the AI

- The research question, framing, and the four-defect taxonomy.
- The scenario schema, the metric definitions, and the feasibility checks that
  distinguish measurement validity from schema validity.
- Every methodological judgement, including how each defect is scored in the
  ablation.
- **Verification of all reported numbers against the persisted per-scenario
  traces.** No figure in the paper is taken on the tooling's word; each is
  recomputed from the recorded execution traces, most of them offline with no
  model access.

## LLMs as evaluated artifacts (not assistance)

Separately from assistance, LLMs appear in this work as the **objects of study**:

- as the **agent under test** (the model whose tool calls we evaluate), and
- as components of two evaluated defenses/measurements: the **LLMJudge** defense
  and the **disclosure assessor**.

These roles are part of the experimental design and are documented in the paper
(Sec. V and VI). They are distinct from the authoring assistance described above.

## Reproducibility

Because every run persists a manifest (model, defense, suite SHA-256, git commit,
package versions) and a self-describing per-scenario trace, the results do not
depend on trusting any AI-generated artifact: they are recomputable from the
traces, and the trace-based analyses require no model access at all. See
[reproducing.md](reproducing.md).
