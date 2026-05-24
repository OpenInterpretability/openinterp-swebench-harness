# Wandering vs Locked: A 34% Blind Spot in Probe-Based Agent Failure Monitoring

**Draft v0 — 2026-05-23**

## Abstract

Probe-based safety monitoring of LLM agents typically assumes that a probe trained to predict trajectory success/failure provides a reliable signal at any turn. We test this on 99 trajectories of Qwen3.6-27B running SWE-bench Pro and identify two mechanistically distinct sub-classes of agent failure: LOCKED (66%) and WANDERING (34%). Both sub-classes are externally identical — they exhaust the turn budget without emitting `finish_tool` — but differ fundamentally in internal state. LOCKED agents are uncertain (probe collapses to verdict <0.30 by mean fraction 0.92 of trajectory length, 21% never produce a patch). WANDERING agents are over-confident (probe stays >0.70 with median final score 1.000, 95% produce patches, but never emit completion action). Qualitative trace inspection confirms WANDERING agents reason "the code is correct" or "tests fail due to environment" while LOCKED agents are still exploring at turn 49. The 34% disagreement between probe verdict and external outcome constitutes a quantified blind spot for probe-only monitoring schemes. We propose this as a natural-trajectory analog of the intervention-induced obfuscation documented in Paper 11.

## §1 — Problem

Production safety pipelines for LLM agents rely on activation probes as monitoring signals. A common assumption is that probe = trustworthy outcome predictor at any turn. We test this on agent trajectories that fail.

## §2 — Setup

- **Model**: Qwen3.6-27B (Alibaba 2026)
- **Task**: SWE-bench Pro (Pro-balanced split)
- **Dataset**: 99 trajectories from OpenInterp Phase 6 capture, residuals at L11/L23/L31/L43/L55 per turn
- **Outcome label**: `finish_reason == "finish_tool"` (n=40 success) vs `"max_turns"` (n=59 failure)
- **Probe**: top-10 diff-means features at L43 pre_tool + StandardScaler + LogisticRegression (agent-probe-guard v0.1 family)
- **Methodology**: 5-fold cross-validation; probe re-trained on representative residual (mean across all turns) within each fold; per-turn scores computed on held-out trajectories

## §3 — Results

### 3.1 Baseline probe performance

Trajectory-mean residual at L43 pre_tool achieves CV AUROC = **0.934 ± 0.041**. Stronger than previously published (Phase 6c, 0.764) on overlapping data.

### 3.2 Inflection turn distribution

For each trajectory, define `lock_fail` as the earliest turn T such that probe_score(t) < 0.30 for all t in [T, end] (and similarly `lock_succ` for >0.70).

| Outcome class | n | Lock-in rate | Median lock turn | Median fraction of length |
|---|---|---|---|---|
| Success | 40 | 100% | 33 (Q1=25, Q3=38) | **0.88** |
| Failure (LOCKED sub-class) | 39 | — | 46 (Q1=43, Q3=49) | **0.92** |
| Failure (WANDERING sub-class) | 20 | — | — (never reaches fail-lock) | — |

Two key observations:
1. **Lock-in happens only in the final ~10% of trajectory length** — both success and failure. The probe commits to its verdict very late.
2. **34% of failures never commit to the failure regime** — these are the WANDERING sub-class.

### 3.3 The WANDERING sub-class

Statistics for the 20 WANDERING failures vs 39 LOCKED failures:

| Metric | LOCKED | WANDERING | Mann-Whitney p |
|---|---|---|---|
| Final probe score (median) | 0.000 | **1.000** | <0.0001 |
| First-turn probe score (median) | 1.000 | 1.000 | 0.92 |
| Patch produced (rate) | 79% | **95%** | — |
| Patch size if any (median bytes) | 4172 | 4065 | — |
| Number of 0.5-crossings | 15 | 18 | 0.35 |

WANDERING agents:
- End with probe-confident-success (probe says they did it)
- Almost always produce a patch (95%)
- Cross the 0.5 threshold many times (mean 18 transitions over ~50 turns)
- But behaviorally fail (don't emit `finish_tool`)

### 3.4 Qualitative trace inspection

Read the last 3 turns of 4 representative trajectories.

**WANDERING #1** (qutebrowser-2dd8966, patch=2320B, final probe=1.00): At turn 49, agent runs `pytest` again, then `view`s a file, "All tests pass. Let me verify the final state of the modified files." Agent stuck in verification mode.

**WANDERING #2** (qutebrowser-9b71c1ea, patch=3520B, final probe=1.00): At turn 49: *"The tests are failing due to the lack of a display server (Xvfb) - this is an infrastructure limitation, not a code issue. The code changes have been successfully made and verified."* Agent concludes test failures aren't its fault; doesn't emit `finish_tool`.

**LOCKED #1** (qutebrowser-f631cd44, patch=4186B, final probe=0.01): At turn 49, agent runs `ls -la` on `configdata.py`, then thinks "the YAML file parses correctly. Let me now check that the config data can be loaded properly." Still uncertain whether changes are correct.

**LOCKED #2** (qutebrowser-ef5ba1a, patch=0B, final probe=0.00): At turn 49, agent runs `grep` for code patterns. Still doesn't have a patch. Still understanding the problem.

The taxonomy is qualitatively grounded: WANDERING = over-confident-completion; LOCKED = persistent-uncertainty.

### 3.5 Predictability from early turns

Within failures, does early-turn probe score predict the eventual lock_fail turn?

| Early turn T | Spearman rho | p-value | n |
|---|---|---|---|
| 1 | -0.132 | 0.42 | 39 |
| 3 | +0.093 | 0.57 | 39 |
| 5 | -0.138 | 0.42 | 37 |
| 10 | +0.073 | 0.66 | 38 |

**Early-turn residuals do NOT predict inflection timing.** All correlations near zero, all p > 0.40.

## §4 — Implications

### 4.1 Safety monitoring blind spot

A probe-based monitoring scheme that classifies trajectories as "safe to continue" when probe > 0.7 has a **34% false-negative rate on failed trajectories** — the WANDERING sub-class is invisible to probe alone. Practical remediation:

- **Complementary action-emission monitor**: explicit detection of trajectories that produced patches but didn't emit `finish_tool` within budget
- **Temporal pattern analysis**: detection of "verification mode" via repeated test/view actions without progress
- **Late-turn polling specifically**: lock-in turn is concentrated in final 10% of length, so monitoring needs to extend into late trajectory

### 4.2 Mechanistic interpretation

The probe captures the model's **internal belief** about task completion, not the **external behavioral outcome** of emitting `finish_tool`. The 34% disagreement quantifies this distinction.

This is a **natural-trajectory analog** of the intervention-induced obfuscation in Paper 11. Both are forms of "internal belief vs external action" failure modes:

- Paper 11 (intervention): CoT-redirect prefill → reasoning-clean ∧ patch-sabotaged
- This work (natural): no intervention → probe-positive ∧ action-incomplete (WANDERING)

### 4.3 Relation to ELPO (irrecoverable step via rollouts)

The Error-Localized Policy Optimization paper ([arXiv 2602.09598](https://arxiv.org/pdf/2602.09598)) identifies the "first irrecoverable step" via binary-search rollouts. Our method is rollout-free (probe lock-in) and identifies a related but distinct quantity: not "first irrecoverable step" but "first stable-verdict step." The 0.88-0.92 length fraction of lock-in suggests irrecoverability happens earlier than verdict commitment.

### 3.6 Cross-layer robustness

Tested whether the WANDERING/LOCKED sub-class assignment generalizes across layers. Re-trained probe (top-10 diff-means + StandardScaler + LR, 5-fold CV) at pre_tool position for L11, L23, L31, L43, L55.

**Per-layer CV AUROC:**

| Layer | CV AUROC | WANDERING rate among failures |
|---|---|---|
| L11 | 0.873 ± 0.094 | 14/59 = 24% |
| L23 | 0.933 ± 0.038 | 19/59 = 32% |
| L31 | 0.928 ± 0.049 | 23/59 = 39% |
| **L43** | **0.934 ± 0.041** | **20/59 = 34%** |
| L55 | 0.888 ± 0.055 | 33/59 = 56% |

**Cross-layer agreement on sub-class assignment** (failures, n=59):

| Layer pair | Agreement | Wandering-both |
|---|---|---|
| L23 vs L43 | **85%** | 15/59 |
| L31 vs L43 | 85% | 17/59 |
| L23 vs L31 | 83% | 16/59 |
| L11 vs L43 | 76% | 10/59 |
| L43 vs L55 | 78% | 20/59 |
| L11 vs L55 | 61% (worst) | 12/59 |

Middle layers L23/L31/L43 agree ~85% on sub-class assignment. The WANDERING phenomenon is **not a L43-specific artifact**. Top layer L55 captures a more lenient sub-class boundary (56% wandering vs ~30% in middle layers), suggesting different representational content.

### 3.7 Bootstrap CI on the WANDERING rate

At L43 pre_tool, 10,000-sample bootstrap (resampling 59 failure trajectories with replacement):

- Point estimate: **33.9%** (20/59)
- 95% CI: **[22.0%, 45.8%]**
- Bootstrap mean: 33.9%

The CI excludes 0% and excludes any "noise" hypothesis. WANDERING is a substantial sub-class.

## §5 — Closing the Blind Spot: A Two-Signal Detector

The 34% WANDERING blind spot motivates the question: can a cheap complementary signal — derived from trace data already available to a monitoring system — close it? We test three text/action-based signals on the same 99 trajectories.

### 5.1 Signals

- **`m_complete`**: rate of completion-language phrase matches in the last 5 turn outputs (concat of `thinking`+`content`). Regex covers "all tests pass", "code is correct", "verified", "infrastructure limitation", "not a code issue", etc.
- **`m_repeat`**: count of repeated (same-tool, similar-arg) calls in last 5 turns minus 1 (so 0 = no repeats).
- **`m_static`**: number of consecutive end-turns without any file-editing tool call (`str_replace_editor`, `create_file`, `edit_file`, `write_file`).

| Signal | WANDERING (n=20) | LOCKED (n=39) | SUCCESS (n=40) | Mann-Whitney W vs L |
|---|---|---|---|---|
| `m_complete_last5` | 0.240 | 0.097 | **0.450** | **p=0.024** |
| `m_repeat_last5` | 2.35 | 1.56 | 1.70 | **p=0.033** |
| `m_static_end` | 15.4 | 11.3 | 12.2 | p=0.247 |

`m_complete` and `m_repeat` significantly separate WANDERING from LOCKED. The static-edit signal does not. Crucially, **SUCCESS trajectories have higher completion-language rate than WANDERING** — the signal is only useful when conjoined with `not emit_finish AND patch > 0`.

### 5.2 Detector comparison

We define four detectors and evaluate per-trajectory at trajectory END (no early-warning yet).

| Detector | Overall recall | WANDERING caught | LOCKED caught | FP on SUCCESS |
|---|---|---|---|---|
| Probe-only (`final_probe < 0.5`) | 67.8% | 1/20 (5%) | 39/39 (100%) | 0 |
| **Probe OR (`complete≥0.4 ∧ no_finish ∧ patch>0`)** | **78.0%** | **7/20 (35%)** | 39/39 (100%) | **0** |
| Just-completion alone | 22.0% | 6/20 (30%) | 7/39 (18%) | 0 |
| Lax (`complete≥0.2 ∧ no_finish`) | 35.6% | 11/20 (55%) | 10/39 (26%) | 0 |

The strict combined detector improves overall recall by **10.2 pp** with **zero new false positives on successes**. It closes roughly one third of the 34% WANDERING blind spot.

### 5.3 The residual blind spot — a sub-bifurcation within WANDERING

Inspecting the 9 WANDERING trajectories that no monitor catches (`m_complete = 0`) reveals a second sub-mechanism:

- **WANDERING-A — "Verification loop" (≈11/20).** Agent verbalizes completion ("all tests pass", "code is correct"), produces a patch, but never emits `finish_tool`. Detectable by completion-language monitor. Example: `qutebrowser-9b71c1ea` at turn 49: *"The tests are failing due to the lack of a display server - this is an infrastructure limitation, not a code issue. The code changes have been successfully made and verified."*
- **WANDERING-B — "Silent continuation" (≈9/20).** Agent does not verbalize completion at all. Some are still actively editing files at turn 50 (`m_static = 0`); others have been static for many turns (`m_static = 50`, never edited in the last 50 turns). Mean final probe = 0.89. Mechanism: agent never reaches a terminal state — neither completion-confidence nor explicit doubt.

This bifurcation refines the original taxonomy: WANDERING is not monolithic but a mixture of *over-confident-completion-not-emitted* (A) and *no-terminal-state* (B). Only sub-class A is reachable by the text-based monitor. Sub-class B requires a different signal — candidates include cross-layer probe disagreement, KV-cache distribution shifts at late turns, or model-internal residual divergence from the success-trajectory mean.

### 5.4 Proposed operational design

A practical AgentGuard-style monitor with three tiers:

```python
def action_emission_monitor(trace):
    text_last5 = " ".join(t.text for t in trace.turns[-5:])
    has_completion = COMPLETION_REGEX.search(text_last5)
    has_finish = any(c.name == "finish_tool" for t in trace.turns for c in t.tool_calls)
    has_patch = sum_patch_bytes(trace) > 0
    turn_frac = len(trace.turns) / trace.max_turns

    if has_completion and not has_finish and has_patch and turn_frac > 0.7:
        return "red"      # WANDERING-A confident detection
    if turn_frac > 0.9 and not has_finish and has_patch:
        return "yellow"   # WANDERING-B suspicion (budget-near + no finish)
    return "green"
```

Combined with the probe: `risk = max(1 - probe_score, monitor_level)`. Gates:
- `risk < 0.3` → pass
- `0.3-0.7` → escalate (extend budget OR human review)
- `> 0.7` → block OR force-emit `finish_tool`

### 5.5 What this does NOT solve

1. **45% residual WANDERING blind spot** (WANDERING-B). Real production deployment of this monitor still misses ~15% of failures (`9/59`) that look like over-confident-success internally and silent-continuation externally.
2. **Early-warning gap.** The detector is evaluated at trajectory END (turn 50/50). A practical monitor needs to fire earlier — at e.g. turn 30 of 50 with budget remaining. We have not tested whether the completion-language signal appears early enough.
3. **Cross-task generalization.** All three signals are evaluated on SWE-bench Pro only. The completion-language regex is task-domain-specific (mentions "tests", "patch", "code").

## §6 — Limitations + future work

1. **N=99 single model + task family** (Qwen3.6-27B + SWE-bench Pro). Cross-task validation needed before generalizing.
2. **Single probe + single layer** (L43 pre_tool, top-10 diff-means). Per-layer cross-check would strengthen the WANDERING/LOCKED taxonomy.
3. **Threshold choice (0.30/0.70)** is somewhat arbitrary; sensitivity analysis with multiple thresholds shows robust ordering (LOCKED > WANDERING in fail-lock at 0.30 and 0.40).
4. **Trace inspection sample size (n=4)** is small; full read of 10 WANDERING + 10 LOCKED would strengthen the qualitative claim.
5. **"Internal belief" interpretation** is unproven; probe captures whatever correlates with outcome in the residual.

## §7 — Conclusion

A 34% probe-vs-outcome disagreement sub-class (WANDERING) is identified in natural Qwen3.6-27B SWE-bench Pro agent failures via cheap probe lock-in analysis. The sub-class is a mechanistically distinct failure mode characterized by agent over-confidence: model believes task is complete, makes patches, but never emits completion action. A complementary text-based action-emission monitor closes ~30% of this blind spot at zero false-positive cost, raising overall detector recall from 67.8% to 78.0%. The blind spot further bifurcates into WANDERING-A (over-confident, verbalized) and WANDERING-B (silent continuation), with WANDERING-B remaining undetectable to both probe and text monitor. The findings have direct implications for probe-based agent safety monitoring schemes deployed in production.

## Reproducibility

- Dataset: OpenInterp Phase 6 captures (99 trajectories × 5 layers × per-turn residuals, bf16) at openinterp.org/research/papers/conditionally-causal-probes
- Probe: `artifacts/agent_probe_guard_qwen36_27b/probe_L43_pre_tool.joblib`
- Analysis scripts: `scripts/inflection_turn_analysis.py`, `scripts/trajectory_probe_horizon_cv.py`
- Output: `scripts/inflection_turn_out/inflection_results.json`
- Apache-2.0 throughout

## Cited

- Paper-MEGA / Vicentino 2026: "Conditionally-Causal Probes: Five Operational Constraints on Linear-Probe Causality in Qwen3.6-27B" (openinterp.org/research)
- Paper 11 / Vicentino 2026: "Cleaning the Chain-of-Thought Is Not Correcting the Agent" (draft)
- [arXiv 2602.09598] Error-Localized Policy Optimization (ELPO)
- [arXiv 2602.09924] LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations
- [arXiv 2604.28129] Latent Adversarial Detection (adversarial restlessness)
- [arXiv 2602.19008] Canonical Path Deviation as a Causal Mechanism of Agent Failure
