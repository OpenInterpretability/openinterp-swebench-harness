# Wandering vs Locked: A 34% Blind Spot in Probe-Based Agent Failure Monitoring (and Four Attempts to Close It)

**Draft v0.2 — 2026-05-24**

## Abstract

Probe-based safety monitoring of LLM agents typically assumes that a probe trained to predict trajectory success/failure provides a reliable signal at any turn. We test this on 99 trajectories of Qwen3.6-27B running SWE-bench Pro across 3 codebases and identify two mechanistically distinct sub-classes of agent failure: LOCKED (66%) and WANDERING (34%). Both are externally identical — they exhaust the turn budget without emitting `finish_tool` — but differ fundamentally in internal state. LOCKED agents are uncertain (probe collapses to <0.30 by mean fraction 0.92 of trajectory length, 21% never produce a patch). WANDERING agents are over-confident (probe stays >0.70 with median final score 1.000, 95% produce patches). The 34% probe-vs-outcome disagreement constitutes a quantified blind spot for probe-only monitoring schemes.

We then test four detector designs to close this blind spot: (v1) a post-hoc text monitor reaches 35% WANDERING recall at 0% false-positive cost; (v2) a naive early-warning extension fails with 15% SUCCESS false positives because SUCCESS agents also verbalize completion-language; (v3) a persistence-based hypothesis is empirically refuted in the opposite direction (SUCCESS agents persist longer in completion verbalization, p = 0.92); (v4) cross-layer probe disagreement (range, std, sign-disagreement across L11/23/31/43/55) succeeds — late-half disagreement separates WANDERING from SUCCESS (p = 0.0028) at 65% recall × 30% FP × 15-turn lead. The combined v1 ∪ v4 detector closes 80% of the blind spot at advisory-tier deployment cost. Signal validates across all 3 codebases.

Mid-layer ablation reveals the discrimination is **edge-layer-driven**: L23/L31/L43 agree with each other, but L11 (surface processing) and L55 (output planning) disagree with the mid-layer consensus more in WANDERING than in SUCCESS. We reframe WANDERING from "stuck-in-verification-loop" to "mid-layer-to-edge-layer alignment failure" — model has internally consolidated a verdict but the surface circuits that translate verdict into `finish_tool` action have not aligned. We provide deployment guidance for three operational tiers (forensics, advisory escalation, autonomous termination — only first two viable).

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

## §6 — Three text-based early-warning extensions — two honest negatives

The §5 detector evaluates at trajectory END (turn 50/50). A practical monitor needs to fire EARLIER — at e.g. turn 30 of 50 with budget remaining. We test three signal candidates derived from agent text output and report all three (the two negatives motivate the reframing in §7).

### 6.1 v2 — Naive early-warning with WAIT clause

Operational signal:
- alert(T, W, B) = ∃ turn t* ≤ T-W with completion-language phrase AND no `finish_tool` by T AND turn-fraction T/n ≥ B

Where W is a wait window (so we don't alarm on agents merely going through a brief verification phase) and B is a budget-fraction gate (so we don't alarm when there's plenty of budget remaining).

Best operating point sweep over (W ∈ {5, 10, 15, 20, 25}, B ∈ {0.0, 0.5, 0.7, 0.8, 0.9}):

| (W, B) | WANDERING recall | SUCCESS FP | Lead time |
|---|---|---|---|
| (15, 0.8) | 40% | 27.5% | 10 turns |
| **(20, 0.8)** | **30%** | **15%** | **10–14 turns** |
| (25, 0.8) | 20% | 10% | 10 turns |

The structural problem: **40/40 SUCCESS trajectories also verbalize completion-language**, then emit `finish_tool` a median of 11 turns later (q25=6, q75=16). The completion-language signal is NOT temporally exclusive to WANDERING. Best operating point at W=20 still has 15% SUCCESS FP — unacceptable for autonomous termination (would kill 15% of successful trajectories). Viable only for advisory escalation if downstream cost is low.

### 6.2 v3 — Persistence-based signal (hypothesis refuted)

Hypothesis: SUCCESS agents perform a brief verification phase (streak of 1–2 consecutive completion-language turns) then emit `finish_tool`; WANDERING agents enter an indefinite verification loop (streak of 3+ consecutive completion-language turns) and never emit `finish_tool`. If true, `streak_max` thresholded at K ≥ 3 separates them.

Empirical streak distribution:

| Sub-class | mean | median | q90 | max |
|---|---|---|---|---|
| SUCCESS   | 2.23 | 2 | 4 | 5 |
| LOCKED    | 0.82 | 1 | 2 | 3 |
| **WANDERING** | **1.75** | **2** | **3** | **6** |

Mann-Whitney WANDERING > SUCCESS one-tail: **p = 0.92** (the opposite direction is significant). SUCCESS agents verbalize completion-language MORE persistently than WANDERING agents.

This refutes the "stuck-in-verification-loop" model of WANDERING. SUCCESS pattern is methodical: verify multiple times (longer streaks), then commit. WANDERING pattern is SCATTERED: occasional completion-utterances interleaved with exploration, never consolidated into a terminal action.

### 6.3 Reframing WANDERING — convergence failure, not loop

The combined v2 (FP unacceptable) and v3 (hypothesis inverted) negatives motivate a mechanistic reframing:

> **WANDERING is not "agent stuck in verification loop". WANDERING is "agent fails to consolidate a terminal decision despite producing intermittent completion signals".**

This is a CONVERGENCE failure, not a LOOP failure. The implication for monitor design: text-based signals that detect "agent says done" are not sufficient because both SUCCESS and WANDERING agents say done. We need signals that detect "agent has NOT consolidated a verdict" — which suggests looking inside the residual stream, not the text output.

## §7 — Cross-layer probe disagreement: a residual-derived early-warning signal

If WANDERING is a convergence-failure mechanism, different layers of the residual stream should encode DIVERGENT verdicts in WANDERING trajectories (the model has not consolidated). SUCCESS trajectories should show cross-layer CONVERGENCE: all layers agree on success at late turns. We test this directly.

### 7.1 Setup

Refit per-layer probes (L11, L23, L31, L43, L55) with the same 5-fold CV pipeline as §3. Score every turn of every test trajectory at every layer. For each (trajectory, turn) compute three disagreement metrics:

- **range(t)** = max_L probe_L(t) − min_L probe_L(t)
- **std(t)** = σ across L of probe_L(t)
- **sign_dis(t)** = fraction of layer-pairs (i, j) with sign(probe_i − 0.5) ≠ sign(probe_j − 0.5)

Then aggregate per-trajectory: late-half mean (turns from n/2 to n).

### 7.2 WANDERING shows higher late-half disagreement (all three metrics)

| Metric | SUCCESS | LOCKED | WANDERING | W vs S p | W vs L p |
|---|---|---|---|---|---|
| range_late | 0.434 | 0.442 | **0.540** | **0.0028** | 0.0070 |
| std_late | 0.175 | 0.178 | **0.218** | **0.0036** | 0.0088 |
| sign_dis_late | 0.207 | 0.212 | **0.267** | **0.0017** | 0.0047 |

All three metrics significantly separate WANDERING from SUCCESS in the late half of trajectories, with consistent direction and effect sizes 24–29% relative. SUCCESS and LOCKED are NOT statistically separated by any of these metrics — both sub-classes converge to a verdict; only WANDERING stays dispersed.

### 7.3 Operational sweep — viable as advisory tier

Operational signal: `alarm(T) = late_half_mean(range[0:T]) > threshold AND no finish by T`.

| (threshold, T_frac) | WANDERING recall | SUCCESS FP | Lead | LOCKED bonus |
|---|---|---|---|---|
| (0.45, 0.70) | 90% (18/20) | 55.0% | 15 turns | 21/39 |
| **(0.52, 0.70)** | **65% (13/20)** | **30.0%** | **15 turns** | 13/39 |
| (0.55, 0.70) | 45% (9/20) | 27.5% | 15 turns | 9/39 |
| (0.60, 0.80) | 35% (7/20) | 15.0% | 10 turns | 4/39 |

The best operating point at (0.52, 0.70): 65% WANDERING recall, 30% SUCCESS FP, 15-turn median lead time before budget end. **First viable early-warning detector** in this work — significantly better than v2's 30%/15% trade-off.

### 7.4 Complementarity with §5 text detector

The text detector and cross-layer detector catch DIFFERENT subsets of WANDERING:

| | v1 catches | v4 catches |
|---|---|---|
| WANDERING-A (verbalized) | 7/20 | 4/20 |
| WANDERING-B (silent) | 0/20 | 9/20 |
| **Total union (v1 ∪ v4)** | | **16/20 (80%)** |

The 9 WANDERING-B trajectories that text monitor missed entirely (`m_complete = 0` — agent never verbalizes "all tests pass") ARE caught by cross-layer disagreement. This is the predicted complementarity from the §5.3 sub-bifurcation: v4 sees the convergence failure in residual that text doesn't expose.

Combined v1 ∪ v4: **80% WANDERING blind-spot closure** (16/20) at 30% SUCCESS FP and 15-turn lead time.

### 7.5 Mid-layer ablation — the signal is edge-driven

Restricting the disagreement computation to mid-layers L23/L31/L43 (all with AUROC ≥ 0.928) collapses the signal:

| Metric | 5-layer (L11/23/31/43/55) | 3-layer (L23/31/43) |
|---|---|---|
| range_late W vs S p | **0.0028** | 0.0701 (n.s.) |
| Best op (recall × FP) | 65% × 30% | 5% × 0% |

Mid-layers agree with each other. The discrimination comes from **edge layers L11 and L55 disagreeing with the mid-layer consensus** more in WANDERING than in SUCCESS. Mechanistic interpretation: WANDERING agents have consolidated a mid-layer verdict but **early surface processing (L11) and output planning (L55) have NOT aligned** with that verdict. The decision-to-action circuit is desynchronized.

This refines §6.3's reframing of WANDERING: not just "convergence failure" but specifically **"mid-layer-verdict-to-edge-layer alignment failure"**. The model "knows" the answer in mid-layers but the surface circuits that would translate that into a `finish_tool` action haven't caught up.

### 7.6 Cross-task validation across 3 repos

Phase 6 has trajectories from 3 codebases. Per-repo Mann-Whitney WANDERING vs SUCCESS on range_late:

| Repo | n_W | n_S | W median | S median | effect | p | direction |
|---|---|---|---|---|---|---|---|
| ansible | 7 | 13 | 0.542 | 0.415 | +0.128 | **0.019** | ✓ W > S |
| openlibrary | 6 | 16 | 0.523 | 0.437 | +0.086 | 0.367 | ✓ W > S |
| qutebrowser | 7 | 11 | 0.568 | 0.444 | +0.124 | **0.035** | ✓ W > S |

**All 3 repos show the same direction** (WANDERING > SUCCESS in late-half disagreement). 2 of 3 individually significant despite n_W = 6–7 (low power); openlibrary trending same direction. The signal is NOT a qutebrowser-specific artifact.

Combined v1 ∪ v4 detector performance per repo:

| Repo | WANDERING recall | SUCCESS FP | LOCKED |
|---|---|---|---|
| ansible | 71% (5/7) | 23% (3/13) | 100% (13/13) |
| openlibrary | 83% (5/6) | 31% (5/16) | 100% (11/11) |
| qutebrowser | 86% (6/7) | 36% (4/11) | 100% (15/15) |

Operational regime stable: recall 71–86%, FP 23–36%. LOCKED 100% in all 3 repos (probe carries this). Signal generalizes across codebases.

## §8 — Deployment guidance

Different operational tiers require different precision/recall trade-offs. We map our detectors to three concrete deployment scenarios.

### 8.1 Tier 1 — Forensics / log analysis (v1 post-hoc only)

Use case: post-trajectory analysis dashboard. Question: "of these N trajectories, which exhibited the WANDERING failure mode?"

Detector: v1 post-hoc (§5).
- Recall: 35% of WANDERING
- FP: 0% on SUCCESS
- Operates on completed trajectories only

This is shippable now. Zero false-positive cost makes it safe for any downstream use including capability assessment, regression detection, agent-comparison studies.

### 8.2 Tier 2 — Advisory escalation (v1 ∪ v4)

Use case: live monitor on running agents. Action when alert fires: extend budget OR human-review, NOT auto-terminate.

Detector: v1 ∪ v4 combined.
- Recall: 80% of WANDERING
- FP: 30% on SUCCESS (false escalations)
- Lead time: 15 turns before budget end

Acceptable when escalation cost is small relative to the WANDERING capability loss being prevented. Examples: enterprise SaaS where human review is cheap, agent training environments where budget extension is free.

### 8.3 Tier 3 — Autonomous termination (NOT VIABLE)

Use case: kill agent on alert to free compute. Requires very low FP rate (< 5%) because every FP is a killed successful trajectory.

**No detector in this work qualifies for Tier 3.** The best operating point with FP < 5% is v3 (5% FP, 10% recall) which is strictly dominated by v1 post-hoc. Future work must find a signal channel with higher precision than text or cross-layer disagreement provide.

Candidate channels for future investigation:
1. **KV-cache attention pattern shifts** — late-turn attention regression toward early-turn context as a signal of "re-evaluating without progress"
2. **Tool-use sequence entropy** — diversity decay in tool calls (agent stuck calling same 2-3 tools)
3. **Action-state divergence in residual** — distance between consecutive turn residuals dropping below threshold despite continued text generation ("frozen agent")
4. **MoE expert routing instability** — for hybrid MoE models, expert selection variance per turn

## §9 — Limitations + future work

1. **N=99 single model + task family** (Qwen3.6-27B + SWE-bench Pro). Cross-task validation needed before generalizing.
2. **Single probe + single layer** (L43 pre_tool, top-10 diff-means). Per-layer cross-check would strengthen the WANDERING/LOCKED taxonomy.
3. **Threshold choice (0.30/0.70)** is somewhat arbitrary; sensitivity analysis with multiple thresholds shows robust ordering (LOCKED > WANDERING in fail-lock at 0.30 and 0.40).
4. **Trace inspection sample size (n=4)** is small; full read of 10 WANDERING + 10 LOCKED would strengthen the qualitative claim.
5. **"Internal belief" interpretation** is unproven; probe captures whatever correlates with outcome in the residual.

## §10 — Conclusion

A 34% probe-vs-outcome disagreement sub-class (WANDERING) is identified in natural Qwen3.6-27B SWE-bench Pro agent failures via cheap probe lock-in analysis. The sub-class is mechanistically distinct: agent believes task is complete (probe high, 95% produce patches) but never emits completion action.

We test four detector designs:
1. **v1 post-hoc text** (§5): 35% recall, 0% FP, no lead — Tier-1 forensics.
2. **v2 naive early-warning text** (§6.1): 30% recall, 15% FP, 10-turn lead — FP unacceptable.
3. **v3 persistence text** (§6.2): 10% recall, 5% FP — hypothesis refuted (SUCCESS more persistent than WANDERING).
4. **v4 cross-layer disagreement** (§7): 65% recall, 30% FP, 15-turn lead — first viable early-warning signal.

Combined v1 ∪ v4 closes **80% of the WANDERING blind spot** at 30% SUCCESS FP and 15-turn lead time. Signal generalizes across 3 codebases (ansible, openlibrary, qutebrowser). Mid-layer ablation reveals the discrimination is driven by **edge layers L11 and L55 disagreeing with mid-layer consensus** — interpreted as mid-layer verdict consolidation that has not yet aligned with surface processing and output planning. WANDERING reframes from "stuck-in-verification-loop" to "mid-layer-to-edge-layer alignment failure".

Deployment guidance: v1 post-hoc is Tier-1 (forensics, ship now). v1 ∪ v4 is Tier-2 (advisory escalation, requires low escalation cost). Tier-3 autonomous termination is NOT viable from any detector in this work — future channels (KV-cache attention shifts, tool-use entropy, residual action-state divergence) needed.

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
