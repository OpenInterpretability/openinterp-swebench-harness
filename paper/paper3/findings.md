# Paper #3 — Findings: Multi-Channel Mechanistic Signatures of Agent WANDERING

**Date**: 2026-05-27
**Status**: Findings validated post-sanity-check; draft pending.
**Data**: N=99 Qwen3.6-27B SWE-bench Pro trajectories (SUCCESS=40, LOCKED=39, WANDERING=20)
**Hardware**: RTX 6000 Pro Blackwell 96 GB (all runs — original Phase 6 + Exp B/L11 re-runs; no H100 was used for SWE-bench Pro work)
**Total compute cost**: $6.19 (Liu et al. taxonomy bridge via OpenRouter Opus 4.7) + ~$0 local

---

## Executive summary

We extract 60 multi-channel features (text / tool-use / residual L11-L55 / temporal) per trajectory and fit a 3-way L1 multinomial classifier (WANDERING vs SUCCESS vs LOCKED) with nested 5×3 CV and 1000-permutation null. Headline: **macro-F1 = 0.636 ± 0.035, z = +5.88, p = 0.001**.

Three findings are paper-grade:

1. **SUCCESS is robustly distinguishable from failures** (AUROC 1.0), dominated by `tool_diversity_count` (the agent uses 3 unique tools vs 2 for failures).

2. **LOCKED vs WANDERING boundary is mechanistically WEAK** (pairwise z=1.81, p=0.035), consistent with Liu et al. (2509.13941) text-based human taxonomy placing ~60% of BOTH sub-classes into category C2.1 (Non-Progressive Iteration).

3. **Stability selection independently surfaces the paper #1 §13 mechanism**: LOCKED dominated by L43 mid-layer convergence (`L43_cosine_consec_late`), WANDERING by L11 edge-layer drift (`L11_drift_first_last`). Confirmed via ultra-conservative analysis (drop all L43 features → L11 signature survives, z=1.88).

**Bonus**: Phase 4 run-to-run stability prediction (n=20 WANDERING; 7 flipped to finish_tool on an independent no-hook re-run, same RTX 6000, temperature=1.0) also surfaces L11_cosine_consec_late as top predictor (AUC 0.725, p=0.096 marginal). Double independent confirmation of L11 as mechanistic locus.

---

## Walk-back-and-rescue (transparency)

Per `feedback_walk_back_and_rescue_paper_structure`, we report the initial leaky-baseline alongside the cleaned analysis:

| Run | Macro-F1 | z | p | Note |
|---|---|---|---|---|
| **Leaky baseline** | 0.987 ± 0.030 | +12.43 | 0.001 | Used `probe_score_*`, `n_turns`, `patch_n_bytes`, `detector_v4_fire_turn` |
| **Clean (primary)** | 0.636 ± 0.035 | +5.88 | 0.001 | Removed 9 label-leaking features |

**Why leaky**: `probe_score_*` features are derived from the same trained probe used to bin sub-class via `lock_fail_0.40` threshold. `n_turns` and `patch_n_bytes` correlate with `finish_reason` (which defines SUCCESS). `detector_v4_fire_turn` correlates with assignment process. Using these as features creates circular reasoning.

**Mitigation**: ultrathink pre-paper sanity check caught this. Clean re-run uses only text + tool + residual L11-L55 + thinking features. F1 drops to 0.636 but remains highly significant. We report both in the paper for full transparency.

---

## Methodology

### Data
- N=99 SWE-bench Pro trajectories collected via OpenInterp custom agent loop on Qwen3.6-27B (`enable_thinking=True`, temperature=1.0)
- HF dataset: `caiovicentino1/swebench-pro-qwen36-27b-phase6` @ revision `a12189c8b92102084140a2dac5f9a27ddd879ace`
- Captures: per-turn residual activations at L11 / L23 / L31 / L43 / L55 (bf16, d_model=5120)
- Sub-class derivation:
  - SUCCESS: `label==1` (test passes after `finish_tool` emission), n=40
  - LOCKED: `label==0 AND lock_fail_0.40 is not None` (probe stably <0.40), n=39
  - WANDERING: `label==0 AND lock_fail_0.40 is None` (probe stays >0.70, no `finish_tool`), n=20

### Feature extraction (4 channels, 63 raw → 60 used)

**TEXT** (11 features): doubt/exploration/completion word densities (per-trajectory + late-half), thinking-length mean/std/final, content-length mean/final, thinking-to-content ratio, n-gram overlap late.

**TOOL** (9 features): Shannon entropy of tool calls (full / first-10 / last-10), unique tools count, top-tool fraction, switching rate, longest same-tool run, calls-per-turn mean, total calls.

**RESIDUAL** (32 features): per-layer (L11/23/31/43/55) norm mean/std/late-mean, cosine of consecutive turn residuals (mean + late-only), first-vs-last drift cosine. Plus 2 cross-layer aggregates (norm range late, norm CV late).

**TEMPORAL** (11 features): n_turns, patch_n_bytes, probe_score_first/final/trajectory-mean/late-mean/volatility, probe_first_threshold_cross_turn, detector_v1/v4/v5_fire_turn.

**Removed in clean run** (label-leakage): `probe_score_*` (6 features, definitional), `n_turns` + `patch_n_bytes` (termination proxies), `detector_v4_fire_turn` (assignment-process correlate). Also dropped 2 all-NaN cols (`detector_v1/v5_fire_turn` — implementation mismatch with paper #1 detectors) + 1 zero-variance col (`probe_first_threshold_cross_turn`). Final: **52 features**.

### Pipeline
- 3-way L1 multinomial logistic regression (saga solver, max_iter=2000)
- Nested cross-validation: outer 5-fold stratified, inner 3-fold GridSearch over C ∈ {1e-3 ... 1e2}
- Modal C selected: 1.0 (well-calibrated; not at grid boundary)
- StandardScaler inside Pipeline (no test-leakage)
- 1000-permutation null (label shuffle, retrain, p-value via Phipson-Smyth `(1 + #{null≥obs})/(1 + n)`)
- Stability selection: 200 bootstrap resamples, feature stable iff selected in ≥80% (Meinshausen-Bühlmann)
- Pairwise contrasts: appendix only, 200-permutation null each

---

## Primary results (3-way classifier, N=99)

| Metric | Value |
|---|---|
| Macro-F1 (5×3 nested CV) | **0.636 ± 0.035** |
| Per-class AUROC: SUCCESS | 1.000 ± 0.000 |
| Per-class AUROC: LOCKED | 0.864 ± 0.030 |
| Per-class AUROC: WANDERING | 0.795 ± 0.062 |
| Null mean ± std | 0.322 ± 0.053 |
| **Z-score** | **+5.88** |
| **p-value (1000 perms)** | **0.001** |

**Confusion matrix** (rows = true, cols = predicted):

```
              pred LOCKED  pred SUCCESS  pred WANDERING
LOCKED        24           0             15
SUCCESS        0          40              0
WANDERING     13           0              7
```

Reading: 40/40 SUCCESS correct. LOCKED: 24/39 correct, 15 misclassified as WANDERING. WANDERING: 7/20 correct, 13 misclassified as LOCKED.

**Pairwise contrasts** (appendix):

| Contrast | macro-F1 | Null mean | z | p |
|---|---|---|---|---|
| W vs S | 1.000 | 0.483 | +6.74 | 0.005 |
| **W vs L** | **0.631** | **0.491** | **+1.81** | **0.035** |
| S vs L | 1.000 | 0.485 | +7.74 | 0.005 |

**Key insight**: SUCCESS is robustly distinct from failures (AUROC 1.0). **LOCKED vs WANDERING separation is barely significant (p=0.035)** — they share mechanistic territory.

---

## Stability-selected features (≥0.80 selection frequency)

| Feature | Class | Sel. freq | L1 coef (full-fit) | Per-class median |
|---|---|---|---|---|
| `tool_diversity_count` | SUCCESS | 1.00 | +4.04 | S: 3.0 / L: 2.0 / W: 2.0 |
| `L43_cosine_consec_late` | LOCKED | 0.835 | +1.21 | S: 0.769 / L: 0.839 / W: 0.814 |
| `L11_drift_first_last` | WANDERING | 0.95 | -0.71 | S: 0.973 / L: 0.975 / W: 0.956 |

Plus next-tier features (0.4-0.8 selection freq):
- LOCKED: `text_doubt_density_late` (0.775), `text_completion_density` (0.735), `L11_drift_first_last` (0.68 — also picks for LOCKED), `L31_norm_std` (0.645)
- WANDERING: `text_thinking_length_final` (0.65), `text_exploration_density` (0.61), `L31_norm_std` (0.56)

### Mechanistic interpretation

**SUCCESS** = high tool diversity (uses 3 unique tools vs 2 for failures). Likely reflects "uses all available agent tools to completion" (median Q25=Q75=3.0 is essentially binary).

**LOCKED** = L43 mid-layer convergence (cosine_consec_late 0.84 vs 0.81 WANDERING vs 0.77 SUCCESS). Mid-layer representation freezes in late trajectory; consistent with "model has settled into a fixed (wrong) verdict and stops updating".

**WANDERING** = L11 edge-layer drift (drift_first_last cosine 0.956 vs 0.975 LOCKED/SUCCESS). Edge-layer representation continues to rotate across trajectory; consistent with "model keeps generating new perspectives but never consolidates to action".

### Effect sizes are SUBTLE

Per-class medians differ by only **2-3 percentage points on cosine 0-1 scale**:
- L11_drift: 0.956 vs 0.975 = 1.9pp gap, IQRs overlap (0.95-0.97 vs 0.96-0.98)
- L43_cosine: 0.839 vs 0.814 vs 0.769 = 2.5pp to 7pp gaps

This is a **statistical regularity** captured by L1 selection across folds, NOT a dramatic visual gap. Papers claiming "mechanism X discriminates" must qualify effect size. We report all medians + IQRs.

---

## Phase 4 — Run-to-run stability prediction (n=20 WANDERING)

**Question**: Can RTX 6000 trajectory features predict which WANDERING agents flip to `finish_tool` on an independent no-hook re-run (same RTX 6000 Pro Blackwell, same deterministic seed, temperature=1.0)?

Ground truth (from `paper/exp_b_results.md` Exp B baseline-nohook):
- 7 / 20 flipped to `finish_tool` on the re-run
- 13 / 20 stayed `max_turns` on the re-run

**Method**: L1 logistic on all 62 features (no nested CV due to small n; used leave-one-out CV instead). 1000-permutation null.

| Model | LOO AUC | Acc @ 0.5 | Null mean ± std | z | p |
|---|---|---|---|---|---|
| L1 logistic | **0.725** | 0.700 | 0.457 ± 0.192 | +1.40 | **0.096** |
| Random Forest | 0.363 | 0.550 | 0.399 ± 0.179 | -0.20 | 0.552 |

**Top L1 features** (positive coefficient = "stayed WANDERING" direction):
1. `L11_cosine_consec_late` (coef = -1.76) ← dominant predictor
2. `tool_entropy_full` (-0.74)
3. `probe_score_first` (-0.64)
4. `L55_cosine_consec_mean` (+0.23)
5. `detector_v4_fire_turn` (-0.23)

**Key insight**: **L11 (edge-layer) signature is the dominant re-run-flip predictor**, independently surfacing what Phase 2 stability selection also identified. Negative coefficient: lower L11 cosine consistency late → more likely to flip on re-run (i.e., less run-stable WANDERING).

**Honest caveat**: p=0.096 is **marginal** (above conventional 0.05). At n=20 with 62 features, AUC=0.725 is suggestive but underpowered. Random Forest underperforms (AUC=0.363), consistent with N=20 limiting nonlinear methods.

---

## Phase 5 — Liu et al. taxonomy bridge

Used Claude Opus 4.7 via OpenRouter to classify all 99 trajectories against Liu et al. (arXiv:2509.13941) hierarchical failure taxonomy. Prompt at `scripts/paper3_liu_judge_prompt.txt`. Cost: $6.19, wall: 638s, 0 parse errors.

**Distribution per sub-class** (primary category):

| Our sub-class | NONE_SUCCESS | C2.1 Non-Progressive Iteration | C1.1 Reproduction Fail | C1.2 Verification Fail | C3 Context Amnesia | Other |
|---|---|---|---|---|---|---|
| SUCCESS (40) | **39 (98%)** | 0 | 0 | 0 | 0 | 1 (B1.2) |
| LOCKED (39) | 0 | **23 (59%)** | 9 (23%) | 3 (8%) | 2 (5%) | 2 (B-codes) |
| WANDERING (20) | 0 | **12 (60%)** | 2 (10%) | 3 (15%) | 0 | 3 (B-codes) |

**Key insight**: Liu et al. text-based human taxonomy places **60% of BOTH LOCKED and WANDERING into C2.1 (Non-Progressive Iteration)**. Independent text-based annotation cannot distinguish these mechanistic sub-types. Our supervised contrast (W vs L pairwise p=0.035) is consistent with this: both axes report the boundary is fuzzy.

**Honest framing for paper**:
> "Two failure modes that human/LLM text-based taxonomies cannot distinguish (Liu et al.'s 'Non-Progressive Iteration' captures 60% of both) are weakly separable via mid- vs edge-layer mechanistic signatures (z=1.81, p=0.035), suggesting that probe-derived sub-class definitions may impose categorical boundaries on what is mechanistically a continuum."

---

## Ultra-conservative sensitivity (sanity check)

To address second-order leakage concern (L43 features are correlated with probe substrate since the probe was trained on L43 residuals), we dropped ALL L43 features + `tool_diversity_count` (dominant SUCCESS) and re-fit pairwise W vs L:

| Metric | Value |
|---|---|
| W vs L LOO AUC | 0.696 ± 0.133 |
| Null mean ± std | 0.498 ± 0.105 |
| z | +1.88 |
| **p** | **0.030** |

**Top features after ultra-conservative drop** (45 features remaining):
1. `L31_norm_late_mean` (coef = -1.02)
2. `L55_cosine_consec_late` (-0.89)
3. **`L11_drift_first_last` (-0.71)** ← survives
4. `text_completion_density` (+0.69)
5. `text_thinking_length_final` (+0.69)

**Verdict**: L11 edge-layer drift survives ultra-conservative L43 drop. Refined mechanism hypothesis: "**edge-layer disturbance**" = L11 drift + L55 instability, not just L11 alone. Consistent with Phase 4 (L11_cosine_late top predictor) and with paper #1 §13 hypothesis.

---

## Honest caveats

### 1. Effect sizes are subtle (2-3pp on cosine scale)
Reviewers will expect "smoking gun". We have statistical signature, not visual gap. Frame appropriately.

### 2. L11_drift correlates with text_thinking_length_final (ρ = 0.63)
Possible alternative interpretation: WANDERING agents think MORE in final turn → different L11 dynamics → L11_drift selected as proxy for "thinking volume", not "edge-layer mechanism". Paper should: (a) report joint analysis, (b) test conditional on thinking_length, (c) acknowledge as open question.

### 3. N=20 WANDERING is binding for within-class analysis
Phase 4 run-to-run stability (n=20) has limited statistical power even with L1 + LOO. Re-run-flip prediction p=0.096 is marginal. Replication at larger N required for strong claim.

### 4. tool_diversity_count is essentially binary
SUCCESS = 3.0 (Q25=Q75=3.0) is suspicious. Likely reflects "did agent reach the third available tool (e.g., `edit` after `bash`+`view`)" rather than "general tool diversity". Real signal, but interpretation matters.

### 5. Second-order L43 leakage (acknowledged, addressed)
L43 is substrate of trained probe. Ultra-conservative drop confirms L11 signature survives, but reviewer should note L43-cosine isn't probe-score-derived directly.

### 6. Single-model scope (Qwen3.6-27B)
All findings on one model on one benchmark (SWE-bench Pro). Cross-model replication is future work. Paper #1 has partial cross-arch (Llama-70B, GPT-5) but only for v5 tool-entropy detector, not the full multi-channel phenotype.

---

## Connection to paper #1 (Tool-Entropy Collapse)

Paper #1 (`project_tool_entropy_paper_published_2026-05-24`, Zenodo DOI 10.5281/zenodo.20368601) §13 hypothesis:
> "WANDERING = mid-layer verdict consolidated but edge-layer (L11/L55) alignment failure"

Paper #3 Phase 2 stability selection (using NONE of paper #1's detector designs as features) independently surfaces:
- LOCKED → L43 mid-layer convergence (cosine_consec_late) = "mid-layer freezes / verdict consolidated"
- WANDERING → L11 edge-layer drift (drift_first_last) = "edge-layer fails to consolidate"

**This is independent mechanism confirmation.** Paper #1 used detector-design + cross-architecture validation. Paper #3 uses supervised contrast + stability selection on internal features. Both arrive at the same mechanistic claim from different paths.

---

## Connection to paper #2 (Two Honest Nulls, draft)

Paper #2 (`project_exp_b_plan_b_null_2026-05-26`) tested paper #1 mechanism causally:
- Exp B Plan B: L55 SUCCESS-donor injection had NO causal effect (α=0.3)
- Exp D: forced-finish counterfactual NULL
- Interpretation #1 of nulls: "L55 isn't the right edge-layer locus"

**Paper #3 Phase 2 (this work) refines paper #2's interpretation**: stability selection identifies **L11 (not L55) as the primary WANDERING discriminator**. Implication: future causal interventions should target L11, not L55. Paper #2's null + paper #3's discovery = coherent refinement of paper #1.

---

## Artifacts

| Artifact | Path |
|---|---|
| Feature CSV (N=99 × 65 cols) | `scripts/inflection_turn_out/features_n99.csv` |
| Liu taxonomy assignments | `scripts/inflection_turn_out/liu_taxonomy_assignments.json` |
| Supervised contrast (LEAKY baseline) | `scripts/inflection_turn_out/paper3_results/` |
| Supervised contrast (CLEAN primary) | `scripts/inflection_turn_out/paper3_results_clean/` |
| Hardware ground truth | `scripts/paper3_hardware_validation.csv` |
| Phase 4 prediction results | `scripts/inflection_turn_out/paper3_hardware_prediction.json` |
| Sanity checks script | `scripts/paper3_sanity_checks.py` |
| Liu taxonomy reference | `paper/paper3/liu_taxonomy_reference.md` |

---

## Next steps

1. **Paper #3 LaTeX draft** (~1-2 days). Sections: Intro / Related Work (cite Liu 2509.13941, "When Agents go Astray" 2509.02360, "Understanding Code Agent Behaviour" 2511.00197, "Beyond the Black Box" 2605.06890, paper #1) / Methods / Results (Phases 1-2 primary, 4-5 secondary) / Discussion / Limitations / Appendix (leaky baseline + ultra-conservative + per-class stats).

2. **Decision: paper #2 vs paper #3 ship order**. Paper #2 still v0.3 draft. Options: (a) ship #2 first then #3, (b) back-to-back Zenodo releases, (c) merge findings into single longer paper.

3. **arXiv submission path**. Per `project_inspect_evals_submission_2026-05-26`, arXiv blocked until 2027. Path: ship Zenodo first, NeurIPS MI Workshop submission (late August CFP) for arXiv ID via venue.

4. **Reproducibility**: scripts already in repo. Add ipynb walkthrough for community + HF dataset card update with paper #3 reference once published.
