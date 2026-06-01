# PRE-REGISTRATION — Context-Rot Has a Mechanistic Fingerprint (and is it causal?)

**Author:** Caio Vicentino · OpenInterpretability · pre-registered 2026-06-01 (before looking at any Stage-1 result).
**Model:** Qwen3.6-27B (reasoning). **Data:** existing Phase-6 N=99 SWE-bench Pro agent trajectories with per-turn residual captures at L11/23/31/43/55 (`swebench_v6_phase6`), labeled `sub_class ∈ {success:40, locked:39, wandering:20}`.

---

## 0. One-line thesis

Long-context degradation ("context rot") — the well-documented behavioral fact that LLMs get worse as context fills, *even with perfect retrieval* (arXiv:2510.05381) — has a **measurable fingerprint in the model's residual/feature geometry that worsens as context fills, diverges between successful and failing runs at matched context length, and is at least partially causal** (restoring the geometry restores behavior). If true, our published WANDERING arc becomes a corollary: WANDERING is the behavioral endpoint of representational context-rot, and the Modality-Matters fresh-interruption rescue works because it **resets the degraded representation**.

## 1. Why this is open (literature, June 2026)

- Context-rot is a named, high-stakes problem (industry guides; "~65% of enterprise AI failures attributed to context drift"). Confirmed to be **representational, not retrieval**: "Context Length Alone Hurts Despite Perfect Retrieval" (arXiv:2510.05381).
- The known **mechanism is attention-level only**: lost-in-the-middle, distractor interference. Targeted web search returned **no residual/SAE-feature-geometry account over context**, and explicitly: *"results focus on compression, not on a mechanistic explanation of why fresh context restores performance."*
- Superposition geometry is studied **statically** (arXiv:2605.00842), not **as a function of context position**. SAE "dark matter" / reconstruction error is **characterized** (arXiv:2410.14670) but never used as a **temporal off-distribution signal over a trajectory**. Residual-stream dynamical-systems / attractors are studied **over depth** (arXiv:2502.12131), not **over agent turns**.
- **Our moat:** full-stack SAE (11 layers) on a reasoning model **×** real long-horizon agent trajectories with per-turn activation capture. Almost nobody has both.

## 2. The fingerprint metrics

Computed per trajectory as a function of turn index `t` (one residual vector per turn per layer = mean over that turn's capture positions).

**Stage 1a — RAW geometry (no SAE, no download, CPU-minutes):**
- **M1 `norm`** — ‖v_t‖ (does the stream inflate?).
- **M2 `velocity`** — 1 − cos(v_t, v_{t−1}) (state movement; → 0 = freezing toward an attractor).
- **M3 `drift`** — cos(v_t, v_0) (how far the state has left its starting region).
- **M4 `eff_dim`** — windowed participation ratio PR = (Σλ)²/Σλ² over the last W turns' covariance (low/collapsing = capacity degrading to a low-D subspace).

**Stage 1b — SAE features (requires `caiovicentino1/qwen36-27b-sae-fullstack`; subsample 1 vec/turn/layer → CPU-feasible):**
- **M5 `L0`** — active-feature count per turn (does sparsity break — more features firing = interference?).
- **M6 `interference`** — mean pairwise cosine of *active* feature decoder directions (geometric overlap = superposition interference) OR activation-mass concentration.
- **M7 `dark_matter`** — SAE reconstruction-error norm ratio ‖v − SAE(v)‖/‖v‖ per turn (rising = model entering states the feature dictionary covers worse = internal OOD).

## 3. Pre-registered predictions

- **P1 (trend):** at least one fingerprint metric changes **monotonically** with turn index within trajectories (signed slope consistent across trajectories).
- **P2 (the real test — early divergence at matched position):** at least one metric **diverges between SUCCESS and WANDERING/failure trajectories at matched EARLY turn index** (in the turn range where both classes still have live trajectories), **before** the behavioral outcome is obvious. Mann-Whitney U, p<0.05, surviving the length-confound control (§5).
- **P3 (Stage 2, predictive):** the fingerprint predicts eventual failure with **lead time and/or AUROC ≥ the behavioral tool-entropy detector** (v1–v6 baselines).
- **P4 (Stage 3, causal/mediation):** (a) re-analyzing the Modality-Matters #4 data, the benefit of the B0 fresh-interruption is **mediated** by restoration of the fingerprint metrics; (b) new runs: an intervention that restores the geometry (context compression / reset / feature denoise) restores behavior, and the representational restoration **mediates** the behavioral rescue.

## 4. Decision gates (set before any result)

- **GO past Stage 1** iff **both P1 and P2 hold for ≥1 metric** (monotone trend AND early matched-position success-vs-failure divergence that survives the length control). Then proceed to Stage 2.
- **NO-GO / honest negative** iff metrics are flat, OR only diverge at LATE turns where only failures survive (= length artifact, not a fingerprint), OR the divergence is fully explained by trajectory length (§5). Document as an honest negative and stop — *do not* salvage.
- Stage 2 GO→Stage 3 iff the fingerprint matches or beats the behavioral baseline on lead-time/AUROC.

## 5. Controls (mandatory — the program's hard-won discipline)

- **Length confound (the #1 trap):** failures run to `max_turns` (longer), successes finish early. So (a) restrict the matched-position contrast to the **early overlap region** where both classes exist; (b) **partial out total trajectory length** from any metric–outcome association (regress metric ~ outcome + length); (c) additionally contrast within **length-matched subsets**. A metric that only "degrades" because failing runs are longer is a **null**.
- **Shuffle-time null:** within-trajectory turn-order shuffle must destroy any genuine temporal trend (per the κ_t C2 control).
- **Position consistency:** use the same capture position aggregation across turns; report sensitivity to position choice.
- **Raw-vs-SAE:** if RAW metrics already pass, the SAE metrics are the *interpretable* layer, not the existence proof — report both; don't let an SAE artifact masquerade as signal.
- **Multiple comparisons:** ≥4 metrics × layers → report which survive correction; pre-commit that **one robust metric** is enough to GO, but the headline must name it honestly.

## 6. Compute & honesty

- **Stage 1a:** existing captures, pure numpy, **CPU**, ~minutes. No GPU, no download.
- **Stage 1b:** + SAE download, subsampled encoding, **CPU**, ~minutes–1h.
- **Stage 3b:** new agent runs on RTX 6000 (only if Stages 1–2 GO).
- Result is reported whichever way it falls. A flat/length-explained Stage 1 is a publishable honest negative ("context-rot has no residual-geometry fingerprint at this granularity") and a clean kill — consistent with [[feedback_walk_back_and_rescue_paper_structure]] and the arc's ethos.

## 7. Artifacts

- `scripts/context_rot_stage1.py` — Stage-1a analysis (+ `--smoke` synthetic self-test that must detect a planted signal and reject a planted null before trusting the real run).
- Output: `scripts/context_rot_out/stage1_metrics.json` + per-metric plots (metric vs turn, by outcome) + the GATE verdict.

---

## 8. Stage-2 ADDENDUM (P3, predictive) — pre-registered 2026-06-01 BEFORE computing any AUROC

Stage-1a produced a QUALIFIED GO on **velocity-freezing** (`STAGE1_RESULTS.md`). Stage 2 asks the decisive product question: **is the representational fingerprint worth more than the cheap probe-free behavioral detector it would replace/augment?** The velocity DIRECTION is already known from Stage 1 (failures freeze); Stage 2 pre-commits the *evaluation procedure* (below) before any predictive number is seen.

**Task.** Primary: WANDERING(20) vs SUCCESS(40), binary. (Secondary: WANDERING vs rest, n=99.)

**Candidate predictor.** Early-window mean `velocity` (same outcome-blind early window as Stage 1) per layer; **primary = L31** (the prereg-passing layer), also report best-layer (flagged as selected) and pooled mean-across-layers.

**Behavioral baselines to beat (from `features_n99.csv`).**
- `tool_entropy_first10` — the **fair early** baseline (same early horizon as velocity).
- `tool_entropy_last10` — the **strong late** baseline (sees the last 10 turns ≈ near outcome; velocity is NOT allowed that info).
- `tool_entropy_full`, and `probe_score_first` (early trajectory-probe) as secondary references.
- `n_turns` — **length floor**: velocity must beat trajectory-length-alone or it is just length.

**Metric.** Rank-AUROC for predicting FAIL under the **pre-registered direction** (velocity: lower→fail; tool-entropy: lower→fail = collapse). 2000× bootstrap 95% CI per AUROC; **paired-bootstrap 95% CI of the difference** AUROC(velocity) − AUROC(tool_entropy_first10) (resample trajectories, recompute both). A feature whose AUROC<0.5 under its pre-registered direction = wrong-direction = no signal (reported as-is, not flipped).

**Lead-time.** velocity alarm = first early-turn with velocity < θ, where θ is calibrated so **≤5% of SUCCESS trajectories ever fire** (matches the Tool-Entropy paper's 5% FP operating point). Compare velocity alarm turn + catch-rate on the 20 WANDERING trajectories vs the deployed `detector_v5_fire_turn` (and v1/v4). Earlier alarm and/or higher catch-rate at matched 5% FP = win.

**GATE (Stage 2 → Stage 3).**
- **GO** iff velocity (at L31 or pooled) **matches or beats** the cheap tool-entropy baseline on **≥1** of {early AUROC: paired-bootstrap CI of the difference vs `first10` includes 0 or is positive; lead-time: fires no later than v5 at matched 5% FP, ideally earlier} **AND is not clearly worse on the other**.
- **NO-GO / honest negative** iff velocity AUROC is clearly below tool-entropy AND shows no lead-time advantage → headline: *"representational freezing is real but adds nothing over the cheap behavioral signal"* — STOP, publish the negative.
- Velocity must also clear the **length floor** (AUROC > AUROC(n_turns)); failing that = length artifact = NO-GO.

## 9. Stage-3 ADDENDUM (P4, causal/mediation) — gated by Stage-2 GO
- **3a (re-analysis, CPU if data exists):** in the Modality-Matters #4 data, does the B0 fresh-interruption benefit get **mediated** by restoration of velocity (post-interruption velocity rises back toward the SUCCESS regime on rescued trajectories, and that rise statistically mediates the finalization gain)? Mediation = bootstrap indirect effect.
- **3b (new runs, GPU — prep only, hand to Caio):** an intervention that directly **restores the geometry** (context reset / compression / feature denoise at the freezing turn) restores behavior, and the representational restoration mediates the behavioral rescue. Honest-negative-publishable either way.

## 10. Stage-1b ADDENDUM (SAE feature reading — secondary, not an existence proof)
Run only as interpretability color on the velocity finding: does the freezing have a feature-level signature (L0 collapse / interference rise / dark-matter rise tracking the velocity drop)? Requires `caiovicentino1/qwen36-27b-sae-fullstack`. Not gated by Stage 2; reported descriptively.
