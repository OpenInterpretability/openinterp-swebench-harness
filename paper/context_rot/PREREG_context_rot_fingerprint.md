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
