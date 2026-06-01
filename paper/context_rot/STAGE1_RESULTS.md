# Stage-1a RESULTS — Context-Rot Mechanistic Fingerprint (RAW geometry)

**Run:** 2026-06-01, CPU, existing Phase-6 N=99 captures (success:40 / wandering:20 / locked:39).
**Code:** `scripts/context_rot_stage1.py` (smoke-validated: detects a planted class signal, rejects a length-only null).
**Output:** `scripts/context_rot_out/stage1_metrics.json`.
**Contrast:** SUCCESS vs WANDERING, per-trajectory **early-window** mean (independent samples), length-controlled.

---

## Headline verdict: **QUALIFIED GO** — one metric (`velocity`) survives the pre-registered conjunction at one layer, with directional support across all 5 layers; but **nothing survives multiple-comparison correction**. Stage 2 is the real test; do not oversell.

## What passed the pre-registered bar (P1 trend ∧ P2 early divergence surviving length control)

Applying the prereg §4 conjunction strictly (the script's automated GATE only enforced P2 — corrected here):

| Layer | Metric | P1 trend p | P2 div MW p | partial_r \| length | succ vs fail (early) | Verdict |
|---|---|---|---|---|---|---|
| **L31** | **velocity** | **0.0026** | **0.0148** | **−0.242** | 0.0630 > 0.0567 | **P1∧P2 PASS** |
| L11 | eff_dim | 1.00 (no trend) | 0.0276 | −0.157 | 2.547 > 2.410 | P2 only → **disqualified** (fails P1) |

Only **L31 `velocity`** clears the full conjunction.

## The real signal: cross-layer directional coherence of `velocity`

`velocity` = 1 − cos(v_t, v_{t−1}) (state movement; → 0 = freezing toward an attractor). Pre-registered M2 prediction: failing trajectories freeze **earlier**. Observed early-window means:

| Layer | success | fail | MW p | partial_r | sign matches thesis? |
|---|---|---|---|---|---|
| L11 | 0.0041 | 0.0039 | 0.297 | +0.018 | ✓ (n.s.) |
| L23 | 0.0248 | 0.0225 | 0.039 | −0.148 | ✓ |
| L31 | 0.0630 | 0.0567 | **0.015** | −0.242 | ✓ |
| L43 | 0.1119 | 0.1023 | 0.038 | −0.138 | ✓ |
| L55 | 0.1876 | 0.1723 | 0.031 | −0.143 | ✓ |

**5/5 layers same direction** (success moves faster early; failure freezes earlier), **4/5 raw p<0.05**, partial-r negative in 4/5 after controlling for trajectory length. Naive sign-test (1/2)^5 = 0.031 — but layers are NOT independent (shared residual stream), so this is *suggestive*, not a rigorous combined p.

## What did NOT hold (honest)

- **Multiple comparisons:** 20 tests (4 metrics × 5 layers). Min p = 0.0148 (L31 velocity). **Nothing survives Bonferroni (0.0025) or BH-FDR (0.05).**
- **`norm`, `drift`, `eff_dim` are not robust fingerprints.** `drift` shows strong within-trajectory trends (trend_p ~1e-6) but **zero success-vs-fail early divergence** (p≈0.71) → pure ordering artifact, exactly the kind of "trend without discrimination" the prereg flagged as a null.
- **`eff_dim` L11** has discrimination (p=0.028) but **no monotone trend** (trend_p=1.0) → fails P1; not a context-rot *progression* signal.
- The automated GATE in the script under-specified the prereg (checked P2 only). Corrected manually here; will tighten the script's gate to P1∧P2.

## Interpretation

The candidate context-rot fingerprint is **representational velocity-freezing**: trajectories that will WANDER show a smaller per-turn state change early on — the residual state settles toward an attractor sooner — and this is *not* explained by trajectory length. It is directionally coherent across the whole stack and mechanistically continuous with the program's prior finding (Tool-Entropy §13: mid-layer verdict consolidation; here the mid-layer L31 is where the velocity contrast is sharpest and trends monotonically). But at Stage 1 it is an **uncorrected, single-layer-clean** signal.

## Decision (per prereg §4)

**Proceed to Stage 2 on `velocity` ONLY**, pre-registering the directional + lead-time hypothesis BEFORE looking:
- **P3 test:** does early-window velocity-freezing predict eventual WANDERING with AUROC and/or lead-time **≥ the behavioral tool-entropy detector (v1–v6)** on the same N=99? If it cannot match the cheap behavioral baseline, the fingerprint is not worth the geometry — honest kill.
- Drop `eff_dim`/`drift`/`norm` from the headline.
- Stage 1b (SAE features L0 / interference / dark_matter) is **secondary** — run it to test whether the velocity-freezing has a feature-level reading (does L0 collapse / dark-matter rise track the freezing?), not as an independent existence proof.

A Stage-2 result where velocity cannot beat tool-entropy = publishable honest negative ("representational freezing is real but adds nothing over the cheap behavioral signal"). A Stage-2 win = the first representational (not attention-level) account of context-rot with lead time.
