# Stage-2 RESULTS — Predictive value of velocity-freezing (P3) — **HONEST NEGATIVE**

**Run:** 2026-06-01, CPU, existing Phase-6 N=99. **Code:** `scripts/context_rot_stage2.py`. **Output:** `scripts/context_rot_out/stage2_predictive.json`.
**Pre-registered** in PREREG §8 BEFORE any AUROC was computed.
**Task:** WANDERING(20) vs SUCCESS(40). Single-feature rank-AUROC (fit-free, no leakage); paired bootstrap for differences.

## Verdict: **NO-GO. STOP.** Representational velocity-freezing is real (Stage 1) but **adds nothing over the cheap probe-free tool-entropy detector**, and is operationally weak as an alarm. Stage 3 (gated by a Stage-2 GO) is NOT entered. The honest negative is the result.

## AUROC table (predict FAIL=WANDERING)

| Signal | AUROC | note |
|---|---|---|
| `n_turns` (length) | **1.000** | postdictive perfect separator — WANDERING runs to max_turns; ANY length-correlate looks predictive |
| `tool_entropy_last10` | **0.888** | the deployed behavioral detector (uses last 10 turns ≈ near outcome) |
| `tool_entropy_full` | 0.835 | |
| **`velocity_L31` (early, the fingerprint)** | **0.695** | prereg-passing layer |
| `velocity_L55` | 0.673 | |
| `velocity_L43` | 0.666 | |
| `velocity_pooled` | 0.660 | |
| `tool_entropy_first10` (fair early baseline) | 0.688 | same early horizon as velocity |
| `velocity_L11` | 0.584 | |
| `probe_score_first` | 0.491 | trajectory probe, early — no early signal |

## Why NO-GO (three independent reasons, any one sufficient)

1. **Ties, does not beat, the fair early behavioral baseline.** velocity_L31 (0.695) vs tool_entropy_first10 (0.688): paired-bootstrap Δ = **+0.0075**, CI95 **[−0.170, +0.211]** — squarely straddles 0. At the only *fair* (early, no-outcome-peek) comparison, the geometry and the cheap behavioral signal are statistically indistinguishable.
2. **Useless as a sharp low-FP alarm.** Calibrated to ≤5% SUCCESS false-positive, velocity catches only **1–3 of 20** WANDERING (L11 1/20, L31 2/20, L55 3/20) and **never co-fires with detector_v5** (n_both_caught = 0 at every layer) → **no lead-time advantage**. Freezing is a soft distributional shift, not a threshold event; tool-entropy collapse is a sharper, more detectable one.
3. **Loses decisively to the deployed detector and to length.** tool_entropy_last10 = 0.888 ≫ 0.695; n_turns = 1.000.

The pre-registered length-floor clause (velocity must beat AUROC(n_turns)) auto-fails here because n_turns is a *postdictive* perfect separator (1.000) in this dataset — that clause is too strict for an *early-prediction* frame and is **not** the reason for the kill. The kill is reasons 1–2: even granting the fair early frame, velocity merely **ties** the cheap early signal and is a **poor alarm**. (Length is noted, not the basis.)

## Interpretation (what this rules out — a real contribution)

Context-rot **does** leave a faint, directionally-coherent representational trace (Stage-1 velocity-freezing, 5/5 layers), **but at this granularity it carries no predictive information beyond the cheap probe-free behavioral detector** it would replace. The internal geometry is **downstream-redundant**, not a privileged early window. This *strengthens* the published WANDERING arc: the behavioral tool-entropy signal is not merely convenient — for this failure mode it is **as good as or better than** reading the residual stream. "Just watch the geometry" is ruled out as a better detector.

## Decision

STOP per PREREG §8 gate. Do NOT run Stage 3 (causal/mediation) or spend SAE compute on Stage 1b — interpreting a signal that adds no operational value is not worth the compute. Publish as a tight honest negative (candidate: short note / appendix to the arc, or a standalone "no-better-than-behavioral" result). Consistent with [[feedback_walk_back_and_rescue_paper_structure]] and the arc ethos: a clean kill, pre-registered, reported whichever way it fell.

**One legitimate caveat on scope (not a salvage):** this tests *prediction/redundancy*, not *causation*. It remains logically possible that the frozen geometry is causally on the path (Stage-3 mediation of the #4 B0 rescue) even while being a redundant *predictor*. But per the pre-registered gate, a predictive null stops the program — chasing causation after a redundancy null would be motivated continuation. If revisited later, it must be re-pre-registered as its own causal question, not as a rescue of this one.
