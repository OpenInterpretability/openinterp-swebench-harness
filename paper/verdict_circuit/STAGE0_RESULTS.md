# Stage-0 RESULTS — The Verdict Circuit (§13 first half) — **PRELIMINARY GO (replicated)**

**Run:** 2026-06-02, CPU, existing Phase-6 N=99 + TopK SAE (k=128) at L23 and L43. **Code:** `scripts/verdict_circuit_stage0.py` (smoke-validated: recovers a planted done-feature, rejects a length-only null). **Pre-registered** in `PREREG_verdict_circuit.md` before any SAE-encoding. **Output:** `scripts/verdict_circuit_out/stage0_L{23,43}.json`.

## Verdict: **PRELIMINARY GO at both mid-layers (L23 strong, L43 weak-directional).** The §13 first half — *WANDERING holds a SUCCESS-typical "done" verdict that LOCKED does not* — is **supported at the SAE-feature level**. NOT a finished result: the semantic label "done verdict" is a hypothesis pending feature interpretation (Stage-1, needs the model), and the §13 second half (why the verdict fails to emit `finish`) is untested.

## What was tested (anti-circular by construction)
A "done-feature" was selected **from SUCCESS only** (fires more at the SUCCESS finish-turn `pre_tool` than at SUCCESS early turns). WANDERING and LOCKED **never touched the selection** — they are the held-out test of the §13 prediction.

| Test | L23 | L43 |
|---|---|---|
| Primary feature | 22358 | 908 |
| **P1** WANDERING final > early (paired) | **p=0.00022** | p=0.022 |
| **P2** WANDERING final > LOCKED final (MW) | **p=1.7e-5, AUROC 0.808** | p=0.014, AUROC 0.599 |
| **C1** survives turn-index control (partial r) | **0.554** | 0.295 |
| top-K(20) aggregate AUROC | **0.886** | 0.767 |
| means: SUCCESS-final / WANDER-final / WANDER-early / LOCKED-final | 1.01 / 0.58 / 0.00 / 0.15 | 1.81 / 0.32 / 0.00 / 0.05 |
| G0 SAE recon cosine (usable ≥0.5) | 0.923 | 0.849 |

**Read:** the done-feature fires strongly at SUCCESS-final, substantially at WANDERING-final (≈57% as strong at L23), essentially not at LOCKED-final, and never at WANDERING-early. WANDERING and SUCCESS *share* the feature at their final turns; LOCKED (which gave up) does not. This is exactly the §13 prediction and it **replicates across two independent mid-layers**.

## Honest caveats (every one matters before this is believed)
1. **Scale / FVU:** median FVU > 1 (vs the SAE's own ~0.4 on training) ⟹ the captures are partly off the SAE's input scale. Magnitudes are therefore **unreliable**; but TopK feature *selection* is invariant to a global input scale and recon cosine is 0.85–0.92, so **which feature fires** — the only thing the analysis uses — is robust. A per-token re-capture matched to SAE training would tighten magnitudes (Stage-1b).
2. **The random-feature null (C2) is degenerate / weak:** with k=128 of 40960, most random features are silent in both classes ⟹ AUROC≈0.5 trivially, so "beats the 95th percentile (0.5)" is easy. **The load-bearing control here is C1 (turn-index), which the feature passes (partial r 0.55 at L23)** — not C2. A stronger specificity null (features matched on SUCCESS-final firing frequency; label-shuffle selection) is owed in Stage-1.
3. **"Done verdict" is a HYPOTHESIS, not established:** Stage-0 shows "a SUCCESS-final feature that WANDERING holds and LOCKED doesn't." That is equally consistent with **"actively engaged at the end"** (SUCCESS+WANDERING are still producing tool calls; LOCKED has given up) as with a literal **"task-complete"** verdict. Distinguishing them requires **interpreting feature 22358/908** (top-activating contexts via the model) — Stage-1. Do not call it the "done verdict" in any external writeup until interpreted.
4. **Only the §13 FIRST half.** Why WANDERING fails to *emit* `finish` (the verdict→action readout) is untested and needs the model (direct-logit-attribution of the feature's decoder direction to the `finish` token; predict it is lower in WANDERING than SUCCESS at equal feature activation). Stage-1.
5. N=20 WANDERING. Modest. Two layers ≠ "a circuit."

## Decision: advance to Stage-1 (needs the model, GPU)
The Stage-0 GO is clean enough (replicated, survives turn control, directionally exact) to justify Stage-1, whose first job is to **try to falsify caveat #3**: interpret feature 22358 (is it semantically completion/conclusion?) and run the readout/DLA test. If feature 22358 turns out to be a generic "active-late" feature, this becomes an honest negative ("WANDERING looks active-late, not done"). If it is genuinely a completion feature with a broken finish-readout in WANDERING, the verdict-circuit thesis closes the arc mechanistically. Either way it is pre-registered and reported as it falls.

Contrast with the just-closed context-rot program: that died at the predictive gate (redundant with behavior). This is a *mechanistic* claim (what the model represents), it survived its pre-registered controls, and it replicated — a real preliminary positive, held to the same skeptical standard.
