# PRE-REGISTRATION — The Verdict Circuit: does WANDERING hold a SUCCESS-typical "task-done" SAE feature it fails to act on?

**Author:** Caio Vicentino · OpenInterpretability · pre-registered 2026-06-02 (before SAE-encoding any capture).
**Model / data:** Qwen3.6-27B; existing Phase-6 N=99 SWE-bench Pro agent trajectories with per-turn, per-position residual captures at L11/23/31/43/55; labels `sub_class ∈ {success:40, locked:39, wandering:20}`. Full-stack TopK SAE (`caiovicentino1/qwen36-27b-sae-fullstack`, k=128, d_sae=40960) at layers {15,19,23,27,35,39,43,47,51,59,63}. **SAE ∩ captures = L23, L43** (Stage-0 runs L43 first, then L23).

---

## 0. Thesis (the central mechanism of the whole arc, never verified as a circuit)
The Tool-Entropy paper's §13 mechanism — *"the mid-layer verdict that the task is done is consolidated, but the edge circuits that translate the verdict into the `finish` action fail to fire"* — is, across four arc papers, only a **correlational** hypothesis. We test its **first half** mechanistically and cheaply: **is there an interpretable SAE feature that marks the "task-done" verdict in SUCCESS, and does WANDERING activate that same feature (holding the verdict) while LOCKED does not (genuinely not done)?** If yes, WANDERING ≠ "still confused" — it is "done but not emitting," and the failure is downstream of the verdict. If no, §13's "verdict consolidated" is literally false at this granularity (also publishable).

## 1. Why this is open (June 2026, verified)
Novelty scan (web/arxiv): **PARTIALLY-OPEN, ranked #1** of three candidates. Nearest prior work localizes a "sufficiency / when-to-stop" signal for **input reading** (arXiv:2502.01025, attention heads, single-turn QA — not an agent terminal action), an agent **memory** failure circuit (arXiv:2605.03354 — not task-completion), and the conceptual **knowledge-action gap** (arXiv:2603.18353 — behavioral, not a finish-action feature). **No one has localized a "task-complete verdict" SAE feature and shown WANDERING holds it but fails to emit, in a real long-horizon coding agent.** Threats-to-novelty to cite: 2502.01025, 2605.03354, 2603.18353.

## 2. The §13 prediction, made precise (Stage-0, CPU, no model)
For each trajectory, at each turn, take the residual at the **`pre_tool`** position (the state immediately before the action is emitted — the §13 locus), SAE-encode it (TopK, k=128) → a sparse feature-activation vector. Define per trajectory: **final turn** = max captured turn (SUCCESS final = the `finish`-emitting turn; WANDERING final = budget exhaustion; LOCKED final = give-up); **early window** = turns 0–2.

- **Selection (SUCCESS only — never sees W/L):** a *done-feature* = a feature whose activation/firing is higher at SUCCESS **final** than SUCCESS **early** (per-trajectory). Pre-commit: rank by (firing-freq at final − at early) among features firing in ≥30% of SUCCESS finals; **primary = top-1**, secondary = top-K (K=20) mean.
- **Held-out §13 test (WANDERING & LOCKED — independent of selection):**
  - **P1 (verdict present in WANDERING):** the selected done-feature is more active at WANDERING **final** than WANDERING **early** (paired, per-trajectory).
  - **P2 (specificity — THE key test):** the done-feature is more active at WANDERING **final** than LOCKED **final** (Mann-Whitney, per-trajectory). Both classes are "ending"; they differ in *done-belief*. WANDERING > LOCKED ⟹ WANDERING holds the success-typical verdict.

## 3. Pre-registered controls (the program's discipline)
- **C1 — turn-index confound:** the done-feature must not be a generic "high-turn / long-context" feature. Partial out turn index; LOCKED-final (also late-ish, but give-up) is the built-in control. A feature equally active at LOCKED-final is a NULL.
- **C2 — random-feature null (specificity):** the done-feature's WANDERING-vs-LOCKED effect must exceed the **95th percentile** of the same effect computed on randomly chosen features. If random features separate W from L just as well, the classes differ globally (length etc.) and the done-feature is not special.
- **C3 — pseudo-replication:** one summary value **per trajectory** (final-window mean), independent samples; never pool (turn,traj) as independent.
- **C4 — SAE-usability gate (G0):** the captures are per-position residuals; the SAE was trained per-token. Before any feature claim, verify the SAE meaningfully reconstructs the `pre_tool` L43 vectors (median fraction-of-variance-explained / reconstruction cosine over a sample). **If FVU is poor**, Stage-0 is **inconclusive** → the existing captures are off-distribution for the SAE and a targeted per-token re-capture (Stage-1b, GPU) is required; this is an honest result, not a kill.

## 4. Decision gates (set before any encoding)
- **GO (verdict-circuit thesis supported)** iff, for a pre-committed (SUCCESS-selected) done-feature: **P1 holds** (WANDERING final > early) **AND P2 holds** (WANDERING final > LOCKED final, MW p<0.05) **AND** the W>L effect clears the random-feature null (C2) **AND** survives the turn-index control (C1). Then proceed to Stage-1 (causal/readout).
- **NO-GO / honest negative (refutes §13's "verdict consolidated")** iff the SUCCESS done-feature does **not** transfer to WANDERING (W-final ≈ W-early), or WANDERING ≈ LOCKED (no done-belief specificity), or the effect is explained by turn-index / random features. Report as "WANDERING does not hold a SUCCESS-typical done-verdict at L43/L23 `pre_tool` — §13's first half is not literally a single-feature fact."
- **INCONCLUSIVE** iff G0 fails (SAE can't read these captures) → spec a per-token re-capture.

## 5. Stage-1 (gated by a Stage-0 GO; needs the model / unembedding, GPU)
The §13 **second half** — *why WANDERING fails to emit*. With the model: direct-logit-attribution of the done-feature's decoder direction to the `finish` tool token; predict DLA(done-feature → finish) is **lower in WANDERING than SUCCESS** despite equal done-feature activation (the verdict→action readout is broken). Optionally a single-feature causal test (clamp the done-feature up at the WANDERING final turn) — but we do **not** pre-promise a rescue (the arc's residual levers are null; this is a mechanism test, not a lever claim).

## 6. Artifacts
`scripts/verdict_circuit_stage0.py` (+ `--smoke`: plant a done-feature in synthetic feature-space; must recover it and reject a length-only null before trusting the real run). Output: `scripts/verdict_circuit_out/stage0_L{43,23}.json`. Result reported whichever way it falls — a clean refutation of §13's first half is as publishable as a confirmation.
