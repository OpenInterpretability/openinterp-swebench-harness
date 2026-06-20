# E0 power-pilot — RESULTS (2026-06-19)

Run: `colab run --gpu G4 scripts/monitor_e0_pilot.py -- --regime R2 --n 22` (Qwen3.6-27B + fullstack SAE L59,
greedy, bf16). Ledger: stdout OILAB_JSON (final state below). n=22 (14 felt over-reach traps + 8 granted controls)
across 6 action domains (files/crypto/db/deploy/email/cloud). Labels: positive = agent emits the irreversible
action on an UNgranted target (11 positives, 11 negatives).

## Headline numbers (n=22, feasibility only — NOT inferential)
| metric | value |
|---|---|
| elicitation rate (felt traps over-reached) | **0.79** (11/14) |
| BB1 action-text AUROC | 0.36 |
| BB2 CoT-text AUROC | 0.71 |
| BB3 external task-grounded AUROC | **0.90** |
| WB1 linear probe (L59, LOO) AUROC | **0.34** |
| WB2 best SAE feature (L59, LOO) AUROC | 0.66 |
| black-to-white boost (best WB − best BB) | **−0.24** |
| F (CoT verbalizes lack of authorization) variance | **0.0** (all 22: F=0) |
| **pre-registered verdict** | **FALLBACK_R1_LED** |

## Interpretation (honest)
The verdict is the PREREG's anticipated branch, and for a deeper reason than "F has no variance":

1. **R2 elicitation is NOT the kill-risk we feared.** 0.79 over-reach on the momentum traps, ≫ the 0.40 gate.
   We can build a balanced R2 set at scale. The single biggest design risk is retired.

2. **H2 (felt≠granted blind spot) reproduces, now quantified as a head-to-head AUROC gap.** In the sincere-error
   regime the internal monitors are blind — linear probe **0.34 (below chance)**, SAE 0.66 (and that is the
   LOO-best of 40960 features at n=22, i.e. optimistic) — while the **external task-grounded check hits 0.90**.
   The cheap action-text monitor is also blind (0.36: it is the same `delete_file`/`send_transaction` call whether
   authorized or not). This extends *Felt, Not Granted* ("allows 21/21") into a monitor head-to-head: the internal
   read loses to the external check on sincere over-reach, across 6 action domains.

3. **Why white-box is blind here (the mechanism):** F=0 on ALL 22 is not a measurement artifact — it is the
   felt≠granted mechanism surfacing in the CoT. The agent *feels* authorized (the momentum framing), so it
   genuinely never verbalizes doubt, and there is no internal "this is unauthorized" state to probe. The signal
   the external check uses (the user never *granted* this specific target) is ground truth the model does not
   represent. So white-box cannot win in R2 by construction.

4. **Therefore H1 (the law: white-box wins where CoT is performative) must be tested on R1, not R2.** R2 is
   *uniformly* performative AND *uniformly* white-box-blind — no variance to correlate. The white-box advantage
   lives where the model internally KNOWS it is misbehaving but hides it (deliberate sabotage / hidden
   scratchpad — SHADE-Arena R1): there CoT-text is obfuscated but the concealed intent is in the activations.

## What this changes (design amendment)
- **R2's role is settled:** the *blind-spot existence proof* for H2 (internal monitors lose to the external,
  task-grounded check on sincere over-reach). Strong, on-brand, reproduces the arc. Keep it as a paper section.
- **H1 moves to R1 (SHADE-Arena).** Next gate = E0-R1: run the same 5-monitor head-to-head on SHADE-Arena
  sabotage task pairs (hidden scratchpad), where faithfulness varies, and test whether the black-to-white boost
  is POSITIVE and tracks (negative) verbalization.
- Do NOT re-run R2 to "fix F" — F=0 is the real phenomenon, not a bug.

## Engineering notes
- Harness validated end-to-end on GPU (deps self-install of transformers-from-main, public SAE download
  unauthenticated, per-trap gen+capture at the action token, 5 monitors, incremental OILAB_JSON). ~10 min total
  on a fresh G4 incl. load. Smoke n=4 caught two `colab run` issues first (no `__file__`, repo not shipped) —
  fixed via embedded traps.
- WB monitors are honest LOO; at n=22 the SAE best-feature AUROC is high-variance/optimistic — treat as a ceiling,
  not an estimate. The probe at 0.34 is the robust read: internals do not separate sincere over-reach.

## Next steps
1. Build E0-R1 on SHADE-Arena (github jkutaso/SHADE-Arena): adapt ≥2 task pairs for Qwen3.6-27B capture, run the
   5-monitor head-to-head + the faithfulness-stratified boost. This is the H1 test.
2. If R1 shows a positive, faithfulness-tracking boost → the law holds → scale (E1). If not → the paper is the
   *boundary* result: white-box helps only in the concealed-intent regime, is blind in sincere error, and the
   operating rule is the regime split.
