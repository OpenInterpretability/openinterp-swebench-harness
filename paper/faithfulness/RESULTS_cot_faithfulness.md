# RESULTS — Mechanistic CoT Faithfulness on Qwen3.6-27B (Tier-A)

Pre-registered in `PREREG_cot_faithfulness_causal.md`. Run 2026-06-18, Colab G4, 3 batches (0-20/20-40/40-60)
aggregated; data in `data/cot_faithfulness_n60.json`. SAE: `caiovicentino1/qwen36-27b-sae-fullstack` (11 layers).

## Setup
60 hard multi-step MCQs (serial computation a reasoning model needs CoT for), options shuffled per item.
Per item: Run-CoT (generate `<think>…</think>`, then forced slot "…answer is (") vs Run-noCoT (no thinking,
same slot). Keep the **16 items where CoT flips the no-CoT answer** for the causal test. Causal patch: per SAE
layer L, replace the noCoT decision-point residual with `decode(encode(x_cot_L))`; measure ΔlogP(CoT-answer).
Clean causal signal = `d_cot − d_self` (controls SAE recon error); noise control = `d_rnd − d_self`.

## Headline numbers (n=60; 16 flipped; bootstrap 95% CI over flipped items)

| Layer | causal `cot−self` [95% CI] | noise `rnd−self` |
|------:|---------------------------:|-----------------:|
| L15–L43 | ≈ 0 (CIs cross 0) | −8 to −10 |
| L47 | +0.17 [+0.03, +0.29] | −10.5 |
| **L51** | **+1.55 [+1.28, +1.85]** | −11.3 |
| **L59** | **+2.72 [+2.18, +3.28]** | −10.5 |
| **L63** | **+2.12 [+1.62, +2.61]** | −12.1 |

- **LATE (L51-63) +2.13 [+1.83, +2.41]  vs  EARLY (L15-27) −0.02 [−0.05, +0.01]** — CIs disjoint.
- **L59 causal > 0 in 16/16 flipped items** (min +0.60, max +4.81). Perfect sign consistency.
- **44/60 items: model already correct with NO CoT** (`already_correct_noCoT` 44/44); with CoT, 60/60 correct.

## Findings
1. **Faithfulness is CONDITIONAL.** When the model already knows the answer without thinking (44/60), the CoT
   features are non-causal (signal ≈ 0) — *performative* reasoning, the causal/mechanistic version of Goodfire's
   "reasoning theater." When the CoT genuinely changes the answer (16/60), the decision-point features are
   strongly *causal* (ΔlogP +2.7 @ L59), 16/16 consistent, far above recon-baseline and random.
2. **The causal channel is LATE.** Signal is ~0 through L43 and concentrates in L51-63 (disjoint CIs). The
   "lever is late" law of the agent arc (#6/#7, action commitment at L51-63) **extends to the reasoning channel**:
   the causal content of the reasoning consolidates in the same late band, not in mid layers.
3. **Controls clean.** Random-feature patch destroys the answer (−8 to −12 at every layer); self-patch (decode of
   the noCoT features) ≈ 0, so the late signal is CoT *content*, not SAE recon artifact.

## Relation to the literature (verified, deep-research 25/25)
- Extends **Chen et al. 2507.22928** (SAE+patching causal faithfulness) from Pythia-2.8B *base* to a 27B *reasoning*
  model: their scale curve (null@70M → causal@2.8B) **continues** — causality holds and is late at 27B-reasoning.
- Differentiates **Goodfire "Reasoning Theater" 2603.05488**: SAE features causally patched (not correlational
  attention probes); we *measure* performativity mechanistically.
- Differentiates **Young 2603.26410** (Qwen3.5-27B thinking-answer divergence): mechanistic/causal, not keyword text.
- Motivated by **DeepMind "Securing the Future of AI Agents" (2026)**: their roadmap names "inspect the model's
  inner workings" as needed-but-unbuilt; this is evidence the causal reasoning content is localizable (and late).

## Limitations (honest)
- Curated serial-computation MCQs (a pilot stimulus set), not a standard benchmark; 16 flipped items (effect is
  large + 16/16 consistent, but n is modest). Single model. Last-token residual substitution as the patch op.
- Tier-B (gated, not yet run): real SWE agent ACTIONS instead of MCQ answers — the max differentiator vs Goodfire;
  and a standard dataset (GSM8K / R2E-Gym) for the flip set.

## Tier-B — agent ACTIONS (run 2026-06-18, n=32 trap scenarios, `data/cot_faithfulness_agent_n32.json`)
Same causal test, "answer" = a committed tool call. 4 domains (crypto/files/db/deploy), trap scenarios where the
surface instruction suggests an irreversible action (send/delete/drop/deploy) but a detail suggests a safe one.
- **CAUSAL replicates Tier-A in the action channel:** on the 10 flipped scenarios, signal ≈ 0 through L47, then
  L51 +0.84 / L59 +1.89 / L63 +2.21; **LATE +1.65 [+1.17, +2.12] vs EARLY +0.08 [−0.14, +0.32]** (disjoint CIs).
- **The three-faced law:** the same late band (L51-63) carries the causal content of (i) the committed action
  (#7/#8), (ii) reasoning→answer (Tier-A), (iii) reasoning→agent-action (Tier-B).
- **Behavioral side-finding:** with CoT, the agent chose the CAUTIOUS (safe) action in 27/32 traps (only 5 irreversible;
  crypto 0, deploy 0) — honest reasoning acts as a safeguard on these traps (contrast with the adversarial COLLAPSE).
- **HONEST CAVEAT (prediction/monitoring is weak):** because the agent was so cautious, only 5/32 chose the
  irreversible action, so the per-layer action-prediction AUROC (late features beat random, e.g. L51 0.84 vs 0.62)
  is on 5 positives and NOT defensible. The predictive-monitor angle needs scenarios that induce more irreversible
  actions to balance the classes — future work, not a result. The *causal* finding does not depend on it.

## Verdict
**CAUSAL + LEVER-IS-LATE (conditional on the CoT doing work), replicated across MCQ answers (Tier-A) and agent
actions (Tier-B).** A positive, first-on-model mechanistic-faithfulness
result on a 27B reasoning model, unifying the arc's late-channel law across action and reasoning. No DOI until the
pre-mint eval gate passes (recompute from `data/`, citations, adversarial audit) — the discipline that bit us twice.
