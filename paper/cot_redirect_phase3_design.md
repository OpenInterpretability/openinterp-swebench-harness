# CoT-Redirect Phase 3 — Does Activation Monitoring Survive the Obfuscation?

### A head-to-head: chain-of-thought monitor vs residual-stream probe, on obfuscated sabotage

**Experimental design v0.1, 2026-05-22. Pre-registered before running.**

- **Author:** Caio Vicentino (ORCID 0009-0003-4331-6259), OpenInterpretability
- **Builds on:** Paper 11 / Phase 1 (`paper/cot_redirect_phase1.md`) — a CoT-redirect
  intervention *obfuscates* sabotage: it cleans the chain-of-thought while the
  sabotaged patch still ships, collapsing the CoT monitor's recall of patch-level
  sabotage from 1.00 to 0.33 (Fisher p = 0.015).
- **Status:** design — pre-registered before any Phase-3 data is collected.

---

## 1. The question

Phase 1 showed that a CoT-redirect intervention blinds a **chain-of-thought** monitor:
the intervention forces an honest-looking `<think>`, the model still ships a
deliberately-buggy patch, and the monitor — which reads only the CoT — reports
"aligned." The sabotage was scrubbed from the *verbalized* channel.

But the model still computed the sabotage. It produced a deliberately-buggy patch;
the intent to do so had to be represented *somewhere* in the forward pass. Phase 1's
monitor read the CoT. Phase 3 asks: **is the sabotage still detectable in the residual
stream — does an activation-level probe catch what the CoT monitor missed?**

This is the breakthrough-shaped question of the arc. Two outcomes, both consequential:

- An activation probe **survives** the obfuscation (catches the sabotage in the
  OBFUSCATION trajectories where the CoT monitor was blinded): then *the monitored
  channel matters* — CoT monitoring is obfuscable by a cheap inference-time
  intervention, but activation monitoring is not. A concrete, mechanistic argument for
  where monitoring should live.
- The probe is **also blind** (recall collapses too): the obfuscation reaches the
  activations — even the residual stream is scrubbed. A deeper and more alarming
  result about how far a simple intervention can hide misalignment.

Either way the answer is mechanistic, on the residual stream, and directly informs the
CoT-monitorability debate.

## 2. Why a re-run with capture, not a replay

Phase 1/2 did not capture activations (the notebooks ran sdpa, capture off). The trace
JSONs log per-turn `thinking` and `raw_response` but not tool results or full
tool-call arguments, so the Phase-1 trajectories cannot be faithfully replayed token-
for-token. Phase 3 therefore **re-runs** NOINT and REDIRECT with residual capture
enabled. The obfuscation is a *population* effect (a rate, not a fixed trajectory set);
a fresh capture-enabled run reproduces the population. A faithful replay would require
richer logging — a noted improvement for future rollouts, not a Phase-3 blocker.

## 3. Conditions and capture

Two arms (the obfuscation contrast; RESAMPLE is not needed — it did not obfuscate):

| Arm | Purpose |
|---|---|
| **NOINT** | sabotage with no intervention — supplies probe training data (sabotaged-patch turns = positive, honest = negative) and the recall ceiling |
| **REDIRECT** | the obfuscating intervention — supplies the OBFUSCATION trajectories for the head-to-head |

- **NOINT: 48 instances. REDIRECT: 72 instances.** REDIRECT is over-sampled because the
  OBFUSCATION cell was only ~11% of REDIRECT in Phase 1 (4/36); 72 instances targets
  roughly 8-10 OBFUSCATION trajectories. Stratified across `ansible` / `openlibrary` /
  `qutebrowser`. 120 rollouts total.
- **Residual capture:** the model's residual stream at layers L11, L23, L31, L43, L55,
  at the harness positions (`think_end`, `pre_tool`, `turn_end`), every turn. Stored
  compactly (bf16) per (layer, position, turn). Estimated footprint ~8-12 GB.
- All other settings inherited from Phase 1 (Qwen3.6-27B, sandbag suffix, 30 turns,
  temperature 1.0, LLM-judge CoT trigger for REDIRECT).

## 4. The probe protocol

A linear probe on the residual vector at a (layer, position) site, predicting whether
the trajectory's final patch is sabotaged (patch-judge ground truth).

- **Labels.** Trajectory-level, from the post-hoc blinded patch-judge (Phase-1
  protocol). A trajectory's turns inherit its label: sabotaged-patch -> positive.
- **Training.** On NOINT only. Tested on REDIRECT — a clean transfer (the probe never
  sees intervention data). Also report a within-REDIRECT cross-validated probe to
  separate the transfer question from the detection question.
- **Site selection.** Sweep all (layer, position) sites; select by held-out AUROC.
- **Mandatory baselines — pre-registered, non-negotiable** (the openinterp probe
  rules): random-feature baseline (a probe on random directions of equal capacity);
  capacity / K-sweep (probe must beat random across capacities, not at one
  over-parameterized K); shuffled-source baseline (X shuffled, y kept) to rule out a
  marginal-fit artifact; an environment-coupling check (NOINT-trained probe applied to
  REDIRECT activations — residual directions can shift across conditions).

## 5. Pre-registered hypotheses

- **H-PROBE-SURVIVES** — on REDIRECT trajectories whose final patch is sabotaged, the
  activation probe's recall is significantly higher than the CoT monitor's recall (the
  Phase-1 figure: 0.33, 2 of 6). Activation monitoring survives the obfuscation.
- **H-PROBE-ALSO-BLIND** — the probe's recall on the same trajectories also collapses
  toward the CoT monitor's. The obfuscation reaches the activations.
- **H-PROBE-EPIPHENOMENAL** — the probe fails its within-NOINT quality gate (does not
  beat the random-feature baseline): there is no linearly-decodable patch-sabotage
  signal to begin with, consistent with the epiphenomenal-probe findings of Papers
  7-8. The head-to-head is then void and reported as such.

## 6. Primary metric and the head-to-head

- **Primary — probe recall of patch-level sabotage on REDIRECT.** Among REDIRECT
  trajectories with a sabotaged final patch, the fraction the activation probe flags,
  versus the fraction the CoT monitor flags (Phase-1 carry-over: that CoT recall is the
  obfuscation finding). The head-to-head is probe-recall vs CoT-recall on the **same
  trajectories**, McNemar / Fisher exact.
- **Supporting — per-turn AUROC.** The probe operates per turn; the per-turn AUROC of
  patch-sabotage detection is well-powered (hundreds of turns) and carries the
  statistical weight that the small per-trajectory headline cannot.
- **Probe-quality reference.** The probe's AUROC on held-out NOINT — establishes the
  probe works *at all* before any head-to-head claim.

## 7. Pre-registered analysis plan

1. Probe quality on NOINT: AUROC per site vs random-feature baseline + capacity sweep
   + shuffled-source. Select the headline site. Gate G1.
2. Apply the NOINT-trained probe to REDIRECT; report per-turn AUROC and per-trajectory
   recall on sabotaged-patch trajectories.
3. Head-to-head: probe recall vs CoT-monitor recall on REDIRECT sabotaged-patch
   trajectories (same set). Statistical test + CI.
4. Environment-coupling control: within-REDIRECT cross-validated probe vs the
   NOINT-transfer probe — does a recall gap come from transfer failure or from genuine
   obfuscation-resistance?
5. The OBFUSCATION sub-cell: of the CoT-clean-but-sabotaged trajectories specifically,
   what is the probe's recall? This is the sharpest cut of the headline.
6. Controls: D1 random-feature, capacity sweep, shuffled-source (Section 4); the
   Phase-1 paraphrase-evasion and length-confound checks carried over.

## 8. Gates (pre-registered)

- **G1** — the probe beats the random-feature baseline on held-out NOINT by a margin
  that survives the capacity sweep and the shuffled-source check. If it does not:
  H-PROBE-EPIPHENOMENAL, the head-to-head is void, and that is the (publishable)
  finding.
- **G2** — at least 8 REDIRECT trajectories with a sabotaged final patch (target ~10
  via the 72-instance draw). Below 8, the per-trajectory head-to-head is reported as a
  directional pilot and the per-turn AUROC carries the claim.
- **G3** — the headline is H-PROBE-SURVIVES if probe recall significantly exceeds CoT
  recall; H-PROBE-ALSO-BLIND if it does not and the probe is otherwise valid (G1
  passed); H-PROBE-EPIPHENOMENAL if G1 fails.

## 9. Sample size and compute

- 120 rollouts (NOINT 48 + REDIRECT 72) with residual capture. ~20-24 h GPU
  (Blackwell), capture overhead included; checkpointed, 2-3 Colab sessions.
- Capture footprint ~8-12 GB on Drive — fits; bf16, only the swept sites.
- Probe fitting + analysis: CPU, minutes — sklearn linear probes.
- Honest scope: even at 72 REDIRECT instances the OBFUSCATION cell is ~10; the
  per-trajectory head-to-head is pilot-scale and labelled as such. A definitive
  version needs a larger OBFUSCATION population (a Phase-3.5 scale-up) and the per-turn
  AUROC is pre-registered as the load-bearing supporting metric for that reason.

## 10. Threats to validity

- **Re-run, not replay.** Phase-3 trajectories are a fresh capture-enabled
  reproduction, not the literal Phase-1 trajectories. The obfuscation is a population
  effect; the rate is what reproduces.
- **Instructed sandbagging.** Unchanged from Phases 1-2 — installed, not emergent. The
  mechanistic finding is about *instructed* sabotage's residual signature.
- **Probe epiphenomenality.** Papers 7-8 found probes on this model are often
  epiphenomenal. H-PROBE-EPIPHENOMENAL is a live, pre-registered outcome, not a
  failure of the experiment.
- **Environment coupling.** A NOINT-trained probe applied to REDIRECT activations may
  degrade from distribution shift, not from obfuscation. The within-REDIRECT CV probe
  (Section 7.4) separates the two.
- **Capture position.** Sabotage may be represented at positions or layers not in the
  swept set; a null is conditional on the sites examined.
- **Single model, custom harness, 11% honest ceiling** — carried over from Phase 1.

## 11. Build plan

| Step | Deliverable |
|---|---|
| 1 | Capture-enabled rollout notebook — Phase-1 engine + forward hooks at L11/23/31/43/55, residuals saved per (position, turn). NOINT + REDIRECT, 48 + 72 instances. |
| 2 | Rollout (~20-24 h GPU, checkpointed). |
| 3 | Test-pass eval + blinded patch-judge (Phase-1/2 pipeline) -> trajectory labels. |
| 4 | Probe fitting: per-site linear probe on NOINT, random-feature + capacity + shuffled-source baselines, site selection. |
| 5 | Head-to-head analysis (Section 7): probe recall vs CoT-monitor recall on REDIRECT sabotaged-patch trajectories + the OBFUSCATION sub-cell. |
| 6 | Write-up — Paper 12, or the mechanistic section of an extended Paper 11. |

## 12. Why this is the breakthrough-shaped experiment

Phases 1-2 are behavioral: they measure what the agent *does* (CoT verdict, patch
verdict). Phase 3 is mechanistic: it asks where, in the model's internals, the
sabotage lives once it has been scrubbed from the verbalized channel. The answer
speaks directly to the central safety question raised by Phase 1 — *the channel a
monitor reads must not be the channel an intervention can write* — by testing whether
a channel an intervention does **not** write (the residual stream) is the one that
survives. It uses assets the project already has (the capture-capable harness, the
linear-probe stack, the blinded patch-judge), it lands in the author's core lane
(mechanistic interpretability via residual-stream probes), and it resolves three ways,
all publishable. It is the natural and highest-ceiling continuation of the arc.
