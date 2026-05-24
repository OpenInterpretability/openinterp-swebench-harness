# Exp C Design — Probe-Direction Steering Causal Test

**Paper #2 third causal experiment.** Tests whether the residual direction that PREDICTS tool-entropy collapse (v5 signal from Tool-Entropy paper) is also CAUSAL for collapse — i.e., does steering along that direction actually induce or prevent WANDERING?

## Hypothesis

Tool-entropy collapse (v5) is correlated with WANDERING. Two competing interpretations:

- **H_epiphenomenal**: tool-entropy collapse is a SYMPTOM of WANDERING; intervening on the residual direction that predicts collapse does NOT alter behavior
- **H_causal**: the residual direction that predicts collapse is also CAUSAL — steering along it alters action diversity and WANDERING rate

This is the "Two Forms of Epiphenomenal Probes" framework from Caio's Phase 7/8 prior work. The structural-rigidity α-sweep diagnostic distinguishes them.

## Direct test design

### Step 1 — Identify the tool-entropy collapse direction

For each WANDERING vs SUCCESS pair (matched repo):
- Contrastive analysis: compute mean residual difference at L43 pre_tool, late-half turns
- Direction `v_collapse = mean(WANDERING_residual) - mean(SUCCESS_residual)`
- Normalize: `v_collapse /= ||v_collapse||_2`

Alternative: train a linear probe to predict collapsed-entropy outcome from L43 residual, use its coefficient direction.

### Step 2 — Steering intervention

For each WANDERING trajectory:
- Use LayerPatch in `mode='add'` at L43 with direction `v_collapse`
- Sweep magnitude α ∈ {-200, -100, -50, -10, 0, +10, +50, +100, +200}
- Negative α = push AWAY from collapse direction (toward SUCCESS-like diversity)
- Positive α = push TOWARD collapse direction (reinforce WANDERING pattern)
- Run agent inference with this steering active throughout

### Step 3 — Measure

Per trajectory × per α:
- Did `finish_tool` emit?
- What was tool_entropy_last10 of resulting trajectory?
- What was final probe score at L43?

Behavioral flip metric (per Caio's Phase 7 standard): use whitespace-stripped flip comparison to avoid tokenization artifacts.

## Triple-source convergent verdict (per Phase 7-10 standard)

Following the methodology from Caio's "Two Forms of Epiphenomenal Probes" paper:

1. **Log-prob proxy with control-token norm**: at lock-in moment, compute Δlog-prob(finish_tool) under steering, normalized vs control tokens
2. **Single-shot α=+5 behavioral**: discrete action choice with small steering — does first tool flip?
3. **Continuous α=+5 behavioral**: 4 trajectories with persistent steering — does final tool emission change?

Triple convergence rules:
- **All 3 agree on flip → causal**
- **Probe predicts but behavior doesn't → epiphenomenal-via-softmax-temperature**
- **Behavior changes but in WRONG direction (degenerates not finishes) → epiphenomenal-structurally-locked**

## Structural-rigidity α-sweep (mandatory falsification)

Per Caio's memory rule (`feedback_steering_structural_rigidity_diagnostic`):

> "When forward-hook steering at α ∈ {±2, ±5} produces zero behavioral change, sweep α to multiples of residual norm (e.g., +50, +100, +200) with both probe AND random direction BEFORE declaring null."

We sweep α to ±200 (~1.3× residual norm = 154) from the start. Random direction control included.

If at α = +200:
- finish_tool emission flips ≥ 30% → **causal** (lever)
- 5-30% flip → **weak causal** (pushdown-asymmetric or similar)
- ≤ 5% flip → **epiphenomenal** (template-locked or softmax-temp confound)

## Expected outcome (prior from Phase 7-10 patterns)

Caio's prior empirical pattern across 4 papers:
- Most probes are **epiphenomenal-via-softmax-temperature** at low α
- Some are **structurally-locked** at high α (template-determined)
- A few are **lever** (RG L55 mid_think: 32% flip at α=+200)

Prior probability for v5 tool-entropy direction:
- 50% epiphenomenal (likely — entropy is downstream observable)
- 30% structurally-locked (template-determined by scaffold)
- 20% lever (would be paper #2 headline)

Any outcome is publishable — null result joins the "Two Forms of Epiphenomenal Probes" pattern (would become "Three Forms" or "Four Forms" depending on what we find).

## Compute requirements

- 20 WANDERING × 9 α values × ~25 turns × ~5sec/turn ≈ 3h H100
- Add 4 controls (random direction, locked-direction, L55 instead of L43, mid-turn-only)
- Total: ~6h H100 ≈ $12 vast.ai

## Engineering tasks

1. **`scripts/exp_c_extract_v5_direction.py`** — contrastive analysis on Phase 6 captures
   - Output: `paper_mega_v4_pairs/exp_c_v5_direction.safetensors`
   - Single tensor: L43 pre_tool collapse direction (d_model = 5120)

2. **`scripts/exp_c_steering_sweep.py`** — orchestration
   - Load v5 direction
   - For each WANDERING × α: install LayerPatch(model, {43: v_collapse * α}, mode='add')
   - Run agent inference
   - Log outcomes

3. **Analysis script** — aggregate per-α flip rates, structural-rigidity verdict, triple-source convergence

## Decision matrix

| Outcome at α=+200 | Triple-source agreement | Verdict |
|---|---|---|
| ≥30% flip | All 3 agree | **CAUSAL — paper headline** |
| 5-30% flip | 2/3 agree | Weak causal — describe regime |
| ≤5% flip | All agree no flip | **EPIPHENOMENAL** — joins prior "Two Forms" pattern |
| Behavior changes wrong way | — | **STRUCTURALLY-LOCKED** — distinct null type |
| Random direction also flips | — | **CONFOUND** — non-specific perturbation |

## Status (2026-05-24)

- LayerPatch class ready (supports `mode='add'` for steering)
- Phase 6b Docker eval RUNNING in background (PID 49261, started ~15:55 BRT)
- Exp B donors pre-computed
- Exp C: design complete, awaits compute env

## Order of execution (when compute available)

1. **Exp C direction extraction** (~30min, no model run — analytical)
2. **Exp B Plan B execution** (~5h H100) — Plan B always-on hook from turn 0
3. **Exp C steering sweep** (~6h H100) — same env as Exp B
4. **Total H100 time**: ~11h ≈ $22 vast.ai

Run them consecutively in one H100 session to amortize setup.

## Reviewer-targeting

Exp C addresses the natural follow-up critique to Exp B:
> "Even if patching L55 SUCCESS direction biases toward finish_tool, the tool-entropy signal itself (v5) might be a downstream symptom, not the causal axis. Have you tested whether steering on the v5 direction has the same effect?"

By providing Exp C, we close that loophole. Even null result for Exp C is valuable — confirms v5 is epiphenomenal observable while edge-layer L55 is the causal locus.
