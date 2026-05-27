# Exp B Plan B Results — L55 SUCCESS-donor Steering (CLEAN NULL CAUSAL)

**Date**: 2026-05-26
**Status**: COMPLETE. Primary (20 instances, hook α=0.3) + Baseline (20 instances, no hook) both shipped.

## TL;DR

L55 SUCCESS-donor always-on hook with mode=add α=0.3 has **no causal effect** on WANDERING-to-finish_tool flip rate. Primary 6/20 (30%) vs Baseline 7/20 (35%), Fisher p=1.000, odds ratio 0.796 (slight HURT direction). Combined with Exp D, this is the **second convergent null** for the Tool-Entropy §13 "mid-layer-to-edge alignment failure can be rescued by SUCCESS-direction injection" hypothesis.

## Final results

| Condition | finish_tool / N | Rate | Reason |
|---|---|---|---|
| Baseline (no hook) | 7/20 | 35% | natural temperature variance |
| Primary (hook L55 SUCCESS-donor α=0.3) | 6/20 | 30% | hook bias |
| **Δ** | **−1** | **−5pp** | **NULL with slight HURT direction** |

**Fisher's exact (contingency)**:
```
              finish_tool  NOT
  primary         6         14
  baseline        7         13
```
- Odds ratio: 0.796
- p-value (two-sided): 1.000
- p-value (primary > baseline): 0.7497

## Per-instance paired comparison

| Instance | Baseline | Primary | Changed? |
|---|---|---|---|
| qutebrowser-2dd896 | max_turns | max_turns | = |
| qutebrowser-9b71c1 | **finish_tool** | max_turns | ★ baseline flipped |
| openlibrary-5de7de | **finish_tool** | **finish_tool** | both |
| openlibrary-7f6b722 | max_turns | **finish_tool** | ★ primary flipped |
| qutebrowser-0d2afd58 | max_turns | max_turns | = |
| ansible-5d253a13 | max_turns | max_turns | = |
| ansible-502270c8 | **finish_tool** | **finish_tool** | both |
| openlibrary-09865f5f | max_turns | max_turns | = |
| openlibrary-c05ccf2c | max_turns | max_turns | = |
| openlibrary-5c6c22f3 | max_turns | max_turns | = |
| openlibrary-bb152d23 | **finish_tool** | max_turns | ★ baseline flipped |
| ansible-a26c325b | max_turns | max_turns | = |
| ansible-3db08adb | **finish_tool** | max_turns | ★ baseline flipped |
| ansible-de01db08 | **finish_tool** | **finish_tool** | both |
| ansible-cd9c4eb5 | **finish_tool** | **finish_tool** | both |
| ansible-4c5ce5a1 | max_turns | max_turns | = |
| qutebrowser-99029144 | max_turns | max_turns | = |
| qutebrowser-85b867fe | max_turns | max_turns | = |
| qutebrowser-cf06f4e3 | max_turns | **finish_tool** | ★ primary flipped |
| qutebrowser-0fc6d110 | max_turns | max_turns | = |

**Summary**:
- 16/20 same outcome (determinism holding under same seed across paired runs)
- 3 baseline-only flips (hook bloqueou natural finish)
- 2 primary-only flips (hook resgatou)
- McNemar (3 vs 2): z ≈ 0.45, p ≈ 0.65 — within noise

## What this means for Tool-Entropy paper §13 hypothesis

**Hypothesis (Tool-Entropy paper, unchanged)**: WANDERING = mid-layer verdict consolidated ("model knows it's done") but edge circuits (L11/L55) fail to translate verdict into `finish_tool` emission.

**Original Exp B Plan B prediction**: persistent injection of SUCCESS-derived L55 residual should bias WANDERING agents toward finish_tool emission.

**Observed**: no causal effect, slight HURT direction. The hypothesis as stated is NOT rescued by simple residual perturbation at L55 with α=0.3.

**Three possible interpretations**:
1. **Hypothesis correct but L55 isn't the right locus** — edge-layer alignment failure may involve a different circuit (L11? specific attention heads? MLP-level?)
2. **Hypothesis correct but α=0.3 too weak** — testable via α=0.7 or α=1.0 sweep
3. **Hypothesis wrong** — WANDERING is not "mid-layer ready, edge fails" but a different mechanism entirely (e.g., attention pattern lock-in, not residual-direction issue)

The mode=add experiment with α=0.3 was designed to give the hypothesis a fair shot without destroying the model (per `feedback_always_on_replace_destructive`). At this magnitude, the answer is null.

## Hardware-determinism methodological discovery

**Critical finding**: Phase 6 was run on RTX 6000 Pro Blackwell (96 GB). All 20 WANDERING instances were classified as `finish_reason=max_turns` (0/20 finish_tool) — that's how they got the WANDERING label.

On H100 80GB with the SAME seed (Runner.seed_for is deterministic by iid) and NO HOOK, baseline shows 7/20 = 35% finish_tool. The classification is not reproducible across GPUs.

**Implications**:
- WANDERING phenotype is partly artifact of single-run + RTX 6000 floating-point determinism
- Hardware drift across GPUs (H100 vs RTX 6000) shifts ~35% of WANDERING-classified instances toward natural finalization
- Future WANDERING labeling protocols should require either (a) multi-seed classification at single GPU, or (b) cross-hardware validation
- The "WANDERING is real" claim from Tool-Entropy paper should be revisited with this caveat — the population is at minimum 35% noise-floor by hardware

## Paper #2 narrative consolidation

Two convergent nulls now in the paper:

1. **Exp D HONEST NULL** (`paper/exp_d_results.md`): WANDERING patches don't pass tests more than SUCCESS patches (ratio 1.37, p=0.71). Rules out "silent success" alternative — patch quality doesn't distinguish sub-classes.

2. **Exp B Plan B NULL** (this doc): L55 SUCCESS-donor injection doesn't rescue WANDERING (Δ=−5pp, p=1.00). Rules out "simple residual perturbation rescues alignment failure" — hook at L55 with α=0.3 is not causal lever.

3. **Methodological discovery**: WANDERING is not hardware-stable — single-run classification on RTX 6000 captures temperature-stochastic phenotype.

## Per Caio's standard

Per [[feedback-walk-back-and-rescue-paper-structure]]: report nulls honestly, structure paper as pre-registered hypothesis test → null verdict → discussion of mechanistic implications. The paper joins the **"Two Forms of Epiphenomenal Probes"** pattern from your Phase 7/8 work as a third null in the series.

## Optional next experiments

- **Exp B at α=0.7 or α=1.0** (~6h H100): structural-rigidity check that α=0.3 wasn't too weak. If stronger α also null → confirms L55 SUCCESS donor isn't a lever regardless of magnitude.
- **Exp C (v5 direction L43 α-sweep)** (~8h H100): tests a different direction (probe-derived v5 collapse direction, not SUCCESS donor) at a different layer (L43 mid vs L55 edge). Includes random direction control + structural-rigidity diagnostic. If Exp C also null = THREE convergent nulls = very strong paper.

## Artifacts

- Drive: `/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_exp_b_c/`
  - `exp_b_primary_add/results.json` — 20 with hook α=0.3
  - `exp_b_baseline_nohook/results.json` — 20 without hook
- Notebooks: `notebooks/nb_exp_b_c_combined_steering.ipynb` (primary), `notebooks/nb_exp_b_baseline.ipynb` (baseline)
- Hardware: NVIDIA H100 80GB (vast.ai)
- Total compute: ~13h H100 (~$26)
