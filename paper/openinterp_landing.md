---
title: "Explore-Consolidate Dynamics in Cross-Probe Coherence Separate Successful and Failed LLM Agent Trajectories"
date: 2026-05-18
author: "Caio Vicentino"
tags: [paper, monitoring, swe-bench, qwen, kappa-t, explore-exploit]
slug: kappa-t-coherence-buildup
summary: |
  Cross-probe coherence κ_t shows a U-shape over agent trajectories.
  Successful agents have HIGH-AMPLITUDE U (big exploration drop + big
  consolidation rise); failed agents have FLAT trajectories. Inverse of
  cardiac uncoupling: failure here is absence of dynamism, not coupling
  collapse. SWE-bench Pro N=99 Qwen3.6-27B; early-half p=0.0002,
  late-half p=0.00004.
---

# Explore-Consolidate Dynamics in Cross-Probe Coherence

**TL;DR.** I introduce a meta-signal for LLM agent monitoring — *cross-probe coherence* κ_t, the mean absolute pairwise correlation across N concurrent per-turn probes. On 99 SWE-bench Pro trajectories from Qwen3.6-27B, two findings:

1. **Mean κ̄ separates outcomes** at AUROC 0.677 (pre-registered, Mann-Whitney p=0.0009).
2. **κ_t shows a U-shape** over each trajectory: it *falls* through an early exploration phase and *rises* through a late consolidation phase. The **amplitude** of this U distinguishes successful from failed agents:
   - Early-half slope: succ **−0.0078**/turn, fail −0.0007/turn — p=0.0002
   - Late-half slope: succ **+0.0149**/turn, fail +0.0025/turn — p=0.00004

Successful agents do a high-amplitude explore-then-consolidate arc; failed agents have flat κ_t — no exploration, no consolidation.

[**Read the full paper (PDF, Zenodo DOI: 10.5281/zenodo.20278983)**](#) | [**GitHub**](https://github.com/OpenInterpretability/openinterp-swebench-harness) | [**Data (Hugging Face)**](https://huggingface.co/datasets/openinterp/kappa-t-coherence-buildup)

---

## Why this matters

Multi-probe LLM monitoring is usually done one axis at a time: one deception probe, one sycophancy probe, one capability probe. When multiple probes exist, they are combined by simple aggregation (averaging logits, OR-gating). This implicitly assumes the joint distribution of probe outputs carries no information beyond the marginals.

I show that assumption can be false. The *covariance structure* across N concurrent probes carries a meta-signal that no single probe captures alone. And the *temporal shape* of that covariance — the U-shape over the trajectory — separates outcomes even when the absolute coherence at any single timepoint does not.

## Inverse of cardiac uncoupling, with a twist

Twenty years of ICU literature shows that healthy patients have **coupled** baseline (HR ↔ BP ↔ RR), and decompensating patients show **cross-vital uncoupling** hours before crisis. The structural similarity to multi-probe LLM monitoring is direct. But the *dynamical signature* differs:

| | Cardiology | LLM agents (this work) |
|---|---|---|
| Healthy / successful | Coupled, **static** baseline | **Oscillatory**: explore-then-consolidate |
| Failure | Coupling collapse from baseline | Flat: no oscillation amplitude |
| Mechanism | Decompensation event | Reasoning deficit (no commit to either phase) |

The structural insight (cross-channel coupling carries health information) transfers; what *successful coupling looks like* — high-static vs oscillatory — is system-specific and requires direct measurement.

## The walk-back-and-rescue story

This paper has a peculiar arc that I want to document explicitly. A first version of the slope statistic — "successful traces show monotonic buildup of κ_t over time" — looked clean (p=0.0003, 12.7× effect ratio). A pre-registered robustness control then revealed it was substantially length-confounded: failure traces are 40% longer (they hit `max_turns=30`), and the original slope statistic was largely explained by that length asymmetry (Mann-Whitney p=0.56 after length regression).

The U-shape decomposition reported above is the **post-control rescue**: length-normalized by construction (early-half/late-half), and the class effects are *stronger* than the original confounded statistic (p=0.0002 and p=0.00004, vs. the original 0.0003).

I think this kind of walk-back-and-rescue is the right epistemic stance for early-stage interpretability. The pre-registered protocol catching its own headline confound — rather than the headline statistic in isolation — is what makes the rescued finding worth reading.

## Methodological discipline

This finding is the sixth probe project in a 36-hour sprint. The five preceding candidates each failed a four-gate pre-registration protocol (silent-refusal, tool-doubt, tool-doubt position-sweep, Inner-Outer paired, κ_t v1 with trace-level labels). And the **sixth project's own bonus statistic** failed a separately pre-registered control (C1 length-confound). Both walks-back happened *because* the protocol existed, before publication.

The full history is in [Appendix B of the paper](#) and the project notes.

## What's next

- **Multi-model validation** (Qwen2.5-7B, Llama-3.1-8B): most important next step
- **Multi-benchmark** (web navigation, retrieval QA, dialog agents)
- **Layer/position sweep** (currently single layer L43, single position `paired_concat`)
- **Pre-registered replication** of the U-shape finding on an independent dataset (it is currently post-hoc relative to the original v2 pre-registration)
- **Causal test** — does steering κ_t change outcomes? Open question; this paper is detection-only
- **agent-probe-guard SDK v0.2** will ship κ_t with early-half and late-half slope estimates as meta-signals alongside individual probe outputs

## Why R$0 compute

All κ_t analysis runs on saved residuals from the existing Phase 6 capture set. No GPU is required beyond the initial Phase 6 run (single A100, ~12 hours, amortized across this and three other papers). κ_t analysis itself takes 35 minutes total on a MacBook Pro CPU.

## Citations and license

Paper and data: CC-BY-4.0. Code: Apache-2.0.

```bibtex
@misc{vicentino2026kappat,
  title  = {Explore-Consolidate Dynamics in Cross-Probe Coherence Separate Successful and Failed LLM Agent Trajectories},
  author = {Vicentino, Caio},
  year   = {2026},
  doi    = {10.5281/zenodo.20278983},
  note   = {OpenInterpretability working paper. Zenodo: https://zenodo.org/record/20278983},
  url    = {https://openinterp.org/research/papers/kappa-t-coherence-buildup},
}
```

If you build on this, I'd like to hear about it — `caio@openinterp.org`.
