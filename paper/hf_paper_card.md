---
license: apache-2.0
language:
- en
tags:
- mechanistic-interpretability
- agent-safety
- llm-agents
- linear-probes
- failure-detection
- swe-bench
- qwen3
- llama
- gpt-5
pretty_name: Tool-Entropy Collapse Paper
size_categories:
- n<1K
---

# Tool-Entropy Collapse: A Cross-Architecture Signature of Agent WANDERING Failure

**Author**: Caio Vicentino (OpenInterpretability) · ORCID [0009-0003-4331-6259](https://orcid.org/0009-0003-4331-6259) · caio@openinterp.org

**Status**: v0.8 (2026-05-24) — submission-ready paper. PDF + LaTeX source + 5 figures included.

**License**: Apache-2.0 (code + paper)

## TL;DR

Probe-based safety monitoring of LLM agents has a **34% blind spot** on Qwen3.6-27B SWE-bench Pro: the WANDERING sub-class where probe says "success" but agent never emits `finish_tool` and hits its budget. We test six detector designs across three signal channels (text, residual cross-layer, action entropy). **Tool-use entropy collapse** is the breakthrough signal — WANDERING agents collapse onto a small set of repeated tool calls (W/S median ratio 0.41 in Qwen AND Llama, 0.71 in GPT-5), enabling a Tier-3 autonomous-termination detector at **70% recall × 5% FP** via combined v1 ∪ v5.

**Cross-architecture validation**: Llama-70b (n=2,315, p<10⁻¹⁵) and GPT-5 router (n=1,419, p=8.9×10⁻³⁵) confirm. **Cross-task validation on METR MALT (15+ task families) is NULL (p=0.81)**, scoping the claim to multi-turn code-execution agent tasks with rich action spaces.

## Headline numbers

| Metric | Value |
|---|---|
| WANDERING prevalence (Qwen Phase 6) | 34% (20/59 failures), 95% CI [22.0%, 45.8%] |
| Tool-entropy W/S ratio (Qwen) | 0.41, p = 1.0×10⁻⁶ |
| Tool-entropy W/S ratio (Llama-70b) | 0.41, p < 10⁻¹⁵ |
| Tool-entropy W/S ratio (GPT-5 router) | 0.71, p = 8.9×10⁻³⁵ |
| Cross-task MALT W/S ratio | 1.007, p = 0.81 (NULL — scoping result) |
| **Tier-3 detector (v1 ∪ v5) recall** | **70%** |
| **Tier-3 detector SUCCESS FP** | **5%** |
| Combined v1 ∪ v4 advisory tier recall | 80% (at 30% FP, 15-turn lead) |

## Files in this card

- `inflection_wandering.pdf` — 11-page compiled PDF (paper)
- `inflection_wandering.tex` — template-agnostic LaTeX source
- `fig1_cross_arch_entropy.pdf` — HEADLINE figure: cross-arch entropy histograms
- `fig2_disagreement_trajectory.pdf` — cross-layer disagreement evolution
- `fig3_detector_comparison.pdf` — 6-detector landscape + 3-tier regions
- `fig4_lab_summary.pdf` — 4-lab W/S ratio bar chart
- `fig5_venn_orthogonality.pdf` — v1 ∩ v4 ∩ v5 captures Venn

## Reproducibility

All scripts + per-trajectory output JSONs at **[github.com/OpenInterpretability/openinterp-swebench-harness](https://github.com/OpenInterpretability/openinterp-swebench-harness)**. Apache-2.0.

Datasets used:
- OpenInterp Phase 6 (own; 99 Qwen3.6-27B SWE-bench Pro trajectories with per-turn residuals at L11/L23/L31/L43/L55 in bf16 safetensors; release planned upon paper acceptance)
- [`nebius/SWE-agent-trajectories`](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) (CC-BY-4.0)
- [`JetBrains-Research/agent-trajectories-swesmith-random-subset`](https://huggingface.co/datasets/JetBrains-Research/agent-trajectories-swesmith-random-subset) (HF public)
- [`SWE-bench/SWE-smith-trajectories`](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) (HF public)
- [`metr-evals/malt-public`](https://huggingface.co/datasets/metr-evals/malt-public) (HF gated, approval required)

## Citation

```
@misc{vicentino2026toolentropy,
  title = {Tool-Entropy Collapse: A Cross-Architecture Signature of Agent WANDERING Failure},
  author = {Vicentino, Caio},
  year = {2026},
  publisher = {Zenodo / OpenInterpretability},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/OpenInterpretability/openinterp-swebench-harness}
}
```

(DOI placeholder — update after Zenodo publish.)

## Strategic positioning

This paper extends the probe-based monitoring paradigm (Apollo deception probes, Anthropic sleeper-agent defection probes) by:
1. Quantifying a **previously unmeasured blind spot** (34% WANDERING failures)
2. Demonstrating a **probe-free complementary signal** (tool-use entropy) that works without model access
3. Validating **cross-architecture** (3 labs, 3 model families, 3 agent scaffolds)
4. Honestly **scoping** via MALT null finding
5. Providing **operational deployment guidance** (3 tiers) for production safety monitoring

The Tier-3 autonomous-termination detector is the first reported in this paradigm to achieve sub-5% false-positive rate at high recall — making production deployment viable.
