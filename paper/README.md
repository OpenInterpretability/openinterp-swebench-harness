# κ_t Explore-Consolidate Paper — Reproducibility

Paper: `kappa_t_coherence_buildup.md` (full v2 — U-shape framing)
Outline: `kappa_t_coherence_buildup_outline.md` (original v1 — superseded but kept for history)

Landing page: `openinterp_landing.md` → openinterp.org/research/papers/kappa-t-coherence-buildup
Zenodo metadata: `zenodo_metadata.yaml`
HF dataset card: `hf_dataset_card.md`
Tweet drafts: `tweet_draft.md`

## Repro pipeline

The full paper can be reproduced on a CPU-only machine in ~30 minutes once you have the Phase 6 captures (single A100, ~12h — see Phase 6 builder in `scripts/`).

```bash
# Phase 6 capture (Colab, GPU)
# Use scripts/build_nb_swebench_v6_phase6_scale.py to generate notebook, run on Colab A100

# κ_t analysis (local Mac, CPU)
python3 scripts/run_kappa_t_v2.py    # v2 with 5 axes — pre-registered mean κ̄ headline
python3 scripts/run_kappa_t_v3.py    # v3 internal replication (9 axes)
python3 scripts/run_kappa_t_controls.py  # 5 robustness controls (C1-C5; C1 reveals length-confound)
python3 scripts/run_kappa_t_c6_length_controlled.py  # early/late half decomposition (U-shape rescue)

# Figures
python3 scripts/make_figures_kappa_t.py     # fig1-fig6
python3 scripts/make_fig7_controls.py       # fig7 (after controls finish)
# fig8 (early/late half) forthcoming
```

Output directories on Drive: `kappa_t_v2/`, `kappa_t_v3/`, `kappa_t_controls/`.

## Figures (paper/figures/)

| Filename | Caption |
|---|---|
| `fig1_kappa_t_trajectories.png` | Per-class mean κ_t over agent turns, bootstrapped 95% bands |
| `fig2_slope_distributions.png` | Per-trace slope distributions, success vs failure |
| `fig3_probe_aurocs.png` | Individual probe AUROCs (v2 and v3 sets) |
| `fig4_probe_correlations.png` | Probe-score pairwise correlation heatmap |
| `fig5_mean_kappa_violin.png` | Per-trace mean κ̄ violin plot by class |
| `fig6_v2_vs_v3_replication.png` | v2 vs v3 gate-by-gate comparison |
| `fig7_controls_summary.png` | 5-control robustness panel (C1-C5) |

## What's NOT here

- The Phase 6 raw `.safetensors` captures — these are ~900MB total and will be uploaded to the HF dataset `openinterp/kappa-t-coherence-buildup` post-publication.
- The Qwen3.6-27B model weights — see https://huggingface.co/Qwen/Qwen3-Next-27B-Instruct (model class `qwen3_5`, requires `transformers` from main).

## Reading order if you're a reviewer

1. **Abstract + §1.3** (5 min) — what's the claim and contribution
2. **§3.2** (2 min) — the headline statistic
3. **§4** (5 min) — robustness controls
4. **Appendix B** (3 min) — walk-back history (why credibility comes from gates, not from the finding in isolation)
5. **§2** (10 min) — full method if §3 looks suspicious
6. **§5–6** (10 min) — discussion + related work

## Authorship & contact

Caio Vicentino (caio@openinterp.org), ORCID 0009-0003-4331-6259, OpenInterpretability.
