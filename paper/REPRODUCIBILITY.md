# Reproducibility map — the WANDERING arc (7 papers + companion note)

Every paper's reproduction artifacts are split the standard way: **code on GitHub** (this repo, Apache-2.0),
**data / activations / intervention state on Hugging Face** (large `*.pt` / captures are git-ignored by design
and live on HF), and the **paper-of-record PDF + permanent DOI on Zenodo**. The model is open-weight
(`Qwen/Qwen3.6-27B`; cross-model: `mistralai/Mistral-7B-Instruct-v0.3`, `mistralai/Mistral-Small-24B-Instruct-2501`).

Audited 2026-06-10: all paths below are git-tracked and pushed to `origin/main`; all HF datasets are public.

| # | Paper (DOI) | Code (this repo) | Pre-reg / EVAL | Data / state (HF) |
|---|---|---|---|---|
| 1 | Tool-Entropy Collapse (10.5281/zenodo.20368601) | `paper/inflection_wandering.tex`, `scripts/v5_tool_entropy.py`, `scripts/inflection_turn_analysis.py`, `scripts/inflection_per_layer_robustness.py` | `paper/preflight_probe_eval*.md` | `caiovicentino1/swebench-pro-qwen36-27b-phase6` |
| 2 | The Right Locus Is Still Not a Rescue Lever (10.5281/zenodo.20490278) | `paper/paper2/`, `scripts/paper2_l11_honest_paired_analysis.py` | in `paper/paper2/` | `caiovicentino1/swebench-phase6-verdict-circuit` |
| 3 | Multi-Channel Signatures (10.5281/zenodo.20490284) | `paper/paper3/`, `scripts/paper3_*` (incl. `paper3_liu_judge_*`), `scripts/inflection_turn_out/paper3_results*/` | in `paper/paper3/` | `…/swebench-phase6-verdict-circuit` (`features_n99.csv`) |
| 4 | Modality Matters (10.5281/zenodo.20490286) | `paper/paper4/`, `paper/exp_b_*`, `paper/exp_d_results.md` | in `paper/paper4/` | bundle (above) |
| 5 | The Verdict Is Not the Lever (10.5281/zenodo.20532769) | `paper/verdict_circuit/`, `scripts/verdict_circuit_stage0.py`, `notebooks/nb_verdict_circuit_{stage1,h4}.ipynb` | `paper/verdict_circuit/PREREG_*.md`, `STAGE*_RESULTS.md` | `…/verdict-circuit` (`results/STAGE*`) |
| 6 | The Lever Is Late (10.5281/zenodo.20534219) | `paper/breakthrough/`, `tools/decision_locator/` + standalone repo `OpenInterpretability/decision-locator` | in `paper/breakthrough/` | bundle (`results/`) |
| 7 | The Lever Generalizes — and It Brakes (10.5281/zenodo.20634838) | `paper/circuit_breaker/`, `scripts/commit_lever_{stages,stats}.py`, `scripts/drive_commitment_lever.py`, `scripts/run_commitment_lever.py`, `scripts/cross_model_lite.py`, `scripts/strengtheners.py` | `paper/circuit_breaker/PREREG_{commitment_lever,cross_model}.md`, `EVAL_{commitment_lever,paper7}.md` | bundle: `results/commit_lever_state.pt`, `commit_lever_partial.json`, `cross_model_*.json`, `strengtheners.json` |
| — | Companion note: No Better Than Behavioral (10.5281/zenodo.20500053) | `paper/context_rot/` | in `paper/context_rot/` | bundle |

## The released tool
`decision-locator` (model-agnostic locate / sweep_patch / steer_generate, prefill-only): standalone public repo
`OpenInterpretability/decision-locator` (pip-installable, CPU self-test + Colab), in-repo copy at
`tools/decision_locator/`. This is the method used in papers #6 and #7.

## How to reproduce (any paper)
1. `git clone https://github.com/OpenInterpretability/openinterp-swebench-harness`
2. Pull the data/state bundle for that paper from the HF dataset in the table (`huggingface_hub`).
3. Run the listed script / notebook (GPU needed for the generation-confirmed interventions in #5–#7; the
   detection / statistics steps are CPU-only). Decision points are deterministic (sorted traces), so the
   intervention numbers are exactly reproducible.

## What is *not* in git (by design)
Raw activation captures and large tensors (`traces/`, `captures/`, `*.pt`, `*.safetensors`) are git-ignored and
hosted on HF — cloning the repo gives all code; the bundle gives the data. The arc PDFs are mirrored at
`caiovicentino1/wandering-arc-papers`.
