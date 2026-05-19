---
license: cc-by-4.0
task_categories:
  - other
tags:
  - mechanistic-interpretability
  - llm-monitoring
  - probing
  - agent-evaluation
  - swe-bench
  - qwen
pretty_name: "κ_t Explore-Consolidate Dynamics — Phase 6 SWE-bench Pro residuals + probe scores"
size_categories:
  - 1K<n<10K
---

# κ_t Explore-Consolidate Dynamics — Dataset

Data accompanying *"Explore-Consolidate Dynamics in Cross-Probe Coherence Separate Successful and Failed LLM Agent Trajectories"* (Vicentino, 2026; openinterp.org/research/papers/kappa-t-coherence-buildup).

## What's in here

- **`captures/`** — 99 `.safetensors` files, one per SWE-bench Pro instance. Each contains residual stream snapshots at 11 layers × 4 positions per agent turn (~40 turns/trace avg), 5120 dim per residual.
- **`traces/`** — 99 `.json` files with full agent rollout: per-turn tool calls, tool results, thinking text, response text, wall-clock timings.
- **`patches/`** — 99 `.patch` files: the unified diff each agent produced (or empty if the agent gave up).
- **`results/phase6_results.json`** — per-instance summary: finished status, n_turns, n_captures, patch size, transformers commit hash.
- **`results/kappa_t_v2_results.json`** — paper headline statistics (5 axes).
- **`results/kappa_t_v3_results.json`** — internal replication (9 axes).
- **`results/kappa_t_controls_results.json`** — 5 robustness controls.

## Model & data provenance

- **Model:** Qwen3.6-27B-Instruct, `enable_thinking=True`, sampled at temperature 1.0
- **`transformers` commit:** `73d9159697a851c85623d0f03fcfbdd863d38975` (from `main`, May 2026)
- **Benchmark:** SWE-bench Pro train split, 99 instances randomly sampled
- **Capture protocol:** see `openinterp-swebench-harness/instrumentation/` for the `LayerTap` + `Runner` interface
- **Compute for capture:** single A100 80GB (Colab Pro+), ~12 hours
- **Compute for κ_t analysis:** MacBook Pro CPU, ~30 minutes total

## How to use

```python
from safetensors import safe_open
import json

iid = 'instance_qutebrowser__qutebrowser-e57b...'
with open(f'captures/{iid}.meta.json') as f:
    meta = json.load(f)
with safe_open(f'captures/{iid}.safetensors', framework='pt') as f:
    # Each record has activation_key, turn_idx, position_label, layer
    rec = meta['records'][0]
    residual = f.get_tensor(rec['activation_key']).float().numpy()  # shape (5120,)
```

To reproduce the paper's headline statistic (slope p=0.0003):

```bash
git clone https://github.com/OpenInterpretability/openinterp-swebench-harness
cd openinterp-swebench-harness
# Edit scripts/run_kappa_t_v2.py to point at the local path of this dataset
python3 scripts/run_kappa_t_v2.py
```

## Limitations

- Single model (Qwen3.6-27B). Multi-model validation pending.
- Single benchmark (SWE-bench Pro). Other agent tasks untested.
- N=99 traces. An attempted N=200 scale-up failed at the data-pipeline level (83% of additional instances had `base_commit` SHAs that could not be retrieved via simple `git clone + checkout`).
- Single layer (L43) × single position (`paired_concat`) reported in the paper.

## Citation

```bibtex
@misc{vicentino2026kappat,
  title  = {Cross-Probe Coherence Buildup in Successful LLM Agent Trajectories},
  author = {Vicentino, Caio},
  year   = {2026},
  doi    = {10.5281/zenodo.20278983},
  note   = {OpenInterpretability working paper. Zenodo: https://zenodo.org/record/20278983},
  url    = {https://openinterp.org/research/papers/kappa-t-coherence-buildup},
}
```

## License

Code: Apache-2.0 (in `openinterp-swebench-harness` repo).
Data + paper: CC-BY-4.0.
