---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - mechanistic-interpretability
  - agent-failure-modes
  - chain-of-thought
  - swe-bench
  - tool-use
  - wandering
  - reasoning
size_categories:
  - n<1K
pretty_name: SWE-bench Pro Qwen3.6-27B Phase 6 — Trajectories + Residuals
---

# SWE-bench Pro Qwen3.6-27B Phase 6 — Trajectories + Residuals

Companion data for the **Tool-Entropy Collapse** paper [(Zenodo DOI 10.5281/zenodo.20368601)](https://zenodo.org/records/20368601) and **Two Honest Nulls** paper #2 (in flight).

99 multi-turn agent trajectories from Qwen3.6-27B running SWE-bench Pro (qutebrowser / openlibrary / ansible), with per-turn residual-stream activations captured at L11 / L23 / L31 / L43 / L55, plus all derived features and labels needed to reproduce both papers and run downstream analyses.

## Use cases

1. **Inspect Evals tool_entropy_collapse eval**: load traces + sub-class labels to test WANDERING detectors
2. **Mech-interp research on residual streams**: 99 × 5-layer × per-turn activations as bf16 safetensors
3. **SWE-bench Pro failure-mode analysis**: reproducibility for any future causal/probing/steering work
4. **Agent monitoring benchmarks**: gold WANDERING/SUCCESS/LOCKED labels at trace level

## Directory structure

```
.
├── README.md                              # this file
├── selected_iids.json                     # which 99 SWE-bench Pro instances
├── phase6_results.json                    # per-instance index (finish_reason, n_turns, paths)
├── phase6_aggregate.json                  # aggregate run stats
├── phase6_n99_verdict.json                # paper #1 verdict numbers
├── kappa_t_per_trace.json                 # κ_t coherence-buildup time series per trace
├── kappa_t_failure_clusters.json          # κ_t cluster analysis (failures)
├── kappa_t_success_clusters.json          # κ_t cluster analysis (successes)
├── phase6c_methodology_sweep.json         # methodology robustness checks
├── phase6c_preview.json                   # methodology preview
├── phase7_steering_pilot.json             # Phase 7 steering pilot output
├── traces/                                # 99 × (trace JSON + agent patch)
│   ├── instance_<repo>_<sha>_v<sha>.json  # per-turn tool_calls, thinking, content, tool_results
│   └── instance_<repo>_<sha>_v<sha>.patch # agent's final submitted/typed patch
├── captures/                              # 99 × (residual safetensors + meta)
│   ├── instance_<...>.safetensors         # per-turn residuals at L11/L23/L31/L43/L55 (bf16)
│   └── instance_<...>.meta.json           # token positions + structure metadata
├── features/                              # derived features (computed in repo scripts)
│   ├── inflection_results.json            # WANDERING/SUCCESS/LOCKED sub-class labels
│   ├── early_warning_results.json         # v1 forensic detector outputs
│   ├── early_warning_v2_results.json      # v2 naive early-warning text extension
│   ├── early_warning_v3_persistence.json  # v3 persistence test (refuted in opposite direction)
│   ├── early_warning_v4_cross_layer.json  # v4 cross-layer probe disagreement
│   ├── early_warning_v4_midlayer.json     # v4 mid-layer ablation
│   ├── early_warning_v4_op_sweep.json     # v4 hyperparameter sweep
│   ├── complementary_monitor.json         # v5 tool-entropy collapse outputs
│   ├── cross_task_validation.json         # METR MALT cross-task null
│   ├── exp_b_determinism_check.json       # Plan B feasibility audit (paper #2)
│   └── exp_d_forced_finish_counterfactual.json # Exp D null verdict (paper #2)
└── phase6b/                               # SWE-bench Pro Docker evaluation results
    └── phase6b_results.json               # 89 instances × 3 conditions (none/golden/agent)
```

## Loading examples

### Load 99 trajectory IDs

```python
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download(repo_id="caiovicentino1/swebench-pro-qwen36-27b-phase6",
                       filename="selected_iids.json", repo_type="dataset")
iids = json.load(open(path))  # list[str]
```

### Load one trace + its residuals

```python
from huggingface_hub import hf_hub_download
import json, safetensors.torch as st

REPO = "caiovicentino1/swebench-pro-qwen36-27b-phase6"
REVISION = "<pinned-sha>"  # see latest SHA below

iid = "instance_ansible__ansible-0ea40e09d1b35bcb69ff4d9cecf3d0defa4b36e8-v30a923fb5c164d6cd18280c02422f75e611e8fb2"

trace_path = hf_hub_download(repo_id=REPO, filename=f"traces/{iid}.json",
                              repo_type="dataset", revision=REVISION)
trace = json.load(open(trace_path))
print(f"finish_reason={trace['finish_reason']}, n_turns={trace['n_turns']}")

caps_path = hf_hub_download(repo_id=REPO, filename=f"captures/{iid}.safetensors",
                             repo_type="dataset", revision=REVISION)
caps = st.load_file(caps_path)
# keys: "t{turn}_pre_tool_p{pos}_L{layer}" → torch.Tensor[d_model=5120] bf16
```

### Load WANDERING/SUCCESS/LOCKED labels

```python
inflection_path = hf_hub_download(repo_id=REPO,
    filename="features/inflection_results.json", repo_type="dataset", revision=REVISION)
infl = json.load(open(inflection_path))
sub_class = {}
for t in infl['per_trajectory']:
    if t['label'] == 1:
        sub_class[t['iid']] = 'success'
    elif t.get('lock_fail_0.40') is not None:
        sub_class[t['iid']] = 'locked'
    else:
        sub_class[t['iid']] = 'wandering'
print(f"SUCCESS={sum(1 for v in sub_class.values() if v=='success')} "
      f"LOCKED={sum(1 for v in sub_class.values() if v=='locked')} "
      f"WANDERING={sum(1 for v in sub_class.values() if v=='wandering')}")
# Expected: SUCCESS=40, LOCKED=39, WANDERING=20
```

## Schema details

### Trace JSON (`traces/<iid>.json`)

```python
{
  "instance_id": str,            # SWE-bench Pro instance ID
  "seed": int,                   # deterministic seed
  "config": {
    "model": "Qwen3.6-27B",
    "temperature": 1.0,
    "top_p": 1.0,
    "thinking_mode": true,
    "capture_layers": [11, 23, 31, 43, 55],
  },
  "finished": bool,
  "finish_reason": "finish_tool" | "max_turns" | "error",
  "wall_seconds": float,
  "n_turns": int,
  "n_captures": int,
  "error": str | null,
  "turns": [
    {
      "turn_idx": int,
      "prompt_tokens": int,
      "new_tokens": int,
      "wall_seconds": float,
      "raw_response": str,
      "thinking": str | null,
      "content": str,
      "tool_calls": [{name, arguments|args, ...}],
      "tool_results": [{output, exit_code, ...}],
      "capture_token_pos": {label: [token_positions]},
      "n_capture_steps": int,
    },
    ...
  ]
}
```

### Captures safetensors (`captures/<iid>.safetensors`)

Tensor keys follow the pattern `t{turn}_{position}_p{pos_idx}_L{layer}`:

- `turn` ∈ 0..n_turns-1
- `position` ∈ {`think_start`, `think_mid`, `think_end`, `pre_tool`, `turn_end`}
- `pos_idx` (positional disambiguator when multiple captures per turn)
- `layer` ∈ {11, 23, 31, 43, 55}

Each tensor: `shape=(d_model=5120,)`, `dtype=bfloat16`.

Companion `<iid>.meta.json` has full token position structure per turn.

### inflection_results.json

```python
{
  "n_trajectories": 99,
  "per_trajectory": [
    {
      "iid": str,
      "n_turns": int,
      "label": 0|1,                  # 1 = SUCCESS, 0 = failure
      "lock_succ_0.70": int|null,    # turn where probe stably > 0.70 (SUCCESS lock-in)
      "lock_fail_0.40": int|null,    # turn where probe stably < 0.40 (LOCKED lock-in)
      "score_trajectory": [float],   # per-turn probe score [0,1]
    },
    ...
  ]
}
```

Sub-class derivation rule:
- `label==1` → **SUCCESS** (n=40)
- `label==0` AND `lock_fail_0.40 is not None` → **LOCKED** (n=39, probe collapsed to < 0.30 by mean fraction 0.92 of trajectory length)
- `label==0` AND `lock_fail_0.40 is None` → **WANDERING** (n=20, probe stays > 0.70 with median final score 1.000, 95% produce patches but no `finish_tool` emission)

## Provenance

- **Model**: Qwen3.6-27B (Alibaba 2026), `enable_thinking=True`, temperature=1.0
- **Task**: SWE-bench Pro (Pro-balanced split), MIT-licensed
- **Scaffold**: OpenInterpretability custom agent loop (forward-hook compatible) — [github.com/OpenInterpretability/openinterp-swebench-harness](https://github.com/OpenInterpretability/openinterp-swebench-harness)
- **Hardware**: NVIDIA RTX 6000 Pro Blackwell (48 GB), bfloat16 inference
- **Run date**: 2026-05-06 → 2026-05-08
- **Phase 6b Docker eval**: completed 2026-05-24 → 2026-05-25 (local Mac CPU + Docker)

## Hardware-determinism caveat (important for WANDERING category)

The WANDERING sub-class label was assigned by single-run classification on RTX 6000 Pro Blackwell (all 20 WANDERING instances had `finish_reason=max_turns` by that single run, with probe scores > 0.70). When the same 20 trajectories are re-run on **H100 80GB with the same deterministic seed and no intervention**, 7/20 = 35% emit `finish_tool` naturally. The WANDERING phenotype is thus partly artifact of single-run + RTX 6000 floating-point determinism; cross-hardware re-classification would tighten the category.

Companion paper #2 (Two Honest Nulls) discusses this and recommends multi-seed classification protocols for future WANDERING labelling.

## Citation

If you use this dataset, please cite the companion papers:

```bibtex
@misc{tool_entropy_2026,
  title={Tool-Entropy Collapse: A Cross-Architecture Signature of Agent WANDERING Failure},
  author={Vicentino, Caio},
  year={2026},
  doi={10.5281/zenodo.20368601},
  url={https://zenodo.org/records/20368601}
}

@misc{two_honest_nulls_2026,
  title={Two Honest Nulls on the WANDERING Mechanism Hypothesis: Pre-registered Causal Tests Refine the Mid-Layer-to-Edge Picture},
  author={Vicentino, Caio},
  year={2026},
  note={Companion to Tool-Entropy Collapse, draft in flight}
}
```

## Related resources

- **Code repository**: [OpenInterpretability/openinterp-swebench-harness](https://github.com/OpenInterpretability/openinterp-swebench-harness)
- **Inspect Evals integration**: [OpenInterpretability/inspect-tool-entropy-collapse](https://github.com/OpenInterpretability/inspect-tool-entropy-collapse) (coming soon)
- **Paper #1 dataset (figures + PDF)**: [caiovicentino1/tool-entropy-collapse-paper](https://huggingface.co/datasets/caiovicentino1/tool-entropy-collapse-paper)
- **OpenInterpretability lab homepage**: [openinterp.org](https://openinterp.org)

## License

Apache 2.0 — free for research and commercial use with attribution.
