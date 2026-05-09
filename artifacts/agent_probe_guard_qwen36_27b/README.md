---
license: apache-2.0
language:
- en
pretty_name: agent-probe-guard probe weights for Qwen3.6-27B
tags:
- mechanistic-interpretability
- probing
- linear-probe
- agent
- qwen
- detection
- mid-reasoning
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: meta
    path: meta.json
---

# agent-probe-guard probe weights for Qwen3.6-27B

**Two-probe activation gate for code agents on Qwen3.6-27B.**

This dataset bundles the trained probe weights, scalers, dim selections, and
provenance metadata used by the `openinterp.AgentProbeGuard` SDK. Detection-only
by design — see the paper.

## What's inside

| File | Layer | Position | Capacity (K) | CV AUROC | Random K-matched | Gap |
|---|---|---|---|---|---|---|
| `probe_L55_thinking.joblib` | 55 | last_prompt_token | 5 | **0.848** | 0.701 | **+0.147** 🟢 |
| `probe_L43_pre_tool.joblib` | 43 | pre_tool *(or last_prompt as v0.1 placeholder)* | 10 | 0.830 | 0.749 | +0.080 |
| `meta.json` | — | — | — | full provenance | — | — |

Both probes are sklearn `LogisticRegression` on `StandardScaler`-transformed
top-K diff-of-means dimensions. Loadable with:

```python
import joblib
art = joblib.load("probe_L55_thinking.joblib")
art["probe"], art["scaler"], art["dims"], art["layer"]
```

## ⚠️ Detect-only at these sites; lever exists at others

The two probes shipped here (L43 pre_tool, L55 thinking) DO NOT lever model
behavior. We confirmed this across three intervention experiments (paper §5.1,
§5.2, §5.3):

- **L43 pre_tool**: probe direction adds a uniform softmax-temperature shift, not
  a target-specific bias. Δrel after control-token normalization ≈ 0.
- **L55 thinking**: residual at last-prompt is structurally downstream of the
  chat template's auto-injected `<think></think>` tokens. Even at α=+200
  (perturbation 86% above ‖residual‖) on the K=5 paper-grade direction, output
  does not change.

If you want to use these for steering, expect null results. If you want to
use them for routing decisions (skip / escalate / proceed), they work.

### But other capability sites DO lever — see paper-5

Subsequent experiments (Phase 11/11b/11d on the same Qwen3.6-27B + same Phase 6
N=99 capture corpus) found that L23 pre_tool, L31 pre_tool, L43 turn_end, and
L55 pre_tool produce **+30 to +60pp probe-vs-random pushdown gaps** at α=−100.
The L31 pre_tool gap (+33-40pp) is **saturation-independent** across code
distributions spanning Qwen pass-rate ~7-89% (HumanEval+MBPP, BigCodeBench,
Codeforces ≥2000). This is the **α=−100 robustness theorem** of paper-5.

The L43 pre_tool probe shipped here was the locus selected from Phase 6c
(N=42 methodology sweep) before we ran Phase 11+11b on the full N=99 corpus.
The N=99 verdict found L43 pre_tool below paper-grade and L31 pre_tool / L43
turn_end as the strongest pushdown levers — see paper-5 §5 for the corrected
locus map. v0.2 of this artifact will swap L43 pre_tool for L31 pre_tool +
add cross-distribution validation metadata.

## 🔬 Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from openinterp import AgentProbeGuard

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.6-27B", dtype="bfloat16",
    device_map="cuda", trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)

guard = AgentProbeGuard.from_pretrained("Qwen/Qwen3.6-27B")
guard.attach(model, tok)

decision = guard.assess(messages, partial_response=current_thought)
# decision.action in {"skip", "escalate", "proceed"}
# decision.scores = {"capability": float, "thinking": float}
```

## v0.1 status — what's solid, what's placeholder

| Probe | Status |
|---|---|
| **L55 thinking** | ✅ **Solid.** Trained on N=240 HotpotQA prompts (4 retrieval conditions × 60 q). Paper-grade gap +0.147 above random K=5 baseline. Phase 8 redux confirmed structural-lock under steering. |
| **L43 capability** | 🟡 **Placeholder.** Phase 1 N=99 SWE-bench Pro trace collection in flight at release time. v0.1 uses nb47b L43 captures with `has_think_v1` labels as a placeholder (effectively a second thinking probe at a different layer). Replace via `scripts/build_agent_probe_guard_hf_artifact.py` once Phase 1 metadata + L43 captures are uploaded to Drive. |

When the L43 probe is upgraded to actual SWE-bench Pro data, the dataset will
be tagged `v0.2.0` and the corresponding meta-json `capability.note` field
removed.

## 🛡️ Three sanity checks

The methodology that protects these numbers:

1. **Random-feature baseline at small N** (paper §3.1) — every AUROC reported
   alongside random K-matched baseline. Gap ≥ +0.10 = paper-grade.
2. **Control-token normalization for steering** (paper §3.2) — Δrel = Δ(target)
   − mean(Δ(controls)). Catches uniform softmax-temperature shifts.
3. **Structural-rigidity α-sweep** (paper §3.3) — sweep α to multiples of
   ‖residual‖ (e.g., ±200) before declaring null. Distinguishes amplitude null
   from template-locked null.

Each check caught a confident-but-wrong claim during the work.

## 📜 Citation

Primary paper for these two probes (epiphenomenal regime):

```bibtex
@inproceedings{vicentino2026twoforms,
  title={Two Forms of Epiphenomenal Probes in Code Agents:
         Mid-Reasoning Capability and Chain-of-Thought Emission in Qwen3.6-27B},
  author={Vicentino, Caio},
  booktitle={NeurIPS 2026 Mechanistic Interpretability Workshop},
  year={2026},
  url={https://github.com/OpenInterpretability/openinterp-swebench-harness}
}
```

For the broader 5-class probe-causality taxonomy + saturation-direction lever
(paper-5, includes the α=−100 robustness theorem):

```bibtex
@article{vicentino2026saturation,
  title={Saturation-Direction Lever: A Five-Class Taxonomy of Probe Causality
         in Qwen3.6-27B},
  author={Vicentino, Caio},
  journal={openinterp.org/research},
  year={2026},
  url={https://openinterp.org/research/papers/saturation-direction-probe-levers}
}
```

## 🔗 Reproduction

- Reproduction harness: <https://github.com/OpenInterpretability/openinterp-swebench-harness>
- SDK (PyPI): <https://pypi.org/project/openinterp/>
- SDK source: <https://github.com/OpenInterpretability/cli/blob/main/openinterp/agent_probe_guard.py>
- Eval doc v6: `paper/preflight_probe_eval_v6_phase8_template_lock.md`
- Workshop draft: `paper/two_forms_epiphenomenal_probes_neurips_mi_2026.md`
- Captures (Drive, ~25MB): `openinterp_runs/nb47b_capture/{L11,L23,L31,L43,L55}_pre_gen_activations.safetensors`

## License

Apache-2.0. Patent grant included. Suitable for enterprise-compliance review.
