# openinterp-swebench-harness

Instrumented agent harness for capturing residual-stream activation trajectories during SWE-bench Pro traces on Qwen3.6-27B. Powers the **paper-5 saturation-direction lever** series — a 12-phase mechanistic study of probe causality on a 27B reasoning agent.

> **Paper-5 published 2026-05-09** — *Saturation-Direction Lever: A Five-Class Taxonomy of Probe Causality in Qwen3.6-27B* — [openinterp.org/research/papers/saturation-direction-probe-levers](https://openinterp.org/research/papers/saturation-direction-probe-levers)
>
> Headline: **α=−100 robustness theorem** — L31 pre_tool capability probe produces +33-40pp probe-vs-random pushdown gap across code distributions spanning Qwen pass-rate ~7-89%, validated through two pre-registered falsification cycles.

## Why a custom harness

OpenHands abstracts `model.generate()` behind multiple layers; vLLM doesn't expose forward hooks at intermediate transformer layers. We need both for our research:
- Direct control of forward pass to install LayerTap hooks at L11/L23/L31/L43/L55
- Decision-point capture (not every token) of bf16 hidden states
- Full reproducibility contract: same seed + same problem → same first 5 turns byte-identical

This harness is intentionally minimal: `transformers` direct, persistent bash via `pexpect`, str_replace_editor reimplemented, single chat loop in `agent/loop.py`.

## Status — paper-5 published 2026-05-09

V1 complete through **Phase 11e** (multi-site α=−100 robustness on Codeforces). 12-phase causal-locus study on Qwen3.6-27B.

### Phase ladder

| Phase | What | Outcome |
|---|---|---|
| 0 — Smoke | 1 problem, full agent loop + 1155 captures + patch | 🟢 G1+G4 PASS |
| 1 — N=20 stratified | 20 Python problems, 12k captures, 55% patch-gen rate | 🟢 G4 GREEN, data CLEAN |
| 2 — Differential probes | 25 probes × 5 positions, AUROC up to 0.958 single, 0.958 LORO | 🟢 STRONG_SIGNAL |
| 5d/6/6c — N=99 + methodology sweep | random-feature baseline + capacity sweep mandatory at N<100 | L43 pre_tool gap +0.269; Phase 5d AUROC=1.000 was over-param |
| 7 — Causality (L43 pre_tool) | log-prob proxy + control-token norm + behavioral steering | 🔵 epiphenomenal (softmax-temp artifact) |
| 8 — CoT-Integrity causality | bidirectional α-sweep ±200 + structural-rigidity diagnostic | 🔵 L55 thinking template-locked |
| 10 — FG/RG causality | RG L55 mid_think shows asymmetric pushup at α=+200 | 🟢 First lever (continuous-quality probe) |
| 11/11b — Capability locus | 4/4 capability sites at decision-bottleneck positions show pushdown-asymmetric lever | 🟢 +30-60pp gaps |
| 12 — Persona falsifier | predicted pushup-asymmetric (continuous), observed pushdown | 🅑 Falsifier #1 → motivates saturation-direction theory |
| 11c — Cross-distribution (BCB) | BCB +33pp at α=−100 (vs +40pp HE+MBPP) | 🟡 Initially proposed saturation-magnitude corollary |
| 11d — Codeforces (lowest saturation) | CF +40pp at α=−100, no pushup emergence | 🅑 Falsifier #2 → corollary dropped, **α=−100 robustness theorem** |
| 11e — Multi-site CF | 4 capability sites × Codeforces ≥2000 | running 2026-05-09 |

### Paper-5 thesis (current)

> **α=−100 robustness theorem**: the L31 pre_tool capability probe direction produces a +33-40pp probe-vs-random pushdown gap across code distributions spanning Qwen3.6-27B pass-rate ~7-89%. The α=−100 locus is **saturation-independent** at moderate amplitude.

Two pre-registered falsification cycles documented (Phase 12 persona + Phase 11d corollary). 8 probes mapped across 5 empirical classes. Saturation-direction principle survives both falsifications as the unifying mechanism.

### Notebooks

`notebooks/nb_swebench_v{0,1,2,3,...,11,11b,11c,11d,11e,12}_*.ipynb` — one standalone Colab per phase. All build scripts in `scripts/build_nb_swebench_v*.py` for reproducibility.

## Architecture

```
openinterp-swebench-harness/
├── config.py              single source of truth for V1
├── runner.py              orchestrator + smoke driver
├── verdict.py             SUCCESS/FAIL/ERROR aggregation
├── agent/
│   ├── loop.py            chat loop multi-turn
│   ├── tools.py           bash + str_replace_editor + finish (OpenAI fn-call)
│   ├── parser.py          parse Qwen function_call output
│   └── prompts.py         system prompt + render template
├── instrumentation/
│   ├── layertap.py        forward hooks at L11/L23/L31/L43/L55
│   └── capture.py         decision-point save into safetensors
├── sandbox/
│   └── exec.py            persistent bash session via pexpect
├── notebooks/                ~13 standalone Colab notebooks, one per phase
├── scripts/                  build_nb_swebench_v*.py — regenerates each notebook
├── paper/                    paper-5 protocol + meta-analysis + sub-papers
└── artifacts/                agent-probe-guard probe weights for HF
```

## Tool definitions (final)

| Tool | Args | Behavior |
|---|---|---|
| `bash` | `command, timeout=120` | persistent shell session, cwd preserved |
| `str_replace_editor` | `command: view\|create\|str_replace\|insert, path, ...` | Anthropic-style file ops |
| `finish` | `summary` | terminate trace |

## Decision points captured

`think_start`, `think_mid` (every 200 tokens in think), `think_end`, `pre_tool_bash`, `pre_tool_edit`, `post_tool_result`, `turn_end`, `finish`. All × 5 layers (L11, L23, L31, L43, L55).

Storage: `captures/{instance_id}.safetensors` with metadata `{instance_id, turn_idx, position_label, token_pos, layer}`.

## Reproducibility contract

```python
config = {
    "model": "Qwen/Qwen3.6-27B",
    "dtype": "bfloat16",
    "temperature": 1.0,        # Qwen reported scaffold
    "top_p": 0.95,             # Qwen reported scaffold
    "max_context": 200_000,
    "max_turns": 50,
    "thinking_mode": True,
    "seed_per_problem": "hash(instance_id) % 2**32",
}
```

## Install

```bash
pip install -e .
```

Requires Python 3.11, transformers 5.x, flash-attn 2.x, pexpect, datasets, safetensors, torch.

## License

Apache-2.0.
