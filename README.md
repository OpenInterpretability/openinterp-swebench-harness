# openinterp-swebench-harness

Instrumented agent harness for capturing SAE feature trajectories during SWE-bench Pro traces on Qwen3.6-27B. Built to support **mechanistic anatomy of agent reasoning failure** research (project_swebench_pro_failure_anatomy).

## Why a custom harness

OpenHands abstracts `model.generate()` behind multiple layers; vLLM doesn't expose forward hooks at intermediate transformer layers. We need both for our research:
- Direct control of forward pass to install LayerTap hooks at L11/L23/L31/L43/L55
- Decision-point capture (not every token) of bf16 hidden states
- Full reproducibility contract: same seed + same problem → same first 5 turns byte-identical

This harness is intentionally minimal: `transformers` direct, persistent bash via `pexpect`, str_replace_editor reimplemented, single chat loop in `agent/loop.py`.

## Status

V1 — Phase 0 SMOKE GREEN (2026-05-05). Phase 1 (20 stratified Python problems) ready to run.

| Gate | Description | Phase 0 | Phase 1 |
|---|---|---|---|
| G1 | End-to-end agent run + captures + patch | 🟢 PASS (46 turns, 1155 caps, 1333 B patch) | re-check at scale |
| G2 | Determinism: same seed → first 5 turns byte-equal | deferred | rerun cell |
| G3 | Soft pass-rate ∈ [30%, 60%] on 20 stratified | deferred | aggregate |
| G4 | Capture audit: shapes/labels match trace log | 🟢 PASS (1155/1155, d_model=5120) | per-problem |
| G5 | Resource ceiling | 🔴 INFO (40min/77GB — fla missing) | scaled targets |

Notebooks: `nb_swebench_v0_smoke.ipynb` (Phase 0, reproducibility reference) + `nb_swebench_v1_phase1.ipynb` (Phase 1, 20 problems with G2/G3 + resume-on-disconnect).

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
└── notebooks/
    └── nb_swebench_v0_smoke.ipynb
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
