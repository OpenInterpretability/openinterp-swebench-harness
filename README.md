# openinterp-swebench-harness

Instrumented agent harness for capturing SAE feature trajectories during SWE-bench Pro traces on Qwen3.6-27B. Built to support **mechanistic anatomy of agent reasoning failure** research (project_swebench_pro_failure_anatomy).

## Why a custom harness

OpenHands abstracts `model.generate()` behind multiple layers; vLLM doesn't expose forward hooks at intermediate transformer layers. We need both for our research:
- Direct control of forward pass to install LayerTap hooks at L11/L23/L31/L43/L55
- Decision-point capture (not every token) of bf16 hidden states
- Full reproducibility contract: same seed + same problem → same first 5 turns byte-identical

This harness is intentionally minimal: `transformers` direct, persistent bash via `pexpect`, str_replace_editor reimplemented, single chat loop in `agent/loop.py`.

## Status

V1 — Phase 0 (smoke). See `notebooks/nb_swebench_v0_smoke.ipynb`.

| Gate | Description | Status |
|---|---|---|
| G1 | 1 problem end-to-end + captures saved | pending |
| G2 | Determinism: rerun byte-identical first 5 turns | pending |
| G3 | Pass rate ∈ [30%, 60%] on 20 stratified problems | pending |
| G4 | Capture audit: shapes/labels/timing match trace log | pending |
| G5 | Resource ceiling: ≤5 min wall + ≤50GB VRAM per problem | pending |

Phase 1+ scale only runs after all 5 gates green.

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
