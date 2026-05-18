"""Generate notebooks/nb_tool_doubt_phase1b_position_sweep.ipynb — STANDALONE diagnostic.

Phase 1 verdict 2026-05-18: best L11 AUROC 0.575, gap +0.065 → 2/4 gates,
substantively RED. Top-5 INSPECT-RAW all error_fake = lexical leak via
"ok": false / "Error:" strings. Truncate ranked BELOW clean = blind to silent
corruption.

Hypothesis to falsify before walk-back: turn_end position (start of next
assistant) was wrong. Doubt might live at other positions.

This notebook rebuilds everything from scratch (model, pairs, captures) since
Colab kernel was reset. Single notebook end-to-end.

Sweep: 10 layers × 4 positions × 60 pairs (same seed-42 pairs as Phase 1).

Decision: any (L, pos) gap > 0.15 → resgata. Else lock walk-back.

Compute: ~15-20 min Colab RTX 6000 (3min download + 5min model load + 5min
sweep capture + 30s train). ~R$2.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_tool_doubt_phase1b_position_sweep.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n", "outputs": [], "execution_count": None}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# Tool-Doubt Phase 1b — Position Sweep (Standalone)

Phase 1 verdict 2/4 gates, lexical leak via `"ok": false` / `"Error:"` strings.
This is the LAST diagnostic before walk-back.

## What it tests
4 positions × 10 layers per (clean, truncate, error_fake) pair:

**Positions:**
- `start_of_tool_msg` — first token of JSON tool block
- `mid_tool` — midpoint of tool message
- `end_of_tool_msg` — `<|im_end|>` of tool message
- `start_of_next_assistant` — what Phase 1 captured (last token of input)

**Layers:** L7, L11, L15, L19, L23, L31, L35, L43, L47, L55

## Decision
- ANY (L, pos) with gap > 0.15 → 🟢 RESGATA Phase 1 with this position
- ZERO combos pass → 🔴 LOCK walk-back, pivot to Inner-Outer Week 2

## Standalone — runs from scratch
~15-20min total: install + model download (~3min, 55GB) + model load + sweep capture + train.
"""),

    code("""
# 1) Install + transformers commit hash (ls-remote fallback)
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q accelerate scipy safetensors huggingface_hub datasets scikit-learn
!pip install -q flash-linear-attention causal-conv1d --no-build-isolation 2>&1 | tail -3 || true

import importlib, subprocess, os
TRANSFORMERS_COMMIT = ''
for pkg in ['transformers', 'sklearn', 'scipy', 'datasets']:
    try:
        m = importlib.import_module(pkg)
        print(f'  {pkg}: {getattr(m, \"__version__\", \"?\")}')
    except ImportError:
        print(f'  {pkg}: MISSING')

# ls-remote fallback (pip wheel strips .git/)
try:
    res = subprocess.run(
        ['git', 'ls-remote', 'https://github.com/huggingface/transformers.git', 'HEAD'],
        capture_output=True, text=True
    )
    if res.returncode == 0 and res.stdout:
        TRANSFORMERS_COMMIT = res.stdout.split()[0]
except Exception as e:
    print(f'  ls-remote failed: {e}')
print(f'\\n🔒 transformers commit: {TRANSFORMERS_COMMIT or \"unknown\"}')
print('\\nRESTART RUNTIME if transformers upgraded.')
"""),

    code("""
# 2) GPU + Drive + Phase 6 traces + SWE-bench Pro dataset
import subprocess, os, json, glob
out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print(f'GPU: {out}')

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs'
except ImportError:
    DRIVE_ROOT = os.path.expanduser('~/openinterp_runs')

PHASE6_TRACES_DIR = os.path.join(DRIVE_ROOT, 'swebench_v6_phase6', 'traces')
assert os.path.isdir(PHASE6_TRACES_DIR), f'Not found: {PHASE6_TRACES_DIR}'
trace_files = sorted(glob.glob(os.path.join(PHASE6_TRACES_DIR, 'instance_*.json')))
print(f'Phase 6 traces: {len(trace_files)}')

from datasets import load_dataset
ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
ds_by_id = {row['instance_id']: row for row in ds}
print(f'SWE-bench Pro instances: {len(ds_by_id)}')

OUT_DIR = os.path.join(DRIVE_ROOT, 'tool_doubt_phase1b_sweep')
DIRS = {k: os.path.join(OUT_DIR, k) for k in ['captures', 'results']}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)
print(f'Output: {OUT_DIR}')
"""),

    code("""
# 3) Load Qwen3.6-27B + clone harness
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'Qwen/Qwen3.6-27B'
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
)
model.eval()
LAYERS = model.model.layers if hasattr(model.model, 'layers') else model.model.language_model.layers
D_MODEL = model.config.hidden_size
N_LAYERS = len(LAYERS)
print(f'Model: d_model={D_MODEL}, n_layers={N_LAYERS}')

if not os.path.exists('/content/openinterp-swebench-harness'):
    !git clone -q https://github.com/OpenInterpretability/openinterp-swebench-harness.git /content/openinterp-swebench-harness
import sys
sys.path.insert(0, '/content/openinterp-swebench-harness')
from agent.prompts import SYSTEM_PROMPT, render_problem
from agent.parser import _strip_think
from agent.tools import TOOLS
print(f'Harness loaded. SYSTEM_PROMPT len={len(SYSTEM_PROMPT)}, tools={[t[\"function\"][\"name\"] for t in TOOLS]}')
"""),

    code("""
# 4) Helpers — same as Phase 1 + position resolver + multi-position capture
import torch, numpy as np, json

PROBE_LAYERS_SWEEP = [7, 11, 15, 19, 23, 31, 35, 43, 47, 55]
POSITIONS = ['start_of_tool_msg', 'mid_tool', 'end_of_tool_msg', 'start_of_next_assistant']


def corrupt_truncate(real_result_dict, n_chars=50):
    r = json.loads(json.dumps(real_result_dict))
    if isinstance(r.get('result'), dict) and 'content' in r['result']:
        c = r['result']['content']
        if isinstance(c, str):
            r['result']['content'] = c[:n_chars] + '...[truncated]'
    return r

def corrupt_error_fake(real_result_dict):
    return {'id': real_result_dict['id'], 'result': {'ok': False, 'error': 'File not found or permission denied.'}}

CORRUPTION_STRATEGIES = {'truncate': corrupt_truncate, 'error_fake': corrupt_error_fake}


def extract_turn0_replay(trace_path, ds_by_id):
    with open(trace_path) as f:
        trace = json.load(f)
    instance_id = trace['instance_id']
    if instance_id not in ds_by_id:
        return None
    instance = dict(ds_by_id[instance_id])
    instance['__workdir__'] = f'/content/work_p6/{instance_id}'
    user_msg = render_problem(instance)
    turn0 = trace['turns'][0]
    if not turn0.get('tool_calls') or not turn0.get('tool_results'):
        return None
    _, body_with_tools = _strip_think(turn0['raw_response'])
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_msg},
        {'role': 'assistant', 'content': body_with_tools},
    ]
    return messages, turn0['tool_results'][0], instance_id


def _len_ids(out):
    if hasattr(out, 'data') and isinstance(out.data, dict) and 'input_ids' in out.data:
        return int(out['input_ids'].shape[-1])
    if torch.is_tensor(out):
        return int(out.shape[-1])
    if isinstance(out, list):
        return len(out[0]) if out and isinstance(out[0], list) else len(out)
    return 0


def _ids_tensor(out):
    if hasattr(out, 'data') and isinstance(out.data, dict) and 'input_ids' in out.data:
        return out['input_ids']
    if torch.is_tensor(out):
        return out
    return torch.tensor([out] if isinstance(out, list) and not isinstance(out[0], list) else out, dtype=torch.long)


def resolve_positions(messages_pre, tool_result_dict):
    messages_with_tool = list(messages_pre) + [
        {'role': 'tool', 'content': json.dumps(tool_result_dict, ensure_ascii=False)[:32000]},
    ]
    out_full = tok.apply_chat_template(messages_with_tool, tools=TOOLS, add_generation_prompt=True, return_tensors='pt', enable_thinking=True)
    ids_full = _ids_tensor(out_full)
    full_len = int(ids_full.shape[-1])

    out_no_gen = tok.apply_chat_template(messages_with_tool, tools=TOOLS, add_generation_prompt=False, return_tensors='pt', enable_thinking=True)
    no_gen_len = _len_ids(out_no_gen)

    out_no_tool = tok.apply_chat_template(list(messages_pre), tools=TOOLS, add_generation_prompt=False, return_tensors='pt', enable_thinking=True)
    no_tool_len = _len_ids(out_no_tool)

    return ids_full, {
        'start_of_tool_msg': no_tool_len,
        'end_of_tool_msg': no_gen_len - 1,
        'mid_tool': (no_tool_len + no_gen_len - 1) // 2,
        'start_of_next_assistant': full_len - 1,
    }


@torch.no_grad()
def capture_at_positions(messages_pre, tool_result, probe_layers=PROBE_LAYERS_SWEEP):
    ids, positions = resolve_positions(messages_pre, tool_result)
    ids = ids.to(model.device)
    attn = torch.ones_like(ids)

    captures = {L: {pos_name: None for pos_name in POSITIONS} for L in probe_layers}
    def make_hook(L):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            for pos_name, pos_idx in positions.items():
                if 0 <= pos_idx < h.shape[1]:
                    captures[L][pos_name] = h[0, pos_idx, :].detach().float().cpu().numpy()
        return hook

    handles = [LAYERS[L].register_forward_hook(make_hook(L)) for L in probe_layers]
    try:
        model(ids, attention_mask=attn, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return captures, positions, int(ids.shape[-1])


print(f'Helpers OK. Layers: {PROBE_LAYERS_SWEEP}, Positions: {POSITIONS}')
"""),

    code("""
# 5) Rebuild same 60 pairs as Phase 1 (deterministic seed=42)
import random
random.seed(42)
N_TRACES = 20

pairs = []
checked = 0
for tf in trace_files:
    if len(pairs) >= N_TRACES * (1 + len(CORRUPTION_STRATEGIES)):
        break
    checked += 1
    ext = extract_turn0_replay(tf, ds_by_id)
    if ext is None:
        continue
    messages_pre, real_tool, instance_id = ext

    pairs.append({'trace_file': tf, 'instance_id': instance_id, 'messages_pre': messages_pre,
                  'tool_result': real_tool, 'label': 0, 'corruption': 'none'})
    for strat_name, strat_fn in CORRUPTION_STRATEGIES.items():
        pairs.append({'trace_file': tf, 'instance_id': instance_id, 'messages_pre': messages_pre,
                      'tool_result': strat_fn(real_tool), 'label': 1, 'corruption': strat_name})

print(f'Built {len(pairs)} pairs from {len(pairs)//3} traces (checked {checked})')
print(f'  Clean: {sum(1 for p in pairs if p[\"label\"]==0)}')
print(f'  Truncate: {sum(1 for p in pairs if p[\"corruption\"]==\"truncate\")}')
print(f'  Error_fake: {sum(1 for p in pairs if p[\"corruption\"]==\"error_fake\")}')
"""),

    code("""
# 6) Capture sweep — 60 pairs × 10 layers × 4 positions
import os, json, time, numpy as np

CAPTURES_FILE = os.path.join(DIRS['captures'], 'sweep_residuals.npz')
META_FILE = os.path.join(DIRS['captures'], 'sweep_meta.json')

if os.path.exists(CAPTURES_FILE) and os.path.exists(META_FILE):
    data = np.load(CAPTURES_FILE)
    sweep_arr = {k: list(data[k]) for k in data.files}
    with open(META_FILE) as f:
        meta = json.load(f)
    print(f'Resumed: {len(meta)} captured')
else:
    sweep_arr = {f'L{L}_{pos}': [] for L in PROBE_LAYERS_SWEEP for pos in POSITIONS}
    meta = []

done = {(m['trace_file'], m['corruption']) for m in meta}
t0 = time.time()
for i, p in enumerate(pairs):
    if (p['trace_file'], p['corruption']) in done:
        continue
    try:
        caps, positions, n_tok = capture_at_positions(p['messages_pre'], p['tool_result'])
        for L in PROBE_LAYERS_SWEEP:
            for pos_name in POSITIONS:
                if caps[L][pos_name] is not None:
                    sweep_arr[f'L{L}_{pos_name}'].append(caps[L][pos_name])
        meta.append({'trace_file': p['trace_file'], 'instance_id': p['instance_id'],
                     'corruption': p['corruption'], 'label': p['label'], 'n_tokens': n_tok,
                     'positions': positions, 'has_capture': True})
    except Exception as e:
        print(f'  [{i}] FAILED: {type(e).__name__}: {e}')
        meta.append({'trace_file': p['trace_file'], 'instance_id': p['instance_id'],
                     'corruption': p['corruption'], 'label': p['label'], 'has_capture': False,
                     'error': f'{type(e).__name__}: {e}'})

    if (i + 1) % 10 == 0 or (i + 1) == len(pairs):
        np.savez(CAPTURES_FILE, **{k: np.stack(v) for k, v in sweep_arr.items() if v})
        with open(META_FILE, 'w') as f:
            json.dump(meta, f, indent=2)
        elapsed = (time.time() - t0) / 60
        n_ok = sum(1 for m in meta if m['has_capture'])
        print(f'  [{i+1}/{len(pairs)}] {elapsed:.1f}min  ok={n_ok}')

np.savez(CAPTURES_FILE, **{k: np.stack(v) for k, v in sweep_arr.items() if v})
with open(META_FILE, 'w') as f:
    json.dump(meta, f, indent=2)
print(f'\\nDONE. {sum(1 for m in meta if m[\"has_capture\"])}/{len(meta)} valid, {(time.time()-t0)/60:.1f}min')
"""),

    code("""
# 7) Train probe per (L, pos) + D1 random baseline, 3 seeds
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

with open(META_FILE) as f:
    meta = json.load(f)
valid_meta = [m for m in meta if m['has_capture']]
data = np.load(CAPTURES_FILE)
y = np.array([m['label'] for m in valid_meta], dtype=int)
print(f'N valid: {len(valid_meta)}, balance: {100*y.mean():.0f}%')

SEEDS = [42, 7, 1337]
sweep_results = {}
for L in PROBE_LAYERS_SWEEP:
    for pos in POSITIONS:
        key = f'L{L}_{pos}'
        if key not in data.files:
            continue
        X = data[key]
        if X.shape[0] != len(y):
            continue
        real_seeds, rand_seeds = [], []
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            real_a, rand_a = [], []
            for tr, te in skf.split(X, y):
                clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
                clf.fit(X[tr], y[tr])
                try: real_a.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
                except ValueError: pass
                rng = np.random.default_rng(seed)
                Xr = rng.standard_normal(X.shape).astype(np.float32)
                clf_r = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
                clf_r.fit(Xr[tr], y[tr])
                try: rand_a.append(roc_auc_score(y[te], clf_r.predict_proba(Xr[te])[:, 1]))
                except ValueError: pass
            real_seeds.append(float(np.mean(real_a)) if real_a else float('nan'))
            rand_seeds.append(float(np.mean(rand_a)) if rand_a else float('nan'))
        sweep_results[(L, pos)] = {
            'real': float(np.mean(real_seeds)),
            'real_std': float(np.std(real_seeds)),
            'rand': float(np.mean(rand_seeds)),
            'gap': float(np.mean(real_seeds) - np.mean(rand_seeds)),
        }
print(f'\\nTrained {len(sweep_results)} (L, pos) probes.')
"""),

    code("""
# 8) Heatmap printout
print('\\n=== REAL AUROC ===')
print(f'{\"Layer\":<6} ' + ' '.join(f'{p[:18]:>18s}' for p in POSITIONS))
for L in PROBE_LAYERS_SWEEP:
    row = f'L{L:<5d} '
    for pos in POSITIONS:
        r = sweep_results.get((L, pos))
        row += f' {r[\"real\"]:>17.3f}' if r else '                  —'
    print(row)

print('\\n=== GAP (real − random) ===')
print(f'{\"Layer\":<6} ' + ' '.join(f'{p[:18]:>18s}' for p in POSITIONS))
for L in PROBE_LAYERS_SWEEP:
    row = f'L{L:<5d} '
    for pos in POSITIONS:
        r = sweep_results.get((L, pos))
        if r:
            gap = r['gap']
            flag = '🟢' if gap > 0.15 else ('🟡' if gap > 0.10 else ' ')
            row += f' {gap:>+15.3f}{flag} '
        else:
            row += '                  —'
    print(row)

if sweep_results:
    best_key = max(sweep_results, key=lambda k: sweep_results[k]['gap'])
    best = sweep_results[best_key]
    print(f'\\nBest (L, pos): L{best_key[0]} {best_key[1]}')
    print(f'  REAL {best[\"real\"]:.3f}±{best[\"real_std\"]:.3f}  GAP {best[\"gap\"]:+.3f}')
"""),

    code("""
# 9) Decision — resgata or walk-back
GAP_THRESHOLD = 0.15
passed = [(k, v) for k, v in sweep_results.items() if v['gap'] > GAP_THRESHOLD]
print(f'\\n=== {len(passed)} (L, pos) combos with gap > {GAP_THRESHOLD} ===')

if passed:
    print('\\n🟢 RESGATE — Phase 1 hypothesis survives at alternate (L, pos).')
    for k, v in sorted(passed, key=lambda x: -x[1]['gap'])[:5]:
        print(f'  L{k[0]} {k[1]}: REAL {v[\"real\"]:.3f}  gap {v[\"gap\"]:+.3f}')
    print('\\nNEXT: re-frame Phase 1 probe to this position. Re-run inspect-raw to confirm not lexical leak.')
else:
    print('\\n🔴 LOCK WALK-BACK — no (L, pos) combo recovers signal.')
    print('Phase 1 hypothesis FALSIFIED across full position × layer sweep.')
    print('\\nSubstantive finding: Qwen3.6-27B encodes EXPLICIT errors (ok=false strings)')
    print('but is BLIND to silent corruption (truncated tool results) at all tested')
    print('layers and positions in pre-generation residual.')
    print('\\nNEXT: walk-back memory + pivot to Inner-Outer divergence probe (Week 2).')

import os
RESULTS_FILE = os.path.join(DIRS['results'], 'sweep_aurocs.json')
with open(RESULTS_FILE, 'w') as f:
    json.dump({
        'per_combo': {f'L{k[0]}_{k[1]}': v for k, v in sweep_results.items()},
        'n_total': int(len(y)), 'n_positive': int(y.sum()),
        'seeds': SEEDS, 'gap_threshold': GAP_THRESHOLD,
        'n_passed': len(passed),
        'verdict': 'RESGATE' if passed else 'WALK_BACK',
        'transformers_commit': TRANSFORMERS_COMMIT,
    }, f, indent=2)
print(f'\\nSaved: {RESULTS_FILE}')
"""),
]


def build():
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NB_PATH}")
    print(f"  {len(cells)} cells")


if __name__ == "__main__":
    build()
