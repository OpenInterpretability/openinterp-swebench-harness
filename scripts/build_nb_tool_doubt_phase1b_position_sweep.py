"""Generate notebooks/nb_tool_doubt_phase1b_position_sweep.ipynb — diagnostic.

Phase 1 verdict 2026-05-18: best layer L11 AUROC 0.575, gap +0.065 → 2/4 gates,
substantively RED. Top-5 INSPECT-RAW showed ALL error_fake (lexical surface
features only); truncate ranked BELOW clean (probe blind to silent corruption).

Hypothesis to falsify before walk-back: position [-1] (start of next assistant
turn) was wrong; doubt might live in residual WHILE reading tool content
(end_of_tool_content) or AT tool message boundary (end_of_tool_msg).

This notebook re-captures the SAME 60 pairs from Phase 1 at 4 positions x 10
layers, trains probes per (L, pos), reports heatmap. If ANY combo gap > 0.15 →
resgata. Otherwise → lock walk-back.

Re-uses model in current Colab session (skip Cells 1-3 if already loaded).

Compute: ~5min Colab if model loaded, ~10min if fresh kernel.
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
# Tool-Doubt Phase 1b — Position Sweep Diagnostic

Phase 1 (2026-05-18) verdict: 2/4 gates, materially RED. Top-5 INSPECT-RAW
showed 100% error_fake → lexical leak via `"ok": false` / `"Error:"` strings.
Truncate ranked BELOW clean → probe blind to silent corruption.

This is the **last diagnostic before walk-back.** Tests whether position
choice (turn_end = start of next assistant) was wrong.

## Positions tested
- `start_of_tool_msg` — first token of the JSON tool message (right after `<|im_start|>tool\\n`)
- `mid_tool` — midpoint of the tool message
- `end_of_tool_msg` — `<|im_end|>` of the tool message
- `start_of_next_assistant` — current Phase 1 position (last token, `<|im_start|>assistant\\n`)

## Layers tested
10 layers: L7, L11, L15, L19, L23, L31, L35, L43, L47, L55

## Decision
- ANY (L, pos) combo with **gap > 0.15** → resgata Phase 1; reframe to that position
- ALL gaps < 0.15 → lock walk-back; pivot to Inner-Outer (Week 2)

## Reuses Phase 1 captures path + model in memory (if running in same Colab session)
"""),

    code("""
# 1) Verify model + pairs + paths already loaded (re-use Phase 1 session)
# If running fresh, must run nb_tool_doubt_phase1.ipynb Cells 1-5 first.
import os, json, torch
try:
    _ = model.config.hidden_size
    _ = tok.eos_token_id
    _ = LAYERS[0]
    _ = pairs[0]
    _ = DIRS['captures']
    _ = TOOLS
    _ = SYSTEM_PROMPT
    print(f'✅ Model + helpers reused. d={D_MODEL}, n_layers={N_LAYERS}, pairs={len(pairs)}')
except NameError as e:
    raise RuntimeError(f'Missing: {e}. Run nb_tool_doubt_phase1 Cells 1-5 first.')
"""),

    code("""
# 2) Helpers — position resolver + multi-position capture
import torch, numpy as np, json

PROBE_LAYERS_SWEEP = [7, 11, 15, 19, 23, 31, 35, 43, 47, 55]
POSITIONS = ['start_of_tool_msg', 'mid_tool', 'end_of_tool_msg', 'start_of_next_assistant']


def resolve_positions(messages_pre, tool_result_dict):
    '''
    Tokenize step-by-step to compute named token positions.
    Returns dict: position_name -> token index in full input_ids.
    '''
    messages_with_tool = list(messages_pre) + [
        {'role': 'tool', 'content': json.dumps(tool_result_dict, ensure_ascii=False)[:32000]},
    ]

    # Full prompt with generation prompt (matches what model sees)
    out_full = tok.apply_chat_template(
        messages_with_tool, tools=TOOLS, add_generation_prompt=True,
        return_tensors='pt', enable_thinking=True,
    )
    if hasattr(out_full, 'data') and isinstance(out_full.data, dict):
        ids_full = out_full['input_ids']
    elif torch.is_tensor(out_full):
        ids_full = out_full
    else:
        ids_full = torch.tensor([out_full], dtype=torch.long)
    full_len = int(ids_full.shape[-1])

    # Without generation prompt: ends at end of tool message
    out_no_gen = tok.apply_chat_template(
        messages_with_tool, tools=TOOLS, add_generation_prompt=False,
        return_tensors='pt', enable_thinking=True,
    )
    if hasattr(out_no_gen, 'data') and isinstance(out_no_gen.data, dict):
        no_gen_len = int(out_no_gen['input_ids'].shape[-1])
    elif torch.is_tensor(out_no_gen):
        no_gen_len = int(out_no_gen.shape[-1])
    else:
        no_gen_len = len(out_no_gen[0]) if isinstance(out_no_gen, list) else full_len

    # Without tool message: ends at end of assistant message (pre-tool context)
    out_no_tool = tok.apply_chat_template(
        list(messages_pre), tools=TOOLS, add_generation_prompt=False,
        return_tensors='pt', enable_thinking=True,
    )
    if hasattr(out_no_tool, 'data') and isinstance(out_no_tool.data, dict):
        no_tool_len = int(out_no_tool['input_ids'].shape[-1])
    elif torch.is_tensor(out_no_tool):
        no_tool_len = int(out_no_tool.shape[-1])
    else:
        no_tool_len = len(out_no_tool[0]) if isinstance(out_no_tool, list) else full_len

    return ids_full, {
        'start_of_tool_msg': no_tool_len,                        # first token of tool block
        'end_of_tool_msg': no_gen_len - 1,                       # last token before assistant prompt
        'mid_tool': (no_tool_len + no_gen_len - 1) // 2,         # midpoint of tool content
        'start_of_next_assistant': full_len - 1,                 # current Phase 1 position
    }


@torch.no_grad()
def capture_at_positions(messages_pre, tool_result, probe_layers=PROBE_LAYERS_SWEEP):
    '''Forward pass; hook captures residual at each target position per layer.'''
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


print(f'Helpers defined. Layers: {PROBE_LAYERS_SWEEP}, Positions: {POSITIONS}')
"""),

    code("""
# 3) Re-capture all 60 pairs across 10 layers × 4 positions
import os, json, time, numpy as np

SWEEP_DIR = os.path.join(DIRS['captures'].replace('captures', 'sweep_captures'))
os.makedirs(SWEEP_DIR, exist_ok=True)
CAPTURES_FILE = os.path.join(SWEEP_DIR, 'sweep_residuals.npz')
META_FILE = os.path.join(SWEEP_DIR, 'sweep_meta.json')

if os.path.exists(CAPTURES_FILE):
    data = np.load(CAPTURES_FILE)
    sweep_arr = {f'L{L}_{pos}': list(data[f'L{L}_{pos}']) for L in PROBE_LAYERS_SWEEP for pos in POSITIONS if f'L{L}_{pos}' in data}
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
        meta.append({
            'trace_file': p['trace_file'],
            'instance_id': p['instance_id'],
            'corruption': p['corruption'],
            'label': p['label'],
            'n_tokens': n_tok,
            'positions': positions,
            'has_capture': True,
        })
    except Exception as e:
        print(f'  [{i}] FAILED: {type(e).__name__}: {e}')
        meta.append({
            'trace_file': p['trace_file'],
            'instance_id': p['instance_id'],
            'corruption': p['corruption'],
            'label': p['label'],
            'has_capture': False,
            'error': f'{type(e).__name__}: {e}',
        })

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
# 4) Train probe per (L, pos) + D1 random baseline; 3 seeds
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

with open(META_FILE) as f:
    meta = json.load(f)
valid_meta = [m for m in meta if m['has_capture']]
data = np.load(CAPTURES_FILE)
y = np.array([m['label'] for m in valid_meta], dtype=int)
print(f'N valid: {len(valid_meta)}, balance: {100*y.mean():.0f}% positive')

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
# 5) Heatmap printout — AUROC and GAP per (L, pos)
print('\\n=== REAL AUROC ===')
print(f'{\"Layer\":<6} ' + ' '.join(f'{p[:14]:>14s}' for p in POSITIONS))
for L in PROBE_LAYERS_SWEEP:
    row = f'L{L:<5d} '
    for pos in POSITIONS:
        r = sweep_results.get((L, pos))
        row += f' {r[\"real\"]:>13.3f}' if r else '             —'
    print(row)

print('\\n=== GAP (real − random) ===')
print(f'{\"Layer\":<6} ' + ' '.join(f'{p[:14]:>14s}' for p in POSITIONS))
for L in PROBE_LAYERS_SWEEP:
    row = f'L{L:<5d} '
    for pos in POSITIONS:
        r = sweep_results.get((L, pos))
        if r:
            gap = r['gap']
            flag = '🟢' if gap > 0.15 else ('🟡' if gap > 0.10 else ' ')
            row += f' {gap:>+12.3f}{flag}'
        else:
            row += '             —'
    print(row)

# Best combo
if sweep_results:
    best_key = max(sweep_results, key=lambda k: sweep_results[k]['gap'])
    best = sweep_results[best_key]
    print(f'\\nBest (L, pos): L{best_key[0]} {best_key[1]}')
    print(f'  REAL {best[\"real\"]:.3f}±{best[\"real_std\"]:.3f}  GAP {best[\"gap\"]:+.3f}')
"""),

    code("""
# 6) Decision — resgata or walk-back
GAP_THRESHOLD = 0.15

passed = [(k, v) for k, v in sweep_results.items() if v['gap'] > GAP_THRESHOLD]
print(f'\\n=== {len(passed)} (L, pos) combos with gap > {GAP_THRESHOLD} ===')

if passed:
    print('\\n🟢 RESGATE — Phase 1 hypothesis survives at alternate (L, pos).')
    for k, v in sorted(passed, key=lambda x: -x[1]['gap'])[:5]:
        print(f'  L{k[0]} {k[1]}: REAL {v[\"real\"]:.3f}  gap {v[\"gap\"]:+.3f}')
    print('\\nNEXT: re-frame Phase 1 probe to this position. Re-run inspect-raw to confirm not lexical leak.')
    print('Save best probe + advance to Phase 2 (causality test).')
else:
    print('\\n🔴 LOCK WALK-BACK — no (L, pos) combo recovers signal.')
    print('Phase 1 hypothesis FALSIFIED across full position × layer sweep.')
    print('\\nFinding: Qwen3.6-27B encodes EXPLICIT errors (ok=false strings)')
    print('but is BLIND to silent corruption (truncated tool results) at all')
    print('tested layers and positions in pre-generation residual.')
    print('\\nNEXT: walk-back memory + pivot to Inner-Outer divergence probe (Week 2 sprint).')

# Save
import os
RESULTS_FILE = os.path.join(SWEEP_DIR, 'sweep_aurocs.json')
with open(RESULTS_FILE, 'w') as f:
    json.dump({
        'per_combo': {f'L{k[0]}_{k[1]}': v for k, v in sweep_results.items()},
        'n_total': int(len(y)),
        'n_positive': int(y.sum()),
        'seeds': SEEDS,
        'gap_threshold': GAP_THRESHOLD,
        'n_passed': len(passed),
        'verdict': 'RESGATE' if passed else 'WALK_BACK',
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
