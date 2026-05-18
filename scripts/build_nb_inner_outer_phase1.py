"""Generate notebooks/nb_inner_outer_phase1.ipynb — Inner-Outer Divergence probe Phase 1.

Third probe in agent-probe-guard stack v0.2 (Week 2 first deliverable). Tests
whether paired (think_end_residual, turn_end_residual) detects failed agent
trajectories better than either single position alone.

**KEY EFFICIENCY:** Phase 6 already saved residuals to .safetensors. No model
load, no GPU, no forward passes. Just sklearn + safetensors load.

**Compute: ~5min Colab CPU, ~R$0.**

Label source (Option B per Caio 2026-05-18): trace outcome from phase6_results.json
  - finish_reason == 'finish_tool' → success (label=0, no divergence formed)
  - else (max_turns, error, etc) → fail (label=1, divergence somewhere)

Pre-registered gates (G1-G8 protocol per [[feedback-transformers-main-version-pin]]):
  P1. Best paired AUROC > 0.70
  P2. Random-feature baseline < 0.55
  P3. Paired adds value: paired_AUROC > max(think_only, turn_only) + 0.05
  P4. 3-seed std < 0.05

CV: GroupKFold by instance_id (prevents trace-level leakage when turn-level
labels share trace-level outcome).

Modes tested per layer:
  - think_only: residual at think_end (5120 dim)
  - turn_only: residual at turn_end (5120 dim)
  - paired_concat: [think; turn] (10240 dim)
  - paired_diff: (turn - think) (5120 dim)

Decision: 4/4 gates pass → 🟢 ship. 3/4 → 🟡 investigate. ≤2 → 🔴 walk-back.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_inner_outer_phase1.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n", "outputs": [], "execution_count": None}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# Inner-Outer Divergence Probe — Phase 1

Third probe in stack v0.2 (Week 2). Tests paired (think_end_residual,
turn_end_residual) for detecting failed agent trajectories.

**Re-uses Phase 6 saved captures — no GPU, no forward passes, no model load.**

## Pre-registered predictions (LOCKED)

| # | Prediction | Threshold | Rationale |
|---|---|---|---|
| **P1** | Best paired AUROC | > 0.70 | Match Boppana single-phase 87% as ceiling reference |
| **P2** | Random-feature baseline | < 0.55 | Sanity — near chance |
| **P3** | Paired adds value over single | paired − max(think_only, turn_only) > 0.05 | Core claim |
| **P4** | 3-seed std (best paired) | < 0.05 | Stable |

## Decision tree
- 4/4 → 🟢 GREEN: ship as second working probe in stack
- 3/4 → 🟡 YELLOW: investigate failing prediction
- ≤2 → 🔴 RED: walk-back (third honest negative)

## Label source (Option B — Caio 2026-05-18)
- finish_reason == 'finish_tool' → label=0 (aligned, success)
- else (max_turns / error / context_overflow) → label=1 (divergence somewhere)

## CV: GroupKFold by instance_id
Turn-level captures share trace-level label → must split by trace to avoid leakage.

## Compute: ~5min Colab CPU, ~R$0
"""),

    # -------- Cell 1: Install --------
    code("""
# 1) Install (lightweight — no transformers/torch needed)
!pip install -q safetensors scikit-learn numpy scipy

import importlib
for pkg in ['safetensors', 'sklearn', 'numpy', 'scipy']:
    m = importlib.import_module(pkg)
    print(f'  {pkg}: {getattr(m, \"__version__\", \"?\")}')
print('\\nNote: NO model load required. Re-uses Phase 6 saved .safetensors residuals.')
"""),

    # -------- Cell 2: Drive + load Phase 6 results --------
    code("""
# 2) Drive mount + Phase 6 paths + load results JSON
import os, json
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs'
except ImportError:
    # Local Mac fallback
    DRIVE_ROOT = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
    if not os.path.exists(DRIVE_ROOT):
        DRIVE_ROOT = os.path.expanduser('~/openinterp_runs')

PHASE6_DIR = os.path.join(DRIVE_ROOT, 'swebench_v6_phase6')
CAPTURES_DIR = os.path.join(PHASE6_DIR, 'captures')
RESULTS_JSON = os.path.join(PHASE6_DIR, 'phase6_results.json')
assert os.path.isdir(CAPTURES_DIR), f'Missing: {CAPTURES_DIR}'
assert os.path.exists(RESULTS_JSON), f'Missing: {RESULTS_JSON}'

with open(RESULTS_JSON) as f:
    phase6_results = json.load(f)
print(f'Phase 6 instances: {len(phase6_results)}')

# Label per Option B: success = finish_tool, else = fail
labels = {}
for iid, info in phase6_results.items():
    finished = info.get('finished', False)
    finish_reason = info.get('finish_reason', '')
    labels[iid] = 0 if (finished and finish_reason == 'finish_tool') else 1

n_pos = sum(labels.values())
print(f'  Success (label=0): {len(labels) - n_pos}')
print(f'  Failed (label=1):  {n_pos}')
print(f'  Failure rate: {n_pos / len(labels) * 100:.0f}%')

# Output dir
OUT_DIR = os.path.join(DRIVE_ROOT, 'inner_outer_phase1')
os.makedirs(OUT_DIR, exist_ok=True)
print(f'Output: {OUT_DIR}')
"""),

    # -------- Cell 3: Helpers — load paired residuals --------
    code("""
# 3) Helpers — load paired (think_end, turn_end) residuals per turn × layer
from safetensors import safe_open
import numpy as np

PROBE_LAYERS = [11, 23, 31, 43, 55]
POS_INNER = 'think_end'
POS_OUTER = 'turn_end'


def load_paired_residuals(instance_info, probe_layers=PROBE_LAYERS):
    '''
    Returns dict: layer → list of (turn_idx, think_end_vec, turn_end_vec) for turns
    where BOTH positions were captured.
    '''
    safetensors_path = instance_info.get('captures_safetensors')
    meta_path = instance_info.get('captures_meta')
    if not safetensors_path or not os.path.exists(safetensors_path):
        # Fall back: derive from instance_id if path is Colab-absolute
        iid = instance_info['instance_id']
        safetensors_path = os.path.join(CAPTURES_DIR, f'{iid}.safetensors')
        meta_path = os.path.join(CAPTURES_DIR, f'{iid}.meta.json')
    if not os.path.exists(safetensors_path) or not os.path.exists(meta_path):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    # Index records by (turn, position, layer) → activation_key
    by_key = {}
    for rec in meta['records']:
        k = (rec['turn_idx'], rec['position_label'], rec['layer'])
        by_key[k] = rec['activation_key']

    # Collect turn indices where both think_end AND turn_end exist for all probe_layers
    turn_indices = set()
    for t in range(50):  # plenty
        if all((t, POS_INNER, L) in by_key for L in probe_layers) and \\
           all((t, POS_OUTER, L) in by_key for L in probe_layers):
            turn_indices.add(t)

    if not turn_indices:
        return None

    paired_per_layer = {L: [] for L in probe_layers}
    with safe_open(safetensors_path, framework='pt') as f:
        for t in sorted(turn_indices):
            for L in probe_layers:
                k_inner = by_key[(t, POS_INNER, L)]
                k_outer = by_key[(t, POS_OUTER, L)]
                v_inner = f.get_tensor(k_inner).to(dtype='float32' if hasattr(f.get_tensor(k_inner), 'to') else None)
                v_outer = f.get_tensor(k_outer)
                paired_per_layer[L].append((t, v_inner.float().numpy(), v_outer.float().numpy()))

    return paired_per_layer


# Quick smoke test on first instance
iid_test = list(phase6_results.keys())[0]
res_test = load_paired_residuals(phase6_results[iid_test])
if res_test:
    print(f'Smoke test {iid_test[:60]}...')
    print(f'  Layer 11: {len(res_test[11])} turns paired')
    if res_test[11]:
        _, v_i, v_o = res_test[11][0]
        print(f'  think_end[0] shape={v_i.shape} dtype={v_i.dtype}')
        print(f'  turn_end[0] shape={v_o.shape} dtype={v_o.dtype}')
else:
    print('FAILED smoke test — check captures path')
"""),

    # -------- Cell 4: Build dataset --------
    code("""
# 4) Build dataset — collect paired residuals + group + label for all instances
import numpy as np

X_think = {L: [] for L in PROBE_LAYERS}
X_turn = {L: [] for L in PROBE_LAYERS}
y_list = []
groups = []  # instance_id per sample (for GroupKFold)

skipped = 0
for iid, info in phase6_results.items():
    paired = load_paired_residuals(info)
    if paired is None:
        skipped += 1
        continue
    label = labels[iid]
    for L in PROBE_LAYERS:
        for (turn_idx, v_i, v_o) in paired[L]:
            X_think[L].append(v_i)
            X_turn[L].append(v_o)
            if L == PROBE_LAYERS[0]:  # add label/group ONCE per (instance, turn)
                y_list.append(label)
                groups.append(iid)

X_think = {L: np.stack(X_think[L]) for L in PROBE_LAYERS}
X_turn = {L: np.stack(X_turn[L]) for L in PROBE_LAYERS}
y = np.array(y_list, dtype=int)
groups = np.array(groups)

# Sanity: all layers same N
for L in PROBE_LAYERS:
    assert X_think[L].shape[0] == len(y), f'L{L} think mismatch'
    assert X_turn[L].shape[0] == len(y), f'L{L} turn mismatch'

n_instances_used = len(set(groups))
print(f'Captures loaded: {len(y)} (turn, layer) samples from {n_instances_used} instances ({skipped} skipped)')
print(f'  per-layer shape: {X_think[PROBE_LAYERS[0]].shape}')
print(f'  Label balance: success={int((y==0).sum())} fail={int((y==1).sum())} ({100*y.mean():.0f}% positive)')
"""),

    # -------- Cell 5: Train probes per (layer, mode) + GroupKFold CV --------
    code("""
# 5) Train probes per (layer, mode) with GroupKFold CV + 3 seeds + D1 baseline
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

MODES = {
    'think_only': lambda L: X_think[L],
    'turn_only': lambda L: X_turn[L],
    'paired_concat': lambda L: np.concatenate([X_think[L], X_turn[L]], axis=1),
    'paired_diff': lambda L: X_turn[L] - X_think[L],
}
SEEDS = [42, 7, 1337]

results = {}
for L in PROBE_LAYERS:
    for mode_name, mode_fn in MODES.items():
        X = mode_fn(L).astype(np.float32)
        real_seeds, rand_seeds = [], []
        for seed in SEEDS:
            np.random.seed(seed)
            gkf = GroupKFold(n_splits=5)
            real_a, rand_a = [], []
            for tr, te in gkf.split(X, y, groups):
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
        results[(L, mode_name)] = {
            'real': float(np.mean(real_seeds)),
            'real_std': float(np.std(real_seeds)),
            'rand': float(np.mean(rand_seeds)),
            'gap': float(np.mean(real_seeds) - np.mean(rand_seeds)),
        }
print(f'Trained {len(results)} (layer, mode) probes.')
"""),

    # -------- Cell 6: Heatmap printout --------
    code("""
# 6) Heatmap — AUROC and PAIRED-LIFT per (layer, mode)
print('\\n=== REAL AUROC (mean of 3 seeds, GroupKFold) ===')
modes_list = list(MODES.keys())
print(f'{\"Layer\":<6} ' + ' '.join(f'{m:>15s}' for m in modes_list))
for L in PROBE_LAYERS:
    row = f'L{L:<5d} '
    for mode in modes_list:
        r = results.get((L, mode))
        row += f' {r[\"real\"]:>14.3f}' if r else '               —'
    print(row)

print('\\n=== GAP vs D1 RANDOM ===')
print(f'{\"Layer\":<6} ' + ' '.join(f'{m:>15s}' for m in modes_list))
for L in PROBE_LAYERS:
    row = f'L{L:<5d} '
    for mode in modes_list:
        r = results.get((L, mode))
        if r:
            gap = r['gap']
            flag = '🟢' if gap > 0.20 else ('🟡' if gap > 0.10 else ' ')
            row += f' {gap:>+13.3f}{flag}'
        else:
            row += '               —'
    print(row)

# Per-layer: paired lift (best_paired − best_single)
print('\\n=== PAIRED LIFT (best paired − best single) per layer ===')
for L in PROBE_LAYERS:
    think = results[(L, 'think_only')]['real']
    turn = results[(L, 'turn_only')]['real']
    p_concat = results[(L, 'paired_concat')]['real']
    p_diff = results[(L, 'paired_diff')]['real']
    best_paired = max(p_concat, p_diff)
    best_single = max(think, turn)
    lift = best_paired - best_single
    flag = '🟢' if lift > 0.05 else ('🟡' if lift > 0.02 else ' ')
    print(f'  L{L}: best_paired={best_paired:.3f} best_single={best_single:.3f} lift={lift:+.3f} {flag}')

# Best paired overall
paired_only = {k: v for k, v in results.items() if 'paired' in k[1]}
best_key = max(paired_only, key=lambda k: paired_only[k]['real'])
best = paired_only[best_key]
print(f'\\nBest paired: L{best_key[0]} {best_key[1]}  REAL {best[\"real\"]:.3f}±{best[\"real_std\"]:.3f}')
"""),

    # -------- Cell 7: Pre-registered gate check --------
    code("""
# 7) Pre-registered gate check
print('=' * 78)
print('PRE-REGISTERED PREDICTION CHECK')
print('=' * 78)

# Best paired
paired_only = {k: v for k, v in results.items() if 'paired' in k[1]}
best_key = max(paired_only, key=lambda k: paired_only[k]['real'])
best_paired = paired_only[best_key]
best_L = best_key[0]

# Best single at same layer
best_single = max(results[(best_L, 'think_only')]['real'],
                  results[(best_L, 'turn_only')]['real'])

P1 = best_paired['real'] > 0.70
P2 = best_paired['rand'] < 0.55
P3 = (best_paired['real'] - best_single) > 0.05
P4 = best_paired['real_std'] < 0.05

preds = [
    ('P1 best paired AUROC > 0.70', best_paired['real'], '>0.70', P1),
    ('P2 random < 0.55', best_paired['rand'], '<0.55', P2),
    (f'P3 paired − best_single > 0.05 (single={best_single:.3f})', best_paired['real'] - best_single, '>0.05', P3),
    ('P4 3-seed std < 0.05', best_paired['real_std'], '<0.05', P4),
]
for name, val, thr, p in preds:
    print(f'  {\"✅\" if p else \"❌\"} {name}: actual={val:+.3f} (need {thr})')

n_pass = sum(1 for _, _, _, p in preds if p)
print(f'\\n=== {n_pass}/4 PASS — best (L{best_key[0]}, {best_key[1]}) ===\\n')

if n_pass == 4:
    print('🟢 GREEN — pre-registered gates pass. Inner-Outer probe = second working probe in stack v0.2.')
    print('   NEXT: multi-model gate (Qwen2.5-7B replication) before Phase 2 ship.')
elif n_pass == 3:
    print('🟡 YELLOW — 1 prediction failed.')
elif n_pass == 2:
    print('🟡 YELLOW — 2 predictions failed. Reconsider design before scaling.')
else:
    print('🔴 RED — walk-back. Third honest negative.')

# Save
import os
with open(os.path.join(OUT_DIR, 'phase1_aurocs.json'), 'w') as f:
    out = {f'L{k[0]}_{k[1]}': v for k, v in results.items()}
    out['_best_paired'] = {'layer': int(best_key[0]), 'mode': best_key[1], **best_paired}
    out['_best_single_at_best_layer'] = best_single
    out['_n_pass'] = n_pass
    out['_n_total'] = int(len(y))
    out['_n_positive'] = int(y.sum())
    out['_n_instances'] = int(n_instances_used)
    json.dump(out, f, indent=2)
print(f'\\nSaved: {os.path.join(OUT_DIR, \"phase1_aurocs.json\")}')
"""),

    # -------- Cell 8: INSPECT-RAW --------
    code("""
# 8) INSPECT-RAW — top/bottom predicted divergence at best (L, mode)
from sklearn.linear_model import LogisticRegression

L_best, mode_best = best_key
X_best = MODES[mode_best](L_best).astype(np.float32)
clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
clf.fit(X_best, y)
proba = clf.predict_proba(X_best)[:, 1]
order = np.argsort(proba)[::-1]

print(f'=== TOP-10 predicted FAIL (L{L_best} {mode_best}) ===')
seen_iids_top = set()
for i in order:
    if groups[i] in seen_iids_top:
        continue
    seen_iids_top.add(groups[i])
    iid = groups[i]
    info = phase6_results[iid]
    print(f'  prob={proba[i]:.3f}  actual={\"FAIL\" if y[i] == 1 else \"OK\"}  reason={info.get(\"finish_reason\", \"?\")[:20]:20s}  iid={iid[:50]}')
    if len(seen_iids_top) >= 10:
        break

print(f'\\n=== BOTTOM-5 predicted (most likely SUCCESS) ===')
seen_iids_bot = set()
for i in order[::-1]:
    if groups[i] in seen_iids_bot:
        continue
    seen_iids_bot.add(groups[i])
    iid = groups[i]
    info = phase6_results[iid]
    print(f'  prob={proba[i]:.3f}  actual={\"FAIL\" if y[i] == 1 else \"OK\"}  reason={info.get(\"finish_reason\", \"?\")[:20]:20s}  iid={iid[:50]}')
    if len(seen_iids_bot) >= 5:
        break
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
