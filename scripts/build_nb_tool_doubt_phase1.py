"""Generate notebooks/nb_tool_doubt_phase1.ipynb — Tool-Result-Doubt probe Phase 1.

Detects whether agent's residual encodes "this tool result is wrong" — captured at
the position where model would start generating turn 1 after seeing turn 0's tool
result (real vs corrupted).

Re-uses SWE-bench Phase 6 traces (~99 unique instances at
/content/drive/MyDrive/openinterp_runs/swebench_v6_phase6/traces/) +
SWE-bench Pro dataset (`ScaleAI/SWE-bench_Pro`) for problem statements.

Single-turn pilot design (Phase 1 pilot is about whether ANY signal exists):
  - Replay turn 0 only: system + user_task + assistant_with_tool_call + tool_result
  - For positives: corrupt the tool_result
  - Capture residual at turn_end position (start of turn 1)
  - 2 corruption strategies: truncate (first 50 chars), error_fake (replace with error)

**GATES EMBEDDED (from feedback memo 2026-05-18 silent-refusal walk-back):**
  G1. Pre-registered predictions in Cell 0 (locked BEFORE measurement)
  G2. transformers commit pinned in Cell 1 (logs hash)
  G3. attention_mask passed explicitly (avoids pad==eos warning)
  G4. Random-feature baseline (paper-MEGA D1)
  G5. Shuffled-source baseline (paper-MEGA D2)
  G6. Multi-seed (3 seeds; std<0.05 = stable)
  G7. Capacity sweep K∈{4,8,16,32,64} (paper-MEGA D3)
  G8. Multi-model gate scheduled if pilot passes (Qwen2.5-7B)

Decision tree (Cell 9): 4/4 → GREEN. 3/4 → YELLOW. ≤2 → RED walk-back.

Target: ~30-40 min Colab RTX 6000, ~R$3-5.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_tool_doubt_phase1.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n", "outputs": [], "execution_count": None}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# Tool-Result-Doubt Probe — Phase 1 (with all 8 gates)

Single-turn pilot. Replay turn 0 of Phase 6 traces; inject corrupted tool result;
capture residual where model would start turn 1. Tests whether residual encodes
"doubt" about tool result quality.

## Pre-registered predictions (GATE G1 — LOCKED BEFORE MEASUREMENT)

| # | Prediction | Threshold | Rationale |
|---|---|---|---|
| **P1** | Best layer AUROC | > 0.70 at N=60 pilot | Matches Phase 6 capability strength |
| **P2** | Random-feature baseline AUROC | < 0.55 | Should be near-chance |
| **P3** | Gap (real − random) | > 0.15 | Below = within N=60 noise |
| **P4** | 3-seed std of best layer | < 0.05 | Stable across seeds |

## Decision tree
- 4/4 → 🟢 GREEN: schedule multi-model gate (Qwen2.5-7B) before Phase 2
- 3/4 → 🟡 YELLOW: investigate failing prediction before scaling
- ≤2 → 🔴 RED: walk-back, document negative (like silent-refusal)

## Setup
- N=60 (20 traces × 3 conditions: clean + truncate + error_fake)
- Forward-only capture at last position after tool message
- 5 layers × 3 seeds = 15 fits
- Compute: ~30-40 min, ~R$3-5
"""),

    # -------- Cell 1: Install + PIN transformers commit (GATE G2) --------
    code("""
# 1) Install + log transformers commit hash (GATE G2)
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q accelerate scipy safetensors huggingface_hub datasets scikit-learn
!pip install -q flash-linear-attention causal-conv1d --no-build-isolation 2>&1 | tail -3 || true

import importlib, subprocess, os
TRANSFORMERS_COMMIT = None
for pkg in ['transformers', 'sklearn', 'scipy', 'datasets']:
    try:
        m = importlib.import_module(pkg)
        print(f'  {pkg}: {getattr(m, \"__version__\", \"?\")}')
    except ImportError:
        print(f'  {pkg}: MISSING')

try:
    import transformers
    tfm_path = os.path.dirname(transformers.__file__)
    TRANSFORMERS_COMMIT = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=tfm_path,
                                          capture_output=True, text=True).stdout.strip()
    print(f'\\n🔒 transformers commit: {TRANSFORMERS_COMMIT}')
    print(f'   Pin via: !pip install -q git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_COMMIT}')
except Exception as e:
    print(f'  Could not get commit: {e}')
print('\\nRESTART RUNTIME if transformers upgraded.')
"""),

    # -------- Cell 2: GPU + Drive + load Phase 6 traces + SWE-bench Pro dataset --------
    code("""
# 2) GPU + Drive + traces + dataset
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

# Confirmed path from local Drive ls (2026-05-18)
PHASE6_TRACES_DIR = os.path.join(DRIVE_ROOT, 'swebench_v6_phase6', 'traces')
assert os.path.isdir(PHASE6_TRACES_DIR), f'Phase 6 traces dir not found: {PHASE6_TRACES_DIR}'

trace_files = sorted(glob.glob(os.path.join(PHASE6_TRACES_DIR, 'instance_*.json')))
print(f'Phase 6 trace files: {len(trace_files)}')

# Load SWE-bench Pro dataset for problem_statement lookup
from datasets import load_dataset
ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
ds_by_id = {row['instance_id']: row for row in ds}
print(f'SWE-bench Pro instances: {len(ds_by_id)}')

# Output dirs
OUT_DIR = os.path.join(DRIVE_ROOT, 'tool_doubt_phase1')
DIRS = {k: os.path.join(OUT_DIR, k) for k in ['captures', 'probes', 'results']}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)
print(f'Output: {OUT_DIR}')
"""),

    # -------- Cell 3: Load Qwen3.6-27B + import harness helpers --------
    code("""
# 3) Load Qwen3.6-27B + clone harness for SYSTEM_PROMPT + body extractor
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

# Clone harness repo for SYSTEM_PROMPT, render_problem, _strip_think, TOOLS
if not os.path.exists('/content/openinterp-swebench-harness'):
    !git clone -q https://github.com/OpenInterpretability/openinterp-swebench-harness.git /content/openinterp-swebench-harness
import sys
sys.path.insert(0, '/content/openinterp-swebench-harness')
from agent.prompts import SYSTEM_PROMPT, render_problem
from agent.parser import _strip_think
from agent.tools import TOOLS
print(f'Harness loaded. SYSTEM_PROMPT len={len(SYSTEM_PROMPT)}, tools={[t[\"function\"][\"name\"] for t in TOOLS]}')
"""),

    # -------- Cell 4: Helpers — single-turn replay + forward-only capture --------
    code("""
# 4) Helpers — extract turn-0 replay context + corruption + forward capture
import torch, numpy as np, json

PROBE_LAYERS = [11, 23, 31, 43, 55]


def corrupt_truncate(real_result_dict, n_chars=50):
    '''Truncate the content field of tool result.'''
    r = json.loads(json.dumps(real_result_dict))  # deep copy
    if isinstance(r.get('result'), dict) and 'content' in r['result']:
        c = r['result']['content']
        if isinstance(c, str):
            r['result']['content'] = c[:n_chars] + '...[truncated]'
    return r


def corrupt_error_fake(real_result_dict):
    '''Replace tool result with fake error.'''
    return {
        'id': real_result_dict['id'],
        'result': {'ok': False, 'error': 'File not found or permission denied.'},
    }


CORRUPTION_STRATEGIES = {
    'truncate': corrupt_truncate,
    'error_fake': corrupt_error_fake,
}


def extract_turn0_replay(trace_path, ds_by_id):
    '''
    Build single-turn replay messages: system + user_task + assistant + tool_result(real).
    Returns (messages_through_assistant, real_tool_result_dict, raw_response_body).
    Tool result NOT yet appended — caller appends real or corrupted version.
    '''
    with open(trace_path) as f:
        trace = json.load(f)
    instance_id = trace['instance_id']
    if instance_id not in ds_by_id:
        return None
    instance = dict(ds_by_id[instance_id])
    instance['__workdir__'] = f'/content/work_p6/{instance_id}'  # match Phase 6 convention
    user_msg = render_problem(instance)

    turn0 = trace['turns'][0]
    if not turn0.get('tool_calls') or not turn0.get('tool_results'):
        return None  # need at least 1 tool call+result in turn 0

    # Reconstruct assistant body (strip thinking, preserve tool_call blocks — matches loop.py:227)
    _, body_with_tools = _strip_think(turn0['raw_response'])

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_msg},
        {'role': 'assistant', 'content': body_with_tools},
    ]
    # Phase 6 only had 1 tool call per turn in most cases; use first
    real_tool_result = turn0['tool_results'][0]
    return messages, real_tool_result, instance_id


@torch.no_grad()
def capture_at_turn_end(messages_through_assistant, tool_result_dict, probe_layers=PROBE_LAYERS):
    '''
    Append tool message (json-serialized result), forward-pass, hook residual at LAST token.
    Uses tools=TOOLS in chat template (matches Phase 6 harness exactly).
    Uses explicit attention_mask (GATE G3).
    '''
    messages = list(messages_through_assistant) + [
        {'role': 'tool', 'content': json.dumps(tool_result_dict, ensure_ascii=False)[:32000]},
    ]

    # Match harness build_input_ids exactly: tools=TOOLS, enable_thinking=True, add_gen_prompt=True
    out = tok.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=True,
        return_tensors='pt', enable_thinking=True,
    )
    if hasattr(out, 'data') and isinstance(out.data, dict) and 'input_ids' in out.data:
        ids = out['input_ids']
    elif torch.is_tensor(out):
        ids = out
    else:
        ids = torch.tensor([out], dtype=torch.long)
    ids = ids.to(model.device)
    attn = torch.ones_like(ids)

    captures = {L: None for L in probe_layers}
    def make_hook(L):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captures[L] = h[0, -1, :].detach().float().cpu().numpy()
        return hook

    handles = [LAYERS[L].register_forward_hook(make_hook(L)) for L in probe_layers]
    try:
        model(ids, attention_mask=attn, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return captures, int(ids.shape[1])


print(f'Helpers defined. PROBE_LAYERS={PROBE_LAYERS}')
print(f'Corruption strategies: {list(CORRUPTION_STRATEGIES.keys())}')
"""),

    # -------- Cell 5: Build pairs --------
    code("""
# 5) Build N=20 traces × 3 conditions = 60 pairs
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

    pairs.append({
        'trace_file': tf,
        'instance_id': instance_id,
        'messages_pre': messages_pre,
        'tool_result': real_tool,
        'label': 0,
        'corruption': 'none',
    })
    for strat_name, strat_fn in CORRUPTION_STRATEGIES.items():
        pairs.append({
            'trace_file': tf,
            'instance_id': instance_id,
            'messages_pre': messages_pre,
            'tool_result': strat_fn(real_tool),
            'label': 1,
            'corruption': strat_name,
        })

print(f'Checked {checked} traces, built {len(pairs)} pairs from {len(pairs)//3} usable traces')
print(f'  Clean: {sum(1 for p in pairs if p[\"label\"]==0)}')
print(f'  Corrupt truncate: {sum(1 for p in pairs if p[\"corruption\"]==\"truncate\")}')
print(f'  Corrupt error_fake: {sum(1 for p in pairs if p[\"corruption\"]==\"error_fake\")}')

# Sample inspect
print(f'\\nFirst pair tool_result (clean):')
print(json.dumps(pairs[0]['tool_result'], indent=2)[:400])
print(f'\\nSecond pair tool_result (truncate):')
print(json.dumps(pairs[1]['tool_result'], indent=2)[:400])
print(f'\\nThird pair tool_result (error_fake):')
print(json.dumps(pairs[2]['tool_result'], indent=2)[:400])
"""),

    # -------- Cell 6: Capture residuals --------
    code("""
# 6) Capture residual at turn_end for all 60 pairs
import os, json, time, numpy as np

CAPTURES_FILE = os.path.join(DIRS['captures'], 'turn_end_residuals.npz')
META_FILE = os.path.join(DIRS['captures'], 'capture_meta.json')

if os.path.exists(CAPTURES_FILE) and os.path.exists(META_FILE):
    data = np.load(CAPTURES_FILE)
    captures_arr = {L: list(data[f'L{L}']) for L in PROBE_LAYERS}
    with open(META_FILE) as f:
        meta = json.load(f)
    print(f'Resumed: {len(meta)} captured')
else:
    captures_arr = {L: [] for L in PROBE_LAYERS}
    meta = []

done = {(m['trace_file'], m['corruption']) for m in meta}
t0 = time.time()
for i, p in enumerate(pairs):
    if (p['trace_file'], p['corruption']) in done:
        continue
    try:
        caps, n_tok = capture_at_turn_end(p['messages_pre'], p['tool_result'])
        for L in PROBE_LAYERS:
            captures_arr[L].append(caps[L])
        meta.append({
            'trace_file': p['trace_file'],
            'instance_id': p['instance_id'],
            'corruption': p['corruption'],
            'label': p['label'],
            'n_tokens': n_tok,
            'has_capture': True,
        })
    except Exception as e:
        print(f'  [{i}] FAILED idx={p[\"instance_id\"][:30]}: {type(e).__name__}: {e}')
        meta.append({
            'trace_file': p['trace_file'],
            'instance_id': p['instance_id'],
            'corruption': p['corruption'],
            'label': p['label'],
            'has_capture': False,
            'error': f'{type(e).__name__}: {e}',
        })

    if (i + 1) % 5 == 0 or (i + 1) == len(pairs):
        np.savez(CAPTURES_FILE, **{f'L{L}': np.stack(captures_arr[L]) for L in PROBE_LAYERS if captures_arr[L]})
        with open(META_FILE, 'w') as f:
            json.dump(meta, f, indent=2)
        elapsed = (time.time() - t0) / 60
        n_ok = sum(1 for m in meta if m['has_capture'])
        print(f'  [{i+1}/{len(pairs)}] {elapsed:.1f}min  ok={n_ok}  failed={len(meta)-n_ok}')

# Final
np.savez(CAPTURES_FILE, **{f'L{L}': np.stack(captures_arr[L]) for L in PROBE_LAYERS if captures_arr[L]})
with open(META_FILE, 'w') as f:
    json.dump(meta, f, indent=2)
n_ok = sum(1 for m in meta if m['has_capture'])
print(f'\\nDONE. {n_ok}/{len(meta)} valid, {(time.time()-t0)/60:.1f}min')
"""),

    # -------- Cell 7: Train + multi-seed + D1 + D2 --------
    code("""
# 7) Probe per layer + 3-seed CV + D1 + D2 baselines (GATES G4, G5, G6)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

with open(META_FILE) as f:
    meta = json.load(f)
valid_meta = [m for m in meta if m['has_capture']]
data = np.load(CAPTURES_FILE)
X_per_layer = {L: data[f'L{L}'] for L in PROBE_LAYERS}
y = np.array([m['label'] for m in valid_meta], dtype=int)
print(f'N valid: {len(valid_meta)}, balance: {100*y.mean():.0f}% positive')

SEEDS = [42, 7, 1337]
results = {}
for L in PROBE_LAYERS:
    X = X_per_layer[L]
    real_per_seed, rand_per_seed, shuf_per_seed = [], [], []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        real_a, rand_a, shuf_a = [], [], []
        for tr, te in skf.split(X, y):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
            clf.fit(Xtr, ytr)
            try: real_a.append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
            except ValueError: pass

            rng = np.random.default_rng(seed)
            Xtr_r = rng.standard_normal(Xtr.shape).astype(np.float32)
            Xte_r = rng.standard_normal(Xte.shape).astype(np.float32)
            clf_r = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
            clf_r.fit(Xtr_r, ytr)
            try: rand_a.append(roc_auc_score(yte, clf_r.predict_proba(Xte_r)[:, 1]))
            except ValueError: pass

            rng2 = np.random.default_rng(seed + 100)
            Xtr_s = Xtr[rng2.permutation(Xtr.shape[0])]
            clf_s = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
            clf_s.fit(Xtr_s, ytr)
            try: shuf_a.append(roc_auc_score(yte, clf_s.predict_proba(Xte)[:, 1]))
            except ValueError: pass

        real_per_seed.append(float(np.mean(real_a)) if real_a else float('nan'))
        rand_per_seed.append(float(np.mean(rand_a)) if rand_a else float('nan'))
        shuf_per_seed.append(float(np.mean(shuf_a)) if shuf_a else float('nan'))

    results[L] = {
        'real_per_seed': real_per_seed,
        'real_mean': float(np.mean(real_per_seed)),
        'real_std': float(np.std(real_per_seed)),
        'rand_mean': float(np.mean(rand_per_seed)),
        'shuf_mean': float(np.mean(shuf_per_seed)),
        'gap': float(np.mean(real_per_seed) - np.mean(rand_per_seed)),
    }
    r = results[L]
    print(f'L{L:2d}: REAL {r[\"real_mean\"]:.3f}±{r[\"real_std\"]:.3f}  D1-rand {r[\"rand_mean\"]:.3f}  D2-shuf {r[\"shuf_mean\"]:.3f}  gap {r[\"gap\"]:+.3f}')

best_L = max(results, key=lambda L: results[L]['real_mean'])
best = results[best_L]
print(f'\\nBest layer: L{best_L} AUROC={best[\"real_mean\"]:.3f}±{best[\"real_std\"]:.3f} gap={best[\"gap\"]:+.3f}')

with open(os.path.join(DIRS['results'], 'phase1_aurocs.json'), 'w') as f:
    json.dump({'per_layer': results, 'best_L': int(best_L), 'n_total': int(len(y)),
               'n_positive': int(y.sum()), 'seeds': SEEDS, 'transformers_commit': TRANSFORMERS_COMMIT}, f, indent=2)
"""),

    # -------- Cell 8: Capacity sweep --------
    code("""
# 8) Capacity sweep PCA-K on best layer (GATE G7)
from sklearn.decomposition import PCA

X_best = X_per_layer[best_L]
K_VALUES = [4, 8, 16, 32, 64]
print(f'Capacity sweep on L{best_L} (N={len(y)})')
sweep = {}
for K in K_VALUES:
    if K > min(X_best.shape):
        continue
    pca = PCA(n_components=K, random_state=42)
    X_pca = pca.fit_transform(X_best)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    real_a, rand_a = [], []
    rng = np.random.default_rng(42)
    X_rand_K = rng.standard_normal((X_best.shape[0], K)).astype(np.float32)
    for tr, te in skf.split(X_pca, y):
        clf = LogisticRegression(class_weight='balanced', max_iter=2000)
        clf.fit(X_pca[tr], y[tr])
        try: real_a.append(roc_auc_score(y[te], clf.predict_proba(X_pca[te])[:, 1]))
        except ValueError: pass
        clf_r = LogisticRegression(class_weight='balanced', max_iter=2000)
        clf_r.fit(X_rand_K[tr], y[tr])
        try: rand_a.append(roc_auc_score(y[te], clf_r.predict_proba(X_rand_K[te])[:, 1]))
        except ValueError: pass
    real_m = float(np.mean(real_a)) if real_a else float('nan')
    rand_m = float(np.mean(rand_a)) if rand_a else float('nan')
    gap = real_m - rand_m
    flag = '🟢' if gap > 0.15 else ('🟡' if gap > 0.05 else '🔴')
    sweep[K] = {'real': real_m, 'rand': rand_m, 'gap': gap}
    print(f'  K={K:3d}: REAL {real_m:.3f}  RAND {rand_m:.3f}  gap {gap:+.3f}  {flag}')

with open(os.path.join(DIRS['results'], 'phase1_capacity_sweep.json'), 'w') as f:
    json.dump({'best_L': int(best_L), 'sweep': sweep}, f, indent=2)
"""),

    # -------- Cell 9: Pre-registered prediction check + decision --------
    code("""
# 9) GATE check — evaluate pre-registered predictions
print('=' * 78)
print('PRE-REGISTERED PREDICTION CHECK (GATE G1)')
print('=' * 78)

P1_pass = best['real_mean'] > 0.70
P2_pass = best['rand_mean'] < 0.55
P3_pass = best['gap'] > 0.15
P4_pass = best['real_std'] < 0.05

preds = [
    ('P1 best AUROC > 0.70', best['real_mean'], '>0.70', P1_pass),
    ('P2 random < 0.55', best['rand_mean'], '<0.55', P2_pass),
    ('P3 gap > 0.15', best['gap'], '>0.15', P3_pass),
    ('P4 3-seed std < 0.05', best['real_std'], '<0.05', P4_pass),
]
for name, val, thr, p in preds:
    print(f'  {\"✅\" if p else \"❌\"} {name}: actual={val:.3f} (need {thr})')

n_pass = sum(1 for _, _, _, p in preds if p)
print(f'\\n=== {n_pass}/4 PASS ===\\n')

if n_pass == 4:
    print('🟢 GREEN — all pre-registered predictions hold.')
    print('   NEXT: schedule multi-model gate (GATE G8) — Qwen2.5-7B replication.')
    print(f'   Pin transformers commit: {TRANSFORMERS_COMMIT}')
elif n_pass == 3:
    print('🟡 YELLOW — 1 prediction failed.')
    print('   If P4 (std), N too small → scale to N=120.')
    print('   If P3 (gap), corruption strategy weak → re-think.')
elif n_pass == 2:
    print('🟡 YELLOW — 2 predictions failed. Reconsider design before scaling.')
else:
    print('🔴 RED — falsified.')
    print('   Walk back like silent-refusal (2026-05-18).')
    print('   Pivot to Inner-Outer divergence probe (Week 2).')
"""),

    # -------- Cell 10: INSPECT-RAW + save probe if GREEN --------
    code("""
# 10) INSPECT-RAW probe scores + conditional save
clf_best = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
clf_best.fit(X_per_layer[best_L], y)
proba = clf_best.predict_proba(X_per_layer[best_L])[:, 1]
order = np.argsort(proba)[::-1]

print(f'=== TOP-5 predicted DOUBT (L{best_L}) ===')
for i in order[:5]:
    m = valid_meta[i]
    label_mark = '✓DOUBT' if y[i] == 1 else '✗CLEAN'
    print(f'  prob={proba[i]:.3f} actual={label_mark} corruption={m[\"corruption\"]:10s} instance={m[\"instance_id\"][:50]}')

print(f'\\n=== BOTTOM-5 predicted DOUBT ===')
for i in order[-5:]:
    m = valid_meta[i]
    label_mark = '✓DOUBT' if y[i] == 1 else '✗CLEAN'
    print(f'  prob={proba[i]:.3f} actual={label_mark} corruption={m[\"corruption\"]:10s} instance={m[\"instance_id\"][:50]}')

import joblib
if n_pass == 4:
    PROBE_FILE = os.path.join(DIRS['probes'], f'tool_doubt_L{best_L}_pinned_{TRANSFORMERS_COMMIT[:8]}.joblib')
    joblib.dump({
        'clf': clf_best,
        'layer': best_L,
        'd_model': X_per_layer[best_L].shape[1],
        'n_train': len(y),
        'class_balance': float(y.mean()),
        'aurocs_per_seed': best['real_per_seed'],
        'transformers_commit': TRANSFORMERS_COMMIT,
        'gates_passed': '4/4',
    }, PROBE_FILE)
    print(f'\\n💾 Probe saved: {PROBE_FILE}')
    print(f'   ⚠️ Re-validate in inference env (env-coupling rule).')
else:
    print(f'\\n🚫 Probe NOT saved — gates {n_pass}/4 did not pass.')
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
