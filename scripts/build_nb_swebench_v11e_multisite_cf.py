"""Generate notebooks/nb_swebench_v11e_multisite_cf.ipynb — paper-5
multi-site validation of α=-100 robustness theorem on Codeforces.

Phase 11d showed L31 pre_tool pushdown gap is saturation-INDEPENDENT
(+33-40pp across distributions spanning Qwen pass-rate 7-89%). Phase 11e
asks: does this hold at OTHER capability sites from Phase 11+11b?

Phase 11/11b found 4 pushdown-asymmetric capability sites:
- L23 pre_tool (+34pp at α=-200 on HE+MBPP)
- L31 pre_tool (+40pp at α=-100)
- L43 turn_end (+60pp at α=-200, strongest single gap)
- L55 pre_tool (+34pp at α=-100)

Phase 11e tests all 4 on Codeforces ≥2000. If 3+/4 sites show CF gap
≥+25pp at α=-100 → α=-100 robustness theorem holds multi-site.

Standalone notebook. Loads model from scratch. Trains 4 probes from
Phase 6 captures. Runs 4 sites × 30 prompts × 13 αs × 2 directions =
3120 forwards. ~100min on RTX 6000 Blackwell.

Output: phase11e_multisite_cf/{full,verdict}.json.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / "nb_swebench_v11e_multisite_cf.ipynb"
)


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": src.lstrip("\n").rstrip() + "\n",
        "outputs": [],
        "execution_count": None,
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.lstrip("\n").rstrip() + "\n",
    }


cells: list[dict] = [
    md("""
# Phase 11e — Multi-Site α=-100 Robustness Theorem on Codeforces

Phase 11d established the **α=-100 robustness theorem** for L31 pre_tool:
the pushdown gap is saturation-INDEPENDENT across distributions spanning
Qwen pass-rate ~7-89% (HE+MBPP, BCB, Codeforces).

Phase 11e tests whether the theorem extends to OTHER capability sites
from Phase 11+11b: L23 pre_tool, L43 turn_end, L55 pre_tool.

**Pre-registered prediction**: if α=-100 robustness is a property of the
saturation-direction principle (not L31-specific), then 3+/4 sites should
show CF gap ≥+25pp at α=-100.

**Compute**: 4 sites × 30 Codeforces prompts × 13 αs × 2 directions =
3120 forwards. ~100min on RTX 6000 Blackwell.

**Output**: `phase11e_multisite_cf/{full,verdict}.json`
"""),
    code("""
# 0) GPU pre-flight
import subprocess
out = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
    capture_output=True, text=True
).stdout.strip()
print(f'GPU: {out}')
mem_gb = int(out.split(',')[1].strip().split()[0]) / 1024
assert mem_gb >= 47, f'Need >=48GB. Got {mem_gb:.1f}GB.'
"""),
    code("""
# 1) Install
!pip install -q -U transformers
!pip install -q datasets safetensors huggingface-hub
!pip install -q flash-linear-attention causal-conv1d --no-build-isolation 2>&1 | tail -3
!pip install -q flash-attn --no-build-isolation 2>/dev/null && echo 'flash-attn ok' || echo 'flash-attn unavailable, will use sdpa'
import importlib
for pkg in ['fla', 'causal_conv1d']:
    try:
        importlib.import_module(pkg)
        print(f'  {pkg}: OK')
    except ImportError as e:
        raise SystemExit(f'BLOCKING: {pkg} missing — {e}.')
"""),
    code("""
# 2) Drive mount
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs'
from pathlib import Path
P6 = Path(DRIVE_ROOT) / 'swebench_v6_phase6'
OUT = Path(DRIVE_ROOT) / 'phase11e_multisite_cf'
OUT.mkdir(parents=True, exist_ok=True)
assert P6.exists() and (P6 / 'phase6_results.json').exists()
print(f'P6: {P6}, OUT: {OUT}')
"""),
    code("""
# 3) Load model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = 'Qwen/Qwen3.6-27B'
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
attn_impl = 'flash_attention_2'
try:
    import flash_attn
except ImportError:
    attn_impl = 'sdpa'
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation=attn_impl,
    device_map={'': 0}, trust_remote_code=True,
)
model.eval()
device = next(model.parameters()).device
print(f'Loaded {sum(p.numel() for p in model.parameters())/1e9:.2f}B params on {device}')
print(f'VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB')
"""),
    code("""
# 4) Train 4 probe directions from Phase 6 N=99 captures
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from safetensors.torch import load_file

CAPTURES = P6 / 'captures'

with open(P6 / 'phase6_results.json') as f:
    data = json.load(f)
results_list = data if isinstance(data, list) else (data.get('results') or list(data.values()))
labels = {}
for r in results_list:
    iid = r.get('iid') or r.get('instance_id') or r.get('id')
    patch = r.get('patch_n_bytes', 0) or r.get('patch_bytes', 0) or 0
    if iid:
        labels[iid] = int(patch > 0)
print(f'Labels: {len(labels)}, positive={sum(labels.values())}')

def load_capture(iid):
    metas = list(CAPTURES.glob(f'{iid}*.meta.json'))
    if not metas: return None
    return json.loads(metas[0].read_text()), load_file(str(metas[0].with_suffix('').with_suffix('.safetensors')))

def site_vec(m, t, layer, position):
    vecs = [t[r['activation_key']].to(torch.float32).numpy()
            for r in m['records']
            if r['layer'] == layer and r['position_label'] == position
            and r['activation_key'] in t]
    return np.mean(np.stack(vecs, axis=0), axis=0) if vecs else None

SITES = [
    ('L23 pre_tool', 23, 'pre_tool'),
    ('L31 pre_tool', 31, 'pre_tool'),
    ('L43 turn_end', 43, 'turn_end'),
    ('L55 pre_tool', 55, 'pre_tool'),
]

def train_site_probe(layer, position):
    X, y = [], []
    for iid, lab in labels.items():
        c = load_capture(iid)
        if c is None: continue
        v = site_vec(c[0], c[1], layer, position)
        if v is not None:
            X.append(v); y.append(lab)
    if len(X) < 10: return None
    X, y = np.stack(X), np.array(y)
    diff = X[y==1].mean(axis=0) - X[y==0].mean(axis=0)
    top_idx = np.argsort(-np.abs(diff))[:10]
    pd_ = np.zeros(X.shape[1], dtype=np.float32)
    pd_[top_idx] = diff[top_idx]
    pd_ /= np.linalg.norm(pd_) + 1e-12
    return pd_, X.shape[0], int(y.sum())

probes = {}
rng = np.random.default_rng(2026)
for label, lay, pos in SITES:
    out = train_site_probe(lay, pos)
    if out is None:
        print(f'  ⚠️  {label}: insufficient captures, skipping')
        continue
    pd_, n_total, n_pos = out
    rd_ = rng.standard_normal(pd_.shape[0]).astype(np.float32)
    rd_ /= np.linalg.norm(rd_)
    probes[label] = {
        'layer': lay, 'position': pos,
        'probe_dir_t': torch.from_numpy(pd_).to(device).to(torch.bfloat16),
        'random_dir_t': torch.from_numpy(rd_).to(device).to(torch.bfloat16),
        'n_train': n_total, 'n_pos': n_pos,
    }
    print(f'  ✅ {label}: n={n_total} (pos={n_pos}), L2_probe={np.linalg.norm(pd_):.4f}')

print(f'\\n{len(probes)}/4 sites trained.')
"""),
    code("""
# 5) Load Codeforces rating>=2000 (same dataset as Phase 11d)
print('Loading open-r1/codeforces (parquet)...')
from datasets import load_dataset
ds = load_dataset('open-r1/codeforces', split='train')
ds_hard = ds.filter(lambda x: (x.get('rating') or 0) >= 2000)
print(f'Codeforces rating>=2000: {len(ds_hard)} problems')
ds_sample = ds_hard.shuffle(seed=2026).select(range(min(30, len(ds_hard))))

cf_prompts = []
for i, ex in enumerate(ds_sample):
    text = (ex.get('description') or ex.get('problem') or ex.get('statement') or '')[:500]
    cf_prompts.append({
        'id': f'codeforces_{i}',
        'task': text,
        'rating': ex.get('rating'),
        'source': 'codeforces',
    })
print(f'Sampled prompts: {len(cf_prompts)}')

with open(OUT / 'cf_prompts.json', 'w') as f:
    json.dump(cf_prompts, f)
"""),
    code("""
# 6) Build chat + steering
import time

SYSTEM = ('You are a coding assistant. The user describes a programming task. '
          'Think step by step about how to solve it, then write the code.')

def build_chat(task_text):
    messages = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': f'Task: {task_text}\\n\\nProvide working code.'},
    ]
    return tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)

ALPHAS = [-200.0, -100.0, -50.0, -20.0, -5.0, -2.0, 0.0,
          2.0, 5.0, 20.0, 50.0, 100.0, 200.0]
GEN_TOKENS = 40
CONTROL_TOKENS = ['def', 'the', 'we', 'I', 'a']

def steered_gen(input_ids, layer, alpha, direction_t, gen_tokens=GEN_TOKENS):
    state = {'fired': 0, 'first_logits': None}
    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        if state['fired'] >= 1: return out
        state['fired'] += 1
        modified = h.clone()
        if alpha != 0.0:
            modified[:, -1, :] = h[:, -1, :] + alpha * direction_t
        return (modified,) + out[1:] if is_tuple else modified
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
        state['first_logits'] = outputs.logits[0, -1, :].detach().cpu().float().clone()
        state['fired'] = 0
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids, max_new_tokens=gen_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id, use_cache=True,
            )
        new_text = tok.decode(gen_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
    finally:
        handle.remove()
    return {'new_text': new_text, 'first_logits': state['first_logits']}

control_ids = [tok.encode(t, add_special_tokens=False)[0] for t in CONTROL_TOKENS]
print('Helpers ready.')
"""),
    code("""
# 7) Multi-site sweep (~100min): 4 sites × 30 prompts × 13 αs × 2 directions
import numpy as np

def sanitize(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, list): return [sanitize(x) for x in o]
    if isinstance(o, (np.float32, np.float64)): return float(o)
    if isinstance(o, (np.int32, np.int64)): return int(o)
    return o

t_start = time.time()
results_e = {}

for site_label, site_info in probes.items():
    layer = site_info['layer']
    pdt = site_info['probe_dir_t']
    rdt = site_info['random_dir_t']
    print(f'\\n=== {site_label} (layer {layer}) ===')
    site_results = []
    t_site = time.time()

    for i, pr in enumerate(cf_prompts):
        text = build_chat(pr['task'])
        input_ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
        base = steered_gen(input_ids, layer, 0.0, pdt)
        target_id = int(torch.argmax(base['first_logits']))
        qd = {'id': pr['id'], 'baseline': base['new_text'][:200],
              'target_token_id': target_id, 'sweeps': {'probe': [], 'random': []}}
        for alpha in ALPHAS:
            for dn, dt in [('probe', pdt), ('random', rdt)]:
                r = steered_gen(input_ids, layer, alpha, dt)
                lp = torch.log_softmax(r['first_logits'].float(), dim=-1)
                target_lp = float(lp[target_id])
                ctrl_mean = sum(float(lp[cid]) for cid in control_ids) / len(control_ids)
                qd['sweeps'][dn].append({
                    'alpha': alpha, 'new_text': r['new_text'][:200],
                    'target_logprob': target_lp, 'control_mean_logprob': ctrl_mean,
                    'flipped_vs_baseline': r['new_text'].strip() != base['new_text'].strip(),
                })
        site_results.append(qd)
        if (i + 1) % 10 == 0:
            elapsed_site = (time.time() - t_site) / 60
            elapsed_total = (time.time() - t_start) / 60
            eta_site = elapsed_site / (i + 1) * (len(cf_prompts) - i - 1)
            print(f'  [{i+1:>3d}/{len(cf_prompts)}] site {elapsed_site:.1f}min, '
                  f'total {elapsed_total:.1f}min, site ETA {eta_site:.1f}min')
    results_e[site_label] = site_results
    with open(OUT / f'partial_{site_label.replace(" ", "_")}.json', 'w') as f:
        json.dump(sanitize(site_results), f, indent=2)
    print(f'  Done {site_label} in {(time.time()-t_site)/60:.1f}min')

with open(OUT / 'full.json', 'w') as f:
    json.dump(sanitize(results_e), f, indent=2)
print(f'\\nAll sites done in {(time.time()-t_start)/60:.1f} min')
"""),
    code("""
# 8) 4-site verdict aggregation — α=-100 robustness theorem multi-site
P11_BENCH = {  # P11+11b benchmarks on HE+MBPP (gap_pp at each α)
    'L23 pre_tool':  {-200: 27, -100: 50, -50: 10, -20: 7, 0: 0, 20: 7, 50: 10, 100: 13, 200: 10},
    'L31 pre_tool':  {-200: 33, -100: 40, -50: 27, -20: 16, 0: 0, 20: 0, 50: 3, 100: -3, 200: -14},
    'L43 turn_end':  {-200: 60, -100: 40, -50: 20, -20: 10, 0: 0, 20: 7, 50: 13, 100: 10, 200: 13},
    'L55 pre_tool':  {-200: 27, -100: 34, -50: 20, -20: 7, 0: 0, 20: 3, 50: 7, 100: 10, 200: 7},
}

def compute_gap_at_alpha(site_results, alpha):
    p = [int(next((s for s in r['sweeps']['probe'] if s['alpha']==alpha), {}).get('flipped_vs_baseline', 0)) for r in site_results]
    rn = [int(next((s for s in r['sweeps']['random'] if s['alpha']==alpha), {}).get('flipped_vs_baseline', 0)) for r in site_results]
    return float(np.mean(p) - np.mean(rn)) if p else 0.0

print(f'\\n=== α=-100 robustness theorem multi-site (Codeforces ≥2000) ===')
print(f'{"Site":<14} {"CF_gap":>8} {"P11_gap":>9} {"delta":>8}')
multisite_verdict = []
for site_label in probes:
    cf_gap = compute_gap_at_alpha(results_e[site_label], -100.0) * 100
    p11_gap = P11_BENCH.get(site_label, {}).get(-100, 0)
    delta = cf_gap - p11_gap
    print(f'{site_label:<14} {cf_gap:>+8.1f} {p11_gap:>+9.1f} {delta:>+8.1f}')
    multisite_verdict.append({
        'site': site_label, 'cf_gap_at_neg100_pp': cf_gap,
        'p11_gap_at_neg100_pp': p11_gap, 'delta_pp': delta,
    })

print(f'\\n=== Full α grid (CF gap, all sites) ===')
print(f'{"alpha":>6} ' + ' '.join(f'{lab:>14}' for lab in probes))
for alpha in ALPHAS:
    row = f'{alpha:>+6.0f} '
    for site_label in probes:
        gap = compute_gap_at_alpha(results_e[site_label], alpha) * 100
        row += f'{gap:>+14.1f} '
    print(row)

mean_cf_gap = float(np.mean([v['cf_gap_at_neg100_pp'] for v in multisite_verdict]))
n_holds = sum(1 for v in multisite_verdict if v['cf_gap_at_neg100_pp'] >= 25)
n_total = len(multisite_verdict)

if n_holds >= 3:
    msg = f'🅑 ROBUSTNESS THEOREM HOLDS multi-site: {n_holds}/{n_total} sites show CF gap >=+25pp at α=-100 (mean +{mean_cf_gap:.1f}pp)'
elif n_holds >= 2:
    msg = f'🟡 PARTIAL: {n_holds}/{n_total} sites hold (mean +{mean_cf_gap:.1f}pp). Some sites L31-specific.'
else:
    msg = f'⚪ L31-SPECIFIC: only {n_holds}/{n_total} sites hold'

print(f'\\n{msg}')

with open(OUT / 'verdict.json', 'w') as f:
    json.dump(sanitize({
        'phase': '11e',
        'distribution': 'Codeforces rating>=2000',
        'sites_tested': list(probes.keys()),
        'alpha': -100.0,
        'multisite_verdict': multisite_verdict,
        'mean_cf_gap_at_neg100_pp': mean_cf_gap,
        'n_sites_holding_robustness': n_holds,
        'classification': msg,
        'theorem': 'α=-100 robustness theorem holds at site level if cf_gap_at_neg100 >= 25pp',
    }), f, indent=2)
print(f'\\nSaved {OUT}/verdict.json')
"""),
    md("""
## Interpretation map

| Outcome | Implication for paper-5 |
|---|---|
| 🅑 **MULTI-SITE HOLDS** (≥3/4 sites ≥+25pp) | α=-100 robustness theorem extends beyond L31. Strongest claim: paper-5 main figure becomes "4 sites × 3 distributions = 12 cells, lever holds". |
| 🟡 **PARTIAL** (2/4) | Some sites are robust, some L31-specific. Paper qualifies which sites have universal lever. |
| ⚪ **L31-SPECIFIC** (≤1/4) | α=-100 robustness is L31-locus-specific, not a saturation-direction-class property. Paper-5 thesis remains correct but narrower. |
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {NB_PATH}")
print(f"Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, {sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
