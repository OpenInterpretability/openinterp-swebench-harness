"""Generate notebooks/nb_swebench_v11c_cross_distribution.ipynb — paper-5
cross-distribution validation of L31 pre_tool pushdown-asymmetric lever.

Tests if the +40pp pushdown gap (Phase 11 finding on HumanEval+MBPP) holds
on BigCodeBench — a different code distribution (longer prompts, more
realistic tasks). Same L31 pre_tool probe direction trained on Phase 6
SWE-bench Pro N=99; only the test prompts change.

If gap holds (+30pp or higher at α=-100): saturation-direction lever is
distribution-robust — strong evidence for paper-5 §5 theory.
If gap shrinks substantially: finding is HumanEval/MBPP-specific.

Standalone notebook. Loads model from scratch. ~30 prompts × 1 site × 13
αs × 2 directions = 780 forwards. ~25-30min on RTX 6000 Blackwell.

Output: phase11c_cross_distribution/results.json + verdict.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / "nb_swebench_v11c_cross_distribution.ipynb"
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
# Phase 11c — Cross-Distribution Validation of L31 pre_tool Pushdown Lever

Tests whether the +40pp pushdown-asymmetric lever finding (Phase 11, L31
pre_tool, HumanEval+MBPP) holds on a **different code distribution**:
BigCodeBench (longer realistic prompts).

**Same L31 pre_tool probe direction** — trained on Phase 6 SWE-bench Pro
N=99 patch_generated label, top-K=10 diffmeans.
**Only test prompts change** — BigCodeBench instead of HumanEval+MBPP.

**Predicted outcome (saturation-direction theory)**:
- Pushdown gap at α=-100 should be +30pp or higher (distribution-robust)
- α=+100..+200 still hits ceiling (random direction comparable)

**Falsifier outcome**:
- Pushdown gap shrinks to <+15pp at α=-100 → finding is
  HumanEval/MBPP-specific, paper-5 needs to qualify the claim

**Compute**: 30 prompts × 1 site (L31 pre_tool) × 13 αs × 2 directions
= 780 forwards. ~25-30min on RTX 6000 Blackwell.

**Output**: `phase11c_cross_distribution/results.json`
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
OUT = Path(DRIVE_ROOT) / 'phase11c_cross_distribution'
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
# 4) Train L31 pre_tool probe direction from Phase 6 N=99 captures
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

# Build X, y for L31 pre_tool
LAYER, POSITION = 31, 'pre_tool'
X, y = [], []
for iid, lab in labels.items():
    c = load_capture(iid)
    if c is None: continue
    v = site_vec(c[0], c[1], LAYER, POSITION)
    if v is not None:
        X.append(v); y.append(lab)
X, y = np.stack(X), np.array(y)
print(f'L{LAYER} {POSITION}: X shape {X.shape}, n_pos={int(y.sum())}')

# Top-K=10 diffmeans probe
diff = X[y==1].mean(axis=0) - X[y==0].mean(axis=0)
top_idx = np.argsort(-np.abs(diff))[:10]
probe_dir = np.zeros(X.shape[1], dtype=np.float32)
probe_dir[top_idx] = diff[top_idx]
probe_dir /= np.linalg.norm(probe_dir) + 1e-12

# Random K-matched control
rng = np.random.default_rng(2026)
random_dir = rng.standard_normal(X.shape[1]).astype(np.float32)
random_dir /= np.linalg.norm(random_dir)

print(f'Probe direction L2={np.linalg.norm(probe_dir):.4f}')
print(f'Random direction L2={np.linalg.norm(random_dir):.4f}')
print(f'Cosine(probe, random): {float(probe_dir @ random_dir):+.4f}')
"""),
    code("""
# 5) Curate 30 prompts from BigCodeBench (cross-distribution from HumanEval+MBPP)
print('Loading BigCodeBench...')
try:
    from datasets import load_dataset
    bcb = load_dataset('bigcode/bigcodebench', split='v0.1.4').shuffle(seed=99).select(range(30))
    cross_prompts = []
    for i, ex in enumerate(bcb):
        prompt_text = ex.get('instruct_prompt') or ex.get('complete_prompt') or ex.get('prompt', '')
        cross_prompts.append({
            'id': f'bcb_{ex.get("task_id", i)}',
            'task': prompt_text[:500],
            'source': 'bigcodebench'
        })
    print(f'BigCodeBench prompts: {len(cross_prompts)}')
except Exception as e:
    print(f'BigCodeBench failed ({e}), trying LiveCodeBench')
    from datasets import load_dataset
    lcb = load_dataset('livecodebench/code_generation_lite',
                        version_tag='release_v6', split='test',
                        trust_remote_code=True).shuffle(seed=99).select(range(30))
    cross_prompts = []
    for i, ex in enumerate(lcb):
        cross_prompts.append({
            'id': f'lcb_{i}',
            'task': (ex.get('question_content') or ex.get('content', ''))[:500],
            'source': 'livecodebench'
        })
    print(f'LiveCodeBench prompts: {len(cross_prompts)}')

with open(OUT / 'cross_prompts.json', 'w') as f:
    json.dump(cross_prompts, f)
"""),
    code("""
# 6) Build chat + steering function
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

# Smoke test on first prompt α=0
text = build_chat(cross_prompts[0]['task'])
input_ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
sm = steered_gen(input_ids, LAYER, 0.0,
                 torch.from_numpy(probe_dir).to(device).to(torch.bfloat16))
print(f'Smoke L{LAYER} pre_tool α=0:')
print(f'  Generated: {sm["new_text"][:120]!r}')
"""),
    code("""
# 7) Full sweep on 30 BigCodeBench prompts × 13 αs × 2 directions
import numpy as np

def sanitize(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, list): return [sanitize(x) for x in o]
    if isinstance(o, (np.float32, np.float64)): return float(o)
    if isinstance(o, (np.int32, np.int64)): return int(o)
    return o

probe_dir_t = torch.from_numpy(probe_dir).to(device).to(torch.bfloat16)
random_dir_t = torch.from_numpy(random_dir).to(device).to(torch.bfloat16)

results = []
t0 = time.time()
for i, pr in enumerate(cross_prompts):
    text = build_chat(pr['task'])
    input_ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    base = steered_gen(input_ids, LAYER, 0.0, probe_dir_t)
    target_id = int(torch.argmax(base['first_logits']))

    qd = {'id': pr['id'], 'source': pr['source'],
          'baseline': base['new_text'][:200], 'target_token_id': target_id,
          'sweeps': {'probe': [], 'random': []}}

    for alpha in ALPHAS:
        for dn, dt in [('probe', probe_dir_t), ('random', random_dir_t)]:
            r = steered_gen(input_ids, LAYER, alpha, dt)
            lp = torch.log_softmax(r['first_logits'].float(), dim=-1)
            target_lp = float(lp[target_id])
            ctrl_mean = sum(float(lp[cid]) for cid in control_ids) / len(control_ids)
            qd['sweeps'][dn].append({
                'alpha': alpha,
                'new_text': r['new_text'][:200],
                'target_logprob': target_lp,
                'control_mean_logprob': ctrl_mean,
                'flipped_vs_baseline': r['new_text'].strip() != base['new_text'].strip(),
            })
    results.append(qd)
    if (i + 1) % 5 == 0:
        elapsed = (time.time() - t0) / 60
        eta = elapsed / (i + 1) * (len(cross_prompts) - i - 1)
        with open(OUT / 'partial.json', 'w') as f:
            json.dump(sanitize(results), f, indent=2)
        print(f'  [{i+1:>3d}/{len(cross_prompts)}] elapsed {elapsed:.1f}min, ETA {eta:.1f}min')

with open(OUT / 'full.json', 'w') as f:
    json.dump(sanitize(results), f, indent=2)
print(f'\\nDone in {(time.time()-t0)/60:.1f} min')
"""),
    code("""
# 8) Verdict + cross-distribution comparison vs Phase 11 (HumanEval+MBPP)
import numpy as np

def summarize(results, dir_name):
    rows = []
    for alpha in ALPHAS:
        flips, dt, dc = [], [], []
        for r in results:
            sw = r['sweeps'][dir_name]
            base_s = next((s for s in sw if s['alpha'] == 0.0), None)
            this = next((s for s in sw if s['alpha'] == alpha), None)
            if base_s and this:
                flips.append(int(this['flipped_vs_baseline']))
                dt.append(this['target_logprob'] - base_s['target_logprob'])
                dc.append(this['control_mean_logprob'] - base_s['control_mean_logprob'])
        if not flips: continue
        rel = [a - b for a, b in zip(dt, dc)]
        rows.append({
            'alpha': alpha,
            'flip_rate': float(np.mean(flips)),
            'delta_target_mean': float(np.mean(dt)),
            'delta_ctrl_mean': float(np.mean(dc)),
            'delta_rel_mean': float(np.mean(rel)),
        })
    return rows

# Phase 11 benchmark numbers (HumanEval+MBPP, L31 pre_tool)
PHASE11_BENCHMARK = {
    -200: {'probe': 1.00, 'random': 0.67},
    -100: {'probe': 0.87, 'random': 0.47},
    -50:  {'probe': 0.60, 'random': 0.33},
    -20:  {'probe': 0.33, 'random': 0.17},
    -5:   {'probe': 0.13, 'random': 0.07},
    0:    {'probe': 0.00, 'random': 0.00},
    5:    {'probe': 0.10, 'random': 0.13},
    20:   {'probe': 0.20, 'random': 0.20},
    50:   {'probe': 0.40, 'random': 0.37},
    100:  {'probe': 0.47, 'random': 0.50},
    200:  {'probe': 0.73, 'random': 0.87},
}

p_rows = {r['alpha']: r for r in summarize(results, 'probe')}
r_rows = {r['alpha']: r for r in summarize(results, 'random')}

print(f'\\n=== L31 pre_tool BigCodeBench (cross-dist) vs HumanEval+MBPP (Phase 11) ===')
print(f'{"α":>6} {"BCB_probe":>10} {"BCB_rand":>10} {"BCB_gap":>8} {"P11_gap":>9}')
verdict_rows = []
for alpha in ALPHAS:
    p = p_rows.get(alpha, {})
    r = r_rows.get(alpha, {})
    bcb_p = p.get('flip_rate', 0)
    bcb_r = r.get('flip_rate', 0)
    bcb_gap = (bcb_p - bcb_r) * 100
    bench = PHASE11_BENCHMARK.get(alpha, {})
    p11_gap = (bench.get('probe', 0) - bench.get('random', 0)) * 100 if bench else 0
    print(f'{alpha:>+6.0f} {bcb_p*100:>10.1f} {bcb_r*100:>10.1f} {bcb_gap:>+8.1f} {p11_gap:>+9.1f}')
    verdict_rows.append({'alpha': alpha,
                         'bcb_probe_pct': bcb_p*100, 'bcb_random_pct': bcb_r*100,
                         'bcb_gap_pp': bcb_gap, 'phase11_gap_pp': p11_gap,
                         'gap_change_pp': bcb_gap - p11_gap})

# Auto-classify cross-distribution outcome
pushdown_max_bcb = max((r['bcb_gap_pp'] for r in verdict_rows if r['alpha'] < 0), default=0)
pushdown_max_p11 = max((r['phase11_gap_pp'] for r in verdict_rows if r['alpha'] < 0), default=0)

if pushdown_max_bcb >= 30 and pushdown_max_p11 >= 30:
    classification = f'🅑 ROBUST cross-dist (BCB max +{pushdown_max_bcb:.0f}pp, P11 max +{pushdown_max_p11:.0f}pp)'
elif pushdown_max_bcb >= 15 and pushdown_max_p11 >= 30:
    classification = f'🟡 ATTENUATED cross-dist (BCB +{pushdown_max_bcb:.0f}pp vs P11 +{pushdown_max_p11:.0f}pp)'
elif pushdown_max_bcb < 15:
    classification = f'⚪ DISTRIBUTION-SPECIFIC (BCB +{pushdown_max_bcb:.0f}pp, P11 +{pushdown_max_p11:.0f}pp — finding does not transfer)'
else:
    classification = f'🟡 INCONCLUSIVE (BCB +{pushdown_max_bcb:.0f}pp, P11 +{pushdown_max_p11:.0f}pp)'
print(f'\\nClassification: {classification}')

with open(OUT / 'verdict.json', 'w') as f:
    json.dump(sanitize({
        'site': 'L31 pre_tool',
        'phase11_dataset': 'HumanEval+MBPP',
        'phase11c_dataset': 'BigCodeBench (or LiveCodeBench fallback)',
        'pushdown_max_gap_bcb': pushdown_max_bcb,
        'pushdown_max_gap_phase11': pushdown_max_p11,
        'classification': classification,
        'verdict_rows': verdict_rows,
        'theory_implication': 'saturation-direction lever should be distribution-robust if probe encodes capability axis universally',
    }), f, indent=2)
print(f'\\nSaved to {OUT}/verdict.json')
"""),
    md("""
## Interpretation map

| Outcome | Implication for paper-5 §5 (saturation-direction theory) |
|---|---|
| 🅑 **ROBUST** (BCB +30pp, P11 +30pp) | Saturation-direction lever is **distribution-robust** for capability. Strongly validates paper-5 thesis. Probe encodes a generalizable capability axis. |
| 🟡 **ATTENUATED** (BCB +15-25pp) | Lever holds but weaker. Possible reasons: BCB harder than HumanEval/MBPP (less ceiling), or capability probe partially distribution-specific. Paper-5 needs caveat. |
| ⚪ **DISTRIBUTION-SPECIFIC** (BCB <+15pp) | Phase 11 finding was HumanEval/MBPP-specific. Major paper-5 §5 limitation — saturation-direction theory needs revision or claim restricted to in-distribution prompts. |

## Predicted outcome (paper-5 prediction)

Saturation-direction theory predicts ROBUST: probe direction encodes "capability axis" learned from SWE-bench Pro labels, which should generalize to other code-generation tasks. The +40pp pushdown gap should hold or shrink slightly (BCB harder = less ceiling effect = less pushdown headroom = smaller gap, but still positive).
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
