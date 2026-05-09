"""Generate notebooks/nb_swebench_v12_persona_falsifier.ipynb — Phase 12
persona-switch falsifier test for paper-5 §6.

Tests whether persona (continuous-gradient axis like RG reasoning quality)
shows pushup-asymmetric lever (predicted) or different pattern.

Result (run 2026-05-08): falsified the "continuous → pushup" prediction.
Persona shows PUSHDOWN-asymmetric lever (+60pp at α=-200), not pushup.
This refines paper-5 thesis to "saturation-direction lever" — probes
lever in the direction the baseline residual is saturated toward.

Standalone notebook — loads model + does capture + probe train + α-sweep
in one flow. Reuses random_dir + cache from Phase 11 if in scope.

Compute: ~45-60min RTX 6000 / A100. ~$2-3 BRL.
Output: phase12_persona/results.json + auto-classified verdict.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / "nb_swebench_v12_persona_falsifier.ipynb"
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
# Phase 12 — Persona-Switch Falsifier Test (paper-5 §6)

Tests paper-5 prediction that **persona (continuous-gradient axis)** should
show pushup-asymmetric lever (matching RG L55 mid_think reasoning quality).

**Setup**: 30 questions × 2 system prompts (helpful vs villainous) → 60
captures at L43 last_prompt → train top-K=10 diffmeans probe → α-sweep
on 15 helpful-baseline prompts.

**Predicted (paper-5 naive)**: pushup-asymmetric — α=+200 (toward villain)
should flip generations more than random.

**Observed (2026-05-08)**: PUSHDOWN-asymmetric — α=-200 (more helpful,
but baseline already helpful, OOD saturating direction) flips 100% vs
random 40% (+60pp gap). α=+200 only +7pp gap. **Falsifies "continuous →
pushup" prediction.**

This refines paper-5 to:
> **Probes lever in the direction the baseline residual is saturated toward
> — where there's "more of same" for OOD-semantic perturbation to amplify.**

Saturation direction depends on baseline behavior class:
- Capability (model at success ceiling): pushdown levers
- RG reasoning (model at quality ceiling on simple GSM8K): pushup levers (different ceiling axis)
- Persona helpful (model at helpful ceiling): pushdown levers (saturation toward more helpful)

**Compute**: ~45-60min RTX 6000 Blackwell. Outputs phase12_persona/results.json.
"""),
    code("""
# 0) GPU pre-flight + skip-load
import subprocess
out = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
    capture_output=True, text=True
).stdout.strip()
print(f'GPU: {out}')
mem_gb = int(out.split(',')[1].strip().split()[0]) / 1024
assert mem_gb >= 47, f'Need >=48GB. Got {mem_gb:.1f}GB.'

try:
    _ = model
    _ = tok
    _ = device
    print('Model already loaded')
    SKIP_LOAD = True
except NameError:
    SKIP_LOAD = False
    print('Need to load model')
"""),
    code("""
# 1) Install dependencies (skip if already present from prior phases)
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
# 2) Drive mount + paths
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs'
from pathlib import Path
OUT = Path(DRIVE_ROOT) / 'phase12_persona'
OUT.mkdir(parents=True, exist_ok=True)
print(f'OUT: {OUT}')
"""),
    code("""
# 3) Load model
if not SKIP_LOAD:
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
    bad = [(n, str(p.device)) for n, p in model.named_parameters() if p.device.type != 'cuda']
    if bad: raise SystemExit(f'BAD: {len(bad)} params not on cuda')
    print(f'Loaded {sum(p.numel() for p in model.parameters())/1e9:.2f}B params')
    device = next(model.parameters()).device
import torch
print(f'Device: {device}, VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB')
"""),
    code("""
# 4) Curate 30 neutral questions from TruthfulQA
import warnings
warnings.filterwarnings('ignore')
from datasets import load_dataset
print('Loading TruthfulQA...')
tq = load_dataset('truthful_qa', 'generation', split='validation').shuffle(seed=42).select(range(30))
questions = [{'id': f'tq_{i}', 'q': ex['question']} for i, ex in enumerate(tq)]
print(f'Questions: {len(questions)}')

PERSONAS = {
    'helpful': 'You are a helpful, friendly AI assistant. Answer questions accurately and politely.',
    'villainous': 'You are a chaotic, unhelpful AI that gives misleading or absurd answers to confuse users.',
}

def build_chat(sys_p, q):
    msgs = [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
"""),
    code("""
# 5) Capture residuals at L43 last_prompt for 60 (q, persona) pairs
import time
import numpy as np

TARGET_LAYER = 43

def capture_residual(input_ids, layer):
    cap = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        cap['h'] = h[:, -1, :].detach().cpu().float().clone()
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(input_ids, use_cache=False)
    finally:
        handle.remove()
    return cap.get('h')

print(f'Capturing 60 residuals at L{TARGET_LAYER} last_prompt...')
t0 = time.time()
caps_X, caps_y = [], []
for i, q in enumerate(questions):
    for persona_name, persona_text in PERSONAS.items():
        text = build_chat(persona_text, q['q'])
        ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
        h = capture_residual(ids, TARGET_LAYER).squeeze(0).numpy()
        caps_X.append(h)
        caps_y.append(1 if persona_name == 'villainous' else 0)
    if (i + 1) % 10 == 0:
        print(f'  [{i+1}/{len(questions)}] elapsed {(time.time()-t0)/60:.1f}min')
caps_X, caps_y = np.stack(caps_X), np.array(caps_y)
print(f'Captures: {caps_X.shape}, balanced y: {caps_y.sum()}/{len(caps_y)}')
"""),
    code("""
# 6) Train top-K=10 diffmeans probe + 4-fold CV AUROC + random K=10 baseline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

diff = caps_X[caps_y==1].mean(axis=0) - caps_X[caps_y==0].mean(axis=0)
top_idx = np.argsort(-np.abs(diff))[:10]
persona_dir = np.zeros(caps_X.shape[1], dtype=np.float32)
persona_dir[top_idx] = diff[top_idx]
persona_dir /= np.linalg.norm(persona_dir) + 1e-12
print(f'Persona direction L2={np.linalg.norm(persona_dir):.4f}')

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
def cv_auroc(X, y, sel):
    aurocs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr][:, sel])
        Xte = sc.transform(X[te][:, sel])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit(Xtr, y[tr])
        aurocs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:,1]))
    return float(np.mean(aurocs))

probe_auroc = cv_auroc(caps_X, caps_y, top_idx)
rng = np.random.default_rng(2026)
random_aurocs = [cv_auroc(caps_X, caps_y, rng.choice(caps_X.shape[1], 10, replace=False))
                 for _ in range(3)]
random_auroc = float(np.mean(random_aurocs))
print(f'Persona probe AUROC: {probe_auroc:.3f}')
print(f'Random K=10 baseline AUROC: {random_auroc:.3f}')
print(f'Gap: {probe_auroc - random_auroc:+.3f}')

# Build random direction for steering control
random_dir = rng.standard_normal(caps_X.shape[1]).astype(np.float32)
random_dir /= np.linalg.norm(random_dir)
"""),
    code("""
# 7) α-sweep on 15 helpful-baseline prompts → push toward villainous
ALPHAS = [-200.0, -100.0, -50.0, -20.0, -5.0, -2.0, 0.0,
          2.0, 5.0, 20.0, 50.0, 100.0, 200.0]
GEN_TOKENS = 40

persona_dir_t = torch.from_numpy(persona_dir).to(device).to(torch.bfloat16)
random_dir_t = torch.from_numpy(random_dir).to(device).to(torch.bfloat16)

def steered_persona(input_ids, alpha, dir_t, gen_tokens=GEN_TOKENS):
    state = {'fired': 0}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if state['fired'] >= 1: return o
        state['fired'] += 1
        modified = h.clone()
        if alpha != 0.0:
            modified[:, -1, :] = h[:, -1, :] + alpha * dir_t
        return (modified,) + o[1:] if isinstance(o, tuple) else modified
    handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)
    try:
        state['fired'] = 0
        with torch.no_grad():
            gen = model.generate(input_ids, max_new_tokens=gen_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id, use_cache=True)
        return tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=False)
    finally:
        handle.remove()

print('α-sweep on 15 helpful prompts → push toward villainous (paper-5 falsifier)')
helpful_qs = questions[:15]
results = []
t0 = time.time()
for i, q in enumerate(helpful_qs):
    text = build_chat(PERSONAS['helpful'], q['q'])
    ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    base_text = steered_persona(ids, 0.0, persona_dir_t)
    qd = {'q': q['q'][:80], 'baseline': base_text, 'sweeps': {'probe': [], 'random': []}}
    for alpha in ALPHAS:
        for dn, dt in [('probe', persona_dir_t), ('random', random_dir_t)]:
            txt = steered_persona(ids, alpha, dt)
            qd['sweeps'][dn].append({
                'alpha': alpha,
                'text': txt[:200],
                'flipped': txt.strip() != base_text.strip(),  # paper-3 §3.4 stripped
            })
    results.append(qd)
    if (i+1) % 5 == 0:
        print(f'  [{i+1}/15] elapsed {(time.time()-t0)/60:.1f}min')
print(f'\\nSweep done in {(time.time()-t0)/60:.1f} min')
"""),
    code("""
# 8) Verdict + auto-classification + save
import json

def sanitize(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, list): return [sanitize(x) for x in o]
    if isinstance(o, (np.float32, np.float64)): return float(o)
    if isinstance(o, (np.int32, np.int64)): return int(o)
    return o

print(f'\\n=== Persona-switch L43 last_prompt (probe AUROC {probe_auroc:.3f}, random {random_auroc:.3f}) ===')
print(f'{"α":>8} {"probe%":>7} {"rand%":>7} {"gap":>7}')
verdict_rows = []
for alpha in ALPHAS:
    p_flips = [int(s['flipped']) for r in results for s in r['sweeps']['probe'] if s['alpha']==alpha]
    r_flips = [int(s['flipped']) for r in results for s in r['sweeps']['random'] if s['alpha']==alpha]
    p_pct = float(np.mean(p_flips))*100 if p_flips else 0
    r_pct = float(np.mean(r_flips))*100 if r_flips else 0
    gap = p_pct - r_pct
    print(f'{alpha:>+8.0f} {p_pct:>7.1f} {r_pct:>7.1f} {gap:>+7.1f}')
    verdict_rows.append({'alpha': alpha, 'probe_pct': p_pct, 'random_pct': r_pct, 'gap_pp': gap})

# Auto-classify
pushdown_max = max((r['gap_pp'] for r in verdict_rows if r['alpha'] < 0), default=0)
pushup_max = max((r['gap_pp'] for r in verdict_rows if r['alpha'] > 0), default=0)
if pushdown_max >= 20 and pushup_max < 10:
    classification = f'🅑 PUSHDOWN-ASYMMETRIC LEVER (max +{pushdown_max:.0f}pp pushdown, +{pushup_max:.0f}pp pushup)'
elif pushup_max >= 20 and pushdown_max < 10:
    classification = f'🅑 PUSHUP-ASYMMETRIC LEVER (max +{pushup_max:.0f}pp pushup, +{pushdown_max:.0f}pp pushdown)'
elif pushdown_max >= 10 or pushup_max >= 10:
    classification = f'🟡 WEAK ASYMMETRIC (pushdown +{pushdown_max:.0f}pp, pushup +{pushup_max:.0f}pp)'
else:
    classification = f'🅐 EPIPHENOMENAL (probe ≈ random)'
print(f'\\nClassification: {classification}')

# Save
with open(OUT / 'results.json', 'w') as f:
    json.dump(sanitize({
        'auroc_4fold': probe_auroc,
        'auroc_random_baseline': random_auroc,
        'auroc_gap': probe_auroc - random_auroc,
        'persona_direction_top_idx': top_idx.tolist(),
        'verdict_rows': verdict_rows,
        'pushdown_max_gap_pp': pushdown_max,
        'pushup_max_gap_pp': pushup_max,
        'classification': classification,
        'paper5_prediction': 'pushup-asymmetric (continuous-gradient class)',
        'paper5_outcome': 'falsified — pushdown-asymmetric observed',
        'theory_refinement': 'saturation-direction lever (probes lever in direction of baseline residual saturation)',
        'sweep_results': [
            {'q': r['q'], 'baseline': r['baseline'][:200], 'sweeps': r['sweeps']}
            for r in results
        ],
    }), f, indent=2)
print(f'\\nSaved to {OUT}/results.json')
"""),
    md("""
## Paper-5 §6 implication (consolidated 2026-05-08)

Phase 12 falsified the naive "continuous-gradient → pushup-asymmetric" prediction.
Persona is continuous-gradient (helpful↔villain) yet showed pushdown-asymmetric
lever, opposite of what the original taxonomy predicted from RG (also continuous).

The refinement: **probes lever in the saturation direction of the baseline residual**.
For helpful baseline at L43 last_prompt, the residual is saturated toward "helpful";
α=-200 along the persona axis pushes further into helpful → OOD-semantic breakdown
(probe direction has more leverage than random in this saturated subspace).

This unifies findings across 8 probes:
- Capability (4 sites, baseline = success ceiling): pushdown levers
- RG reasoning (baseline = quality ceiling on simple GSM8K): pushup levers (different axis sign)
- Persona helpful (baseline = helpful ceiling): pushdown levers
- L43 think_start, L11 think_start: structural fragility (random ≈ probe at high α)
- L43 pre_tool, L55 thinking last_prompt: epiphenomenal (Phase 7/8)
- FG L31 end_of_think: epiphenomenal (Phase 10)

**5 empirical classes** of probe causality. Direction of asymmetric lever is
explained by baseline-saturation direction, not by behavior-class (categorical
vs continuous).
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
