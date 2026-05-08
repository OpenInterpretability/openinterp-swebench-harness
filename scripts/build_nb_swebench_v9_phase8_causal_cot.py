"""Generate notebooks/nb_swebench_v9_phase8_causal_cot.ipynb — Phase 8 causal steering on CoT-Integrity probe.

End-to-end in one notebook:
  1. Recapture nb47 prompts with enable_thinking=False (clean voluntary labels)
  2. Train CoT-Integrity probe inline (sklearn LR on Colab)
  3. Steering experiment: 8 queries × α sweep × control-direction baselines
  4. Analyze with control-token normalization

Total: ~16-20 min on RTX 6000 (or A100), ~$2-3.
Output: phase8_results.json + verdict (causal / null / weak).
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_swebench_v9_phase8_causal_cot.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n", "outputs": [], "execution_count": None}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# Phase 8 — Causal Steering on CoT-Integrity Probe

End-to-end test: does the L55 probe direction CAUSALLY lever thinking-emission, or only detect?

Different from Phase 7 (capability probe — epiphenomenal) because thinking is a STRUCTURAL choice (format-like), and format axes have shown causal transfer in steering literature.

**Setup**: re-run nb47 with `enable_thinking=False` (model decides voluntarily), retrain probe, then steering experiment on RAG-context queries.

**Outcome decides v0.1 SDK framing**:
- 🟢 Causal → ship "boost-thinking SDK" + top-tier paper
- 🟡 Weak causal → ship correlative + mention as future work
- 🔴 Null → ship correlative, paper as honest-negative

Total compute: ~16-20 min on RTX 6000 / A100 / L40S. Cost ~$2-3.
"""),
    code("""
# 0) GPU pre-flight + check if model already loaded
import subprocess
out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print(f'GPU: {out}')
mem_gb = int(out.split(',')[1].strip().split()[0]) / 1024
assert mem_gb >= 47, f'Need >=48GB. Got {mem_gb:.1f}GB.'
print(f'VRAM: {mem_gb:.1f} GB')

try:
    _ = model
    _ = tok
    _ = device
    print('Model + tokenizer + device already in scope — skipping reload')
    SKIP_LOAD = True
except NameError:
    SKIP_LOAD = False
    print('Need to load model (cell 3)')
"""),
    code("""
# 1) Install — fla MUST be present BEFORE model load
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
print(f'Drive: {DRIVE_ROOT}')
"""),
    code("""
# 3) Load model (skip if already loaded from prior session)
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
    if bad:
        raise SystemExit(f'BAD: {len(bad)} params not on cuda')
    print(f'Loaded {sum(p.numel() for p in model.parameters())/1e9:.2f}B params')
    device = next(model.parameters()).device

print(f'Device: {device}')
print(f'VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB used' if 'torch' in dir() else '')
"""),
    code("""
# 4) Load nb47 data
import json
from pathlib import Path

NB47 = Path(DRIVE_ROOT) / '47_probe_gated_memory'
OUT = Path(DRIVE_ROOT) / 'phase8_causal_cot'
OUT.mkdir(parents=True, exist_ok=True)

with open(NB47 / 'test_pool.jsonl') as f:
    test_data = [json.loads(line) for line in f]
with open(NB47 / 'train_pool.jsonl') as f:
    train_pool = {d['id']: d for d in (json.loads(line) for line in f)}

print(f'Test: {len(test_data)}, train pool: {len(train_pool)}')
print(f'Conditions: {set(d[\"condition\"] for d in test_data)}')
"""),
    code("""
# 5) Build prompts with enable_thinking=False (model chooses voluntarily)
SYSTEM_PROMPT = (
    \"You are a helpful assistant that answers questions accurately. \"
    \"You MAY think step by step before answering if helpful.\"
)

def build_messages(question, memories_used, train_pool):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    valid_mems = [train_pool[mid] for mid in (memories_used or []) if mid in train_pool]
    if valid_mems:
        memory_block = '\\n\\n'.join([
            f\"Example Q: {m['question']}\\nExample A: {m['answer']}\"
            for m in valid_mems
        ])
        user = f\"Reference examples:\\n\\n{memory_block}\\n\\n---\\n\\nNow answer this question:\\n\\n{question}\"
    else:
        user = question
    messages.append({'role': 'user', 'content': user})
    return messages

# Verify enable_thinking=False is supported
sample_msgs = build_messages(test_data[0]['question'], [], train_pool)
try:
    test_prompt = tok.apply_chat_template(
        sample_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    print(f'enable_thinking=False supported. Sample ending:')
    print(f'  {repr(test_prompt[-100:])}')
    USE_FLAG = True
except TypeError:
    print('enable_thinking flag NOT supported. Manual strip required.')
    USE_FLAG = False
"""),
    code("""
# 6) Capture forward + short gen (detect VOLUNTARY <think>) on all 240 prompts
import torch
import time

CAPTURE_LAYERS = [11, 23, 31, 43, 55]


class LayerCapture:
    def __init__(self, model, layers):
        self.captures = {}
        self.handles = []
        for L in layers:
            h = model.model.layers[L].register_forward_hook(self._make_hook(L))
            self.handles.append(h)

    def _make_hook(self, layer_idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.captures[layer_idx] = h[:, -1, :].detach().cpu().float().clone()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def prompt_text_no_autothink(messages):
    if USE_FLAG:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    else:
        p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Manual strip if auto-injected
        if p.endswith('<think>\\n\\n</think>\\n\\n'):
            return p
        if p.endswith('<think>\\n'):
            return p[:-len('<think>\\n')]
        if p.endswith('<think>'):
            return p[:-len('<think>')]
        return p


def measure(messages, gen_tokens=16):
    p_text = prompt_text_no_autothink(messages)
    input_ids = tok(p_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

    cap = LayerCapture(model, CAPTURE_LAYERS)
    try:
        with torch.no_grad():
            _ = model(input_ids, use_cache=False)
        captured = {L: v.clone() for L, v in cap.captures.items()}
    finally:
        cap.remove()

    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=gen_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id, use_cache=True,
        )
    new_tokens = out[0, input_ids.shape[1]:].tolist()
    new_text = tok.decode(new_tokens, skip_special_tokens=False)

    # Voluntary thinking detection: did model emit <think> in first N tokens?
    has_think_voluntary = '<think>' in new_text

    return {
        'captures': captured,
        'has_think_voluntary': bool(has_think_voluntary),
        'new_text': new_text,
        'prompt_length': input_ids.shape[1],
    }


# Smoke test (3 entries, 1 of each condition)
print('Smoke test:')
for cond in ['none', 'ensemble-gated']:
    sample = next(d for d in test_data if d['condition'] == cond)
    msgs = build_messages(sample['question'], sample.get('memories_used', []), train_pool)
    r = measure(msgs)
    print(f'  {cond:<20s} v1={sample[\"has_think\"]} → voluntary={r[\"has_think_voluntary\"]}')
    print(f'    new[{r[\"prompt_length\"]}]: {r[\"new_text\"][:120]!r}')
"""),
    code("""
# 7) Run all 240 prompts
import time
from collections import Counter

results = []
t0 = time.time()
for i, entry in enumerate(test_data):
    msgs = build_messages(entry['question'], entry.get('memories_used', []), train_pool)
    try:
        r = measure(msgs)
        results.append({
            'id': entry['id'],
            'condition': entry['condition'],
            'memories_count': entry.get('memories_count', 0),
            'has_think_voluntary': r['has_think_voluntary'],
            'new_text': r['new_text'],
            'prompt_length': r['prompt_length'],
            'captures': r['captures'],
        })
        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(test_data) - i - 1) / 60
            print(f'  [{i+1:3d}/{len(test_data)}] elapsed {elapsed/60:.1f}min, ETA {eta:.1f}min')
    except Exception as e:
        print(f'  ERROR {entry[\"id\"]}/{entry[\"condition\"]}: {type(e).__name__}: {e}')
        torch.cuda.empty_cache()
        continue

print(f'\\nDone {len(results)}/{len(test_data)} in {(time.time()-t0)/60:.1f} min')

# Voluntary has_think rate per condition (this is the GROUND TRUTH for our setup)
print('\\nVoluntary has_think rate per condition:')
by_cond = {}
for r in results:
    by_cond.setdefault(r['condition'], []).append(r['has_think_voluntary'])
for cond in ['none', 'ensemble-gated', 'all-admit', 'random-50']:
    vals = by_cond.get(cond, [])
    if vals:
        rate = sum(vals) / len(vals) * 100
        print(f'  {cond:<20s}: {sum(vals):>3d}/{len(vals)} = {rate:.1f}%')
"""),
    code("""
# 8) Train CoT-Integrity probe inline (sklearn LR on Colab CPU)
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

LAYER_TARGET = 55  # best layer from previous sweep
TOP_K = 50         # capacity from L1-LR sweep

# Build X, y
y = np.array([1 if r['has_think_voluntary'] else 0 for r in results], dtype=int)
print(f'N={len(y)}, voluntary_pos={int(y.sum())} ({y.mean()*100:.1f}%)')

X = np.stack([r['captures'][LAYER_TARGET].squeeze(0).numpy() for r in results], axis=0)
print(f'X shape: {X.shape}')

# Train probe with L1 C=1.0 (best from prior sweep)
sc = StandardScaler()
X_s = np.nan_to_num(sc.fit_transform(X), nan=0.0, posinf=0.0, neginf=0.0)
clf = LogisticRegression(C=1.0, max_iter=5000, penalty='l1', solver='saga', class_weight='balanced')
clf.fit(X_s, y)

# 4-fold CV AUROC
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
cv_aurocs = []
for tr, te in skf.split(X, y):
    sc_cv = StandardScaler()
    Xtr = np.nan_to_num(sc_cv.fit_transform(X[tr]), nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(sc_cv.transform(X[te]), nan=0.0, posinf=0.0, neginf=0.0)
    clf_cv = LogisticRegression(C=1.0, max_iter=5000, penalty='l1', solver='saga', class_weight='balanced')
    clf_cv.fit(Xtr, y[tr])
    cv_aurocs.append(roc_auc_score(y[te], clf_cv.predict_proba(Xte)[:, 1]))
print(f'L{LAYER_TARGET} L1 C=1.0 CV AUROC: {np.mean(cv_aurocs):.3f}')

# Build steering direction (full LR coefficients, normalized)
direction = clf.coef_.flatten() / sc.scale_
direction = np.nan_to_num(direction, nan=0.0, posinf=0.0, neginf=0.0)
norm = np.linalg.norm(direction)
direction /= norm
print(f'Direction norm before normalize: {norm:.3f}')

# Save direction + probe metadata
import json
with open(OUT / 'probe_direction.json', 'w') as f:
    json.dump({
        'layer': LAYER_TARGET,
        'method': 'L1-LR C=1.0 enable_thinking=False',
        'cv_auroc': float(np.mean(cv_aurocs)),
        'cv_aurocs_per_fold': [float(a) for a in cv_aurocs],
        'n_total': len(y),
        'n_pos': int(y.sum()),
    }, f, indent=2)
"""),
    code("""
# 9) Steering experiment — does +α direction make model emit <think> more?
# Pick 8 queries: 4 from "ensemble-gated" (RAG suppressed thinking, low voluntary rate)
#                 4 from "none" (high baseline, control)
import torch

direction_t = torch.from_numpy(direction).to(device).to(torch.bfloat16)

# Build random direction control
rng = np.random.default_rng(2026)
random_dir = rng.standard_normal(X.shape[1])
random_dir /= np.linalg.norm(random_dir)
random_dir_t = torch.from_numpy(random_dir.astype(np.float32)).to(device).to(torch.bfloat16)

# Pick test traces — those where probe predicted thinking but RAG might suppress
# For 'ensemble-gated' condition (RAG present), pick ones where probe says HIGH thinking
# (so steering toward thinking should produce visible behavioral change)
ensemble_results = [r for r in results if r['condition'] == 'ensemble-gated']
none_results = [r for r in results if r['condition'] == 'none']

# Score by probe
ensemble_scores = []
for r in ensemble_results:
    act = r['captures'][LAYER_TARGET].squeeze(0).numpy().reshape(1, -1)
    score = clf.predict_proba(np.nan_to_num(sc.transform(act), nan=0.0))[0, 1]
    ensemble_scores.append((r, float(score)))

# Sort by score desc, pick top 4 (probe predicts thinking but RAG present — interesting test case)
ensemble_scores.sort(key=lambda x: -x[1])
test_targets = [(r, s, 'ensemble-gated') for r, s in ensemble_scores[:4]]

# Pick 4 from 'none' similarly (control — expect baseline thinking already)
none_scores = []
for r in none_results:
    act = r['captures'][LAYER_TARGET].squeeze(0).numpy().reshape(1, -1)
    score = clf.predict_proba(np.nan_to_num(sc.transform(act), nan=0.0))[0, 1]
    none_scores.append((r, float(score)))
none_scores.sort(key=lambda x: -x[1])
test_targets += [(r, s, 'none') for r, s in none_scores[:4]]

print(f'Test targets: {len(test_targets)}')
for r, s, c in test_targets:
    print(f'  [{c:<15s}] probe_score={s:.3f}  voluntary_baseline={r[\"has_think_voluntary\"]}  {r[\"id\"]}')
"""),
    code("""
# 10) Run steering: for each target, generate under α=0, α=+2, α=+5, +random α=2
import torch

ALPHAS_PROBE = [0.0, 2.0, 5.0]
ALPHA_RANDOM = 2.0  # control

def gen_steered(input_ids, alpha, direction_vec):
    state = {'fired': 0}
    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        if state['fired'] >= 1:  # one-time intervention on prefill
            return out
        state['fired'] += 1
        modified = h.clone()
        if modified.dim() == 3:
            modified[:, -1, :] = h[:, -1, :] + alpha * direction_vec
        return (modified,) + out[1:] if is_tuple else modified
    handle = model.model.layers[LAYER_TARGET].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids.to(device), max_new_tokens=24, do_sample=False,
                pad_token_id=tok.eos_token_id, use_cache=True,
            )
        return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=False)
    finally:
        handle.remove()


steering_results = []
for r, score, cond in test_targets:
    msgs = build_messages(r['question'], r.get('memories_used', []), train_pool)
    p_text = prompt_text_no_autothink(msgs)
    input_ids = tok(p_text, return_tensors='pt', add_special_tokens=False).input_ids

    print(f'\\n[{cond}] probe={score:.3f} | {r[\"id\"]}')
    print(f'  baseline voluntary: {r[\"has_think_voluntary\"]}')
    case = {'id': r['id'], 'condition': cond, 'probe_score': score,
            'baseline_voluntary': r['has_think_voluntary']}
    for alpha in ALPHAS_PROBE:
        text = gen_steered(input_ids, alpha, direction_t)
        has_think = '<think>' in text
        case[f'alpha_{alpha}'] = {'has_think': has_think, 'text_60': text.replace('\\n', ' ')[:60]}
        marker = '🟢' if has_think else '⚪'
        print(f'  α=+{alpha:.1f} probe : {marker} has_think={has_think} | {text.replace(chr(10), \" \")[:60]}')
    text_rand = gen_steered(input_ids, ALPHA_RANDOM, random_dir_t)
    has_think_rand = '<think>' in text_rand
    case[f'random_alpha_{ALPHA_RANDOM}'] = {'has_think': has_think_rand, 'text_60': text_rand.replace('\\n', ' ')[:60]}
    print(f'  α=+{ALPHA_RANDOM:.1f} RAND  : {\"🟢\" if has_think_rand else \"⚪\"} has_think={has_think_rand} | {text_rand.replace(chr(10), \" \")[:60]}')
    steering_results.append(case)

# Save
with open(OUT / 'phase8_steering_results.json', 'w') as f:
    json.dump(steering_results, f, indent=2)
print(f'\\nSaved {OUT}/phase8_steering_results.json')
"""),
    code("""
# 11) Verdict — control-token-normalized analysis
# For each test target, compute Δ(has_think rate) vs α=0, separated by condition + by ensemble vs none
import json

# Aggregate
ensemble_cases = [c for c in steering_results if c['condition'] == 'ensemble-gated']
none_cases = [c for c in steering_results if c['condition'] == 'none']

print('=== HAS_THINK RATE BY ALPHA ===')
print(f'{\"condition\":<18s} {\"α=0\":>6s} {\"α=+2\":>7s} {\"α=+5\":>7s} {\"α=+2 RAND\":>11s}')
for cond_name, cases in [('ensemble (RAG)', ensemble_cases), ('none (no RAG)', none_cases)]:
    if not cases: continue
    n = len(cases)
    rates = {}
    for key, label in [('alpha_0.0', 'α=0'), ('alpha_2.0', 'α=+2'), ('alpha_5.0', 'α=+5'),
                       (f'random_alpha_{ALPHA_RANDOM}', 'α=+2 RAND')]:
        rates[label] = sum(1 for c in cases if c[key]['has_think']) / n
    print(f'{cond_name:<18s} {rates[\"α=0\"]:>5.0%} {rates[\"α=+2\"]:>6.0%} {rates[\"α=+5\"]:>6.0%} {rates[\"α=+2 RAND\"]:>10.0%}')

print()

# Causality verdict (focus ensemble — RAG context, where intervention matters most)
if ensemble_cases:
    n = len(ensemble_cases)
    rate_a0 = sum(1 for c in ensemble_cases if c['alpha_0.0']['has_think']) / n
    rate_a2 = sum(1 for c in ensemble_cases if c['alpha_2.0']['has_think']) / n
    rate_a5 = sum(1 for c in ensemble_cases if c['alpha_5.0']['has_think']) / n
    rate_rand = sum(1 for c in ensemble_cases if c[f'random_alpha_{ALPHA_RANDOM}']['has_think']) / n

    delta_2 = (rate_a2 - rate_a0) * 100
    delta_2_vs_rand = (rate_a2 - rate_rand) * 100
    delta_5 = (rate_a5 - rate_a0) * 100

    print(f'=== ENSEMBLE-GATED (RAG context) — causality test ===')
    print(f'Δ has_think α=+2 vs α=0:    {delta_2:+.1f}pp')
    print(f'Δ has_think α=+5 vs α=0:    {delta_5:+.1f}pp')
    print(f'Δ has_think α=+2 vs RAND:   {delta_2_vs_rand:+.1f}pp  ← key control')
    print()
    if delta_2_vs_rand >= 25:
        print(f'🟢 STRONG CAUSAL: probe direction levers thinking emergence (+{delta_2_vs_rand:.0f}pp vs random)')
    elif delta_2_vs_rand >= 10:
        print(f'🟡 WEAK CAUSAL: signal exists but small (+{delta_2_vs_rand:.0f}pp vs random)')
    elif delta_2_vs_rand >= 0:
        print(f'🔵 BORDERLINE: marginal effect (+{delta_2_vs_rand:.0f}pp vs random) — likely noise at N=4')
    else:
        print(f'🔴 NULL or REVERSED: no causal lever found')

# Save aggregate
with open(OUT / 'phase8_verdict.json', 'w') as f:
    json.dump({
        'ensemble_n': len(ensemble_cases),
        'none_n': len(none_cases),
        'ensemble_rates': {a: sum(1 for c in ensemble_cases if c[k]['has_think'])/max(len(ensemble_cases),1)
                           for a, k in [('a0', 'alpha_0.0'), ('a2', 'alpha_2.0'),
                                        ('a5', 'alpha_5.0'), ('a2_rand', f'random_alpha_{ALPHA_RANDOM}')]},
    }, f, indent=2)
print(f'\\nSaved verdict to {OUT}/phase8_verdict.json')
"""),
    md("""
## Decision tree based on cell 11 verdict

| Verdict | Implication | Next step |
|---|---|---|
| 🟢 STRONG CAUSAL (Δrand ≥ +25pp) | Probe levers thinking — game-changer | Build SDK with "boost mode", paper top-tier |
| 🟡 WEAK CAUSAL (+10-25pp) | Lever exists, fragile | Ship correlative SDK, mention boost as future |
| 🔵 BORDERLINE (0-10pp) | Likely noise at N=4 — re-run with more traces | Cheap: increase N to 12 |
| 🔴 NULL or REVERSED | Epiphenomenal like Phase 7 | Ship correlative SDK as planned, paper as honest-negative |

## Sample size caveat

N=4 ensemble cases is SMALL. Even +50pp effect can be coincidence at this N. If verdict is borderline-positive, definitely scale to N=12 (cheap) before claiming causal.

## Next steps after verdict

- Save direction + probe artifacts to HF for SDK
- Update paper draft (eval v6 = causality verdict)
- Decide v0.1 SDK feature scope (detect-only vs detect+boost)
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "colab": {"name": "nb_swebench_v9_phase8_causal_cot.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
