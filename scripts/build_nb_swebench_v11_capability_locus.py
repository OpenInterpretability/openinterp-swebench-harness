"""Generate notebooks/nb_swebench_v11_capability_locus.ipynb — paper-5 capability
locus extension. Tests whether the 4 strongest capability probes from Phase 6
N=99 verdict are causal levers or detection-only.

Sites tested (selected from Phase 6 N=99 verdict, all gap ≥ +0.146):
  1. L43 think_start (AUROC 0.966, gap +0.146) — highest absolute AUROC
  2. L11 think_start (AUROC 0.795, gap +0.194) — earliest, likely input-locked
  3. L31 pre_tool   (AUROC 0.926, gap +0.164) — comparable to Phase 7 L43
  4. L55 pre_tool   (AUROC 0.930, gap +0.171) — late-layer pre-action

Protocol (paper-3 §3 sanity stack + paper-5 protocol):
  - α sweep ∈ {-200, -100, -50, -20, -5, -2, 0, +2, +5, +20, +50, +100, +200}
  - Probe direction (top-K=10 diffmeans) vs random K-matched direction
  - Bidirectional (pushup + pushdown via signed α)
  - Whitespace-stripped flip metric (paper-3 §3.4)
  - Control-token normalization for log-prob shifts (paper-3 §3.2)

Predicted outcomes (causal locus theory):
  🅐 All 4 sites null → categorical-decision-class confirmed at 4 new sites
  🅑 Any site lever → first capability lever, theory refines (locality found?)
  🟡 Mixed → behavior-class taxonomy needs further axis

Compute: 30 prompts × 4 sites × 13 αs × 2 directions ≈ 3120 forwards.
~2-2.5h on RTX 6000 Blackwell. ~$3-4 BRL.

Output: phase11_capability_locus_results.json + per-site verdict.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / "nb_swebench_v11_capability_locus.ipynb"
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
# Phase 11 — Capability Locus Extension (paper-5 closing experiment)

Tests whether the 4 strongest capability probes from Phase 6 N=99 verdict
are causal levers or detection-only.

**4 sites** (all gap ≥ +0.146 in N=99 verdict):
- L43 think_start (AUROC 0.966) — highest AUROC
- L11 think_start (AUROC 0.795, gap +0.194) — earliest, predicted input-locked
- L31 pre_tool (AUROC 0.926)
- L55 pre_tool (AUROC 0.930)

**Predicted outcome (causal locus theory)**: capability is a categorical
decision → all 4 sites epiphenomenal (input-locked, like Phase 8 thinking).
A single lever among the 4 would refine the theory.

**Protocol**: Phase 7 + 8 + Phase 10 stack
- α ∈ {-200, -100, -50, -20, -5, -2, 0, +2, +5, +20, +50, +100, +200}
- Probe direction (top-K=10 diffmeans on Phase 6 patch_generated label) vs random K-matched
- Whitespace-stripped flip metric (paper-3 §3.4 Phase 10 lesson)
- Control-token normalization (paper-3 §3.2)

**Compute**: ~30 prompts × 4 sites × 13 αs × 2 directions ≈ 3120 forwards.
~2-2.5h on RTX 6000.

Drive: `phase11_capability_locus/`. Resume on partial saves every 5 prompts.
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
# 1) Install — fla MUST be present BEFORE model load
!pip install -q -U transformers
!pip install -q datasets safetensors huggingface-hub joblib scikit-learn
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
OUT = Path(DRIVE_ROOT) / 'phase11_capability_locus'
OUT.mkdir(parents=True, exist_ok=True)
print(f'P6: {P6}, OUT: {OUT}')
assert P6.exists() and (P6 / 'phase6_results.json').exists()
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
# 4) Load Phase 6 labels + train probe directions for 4 target sites
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from safetensors.torch import load_file
from sklearn.preprocessing import StandardScaler

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

print('Caching captures...')
cache = {iid: load_capture(iid) for iid in labels}
cache = {k: v for k, v in cache.items() if v is not None}
print(f'Cached {len(cache)}/{len(labels)}')

# 4 target sites from Phase 6 N=99 verdict (all gap >= +0.146)
SITES = [
    {'layer': 43, 'position': 'think_start', 'auroc_n99': 0.966, 'gap': 0.146},
    {'layer': 11, 'position': 'think_start', 'auroc_n99': 0.795, 'gap': 0.194},
    {'layer': 31, 'position': 'pre_tool', 'auroc_n99': 0.926, 'gap': 0.164},
    {'layer': 55, 'position': 'pre_tool', 'auroc_n99': 0.930, 'gap': 0.171},
]

# For each site: train top-K=10 diffmeans probe direction
print('\\nTraining probe directions:')
for s in SITES:
    X, y = [], []
    for iid, lab in labels.items():
        if iid not in cache: continue
        v = site_vec(cache[iid][0], cache[iid][1], s['layer'], s['position'])
        if v is not None:
            X.append(v); y.append(lab)
    X = np.stack(X); y = np.array(y)
    # top-K=10 diffmeans (signed direction)
    diff = X[y==1].mean(axis=0) - X[y==0].mean(axis=0)
    abs_d = np.abs(diff)
    top_idx = np.argsort(-abs_d)[:10]
    direction = np.zeros(X.shape[1], dtype=np.float32)
    direction[top_idx] = diff[top_idx]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    s['direction'] = direction
    s['top_idx'] = top_idx.tolist()
    s['n_train'] = len(X)
    print(f"  L{s['layer']} {s['position']}: dir norm {np.linalg.norm(direction):.4f}, n_train={len(X)}")

# Random K-matched direction (one shared across sites for control)
rng = np.random.default_rng(2026)
random_dir = rng.standard_normal(SITES[0]['direction'].shape[0]).astype(np.float32)
random_dir /= np.linalg.norm(random_dir)
print(f'\\nRandom direction norm: {np.linalg.norm(random_dir):.4f}')
print(f'Cosine(L43-think_start, random): {float(SITES[0]["direction"] @ random_dir):+.4f}')
"""),
    code("""
# 5) Curate test prompts — sample SWE-bench-like prompts (HumanEval as proxy)
# We need a balanced set: ~50% likely-to-solve and ~50% likely-to-fail
# Use HumanEval (easier baseline) + LiveCodeBench-style hard problems
from datasets import load_dataset

print('Loading HumanEval...')
he = load_dataset('openai/openai_humaneval', split='test').shuffle(seed=42).select(range(15))
print('Loading MBPP...')
mbpp = load_dataset('mbpp', 'sanitized', split='test').shuffle(seed=42).select(range(15))

prompts = []
for ex in he:
    prompts.append({'id': f'he_{ex["task_id"]}', 'task': ex['prompt'][:500],
                    'source': 'humaneval'})
for ex in mbpp:
    prompts.append({'id': f'mbpp_{ex["task_id"]}', 'task': ex['prompt'][:500],
                    'source': 'mbpp'})

print(f'Total prompts: {len(prompts)}')
with open(OUT / 'prompts.json', 'w') as f:
    json.dump(prompts, f)
"""),
    code("""
# 6) Build chat prompts + steering function (Phase 8 reuse)
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
    state = {'fired': 0, 'pre_norm': None, 'post_norm': None, 'first_logits': None}
    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        if state['fired'] >= 1: return out
        state['fired'] += 1
        state['pre_norm'] = float(h[:, -1, :].norm().item())
        modified = h.clone()
        if alpha != 0.0:
            modified[:, -1, :] = h[:, -1, :] + alpha * direction_t
        state['post_norm'] = float(modified[:, -1, :].norm().item())
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
    return {
        'pre_norm': state['pre_norm'], 'post_norm': state['post_norm'],
        'new_text': new_text, 'first_logits': state['first_logits'],
    }

control_ids = [tok.encode(t, add_special_tokens=False)[0] for t in CONTROL_TOKENS]
print(f'Control token ids: {dict(zip(CONTROL_TOKENS, control_ids))}')

# Smoke test on first prompt at L43 think_start, α=0
text = build_chat(prompts[0]['task'])
input_ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
sm = steered_gen(input_ids, 43, 0.0,
                 torch.from_numpy(SITES[0]['direction']).to(device).to(torch.bfloat16))
print(f'L43 think_start α=0: norm {sm["pre_norm"]:.2f} → {sm["post_norm"]:.2f}')
print(f'Generated: {sm["new_text"][:120]!r}')
"""),
    code("""
# 7) Run α-sweep on all 4 sites × 30 prompts × 13 αs × 2 directions
# Save partial every 5 prompts.
import time
results = []
random_dir_t = torch.from_numpy(random_dir).to(device).to(torch.bfloat16)
SITE_DIR_TS = {f"L{s['layer']}_{s['position']}":
               torch.from_numpy(s['direction']).to(device).to(torch.bfloat16)
               for s in SITES}

t0 = time.time()
for i, pr in enumerate(prompts):
    text = build_chat(pr['task'])
    input_ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    prompt_data = {'id': pr['id'], 'source': pr['source'], 'sites': {}}

    for s in SITES:
        site_key = f"L{s['layer']}_{s['position']}"
        # Baseline α=0
        base = steered_gen(input_ids, s['layer'], 0.0, SITE_DIR_TS[site_key])
        target_id = int(torch.argmax(base['first_logits']))
        target_str = tok.decode([target_id])

        site_results = {
            'target_token': target_str, 'target_token_id': target_id,
            'baseline_text': base['new_text'], 'pre_norm': base['pre_norm'],
            'sweeps': {'probe': [], 'random': []},
        }

        for alpha in ALPHAS:
            for dir_name, dir_t in [('probe', SITE_DIR_TS[site_key]), ('random', random_dir_t)]:
                r = steered_gen(input_ids, s['layer'], alpha, dir_t)
                lp = torch.log_softmax(r['first_logits'].float(), dim=-1)
                target_lp = float(lp[target_id])
                ctrl_lps = [float(lp[cid]) for cid in control_ids]
                ctrl_mean = sum(ctrl_lps) / len(ctrl_lps)
                site_results['sweeps'][dir_name].append({
                    'alpha': alpha,
                    'pre_norm': r['pre_norm'], 'post_norm': r['post_norm'],
                    'new_text': r['new_text'][:200],
                    'target_logprob': target_lp,
                    'control_mean_logprob': ctrl_mean,
                    'flipped_vs_baseline': r['new_text'].strip() != base['new_text'].strip(),
                })
        prompt_data['sites'][site_key] = site_results

    results.append(prompt_data)
    if (i + 1) % 5 == 0:
        elapsed = (time.time() - t0) / 60
        eta = elapsed / (i + 1) * (len(prompts) - i - 1)
        with open(OUT / 'phase11_partial.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  [{i+1:>3d}/{len(prompts)}] elapsed {elapsed:.1f}min, ETA {eta:.1f}min — partial saved')

with open(OUT / 'phase11_full.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nDone in {(time.time()-t0)/60:.1f} min')
"""),
    code("""
# 8) Verdict tables per site — stripped flip + Δrel
import numpy as np

def summarize(results, site_key, dir_name):
    rows = []
    for alpha in ALPHAS:
        flips, dt, dc = [], [], []
        for r in results:
            sw = r['sites'].get(site_key, {}).get('sweeps', {}).get(dir_name, [])
            base = next((s for s in sw if s['alpha'] == 0.0), None)
            this = next((s for s in sw if s['alpha'] == alpha), None)
            if base and this:
                flips.append(int(this['flipped_vs_baseline']))
                dt.append(this['target_logprob'] - base['target_logprob'])
                dc.append(this['control_mean_logprob'] - base['control_mean_logprob'])
        if not flips:
            continue
        rel = [a - b for a, b in zip(dt, dc)]
        rows.append({
            'alpha': alpha,
            'flip_rate': float(np.mean(flips)),
            'delta_target_mean': float(np.mean(dt)),
            'delta_ctrl_mean': float(np.mean(dc)),
            'delta_rel_mean': float(np.mean(rel)),
        })
    return rows

verdicts = {}
for s in SITES:
    site_key = f"L{s['layer']}_{s['position']}"
    p_rows = {r['alpha']: r for r in summarize(results, site_key, 'probe')}
    r_rows = {r['alpha']: r for r in summarize(results, site_key, 'random')}

    print(f'\\n=== {site_key} (AUROC N=99: {s["auroc_n99"]:.3f}, gap +{s["gap"]:.3f}) ===')
    print(f'{"α":>8} {"probe_flip%":>11} {"random_flip%":>13} {"probe_Δrel":>11} {"random_Δrel":>13}')
    for alpha in ALPHAS:
        p = p_rows.get(alpha, {})
        r = r_rows.get(alpha, {})
        print(f'{alpha:>+8.0f} {p.get("flip_rate", 0)*100:>11.1f} {r.get("flip_rate", 0)*100:>13.1f} '
              f'{p.get("delta_rel_mean", 0):>+11.3f} {r.get("delta_rel_mean", 0):>+13.3f}')

    # Auto-verdict at α=±200
    high = max(p_rows.keys())
    p_hi = p_rows[high]['flip_rate']
    r_hi = r_rows[high]['flip_rate']
    p_drel = abs(p_rows[high]['delta_rel_mean'])
    if p_hi > 0.30 and r_hi < 0.30:
        v = f'🅑 LEVER (probe {p_hi*100:.0f}% > random {r_hi*100:.0f}%)'
    elif p_hi < 0.10 and r_hi < 0.10 and p_drel < 0.10:
        v = f'🅐 EPIPHENOMENAL (both flat, Δrel {p_drel:.3f})'
    elif p_hi > 0.10 and p_drel < 0.10:
        v = f'🟡 SOFTMAX-TEMP (flip without Δrel)'
    elif abs(p_hi - r_hi) < 0.10:
        v = f'🟡 STRUCTURAL FRAGILITY (probe ≈ random at high α)'
    else:
        v = f'🟡 INCONCLUSIVE'
    print(f'Verdict: {v}')
    verdicts[site_key] = {
        'site': s, 'verdict': v,
        'probe_summary': list(p_rows.values()),
        'random_summary': list(r_rows.values()),
    }

with open(OUT / 'phase11_verdict.json', 'w') as f:
    json.dump({'sites': verdicts, 'protocol': 'paper5_causal_locus_protocol.md'}, f, indent=2)
print(f'\\nFull verdict saved to {OUT}/phase11_verdict.json')
"""),
    md("""
## Interpretation map (paper-5 closing)

| Combined | Implication |
|---|---|
| 4/4 epiphenomenal | Capability = categorical decision class, locus in input tokens. Theory survives. |
| 1/4 lever | First capability lever. Investigate which property of that site differs from the other 3. |
| 2-3/4 lever | Theory needs major refinement — capability is partially residual-controllable. |
| Mixed (epiphenomenal at L11 think_start but lever at L55 pre_tool) | Locus is layer-dependent for capability. |

If 4/4 epiphenomenal: paper-5 closes with 4-class taxonomy + 8 probes mapped (FG/RG/L43/L55-thinking + 4 capability sites). Ready for NeurIPS workshop submission Aug 29.
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
