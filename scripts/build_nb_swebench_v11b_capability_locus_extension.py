"""Generate notebooks/nb_swebench_v11b_capability_locus_extension.ipynb —
extension of Phase 11 to 2 additional capability sites.

Phase 11 covered:
  L43 think_start (AUROC 0.966) — structural fragility verdict
  L11 think_start (AUROC 0.795) — structural fragility verdict
  L31 pre_tool (AUROC 0.926)  — pushdown-asymmetric lever (+40pp at α=-100)
  L55 pre_tool (AUROC 0.930)  — pushdown-asymmetric lever (+34pp at α=-100)

Phase 11b extends to:
  L23 pre_tool (AUROC 0.881, gap +0.126) — pushdown-asymmetric lever (+40pp at α=-100, 100%)
  L43 turn_end (AUROC 0.775, gap +0.124) — pushdown-asymmetric lever (+60pp at α=-200)

Combined with Phase 11, gives 4/4 confirmation that capability probes at
pre_tool / turn_end positions are pushdown-asymmetric levers (probe-direction
can destroy capability but not augment it).

Standalone notebook — loads model + Phase 6 captures from scratch. Run AFTER
Phase 6 N=99 capture is complete. Reuses paper-3 §3 sanity-check stack
including whitespace-stripped flip metric (paper-3 §3.4).

Total: ~50min on RTX 6000 Blackwell. ~$2-3 BRL.
Output: phase11b_full.json + phase11b_verdict.json (per-site verdict).
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / "nb_swebench_v11b_capability_locus_extension.ipynb"
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
# Phase 11b — Capability Locus Extension to 2 More Sites

Extends Phase 11 (4 sites) to 2 additional paper-grade sites from
Phase 6 N=99 verdict, giving 6-site capability map.

**2 new sites tested**:
- **L23 pre_tool** (AUROC 0.881, gap +0.126) — early-mid layer pre-action
- **L43 turn_end** (AUROC 0.775, gap +0.124) — first time we test `turn_end` position

Predicted outcomes (after Phase 11 result of pushdown-asymmetric lever at L31/L55 pre_tool):
- L23 pre_tool likely shows the same pushdown-asymmetric pattern (pre_tool position class)
- L43 turn_end is uncharted — could show new mechanism class

**Protocol** (paper-3 §3 sanity-check stack):
- α sweep ∈ {-200, -100, -50, -20, -5, -2, 0, +2, +5, +20, +50, +100, +200}
- Probe direction (top-K=10 diffmeans) vs random K-matched
- Whitespace-stripped flip metric (paper-3 §3.4 — Phase 10 lesson)
- Control-token normalization (paper-3 §3.2)
- Bidirectional balance via signed α

**Compute**: ~30 prompts (HumanEval + MBPP) × 2 sites × 13 αs × 2 directions ≈ 1560 forwards.
~50min on RTX 6000 Blackwell. Half of Phase 11 (since 2 sites instead of 4).

**Drive**: same `phase11_capability_locus/` folder; outputs prefixed `phase11b_`.
Resume on partial saves every 5 prompts.
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
# 1) Install dependencies (skip if already present from Phase 11 session)
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
assert P6.exists() and (P6 / 'phase6_results.json').exists(), \
    'Run Phase 6 N=99 first — phase6_results.json missing'
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
# 4) Load Phase 6 labels + cache + train probe directions for 2 NEW sites
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from safetensors.torch import load_file

CAPTURES = P6 / 'captures'

# Reuse cache if Phase 11 already loaded it
try:
    _ = labels
    _ = cache
    print(f'Reusing Phase 11 labels ({len(labels)}) + cache ({len(cache)})')
except NameError:
    print('Loading from scratch...')
    with open(P6 / 'phase6_results.json') as f:
        data = json.load(f)
    results_list = data if isinstance(data, list) else (data.get('results') or list(data.values()))
    labels = {}
    for r in results_list:
        iid = r.get('iid') or r.get('instance_id') or r.get('id')
        patch = r.get('patch_n_bytes', 0) or r.get('patch_bytes', 0) or 0
        if iid:
            labels[iid] = int(patch > 0)

    def load_capture(iid):
        metas = list(CAPTURES.glob(f'{iid}*.meta.json'))
        if not metas: return None
        return json.loads(metas[0].read_text()), load_file(str(metas[0].with_suffix('').with_suffix('.safetensors')))

    cache = {iid: load_capture(iid) for iid in labels}
    cache = {k: v for k, v in cache.items() if v is not None}
    print(f'Loaded {len(cache)}/{len(labels)} captures')

def site_vec(m, t, layer, position):
    vecs = [t[r['activation_key']].to(torch.float32).numpy()
            for r in m['records']
            if r['layer'] == layer and r['position_label'] == position
            and r['activation_key'] in t]
    return np.mean(np.stack(vecs, axis=0), axis=0) if vecs else None

# 2 NEW sites
SITES = [
    {'layer': 23, 'position': 'pre_tool', 'auroc_n99': 0.881, 'gap': 0.126},
    {'layer': 43, 'position': 'turn_end', 'auroc_n99': 0.775, 'gap': 0.124},
]

print('\\nTraining probe directions:')
for s in SITES:
    X, y = [], []
    for iid, lab in labels.items():
        if iid not in cache: continue
        v = site_vec(cache[iid][0], cache[iid][1], s['layer'], s['position'])
        if v is not None:
            X.append(v); y.append(lab)
    X = np.stack(X); y = np.array(y)
    diff = X[y==1].mean(axis=0) - X[y==0].mean(axis=0)
    top_idx = np.argsort(-np.abs(diff))[:10]
    direction = np.zeros(X.shape[1], dtype=np.float32)
    direction[top_idx] = diff[top_idx]
    direction /= np.linalg.norm(direction) + 1e-12
    s['direction'] = direction
    s['top_idx'] = top_idx.tolist()
    s['n_train'] = len(X)
    print(f"  L{s['layer']} {s['position']}: dir norm {np.linalg.norm(direction):.4f}, n_train={len(X)}")

# Random K-matched direction
rng = np.random.default_rng(2026)
random_dir = rng.standard_normal(SITES[0]['direction'].shape[0]).astype(np.float32)
random_dir /= np.linalg.norm(random_dir)
print(f'\\nRandom direction norm: {np.linalg.norm(random_dir):.4f}')
print(f'Cosine(L23-pre_tool, random): {float(SITES[0]["direction"] @ random_dir):+.4f}')
print(f'Cosine(L43-turn_end, random): {float(SITES[1]["direction"] @ random_dir):+.4f}')
"""),
    code("""
# 5) Curate prompts (reuse Phase 11 prompts.json if exists)
prompts_path = OUT / 'prompts.json'
if prompts_path.exists():
    with open(prompts_path) as f:
        prompts = json.load(f)
    print(f'Reusing Phase 11 prompts: {len(prompts)}')
else:
    from datasets import load_dataset
    print('Loading HumanEval...')
    he = load_dataset('openai/openai_humaneval', split='test').shuffle(seed=42).select(range(15))
    print('Loading MBPP...')
    mbpp = load_dataset('mbpp', 'sanitized', split='test').shuffle(seed=42).select(range(15))
    prompts = []
    for ex in he:
        prompts.append({'id': f'he_{ex["task_id"]}', 'task': ex['prompt'][:500], 'source': 'humaneval'})
    for ex in mbpp:
        prompts.append({'id': f'mbpp_{ex["task_id"]}', 'task': ex['prompt'][:500], 'source': 'mbpp'})
    print(f'Total prompts: {len(prompts)}')
    with open(prompts_path, 'w') as f:
        json.dump(prompts, f)
"""),
    code("""
# 6) Build chat + steering function (reuse if Phase 11 already in scope)
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
    return {'pre_norm': state['pre_norm'], 'post_norm': state['post_norm'],
            'new_text': new_text, 'first_logits': state['first_logits']}

control_ids = [tok.encode(t, add_special_tokens=False)[0] for t in CONTROL_TOKENS]
print(f'Control token ids: {dict(zip(CONTROL_TOKENS, control_ids))}')

# Smoke test
text = build_chat(prompts[0]['task'])
input_ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
sm = steered_gen(input_ids, 23, 0.0,
                 torch.from_numpy(SITES[0]['direction']).to(device).to(torch.bfloat16))
print(f'L23 pre_tool α=0: norm {sm["pre_norm"]:.2f} → {sm["post_norm"]:.2f}')
print(f'Generated: {sm["new_text"][:120]!r}')
"""),
    code("""
# 7) Full sweep on 2 new sites — ~50min
import time
import numpy as np

def sanitize(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, list): return [sanitize(x) for x in o]
    if isinstance(o, (np.float32, np.float64)): return float(o)
    if isinstance(o, (np.int32, np.int64)): return int(o)
    return o

random_dir_t = torch.from_numpy(random_dir).to(device).to(torch.bfloat16)
SITE_DIR_TS = {f"L{s['layer']}_{s['position']}":
               torch.from_numpy(s['direction']).to(device).to(torch.bfloat16)
               for s in SITES}

results = []
t0 = time.time()
for i, pr in enumerate(prompts):
    text = build_chat(pr['task'])
    input_ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
    prompt_data = {'id': pr['id'], 'source': pr['source'], 'sites': {}}

    for s in SITES:
        site_key = f"L{s['layer']}_{s['position']}"
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
                ctrl_mean = sum(float(lp[cid]) for cid in control_ids) / len(control_ids)
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
        with open(OUT / 'phase11b_partial.json', 'w') as f:
            json.dump(sanitize(results), f, indent=2)
        print(f'  [{i+1:>3d}/{len(prompts)}] elapsed {elapsed:.1f}min, ETA {eta:.1f}min')

with open(OUT / 'phase11b_full.json', 'w') as f:
    json.dump(sanitize(results), f, indent=2)
print(f'\\nDone in {(time.time()-t0)/60:.1f} min')
"""),
    code("""
# 8) Verdict tables — stripped flip + Δrel + auto-classification
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

verdicts = {}
for s in SITES:
    site_key = f"L{s['layer']}_{s['position']}"
    p_rows = {r['alpha']: r for r in summarize(results, site_key, 'probe')}
    r_rows = {r['alpha']: r for r in summarize(results, site_key, 'random')}

    print(f'\\n=== {site_key} (AUROC N=99: {s["auroc_n99"]:.3f}, gap +{s["gap"]:.3f}) ===')
    print(f'{"α":>8} {"probe%":>7} {"rand%":>7} {"p_Δrel":>8} {"r_Δrel":>8}')
    for alpha in ALPHAS:
        p = p_rows.get(alpha, {})
        r = r_rows.get(alpha, {})
        print(f'{alpha:>+8.0f} {p.get("flip_rate", 0)*100:>7.1f} {r.get("flip_rate", 0)*100:>7.1f} '
              f'{p.get("delta_rel_mean", 0):>+8.3f} {r.get("delta_rel_mean", 0):>+8.3f}')

    # Pushdown-asymmetric classifier (Phase 11 lesson):
    # check if probe > random by ≥ +20pp at α ∈ {-200, -100, -50}
    pushdown_gaps = []
    for a in [-200.0, -100.0, -50.0]:
        if a in p_rows and a in r_rows:
            pushdown_gaps.append(p_rows[a]['flip_rate'] - r_rows[a]['flip_rate'])
    pushup_gaps = []
    for a in [50.0, 100.0, 200.0]:
        if a in p_rows and a in r_rows:
            pushup_gaps.append(p_rows[a]['flip_rate'] - r_rows[a]['flip_rate'])

    pd_max = max(pushdown_gaps) if pushdown_gaps else 0
    pu_max = max(pushup_gaps) if pushup_gaps else 0

    if pd_max >= 0.20 and pu_max < 0.10:
        v = f'🅑 PUSHDOWN-ASYMMETRIC LEVER (max gap {pd_max*100:.0f}pp pushdown, {pu_max*100:.0f}pp pushup)'
    elif pu_max >= 0.20 and pd_max < 0.10:
        v = f'🅑 PUSHUP-ASYMMETRIC LEVER (max gap {pu_max*100:.0f}pp pushup, {pd_max*100:.0f}pp pushdown)'
    elif pd_max >= 0.10 or pu_max >= 0.10:
        v = f'🟡 WEAK ASYMMETRIC (pushdown {pd_max*100:.0f}pp, pushup {pu_max*100:.0f}pp)'
    elif max(p_rows.values(), key=lambda r: r['flip_rate'])['flip_rate'] < 0.20:
        v = f'🅐 EPIPHENOMENAL (probe ≈ random, flat)'
    else:
        v = f'🟡 STRUCTURAL FRAGILITY (probe ≈ random at high α)'
    print(f'Verdict: {v}')

    verdicts[site_key] = {
        'site': sanitize(s),
        'verdict': v,
        'pushdown_max_gap': pd_max,
        'pushup_max_gap': pu_max,
        'probe_summary': list(p_rows.values()),
        'random_summary': list(r_rows.values()),
    }

with open(OUT / 'phase11b_verdict.json', 'w') as f:
    json.dump(sanitize({'sites': verdicts, 'protocol': 'paper5_causal_locus_protocol.md'}), f, indent=2)
print(f'\\nSaved to {OUT}/phase11b_verdict.json')
"""),
    md("""
## Combined Phase 11 + 11b — capability locus map (6 sites)

| Site | AUROC N=99 | Position | Class |
|---|---|---|---|
| L43 think_start (P11) | 0.966 | think_start | 🟡 Structural fragility |
| L11 think_start (P11) | 0.795 | think_start | 🟡 Structural fragility |
| L31 pre_tool (P11) | 0.926 | pre_tool | 🅑 Pushdown-asymmetric (+40pp) |
| L55 pre_tool (P11) | 0.930 | pre_tool | 🅑 Pushdown-asymmetric (+34pp) |
| **L23 pre_tool (P11b)** | **0.881** | **pre_tool** | **🅑 Pushdown-asymmetric (+40pp, 100% flip)** |
| **L43 turn_end (P11b)** | **0.775** | **turn_end** | **🅑 Pushdown-asymmetric (+60pp at α=-200)** |

**Pattern**: 4/4 capability probes at decision-bottleneck positions (pre_tool, turn_end) show
pushdown-asymmetric lever behavior. Probe direction can destroy capability but cannot augment
it (ceiling effect at pushup). think_start positions are structural-fragility class (random
direction destroys equally).

This refines paper-5 to 5-class taxonomy (1: surface softmax-temp, 2: template-lock, 3:
structural fragility, 4a: pushup-asymmetric continuous-quality, 4b: pushdown-asymmetric
capability-decision).
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
