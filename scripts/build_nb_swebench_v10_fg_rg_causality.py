"""Generate notebooks/nb_swebench_v10_fg_rg_causality.ipynb — paper-5 first
parallel experiment.

Tests whether FabricationGuard (L31 end_of_think) and ReasoningGuard (L55
mid_think) probe directions are causal levers or detection-only — closing
a gap our Paper-5 meta-analysis flagged as "never tested" despite both
probes being used as the DPO reward signal in paper-2.

Protocol = Phase 7 + Phase 8 reused on two new probes:
  - α sweep ∈ {-200, -100, -50, -20, -5, -2, 0, +2, +5, +20, +50, +100, +200}
  - probe direction vs random K-matched direction (control)
  - bidirectional (pushup + pushdown targets)
  - control-token normalization for log-prob shifts
  - structural-rigidity diagnostic at α >> ‖h‖

Predicted outcomes (informative either way):
  🅐 FG/RG also null → 4/4 probes epiphenomenal, paper-5 strong evidence
  🅑 FG/RG lever → first counter-example, theory refines to "task-trained
       probes lever, post-hoc don't"

Total: ~1.5-2h on RTX 6000 Blackwell or A100. ~$3-5 BRL.
Output: phase10_fg_rg_results.json + verdict tables.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / "nb_swebench_v10_fg_rg_causality.ipynb"
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
# Phase 10 — FG/RG Causality Test (Paper-5 first parallel experiment)

Tests whether **FabricationGuard** (L31 end_of_think, AUROC 0.81 HaluEval)
and **ReasoningGuard** (L55 mid_think, AUROC 0.888 GSM8K) probe directions
are causal levers or detection-only.

This closes a gap our paper-5 meta-analysis flagged: both probes were used
as the DPO reward signal in paper-2, but **never tested causally**.

**Predicted outcomes** (both informative):
- 🅐 FG/RG produce α=±200 zero behavioral change with control-token Δrel ≈ 0
  → 4/4 tested probes epiphenomenal, paper-5 strong claim
- 🅑 FG/RG lever (φ ≥ 0.30 vs random) at some α
  → first counter-example, theory refines to "task-reward-trained probes
     lever, post-hoc-trained probes don't"

**Compute**: ~1.5-2h, RTX 6000 Blackwell or A100.

**Drive checkpointing**: every probe × every α saved. Resumable.
"""),
    code("""
# 0) GPU pre-flight + skip-load if model already in scope
import subprocess
out = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
    capture_output=True, text=True
).stdout.strip()
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
OUT = Path(DRIVE_ROOT) / 'phase10_fg_rg_causality'
OUT.mkdir(parents=True, exist_ok=True)
print(f'Output: {OUT}')
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
        MODEL,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map={'': 0},
        trust_remote_code=True,
    )
    model.eval()
    bad = [(n, str(p.device)) for n, p in model.named_parameters()
           if p.device.type != 'cuda']
    if bad:
        raise SystemExit(f'BAD: {len(bad)} params not on cuda')
    print(f'Loaded {sum(p.numel() for p in model.parameters())/1e9:.2f}B params')
    device = next(model.parameters()).device

import torch
print(f'Device: {device}')
print(f'VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB used')
"""),
    code("""
# 4) Load FG and RG probe directions from HuggingFace
from huggingface_hub import hf_hub_download
import joblib
import numpy as np

# FabricationGuard — L31 end_of_think
FG_REPO = 'caiovicentino1/FabricationGuard-linearprobe-qwen36-27b'
RG_REPO = 'caiovicentino1/ReasoningGuard-linearprobe-qwen36-27b'

def load_probe(repo_id, filename='probe.joblib'):
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset')
    except Exception:
        # fallback: model repo
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type='model')
    bundle = joblib.load(path)
    return bundle

fg_bundle = load_probe(FG_REPO)
rg_bundle = load_probe(RG_REPO)
print(f'FG keys: {list(fg_bundle.keys())}')
print(f'RG keys: {list(rg_bundle.keys())}')

# Extract probe direction in raw residual space
def extract_direction(bundle):
    clf = bundle.get('clf') or bundle.get('classifier') or bundle.get('model')
    coef = clf.coef_.ravel().astype(np.float32)
    sc = bundle.get('scaler') or bundle.get('sc')
    if sc is not None:
        coef = coef / sc.scale_.astype(np.float32)
    direction = coef / (np.linalg.norm(coef) + 1e-12)
    return direction

fg_direction = extract_direction(fg_bundle)
rg_direction = extract_direction(rg_bundle)
print(f'FG direction shape: {fg_direction.shape}, norm: {np.linalg.norm(fg_direction):.4f}')
print(f'RG direction shape: {rg_direction.shape}, norm: {np.linalg.norm(rg_direction):.4f}')

# Probe metadata for verdict context
fg_layer = fg_bundle.get('layer', 31)
fg_position = fg_bundle.get('position', 'end_of_think')
rg_layer = rg_bundle.get('layer', 55)
rg_position = rg_bundle.get('position', 'mid_think')
print(f'FG: L{fg_layer} {fg_position}')
print(f'RG: L{rg_layer} {rg_position}')
"""),
    code("""
# 5) Curate prompt sets
# FG: ambiguous factual questions where model might hallucinate (HaluEval-style)
# RG: GSM8K math problems with clear correct answers
import json
from datasets import load_dataset

# HaluEval QA — 25 hallucinate-prone (specific, niche) + 25 grounded (common)
print('Loading HaluEval QA...')
try:
    halu = load_dataset('pminervini/HaluEval', 'qa', split='data').shuffle(seed=42).select(range(120))
except Exception:
    halu = load_dataset('json', data_files={
        'data': 'https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json'
    }, split='data').shuffle(seed=42).select(range(120))

# Format: simple QA prompts
fg_prompts = []
for ex in halu:
    q = ex.get('question') or ex.get('user_query', '')
    if q and len(q) < 200:
        fg_prompts.append({'id': len(fg_prompts), 'question': q})
    if len(fg_prompts) >= 50:
        break
print(f'FG prompts: {len(fg_prompts)}')

# GSM8K — 50 random
print('Loading GSM8K...')
gsm = load_dataset('gsm8k', 'main', split='test').shuffle(seed=42).select(range(50))
rg_prompts = []
for ex in gsm:
    q = ex['question']
    a = ex['answer'].split('####')[-1].strip()
    rg_prompts.append({
        'id': len(rg_prompts),
        'question': q,
        'gold_answer': a,
    })
print(f'RG prompts: {len(rg_prompts)}')

with open(OUT / 'fg_prompts.json', 'w') as f:
    json.dump(fg_prompts, f)
with open(OUT / 'rg_prompts.json', 'w') as f:
    json.dump(rg_prompts, f)
"""),
    code("""
# 6) Build chat prompts + capture baseline activations
import torch

SYSTEM_FG = 'You are a helpful assistant. Answer the question concisely. If you do not know, say so.'
SYSTEM_RG = 'You are a math tutor. Solve the problem step by step. End with: Final answer: <number>.'

def build_chat(system_prompt, user_query):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_query},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,  # let model think — probes were trained on think activations
    )
    return text


def capture_residual(input_ids, layer):
    captures = {}
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captures['h'] = h[:, -1, :].detach().cpu().float().clone()
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(input_ids, use_cache=False)
    finally:
        handle.remove()
    return captures.get('h')


# Smoke test: capture for first FG prompt at L31
sample_text = build_chat(SYSTEM_FG, fg_prompts[0]['question'])
ids = tok(sample_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
h = capture_residual(ids, fg_layer)
print(f'Sample capture L{fg_layer} shape: {h.shape}, norm: {h.norm().item():.2f}')
"""),
    code("""
# 7) Steering function — Phase 7/8 reuse
import torch

ALPHAS = [-200.0, -100.0, -50.0, -20.0, -5.0, -2.0, 0.0,
          2.0, 5.0, 20.0, 50.0, 100.0, 200.0]
GEN_TOKENS = 40

def steered_forward_and_gen(input_ids, layer, alpha, direction_t, gen_tokens=GEN_TOKENS):
    \"\"\"One-shot steering at last position of prefill.
    Returns: dict with generated text + first-token logits + residual norms.
    \"\"\"
    state = {'fired': 0, 'pre_norm': None, 'post_norm': None,
             'first_logits': None}

    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        if state['fired'] >= 1:
            return out
        state['fired'] += 1
        state['pre_norm'] = float(h[:, -1, :].norm().item())
        modified = h.clone()
        if alpha != 0.0:
            modified[:, -1, :] = h[:, -1, :] + alpha * direction_t
        state['post_norm'] = float(modified[:, -1, :].norm().item())
        return (modified,) + out[1:] if is_tuple else modified

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        # Capture first-token logits for log-prob analysis
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
        state['first_logits'] = outputs.logits[0, -1, :].detach().cpu().float().clone()

        # Re-fire hook for generation (need fresh state)
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
        'pre_norm': state['pre_norm'],
        'post_norm': state['post_norm'],
        'delta_norm': (state['post_norm'] or 0) - (state['pre_norm'] or 0),
        'new_text': new_text,
        'first_logits': state['first_logits'],
    }


# Smoke test
res = steered_forward_and_gen(ids, fg_layer, alpha=0.0,
                              direction_t=torch.from_numpy(fg_direction).to(device).to(torch.bfloat16))
print(f'α=0 baseline norm {res[\"pre_norm\"]:.2f} → {res[\"post_norm\"]:.2f}')
print(f'Generated: {res[\"new_text\"][:120]!r}')
"""),
    code("""
# 8) Build random K-matched direction (control)
import numpy as np
import torch

def build_random_direction(d_model, seed=2026):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d_model).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

random_direction = build_random_direction(fg_direction.shape[0])
print(f'Random direction L2: {np.linalg.norm(random_direction):.4f}')
print(f'Cosine(FG, random): {float(fg_direction @ random_direction):+.4f}')
print(f'Cosine(RG, random): {float(rg_direction @ random_direction):+.4f}')
print(f'Cosine(FG, RG): {float(fg_direction @ rg_direction):+.4f}')
"""),
    code("""
# 9) Phase A — FG sweep: 50 prompts × 13 αs × 2 directions = 1300 forward+gen
# But we go 8 prompts (smoke) first to validate before full sweep
import time
import json

CONTROL_TOKENS_FG = ['the', 'a', 'is', 'and', 'I']  # generic, similar baseline log-prob

fg_dir_t = torch.from_numpy(fg_direction).to(device).to(torch.bfloat16)
rand_dir_t = torch.from_numpy(random_direction).to(device).to(torch.bfloat16)

# Get token ids for control tokens
control_ids_fg = [tok.encode(t, add_special_tokens=False)[0] for t in CONTROL_TOKENS_FG]
print(f'FG control token ids: {dict(zip(CONTROL_TOKENS_FG, control_ids_fg))}')

def measure_logprobs(first_logits, target_token_id, control_token_ids):
    \"\"\"Compute log-prob of target + mean of controls. All from same softmax.\"\"\"
    log_probs = torch.log_softmax(first_logits.float(), dim=-1)
    target_lp = float(log_probs[target_token_id])
    control_lps = [float(log_probs[cid]) for cid in control_token_ids]
    return target_lp, sum(control_lps) / len(control_lps), control_lps


# Pick first 8 FG prompts as smoke
SMOKE_N = 8
fg_results = []
t0 = time.time()
for i, pr in enumerate(fg_prompts[:SMOKE_N]):
    text = build_chat(SYSTEM_FG, pr['question'])
    input_ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

    # Baseline α=0 to determine target token (greedy first token)
    base_res = steered_forward_and_gen(input_ids, fg_layer, alpha=0.0, direction_t=fg_dir_t)
    target_token_id = int(torch.argmax(base_res['first_logits']))
    target_token_str = tok.decode([target_token_id])

    prompt_results = {
        'id': pr['id'], 'question': pr['question'][:100],
        'target_token': target_token_str, 'target_token_id': target_token_id,
        'baseline_text': base_res['new_text'],
        'pre_norm': base_res['pre_norm'],
        'sweeps': {'fg_probe': [], 'random': []},
    }

    for alpha in ALPHAS:
        for dir_name, dir_t in [('fg_probe', fg_dir_t), ('random', rand_dir_t)]:
            r = steered_forward_and_gen(input_ids, fg_layer, alpha=alpha, direction_t=dir_t)
            target_lp, ctrl_mean_lp, ctrl_lps = measure_logprobs(
                r['first_logits'], target_token_id, control_ids_fg
            )
            prompt_results['sweeps'][dir_name].append({
                'alpha': alpha,
                'pre_norm': r['pre_norm'], 'post_norm': r['post_norm'],
                'new_text': r['new_text'],
                'target_logprob': target_lp,
                'control_mean_logprob': ctrl_mean_lp,
                'control_logprobs': ctrl_lps,
                'flipped_vs_baseline': r['new_text'] != base_res['new_text'],
            })
    fg_results.append(prompt_results)
    print(f'  FG [{i+1}/{SMOKE_N}] {pr[\"question\"][:60]!r}... target=\"{target_token_str}\"  elapsed {(time.time()-t0)/60:.1f}min')

# Save smoke results
with open(OUT / 'fg_smoke.json', 'w') as f:
    json.dump(fg_results, f, indent=2)
print(f'\\nFG smoke done in {(time.time()-t0)/60:.1f} min, saved to fg_smoke.json')
"""),
    code("""
# 10) Quick verdict on FG smoke — is there ANY signal?
import numpy as np

def summarize_sweep(results, dir_name):
    \"\"\"For each α, compute % flipped + Δrel mean across prompts.\"\"\"
    rows = []
    alphas_seen = sorted({s['alpha'] for r in results for s in r['sweeps'][dir_name]})
    for alpha in alphas_seen:
        flips = []
        deltas_target = []
        deltas_ctrl = []
        # Find baseline (α=0) target lp + ctrl lp per prompt
        for r in results:
            base = next((s for s in r['sweeps'][dir_name] if s['alpha'] == 0.0), None)
            this_alpha = next((s for s in r['sweeps'][dir_name] if s['alpha'] == alpha), None)
            if base and this_alpha:
                flips.append(int(this_alpha['flipped_vs_baseline']))
                deltas_target.append(this_alpha['target_logprob'] - base['target_logprob'])
                deltas_ctrl.append(this_alpha['control_mean_logprob'] - base['control_mean_logprob'])
        if not flips:
            continue
        rel_deltas = [dt - dc for dt, dc in zip(deltas_target, deltas_ctrl)]
        rows.append({
            'alpha': alpha,
            'n': len(flips),
            'flip_rate': float(np.mean(flips)),
            'delta_target_mean': float(np.mean(deltas_target)),
            'delta_ctrl_mean': float(np.mean(deltas_ctrl)),
            'delta_rel_mean': float(np.mean(rel_deltas)),
            'delta_rel_std': float(np.std(rel_deltas)),
        })
    return rows


print('=== FG L31 — probe direction ===')
print(f'{"α":>8} {"flip%":>6} {"Δtarget":>9} {"Δctrl":>9} {"Δrel":>9} {"std":>9}')
for row in summarize_sweep(fg_results, 'fg_probe'):
    print(f'{row[\"alpha\"]:>+8.0f} {row[\"flip_rate\"]*100:>6.1f} '
          f'{row[\"delta_target_mean\"]:>+9.3f} {row[\"delta_ctrl_mean\"]:>+9.3f} '
          f'{row[\"delta_rel_mean\"]:>+9.3f} {row[\"delta_rel_std\"]:>9.3f}')

print('\\n=== FG L31 — random direction (control) ===')
print(f'{"α":>8} {"flip%":>6} {"Δtarget":>9} {"Δctrl":>9} {"Δrel":>9} {"std":>9}')
for row in summarize_sweep(fg_results, 'random'):
    print(f'{row[\"alpha\"]:>+8.0f} {row[\"flip_rate\"]*100:>6.1f} '
          f'{row[\"delta_target_mean\"]:>+9.3f} {row[\"delta_ctrl_mean\"]:>+9.3f} '
          f'{row[\"delta_rel_mean\"]:>+9.3f} {row[\"delta_rel_std\"]:>9.3f}')

print('\\n--- decision rule ---')
print('• If FG probe flip% at α=±200 > 30 AND random flip% < 30 → 🅑 LEVER')
print('• If both flip% ≈ 0 AND |Δrel| < 0.1 across α → 🅐 EPIPHENOMENAL')
print('• If FG flip% > 0 but Δrel ≈ 0 → softmax-temp artifact (paper-3 §6.1)')
"""),
    code("""
# 11) Phase B — RG sweep: same protocol on L55 mid_think
# Note: ReasoningGuard was trained at mid_think position. We steer at last
# prompt position (consistent with Phase 7/8 protocol). If null, future v2
# can repeat at probe-trained position.

CONTROL_TOKENS_RG = ['the', 'we', 'so', 'first', 'let']

rg_dir_t = torch.from_numpy(rg_direction).to(device).to(torch.bfloat16)
control_ids_rg = [tok.encode(t, add_special_tokens=False)[0] for t in CONTROL_TOKENS_RG]
print(f'RG control token ids: {dict(zip(CONTROL_TOKENS_RG, control_ids_rg))}')

rg_results = []
t0 = time.time()
for i, pr in enumerate(rg_prompts[:SMOKE_N]):
    text = build_chat(SYSTEM_RG, pr['question'])
    input_ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

    base_res = steered_forward_and_gen(input_ids, rg_layer, alpha=0.0, direction_t=rg_dir_t)
    target_token_id = int(torch.argmax(base_res['first_logits']))
    target_token_str = tok.decode([target_token_id])

    prompt_results = {
        'id': pr['id'], 'question': pr['question'][:100],
        'gold_answer': pr['gold_answer'],
        'target_token': target_token_str, 'target_token_id': target_token_id,
        'baseline_text': base_res['new_text'],
        'pre_norm': base_res['pre_norm'],
        'sweeps': {'rg_probe': [], 'random': []},
    }

    for alpha in ALPHAS:
        for dir_name, dir_t in [('rg_probe', rg_dir_t), ('random', rand_dir_t)]:
            r = steered_forward_and_gen(input_ids, rg_layer, alpha=alpha, direction_t=dir_t)
            target_lp, ctrl_mean_lp, ctrl_lps = measure_logprobs(
                r['first_logits'], target_token_id, control_ids_rg
            )
            prompt_results['sweeps'][dir_name].append({
                'alpha': alpha,
                'pre_norm': r['pre_norm'], 'post_norm': r['post_norm'],
                'new_text': r['new_text'],
                'target_logprob': target_lp,
                'control_mean_logprob': ctrl_mean_lp,
                'flipped_vs_baseline': r['new_text'] != base_res['new_text'],
            })
    rg_results.append(prompt_results)
    print(f'  RG [{i+1}/{SMOKE_N}] {pr[\"question\"][:60]!r}... target=\"{target_token_str}\"  elapsed {(time.time()-t0)/60:.1f}min')

with open(OUT / 'rg_smoke.json', 'w') as f:
    json.dump(rg_results, f, indent=2)
print(f'\\nRG smoke done in {(time.time()-t0)/60:.1f} min')
"""),
    code("""
# 12) Quick verdict on RG smoke
print('=== RG L55 — probe direction ===')
print(f'{"α":>8} {"flip%":>6} {"Δtarget":>9} {"Δctrl":>9} {"Δrel":>9} {"std":>9}')
for row in summarize_sweep(rg_results, 'rg_probe'):
    print(f'{row[\"alpha\"]:>+8.0f} {row[\"flip_rate\"]*100:>6.1f} '
          f'{row[\"delta_target_mean\"]:>+9.3f} {row[\"delta_ctrl_mean\"]:>+9.3f} '
          f'{row[\"delta_rel_mean\"]:>+9.3f} {row[\"delta_rel_std\"]:>9.3f}')

print('\\n=== RG L55 — random direction (control) ===')
print(f'{"α":>8} {"flip%":>6} {"Δtarget":>9} {"Δctrl":>9} {"Δrel":>9} {"std":>9}')
for row in summarize_sweep(rg_results, 'random'):
    print(f'{row[\"alpha\"]:>+8.0f} {row[\"flip_rate\"]*100:>6.1f} '
          f'{row[\"delta_target_mean\"]:>+9.3f} {row[\"delta_ctrl_mean\"]:>+9.3f} '
          f'{row[\"delta_rel_mean\"]:>+9.3f} {row[\"delta_rel_std\"]:>9.3f}')
"""),
    code("""
# 13) DECISION: scale to full sweep (50 prompts × 13 αs × 2 dirs each)
# Only run full sweep if smoke shows interesting signal.
# Manual decision: review smoke verdict above, then set RUN_FULL = True

RUN_FULL = True   # set False if smoke verdict is clearly flat (skip wasted compute)

if RUN_FULL:
    print('Running full FG sweep (50 prompts)...')
    fg_full = []
    t0 = time.time()
    for i, pr in enumerate(fg_prompts):
        text = build_chat(SYSTEM_FG, pr['question'])
        input_ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
        base_res = steered_forward_and_gen(input_ids, fg_layer, alpha=0.0, direction_t=fg_dir_t)
        target_token_id = int(torch.argmax(base_res['first_logits']))

        prompt_results = {
            'id': pr['id'], 'question': pr['question'][:200],
            'target_token': tok.decode([target_token_id]),
            'target_token_id': target_token_id,
            'baseline_text': base_res['new_text'],
            'sweeps': {'fg_probe': [], 'random': []},
        }
        for alpha in ALPHAS:
            for dir_name, dir_t in [('fg_probe', fg_dir_t), ('random', rand_dir_t)]:
                r = steered_forward_and_gen(input_ids, fg_layer, alpha=alpha, direction_t=dir_t)
                target_lp, ctrl_mean_lp, _ = measure_logprobs(
                    r['first_logits'], target_token_id, control_ids_fg
                )
                prompt_results['sweeps'][dir_name].append({
                    'alpha': alpha,
                    'pre_norm': r['pre_norm'], 'post_norm': r['post_norm'],
                    'new_text': r['new_text'][:200],
                    'target_logprob': target_lp,
                    'control_mean_logprob': ctrl_mean_lp,
                    'flipped_vs_baseline': r['new_text'] != base_res['new_text'],
                })
        fg_full.append(prompt_results)
        if (i + 1) % 10 == 0:
            elapsed = (time.time() - t0) / 60
            eta = elapsed / (i + 1) * (len(fg_prompts) - i - 1)
            with open(OUT / 'fg_full_partial.json', 'w') as f:
                json.dump(fg_full, f, indent=2)
            print(f'  FG [{i+1:>3d}/{len(fg_prompts)}] elapsed {elapsed:.1f}min, ETA {eta:.1f}min — saved partial')

    with open(OUT / 'fg_full.json', 'w') as f:
        json.dump(fg_full, f, indent=2)
    print(f'\\nFG full done in {(time.time()-t0)/60:.1f} min')
else:
    print('Skipping FG full sweep (smoke flat).')
    fg_full = fg_results
"""),
    code("""
# 14) Same for RG full sweep
if RUN_FULL:
    print('Running full RG sweep (50 prompts)...')
    rg_full = []
    t0 = time.time()
    for i, pr in enumerate(rg_prompts):
        text = build_chat(SYSTEM_RG, pr['question'])
        input_ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
        base_res = steered_forward_and_gen(input_ids, rg_layer, alpha=0.0, direction_t=rg_dir_t)
        target_token_id = int(torch.argmax(base_res['first_logits']))

        prompt_results = {
            'id': pr['id'], 'question': pr['question'][:200],
            'gold_answer': pr['gold_answer'],
            'target_token': tok.decode([target_token_id]),
            'target_token_id': target_token_id,
            'baseline_text': base_res['new_text'],
            'sweeps': {'rg_probe': [], 'random': []},
        }
        for alpha in ALPHAS:
            for dir_name, dir_t in [('rg_probe', rg_dir_t), ('random', rand_dir_t)]:
                r = steered_forward_and_gen(input_ids, rg_layer, alpha=alpha, direction_t=dir_t)
                target_lp, ctrl_mean_lp, _ = measure_logprobs(
                    r['first_logits'], target_token_id, control_ids_rg
                )
                prompt_results['sweeps'][dir_name].append({
                    'alpha': alpha,
                    'pre_norm': r['pre_norm'], 'post_norm': r['post_norm'],
                    'new_text': r['new_text'][:200],
                    'target_logprob': target_lp,
                    'control_mean_logprob': ctrl_mean_lp,
                    'flipped_vs_baseline': r['new_text'] != base_res['new_text'],
                })
        rg_full.append(prompt_results)
        if (i + 1) % 10 == 0:
            elapsed = (time.time() - t0) / 60
            eta = elapsed / (i + 1) * (len(rg_prompts) - i - 1)
            with open(OUT / 'rg_full_partial.json', 'w') as f:
                json.dump(rg_full, f, indent=2)
            print(f'  RG [{i+1:>3d}/{len(rg_prompts)}] elapsed {elapsed:.1f}min, ETA {eta:.1f}min — saved partial')

    with open(OUT / 'rg_full.json', 'w') as f:
        json.dump(rg_full, f, indent=2)
    print(f'\\nRG full done in {(time.time()-t0)/60:.1f} min')
else:
    print('Skipping RG full sweep.')
    rg_full = rg_results
"""),
    code("""
# 15) Final verdict tables — full sweep
print('=== FG L31 (FabricationGuard) — full sweep ===')
print(f'{"α":>8} {"probe_flip%":>11} {"random_flip%":>13} {"probe_Δrel":>11} {"random_Δrel":>13}')
fg_probe_rows = {r['alpha']: r for r in summarize_sweep(fg_full, 'fg_probe')}
fg_rand_rows = {r['alpha']: r for r in summarize_sweep(fg_full, 'random')}
for alpha in ALPHAS:
    p = fg_probe_rows.get(alpha, {})
    rd = fg_rand_rows.get(alpha, {})
    print(f'{alpha:>+8.0f} {p.get(\"flip_rate\", 0)*100:>11.1f} {rd.get(\"flip_rate\", 0)*100:>13.1f} '
          f'{p.get(\"delta_rel_mean\", 0):>+11.3f} {rd.get(\"delta_rel_mean\", 0):>+13.3f}')

print('\\n=== RG L55 (ReasoningGuard) — full sweep ===')
print(f'{"α":>8} {"probe_flip%":>11} {"random_flip%":>13} {"probe_Δrel":>11} {"random_Δrel":>13}')
rg_probe_rows = {r['alpha']: r for r in summarize_sweep(rg_full, 'rg_probe')}
rg_rand_rows = {r['alpha']: r for r in summarize_sweep(rg_full, 'random')}
for alpha in ALPHAS:
    p = rg_probe_rows.get(alpha, {})
    rd = rg_rand_rows.get(alpha, {})
    print(f'{alpha:>+8.0f} {p.get(\"flip_rate\", 0)*100:>11.1f} {rd.get(\"flip_rate\", 0)*100:>13.1f} '
          f'{p.get(\"delta_rel_mean\", 0):>+11.3f} {rd.get(\"delta_rel_mean\", 0):>+13.3f}')

# Auto-verdict
def verdict_for(probe_rows, rand_rows, name):
    # At α=±200, probe vs random
    high_a = max(probe_rows.keys())
    p_flip = probe_rows[high_a]['flip_rate']
    r_flip = rand_rows[high_a]['flip_rate']
    p_drel = abs(probe_rows[high_a]['delta_rel_mean'])
    if p_flip > 0.30 and r_flip < 0.30:
        return f'🅑 LEVER (probe flip {p_flip*100:.0f}% > random {r_flip*100:.0f}%)'
    elif p_flip < 0.10 and r_flip < 0.10 and p_drel < 0.10:
        return f'🅐 EPIPHENOMENAL (probe flip {p_flip*100:.0f}% ≈ random {r_flip*100:.0f}%, Δrel {p_drel:.3f})'
    elif p_flip > 0.10 and p_drel < 0.10:
        return f'🟡 SOFTMAX-TEMP (flip without Δrel)'
    else:
        return f'🟡 INCONCLUSIVE — review tables'

print(f'\\nFG verdict: {verdict_for(fg_probe_rows, fg_rand_rows, \"FG\")}')
print(f'RG verdict: {verdict_for(rg_probe_rows, rg_rand_rows, \"RG\")}')

# Save consolidated verdict
final_verdict = {
    'fg_layer': fg_layer, 'fg_position': fg_position,
    'rg_layer': rg_layer, 'rg_position': rg_position,
    'fg_alphas': list(ALPHAS),
    'fg_probe_summary': list(fg_probe_rows.values()),
    'fg_random_summary': list(fg_rand_rows.values()),
    'rg_probe_summary': list(rg_probe_rows.values()),
    'rg_random_summary': list(rg_rand_rows.values()),
    'fg_verdict': verdict_for(fg_probe_rows, fg_rand_rows, 'FG'),
    'rg_verdict': verdict_for(rg_probe_rows, rg_rand_rows, 'RG'),
    'protocol': 'paper5_causal_locus_protocol.md',
    'n_fg': len(fg_full),
    'n_rg': len(rg_full),
}
with open(OUT / 'phase10_verdict.json', 'w') as f:
    json.dump(final_verdict, f, indent=2)
print(f'\\nFinal verdict saved to {OUT}/phase10_verdict.json')
"""),
    md("""
## Interpretation map

| Combined verdict | Implication for paper-5 |
|---|---|
| FG 🅐 + RG 🅐 | 4/4 tested probes epiphenomenal. Strong evidence for "probes detect downstream of locus" thesis. |
| FG 🅑 + RG 🅑 | First counter-examples. Theory refines to "task-reward-trained probes lever; post-hoc-trained don't". Major paper finding. |
| FG 🅑 + RG 🅐 (or vice versa) | Asymmetry — investigate why one trained-as-reward probe levers but the other doesn't. Probably reflects which behavior IS in residual vs in input/template. |
| FG 🟡 + RG 🟡 | Softmax-temperature class for both. Same as Phase 7. Consistent epiphenomenal regime, refined diagnosis. |

## Next steps after this notebook

If 🅐 → write paper-5 with FG/RG as evidence #3 + #4. Run persona-switch as calibrator.

If 🅑 → write counter-example finding. Investigate which property of FG/RG enables levering (training-on-reward? specific-position? layer choice?). May lead to "probe causality predictability" sub-paper.

Either way, this is the first time we test whether the probes used as our DPO reward signal actually have causal authority. Closes a paper-2 + paper-3 gap.
"""),
]

# Build notebook structure
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {NB_PATH}")
print(f"Cells: {len(cells)}")
