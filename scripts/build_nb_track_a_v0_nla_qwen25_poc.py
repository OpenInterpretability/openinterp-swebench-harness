"""Generate notebooks/nb_track_a_v0_nla_qwen25_poc.ipynb — Track A POC.

Validate that the released Anthropic NLA for Qwen2.5-7B-Instruct (kitft/nla-qwen2.5-7b-L20)
produces useful verbalizations BEFORE committing to a full Phase-6-style harness on
Qwen2.5-7B with paper-5 site-partition methodology.

Two-phase POC:

Phase A (in-distribution sanity): 10 chat prompts. Capture L20 residual at the
last user-prompt token. Run AV → text. Run AR → reconstructed activation.
Compute FVE (target: ≥0.5, paper claims 0.752 in-distribution). Print sample
explanations to qualitatively assess confabulation.

Phase B (OOD comparison): 10 agent-format prompts with tool-call schema.
Capture L20 at the same position. Run NLA. Compare FVE distributions and
explanation quality between Phase A and Phase B.

If Phase B FVE drop is severe (>0.2) or explanations are clearly confabulated,
Track A pivots from "apply NLA to agent traces" to either:
  - Negative-result paper (NLA distribution-shift on agent loops)
  - Grant-application path for domain-specific NLA training (Track B)

If Phase B FVE drop is mild (<0.1) and explanations are coherent, proceed
with full Track A harness (Phase 16+).

Compute: ~$5-10 Colab Pro+, ~30-45min wall clock. Single RTX 6000 48GB.

Output: phase16_track_a_poc/{phase_a, phase_b}_{activations, explanations,
reconstructions, fve}.json + verdict.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "notebooks"
    / "nb_track_a_v0_nla_qwen25_poc.ipynb"
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
# Track A POC — Natural Language Autoencoder × Qwen2.5-7B

Validates whether `kitft/nla-qwen2.5-7b-L20` (Anthropic's released NLA at layer
20 of Qwen2.5-7B-Instruct) produces useful verbalizations on:

- **Phase A (in-distribution)**: simple chat prompts matching NLA training
  data (UltraFineWeb + WildChat).
- **Phase B (OOD)**: agent-format prompts with tool-call schema, matching our
  paper-5 SWE-bench Pro capture distribution.

If Phase B shows severe FVE drop or qualitatively confabulated explanations,
Track A pivots to a domain-specific NLA training path or a negative-result
methodology paper. If both phases match paper-claimed FVE 0.752, full Track
A harness build proceeds.

**Compute**: ~$5-10 Colab Pro+, ~30-45min wall clock. Sequential model loading
(target → AV → AR) to fit in 48GB VRAM.

**References**:
- NLA paper: <https://transformer-circuits.pub/2026/nla/>
- HF: `kitft/nla-qwen2.5-7b-L20-av`, `kitft/nla-qwen2.5-7b-L20-ar`
- GitHub: <https://github.com/kitft/natural_language_autoencoders>
- Paper-5 v5: <https://openinterp.org/research/papers/saturation-direction-probe-levers>
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
assert mem_gb >= 23, f'Need >=24GB. Got {mem_gb:.1f}GB. Sequential loading needs RTX 6000 / A40 / A100.'
"""),
    code("""
# 1) Install
!pip install -q -U transformers accelerate safetensors huggingface-hub
import importlib
for pkg in ['transformers', 'accelerate', 'safetensors', 'huggingface_hub']:
    m = importlib.import_module(pkg)
    print(f'  {pkg}: {getattr(m, "__version__", "?")}')
"""),
    code("""
# 2) Drive mount + output dir
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs'
from pathlib import Path
OUT = Path(DRIVE_ROOT) / 'phase16_track_a_poc'
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path('/content/hf_cache')
CACHE.mkdir(exist_ok=True)
print(f'OUT: {OUT}')
print(f'HF cache: {CACHE}')
"""),
    code("""
# 3) Download NLA AV/AR + load NLA config from kitft repo
# AV is embedding-only loadable; we need full AV LM for raw-transformers inference path.
# AR is full LM truncated.
import os
os.environ['HF_HUB_CACHE'] = str(CACHE)
os.environ['HF_HOME'] = str(CACHE)

from huggingface_hub import snapshot_download

AV_REPO = 'kitft/nla-qwen2.5-7b-L20-av'
AR_REPO = 'kitft/nla-qwen2.5-7b-L20-ar'
TARGET_MODEL = 'Qwen/Qwen2.5-7B-Instruct'

print('Downloading AV (Activation Verbalizer) ~16GB...')
av_path = snapshot_download(AV_REPO, cache_dir=str(CACHE))
print(f'  AV at: {av_path}')

print('Downloading AR (Activation Reconstructor) ~10GB...')
ar_path = snapshot_download(AR_REPO, cache_dir=str(CACHE))
print(f'  AR at: {ar_path}')

# nla_meta.yaml in repo defines injection_scale, injection_char, prompt templates.
import yaml
meta_path = Path(av_path) / 'nla_meta.yaml'
if not meta_path.exists():
    # Try AR
    meta_path = Path(ar_path) / 'nla_meta.yaml'
assert meta_path.exists(), f'nla_meta.yaml not found in AV or AR repo'
META = yaml.safe_load(open(meta_path))
print('\\nNLA meta:')
for k, v in META.items():
    if isinstance(v, str) and len(v) > 100:
        print(f'  {k}: {v[:100]}...')
    else:
        print(f'  {k}: {v}')
"""),
    code("""
# 4) Load target Qwen2.5-7B-Instruct + capture activations at L20 on test prompts
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f'Loading target {TARGET_MODEL}...')
tok = AutoTokenizer.from_pretrained(TARGET_MODEL, cache_dir=str(CACHE))
target = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL, dtype=torch.bfloat16,
    attn_implementation='sdpa',  # Qwen2.5 works fine with sdpa, no FLA needed
    device_map={'': 0}, cache_dir=str(CACHE),
)
target.eval()
device = next(target.parameters()).device
print(f'  Loaded {sum(p.numel() for p in target.parameters())/1e9:.2f}B params')
print(f'  d_model: {target.config.hidden_size}, n_layers: {target.config.num_hidden_layers}')
print(f'  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

LAYER = 20  # NLA training layer
assert LAYER < target.config.num_hidden_layers
"""),
    code("""
# 5) Phase A — in-distribution chat prompts (10 prompts matching NLA training data)
PHASE_A_PROMPTS = [
    'What is the capital of France?',
    'Explain the difference between supervised and unsupervised learning.',
    'Write a haiku about autumn leaves.',
    'What are the main causes of climate change?',
    'How does photosynthesis work in plants?',
    'Describe the plot of Romeo and Juliet briefly.',
    'What is recursion in programming?',
    'List three benefits of regular exercise.',
    'How do solar panels generate electricity?',
    'What is the meaning of life?',
]

def capture_l20_activation(model, tok, user_text, layer=LAYER, position='last_user'):
    '''Forward pass, capture residual at layer `layer` at the last user-prompt token.'''
    messages = [{'role': 'user', 'content': user_text}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok(text, return_tensors='pt').input_ids.to(device)

    captured = {}
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        # h shape: [batch, seq_len, d_model]
        # capture last token of input (just before assistant generation starts)
        captured['act'] = h[0, -1, :].detach().clone()
        captured['seq_len'] = h.shape[1]
        captured['d_model'] = h.shape[-1]

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(input_ids, use_cache=False)
    finally:
        handle.remove()
    return captured['act'], captured['seq_len']

# Capture Phase A
phase_a_acts = []
for i, prompt in enumerate(PHASE_A_PROMPTS):
    act, seq_len = capture_l20_activation(target, tok, prompt)
    phase_a_acts.append({'id': f'a{i}', 'prompt': prompt, 'seq_len': seq_len,
                         'act_l2': float(act.float().norm()), 'act': act.cpu().float()})
    print(f'  [a{i}] seq_len={seq_len:>4d}, ‖h‖={phase_a_acts[-1]["act_l2"]:.2f}')

print(f'\\nPhase A captured: {len(phase_a_acts)} activations at L{LAYER}, d={phase_a_acts[0]["act"].shape[0]}')
"""),
    code("""
# 6) Phase B — OOD agent-format prompts with tool-call schema
AGENT_SYSTEM = '''You are a coding assistant with access to tools.

Available tools:
- bash(command: str) — execute shell command
- str_replace_editor(command: str, path: str, ...) — edit files
- finish(summary: str) — terminate

Use tools by emitting <tool_call>{"name": "...", "arguments": {...}}</tool_call>.'''

PHASE_B_PROMPTS = [
    'Find all Python files modified in the last 24 hours and show their sizes.',
    'In the project, locate the file containing the implementation of `LayerNorm` and propose a unit test.',
    'Run the existing test suite, identify failing tests, and produce a patch.',
    'Read README.md, then explore the src/ directory to understand the build pipeline.',
    'There is a memory leak in `processor.py`; instrument it with print statements and identify the leak.',
    'Implement a CLI flag `--verbose` in `main.py` that prints debug info.',
    'Refactor the duplicated code in `utils/parser.py` and `utils/lexer.py` into a shared module.',
    'Find all calls to `requests.get` without a timeout parameter and fix them.',
    'Add type hints to all functions in `models/database.py`.',
    'Investigate why `test_async_handler.py::test_concurrent_writes` is flaky and propose a fix.',
]

phase_b_acts = []
for i, user_prompt in enumerate(PHASE_B_PROMPTS):
    messages = [
        {'role': 'system', 'content': AGENT_SYSTEM},
        {'role': 'user', 'content': user_prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok(text, return_tensors='pt').input_ids.to(device)

    captured = {}
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['act'] = h[0, -1, :].detach().clone()
        captured['seq_len'] = h.shape[1]

    handle = target.model.layers[LAYER].register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = target(input_ids, use_cache=False)
    finally:
        handle.remove()

    phase_b_acts.append({'id': f'b{i}', 'prompt': user_prompt, 'seq_len': captured['seq_len'],
                         'act_l2': float(captured['act'].float().norm()),
                         'act': captured['act'].cpu().float()})
    print(f'  [b{i}] seq_len={captured["seq_len"]:>4d}, ‖h‖={phase_b_acts[-1]["act_l2"]:.2f}')

# Save activations + free target model VRAM
import torch
all_acts_meta = [{k: v for k, v in a.items() if k != 'act'} for a in phase_a_acts + phase_b_acts]
with open(OUT / 'all_activations_meta.json', 'w') as f:
    json.dump(all_acts_meta, f, indent=2)
torch.save({'phase_a': phase_a_acts, 'phase_b': phase_b_acts}, OUT / 'all_activations.pt')

print(f'\\nReleasing target model VRAM...')
del target
torch.cuda.empty_cache()
import gc; gc.collect()
print(f'  VRAM after release: {torch.cuda.memory_allocated()/1024**3:.1f} GB')
"""),
    code("""
# 7) Load AV (Activation Verbalizer) and generate explanations
# Per kitft inference: AV is full LM with embedding injection at marked positions.
# Raw transformers path: replace embedding row at injection token, generate via inputs_embeds.

print('Loading AV (Activation Verbalizer) Qwen2.5 family ~8B...')
av_tok = AutoTokenizer.from_pretrained(av_path)
av = AutoModelForCausalLM.from_pretrained(
    av_path, dtype=torch.bfloat16, attn_implementation='sdpa',
    device_map={'': 0},
)
av.eval()
print(f'  AV loaded: {sum(p.numel() for p in av.parameters())/1e9:.2f}B params')
print(f'  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

# Read injection config from META
INJ_SCALE = META.get('injection_scale', 150.0)
INJ_CHAR = META.get('injection_char', '㈎')  # Qwen NLA uses U+320E
ACTOR_TEMPLATE = META.get('prompt_templates', {}).get('av',
    'Explain what concept is encoded in this activation: {injection_char}')
print(f'\\n  injection_scale: {INJ_SCALE}')
print(f'  injection_char: {INJ_CHAR!r}')
print(f'  actor template: {ACTOR_TEMPLATE[:120]}')

# Verify injection token IDs
inj_ids = av_tok.encode(INJ_CHAR, add_special_tokens=False)
print(f'  injection_char token IDs: {inj_ids}')
assert len(inj_ids) == 1, f'Expected single token for INJ_CHAR; got {inj_ids}'
INJ_ID = inj_ids[0]
"""),
    code("""
# 8) Generate verbalizations for all 20 captured activations
import re

def normalize_activation(v, target_scale=INJ_SCALE):
    '''Match kitft normalize_activation: L2-normalize then rescale.'''
    norm = v.float().norm().clamp_min(1e-12)
    return (v / norm * target_scale).to(v.dtype)

def verbalize(av_model, av_tok, raw_activation, max_new_tokens=200, temperature=1.0):
    '''Run AV: replace embedding at injection token with normalized activation, generate.'''
    content = ACTOR_TEMPLATE.format(injection_char=INJ_CHAR)
    text = av_tok.apply_chat_template(
        [{'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=True,
    )
    input_ids = av_tok(text, return_tensors='pt').input_ids.to(device)
    embed_layer = av_model.get_input_embeddings()
    embeds = embed_layer(input_ids)  # [1, T, d_model_av]

    # Find injection position
    inj_positions = (input_ids[0] == INJ_ID).nonzero(as_tuple=True)[0]
    assert len(inj_positions) >= 1, f'INJ_ID {INJ_ID} not found in tokenized template'
    inj_pos = inj_positions[0].item()

    # Normalize activation and inject
    v_scaled = normalize_activation(raw_activation.to(device).to(torch.bfloat16))
    embeds[0, inj_pos] = v_scaled

    with torch.no_grad():
        out = av_model.generate(
            inputs_embeds=embeds, attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens, do_sample=temperature > 0,
            temperature=temperature, pad_token_id=av_tok.eos_token_id,
        )
    # out is the generated continuation only when using inputs_embeds (no input_ids prefix)
    text_out = av_tok.decode(out[0], skip_special_tokens=True)

    # Extract <explanation>...</explanation> if present
    m = re.search(r'<explanation>\\s*(.*?)\\s*</explanation>', text_out, re.DOTALL)
    return m.group(1).strip() if m else text_out.strip()

import time
explanations = []
for entry in phase_a_acts + phase_b_acts:
    t0 = time.time()
    expl = verbalize(av, av_tok, entry['act'])
    elapsed = time.time() - t0
    explanations.append({'id': entry['id'], 'prompt': entry['prompt'][:80],
                          'explanation': expl, 'gen_seconds': elapsed})
    print(f'  [{entry["id"]}] ({elapsed:.1f}s) {expl[:120]}')

with open(OUT / 'explanations.json', 'w') as f:
    json.dump(explanations, f, indent=2)
print(f'\\nSaved {len(explanations)} explanations.')

# Free AV
del av
torch.cuda.empty_cache()
import gc; gc.collect()
print(f'VRAM after AV release: {torch.cuda.memory_allocated()/1024**3:.1f} GB')
"""),
    code("""
# 9) Load AR (Activation Reconstructor) and reconstruct activations from explanations
print('Loading AR (Activation Reconstructor) ~5B truncated...')
ar_tok = AutoTokenizer.from_pretrained(ar_path)
ar = AutoModelForCausalLM.from_pretrained(
    ar_path, dtype=torch.bfloat16, attn_implementation='sdpa',
    device_map={'': 0},
)
ar.eval()
# Per kitft: AR has lm_head replaced with Identity, final norm stripped, value_head loaded
# For POC simplicity, take last_hidden_state[:,-1,:] as the reconstruction estimate
print(f'  AR loaded: {sum(p.numel() for p in ar.parameters())/1e9:.2f}B params')
print(f'  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

CRITIC_TEMPLATE = META.get('prompt_templates', {}).get('ar',
    'Reconstruct the activation from this explanation: {explanation}')

def reconstruct(ar_model, ar_tok, explanation):
    text = CRITIC_TEMPLATE.format(explanation=explanation)
    input_ids = ar_tok(text, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = ar_model(input_ids, output_hidden_states=True, use_cache=False)
    # Use the AR's final-layer hidden state at last token as reconstruction
    # (AR was trained to produce reconstruction at this position)
    last_h = out.hidden_states[-1][0, -1, :].cpu().float()
    return last_h

reconstructions = []
for entry, expl_entry in zip(phase_a_acts + phase_b_acts, explanations):
    rec = reconstruct(ar, ar_tok, expl_entry['explanation'])
    reconstructions.append({'id': entry['id'], 'reconstruction': rec})

print(f'  Reconstructed {len(reconstructions)} activations.')
"""),
    code("""
# 10) Compute FVE per pair, aggregate, decide verdict
import numpy as np

def fve(original, reconstructed):
    '''Fraction of Variance Explained, normalized scale per kitft scoring.'''
    o = original.float().flatten()
    r = reconstructed.float().flatten()
    # L2-normalize both, scale by sqrt(d_model) per paper
    sd = np.sqrt(len(o))
    o_n = o / o.norm() * sd
    r_n = r / r.norm() * sd
    mse = ((o_n - r_n) ** 2).mean().item()
    cos = float((o_n @ r_n) / (o_n.norm() * r_n.norm()))
    # FVE = 1 - mse / mean_baseline_mse
    # Mean baseline: ||o_n - mean(o_n)||²  for the whole set
    return mse, cos

# Per-entry scores
scores = []
for entry, rec_entry in zip(phase_a_acts + phase_b_acts, reconstructions):
    mse, cos = fve(entry['act'], rec_entry['reconstruction'])
    scores.append({'id': entry['id'], 'mse': mse, 'cos': cos})
    print(f'  [{entry["id"]}] mse={mse:.4f}  cos={cos:+.4f}')

# Mean baseline FVE: compare each reconstruction to the MEAN of all originals
all_originals = torch.stack([e['act'].float() for e in phase_a_acts + phase_b_acts])
mean_o = all_originals.mean(dim=0)
mean_o_norm = (mean_o / mean_o.norm() * np.sqrt(len(mean_o)))
baselines = []
for entry in phase_a_acts + phase_b_acts:
    o_n = (entry['act'].float() / entry['act'].float().norm() * np.sqrt(len(entry['act'])))
    baselines.append(((o_n - mean_o_norm) ** 2).mean().item())
mean_baseline_mse = float(np.mean(baselines))

# FVE per entry vs mean baseline
for s, b in zip(scores, baselines):
    s['fve'] = 1.0 - s['mse'] / b if b > 0 else float('nan')

phase_a_scores = scores[:10]
phase_b_scores = scores[10:]
phase_a_fve = float(np.mean([s['fve'] for s in phase_a_scores]))
phase_b_fve = float(np.mean([s['fve'] for s in phase_b_scores]))
phase_a_cos = float(np.mean([s['cos'] for s in phase_a_scores]))
phase_b_cos = float(np.mean([s['cos'] for s in phase_b_scores]))

print(f'\\n========== TRACK A POC VERDICT ==========')
print(f'Phase A (in-distribution chat) — mean FVE: {phase_a_fve:+.3f}, cos: {phase_a_cos:+.3f}')
print(f'Phase B (OOD agent-format)     — mean FVE: {phase_b_fve:+.3f}, cos: {phase_b_cos:+.3f}')
print(f'Distribution shift Δ FVE: {phase_b_fve - phase_a_fve:+.3f}')
print(f'(Paper claims FVE 0.752 in-distribution on Qwen2.5-7B L20)')
print(f'========================================')

# Verdict logic
if phase_a_fve >= 0.5 and abs(phase_b_fve - phase_a_fve) < 0.15:
    verdict = '🟢 PROCEED with Track A full harness — NLA generalizes to agent-format'
elif phase_a_fve >= 0.5 and (phase_b_fve - phase_a_fve) < -0.15:
    verdict = '🟡 PIVOT — NLA distribution-shift is real; consider negative-result paper OR domain-specific NLA training (Track B grant)'
elif phase_a_fve < 0.3:
    verdict = '🔴 BLOCKED — NLA inference path is broken or POC implementation has bug. Debug before continuing.'
else:
    verdict = '🟡 INCONCLUSIVE — borderline FVE; expand sample size before commit'

print(f'\\nVerdict: {verdict}')

with open(OUT / 'verdict.json', 'w') as f:
    json.dump({
        'phase_a': {'mean_fve': phase_a_fve, 'mean_cos': phase_a_cos, 'scores': phase_a_scores},
        'phase_b': {'mean_fve': phase_b_fve, 'mean_cos': phase_b_cos, 'scores': phase_b_scores},
        'distribution_shift': phase_b_fve - phase_a_fve,
        'verdict': verdict,
        'paper_claim_in_dist_fve': 0.752,
    }, f, indent=2)
print(f'\\nSaved {OUT}/verdict.json')
"""),
    code("""
# 11) Qualitative inspection — print explanations side-by-side with prompts
print('========== PHASE A: in-distribution chat ==========')
for entry, expl in zip(phase_a_acts, explanations[:10]):
    print(f'\\n[{entry["id"]}] PROMPT: {entry["prompt"]}')
    print(f'  EXPLANATION: {expl["explanation"]}')

print('\\n\\n========== PHASE B: OOD agent-format ==========')
for entry, expl in zip(phase_b_acts, explanations[10:]):
    print(f'\\n[{entry["id"]}] PROMPT: {entry["prompt"][:100]}')
    print(f'  EXPLANATION: {expl["explanation"]}')

print('\\n\\nManual qualitative assessment:')
print('  - Are Phase B explanations coherent or confabulated?')
print('  - Do explanations reference tool calls, code, or agent-loop semantics?')
print('  - Is there a noticeable quality gap between Phase A and Phase B?')
"""),
    md("""
## Decision tree after running this notebook

| Phase A FVE | Phase B FVE | Action |
|---|---|---|
| ≥ 0.5 | within 0.15 of A | 🟢 Build full Track A harness (Phase 16+). NLA generalizes to agent format. |
| ≥ 0.5 | drops > 0.15 below A | 🟡 Pivot. Either negative-result paper ("NLA distribution-shift on agent loops") or apply for grant compute to train domain-specific NLA. |
| < 0.5 | n/a | 🔴 Implementation issue. Debug template, injection scale, or attention impl before reading verdict. |

**If 🟢**: proceed to Phase 16 — port the Phase 6 SWE-bench Pro harness to
Qwen2.5-7B, train probes at L20, run α-sweeps, apply NLA at decision-bottleneck
positions, test paper-5 site-partition predictions semantically.

**If 🟡 (most likely)**: write negative-result paper-7 — *"Off-the-shelf NLAs do
not generalize to agent-loop activation distributions: distribution-shift
analysis on Qwen2.5-7B L20"*. Apply LTFF/OpenPhil/TRC for domain-specific NLA
training compute. Track A becomes Track A-prime (grant-funded).

**If 🔴**: debug. Most likely culprits — wrong INJ_CHAR token, wrong
INJ_SCALE, AR template missing, attention impl mismatch.
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
