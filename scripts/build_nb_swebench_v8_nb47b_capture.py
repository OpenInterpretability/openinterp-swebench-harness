"""Generate notebooks/nb_swebench_v8_nb47b_capture.ipynb — instrument nb47 with activation capture.

Reuses nb47 v1 question pool (60 HotpotQA questions × 4 RAG conditions = 240 prompts) but
adds capture hooks at L11/L23/L31/L43/L55 to extract internal states. Measures has_think
ourselves via short generation (~5 tokens) so labels are tied to our prompt format.

Output: captures + labels ready for CoT-Integrity probe training (locally on Mac).

Cost: ~5-10 min on RTX 6000 / A100 (240 × 1 forward + 5-token gen each).
Runs in parallel to Phase 6 N=99 — separate Colab session.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_swebench_v8_nb47b_capture.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n", "outputs": [], "execution_count": None}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# nb47b — Instrument RAG/CoT Collapse with Activation Capture

Phase 2 of breakthrough Path 2 (ProbeGated-RAG). nb47 v1 measured outcome (has_think drops 73→50% under RAG) but did NOT capture activations. This notebook re-runs the same 60 HotpotQA questions × 4 RAG conditions (240 prompts) with capture hooks at L11/L23/L31/L43/L55.

Output: 240 captures × 5 layers + has_think labels → train CoT-Integrity probe locally on Mac afterwards.

**Runs in parallel to Phase 6 N=99 trace collection** — separate Colab session, does not interfere.

Cost: ~5-10 min compute on RTX 6000 / A100 / L40S (≥48GB VRAM).
"""),
    code("""
# 0) GPU pre-flight
import subprocess
out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print(f'GPU: {out}')
mem_gb = int(out.split(',')[1].strip().split()[0]) / 1024
print(f'VRAM: {mem_gb:.1f} GB')
assert mem_gb >= 47, f'Need >=48GB. Got {mem_gb:.1f}GB.'
"""),
    code("""
# 1) Install — same stack as Phase 7
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
        raise SystemExit(f'BLOCKING: {pkg} missing — {e}. Restart runtime + rerun.')
"""),
    code("""
# 2) Drive mount
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs'
print(f'Drive root: {DRIVE_ROOT}')
"""),
    code("""
# 3) Load Qwen3.6-27B (single GPU, no offload)
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

# Verify all params on cuda
bad = [(n, str(p.device)) for n, p in model.named_parameters() if p.device.type != 'cuda']
if bad:
    print(f'BAD: {len(bad)} params not on cuda. First 5:')
    for n, d in bad[:5]: print(f'  {n}: {d}')
    raise SystemExit('Aborting — would cause CPU-offload slowdown.')
print(f'OK: all {sum(1 for _ in model.parameters())} params on cuda')
print(f'Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params')
print(f'VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB used')
device = next(model.parameters()).device
"""),
    code("""
# 4) Load nb47 data + understand condition mapping
import json
from pathlib import Path

NB47 = Path(DRIVE_ROOT) / '47_probe_gated_memory'
OUT_DIR = Path(DRIVE_ROOT) / 'nb47b_capture'
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(NB47 / 'test_pool.jsonl') as f:
    test_data = [json.loads(line) for line in f]

with open(NB47 / 'train_pool.jsonl') as f:
    train_pool = {d['id']: d for d in (json.loads(line) for line in f)}

with open(NB47 / 'memory_banks.json') as f:
    banks = json.load(f)

print(f'Test entries: {len(test_data)}, train pool: {len(train_pool)}, banks: {list(banks.keys())}')

# Sample
print(f'\\nSample test entry:')
sample = test_data[0]
for k, v in sample.items():
    if isinstance(v, str) and len(v) > 100:
        print(f'  {k}: {v[:100]}...')
    else:
        print(f'  {k}: {v}')

# Distribution
from collections import Counter
cond_counts = Counter(d['condition'] for d in test_data)
print(f'\\nCondition distribution: {dict(cond_counts)}')
"""),
    code("""
# 5) Build prompts — well-defined format using retrieved memory exemplars
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions accurately. "
    "Think step by step before answering."
)

def build_messages(question, memories_used, train_pool):
    \"\"\"Construct chat messages: system + memory exemplars (Q/A pairs) + question.\"\"\"
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


# Smoke-test prompt building
sample_msgs = build_messages(test_data[10]['question'], test_data[10].get('memories_used', []), train_pool)
prompt_text = tok.apply_chat_template(sample_msgs, tokenize=False, add_generation_prompt=True)
print(f'Sample prompt ({test_data[10][\"condition\"]}):\\n{prompt_text[:500]}...')
print(f'\\n[truncated, total length {len(prompt_text)} chars]')
"""),
    code("""
# 6) Capture forward + 5-token generation for has_think detection
import torch
from safetensors.torch import save_file

CAPTURE_LAYERS = [11, 23, 31, 43, 55]
THINK_TOKEN_IDS = set()

# Find <think> token IDs (Qwen3.6 uses '<think>' as a special-ish marker)
for s in ['<think>', '<think>\\n', ' <think>']:
    ids = tok.encode(s, add_special_tokens=False)
    for tid in ids:
        decoded = tok.decode([tid])
        if 'think' in decoded.lower():
            THINK_TOKEN_IDS.add(tid)
print(f'<think> token ids: {THINK_TOKEN_IDS}')


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
    def reset(self):
        self.captures = {}
    def remove(self):
        for h in self.handles: h.remove()


def measure_with_capture(messages):
    \"\"\"Forward with hooks at last token of prompt; then generate 8 tokens to detect <think>.\"\"\"
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok(prompt_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

    cap = LayerCapture(model, CAPTURE_LAYERS)
    try:
        # Forward + capture at last prompt token
        with torch.no_grad():
            _ = model(input_ids, use_cache=False)
        captured = {L: v.clone() for L, v in cap.captures.items()}
    finally:
        cap.remove()

    # Short generation (no hooks) to detect <think>
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=8, do_sample=False,
            pad_token_id=tok.eos_token_id, use_cache=True,
        )
    new_tokens = out[0, input_ids.shape[1]:].tolist()
    new_text = tok.decode(new_tokens, skip_special_tokens=False)
    has_think = '<think>' in new_text or any(t in THINK_TOKEN_IDS for t in new_tokens)

    return {
        'captures': captured,
        'has_think_pred': bool(has_think),
        'new_tokens_text': new_text,
        'prompt_length': input_ids.shape[1],
    }
"""),
    code("""
# 7) Run all 240 prompts (60 questions × 4 conditions)
import time
from collections import defaultdict

results = []
t0 = time.time()
print(f'Running {len(test_data)} prompts at {time.strftime(\"%H:%M:%S\")}...')

for i, entry in enumerate(test_data):
    msgs = build_messages(entry['question'], entry.get('memories_used', []), train_pool)
    try:
        r = measure_with_capture(msgs)
        result = {
            'id': entry['id'],
            'condition': entry['condition'],
            'memories_count': entry.get('memories_count', 0),
            'has_think_v1': entry['has_think'],          # nb47 v1 label (different prompt format)
            'has_think_v2': r['has_think_pred'],          # our prompt format
            'new_text_v2': r['new_tokens_text'],
            'prompt_length': r['prompt_length'],
            'captures': r['captures'],
        }
        results.append(result)

        if (i+1) % 30 == 0:
            elapsed = time.time() - t0
            eta_min = elapsed / (i+1) * (len(test_data) - i - 1) / 60
            print(f'  [{i+1:3d}/{len(test_data)}] elapsed {elapsed/60:.1f}min, ETA {eta_min:.1f}min')
    except Exception as e:
        print(f'  ERROR on {entry[\"id\"]}/{entry[\"condition\"]}: {type(e).__name__}: {e}')
        torch.cuda.empty_cache()
        continue

print(f'\\nDone: {len(results)}/{len(test_data)} in {(time.time()-t0)/60:.1f} min')
"""),
    code("""
# 8) Quick has_think rate verification — does our format reproduce nb47 finding?
from collections import defaultdict

by_cond_v1 = defaultdict(list)
by_cond_v2 = defaultdict(list)
for r in results:
    by_cond_v1[r['condition']].append(r['has_think_v1'])
    by_cond_v2[r['condition']].append(r['has_think_v2'])

print(f'{\"condition\":<20s} {\"n\":>4s} {\"v1 (nb47)\":>11s} {\"v2 (ours)\":>11s} {\"delta\":>7s}')
for cond in ['none', 'ensemble-gated', 'all-admit', 'random-50']:
    v1 = by_cond_v1.get(cond, [])
    v2 = by_cond_v2.get(cond, [])
    if not v1: continue
    rate_v1 = sum(v1)/len(v1) * 100
    rate_v2 = sum(v2)/len(v2) * 100
    delta = rate_v2 - rate_v1
    print(f'{cond:<20s} {len(v1):>4d} {rate_v1:>10.1f}% {rate_v2:>10.1f}% {delta:>+6.1f}pp')

print('\\nIf v2 (ours) shows similar drop pattern → our prompt reproduces nb47 effect')
print('If v2 collapse pattern differs → prompt format matters; treat v2 as ground truth label')
"""),
    code("""
# 9) Save captures + labels to Drive
import torch
from safetensors.torch import save_file
from pathlib import Path
import json

OUT = Path(DRIVE_ROOT) / 'nb47b_capture'
OUT.mkdir(parents=True, exist_ok=True)

# Collect tensors per layer (each tensor is (n_results, d_model))
layer_tensors = {L: [] for L in CAPTURE_LAYERS}
metadata = []
for r in results:
    metadata.append({
        'id': r['id'],
        'condition': r['condition'],
        'memories_count': r['memories_count'],
        'has_think_v1': r['has_think_v1'],
        'has_think_v2': r['has_think_v2'],
        'new_text_v2': r['new_text_v2'],
        'prompt_length': r['prompt_length'],
    })
    for L in CAPTURE_LAYERS:
        layer_tensors[L].append(r['captures'][L].squeeze(0))

# Stack and save per layer
for L in CAPTURE_LAYERS:
    stacked = torch.stack(layer_tensors[L], dim=0)  # (N, d_model)
    save_file({'activations': stacked}, str(OUT / f'L{L}_pre_gen_activations.safetensors'))
    print(f'Saved L{L}: shape {stacked.shape}')

# Save metadata
with open(OUT / 'metadata.json', 'w') as f:
    json.dump({
        'n_samples': len(results),
        'capture_layers': CAPTURE_LAYERS,
        'capture_position': 'last_prompt_token (pre-generation)',
        'system_prompt': SYSTEM_PROMPT,
        'records': metadata,
    }, f, indent=2)

print(f'\\nSaved to {OUT}')
print(f'Total: {len(results)} captures × {len(CAPTURE_LAYERS)} layers')
print(f'\\nNext: train CoT-Integrity probe locally on Mac (sklearn LR over activations)')
"""),
    md("""
## Output structure

```
nb47b_capture/
├── L11_pre_gen_activations.safetensors  (240 × 5120)
├── L23_pre_gen_activations.safetensors
├── L31_pre_gen_activations.safetensors
├── L43_pre_gen_activations.safetensors
├── L55_pre_gen_activations.safetensors
└── metadata.json  (240 records: id, condition, has_think_v1, has_think_v2, prompt_length)
```

## Next steps (locally on Mac, no GPU needed)

1. Read activations + labels via `pandas` + `safetensors`
2. Methodology sweep (random-feature baseline + capacity sweep + L1-LR + PCA) per layer
3. Identify best (layer, K) for predicting `has_think_v2 (ours)` from prompt activation
4. Goal AUROC ≥ 0.75 on held-out 20% — that's our CoT-Integrity probe v0.1

## Decision tree based on results

| Outcome | Implication |
|---|---|
| AUROC ≥ 0.85 + gap ≥ +0.20 vs random | 🟢 strong signal, build SDK day 3-5 |
| AUROC 0.70–0.85, gap +0.10 to +0.20 | 🟡 moderate, ship as preview SDK |
| AUROC ≤ 0.70 or gap ≤ +0.10 | 🔴 probe fails to detect collapse, pivot to per-token analysis |

The goal: a probe that, given a query + RAG context, predicts whether the LLM will preserve thinking — BEFORE generation. That's the breakthrough piece.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "colab": {"name": "nb_swebench_v8_nb47b_capture.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
