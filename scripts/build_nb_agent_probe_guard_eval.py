"""Generate notebooks/nb_agent_probe_guard_eval.ipynb — Colab eval for v0.1 SDK.

Self-contained Colab notebook that:
  1. Mounts Drive (for nb47b captures + helpers)
  2. Installs openinterp[full] from PyPI v0.3.0
  3. Loads Qwen3.6-27B + tokenizer (~3 min, 50GB VRAM)
  4. Loads nb47b prompts + builds messages (HotpotQA + memory pool)
  5. Validates SDK loads HF artifact via AgentProbeGuard.from_pretrained
  6. Runs end-to-end assess() on N=240 prompts, measures GPU latency + AUROC
  7. Saves results to Drive + prints verdict (matches local eval claim)

Total compute: ~6-8 min after model load (240 × ~50ms forward passes).
Cost: ~$0.10 on RTX 6000 / A100.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_agent_probe_guard_eval.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n",
            "outputs": [], "execution_count": None}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# agent-probe-guard v0.1 — End-to-End Colab Eval

Validates the full pipeline: PyPI install → HF artifact download → SDK loads probes → GPU forward pass with hooks → AUROC matches paper claims → latency under target.

**Compute**: ~6-8 min after model load. Cost ~$0.10 on RTX 6000.

**Outputs** saved to `Drive/openinterp_runs/agent_probe_guard_eval/`:
- `eval_results.json` (per-prompt scores, actions, latencies)
- `eval_summary.json` (aggregate AUROC, distribution, p50/p95/p99)

Reproduces the **local eval verdict**:
- thinking AUROC ≈ 0.855
- capability AUROC ≈ 0.863
- skip 21.2% / escalate 30.0% / proceed 48.8%
- skip→y=0 accuracy 86.3%, proceed→y=1 accuracy 82.1%
"""),

    code("""
# 0) GPU pre-flight + Drive mount
import subprocess
out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print(f'GPU: {out}')
mem_gb = int(out.split(',')[1].strip().split()[0]) / 1024
assert mem_gb >= 47, f'Need >=48GB. Got {mem_gb:.1f}GB. Use RTX 6000 / A100 / L40S.'
print(f'VRAM: {mem_gb:.1f} GB ✓')

from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs'
import os; os.makedirs(DRIVE_ROOT, exist_ok=True)
"""),

    code("""
# 1) Install openinterp v0.3.0 from PyPI + Qwen3.6 deps
!pip install -q --upgrade "openinterp[full]"
!pip install -q transformers safetensors

# Qwen3.6 GDN/standard hybrid attention deps (skip if already installed)
import importlib
try:
    importlib.import_module('fla')
    print('fla already installed')
except ImportError:
    !pip install -q flash-linear-attention causal-conv1d --no-build-isolation 2>&1 | tail -3

import openinterp
print(f'openinterp v{openinterp.__version__}')
print(f'AgentProbeGuard:', openinterp.AgentProbeGuard)
"""),

    code("""
# 2) Load Qwen3.6-27B (bf16, ~3 min on RTX 6000)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen3.6-27B'
print(f'Loading {MODEL} ...')

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

attn_impl = 'flash_attention_2' if torch.cuda.is_available() else 'eager'
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation=attn_impl,
        device_map={'': 0}, trust_remote_code=True,
    )
except Exception as e:
    print(f'flash_attention_2 failed: {e}; falling back to sdpa')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map={'': 0}, trust_remote_code=True,
    )
model.eval()
device = next(model.parameters()).device
print(f'Loaded {sum(p.numel() for p in model.parameters())/1e9:.2f}B params on {device}')
print(f'VRAM used: {torch.cuda.memory_allocated()/1024**3:.1f} GB')
"""),

    code("""
# 3) Load nb47b prompts (assumes you ran nb47b_capture before, OR we synth from HotpotQA)
import json
from pathlib import Path

NB47 = Path(DRIVE_ROOT) / '47_probe_gated_memory'
NB47B = Path(DRIVE_ROOT) / 'nb47b_capture'

if (NB47 / 'test_pool.jsonl').exists() and (NB47 / 'train_pool.jsonl').exists() and (NB47B / 'metadata.json').exists():
    print('Using existing nb47b data on Drive')
    with open(NB47 / 'test_pool.jsonl') as f:
        test_data = [json.loads(line) for line in f]
    with open(NB47 / 'train_pool.jsonl') as f:
        train_pool = {d['id']: d for d in (json.loads(line) for line in f)}
    nb47b_meta = json.loads((NB47B / 'metadata.json').read_text())
    nb47b_records = nb47b_meta['records']
    print(f'  test_data: {len(test_data)} entries')
    print(f'  train_pool: {len(train_pool)} entries')
    print(f'  nb47b_records: {len(nb47b_records)}')
else:
    print('nb47b data not found on Drive. Falling back to HotpotQA validation samples.')
    !pip install -q datasets 2>&1 | tail -1
    from datasets import load_dataset
    ds = load_dataset('hotpot_qa', 'distractor', split='validation', trust_remote_code=True)
    samples = list(ds.select(range(60)))
    test_data = [{
        'id': f'hotpot_{i}',
        'question': s['question'],
        'answer': s['answer'],
        'condition': 'none',
        'memories_used': [],
        'memories_count': 0,
    } for i, s in enumerate(samples)]
    train_pool = {}
    nb47b_records = [{'id': d['id'], 'condition': 'none',
                      'memories_count': 0, 'has_think_v1': False} for d in test_data]
    print(f'  Synthesized {len(test_data)} HotpotQA samples')
"""),

    code("""
# 4) Define build_messages + prompt_text_no_autothink (matches nb47b capture pipeline)
SYSTEM_PROMPT = (
    'You are a helpful assistant that answers questions accurately. '
    'You MAY think step by step before answering if helpful.'
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


# Detect enable_thinking flag
sample_msgs = build_messages(test_data[0]['question'], [], train_pool)
try:
    _ = tok.apply_chat_template(sample_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    USE_FLAG = True
except TypeError:
    USE_FLAG = False
print(f'enable_thinking flag supported: {USE_FLAG}')


def prompt_text_no_autothink(messages):
    if USE_FLAG:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if p.endswith('<think>\\n\\n</think>\\n\\n'):
        return p
    if p.endswith('<think>\\n'):
        return p[:-len('<think>\\n')]
    if p.endswith('<think>'):
        return p[:-len('<think>')]
    return p


sample_text = prompt_text_no_autothink(sample_msgs)
print(f'Sample prompt end: {sample_text[-60:]!r}')
"""),

    code("""
# 5) Load AgentProbeGuard via SDK (pulls from HF dataset caiovicentino1/agent-probe-guard-qwen36-27b)
from openinterp import AgentProbeGuard

guard = AgentProbeGuard.from_pretrained('Qwen/Qwen3.6-27B')
guard.attach(model, tok)
print(repr(guard))
print(f'capability dims (K=10): {guard.capability_dims}')
print(f'thinking dims (K=5): {guard.thinking_dims}')
print(f'thresholds: {guard.thresholds}')
"""),

    code("""
# 6) Smoke test — single assess() on first prompt
import time

sample_msgs = build_messages(test_data[0]['question'], test_data[0].get('memories_used', []), train_pool)
sample_text = prompt_text_no_autothink(sample_msgs)

t0 = time.perf_counter()
decision = guard.assess(prompt_text=sample_text)
elapsed_ms = (time.perf_counter() - t0) * 1000
print(f'Single assess() latency: {elapsed_ms:.1f} ms')
print(f'Decision: {decision.as_dict()}')
"""),

    code("""
# 7) Full eval on all 240 prompts
import numpy as np
import time
from collections import Counter
from sklearn.metrics import roc_auc_score

# Build held-out labels from nb47b
y_v1 = np.array([1 if r['has_think_v1'] else 0 for r in nb47b_records], dtype=int)
test_by_id = {d['id']: d for d in test_data}

results = []
latencies = []
y_seen = []
t_start = time.time()

for i, rec in enumerate(nb47b_records):
    src = test_by_id.get(rec['id'])
    if src is None:
        continue
    msgs = build_messages(src['question'], src.get('memories_used', []), train_pool)
    p_text = prompt_text_no_autothink(msgs)

    t0 = time.perf_counter()
    decision = guard.assess(prompt_text=p_text)
    latencies.append((time.perf_counter() - t0) * 1000)

    results.append({
        'id': rec['id'],
        'condition': rec.get('condition', 'unknown'),
        'has_think_v1': bool(y_v1[i]),
        'action': decision.action,
        'capability': decision.scores['capability'],
        'thinking': decision.scores['thinking'],
        'reason': decision.reason,
    })
    y_seen.append(int(y_v1[i]))

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(nb47b_records) - i - 1)
        print(f'  [{i+1:3d}/{len(nb47b_records)}] elapsed {elapsed:.0f}s, ETA {eta:.0f}s')

print(f'\\nDone {len(results)} prompts in {(time.time()-t_start)/60:.1f} min')
"""),

    code("""
# 8) Compute aggregates + verdict
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score

scores_thk = [r['thinking'] for r in results]
scores_cap = [r['capability'] for r in results]
actions = [r['action'] for r in results]
lat = np.array(latencies)
y_arr = np.array(y_seen)

print('===== END-TO-END EVAL RESULTS =====')
print(f'N = {len(results)}')
print(f'thinking AUROC: {roc_auc_score(y_arr, scores_thk):.4f}  (paper claim ≈ 0.855)')
print(f'capability AUROC: {roc_auc_score(y_arr, scores_cap):.4f}  (paper claim ≈ 0.863)')

print(f'\\nDecision distribution (default thresholds):')
counter = Counter(actions)
n = len(actions)
for a in ('skip', 'escalate', 'proceed'):
    c = counter.get(a, 0)
    print(f'  {a:10s}: {c:>3d}/{n} ({c/n*100:.1f}%)')

# Routing accuracy
sk_idx = [i for i, a in enumerate(actions) if a == 'skip']
pr_idx = [i for i, a in enumerate(actions) if a == 'proceed']
sk_neg = float(np.mean([1 - y_arr[i] for i in sk_idx])) if sk_idx else 0
pr_pos = float(np.mean([y_arr[i] for i in pr_idx])) if pr_idx else 0
print(f'\\nRouting accuracy:')
print(f'  skip → y=0 rate: {sk_neg*100:.1f}%  (paper claim 86.3%)')
print(f'  proceed → y=1 rate: {pr_pos*100:.1f}%  (paper claim 82.1%)')

print(f'\\nFull assess() latency (GPU forward + 2 probes):')
print(f'  p50:  {np.percentile(lat, 50):.1f} ms')
print(f'  p95:  {np.percentile(lat, 95):.1f} ms')
print(f'  p99:  {np.percentile(lat, 99):.1f} ms')
print(f'  mean: {lat.mean():.1f} ms')
print(f'  total: {lat.sum()/1000:.1f} s for N={len(lat)}')

# Verdict
print(f'\\n========== VERDICT ==========')
auroc_thk = roc_auc_score(y_arr, scores_thk)
auroc_cap = roc_auc_score(y_arr, scores_cap)
checks = [
    ('thinking AUROC ≥ 0.80', auroc_thk >= 0.80),
    ('capability AUROC ≥ 0.80', auroc_cap >= 0.80),
    ('skip distribution sane (10-40%)', 0.10 <= counter['skip']/n <= 0.40),
    ('p95 latency < 200ms', np.percentile(lat, 95) < 200),
    ('skip→y=0 accuracy ≥ 75%', sk_neg >= 0.75),
    ('proceed→y=1 accuracy ≥ 70%', pr_pos >= 0.70),
]
for check, passed in checks:
    print(f'  {\"✓\" if passed else \"✗\"} {check}')
all_pass = all(p for _, p in checks)
print(f'\\nResult: {\"🟢 ALL CHECKS PASSED — v0.1 SDK validated end-to-end\" if all_pass else \"🔴 SOME CHECKS FAILED — investigate\"}')
"""),

    code("""
# 9) Save results to Drive
import json, time
from pathlib import Path

OUT = Path(DRIVE_ROOT) / 'agent_probe_guard_eval'
OUT.mkdir(parents=True, exist_ok=True)

with open(OUT / 'eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved {OUT / \"eval_results.json\"} ({len(results)} entries)')

import numpy as np
summary = {
    'sdk_version': openinterp.__version__,
    'artifact_repo': 'caiovicentino1/agent-probe-guard-qwen36-27b',
    'n': len(results),
    'thinking_auroc': float(roc_auc_score(y_arr, scores_thk)),
    'capability_auroc': float(roc_auc_score(y_arr, scores_cap)),
    'decision_dist': {a: counter.get(a, 0) / n for a in ('skip', 'escalate', 'proceed')},
    'skip_neg_rate': sk_neg,
    'proceed_pos_rate': pr_pos,
    'latency_ms': {
        'p50': float(np.percentile(lat, 50)),
        'p95': float(np.percentile(lat, 95)),
        'p99': float(np.percentile(lat, 99)),
        'mean': float(lat.mean()),
    },
    'gpu': out,
    'completed_at_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
}
with open(OUT / 'eval_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Saved {OUT / \"eval_summary.json\"}')

guard.detach()
print('\\nDone. Detached hooks. SDK v0.1 GPU eval complete.')
"""),
]


nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1))
n_code = sum(1 for c in cells if c["cell_type"] == "code")
n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
print(f'Wrote {NB_PATH}')
print(f'Cells: {len(cells)} ({n_code} code, {n_md} md)')
