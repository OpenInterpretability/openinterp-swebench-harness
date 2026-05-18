"""Generate notebooks/nb_kappa_t_scaleup_phase6c.ipynb — scale-up + multi-model batch.

After κ_t v2/v3 GREEN on N=99 Phase 6 traces, this notebook prepares the
2 most critical next steps for paper-grade external replication:

  STAGE A — Phase 6c scale-up on Qwen3.6-27B (continuation: instances 100-200)
    ~4-5h Colab RTX 6000, ~R$25-35
    Uses same harness as Phase 6, deterministic continuation

  STAGE B — Local κ_t v2 analysis on combined N≈200 traces
    Runs locally on Mac CPU (no GPU needed) after STAGE A completes
    Confirms slope finding survives 2× the data

  STAGE C (OPTIONAL) — Multi-model Qwen3.5-7B on subset of Phase 6 instances
    Tests cross-family generalization of κ_t buildup pattern
    ~2-3h Colab RTX 6000, ~R$15-25

Each stage saves to Drive with deterministic resume support. User can run
A → B → (optionally C) in sequence or pause between stages.

Hard rules applied:
  - transformers from main + commit hash pin
  - flash-linear-attention for GDN speed
  - dtype=bfloat16 (not torch_dtype)
  - Drive checkpoint every 10% progress
  - Pre-registered gates from κ_t paper (see paper outline)
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_kappa_t_scaleup_phase6c.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n", "outputs": [], "execution_count": None}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# κ_t Scale-Up Phase 6c — External Replication

After κ_t v2/v3 GREEN on N=99 Phase 6 traces (2026-05-18), this notebook
runs the **scale-up to N=200** required for paper-grade external replication.

## What this notebook does

**STAGE A** — Continuation Phase 6c on Qwen3.6-27B, instances 100-200 (~5h, R$25-35).
Uses same harness as Phase 6. Saves residuals to Drive in same `.safetensors` format.

**STAGE B** — Local κ_t v2 analysis on combined N≈200 (run on Mac CPU after STAGE A).
Re-tests pre-registered gates with 2× data.

**STAGE C** (optional) — Multi-model Qwen3.5-7B on Phase 6 subset (cross-family).

## Pre-registered gates (LOCKED before measurement)

Same as κ_t v2 verified on N=99:
- G1 κ_t AUROC > 0.65 (was 0.677 at N=99)
- G2 gap vs shuffled > 0.10 (was +0.176)
- G3 Mann-Whitney p < 0.05 (was 0.003)
- G4 EARLY κ_t > 0.60 (was 0.572, failed)
- G5 slope p < 0.05 (was 0.0003, headline)

**Pass threshold for paper:** 4/5 gates including G5 slope.

## Compute summary

| Stage | Where | Time | Cost |
|---|---|---|---|
| A. Scale-up | Colab RTX 6000 | 4-5h | R$25-35 |
| B. κ_t local | Mac CPU | 15-20min | R$0 |
| C. Multi-model (opt) | Colab RTX 6000 | 2-3h | R$15-25 |
"""),

    # ===================== STAGE A — Scale-up =====================
    md("""
## STAGE A — Phase 6c Continuation (Qwen3.6-27B, instances 100-200)
"""),

    # Cell 1: Install + transformers commit
    code("""
# A.1) Install + log transformers commit (hard rule: pin per run)
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q accelerate scipy safetensors huggingface_hub datasets
!pip install -q flash-linear-attention causal-conv1d --no-build-isolation 2>&1 | tail -3 || true

import importlib, subprocess, os
TRANSFORMERS_COMMIT = ''
for pkg in ['transformers', 'datasets', 'fla']:
    try:
        m = importlib.import_module(pkg)
        print(f'  {pkg}: {getattr(m, \"__version__\", \"OK\")}')
    except ImportError:
        print(f'  {pkg}: MISSING')

try:
    res = subprocess.run(
        ['git', 'ls-remote', 'https://github.com/huggingface/transformers.git', 'HEAD'],
        capture_output=True, text=True
    )
    if res.returncode == 0 and res.stdout:
        TRANSFORMERS_COMMIT = res.stdout.split()[0]
    print(f'\\n🔒 transformers commit: {TRANSFORMERS_COMMIT}')
except Exception as e:
    print(f'  ls-remote failed: {e}')

print('\\nRESTART RUNTIME if transformers upgraded.')
"""),

    # Cell 2: Drive + clone harness
    code("""
# A.2) Mount Drive + clone harness + verify pre-existing Phase 6 captures
import subprocess, os, json
out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print(f'GPU: {out}')

from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs'

if not os.path.exists('/content/openinterp-swebench-harness'):
    !git clone -q https://github.com/OpenInterpretability/openinterp-swebench-harness.git /content/openinterp-swebench-harness
!cd /content/openinterp-swebench-harness && git pull -q
import sys
sys.path.insert(0, '/content/openinterp-swebench-harness')

PHASE6_DIR = os.path.join(DRIVE_ROOT, 'swebench_v6_phase6')
PHASE6C_DIR = os.path.join(DRIVE_ROOT, 'swebench_v6c_phase6c')
os.makedirs(PHASE6C_DIR, exist_ok=True)
os.makedirs(os.path.join(PHASE6C_DIR, 'captures'), exist_ok=True)
os.makedirs(os.path.join(PHASE6C_DIR, 'traces'), exist_ok=True)

# Read Phase 6 selected_iids to know what was already done
with open(os.path.join(PHASE6_DIR, 'selected_iids.json')) as f:
    phase6_iids = set(json.load(f))
print(f'Phase 6 already covered: {len(phase6_iids)} instances')
print(f'Output dir: {PHASE6C_DIR}')
"""),

    # Cell 3: Load SWE-bench Pro + select continuation instances
    code("""
# A.3) Select next 100 SWE-bench Pro instances (stratified by length × repo)
from datasets import load_dataset
from collections import Counter
import numpy as np, random

ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
print(f'Total SWE-bench Pro: {len(ds)}')

# Filter Python-ish, exclude already-done
ds_list = [row for row in ds if row['instance_id'] not in phase6_iids]
print(f'Remaining (not in Phase 6): {len(ds_list)}')

# Stratify by problem_statement length × repo (same as Phase 6 builder)
def length_bucket(L):
    if L < 1500: return 'short'
    elif L < 3500: return 'mid'
    else: return 'long'

random.seed(43)  # different seed from Phase 6 (was 42)
buckets = {}
for row in ds_list:
    key = (length_bucket(len(row.get('problem_statement') or '')), row.get('repo'))
    buckets.setdefault(key, []).append(row)
for k in buckets:
    random.shuffle(buckets[k])

N_TARGET = 100
selected = []
i = 0
keys = list(buckets.keys())
while len(selected) < N_TARGET and any(buckets[k] for k in keys):
    k = keys[i % len(keys)]
    if buckets[k]:
        selected.append(buckets[k].pop())
    i += 1

print(f'Selected {len(selected)} new instances')
print(f'  Buckets: {Counter(length_bucket(len(x.get(\"problem_statement\") or \"\")) for x in selected)}')
print(f'  Repos: {Counter(x[\"repo\"] for x in selected)}')

# Save selected
with open(os.path.join(PHASE6C_DIR, 'selected_iids.json'), 'w') as f:
    json.dump([x['instance_id'] for x in selected], f, indent=2)
"""),

    # Cell 4: Load model
    code("""
# A.4) Load Qwen3.6-27B
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'Qwen/Qwen3.6-27B'
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
)
model.eval()
print(f'Loaded. d_model={model.config.hidden_size}, n_layers={len(model.model.layers)}')
"""),

    # Cell 5: Run agent loop on selected instances
    code("""
# A.5) Run agent loop on 100 instances. Uses harness AgentLoop + Tap + Capture.
# Saves to PHASE6C_DIR/traces + captures/ (same format as Phase 6).
# Resume-safe: skip already-done instances.

import time, json
from pathlib import Path
from agent.loop import AgentLoop, AgentResult
from agent.prompts import render_problem
from instrumentation.tap import ResidualTap
from instrumentation.capture import CaptureBuffer
from sandbox.exec import BashSession

CONFIG = type('Cfg', (), {
    'model': 'Qwen/Qwen3.6-27B',
    'temperature': 1.0,
    'top_p': 0.95,
    'thinking_mode': True,
    'capture_layers': [11, 23, 31, 43, 55],
    'max_turns': 30,
    'max_context': 32768,
    'max_invalid_tools_in_row': 5,
})()

TRACES_DIR = os.path.join(PHASE6C_DIR, 'traces')
CAPTURES_DIR = os.path.join(PHASE6C_DIR, 'captures')
RESULTS_FILE = os.path.join(PHASE6C_DIR, 'phase6c_results.json')

# Resume support
results = {}
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    print(f'Resumed: {len(results)} done')

# Filter to remaining
remaining = [r for r in selected if r['instance_id'] not in results]
print(f'Running {len(remaining)} new instances')

t_start = time.time()
for i, instance in enumerate(remaining):
    iid = instance['instance_id']
    try:
        instance['__workdir__'] = f'/content/work_p6c/{iid}'
        os.makedirs(instance['__workdir__'], exist_ok=True)
        bash = BashSession(cwd=instance['__workdir__'])
        tap = ResidualTap(model, capture_layers=CONFIG.capture_layers)
        cap = CaptureBuffer()
        loop = AgentLoop(model=model, tokenizer=tok, config=CONFIG,
                         bash_session=bash, tap=tap, capture_buffer=cap,
                         instance_id=iid, seed=42)
        problem_msg = render_problem(instance)
        result = loop.run(problem_msg)

        # Save trace JSON
        trace_path = os.path.join(TRACES_DIR, f'{iid}.json')
        with open(trace_path, 'w') as f:
            json.dump({
                'instance_id': iid,
                'seed': 42,
                'config': {'model': CONFIG.model, 'temperature': CONFIG.temperature,
                           'top_p': CONFIG.top_p, 'thinking_mode': CONFIG.thinking_mode,
                           'capture_layers': CONFIG.capture_layers},
                'finished': result.finished,
                'finish_reason': result.finish_reason,
                'n_turns': len(result.turns),
                'n_captures': sum(t.n_capture_steps for t in result.turns),
                'wall_seconds': time.time() - t_start,
                'error': result.error,
                'turns': [t.__dict__ for t in result.turns],
            }, f, indent=1)

        # Save captures
        cap_path = os.path.join(CAPTURES_DIR, f'{iid}.safetensors')
        meta_path = os.path.join(CAPTURES_DIR, f'{iid}.meta.json')
        cap.save(cap_path, meta_path, instance_id=iid)

        # Record result
        results[iid] = {
            'instance_id': iid,
            'finished': result.finished,
            'finish_reason': result.finish_reason,
            'n_turns': len(result.turns),
            'wall_seconds': time.time() - t_start,
            'transformers_commit': TRANSFORMERS_COMMIT,
        }
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

        elapsed = (time.time() - t_start) / 60
        print(f'  [{i+1}/{len(remaining)}] {iid[:50]} | turns={len(result.turns)} | {elapsed:.1f}min total')

    except Exception as e:
        print(f'  [{i+1}] FAILED {iid[:50]}: {type(e).__name__}: {e}')
        results[iid] = {'error': f'{type(e).__name__}: {e}'}
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)

print(f'\\nDONE. {len(results)} instances total, {(time.time()-t_start)/60:.1f}min')
"""),

    # ===================== STAGE B — Local analysis instructions =====================
    md("""
## STAGE B — Local κ_t analysis (run on Mac after STAGE A completes)

After STAGE A finishes, run on your local Mac:

```bash
cd /Volumes/SSD\\ Major/fish/openinterp-swebench-harness
# Edit scripts/run_kappa_t_v2.py to point at swebench_v6c_phase6c/captures + traces
# OR: combined-mode runner reading from BOTH phase6 AND phase6c
python3 scripts/run_kappa_t_v2.py  # ~15-20min CPU, no GPU
```

Decision rule:
- 4/5 gates pass + slope p < 0.05 → 🟢 paper-grade external replication
- 3/5 → 🟡 partial — investigate which gate(s) failed at scale
- ≤2/5 → 🔴 walk-back v2 finding (was Phase 6-specific)
"""),

    # ===================== STAGE C — Multi-model (optional) =====================
    md("""
## STAGE C (optional) — Multi-Model Qwen3.5-7B Cross-Family

Tests whether κ_t buildup pattern transfers to a different Qwen family
(Qwen3.5 is non-reasoning, Qwen3.6 is reasoning — different training regime).

Caveat: Qwen3.5 doesn't have `<think>` natively. Would capture only `pre_tool`
and `turn_end` positions. Adapt position resolver accordingly.

Run only if Stage B passes (i.e., scale-up confirms finding).
"""),

    code("""
# C.1) Skeleton for Qwen3.5-7B run — UNFINISHED, fill in based on Stage B result
# Uncomment + complete only after deciding to run Stage C
'''
MODEL_ID_C = 'Qwen/Qwen2.5-7B-Instruct'  # or Qwen3.5-7B if available
# ... load model
# ... adapt capture pipeline (no think_end since no native <think>)
# ... run on subset of Phase 6 instances (e.g., 50 most representative)
'''
print('Stage C is optional — only run if Stage B passes (scale-up confirms).')
"""),
]


def build():
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NB_PATH}")
    print(f"  {len(cells)} cells")


if __name__ == "__main__":
    build()
