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

    # Cell 4: Load model + HarnessConfig + Runner
    code("""
# A.4) Load Qwen3.6-27B + HarnessConfig pointing at PHASE6C_DIR + Runner
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from config import HarnessConfig
from runner import Runner

MODEL_ID = 'Qwen/Qwen3.6-27B'
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map='auto', trust_remote_code=True,
)
model.eval()
print(f'Loaded. d_model={model.config.hidden_size}, n_layers={len(model.model.layers)}')

cfg = HarnessConfig(
    work_root=Path(PHASE6C_DIR) / 'work',
    capture_root=Path(PHASE6C_DIR) / 'captures',
    trace_root=Path(PHASE6C_DIR) / 'traces',
)
print(f'Config: model={cfg.model}, layers={cfg.capture_layers}, max_turns={cfg.max_turns}')
print(f'  work_root: {cfg.work_root}')
print(f'  capture_root: {cfg.capture_root}')
print(f'  trace_root: {cfg.trace_root}')

runner = Runner(model=model, tokenizer=tok, config=cfg)
print('Runner ready.')
"""),

    # Cell 5: prep_workdir + run_one loop with resume
    code("""
# A.5) Run 100 instances via Runner.run_one with prep_workdir + resume + audit + checkpoint.
# Same pattern as Phase 6 builder (scripts/build_nb_swebench_v6_phase6_scale.py).

import subprocess, json, time, torch
from pathlib import Path
from instrumentation.capture import audit_captures

def prep_workdir(instance, workdir):
    workdir = Path(workdir)
    if (workdir / '.git').exists():
        return
    repo, base = instance['repo'], instance['base_commit']
    workdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--quiet', f'https://github.com/{repo}', str(workdir)], check=True)
    subprocess.run(['git', '-c', 'advice.detachedHead=false', 'checkout', '--quiet', base], cwd=str(workdir), check=True)
    test_patch = instance.get('test_patch')
    if test_patch:
        subprocess.run(['git', 'apply', '--allow-empty', '-'], input=test_patch, text=True, cwd=str(workdir))
    subprocess.run(['git', 'add', '-A'], cwd=str(workdir), check=False)
    subprocess.run(['git', 'commit', '--quiet', '--no-verify', '-m', 'baseline', '--allow-empty'], cwd=str(workdir), check=False)


CHECKPOINT = Path(PHASE6C_DIR) / 'phase6c_results.json'
results: dict = {}
if CHECKPOINT.exists():
    results = json.loads(CHECKPOINT.read_text())
    print(f'Resuming — {len(results)} already done')

t_start = time.time()
for i, inst in enumerate(selected):
    iid = inst['instance_id']
    if iid in results and results[iid].get('finished') is not None:
        continue
    print(f'[{i+1:3d}/{len(selected)}] RUN  {iid[:60]} | {inst[\"repo\"]}')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        out = runner.run_one(inst, prepare_workdir_fn=prep_workdir)
        audit = audit_captures(out['captures_meta'])
        out['audit_ok'] = audit['ok']
        out['vram_peak_gb'] = round(torch.cuda.max_memory_allocated() / (1024**3), 1)
        out['transformers_commit'] = TRANSFORMERS_COMMIT
    except Exception as e:
        out = {
            'instance_id': iid, 'finished': False, 'finish_reason': 'error',
            'error': f'{type(e).__name__}: {e}', 'wall_seconds': 0.0,
            'n_turns': 0, 'n_captures': 0, 'audit_ok': False, 'vram_peak_gb': 0.0,
            'transformers_commit': TRANSFORMERS_COMMIT,
        }
        print(f'  ERROR: {out[\"error\"]}')
    results[iid] = out
    CHECKPOINT.write_text(json.dumps(results, indent=2))
    elapsed_min = (time.time() - t_start) / 60
    avg_min = elapsed_min / max(1, len([r for r in results.values() if r.get('wall_seconds', 0) > 0]))
    eta_min = avg_min * (len(selected) - len(results))
    print(f'  wall={out[\"wall_seconds\"]/60:.1f}min  turns={out[\"n_turns\"]:2d}  caps={out[\"n_captures\"]}  vram={out.get(\"vram_peak_gb\", 0):.1f}GB  patch={out.get(\"patch_n_bytes\", 0)}B | elapsed {elapsed_min:.0f}m ETA {eta_min:.0f}m')

print(f'\\n=== Phase 6c COMPLETE: {len(results)}/{len(selected)} ===')
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
