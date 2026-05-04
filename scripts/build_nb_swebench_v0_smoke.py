"""Generate notebooks/nb_swebench_v0_smoke.ipynb — Phase 0 smoke test for SWE-bench harness V1.

Run from the harness repo root:
    python scripts/build_nb_swebench_v0_smoke.py

The notebook is Colab-targeted (RTX PRO 6000 Blackwell 96GB) and validates G1+G4+G5.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_swebench_v0_smoke.ipynb"


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": src.lstrip("\n").rstrip() + "\n",
        "outputs": [],
        "execution_count": None,
    }


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# SWE-bench Pro Failure Anatomy — Phase 0 Smoke (V1 harness)

Validate the harness V1 end-to-end on **1 SWE-bench Pro instance** with Qwen3.6-27B.

Phase 0 gates checked here:
- **G1** end-to-end: agent runs, ≥1 turn, captures saved, patch produced
- **G4** capture audit: every record in meta has matching tensor of expected shape
- **G5** resource ceiling: ≤5 min wall, ≤50GB VRAM peak

G2 (determinism) and G3 (pass-rate sanity, 20 problems) are Phase 1 scope.

Hardware target: **RTX PRO 6000 Blackwell 96GB** on Colab Pro+ runtime.
"""),
    code("""
# 1) Install deps. transformers 5.x for Qwen3.6 classes (per memory rule).
!pip install -q -U transformers
!pip install -q datasets safetensors pexpect huggingface-hub
# flash-attn is best-effort — fall back to sdpa if it fails to build
!pip install -q flash-attn --no-build-isolation 2>/dev/null && echo 'flash-attn ok' || echo 'flash-attn unavailable, will use sdpa'
"""),
    code("""
# 2) Mount Drive for persistent captures + traces
import os
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs/swebench_v0'
os.makedirs(DRIVE_ROOT, exist_ok=True)
print('Drive root:', DRIVE_ROOT)
"""),
    code("""
# 3) Pull the harness code. Two paths:
#    A) GitHub clone (preferred, after push)
#    B) Drive zip fallback if you uploaded `openinterp-swebench-harness.zip` to Drive
import os, sys, subprocess

HARNESS_PATH = '/content/openinterp-swebench-harness'
GITHUB_URL = 'https://github.com/OpenInterpretability/openinterp-swebench-harness'
DRIVE_ZIP = '/content/drive/MyDrive/openinterp_runs/openinterp-swebench-harness.zip'

if not os.path.exists(HARNESS_PATH):
    if os.path.exists(DRIVE_ZIP):
        subprocess.run(['unzip', '-q', '-o', DRIVE_ZIP, '-d', '/content/'], check=True)
        # Adjust dir name if the zip extracts to a different folder
        if not os.path.exists(HARNESS_PATH):
            for cand in os.listdir('/content/'):
                if 'swebench-harness' in cand:
                    os.rename(f'/content/{cand}', HARNESS_PATH)
                    break
    else:
        subprocess.run(['git', 'clone', GITHUB_URL, HARNESS_PATH], check=True)

sys.path.insert(0, HARNESS_PATH)
print('Harness path:', HARNESS_PATH)
print('Files:', sorted(os.listdir(HARNESS_PATH))[:20])
"""),
    code("""
# 4) Configure paths to use Drive for outputs that should survive runtime restarts
from pathlib import Path
from config import HarnessConfig

cfg = HarnessConfig(
    work_root=Path('/content/work'),                    # ephemeral repo clones
    capture_root=Path(DRIVE_ROOT) / 'captures',         # persistent
    trace_root=Path(DRIVE_ROOT) / 'traces',             # persistent
)
print(cfg)
"""),
    code("""
# 5) Load Qwen3.6-27B in bf16 with flash_attention_2 (sdpa fallback)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen3.6-27B'
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

attn_impl = 'flash_attention_2'
try:
    import flash_attn  # noqa
except ImportError:
    attn_impl = 'sdpa'
print('attn_implementation:', attn_impl)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_impl,
    device_map='auto',
    trust_remote_code=True,
)
model.eval()
print('Model loaded:', sum(p.numel() for p in model.parameters()) / 1e9, 'B params')
print('Device map:', getattr(model, 'hf_device_map', 'cuda'))
"""),
    code("""
# 6) Load SWE-bench Pro and pick 1 instance
from datasets import load_dataset

ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
print('Total instances:', len(ds))
print('Columns:', ds.column_names)

# Pick a known-tractable instance (small, well-known repo). Adjust index after first run.
PICK_IDX = 0
inst = dict(ds[PICK_IDX])
print('Picked:', inst.get('instance_id'), '|', inst.get('repo'), '@', (inst.get('base_commit') or '')[:8])
print('Problem (first 400 chars):')
print('  ' + (inst.get('problem_statement') or '')[:400].replace('\\n', '\\n  '))
"""),
    code("""
# 7) Workdir prep: shallow clone repo at base_commit, apply test_patch if present
import subprocess
from pathlib import Path

def prep_workdir(instance, workdir):
    workdir = Path(workdir)
    if (workdir / '.git').exists():
        return
    repo = instance['repo']
    base = instance['base_commit']
    workdir.mkdir(parents=True, exist_ok=True)
    print(f'Cloning {repo} @ {base[:8]} -> {workdir}')
    # Shallow clone is faster but base_commit may be old; do full clone for safety
    subprocess.run(['git', 'clone', '--quiet', f'https://github.com/{repo}', str(workdir)], check=True)
    subprocess.run(['git', '-c', 'advice.detachedHead=false', 'checkout', '--quiet', base], cwd=str(workdir), check=True)
    test_patch = instance.get('test_patch')
    if test_patch:
        proc = subprocess.run(['git', 'apply', '--allow-empty', '-'], input=test_patch, text=True, cwd=str(workdir))
        if proc.returncode != 0:
            print('  warn: test_patch did not apply cleanly (continuing — agent will work on the issue itself)')
    # Make the agent's diffs anchorable: stage so `git diff HEAD` shows agent changes
    subprocess.run(['git', 'add', '-A'], cwd=str(workdir), check=False)
    subprocess.run(['git', 'commit', '--quiet', '--no-verify', '-m', 'baseline (test_patch applied)', '--allow-empty'], cwd=str(workdir), check=False)

prep_workdir(inst, cfg.work_root / inst['instance_id'])
print('Workdir ready')
"""),
    code("""
# 8) Run the harness on 1 problem (G1)
from runner import Runner
import torch

torch.cuda.reset_peak_memory_stats()
runner = Runner(model=model, tokenizer=tok, config=cfg)
out = runner.run_one(inst, prepare_workdir_fn=prep_workdir)

print('\\n=== Phase 0 Result ===')
for k, v in out.items():
    print(f'  {k}: {v}')
"""),
    code("""
# 9) G4 — capture audit
from instrumentation.capture import audit_captures

audit = audit_captures(out['captures_meta'])
print('Audit:', {k: audit[k] for k in audit if k != 'issues'})
if audit.get('issues'):
    print('Issues:')
    for line in audit['issues'][:10]:
        print('  -', line)
g4 = audit['ok']
print('\\nG4 capture audit:', 'PASS' if g4 else 'FAIL')
"""),
    code("""
# 10) G5 — resource ceiling
import torch

vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
wall_min = out['wall_seconds'] / 60
print(f'Wall: {wall_min:.2f} min  (gate ≤5 min)')
print(f'VRAM peak: {vram_peak_gb:.1f} GB  (gate ≤50 GB)')
g5 = (wall_min <= 5.0) and (vram_peak_gb <= 50.0)
print('G5:', 'PASS' if g5 else 'INFO (single problem; not blocking)')
"""),
    code("""
# 11) G1 — end-to-end roll-up
g1 = (out['n_captures'] > 0) and (out['n_turns'] > 0)

print('\\n=========== Phase 0 Gate Summary ===========')
print(f'G1 end-to-end : {\"PASS\" if g1 else \"FAIL\"}  (n_turns={out[\"n_turns\"]}, n_captures={out[\"n_captures\"]}, reason={out[\"finish_reason\"]})')
print(f'G4 capture    : {\"PASS\" if g4 else \"FAIL\"}')
print(f'G5 resources  : {\"PASS\" if g5 else \"INFO\"}')

assert g1, 'G1 FAILED — debug before scaling.'
assert g4, 'G4 FAILED — capture audit issues.'
print('\\nPhase 0 SMOKE GREEN — proceed to Phase 1 (20 stratified problems).')
"""),
    code("""
# 12) Inspect first turn for sanity (no model output reproduction — just structural)
import json

with open(out['trace_path']) as f:
    trace = json.load(f)
t0 = trace['turns'][0]
print('Turn 0:')
print(f'  prompt_tokens : {t0[\"prompt_tokens\"]}')
print(f'  new_tokens    : {t0[\"new_tokens\"]}')
print(f'  wall_s        : {t0[\"wall_seconds\"]:.1f}')
print(f'  n_capture_steps (forward passes): {t0[\"n_capture_steps\"]}')
print(f'  capture token positions: {t0[\"capture_token_pos\"]}')
print(f'  thinking (first 300 chars):')
think = t0.get('thinking') or '(none)'
print('    ' + think[:300].replace('\\n', '\\n    '))
print(f'  tool calls in turn 0: {[tc[\"name\"] for tc in t0[\"tool_calls\"]]}')
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "colab": {"name": "nb_swebench_v0_smoke.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
