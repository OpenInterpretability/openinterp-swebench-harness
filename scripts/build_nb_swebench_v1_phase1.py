"""Generate notebooks/nb_swebench_v1_phase1.ipynb — Phase 1 of failure anatomy.

20 stratified Python-only SWE-bench Pro problems. Validates G2 (determinism),
G3 (pass-rate sanity), and re-checks G1+G4+G5 at scale. Resume-on-disconnect
via per-problem Drive checkpoint.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_swebench_v1_phase1.ipynb"


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
# SWE-bench Pro Failure Anatomy — Phase 1 (20 stratified Python)

After Phase 0 SMOKE green (G1 + G4), this notebook scales to **20 stratified Python-only
problems** to validate:

- **G2** determinism — same problem 2× same seed → first 5 turns byte-equal
- **G3** pass-rate sanity — soft pass (agent finished + non-empty patch) ∈ [30%, 60%]
- **G1/G4/G5** reconfirmed at scale

Wall-time budget: ~4 hours with `fla + causal-conv1d` installed (GDN fast path).

Resume-on-disconnect: per-problem checkpoint to Drive. Re-running this notebook
skips problems already completed.

Hardware target: **RTX PRO 6000 Blackwell 96GB** Colab Pro+.
"""),
    code("""
# 1) Install — transformers 5.x + GDN fast-path libs (fla + causal-conv1d) + harness deps
!pip install -q -U transformers
!pip install -q datasets safetensors pexpect huggingface-hub
!pip install -q flash-linear-attention causal-conv1d --no-build-isolation 2>&1 | tail -3
!pip install -q flash-attn --no-build-isolation 2>/dev/null && echo 'flash-attn ok' || echo 'flash-attn unavailable'
"""),
    code("""
# 2) Mount Drive — captures + traces + checkpoint persist across runs
import os
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs/swebench_v1_phase1'
os.makedirs(DRIVE_ROOT, exist_ok=True)
print('Drive root:', DRIVE_ROOT)
"""),
    code("""
# 3) Pull latest harness from GitHub
import os, sys, subprocess
HARNESS_PATH = '/content/openinterp-swebench-harness'
GITHUB_URL = 'https://github.com/OpenInterpretability/openinterp-swebench-harness'
if os.path.exists(HARNESS_PATH):
    subprocess.run(['git', '-C', HARNESS_PATH, 'pull', '--quiet'], check=False)
else:
    subprocess.run(['git', 'clone', GITHUB_URL, HARNESS_PATH], check=True)
sys.path.insert(0, HARNESS_PATH)
print('Harness HEAD:', subprocess.run(['git', '-C', HARNESS_PATH, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip())
"""),
    code("""
# 4) Config — Phase 1 defaults (max_turns=30, bash_max_output=32k come from config.py)
from pathlib import Path
from config import HarnessConfig

cfg = HarnessConfig(
    work_root=Path('/content/work'),
    capture_root=Path(DRIVE_ROOT) / 'captures',
    trace_root=Path(DRIVE_ROOT) / 'traces',
)
print(f'max_turns={cfg.max_turns}  bash_max_output={cfg.bash_max_output_bytes}')
print(f'capture_layers={cfg.capture_layers}')
"""),
    code("""
# 5) Load Qwen3.6-27B with flash-attn + GDN fast path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen3.6-27B'
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

attn_impl = 'flash_attention_2'
try:
    import flash_attn  # noqa
except ImportError:
    attn_impl = 'sdpa'

# Verify GDN fast path is available
try:
    import fla, causal_conv1d  # noqa
    print('GDN fast path: AVAILABLE (fla + causal-conv1d)')
except ImportError as e:
    print(f'WARN GDN fast path missing: {e} — expect ~5-10x slower')

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    attn_implementation=attn_impl,
    device_map='auto',
    trust_remote_code=True,
)
model.eval()
print(f'Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params, attn={attn_impl}')
"""),
    code("""
# 6) Load SWE-bench Pro and stratified-sample 20 Python-only problems
from datasets import load_dataset
import random

ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
py = ds.filter(lambda x: x.get('repo_language') == 'Python')
print(f'Python instances: {len(py)} / {len(ds)} total')

# Stratify by problem_statement length: 3 buckets (short, medium, long)
import numpy as np
lengths = np.array([len(x.get('problem_statement') or '') for x in py])
q33, q66 = float(np.quantile(lengths, 0.33)), float(np.quantile(lengths, 0.66))
print(f'Length quartiles: q33={int(q33)} q66={int(q66)}')

short_idx = [i for i, L in enumerate(lengths) if L < q33]
med_idx   = [i for i, L in enumerate(lengths) if q33 <= L < q66]
long_idx  = [i for i, L in enumerate(lengths) if L >= q66]

PHASE1_SEED = 42
rng = random.Random(PHASE1_SEED)
rng.shuffle(short_idx); rng.shuffle(med_idx); rng.shuffle(long_idx)

picked_idx = short_idx[:7] + med_idx[:7] + long_idx[:6]
selected = [dict(py[i]) for i in picked_idx]

print(f'\\nSelected {len(selected)} problems:')
for i, inst in enumerate(selected):
    print(f"  [{i:2d}] {inst['instance_id'][:60]:60s} | {inst['repo']:30s} | len={len(inst.get('problem_statement') or ''):5d}")
"""),
    code("""
# 7) Workdir preparation
import subprocess
from pathlib import Path

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
        proc = subprocess.run(['git', 'apply', '--allow-empty', '-'], input=test_patch, text=True, cwd=str(workdir))
        if proc.returncode != 0:
            print(f'  warn: test_patch did not apply for {instance[\"instance_id\"][:40]}')
    subprocess.run(['git', 'add', '-A'], cwd=str(workdir), check=False)
    subprocess.run(['git', 'commit', '--quiet', '--no-verify', '-m', 'baseline', '--allow-empty'], cwd=str(workdir), check=False)
"""),
    code("""
# 8) Run 20 problems sequentially with per-problem Drive checkpoint
from runner import Runner
from instrumentation.capture import audit_captures
import torch, json, time
from pathlib import Path

CHECKPOINT = Path(DRIVE_ROOT) / 'phase1_results.json'
results: dict[str, dict] = {}
if CHECKPOINT.exists():
    results = json.loads(CHECKPOINT.read_text())
    print(f'Resuming — {len(results)} problems already done')

runner = Runner(model=model, tokenizer=tok, config=cfg)

for i, inst in enumerate(selected):
    iid = inst['instance_id']
    if iid in results and results[iid].get('finished') is not None:
        print(f'[{i+1:2d}/{len(selected)}] SKIP {iid[:60]}')
        continue
    print(f'\\n[{i+1:2d}/{len(selected)}] RUN  {iid[:60]} | {inst[\"repo\"]}')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        out = runner.run_one(inst, prepare_workdir_fn=prep_workdir)
        audit = audit_captures(out['captures_meta'])
        out['audit_ok'] = audit['ok']
        out['vram_peak_gb'] = round(torch.cuda.max_memory_allocated() / (1024**3), 1)
    except Exception as e:
        out = {
            'instance_id': iid,
            'finished': False,
            'finish_reason': 'error',
            'error': f'{type(e).__name__}: {e}',
            'wall_seconds': 0.0,
            'n_turns': 0,
            'n_captures': 0,
            'audit_ok': False,
            'vram_peak_gb': 0.0,
        }
        print(f'  ERROR: {out[\"error\"]}')
    results[iid] = out
    CHECKPOINT.write_text(json.dumps(results, indent=2))
    print(f'  wall={out[\"wall_seconds\"]/60:.1f}min  turns={out[\"n_turns\"]:2d}  caps={out[\"n_captures\"]}  vram={out.get(\"vram_peak_gb\", 0):.1f}GB  patch_bytes={out.get(\"patch_n_bytes\", 0)}')

print(f'\\n=== Phase 1 complete: {len(results)}/{len(selected)} ===')
"""),
    code("""
# 9) G2 — determinism check on problem #0 (rerun with same seed)
# We re-run the agent with the same seed and compare the first 5 turns of generated text.
import json
from agent import AgentLoop
from sandbox import BashSession
from instrumentation import LayerTap, CaptureBuffer
from agent.prompts import render_problem
import torch

target = selected[0]
iid = target['instance_id']
seed = cfg.seed_for(iid)
print(f'Rerunning {iid} with seed={seed} for G2 determinism check')

# Read original turns from saved trace
orig_trace_path = cfg.trace_root / f'{iid}.json'
if not orig_trace_path.exists():
    print(f'No prior trace at {orig_trace_path} — skip G2')
    g2 = None
else:
    orig_trace = json.loads(orig_trace_path.read_text())
    orig_first_5 = [t['raw_response'] for t in orig_trace['turns'][:5]]

    # Rerun (capped at 5 turns to save time — use a temp config)
    from dataclasses import replace
    cfg_g2 = replace(cfg, max_turns=5)
    workdir_g2 = Path('/content/work_g2') / iid
    workdir_g2.parent.mkdir(parents=True, exist_ok=True)
    if workdir_g2.exists():
        import shutil
        shutil.rmtree(workdir_g2)
    prep_workdir(target, workdir_g2)
    target_w = {**target, '__workdir__': str(workdir_g2)}

    cap_buf2 = CaptureBuffer(instance_id=iid + '__rerun')
    tap2 = LayerTap(model, cfg_g2.capture_layers).attach()
    bash2 = BashSession(workdir_g2, default_timeout=cfg_g2.bash_timeout_default, max_output_bytes=cfg_g2.bash_max_output_bytes)
    try:
        loop2 = AgentLoop(model=model, tokenizer=tok, config=cfg_g2, bash_session=bash2, tap=tap2, capture_buffer=cap_buf2, instance_id=iid+'__rerun', seed=seed)
        rerun = loop2.run(render_problem(target_w))
    finally:
        tap2.detach(); bash2.close()

    rerun_first_5 = [t.raw_response for t in rerun.turns[:5]]
    n_check = min(len(orig_first_5), len(rerun_first_5))
    matches = [orig_first_5[i] == rerun_first_5[i] for i in range(n_check)]
    print(f'G2 determinism: {sum(matches)}/{n_check} of first 5 turns byte-equal')
    g2 = (n_check > 0) and all(matches)
    print('G2:', 'PASS' if g2 else 'FAIL')
"""),
    code("""
# 10) G3 — soft pass-rate sanity (finished + non-empty patch)
import json

soft_pass = sum(1 for r in results.values() if r.get('finished') and r.get('patch_n_bytes', 0) > 0)
total = len(results)
rate = soft_pass / max(total, 1)
print(f'Soft pass-rate: {soft_pass}/{total} = {rate*100:.1f}%')
print(f'Sanity gate: 30% ≤ rate ≤ 60%')
g3 = 0.30 <= rate <= 0.60
print('G3:', 'PASS' if g3 else f'OUT OF RANGE (Qwen reports 53.5% on full Pro — investigate)')

# Breakdown
from collections import Counter
finish_reasons = Counter(r.get('finish_reason', 'error') for r in results.values())
print('\\nFinish reasons:', dict(finish_reasons))
"""),
    code("""
# 11) G4 + G5 at scale + final roll-up
import torch

audit_ok = sum(1 for r in results.values() if r.get('audit_ok'))
audit_total = len(results)
g4 = audit_ok == audit_total
print(f'G4 audit: {audit_ok}/{audit_total} captures pass')

walls = [r['wall_seconds']/60 for r in results.values() if r.get('wall_seconds', 0) > 0]
vrams = [r.get('vram_peak_gb', 0) for r in results.values() if r.get('vram_peak_gb', 0) > 0]
if walls:
    print(f'\\nWall time per problem (min):  median={sorted(walls)[len(walls)//2]:.1f}  max={max(walls):.1f}  total={sum(walls):.1f}')
if vrams:
    print(f'VRAM peak (GB):  median={sorted(vrams)[len(vrams)//2]:.1f}  max={max(vrams):.1f}')

g5 = walls and max(walls) <= 15.0 and (not vrams or max(vrams) <= 80.0)
print(f'G5 (scaled targets ≤15min/≤80GB): {\"PASS\" if g5 else \"INFO\"}')

print('\\n=========== Phase 1 Gate Summary ===========')
print(f'G2 determinism : {\"PASS\" if g2 else (\"FAIL\" if g2 is False else \"SKIPPED\")}')
print(f'G3 pass-rate   : {\"PASS\" if g3 else f\"INFO (rate={rate*100:.1f}%)\"}')
print(f'G4 capture     : {\"PASS\" if g4 else \"FAIL\"}')
print(f'G5 resources   : {\"PASS\" if g5 else \"INFO\"}')
"""),
    code("""
# 12) Save aggregate Phase 1 report to Drive
import json
from pathlib import Path

report = {
    'phase': 1,
    'n_problems': len(results),
    'soft_pass_rate': rate,
    'soft_pass_count': soft_pass,
    'finish_reasons': dict(finish_reasons),
    'audit_ok_count': audit_ok,
    'wall_minutes_total': float(sum(walls)) if walls else 0.0,
    'wall_minutes_median': float(sorted(walls)[len(walls)//2]) if walls else 0.0,
    'wall_minutes_max': float(max(walls)) if walls else 0.0,
    'vram_gb_max': float(max(vrams)) if vrams else 0.0,
    'gates': {
        'G2_determinism': bool(g2) if g2 is not None else None,
        'G3_passrate_in_range': bool(g3),
        'G4_capture_audit': bool(g4),
        'G5_resources': bool(g5) if walls else None,
    },
    'instance_results': {
        iid: {
            'finished': r.get('finished'),
            'finish_reason': r.get('finish_reason'),
            'wall_min': round(r.get('wall_seconds', 0)/60, 1),
            'n_turns': r.get('n_turns', 0),
            'n_captures': r.get('n_captures', 0),
            'patch_bytes': r.get('patch_n_bytes', 0),
            'audit_ok': r.get('audit_ok', False),
            'vram_gb': r.get('vram_peak_gb', 0),
            'error': r.get('error'),
        } for iid, r in results.items()
    },
}

report_path = Path(DRIVE_ROOT) / 'phase1_report.json'
report_path.write_text(json.dumps(report, indent=2))
print(f'Wrote {report_path}')
print(f'\\nReport summary:')
print(json.dumps({k: v for k, v in report.items() if k != 'instance_results'}, indent=2))
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "colab": {"name": "nb_swebench_v1_phase1.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
