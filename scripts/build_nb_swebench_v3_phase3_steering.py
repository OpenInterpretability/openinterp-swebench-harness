"""Generate notebooks/nb_swebench_v3_phase3_steering.ipynb — Phase 3 causal validation.

Tests whether the Phase 2 probe direction (L31 turn_end) is CAUSALLY responsible for
patch generation differences, or merely correlated.

Method: steering vector
  v = normalize(mean(success_L31_turn_end) - mean(fail_L31_turn_end))
  Hook layer 31 forward output: hs += alpha * v during full re-run of 5 fail traces.
  Compare patch generation under alpha sweep.

Verdict criterion: ≥3/5 fail traces produce non-empty patches at any tested alpha → causal.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_swebench_v3_phase3_steering.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n", "outputs": [], "execution_count": None}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# SWE-bench Pro Failure Anatomy — Phase 3 Causal Steering

Phase 2 verdict: 🟢 STRONG_SIGNAL — probes at L31 turn_end achieve AUROC 0.958 distinguishing
patch-success from patch-fail traces. **But is this correlational or causal?**

Phase 3 tests causality via **steering vector**:

1. Compute v = normalize(mean(success @ L31 turn_end) - mean(fail @ L31 turn_end))
2. For 5 fail traces from Phase 1, re-run with forward hook on layer 31 adding α·v
3. Sweep α ∈ {0, 0.5, 1, 2, 5} (where 0 = control)
4. Compare patch generation: did α > 0 produce patches in originally-failing traces?

**Verdict criteria**:
- 🟢 CAUSAL if ≥3/5 fail traces produce non-empty patches at any α > 0 (and α = 0 control matches Phase 1)
- 🟡 PARTIAL if 1-2/5 produce patches
- 🔴 CORRELATIONAL ONLY if 0/5 produce patches

Hardware target: **RTX PRO 6000 Blackwell 96GB** Colab Pro+. ~1.5h compute total.
"""),
    code("""
# 1) Install
!pip install -q -U transformers
!pip install -q datasets safetensors pexpect huggingface-hub
!pip install -q flash-linear-attention causal-conv1d --no-build-isolation 2>&1 | tail -3
!pip install -q flash-attn --no-build-isolation 2>/dev/null && echo 'flash-attn ok' || echo 'flash-attn unavailable'
"""),
    code("""
# 2) Mount Drive + locate Phase 1/2 outputs
import os, json
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs/swebench_v1_phase1'
PHASE3_ROOT = '/content/drive/MyDrive/openinterp_runs/swebench_v3_phase3'
os.makedirs(PHASE3_ROOT, exist_ok=True)
assert os.path.exists(f'{DRIVE_ROOT}/phase2_report.json'), 'Phase 2 must complete first'
print('Phase 1 root:', DRIVE_ROOT)
print('Phase 3 root:', PHASE3_ROOT)
"""),
    code("""
# 3) Pull harness
import os, sys, subprocess
HARNESS_PATH = '/content/openinterp-swebench-harness'
GITHUB_URL = 'https://github.com/OpenInterpretability/openinterp-swebench-harness'
if os.path.exists(HARNESS_PATH):
    subprocess.run(['git', '-C', HARNESS_PATH, 'pull', '--quiet'], check=False)
else:
    subprocess.run(['git', 'clone', GITHUB_URL, HARNESS_PATH], check=True)
sys.path.insert(0, HARNESS_PATH)
print('HEAD:', subprocess.run(['git', '-C', HARNESS_PATH, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip())
"""),
    code("""
# 4) Compute steering vector v from Phase 1 captures
import json
from pathlib import Path
from safetensors.torch import load_file
import torch
import numpy as np
import re

with open(f'{DRIVE_ROOT}/phase1_report.json') as f:
    p1_report = json.load(f)
results = p1_report['instance_results']
success_iids = [iid for iid, r in results.items() if r.get('patch_bytes', 0) > 0]
fail_iids = [iid for iid, r in results.items() if r.get('patch_bytes', 0) == 0]
print(f'Success: {len(success_iids)}  Fail: {len(fail_iids)}')

TARGET_LAYER = 31
TARGET_POSITION = 'turn_end'
CAPTURE_DIR = Path(f'{DRIVE_ROOT}/captures')

def load_layer_pos_mean(iid: str, layer: int, pos: str) -> np.ndarray | None:
    meta_path = CAPTURE_DIR / f'{iid}.meta.json'
    weights_path = CAPTURE_DIR / f'{iid}.safetensors'
    if not meta_path.exists() or not weights_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    tensors = load_file(str(weights_path))
    vecs = []
    for rec in meta['records']:
        if rec['layer'] == layer and rec['position_label'] == pos:
            key = rec['activation_key']
            if key in tensors:
                vecs.append(tensors[key].to(torch.float32).numpy())
    if not vecs:
        return None
    return np.mean(np.stack(vecs, axis=0), axis=0)

success_means = np.stack([m for iid in success_iids if (m := load_layer_pos_mean(iid, TARGET_LAYER, TARGET_POSITION)) is not None])
fail_means = np.stack([m for iid in fail_iids if (m := load_layer_pos_mean(iid, TARGET_LAYER, TARGET_POSITION)) is not None])
print(f'success_means: {success_means.shape}  fail_means: {fail_means.shape}')

v = success_means.mean(axis=0) - fail_means.mean(axis=0)
v_norm_l2 = float(np.linalg.norm(v))
v_normalized = v / (v_norm_l2 + 1e-10)
v_torch = torch.tensor(v_normalized, dtype=torch.bfloat16)

# Reference activation magnitude
ref_act_mag = float(np.mean([np.linalg.norm(m) for m in success_means]))
print(f'\\nSteering vector v at L{TARGET_LAYER} {TARGET_POSITION}:')
print(f'  ||v||_2 (raw diff)      = {v_norm_l2:.3f}')
print(f'  Reference ||activation|| = {ref_act_mag:.3f}')
print(f'  scale ratio (v / ref)   = {v_norm_l2 / ref_act_mag:.3f}')
print(f'  v_normalized stored on CPU as bfloat16 ({v_torch.shape})')

torch.save({'v': v_torch, 'layer': TARGET_LAYER, 'position': TARGET_POSITION,
            'success_iids': success_iids, 'fail_iids': fail_iids,
            'v_norm_l2': v_norm_l2, 'ref_act_mag': ref_act_mag}, f'{PHASE3_ROOT}/steering_vector.pt')
print(f'\\nSaved steering vector → {PHASE3_ROOT}/steering_vector.pt')
"""),
    code("""
# 5) Load Qwen3.6-27B
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = 'Qwen/Qwen3.6-27B'

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
attn_impl = 'flash_attention_2'
try:
    import flash_attn  # noqa
except ImportError:
    attn_impl = 'sdpa'
try:
    import fla, causal_conv1d  # noqa
    print('GDN fast path: AVAILABLE')
except ImportError as e:
    print(f'WARN GDN fast path missing: {e}')

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation=attn_impl,
    device_map='auto', trust_remote_code=True,
)
model.eval()
print(f'Loaded {sum(p.numel() for p in model.parameters())/1e9:.2f}B params')

# Move steering vector to model device
v_torch = v_torch.to(next(model.parameters()).device)
ref_act_mag_torch = torch.tensor(ref_act_mag, dtype=torch.bfloat16, device=v_torch.device)
"""),
    code("""
# 6) Steering hook factory: hook layer 31 output to add alpha * v at every position
import torch
import torch.nn as nn

class SteeringHook:
    \"\"\"Forward post-hook that adds alpha * v_normalized * scale to every token position
    of layer 31's output. scale is reference activation magnitude so alpha is in
    units of 'fraction of typical activation magnitude'.\"\"\"
    def __init__(self, v: torch.Tensor, scale: float, alpha: float):
        self.v = v
        self.scale = scale
        self.alpha = alpha
        self.handle = None

    def hook(self, module, inputs, output):
        if self.alpha == 0.0:
            return output
        if isinstance(output, tuple):
            hs, *rest = output
        else:
            hs, rest = output, []
        if not torch.is_tensor(hs):
            return output
        delta = (self.alpha * self.scale) * self.v.to(hs.device, hs.dtype)
        steered = hs + delta
        if rest:
            return (steered, *rest)
        return steered

    def attach(self, layer_module: nn.Module):
        self.handle = layer_module.register_forward_hook(self.hook)
        return self

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

def get_layer_module(model, idx: int) -> nn.Module:
    for path in ('model.model.layers', 'model.layers', 'transformer.h'):
        cur = model
        ok = True
        for part in path.split('.'):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            try:
                return cur[idx]
            except (TypeError, IndexError):
                continue
    raise RuntimeError(f'cannot find layer {idx}')

# Smoke test the hook attaches and detaches cleanly
target_layer_module = get_layer_module(model, TARGET_LAYER)
print(f'Target layer module: {type(target_layer_module).__name__}')
sh = SteeringHook(v_torch, ref_act_mag, alpha=0.0).attach(target_layer_module)
sh.detach()
print('SteeringHook attach/detach OK')
"""),
    code("""
# 7) Workdir prep (same as Phase 1)
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
        subprocess.run(['git', 'apply', '--allow-empty', '-'], input=test_patch, text=True, cwd=str(workdir))
    subprocess.run(['git', 'add', '-A'], cwd=str(workdir), check=False)
    subprocess.run(['git', 'commit', '--quiet', '--no-verify', '-m', 'baseline', '--allow-empty'], cwd=str(workdir), check=False)
"""),
    code("""
# 8) Pick 5 fail traces + load instance metadata from dataset
from datasets import load_dataset

ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
ds_by_iid = {x['instance_id']: dict(x) for x in ds}

# Sort fail iids by Phase 1 wall time descending — pick the ones the agent worked hardest on
fail_with_walls = [(iid, results[iid].get('wall_seconds', 0)) for iid in fail_iids]
fail_with_walls.sort(key=lambda x: -x[1])
PICK = [iid for iid, _ in fail_with_walls[:5]]
fail_instances = []
for iid in PICK:
    if iid in ds_by_iid:
        fail_instances.append(ds_by_iid[iid])
    else:
        print(f'WARN {iid} not in dataset')
print(f'Picked {len(fail_instances)} fail traces:')
for inst in fail_instances:
    p1 = results[inst['instance_id']]
    print(f"  {inst['instance_id'][:60]:60s}  wall={p1[\"wall_seconds\"]/60:.1f}min  turns={p1[\"n_turns\"]}")
"""),
    code("""
# 9) Run alpha sweep on each fail trace
from pathlib import Path
from config import HarnessConfig
from runner import Runner
import torch, json, time
from dataclasses import replace

ALPHAS = [0.0, 0.5, 1.0, 2.0, 5.0]
cfg = HarnessConfig(
    work_root=Path('/content/work_phase3'),
    capture_root=Path(PHASE3_ROOT) / 'captures',
    trace_root=Path(PHASE3_ROOT) / 'traces',
    max_turns=30,
)

results_p3 = {}
checkpoint_path = Path(PHASE3_ROOT) / 'phase3_results.json'
if checkpoint_path.exists():
    results_p3 = json.loads(checkpoint_path.read_text())
    print(f'Resuming — {len(results_p3)} runs already complete')

for inst in fail_instances:
    iid = inst['instance_id']
    for alpha in ALPHAS:
        run_key = f'{iid}__alpha{alpha:.1f}'
        if run_key in results_p3 and results_p3[run_key].get('finished') is not None:
            continue
        # Use unique trace_root per alpha to avoid file collisions
        cfg_run = replace(
            cfg,
            work_root=Path(f'/content/work_phase3/alpha_{alpha:.1f}'),
            capture_root=Path(PHASE3_ROOT) / f'alpha_{alpha:.1f}' / 'captures',
            trace_root=Path(PHASE3_ROOT) / f'alpha_{alpha:.1f}' / 'traces',
        )
        runner = Runner(model=model, tokenizer=tok, config=cfg_run)
        # Attach steering hook
        sh = SteeringHook(v_torch, ref_act_mag, alpha=alpha).attach(target_layer_module)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f'\\n[{iid[:60]:60s}] alpha={alpha}')
        try:
            t0 = time.time()
            out = runner.run_one(inst, prepare_workdir_fn=prep_workdir)
            wall = time.time() - t0
            out['vram_peak_gb'] = round(torch.cuda.max_memory_allocated() / (1024**3), 1)
            out['alpha'] = alpha
        except Exception as e:
            out = {'instance_id': iid, 'alpha': alpha, 'finished': False, 'finish_reason': 'error',
                   'error': f'{type(e).__name__}: {e}', 'wall_seconds': 0.0, 'n_turns': 0,
                   'n_captures': 0, 'patch_n_bytes': 0, 'vram_peak_gb': 0.0}
            print(f'  ERROR: {out[\"error\"]}')
        finally:
            sh.detach()
        results_p3[run_key] = out
        checkpoint_path.write_text(json.dumps(results_p3, indent=2))
        print(f'  wall={out[\"wall_seconds\"]/60:.1f}min  turns={out[\"n_turns\"]}  patch={out.get(\"patch_n_bytes\",0)}  reason={out.get(\"finish_reason\")}')

print(f'\\n=== Phase 3 done: {len(results_p3)}/{len(fail_instances)*len(ALPHAS)} ===')
"""),
    code("""
# 10) Aggregate: per-instance, per-alpha — did patches appear?
import numpy as np
from collections import defaultdict

table = defaultdict(dict)
for run_key, r in results_p3.items():
    iid, alpha_str = run_key.rsplit('__alpha', 1)
    alpha = float(alpha_str)
    table[iid][alpha] = {
        'patch_bytes': r.get('patch_n_bytes', 0),
        'finished': r.get('finished', False),
        'finish_reason': r.get('finish_reason'),
        'turns': r.get('n_turns', 0),
        'wall_min': r.get('wall_seconds', 0) / 60,
    }

print('Patch bytes by (instance, alpha):')
print(f'  {\" \" * 50}', '  '.join(f'a={a}' for a in ALPHAS))
patched_at_any = 0
for iid in sorted(table.keys()):
    row_data = table[iid]
    cells_str = '  '.join(f'{row_data.get(a, {}).get(\"patch_bytes\", 0):5d}' for a in ALPHAS)
    print(f'  {iid[:48]:48s}  {cells_str}')
    if any(row_data.get(a, {}).get('patch_bytes', 0) > 0 and a > 0 for a in ALPHAS):
        patched_at_any += 1

print(f'\\n=== Fail traces that produced a patch under steering (alpha>0): {patched_at_any}/{len(table)} ===')

baseline_zero = sum(1 for iid in table if table[iid].get(0.0, {}).get('patch_bytes', 0) == 0)
print(f'Baseline (alpha=0) check: {baseline_zero}/{len(table)} kept their no-patch state')
"""),
    code("""
# 11) Verdict + save report
import json

n_total = len(table)
n_recovered = patched_at_any  # traces that gained a patch under steering

if n_recovered >= 3:
    verdict = 'CAUSAL'
    color = '🟢'
elif n_recovered >= 1:
    verdict = 'PARTIAL'
    color = '🟡'
else:
    verdict = 'CORRELATIONAL_ONLY'
    color = '🔴'

print('=========== Phase 3 Verdict ===========')
print(f'Steering target: L{TARGET_LAYER} {TARGET_POSITION}')
print(f'Steering vector ||v||_2 = {v_norm_l2:.3f}, scale = {ref_act_mag:.3f}')
print(f'Alphas tested: {ALPHAS}')
print(f'Fail traces recovered (patch>0 at any alpha>0): {n_recovered}/{n_total}')
print(f'\\n{color} Verdict: {verdict}')

if verdict == 'CAUSAL':
    print('  → Phase 2 finding has causal evidence. Probe direction is meaningful intervention.')
    print('  → Phase 4 = scale + paper draft.')
elif verdict == 'PARTIAL':
    print('  → Mixed evidence. Try larger alpha sweep or different (layer, position) target.')
else:
    print('  → No causal effect. Probe was correlational. Pivot: try L43 or L23 target, or scale Phase 1.5.')

report = {
    'phase': 3,
    'target': {'layer': TARGET_LAYER, 'position': TARGET_POSITION},
    'steering_vector_l2': v_norm_l2,
    'reference_activation_magnitude': ref_act_mag,
    'alphas_tested': ALPHAS,
    'n_fail_traces': n_total,
    'n_recovered_under_steering': n_recovered,
    'baseline_no_patch_preserved': baseline_zero,
    'verdict': verdict,
    'per_instance': {iid: {f'alpha_{a}': table[iid].get(a, {}) for a in ALPHAS} for iid in table},
}
out_path = Path(PHASE3_ROOT) / 'phase3_report.json'
out_path.write_text(json.dumps(report, indent=2))
print(f'\\nWrote {out_path}')
"""),
    code("""
# 12) Inspect a recovered trace (if any) — read the patch
import json
from pathlib import Path

recovered_runs = []
for run_key, r in results_p3.items():
    iid, alpha_str = run_key.rsplit('__alpha', 1)
    alpha = float(alpha_str)
    if alpha > 0 and r.get('patch_n_bytes', 0) > 0:
        recovered_runs.append((iid, alpha, r.get('patch_n_bytes', 0)))

if not recovered_runs:
    print('No recovered traces to inspect.')
else:
    recovered_runs.sort(key=lambda x: -x[2])
    iid, alpha, n_bytes = recovered_runs[0]
    print(f'Inspecting biggest recovered patch:')
    print(f'  instance: {iid}')
    print(f'  alpha:    {alpha}')
    print(f'  patch:    {n_bytes} bytes\\n')
    patch_path = Path(PHASE3_ROOT) / f'alpha_{alpha:.1f}' / 'traces' / f'{iid}.patch'
    if patch_path.exists():
        print(patch_path.read_text()[:3000])
    else:
        print(f'patch file not found: {patch_path}')
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "colab": {"name": "nb_swebench_v3_phase3_steering.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
