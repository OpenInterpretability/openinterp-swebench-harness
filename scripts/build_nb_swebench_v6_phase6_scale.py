"""Generate notebooks/nb_swebench_v6_phase6_scale.ipynb — Phase 1.5 scale-up to N=100.

Larger N to address paper reviewer concerns about N=17 statistics. Same protocol
as Phase 1 but max_turns=50 + 5x more instances + finer stratification.

Targets the two strict-reviewer concerns from preflight_probe_eval.md:
  - "AUROC=1.000 with 4 positives is consistent with chance" — bigger N gives proper power
  - "Permutation null test missing" — Phase 6b includes it on the larger sample

Wall-time estimate: 100 × ~12 min = ~20h Colab. Resume-on-disconnect via per-problem
Drive checkpoint. Recommended: 4 batches of 25 over 2-3 days.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_swebench_v6_phase6_scale.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n", "outputs": [], "execution_count": None}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# SWE-bench Pro Failure Anatomy — Phase 6 Scale-Up (N=100)

Phase 1 (N=20) showed AUROC = 1.000 at L43/L55 think_start with TRUE Docker labels.
Bootstrap CI tight, LORO 1.0 across 3 repos. **But N=4 positives is small.**

Phase 6 scales to **N=100 stratified Python problems** with `max_turns=50` (was 30) for
proper statistical power. Expected outcome:

| Result | Interpretation |
|---|---|
| AUROC ≥ 0.95 | 🟢 Phase 1 finding holds. Paper-grade. |
| AUROC 0.85–0.95 | 🟡 Strong but small-N artifact partial. Paper still publishable. |
| AUROC < 0.85 | 🔴 Phase 1 was N=4 lucky. Paper needs major reframe. |

**Sequencing**:
1. This notebook = trace collection (~20h Colab on RTX PRO 6000)
2. Phase 6b notebook (separate) = Docker eval on local Mac (~5h, no Colab)
3. Phase 6c (light) = re-do Phase 5b/5c analysis on 100-instance label set

**Resume**: per-problem Drive checkpoint; can disconnect and resume. Recommended: 4
batches of 25 problems over 2-3 days.
"""),
    code("""
# 1) Install — same stack as Phase 1
!pip install -q -U transformers
!pip install -q datasets safetensors pexpect huggingface-hub
!pip install -q flash-linear-attention causal-conv1d --no-build-isolation 2>&1 | tail -3
!pip install -q flash-attn --no-build-isolation 2>/dev/null && echo 'flash-attn ok' || echo 'flash-attn unavailable, will use sdpa'
"""),
    code("""
# 2) Drive mount
import os
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs/swebench_v6_phase6'
os.makedirs(DRIVE_ROOT, exist_ok=True)
print('Drive root:', DRIVE_ROOT)
"""),
    code("""
# 3) Pull harness latest
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
# 4) Config — Phase 6 defaults (max_turns BUMPED to 50)
from pathlib import Path
from dataclasses import replace
from config import HarnessConfig, DEFAULT

cfg = replace(
    DEFAULT,
    work_root=Path('/content/work_p6'),
    capture_root=Path(DRIVE_ROOT) / 'captures',
    trace_root=Path(DRIVE_ROOT) / 'traces',
    max_turns=50,              # ← was 30 in Phase 1
    bash_max_output_bytes=32_000,
)
print(f'max_turns={cfg.max_turns}  bash_max_output={cfg.bash_max_output_bytes}')
print(f'capture_layers={cfg.capture_layers}')
"""),
    code("""
# 5) Load Qwen3.6-27B (same as Phase 1)
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
print(f'Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params')
"""),
    code("""
# 6) Stratified sample of 100 Python problems (excluding Phase 1's 20 to avoid double-count)
from datasets import load_dataset
import random
import numpy as np
import json

ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')
py = ds.filter(lambda x: x.get('repo_language') == 'python')
print(f'Python instances: {len(py)} / {len(ds)}')

# Exclude Phase 1 instances
PHASE1_PATH = '/content/drive/MyDrive/openinterp_runs/swebench_v1_phase1/phase1_report.json'
phase1_iids: set = set()
if os.path.exists(PHASE1_PATH):
    with open(PHASE1_PATH) as f:
        phase1_iids = set(json.load(f)['instance_results'].keys())
    print(f'Excluding {len(phase1_iids)} Phase 1 instances from sample')

py_filtered = [x for x in py if x['instance_id'] not in phase1_iids]
print(f'Python pool after exclusion: {len(py_filtered)}')

# Stratify by problem_statement length × repo
lengths = np.array([len(x.get('problem_statement') or '') for x in py_filtered])
q33, q66 = float(np.quantile(lengths, 0.33)), float(np.quantile(lengths, 0.66))

def length_bucket(L: float) -> int:
    return 0 if L < q33 else (1 if L < q66 else 2)

# Group by (repo, length bucket)
from collections import defaultdict
groups: dict[tuple[str, int], list[int]] = defaultdict(list)
for i, x in enumerate(py_filtered):
    groups[(x['repo'], length_bucket(lengths[i]))].append(i)

PHASE6_SEED = 137
rng = random.Random(PHASE6_SEED)
N = 100
selected_idx: list[int] = []
all_groups = list(groups.keys())
rng.shuffle(all_groups)
for gk in all_groups:
    if len(selected_idx) >= N:
        break
    bucket = groups[gk][:]
    rng.shuffle(bucket)
    take = min(len(bucket), max(1, N // len(all_groups)))
    selected_idx.extend(bucket[:take])
selected_idx = selected_idx[:N]
selected = [dict(py_filtered[i]) for i in selected_idx]

print(f'\\nSelected {len(selected)} problems across:')
from collections import Counter
print(f'  repos: {Counter(x[\"repo\"] for x in selected).most_common()}')
print(f'  buckets: {Counter(length_bucket(len(x.get(\"problem_statement\") or \"\")) for x in selected)}')

# Save selection for repro
with open(f'{DRIVE_ROOT}/selected_iids.json', 'w') as f:
    json.dump([x['instance_id'] for x in selected], f, indent=2)
print(f'Selection saved to {DRIVE_ROOT}/selected_iids.json')
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
# 8) Sequential loop with per-problem Drive checkpoint
from runner import Runner
from instrumentation.capture import audit_captures
import torch, json, time
from pathlib import Path

CHECKPOINT = Path(DRIVE_ROOT) / 'phase6_results.json'
results: dict[str, dict] = {}
if CHECKPOINT.exists():
    results = json.loads(CHECKPOINT.read_text())
    print(f'Resuming — {len(results)} problems already done')

runner = Runner(model=model, tokenizer=tok, config=cfg)

t_start = time.time()
for i, inst in enumerate(selected):
    iid = inst['instance_id']
    if iid in results and results[iid].get('finished') is not None:
        continue
    print(f'[{i+1:3d}/{len(selected)}] RUN  {iid[:65]} | {inst[\"repo\"]}')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        out = runner.run_one(inst, prepare_workdir_fn=prep_workdir)
        audit = audit_captures(out['captures_meta'])
        out['audit_ok'] = audit['ok']
        out['vram_peak_gb'] = round(torch.cuda.max_memory_allocated() / (1024**3), 1)
    except Exception as e:
        out = {
            'instance_id': iid, 'finished': False, 'finish_reason': 'error',
            'error': f'{type(e).__name__}: {e}', 'wall_seconds': 0.0,
            'n_turns': 0, 'n_captures': 0, 'audit_ok': False, 'vram_peak_gb': 0.0,
        }
        print(f'  ERROR: {out[\"error\"]}')
    results[iid] = out
    CHECKPOINT.write_text(json.dumps(results, indent=2))
    elapsed_min = (time.time() - t_start) / 60
    avg_min = elapsed_min / max(1, len(results))
    eta_min = avg_min * (len(selected) - len(results))
    print(f'  wall={out[\"wall_seconds\"]/60:.1f}min  turns={out[\"n_turns\"]:2d}  caps={out[\"n_captures\"]}  vram={out.get(\"vram_peak_gb\", 0):.1f}GB  patch={out.get(\"patch_n_bytes\", 0)} | elapsed {elapsed_min:.0f}m, ETA {eta_min:.0f}m')

print(f'\\n=== Phase 6 COMPLETE: {len(results)}/{len(selected)} ===')
"""),
    code("""
# 9) Quick stats summary
from collections import Counter
import numpy as np

finishes = Counter(r.get('finish_reason', 'error') for r in results.values())
patch_count = sum(1 for r in results.values() if r.get('patch_n_bytes', 0) > 0)
walls = [r.get('wall_seconds', 0)/60 for r in results.values() if r.get('wall_seconds', 0) > 0]
vrams = [r.get('vram_peak_gb', 0) for r in results.values() if r.get('vram_peak_gb', 0) > 0]
audit_ok = sum(1 for r in results.values() if r.get('audit_ok'))

print(f'Patches generated: {patch_count}/{len(results)} ({patch_count/len(results)*100:.1f}%)')
print(f'Audit pass: {audit_ok}/{len(results)}')
print(f'Wall time per problem (min): median={np.median(walls):.1f}, max={np.max(walls):.1f}, total={sum(walls):.0f}')
print(f'VRAM peak (GB): median={np.median(vrams):.1f}, max={np.max(vrams):.1f}')
print(f'Finish reasons: {dict(finishes)}')

# Save aggregate
agg = {
    'phase': 6,
    'n_target': len(selected),
    'n_done': len(results),
    'n_patches': patch_count,
    'n_audit_ok': audit_ok,
    'finish_reasons': dict(finishes),
    'wall_min_total': float(sum(walls)),
    'wall_min_median': float(np.median(walls)) if walls else 0.0,
    'vram_gb_max': float(max(vrams)) if vrams else 0.0,
}
with open(f'{DRIVE_ROOT}/phase6_aggregate.json', 'w') as f:
    json.dump(agg, f, indent=2)
print(f'\\nWrote {DRIVE_ROOT}/phase6_aggregate.json')
"""),
    md("""
## Next steps after Phase 6 trace collection

1. **Phase 6b (local Mac, ~5-8h)**: run Docker evaluation on every patched instance.
   - Pull `jefzda/sweap-images:{dockerhub_tag}` per instance
   - 3 conditions: none / golden / agent_phase6
   - Save verdicts to `phase6b_docker_verdicts.json`

2. **Phase 6c (local CPU, ~10 min)**: re-do Phase 5b/5c analysis on N=100 with TRUE labels.
   - Single-probe AUROC at L43/L55 think_start
   - Bootstrap 1000 resamples (now with N≈80-100 + 30-50 positives — proper power)
   - **Permutation null test** (1000 shuffles)
   - LORO across all repos in N=100

3. **Paper update**: replace N=17 numbers with N=100 numbers. AUROC and CIs will tighten
   substantially. Permutation null gives p-value.

**Decision tree post-Phase 6c**:
- Phase 1 finding holds (AUROC ≥ 0.85): paper accepted with revisions
- Phase 1 was small-N lucky (AUROC 0.65-0.85): reframe paper as "promising signal but does not generalize at scale"
- Total fail (AUROC < 0.65): pivot to a different layer/position or accept null result
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "colab": {"name": "nb_swebench_v6_phase6_scale.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
