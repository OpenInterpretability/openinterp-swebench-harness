"""Generate notebooks/nb_swebench_v7_phase7_steering_pilot.ipynb — Phase 7 micro-pilot.

Cheap log-prob proxy for causality at L43 pre_tool. Runs on a SEPARATE Colab
session from Phase 6 trace collection (so doesn't interfere). ~10-15 min compute,
~$1-2 cost.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_swebench_v7_phase7_steering_pilot.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n", "outputs": [], "execution_count": None}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# SWE-bench Phase 7 — Steering Micro-Pilot at L43 pre_tool

Cheap causality proxy. **Runs on a SEPARATE Colab session** from Phase 6
trace collection — does not interfere.

What we test: does adding +α × probe_direction at L43 pre_tool position shift
the model's next-token log-probabilities toward "finish" (signaling solve) for
traces that originally failed?

If yes → probe is causal (paper goes top-tier)
If no → probe is epiphenomenal / correlative (paper still publishes positive
finding at correlative level + intervention failure as separate finding)

Cost: ~10-15 min on RTX 6000 / A100 / L40S. Single forward per (trace, α) pair.

**Sequence**: roda em paralelo ao Phase 6 (Colab session #1). Quando Phase 6
terminar (~10h), Phase 7c full = re-run completo de N traces se este pilot
sugerir causalidade.
"""),
    code("""
# 1) Install — same as Phase 6 (need fla for GDN fast path)
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
        raise SystemExit(f'BLOCKING: {pkg} missing — {e}. Restart runtime + rerun this cell.')
"""),
    code("""
# 2) GPU pre-flight
import subprocess
out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print(f'GPU: {out}')
mem_gb = int(out.split(',')[1].strip().split()[0]) / 1024
print(f'VRAM: {mem_gb:.1f} GB')
assert mem_gb >= 47, f'Need >=48GB. Got {mem_gb:.1f}GB.'
"""),
    code("""
# 3) Drive mount
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs'
print(f'Drive root: {DRIVE_ROOT}')
"""),
    code("""
# 4) Pull harness latest (gets phase7 script)
import os, sys, subprocess
HARNESS = '/content/openinterp-swebench-harness'
GITHUB_URL = 'https://github.com/OpenInterpretability/openinterp-swebench-harness'
if os.path.exists(HARNESS):
    subprocess.run(['git', '-C', HARNESS, 'pull', '--quiet'], check=False)
else:
    subprocess.run(['git', 'clone', GITHUB_URL, HARNESS], check=True)
sys.path.insert(0, HARNESS)
print('HEAD:', subprocess.run(['git', '-C', HARNESS, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip())
"""),
    code("""
# 5) Run Phase 7 micro-pilot
%cd /content/openinterp-swebench-harness
!python3 scripts/phase7_steering_micro_pilot.py --n-traces 8 --alphas -2 0 1 2 5
"""),
    code("""
# 6) Inspect results + plot
import json
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = f'{DRIVE_ROOT}/swebench_v6_phase6/phase7_steering_pilot.json'
with open(OUT_PATH) as f:
    r = json.load(f)

# Plot Δlog-prob(finish) vs α, separated by verdict
alphas = [float(a) for a in r['alphas']]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, vtype in enumerate(['fails', 'solves']):
    ax = axes[i]
    for iid, res in r['results'].items():
        if res['verdict'] != vtype:
            continue
        lp_per_a = res['logprobs_per_alpha']
        ys = [lp_per_a.get(str(a), {}).get('finish', np.nan) - lp_per_a.get('0.0', {}).get('finish', np.nan)
              for a in alphas]
        ax.plot(alphas, ys, alpha=0.5, label=f'score={res[\"probe_score\"]:.2f}')
    ax.set_title(f'{vtype.upper()}  Δlog-prob(finish) vs α at L43 pre_tool')
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('α (steering coefficient)')
    if i == 0:
        ax.set_ylabel('Δ log-prob(finish) vs α=0')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{DRIVE_ROOT}/swebench_v6_phase6/phase7_logprob_shift.png', dpi=150)
plt.show()

agg = r['aggregate']
print('\\n=== AGGREGATE ===')
print(f'FAILS (n={agg[\"n_fails\"]}): mean Δlog-prob(finish) at α=+2: {agg[\"fails_mean_shift_finish_alpha2\"]:+.3f}')
print(f'SOLVES (n={agg[\"n_solves\"]}): mean Δlog-prob(finish) at α=+2: {agg[\"solves_mean_shift_finish_alpha2\"]:+.3f}')
"""),
    md("""
## Interpretation guide

**If FAILS shift mean > +0.5 at α=+2**: probe direction CAUSALLY pushes model toward "finish"
even on traces that originally failed → causal interpretation. Paper top-tier.

**If FAILS shift mean ≈ 0 (-0.2 to +0.2)**: probe is epiphenomenal — direction reads but
does not lever capability. Honest negative result. Paper still publishes correlative finding
(AUROC 0.83 is real) plus intervention-failure as standalone result.

**If FAILS shift mean < -0.5**: probe direction reads "I will fail" not "I can solve" —
reversed sign. Important interpretive correction.

**Phase 7 full (post-pilot, if pilot suggests causal)**: re-run 10 failed traces fully with
α=+2 steering applied at the hook point. Run Docker eval. Compare verdicts. ~$5-10, ~2-3h.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "colab": {"name": "nb_swebench_v7_phase7_steering_pilot.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
