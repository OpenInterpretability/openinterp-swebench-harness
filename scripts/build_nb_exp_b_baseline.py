"""Build Exp B Baseline notebook (no-hook control) — companion to Exp B Primary.

Tests: do the same 20 WANDERING trajectories naturally emit finish_tool when re-run
WITHOUT any hook, using same seed (Runner.seed_for) and same H100 hardware?

If baseline = 0/20 → Exp B Primary 6/20 is CAUSAL (Fisher p ≈ 0.02)
If baseline = 1-2/20 → Hook effect is real but smaller
If baseline ≥ 5/20 → Hook effect not significant vs natural stochasticity

Output: notebooks/nb_exp_b_baseline.ipynb
"""
import json
from pathlib import Path

OUT = Path('/Volumes/SSD Major/fish/openinterp-swebench-harness/notebooks/nb_exp_b_baseline.ipynb')


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}

def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": "\n".join(lines)}


cells = []

cells.append(md(
    "# Exp B Baseline — No-hook Control (paper #2)",
    "",
    "**Companion to Exp B Primary** which ran 20 WANDERING with L55 SUCCESS-donor hook (mode=add, α=0.3) and got 6/20 = 30% finish_tool.",
    "",
    "**This notebook**: re-runs the SAME 20 WANDERING with NO HOOK at SAME seed (Runner default) on SAME H100 hardware. Isolates hook effect vs natural stochasticity.",
    "",
    "## Decision matrix",
    "",
    "| Baseline flip rate | Interpretation |",
    "|---|---|",
    "| 0/20 | **CAUSAL** — Fisher exact p ≈ 0.02 vs 6/20 hook |",
    "| 1-2/20 | Strong hook effect (still likely causal) |",
    "| 3-4/20 | Marginal — controls C1-C3 essential next |",
    "| ≥5/20 | Hook NOT causal — natural variance explains |",
    "",
    "## Setup",
    "- Same install pattern as Exp B Primary (transformers main + fla + Qwen3.6-27B)",
    "- Same WANDERING IIDs (20)",
    "- Same Runner config (max_turns=50, seed deterministic by iid)",
    "- NO LayerPatch, NO hook installation",
    "- ETA: ~6.5h on H100 (matches Primary ETA since hook is <1% overhead)",
    "",
    "## What this notebook does NOT need",
    "- Donor residuals (not used)",
    "- v5 direction (not used)",
    "- LayerPatch class (not imported)",
    "- WANDERING assignment file (only need the IIDs)",
))

# Setup cells (mirror Exp B Primary)
cells.append(code(
    "# 0) GPU pre-flight — same H100 requirement as Primary",
    "import subprocess",
    "out = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],",
    "                     capture_output=True, text=True).stdout.strip()",
    "print(out)",
    "vram_gb = int(out.split(',')[1].strip().split()[0]) // 1024",
    "assert vram_gb >= 70, f'Need ≥70GB VRAM, got {vram_gb}GB. Aborting.'",
    "print(f'VRAM OK: {vram_gb}GB')",
))

cells.append(code(
    "# 1) Install — same as Exp B Primary",
    "%%capture",
    "!pip install -q --upgrade pip",
    "!pip uninstall -y transformers > /dev/null 2>&1 || true",
    "!pip install -q git+https://github.com/huggingface/transformers.git",
    "!pip install -q accelerate sentencepiece safetensors einops",
    "!pip install -q causal-conv1d --no-build-isolation",
    "!pip install -q fla-org --no-build-isolation || pip install -q flash-linear-attention",
    "!pip install -q huggingface_hub datasets",
))

cells.append(code(
    "# 1b) Log transformers commit (reproducibility)",
    "import importlib, transformers, subprocess, os",
    "print(f'transformers version: {transformers.__version__}')",
    "site = os.path.dirname(transformers.__file__)",
    "h = subprocess.run(['git','-C',site,'rev-parse','HEAD'], capture_output=True, text=True).stdout.strip()",
    "print(f'transformers commit: {h[:12] if h else \"(not a git repo)\"}')",
))

cells.append(code(
    "# 2) Drive mount + paths",
    "from google.colab import drive",
    "drive.mount('/content/drive')",
    "",
    "from pathlib import Path",
    "DRIVE_ROOT = Path('/content/drive/MyDrive/openinterp_runs/swebench_exp_b_c')",
    "DRIVE_ROOT.mkdir(parents=True, exist_ok=True)",
    "PRIMARY_DRIVE = DRIVE_ROOT / 'exp_b_primary_add'  # for comparison",
    "BASELINE_DRIVE = DRIVE_ROOT / 'exp_b_baseline_nohook'",
    "BASELINE_DRIVE.mkdir(parents=True, exist_ok=True)",
    "print(f'Output → {BASELINE_DRIVE}')",
    "print(f'Primary (for comparison) → {PRIMARY_DRIVE}')",
    "if PRIMARY_DRIVE.exists():",
    "    import json",
    "    prim = json.loads((PRIMARY_DRIVE / 'results.json').read_text())",
    "    n_ft = sum(1 for r in prim.values() if r.get('finish_reason') == 'finish_tool')",
    "    print(f'  Primary results loaded: {len(prim)} instances, {n_ft} finish_tool')",
))

cells.append(code(
    "# 3) Pull harness latest",
    "import os, subprocess",
    "REPO_DIR = '/content/openinterp-swebench-harness'",
    "if not os.path.exists(REPO_DIR):",
    "    !git clone --quiet https://github.com/OpenInterpretability/openinterp-swebench-harness.git $REPO_DIR",
    "else:",
    "    !cd $REPO_DIR && git fetch --quiet && git reset --hard origin/main",
    "%cd $REPO_DIR",
    "import sys",
    "if REPO_DIR not in sys.path:",
    "    sys.path.insert(0, REPO_DIR)",
    "print(subprocess.check_output(['git', 'log', '-1', '--oneline']).decode().strip())",
))

cells.append(code(
    "# 4) Load Qwen3.6-27B (same as Primary)",
    "import torch",
    "from transformers import AutoTokenizer, AutoModelForCausalLM",
    "",
    "MODEL_NAME = 'Qwen/Qwen3.6-27B'",
    "print(f'Loading {MODEL_NAME}...')",
    "tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)",
    "model = AutoModelForCausalLM.from_pretrained(",
    "    MODEL_NAME, trust_remote_code=True,",
    "    torch_dtype=torch.bfloat16,",
    "    device_map={'': 0},",
    "    low_cpu_mem_usage=True,",
    ")",
    "model.eval()",
    "device = next(model.parameters()).device",
    "print(f'Model on {device}, n_layers={len(model.model.layers)}')",
))

cells.append(code(
    "# 5) Load WANDERING IIDs from assignment file (only need the IIDs, not donors)",
    "import json",
    "ASSETS = Path(REPO_DIR) / 'paper_mega_v4_pairs'",
    "assignment = json.load(open(ASSETS / 'exp_b_wandering_assignment.json'))",
    "wand_iids_all = list(assignment.keys())",
    "print(f'WANDERING IIDs: {len(wand_iids_all)}')",
    "",
    "from datasets import load_dataset",
    "ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')",
    "by_iid = {ex['instance_id']: ex for ex in ds}",
    "wand_iids = [iid for iid in wand_iids_all if iid in by_iid]",
    "print(f'WANDERING runnable: {len(wand_iids)}/{len(wand_iids_all)}')",
))

cells.append(code(
    "# 6) Runner setup — SAME config as Exp B Primary (max_turns=50, default seed)",
    "from runner import Runner",
    "from instrumentation.capture import audit_captures",
    "from config import HarnessConfig, DEFAULT",
    "from dataclasses import replace",
    "import subprocess",
    "",
    "cfg = replace(",
    "    DEFAULT,",
    "    work_root=Path('/content/work_baseline'),",
    "    capture_root=BASELINE_DRIVE / 'captures',",
    "    trace_root=BASELINE_DRIVE / 'traces',",
    "    max_turns=50,",
    "    bash_max_output_bytes=32_000,",
    ")",
    "runner = Runner(model=model, tokenizer=tok, config=cfg)",
    "print(f'max_turns={cfg.max_turns}')",
    "",
    "def prep_workdir(instance, workdir):",
    "    workdir = Path(workdir)",
    "    if (workdir / '.git').exists(): return",
    "    repo, base = instance['repo'], instance['base_commit']",
    "    workdir.mkdir(parents=True, exist_ok=True)",
    "    subprocess.run(['git','clone','--quiet',f'https://github.com/{repo}',str(workdir)], check=True)",
    "    subprocess.run(['git','-c','advice.detachedHead=false','checkout','--quiet',base], cwd=str(workdir), check=True)",
    "    tp = instance.get('test_patch')",
    "    if tp:",
    "        subprocess.run(['git','apply','--allow-empty','-'], input=tp, text=True, cwd=str(workdir))",
    "    subprocess.run(['git','add','-A'], cwd=str(workdir), check=False)",
    "    subprocess.run(['git','commit','--quiet','--no-verify','-m','baseline','--allow-empty'], cwd=str(workdir), check=False)",
    "print('Runner ready, prep_workdir defined.')",
))

cells.append(md(
    "---",
    "## Run Baseline (no hook)",
    "",
    "Same code path as Exp B Primary except NO LayerPatch is installed. Pure agent loop with default seed.",
))

cells.append(code(
    "# 7) Baseline execution loop — NO hook, just runner.run_one() directly",
    "import json, time",
    "",
    "CKPT = BASELINE_DRIVE / 'results.json'",
    "results = json.loads(CKPT.read_text()) if CKPT.exists() else {}",
    "print(f'Resuming Baseline: {len(results)} of {len(wand_iids)} done')",
    "",
    "t_start = time.time()",
    "for i, iid in enumerate(wand_iids):",
    "    if iid in results and results[iid].get('finished') is not None:",
    "        continue",
    "    repo = assignment[iid]['repo']",
    "    print(f'[{i+1:2d}/{len(wand_iids)}] RUN  {iid[:50]} | repo={repo}')",
    "    torch.cuda.empty_cache()",
    "    torch.cuda.reset_peak_memory_stats()",
    "    t0 = time.time()",
    "    try:",
    "        out = runner.run_one(by_iid[iid], prepare_workdir_fn=prep_workdir)",
    "    except Exception as e:",
    "        out = {'instance_id': iid, 'finished': False,",
    "               'finish_reason':'error', 'error': f'{type(e).__name__}: {e}'}",
    "    out['wall_seconds'] = time.time() - t0",
    "    out['vram_peak_gb'] = round(torch.cuda.max_memory_allocated() / (1024**3), 1)",
    "    out['run_tag'] = 'exp_b_baseline_nohook'",
    "    results[iid] = out",
    "    CKPT.write_text(json.dumps(results, indent=2, default=str))",
    "    elapsed = (time.time() - t_start) / 60",
    "    n_done = sum(1 for r in results.values() if r.get('finished') is not None)",
    "    avg = elapsed / max(1, n_done)",
    "    eta = avg * (len(wand_iids) - n_done)",
    "    print(f'  wall={out[\"wall_seconds\"]/60:.1f}min  finish={out[\"finish_reason\"]:14s}  '",
    "          f'turns={out.get(\"n_turns\",0):2d}  vram={out.get(\"vram_peak_gb\",0):.1f}GB  '",
    "          f'| elapsed {elapsed:.0f}m, ETA {eta:.0f}m')",
    "",
    "print(f'\\n=== Exp B Baseline COMPLETE: {len(results)}/{len(wand_iids)} ===')",
))

cells.append(code(
    "# 8) Verdict — compare baseline vs primary",
    "import json",
    "from scipy.stats import fisher_exact",
    "",
    "baseline = json.loads(CKPT.read_text())",
    "primary  = json.loads((PRIMARY_DRIVE / 'results.json').read_text())",
    "",
    "def count_ft(d):",
    "    return sum(1 for r in d.values() if r.get('finish_reason') == 'finish_tool')",
    "",
    "n_base = len(baseline)",
    "n_prim = len(primary)",
    "ft_base = count_ft(baseline)",
    "ft_prim = count_ft(primary)",
    "",
    "print(f'=== Comparison ===')",
    "print(f'  Baseline (no hook):    {ft_base}/{n_base} = {100*ft_base/n_base:.0f}% finish_tool')",
    "print(f'  Primary (hook α=0.3):  {ft_prim}/{n_prim} = {100*ft_prim/n_prim:.0f}% finish_tool')",
    "print()",
    "",
    "# Fisher's exact",
    "#         finish_tool  not_finish_tool",
    "# primary  ft_prim      n_prim - ft_prim",
    "# baseline ft_base      n_base - ft_base",
    "table = [[ft_prim, n_prim - ft_prim],",
    "         [ft_base, n_base - ft_base]]",
    "print(f'Contingency table:')",
    "print(f'              finish_tool  NOT')",
    "print(f'  primary       {table[0][0]:>3d}        {table[0][1]:>3d}')",
    "print(f'  baseline      {table[1][0]:>3d}        {table[1][1]:>3d}')",
    "",
    "or_, p_two = fisher_exact(table, alternative='two-sided')",
    "_, p_greater = fisher_exact(table, alternative='greater')  # primary > baseline?",
    "print(f'\\nFishers exact:')",
    "print(f'  Odds ratio:               {or_:.3f}')",
    "print(f'  p-value (two-sided):      {p_two:.4f}')",
    "print(f'  p-value (primary > base): {p_greater:.4f}')",
    "",
    "# Per-instance flip table (which instances differ between conditions)",
    "print(f'\\n=== Per-instance comparison ===')",
    "print(f'{\"iid\":<50}  {\"baseline\":>14}  {\"primary\":>14}  changed?')",
    "for iid in primary:",
    "    if iid not in baseline: continue",
    "    bfr = baseline[iid].get('finish_reason','')",
    "    pfr = primary[iid].get('finish_reason','')",
    "    marker = '★' if bfr != pfr else ''",
    "    print(f'  {iid[:50]:<50}  {bfr:>14}  {pfr:>14}  {marker}')",
    "",
    "# Verdict",
    "diff = ft_prim - ft_base",
    "if p_two < 0.05 and ft_prim > ft_base:",
    "    print(f'\\n>>> CAUSAL — hook produces {diff} more finish_tool emissions, p={p_two:.4f}')",
    "elif p_two < 0.1 and ft_prim > ft_base:",
    "    print(f'\\n>>> Weak causal — borderline significance, p={p_two:.4f}')",
    "elif ft_prim == ft_base:",
    "    print(f'\\n>>> No effect — hook does not alter flip rate')",
    "elif ft_prim < ft_base:",
    "    print(f'\\n>>> REVERSE — hook DECREASES finish_tool emissions (unusual)')",
    "else:",
    "    print(f'\\n>>> NOT SIGNIFICANT — {ft_prim} vs {ft_base}, p={p_two:.4f}')",
))

cells.append(md(
    "## After completion",
    "",
    "If CAUSAL (p<0.05) → write paper #2 §4 with this finding. Pulls Exp C onto critical path.",
    "If NOT SIGNIFICANT → either α=0.3 too weak (try α=0.7), or L55 not causal lever. Pivot Exp C as confirmatory.",
    "",
    "Local analysis: `python3 scripts/analyze_exp_b_paired.py` will produce paper-ready figure of flip-rate comparison + McNemar's test on the paired design.",
))

# Write
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "machine_shape": "hm"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f'Wrote {OUT}')
print(f'Cells: {len(cells)}')
print(f'Size: {OUT.stat().st_size:,} bytes')
