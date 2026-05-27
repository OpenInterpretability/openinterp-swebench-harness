"""Build Paper #2 v2 L11 SUCCESS-donor α-sweep PILOT notebook.

Companion to paper #2 Exp B (L55) — extends to L11 per paper #3 Phase 2 finding
that L11 edge-layer drift is dominant WANDERING discriminator.

Design:
  - PILOT scale: N=5 WANDERING × 4 α values = 20 runs
  - α values NORM-MATCHED to L55 Exp B (Exp B used α=0.3 with L55_norm ~127;
    L11_norm ~33, so α=1.15 at L11 ≈ Exp B perturbation magnitude)
    Sweep: α ∈ {0.35, 1.15, 2.3, 4.6} = weak / Exp-B-equivalent / 2× / 4×
  - mode='add' (per feedback_always_on_replace_destructive)
  - SUCCESS donor per-repo (same matching as Exp B)
  - Compares to existing baseline_nohook (n=20 in Drive)

Compute estimate: 20 runs × ~20min = ~7h H100 (~$14 vast.ai @ $2/h)

Output: notebooks/nb_paper2_v2_l11_pilot.ipynb
"""
import json
from pathlib import Path

OUT = Path('/Volumes/SSD Major/fish/openinterp-swebench-harness/notebooks/nb_paper2_v2_l11_pilot.ipynb')


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}

def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": "\n".join(lines)}


cells = []

# Title
cells.append(md(
    "# Paper #2 v2 — L11 SUCCESS-donor α-sweep PILOT",
    "",
    "**Tests paper #3 Phase 2 finding causally**: L11 edge-layer drift is dominant WANDERING discriminator (stability_freq=0.95). If L11 is the causal locus (not L55 as Exp B initially assumed), injecting SUCCESS-derived L11 residual should rescue WANDERING agents toward `finish_tool`.",
    "",
    "## Refinement of paper #2 Exp B",
    "- Exp B (L55, α=0.3, mode=add): **NULL** (6/20 primary vs baseline 7/20)",
    "- Interpretation #1 from paper #2 nulls: L55 not be the right edge-locus",
    "- Paper #3 supervised contrast: L11_drift_first_last selection_freq=0.95 (top WANDERING)",
    "- Paper #3 Phase 4 cross-hardware: L11_cosine_consec_late top predictor (coef=-1.76)",
    "- → Double independent confirmation: L11 (not L55) is the real edge-layer locus",
    "",
    "## α-sweep design (norm-matched to Exp B)",
    "- L11 donor norm ~33 vs L55 norm ~127 → α=0.3 at L11 is 4× WEAKER than Exp B equivalent",
    "- Sweep α ∈ {0.35, 1.15, 2.3, 4.6}: weak / Exp-B-equivalent / 2× / 4×",
    "- Per `feedback-steering-structural-rigidity-diagnostic`: structural-rigidity check at 4 scales",
    "",
    "## Compute",
    "- 5 WANDERING × 4 α = 20 runs × ~20min = ~7h H100 (~$14)",
    "- Resume-on-disconnect via per-instance Drive checkpoint (same pattern as Exp B)",
    "",
    "## Pre-computed assets",
    "- `paper_mega_v4_pairs/exp_l11_donors.safetensors` — per-repo L11 SUCCESS + LOCKED donors",
    "- `paper_mega_v4_pairs/exp_l11_donors_metadata.json` — norms + lock_turns",
))

# Setup
cells.append(code(
    "# 0) GPU pre-flight",
    "import subprocess",
    "out = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],",
    "                     capture_output=True, text=True).stdout.strip()",
    "print(out)",
    "vram_gb = int(out.split(',')[1].strip().split()[0]) // 1024",
    "assert vram_gb >= 70, f'Need >=70GB VRAM, got {vram_gb}GB. Aborting.'",
    "print(f'VRAM OK: {vram_gb}GB')",
))

cells.append(code(
    "# 1) Install — transformers main + fla + accelerate (per Caio pinning rule)",
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
    "# 1b) Log transformers commit",
    "import transformers, subprocess, os",
    "print(f'transformers version: {transformers.__version__}')",
    "site = os.path.dirname(transformers.__file__)",
    "h = subprocess.run(['git','-C',site,'rev-parse','HEAD'], capture_output=True, text=True).stdout.strip()",
    "print(f'transformers commit: {h[:12] if h else \"(not git)\"}')",
))

cells.append(code(
    "# 2) Drive mount + paths",
    "from google.colab import drive",
    "drive.mount('/content/drive')",
    "",
    "from pathlib import Path",
    "DRIVE_ROOT = Path('/content/drive/MyDrive/openinterp_runs/swebench_paper2_v2_l11_pilot')",
    "DRIVE_ROOT.mkdir(parents=True, exist_ok=True)",
    "PHASE6_DRIVE = Path('/content/drive/MyDrive/openinterp_runs/swebench_v6_phase6')",
    "BASELINE_DIR = Path('/content/drive/MyDrive/openinterp_runs/swebench_exp_b_c/exp_b_baseline_nohook')",
    "print(f'Output -> {DRIVE_ROOT}')",
    "print(f'Phase 6 -> {PHASE6_DRIVE}')",
    "print(f'Baseline (compare against) -> {BASELINE_DIR}')",
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
    "# 4) Load Qwen3.6-27B",
    "import torch",
    "from transformers import AutoTokenizer, AutoModelForCausalLM",
    "MODEL_NAME = 'Qwen/Qwen3.6-27B'",
    "print(f'Loading {MODEL_NAME}...')",
    "tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)",
    "model = AutoModelForCausalLM.from_pretrained(",
    "    MODEL_NAME, trust_remote_code=True,",
    "    torch_dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True,",
    ")",
    "model.eval()",
    "device = next(model.parameters()).device",
    "print(f'Model on {device}, n_layers={len(model.model.layers)}, d_model={model.config.hidden_size}')",
))

# Load L11 donors
cells.append(code(
    "# 5) Load L11 donors",
    "import json, safetensors.torch as st",
    "ASSETS = Path(REPO_DIR) / 'paper_mega_v4_pairs'",
    "l11_donors = st.load_file(str(ASSETS / 'exp_l11_donors.safetensors'))",
    "l11_meta = json.load(open(ASSETS / 'exp_l11_donors_metadata.json'))",
    "print('=== L11 donors (L2 norm) ===')",
    "for k, t in sorted(l11_donors.items()):",
    "    print(f'  {k:50s}  shape={tuple(t.shape)}  L2={t.float().norm().item():.2f}')",
))

# Pick pilot instances
cells.append(code(
    "# 6) Pick 5 WANDERING pilot iids (same as baseline_nohook, stratified by repo)",
    "BASELINE_RESULTS = BASELINE_DIR / 'results.json'",
    "baseline = json.load(open(BASELINE_RESULTS))",
    "baseline_iids = list(baseline.keys()) if isinstance(baseline, dict) else [r['iid'] for r in baseline]",
    "print(f'Baseline iids: {len(baseline_iids)}')",
    "",
    "from collections import defaultdict",
    "import random; random.seed(42)",
    "by_repo = defaultdict(list)",
    "for iid in baseline_iids:",
    "    if 'qutebrowser' in iid: by_repo['qutebrowser'].append(iid)",
    "    elif 'openlibrary' in iid: by_repo['openlibrary'].append(iid)",
    "    elif 'ansible' in iid: by_repo['ansible'].append(iid)",
    "",
    "PILOT_IIDS = []",
    "for repo, n in [('qutebrowser', 2), ('openlibrary', 2), ('ansible', 1)]:",
    "    PILOT_IIDS.extend(sorted(by_repo[repo])[:n])  # deterministic",
    "print(f'Pilot iids ({len(PILOT_IIDS)}):')",
    "for iid in PILOT_IIDS:",
    "    print(f'  {iid}')",
))

# LayerPatch + runner + workdir
cells.append(code(
    "# 7) LayerPatch + runner + workdir prep (mirrors Exp B pattern)",
    "from instrumentation.layerpatch import LayerPatch",
    "from runner import Runner",
    "from config import DEFAULT",
    "from dataclasses import replace",
    "from datasets import load_dataset",
    "",
    "ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')",
    "by_iid = {ex['instance_id']: ex for ex in ds}",
    "missing = [iid for iid in PILOT_IIDS if iid not in by_iid]",
    "assert not missing, f'Pilot iids missing from dataset: {missing}'",
    "print(f'All {len(PILOT_IIDS)} pilot iids resolved.')",
    "",
    "cfg = replace(",
    "    DEFAULT,",
    "    work_root=Path('/content/work_l11_pilot'),",
    "    capture_root=DRIVE_ROOT / 'captures',",
    "    trace_root=DRIVE_ROOT / 'traces',",
    "    capture_layers=[],  # no captures needed for intervention",
    "    max_turns=50,",
    "    bash_max_output_bytes=32_000,",
    ")",
    "print(f'max_turns={cfg.max_turns}')",
    "",
    "runner = Runner(model=model, tokenizer=tok, config=cfg)",
    "print('Runner ready.')",
))

cells.append(code(
    "# 8) Workdir prep helper (clone + checkout base + test_patch)",
    "import subprocess",
    "",
    "def prep_workdir(instance, workdir):",
    "    workdir = Path(workdir)",
    "    if (workdir / '.git').exists(): return",
    "    repo, base = instance['repo'], instance['base_commit']",
    "    workdir.mkdir(parents=True, exist_ok=True)",
    "    subprocess.run(['git', 'clone', '--quiet', f'https://github.com/{repo}', str(workdir)], check=True)",
    "    subprocess.run(['git','-c','advice.detachedHead=false','checkout','--quiet',base], cwd=str(workdir), check=True)",
    "    tp = instance.get('test_patch')",
    "    if tp:",
    "        subprocess.run(['git','apply','--allow-empty','-'], input=tp, text=True, cwd=str(workdir))",
    "    subprocess.run(['git','add','-A'], cwd=str(workdir), check=False)",
    "    subprocess.run(['git','commit','--quiet','--no-verify','-m','baseline','--allow-empty'], cwd=str(workdir), check=False)",
))

cells.append(code(
    "# 9) run_with_hook helper (mirrors Exp B run_one_with_hook)",
    "import time, gc",
    "",
    "def run_with_hook(instance, *, layer, donor_tensor, alpha, mode='add', run_tag=''):",
    "    \"\"\"Single agent loop with LayerPatch always-on at given layer.\"\"\"",
    "    eff_patch = donor_tensor.to(device).to(model.dtype) * float(alpha) if mode == 'add' else donor_tensor.to(device).to(model.dtype)",
    "    torch.cuda.empty_cache()",
    "    torch.cuda.reset_peak_memory_stats()",
    "    patch = LayerPatch(model, {layer: eff_patch}, mode=mode)",
    "    patch.attach()",
    "    t0 = time.time()",
    "    try:",
    "        out = runner.run_one(instance, prepare_workdir_fn=prep_workdir)",
    "    except Exception as e:",
    "        out = {'instance_id': instance['instance_id'], 'finished': False,",
    "               'finish_reason':'error', 'error': f'{type(e).__name__}: {e}'}",
    "    finally:",
    "        ncalls = patch.n_calls_per_layer()",
    "        patch.detach()",
    "    out['wall_seconds'] = time.time() - t0",
    "    out['hook_n_calls'] = ncalls",
    "    out['hook_layer'] = layer",
    "    out['hook_mode'] = mode",
    "    out['hook_alpha'] = float(alpha)",
    "    out['hook_patch_l2'] = float(eff_patch.float().norm().item())",
    "    out['run_tag'] = run_tag",
    "    out['vram_peak_gb'] = round(torch.cuda.max_memory_allocated() / (1024**3), 1)",
    "    gc.collect(); torch.cuda.empty_cache()",
    "    return out",
))

# α-sweep execution
cells.append(code(
    "# 10) α-sweep execution: 5 iids × 4 α = 20 runs",
    "ALPHAS = [0.35, 1.15, 2.3, 4.6]  # norm-matched: weak / Exp-B-equivalent / 2× / 4×",
    "HOOK_LAYER = 11  # paper #3 Phase 2 dominant feature",
    "",
    "RUNS_DIR = DRIVE_ROOT / 'runs'",
    "RUNS_DIR.mkdir(parents=True, exist_ok=True)",
    "",
    "results = {}",
    "for iid in PILOT_IIDS:",
    "    if 'qutebrowser' in iid: repo = 'qutebrowser'",
    "    elif 'openlibrary' in iid: repo = 'openlibrary'",
    "    elif 'ansible' in iid: repo = 'ansible'",
    "    donor_key = f'L11_donor_success_{repo}'",
    "    donor = l11_donors[donor_key]",
    "    print(f'\\n=== {iid[:60]}... donor={donor_key} L2={donor.float().norm():.2f} ===')",
    "    for alpha in ALPHAS:",
    "        run_tag = f'{iid}__alpha_{alpha:.2f}'",
    "        out_path = RUNS_DIR / f'{run_tag}.json'",
    "        if out_path.exists():",
    "            print(f'  alpha={alpha:.2f}: SKIP (exists)')",
    "            results[run_tag] = json.load(open(out_path))",
    "            continue",
    "        print(f'  alpha={alpha:.2f}: running...')",
    "        out = run_with_hook(by_iid[iid], layer=HOOK_LAYER, donor_tensor=donor,",
    "                            alpha=alpha, mode='add', run_tag=run_tag)",
    "        out['donor_key'] = donor_key",
    "        out['iid'] = iid",
    "        out['alpha'] = alpha",
    "        json.dump(out, open(out_path, 'w'), indent=2)",
    "        results[run_tag] = out",
    "        print(f'    finish={out.get(\"finish_reason\")} turns={out.get(\"n_turns\")} ({out[\"wall_seconds\"]:.0f}s)')",
))

# Analysis
cells.append(code(
    "# 11) Per-α aggregate",
    "from collections import defaultdict",
    "",
    "all_results = {}",
    "for p in RUNS_DIR.glob('*__alpha_*.json'):",
    "    r = json.load(open(p))",
    "    all_results[p.stem] = r",
    "",
    "by_alpha = defaultdict(lambda: {'finish_tool': 0, 'max_turns': 0, 'error': 0, 'total': 0})",
    "for run_id, r in all_results.items():",
    "    a = r.get('alpha')",
    "    if a is None: continue",
    "    by_alpha[a]['total'] += 1",
    "    fr = r.get('finish_reason')",
    "    if fr == 'finish_tool': by_alpha[a]['finish_tool'] += 1",
    "    elif fr == 'max_turns': by_alpha[a]['max_turns'] += 1",
    "    else: by_alpha[a]['error'] += 1",
    "",
    "print(f'{\"alpha\":<8} {\"finish_tool\":<14} {\"max_turns\":<12} {\"error\":<8} {\"total\":<8}')",
    "for a in sorted(by_alpha.keys()):",
    "    s = by_alpha[a]",
    "    print(f'{a:<8.2f} {s[\"finish_tool\"]}/{s[\"total\"]:<13} {s[\"max_turns\"]}/{s[\"total\"]:<11} {s[\"error\"]:<8} {s[\"total\"]}')",
    "",
    "agg_path = DRIVE_ROOT / 'pilot_aggregate.json'",
    "json.dump({str(k): v for k, v in by_alpha.items()}, open(agg_path, 'w'), indent=2)",
    "print(f'\\nSaved {agg_path}')",
))

cells.append(code(
    "# 12) Decision gate",
    "any_flips = any(s['finish_tool'] >= 1 for s in by_alpha.values())",
    "max_rate = max((s['finish_tool'] / max(s['total'], 1) for s in by_alpha.values()), default=0)",
    "print(f'Max alpha flip rate: {max_rate:.1%}')",
    "",
    "if max_rate >= 0.4:",
    "    best_a = max(by_alpha, key=lambda a: by_alpha[a]['finish_tool'])",
    "    print(f'DECISION: Strong signal -> Phase 1 full N=20 at alpha={best_a}')",
    "elif max_rate >= 0.2:",
    "    print('DECISION: Marginal signal -> Phase 1 worth running, lower priority')",
    "elif any_flips:",
    "    print('DECISION: Weak signal -> consider Phase 1b (joint L11+L55)')",
    "else:",
    "    print('DECISION: NULL -> early exit; paper #2 ships as Three Honest Nulls + paper #3 stands on statistical')",
))

cells.append(md(
    "## Done — push aggregate back to repo for analysis",
    "",
    "Output files in `DRIVE_ROOT`:",
    "- `pilot_aggregate.json` — per-α flip rates",
    "- `runs/*.json` — per-run details (finish_reason, n_turns, hook calls, etc.)",
    "",
    "Next step (back on local):",
    "1. Sync DRIVE_ROOT to repo via `rsync` or manual download",
    "2. Compute Fisher exact / McNemar paired vs baseline_nohook (n=20, 7 flips)",
    "3. Apply decision gate -> Phase 1 or stop",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=2))
print(f"Wrote {OUT} ({len(cells)} cells)")
