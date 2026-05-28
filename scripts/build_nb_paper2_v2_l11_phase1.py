"""Build Paper #2 v2 L11 Phase 1 notebook — full N=20 × two magnitudes.

Follows pilot (N=5) which revealed per-instance rescue-magnitude windows:
  - qutebrowser-0d2afd58 rescued @ α=1.15 (not 0.70)
  - openlibrary-09865f5f rescued @ α=0.70 (crashed @ 1.15)

Phase 1 design:
  - All 20 WANDERING iids × α ∈ {0.7, 1.15} (resume skips 10 pilot-done)
  - No-hook determinism check (5 iids) — confirm RTX 6000 baseline ~0/20
  - Random-direction control (targeted on rescued subset) — direction-specificity
  - Enhanced analysis: per-instance best-α, rescue-at-either, Fisher exact vs 0/20

Uses SAME DRIVE_ROOT/runs as pilot for resume. Analysis filters to α ∈ {0.7, 1.15}.

Compute: ~30 new hook + 5 no-hook + ~6 control = ~41 runs ≈ ~16h RTX 6000 (~$18)

Output: notebooks/nb_paper2_v2_l11_phase1.ipynb
"""
import json
from pathlib import Path

OUT = Path('/Volumes/SSD Major/fish/openinterp-swebench-harness/notebooks/nb_paper2_v2_l11_phase1.ipynb')


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}

def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": "\n".join(lines)}


cells = []

cells.append(md(
    "# Paper #2 v2 — L11 Phase 1 (full N=20 × two magnitudes)",
    "",
    "**Pilot (N=5) revealed per-instance rescue-magnitude windows.** Phase 1 tests at scale.",
    "",
    "## Pilot recap",
    "- qutebrowser-0d2afd58: rescued @ α=1.15 (too weak @ 0.70)",
    "- openlibrary-09865f5f: rescued @ α=0.70 (CRASHED @ 1.15)",
    "- → Each WANDERING instance has its own rescue-magnitude window",
    "",
    "## Phase 1 design",
    "- **Main**: all 20 WANDERING × α ∈ {0.7, 1.15} (resume skips 10 pilot-done → 30 new runs)",
    "- **Determinism**: 5 no-hook runs (confirm RTX 6000 baseline ~0/20 on THIS instance)",
    "- **Control**: random-direction on rescued subset (direction-specificity)",
    "",
    "## Pre-registered tests",
    "- PRIMARY: rescue-at-EITHER-α vs baseline 0/20 → Fisher exact, significant if >=4/20",
    "- SECONDARY: per-instance best-magnitude mapping; crash rate per α; per-repo patterns",
    "- CONTROL: SUCCESS-donor rescue rate > random-direction rescue rate",
    "",
    "## Compute: ~41 runs ≈ ~16h RTX 6000 (~$18)",
    "Resume-safe: per-run Drive checkpoint. Re-run any cell after disconnect.",
))

# Setup cells (identical to pilot)
cells.append(code(
    "# 0) GPU pre-flight",
    "import subprocess",
    "out = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],",
    "                     capture_output=True, text=True).stdout.strip()",
    "print(out)",
    "vram_gb = int(out.split(',')[1].strip().split()[0]) // 1024",
    "assert vram_gb >= 70, f'Need >=70GB VRAM, got {vram_gb}GB.'",
    "print(f'VRAM OK: {vram_gb}GB')",
))

cells.append(code(
    "# 1) Install",
    "%%capture",
    "!pip install -q --upgrade pip",
    "!pip uninstall -y transformers > /dev/null 2>&1 || true",
    "!pip install -q git+https://github.com/huggingface/transformers.git",
    "!pip install -q accelerate sentencepiece safetensors einops",
    "!pip install -q causal-conv1d --no-build-isolation",
    "!pip install -q fla-org --no-build-isolation || pip install -q flash-linear-attention",
    "!pip install -q huggingface_hub datasets scipy",
))

cells.append(code(
    "# 1b) Log transformers commit",
    "import transformers, subprocess, os",
    "site = os.path.dirname(transformers.__file__)",
    "h = subprocess.run(['git','-C',site,'rev-parse','HEAD'], capture_output=True, text=True).stdout.strip()",
    "print(f'transformers {transformers.__version__}, commit {h[:12] if h else \"(not git)\"}')",
))

cells.append(code(
    "# 2) Drive + paths (SAME dir as pilot for resume)",
    "from google.colab import drive",
    "drive.mount('/content/drive')",
    "from pathlib import Path",
    "DRIVE_ROOT = Path('/content/drive/MyDrive/openinterp_runs/swebench_paper2_v2_l11_pilot')",
    "RUNS_DIR = DRIVE_ROOT / 'runs'",
    "RUNS_DIR.mkdir(parents=True, exist_ok=True)",
    "BASELINE_DIR = Path('/content/drive/MyDrive/openinterp_runs/swebench_exp_b_c/exp_b_baseline_nohook')",
    "print(f'Runs dir (resume-aware) -> {RUNS_DIR}')",
    "existing = list(RUNS_DIR.glob('*.json'))",
    "print(f'Existing run files (pilot): {len(existing)}')",
))

cells.append(code(
    "# 3) Pull harness",
    "import os, subprocess",
    "REPO_DIR = '/content/openinterp-swebench-harness'",
    "if not os.path.exists(REPO_DIR):",
    "    !git clone --quiet https://github.com/OpenInterpretability/openinterp-swebench-harness.git $REPO_DIR",
    "else:",
    "    !cd $REPO_DIR && git fetch --quiet && git reset --hard origin/main",
    "%cd $REPO_DIR",
    "import sys",
    "if REPO_DIR not in sys.path: sys.path.insert(0, REPO_DIR)",
    "print(subprocess.check_output(['git','log','-1','--oneline']).decode().strip())",
))

cells.append(code(
    "# 4) Load Qwen3.6-27B",
    "import torch",
    "from transformers import AutoTokenizer, AutoModelForCausalLM",
    "MODEL_NAME = 'Qwen/Qwen3.6-27B'",
    "tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)",
    "model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True,",
    "    torch_dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True)",
    "model.eval()",
    "device = next(model.parameters()).device",
    "print(f'Model on {device}, n_layers={len(model.model.layers)}, d_model={model.config.hidden_size}')",
))

cells.append(code(
    "# 5) Load L11 donors",
    "import json, safetensors.torch as st",
    "ASSETS = Path(REPO_DIR) / 'paper_mega_v4_pairs'",
    "l11_donors = st.load_file(str(ASSETS / 'exp_l11_donors.safetensors'))",
    "print(f'Loaded {len(l11_donors)} donors')",
))

cells.append(code(
    "# 6) ALL 20 WANDERING iids (from baseline_nohook)",
    "baseline = json.load(open(BASELINE_DIR / 'results.json'))",
    "ALL_IIDS = list(baseline.keys()) if isinstance(baseline, dict) else [r['iid'] for r in baseline]",
    "print(f'All WANDERING iids: {len(ALL_IIDS)}')",
    "def repo_of(iid):",
    "    if 'qutebrowser' in iid: return 'qutebrowser'",
    "    if 'openlibrary' in iid: return 'openlibrary'",
    "    if 'ansible' in iid: return 'ansible'",
    "    raise ValueError(iid)",
    "from collections import Counter",
    "print(f'Repo dist: {Counter(repo_of(i) for i in ALL_IIDS)}')",
))

cells.append(code(
    "# 7) Runner + LayerPatch + dataset",
    "from instrumentation.layerpatch import LayerPatch",
    "from runner import Runner",
    "from config import DEFAULT",
    "from dataclasses import replace",
    "from datasets import load_dataset",
    "ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')",
    "by_iid = {ex['instance_id']: ex for ex in ds}",
    "missing = [i for i in ALL_IIDS if i not in by_iid]",
    "assert not missing, f'Missing: {missing}'",
    "cfg = replace(DEFAULT, work_root=Path('/content/work_l11_p1'),",
    "    capture_root=DRIVE_ROOT/'captures', trace_root=DRIVE_ROOT/'traces',",
    "    capture_layers=[], max_turns=50, bash_max_output_bytes=32_000)",
    "runner = Runner(model=model, tokenizer=tok, config=cfg)",
    "print('Runner ready.')",
))

cells.append(code(
    "# 8) workdir prep + run_with_hook helpers",
    "import subprocess, time, gc",
    "def prep_workdir(instance, workdir):",
    "    workdir = Path(workdir)",
    "    if (workdir/'.git').exists(): return",
    "    repo, base = instance['repo'], instance['base_commit']",
    "    workdir.mkdir(parents=True, exist_ok=True)",
    "    subprocess.run(['git','clone','--quiet',f'https://github.com/{repo}',str(workdir)], check=True)",
    "    subprocess.run(['git','-c','advice.detachedHead=false','checkout','--quiet',base], cwd=str(workdir), check=True)",
    "    tp = instance.get('test_patch')",
    "    if tp: subprocess.run(['git','apply','--allow-empty','-'], input=tp, text=True, cwd=str(workdir))",
    "    subprocess.run(['git','add','-A'], cwd=str(workdir), check=False)",
    "    subprocess.run(['git','commit','--quiet','--no-verify','-m','baseline','--allow-empty'], cwd=str(workdir), check=False)",
    "",
    "def run_with_hook(instance, *, layer, donor_tensor, alpha, mode='add', run_tag=''):",
    "    if donor_tensor is not None:",
    "        eff = donor_tensor.to(device).to(model.dtype) * float(alpha)",
    "        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()",
    "        patch = LayerPatch(model, {layer: eff}, mode=mode); patch.attach()",
    "    else:",
    "        patch = None",
    "    t0 = time.time()",
    "    try:",
    "        out = runner.run_one(instance, prepare_workdir_fn=prep_workdir)",
    "    except Exception as e:",
    "        out = {'instance_id': instance['instance_id'], 'finished': False,",
    "               'finish_reason':'error','error':f'{type(e).__name__}: {e}'}",
    "    finally:",
    "        if patch is not None:",
    "            out_ncalls = patch.n_calls_per_layer(); patch.detach()",
    "        else:",
    "            out_ncalls = {}",
    "    out['wall_seconds'] = time.time()-t0",
    "    out['hook_layer'] = layer if donor_tensor is not None else None",
    "    out['hook_alpha'] = float(alpha) if donor_tensor is not None else 0.0",
    "    out['run_tag'] = run_tag",
    "    gc.collect(); torch.cuda.empty_cache()",
    "    return out",
))

# Determinism check
cells.append(code(
    "# 9) DETERMINISM CHECK — 5 no-hook runs on current RTX 6000",
    "# Confirms baseline ~0/N on THIS instance (Phase 6 originals were 0/20 by definition)",
    "DETERM_IIDS = ALL_IIDS[:5]",
    "print('Determinism check (no-hook) on 5 iids:')",
    "determ = {}",
    "for iid in DETERM_IIDS:",
    "    tag = f'{iid}__nohook'",
    "    p = RUNS_DIR / f'{tag}.json'",
    "    if p.exists():",
    "        determ[iid] = json.load(open(p)); print(f'  SKIP {iid[:50]} (exists): {determ[iid].get(\"finish_reason\")}')",
    "        continue",
    "    out = run_with_hook(by_iid[iid], layer=11, donor_tensor=None, alpha=0.0, run_tag=tag)",
    "    out['iid']=iid; out['condition']='nohook'",
    "    json.dump(out, open(p,'w'), indent=2); determ[iid]=out",
    "    print(f'  {iid[:50]}: {out.get(\"finish_reason\")} T{out.get(\"n_turns\")} ({out[\"wall_seconds\"]:.0f}s)')",
    "n_flip = sum(1 for o in determ.values() if o.get('finish_reason')=='finish_tool')",
    "print(f'\\nDeterminism: {n_flip}/5 finish_tool (expect 0 if RTX 6000 deterministic vs Phase 6)')",
    "print('  -> 0/5: clean baseline, hook rescues are causal')",
    "print('  -> >0/5: hardware drift, subtract from hook rescue rate')",
))

# Main alpha-sweep
cells.append(code(
    "# 10) MAIN: all 20 iids × α ∈ {0.7, 1.15} (resume skips pilot-done)",
    "ALPHAS = [0.7, 1.15]",
    "HOOK_LAYER = 11",
    "results = {}",
    "for iid in ALL_IIDS:",
    "    repo = repo_of(iid)",
    "    donor = l11_donors[f'L11_donor_success_{repo}']",
    "    for alpha in ALPHAS:",
    "        tag = f'{iid}__alpha_{alpha:.2f}'",
    "        p = RUNS_DIR / f'{tag}.json'",
    "        if p.exists():",
    "            results[tag] = json.load(open(p)); continue",
    "        print(f'{repo[:4]} {iid[40:60]} α={alpha:.2f}: running...')",
    "        out = run_with_hook(by_iid[iid], layer=HOOK_LAYER, donor_tensor=donor, alpha=alpha, run_tag=tag)",
    "        out['iid']=iid; out['alpha']=alpha; out['donor_key']=f'L11_donor_success_{repo}'",
    "        json.dump(out, open(p,'w'), indent=2); results[tag]=out",
    "        print(f'    {out.get(\"finish_reason\")} T{out.get(\"n_turns\")} ({out[\"wall_seconds\"]:.0f}s)')",
    "print(f'\\nMain done. {len(results)} runs in results.')",
))

# Enhanced analysis
cells.append(code(
    "# 11) ENHANCED ANALYSIS — per-instance windows + rescue-at-either + Fisher",
    "from collections import defaultdict",
    "from scipy.stats import fisher_exact",
    "",
    "# Reload all hook runs (α in 0.7, 1.15)",
    "hook_runs = defaultdict(dict)  # iid -> {alpha: outcome}",
    "for p in RUNS_DIR.glob('*__alpha_*.json'):",
    "    r = json.load(open(p)); a = r.get('alpha')",
    "    if a in (0.7, 1.15): hook_runs[r['iid']][a] = r.get('finish_reason')",
    "",
    "# Per-instance best outcome (rescue at EITHER α)",
    "rescued_either = []; per_inst = {}",
    "for iid, by_a in hook_runs.items():",
    "    outcomes = list(by_a.values())",
    "    rescued = any(o == 'finish_tool' for o in by_a.values())",
    "    crashed = any(o == 'invalid_tools' for o in by_a.values())",
    "    rescue_a = [a for a,o in by_a.items() if o=='finish_tool']",
    "    per_inst[iid] = {'outcomes': by_a, 'rescued': rescued, 'crashed': crashed, 'rescue_alpha': rescue_a}",
    "    if rescued: rescued_either.append(iid)",
    "",
    "n_total = len(hook_runs)",
    "n_rescued = len(rescued_either)",
    "print(f'Instances tested (both α): {n_total}')",
    "print(f'Rescued at EITHER α: {n_rescued}/{n_total} = {n_rescued/max(n_total,1):.1%}')",
    "",
    "# Per-instance window table",
    "print('\\nPer-instance windows:')",
    "for iid, d in sorted(per_inst.items()):",
    "    o07 = d['outcomes'].get(0.7,'?'); o115 = d['outcomes'].get(1.15,'?')",
    "    flag = ' <-- RESCUE' if d['rescued'] else ('  (crash)' if d['crashed'] else '')",
    "    print(f'  {iid[40:65]:<26} 0.7={o07:<13} 1.15={o115:<13}{flag}')",
    "",
    "# Fisher exact vs baseline 0/N (use determinism result)",
    "n_baseline_flip = sum(1 for o in determ.values() if o.get('finish_reason')=='finish_tool') if 'determ' in dir() else 0",
    "# Contingency: [[hook_rescue, hook_norescue],[base_rescue, base_norescue]]",
    "# Baseline: use determinism N (5) OR definitional 0/20",
    "table = [[n_rescued, n_total - n_rescued], [0, 20]]  # vs definitional 0/20 baseline",
    "odds, p = fisher_exact(table, alternative='greater')",
    "print(f'\\nFisher exact (rescue {n_rescued}/{n_total} vs baseline 0/20):')",
    "print(f'  odds ratio: {odds:.2f}, p-value (one-sided): {p:.4f}')",
    "print(f'  {\"SIGNIFICANT\" if p < 0.05 else \"not significant\"} at α=0.05')",
    "",
    "# Per-repo + crash characterization",
    "print('\\nPer-α outcome counts:')",
    "for a in [0.7, 1.15]:",
    "    outs = [d['outcomes'].get(a) for d in per_inst.values() if a in d['outcomes']]",
    "    ft = outs.count('finish_tool'); mt = outs.count('max_turns'); it = outs.count('invalid_tools')",
    "    print(f'  α={a}: finish_tool={ft}, max_turns={mt}, invalid_tools={it}, total={len(outs)}')",
    "",
    "summary = {'n_total': n_total, 'n_rescued_either': n_rescued,",
    "           'rescue_rate': n_rescued/max(n_total,1), 'fisher_p': float(p), 'odds_ratio': float(odds),",
    "           'per_instance': {k: {'o07': v['outcomes'].get(0.7), 'o115': v['outcomes'].get(1.15),",
    "                                 'rescued': v['rescued']} for k,v in per_inst.items()},",
    "           'rescued_iids': rescued_either}",
    "json.dump(summary, open(DRIVE_ROOT/'phase1_summary.json','w'), indent=2)",
    "print(f'\\nSaved {DRIVE_ROOT}/phase1_summary.json')",
))

# Random direction control
cells.append(code(
    "# 12) RANDOM-DIRECTION CONTROL (targeted on rescued subset)",
    "# Direction-specificity: does SUCCESS-donor rescue MORE than random-direction at same α?",
    "rng_donor = l11_donors['L11_random_direction_norm_matched']",
    "control_results = {}",
    "if rescued_either:",
    "    print(f'Running random-direction control on {len(rescued_either)} rescued iids at their rescue-α:')",
    "    for iid in rescued_either:",
    "        for a in per_inst[iid]['rescue_alpha']:",
    "            tag = f'{iid}__RANDOM_alpha_{a:.2f}'",
    "            p = RUNS_DIR / f'{tag}.json'",
    "            if p.exists(): control_results[tag]=json.load(open(p)); print(f'  SKIP {tag[40:]}'); continue",
    "            out = run_with_hook(by_iid[iid], layer=11, donor_tensor=rng_donor, alpha=a, run_tag=tag)",
    "            out['iid']=iid; out['alpha']=a; out['condition']='random_direction'",
    "            json.dump(out, open(p,'w'), indent=2); control_results[tag]=out",
    "            print(f'  {iid[40:60]} RANDOM α={a:.2f}: {out.get(\"finish_reason\")} T{out.get(\"n_turns\")}')",
    "    n_rand_rescue = sum(1 for o in control_results.values() if o.get('finish_reason')=='finish_tool')",
    "    print(f'\\nRandom-direction rescued: {n_rand_rescue}/{len(control_results)}')",
    "    print(f'SUCCESS-donor rescued: {n_rescued}/{n_total}')",
    "    print('  -> random << success: SUCCESS-direction is specific (causal)')",
    "    print('  -> random ~ success: effect is perturbation magnitude, not direction')",
    "else:",
    "    print('No rescued instances — skip control.')",
))

# Shutdown
cells.append(code(
    "# 13) Save + shutdown",
    "import time, os",
    "print('=== Flushing Drive ===')",
    "try:",
    "    from google.colab import drive; drive.flush_and_unmount(); print('flushed.')",
    "except Exception as e: print(f'(skip flush: {e})')",
    "print('\\n=== SHUTDOWN IN 30s (Kernel->Interrupt to cancel) ===')",
    "time.sleep(30)",
    "os.system('sudo shutdown -h now')",
))

cells.append(md(
    "## Phase 1 outputs (in DRIVE_ROOT)",
    "- `phase1_summary.json` — rescue rate, Fisher p, per-instance windows",
    "- `runs/*.json` — all per-run details (hook + nohook + random control)",
    "",
    "## Next (local):",
    "1. Sync phase1_summary.json + runs/ to repo",
    "2. Paper #2 v2 writeup: title depends on rescue rate + Fisher significance",
    "   - If significant + random-control clean: 'L11 as Causal Edge-Locus with Per-Instance Windows'",
    "   - If marginal: 'L11 Borderline Causal: Per-Instance Magnitude Sensitivity'",
))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
      "language_info": {"name":"python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=2))
print(f"Wrote {OUT} ({len(cells)} cells)")
