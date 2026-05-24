"""Build the combined Exp B + Exp C notebook for paper #2 causal experiments.

Two experiments in one Colab session (~10-14h H100):
  Exp B Primary: L55 SUCCESS-donor patching, always-on hook from turn 0
                 19 WANDERING × primary (~6h) [controls C1-C3 deferred to Phase 2]
  Exp C Pilot:   v5 collapse direction α-sweep at L43 pre_tool (mode=add)
                 5 WANDERING × 5 α values × 2 directions (probe vs random)

Output: notebooks/nb_exp_b_c_combined_steering.ipynb
"""
import json
from pathlib import Path

OUT = Path('/Volumes/SSD Major/fish/openinterp-swebench-harness/notebooks/nb_exp_b_c_combined_steering.ipynb')


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}

def code(*lines):
    src = "\n".join(lines)
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


cells = []

# ---------------------------------------------------------------------------
# Title + design summary
# ---------------------------------------------------------------------------
cells.append(md(
    "# Exp B + Exp C — Causal Steering Experiments (paper #2)",
    "",
    "**Two causal experiments combined.** Tests the mid-layer-to-edge alignment hypothesis from Tool-Entropy paper §13 via two complementary steering interventions.",
    "",
    "## Exp B Primary — L55 SUCCESS-donor activation patching",
    "- For each of 19 WANDERING trajectories, install LayerPatch(L55, SUCCESS-donor, mode=replace) from turn 0",
    "- Per-repo donor: mean L55 pre_tool residual across matched SUCCESS trajectories",
    "- Run fresh agent loop with hook active throughout (max_turns=50)",
    "- Measure: did finish_tool emit? n_turns? tool_entropy_last10?",
    "- **Hypothesis**: persistent injection of SUCCESS-derived L55 residual biases WANDERING agents toward finish_tool emission",
    "",
    "## Exp C Pilot — v5 collapse direction α-sweep (L43 pre_tool, mode=add)",
    "- v5 direction = mean(WANDERING_residual) - mean(SUCCESS_residual) at L43 pre_tool late-half turns",
    "- 5 WANDERING × 5 α ∈ {-100, -20, 0, +20, +100} × 2 directions (probe vs random matched-norm)",
    "- Structural-rigidity α-sweep (per [[feedback-steering-structural-rigidity-diagnostic]])",
    "- **Hypothesis**: if v5 direction is CAUSAL, push -α should reduce WANDERING; push +α should increase",
    "",
    "## Compute estimate (H100 80GB)",
    "- Exp B: 19 runs × ~20min = ~6h",
    "- Exp C: 25 runs × ~20min = ~8h",
    "- Total: ~14h (vast.ai H100 @ ~$2/h ≈ $28)",
    "",
    "## Controls deferred (Phase 2 notebook)",
    "- C1 random direction L55, C2 LOCKED donor L55, C3 L43 mid-layer SUCCESS donor",
    "- C4 amplitude sweep on L55 donor (mode=add) at α ∈ {2,5,20}",
    "- Only run if Exp B primary shows ≥30% flip rate (else null is already informative)",
    "",
    "## Pre-computed assets (in repo: `paper_mega_v4_pairs/`)",
    "- `exp_b_donors.safetensors` — 13 donor tensors L43+L55 per (repo × class)",
    "- `exp_b_wandering_assignment.json` — 19 WANDERING → donor mapping (zero missing)",
    "- `exp_c_v5_direction.safetensors` — v5 collapse direction (normalized, d_model=5120)",
    "- `exp_c_v5_direction_metadata.json` — reference L2 norm (=58) for α scaling",
))

# ---------------------------------------------------------------------------
# Setup cells
# ---------------------------------------------------------------------------
cells.append(code(
    "# 0) GPU pre-flight — ABORT if not H100 / A100-80GB",
    "import subprocess",
    "out = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],",
    "                     capture_output=True, text=True).stdout.strip()",
    "print(out)",
    "vram_gb = int(out.split(',')[1].strip().split()[0]) // 1024",
    "assert vram_gb >= 70, f'Need ≥70GB VRAM, got {vram_gb}GB. Aborting.'",
    "print(f'VRAM OK: {vram_gb}GB')",
))

cells.append(code(
    "# 1) Install — fla MUST be present BEFORE model load (else GDN fallback = 10× slower)",
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
    "# 1b) Log commit hash of transformers (per Caio's pinning rule)",
    "import importlib, sys",
    "import transformers",
    "print(f'transformers version: {transformers.__version__}')",
    "import subprocess, os",
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
    "PHASE6_DRIVE = Path('/content/drive/MyDrive/openinterp_runs/swebench_v6_phase6')",
    "print(f'Output → {DRIVE_ROOT}')",
    "print(f'Phase 6 data → {PHASE6_DRIVE}')  # should exist with traces + phase6_results.json",
))

cells.append(code(
    "# 3) Pull harness latest (has agent loop, runner, instrumentation, donors, v5 direction)",
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
    "# 4) Load Qwen3.6-27B + tokenizer (single GPU, no offload — same as Phase 6)",
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
    "print(f'Model on {device}, dtype={model.dtype}, n_layers={len(model.model.layers)}')",
    "print(f'd_model={model.config.hidden_size}')",
))

# ---------------------------------------------------------------------------
# Load assets
# ---------------------------------------------------------------------------
cells.append(code(
    "# 5) Load pre-computed assets (donors + v5 direction + WANDERING assignment)",
    "import json",
    "import safetensors.torch as st",
    "",
    "ASSETS = Path(REPO_DIR) / 'paper_mega_v4_pairs'",
    "donors = st.load_file(str(ASSETS / 'exp_b_donors.safetensors'))",
    "v5dir  = st.load_file(str(ASSETS / 'exp_c_v5_direction.safetensors'))",
    "assignment = json.load(open(ASSETS / 'exp_b_wandering_assignment.json'))",
    "v5_meta = json.load(open(ASSETS / 'exp_c_v5_direction_metadata.json'))",
    "",
    "print('=== Donors ===')",
    "for k, t in donors.items():",
    "    print(f'  {k:50s}  shape={tuple(t.shape)}  L2={t.float().norm().item():.2f}')",
    "print()",
    "print('=== v5 direction ===')",
    "for k, t in v5dir.items():",
    "    print(f'  {k:50s}  shape={tuple(t.shape)}  L2={t.float().norm().item():.2f}')",
    "print(f'  reference residual L2 for α scaling: {v5_meta[\"reference_residual_l2_for_alpha_scaling\"]:.0f}')",
    "print()",
    "print(f'=== WANDERING assignment: {len(assignment)} trajectories ===')",
    "from collections import Counter",
    "repo_counts = Counter(w['repo'] for w in assignment.values())",
    "print(f'  Repos: {dict(repo_counts)}')",
))

cells.append(code(
    "# 6) Load Phase 6 instance metadata (need repo / base_commit / problem_statement to re-run agent)",
    "from datasets import load_dataset",
    "ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')",
    "by_iid = {ex['instance_id']: ex for ex in ds}",
    "print(f'SWE-bench Pro loaded: {len(by_iid)} instances')",
    "",
    "# Verify all WANDERING instance_ids resolve",
    "missing = [iid for iid in assignment if iid not in by_iid]",
    "print(f'WANDERING IIDs missing from dataset: {len(missing)}')",
    "if missing: print('  ', missing[:3])",
    "wand_iids = [iid for iid in assignment if iid in by_iid]",
    "print(f'WANDERING runnable: {len(wand_iids)}/{len(assignment)}')",
))

# ---------------------------------------------------------------------------
# LayerPatch + agent runner helper
# ---------------------------------------------------------------------------
cells.append(code(
    "# 7) LayerPatch utility (from instrumentation/layerpatch.py) + agent runner helper",
    "from instrumentation.layerpatch import LayerPatch, make_random_patch",
    "from runner import Runner",
    "from instrumentation.capture import audit_captures",
    "from config import HarnessConfig, DEFAULT",
    "from dataclasses import replace",
    "",
    "cfg = replace(",
    "    DEFAULT,",
    "    work_root=Path('/content/work_exp_bc'),",
    "    capture_root=DRIVE_ROOT / 'captures',",
    "    trace_root=DRIVE_ROOT / 'traces',",
    "    max_turns=50,",
    "    bash_max_output_bytes=32_000,",
    ")",
    "print(f'max_turns={cfg.max_turns}, capture_root={cfg.capture_root}')",
    "",
    "runner = Runner(model=model, tokenizer=tok, config=cfg)",
    "print('Runner ready.')",
))

cells.append(code(
    "# 8) Workdir prep (same as Phase 6) + run_with_hook helper",
    "import subprocess, time",
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
    "",
    "def run_with_hook(instance, *, layer, patch_tensor, mode='replace', alpha=1.0, run_tag=''):",
    "    \"\"\"Run a single agent loop with LayerPatch hook installed throughout.",
    "    Returns runner.run_one output + hook call count + tag.",
    "    \"\"\"",
    "    if mode == 'add':",
    "        eff_patch = patch_tensor * float(alpha)",
    "    else:",
    "        eff_patch = patch_tensor",
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
    "    return out",
))

# ---------------------------------------------------------------------------
# Exp B section
# ---------------------------------------------------------------------------
cells.append(md(
    "---",
    "## Exp B — Primary L55 SUCCESS-donor patching",
    "",
    "For each WANDERING trajectory: install hook at L55 with per-repo SUCCESS donor in `mode='replace'`. Hook is active from turn 0 throughout the agent's max_turns=50 fresh run.",
    "",
    "Decision rule:",
    "- ≥30% flip to finish_tool → CAUSAL — paper headline",
    "- 10-30% flip → weak causal (run controls C1-C3 next)",
    "- ≤10% flip → null (mechanism is more complex than persistent residual injection)",
))

cells.append(code(
    "# 9) Exp B Primary execution loop — sequential with per-instance Drive checkpoint",
    "import json, time",
    "",
    "EXP_B_OUT = DRIVE_ROOT / 'exp_b_primary'",
    "EXP_B_OUT.mkdir(parents=True, exist_ok=True)",
    "CKPT_B = EXP_B_OUT / 'results.json'",
    "",
    "results_b = json.loads(CKPT_B.read_text()) if CKPT_B.exists() else {}",
    "print(f'Resuming Exp B: {len(results_b)} of {len(wand_iids)} done')",
    "",
    "t_start = time.time()",
    "for i, iid in enumerate(wand_iids):",
    "    if iid in results_b and results_b[iid].get('finished') is not None:",
    "        continue",
    "    repo = assignment[iid]['repo']",
    "    donor_key = assignment[iid]['donors_available']['L55_success']",
    "    if donor_key == 'MISSING':",
    "        print(f'[{i+1:2d}/{len(wand_iids)}] SKIP {iid[:50]} — no L55 SUCCESS donor for {repo}')",
    "        continue",
    "    donor = donors[donor_key]",
    "    print(f'[{i+1:2d}/{len(wand_iids)}] RUN  {iid[:50]} | repo={repo} | donor={donor_key} (L2={donor.float().norm():.1f})')",
    "    out = run_with_hook(by_iid[iid], layer=55, patch_tensor=donor, mode='replace',",
    "                        run_tag=f'exp_b_primary_{repo}')",
    "    results_b[iid] = out",
    "    CKPT_B.write_text(json.dumps(results_b, indent=2, default=str))",
    "    elapsed = (time.time() - t_start) / 60",
    "    avg = elapsed / max(1, sum(1 for r in results_b.values() if r.get('finished') is not None))",
    "    eta = avg * (len(wand_iids) - len(results_b))",
    "    print(f'  wall={out[\"wall_seconds\"]/60:.1f}min  finish={out[\"finish_reason\"]:8s}  '",
    "          f'turns={out.get(\"n_turns\",0):2d}  vram={out.get(\"vram_peak_gb\",0):.1f}GB  '",
    "          f'hook_calls={list(out[\"hook_n_calls\"].values())[0]}  | elapsed {elapsed:.0f}m, ETA {eta:.0f}m')",
    "",
    "print(f'\\n=== Exp B Primary COMPLETE: {len(results_b)}/{len(wand_iids)} ===')",
))

cells.append(code(
    "# 10) Exp B Primary quick verdict",
    "results_b = json.loads(CKPT_B.read_text())",
    "n = len(results_b)",
    "n_finished = sum(1 for r in results_b.values() if r.get('finished'))",
    "n_finish_tool = sum(1 for r in results_b.values() if r.get('finish_reason') == 'finish_tool')",
    "n_max_turns = sum(1 for r in results_b.values() if r.get('finish_reason') == 'max_turns')",
    "n_error = sum(1 for r in results_b.values() if r.get('finish_reason') == 'error')",
    "",
    "print(f'=== Exp B Primary verdict (N={n} WANDERING) ===')",
    "print(f'  finished=True:       {n_finished}/{n} ({100*n_finished/max(1,n):.1f}%)')",
    "print(f'  finish_reason=finish_tool: {n_finish_tool}/{n} ({100*n_finish_tool/max(1,n):.1f}%)')",
    "print(f'  finish_reason=max_turns:   {n_max_turns}/{n}')",
    "print(f'  finish_reason=error:       {n_error}/{n}')",
    "print()",
    "print('Comparison to baseline:')",
    "print('  WANDERING baseline (no hook): 0/19 finish_tool by definition (that\\'s what defines WANDERING)')",
    "print()",
    "flip_rate = n_finish_tool / max(1, n - n_error)",
    "if flip_rate >= 0.30:",
    "    print(f'>>> FLIP RATE {100*flip_rate:.0f}% — CAUSAL signal. Proceed to controls C1-C3.')",
    "elif flip_rate >= 0.10:",
    "    print(f'>>> FLIP RATE {100*flip_rate:.0f}% — weak signal. Controls would clarify.')",
    "else:",
    "    print(f'>>> FLIP RATE {100*flip_rate:.0f}% — null. Persistent L55 replacement does not bias toward finish_tool.')",
    "    print('    Implication: mechanism is NOT simple residual injection. Edge-layer alignment may need turn-specific timing.')",
))

# ---------------------------------------------------------------------------
# Exp C section
# ---------------------------------------------------------------------------
cells.append(md(
    "---",
    "## Exp C — v5 collapse direction α-sweep (pilot N=5)",
    "",
    "Tests whether the residual direction that PREDICTS tool-entropy collapse is also CAUSAL for collapse. Structural-rigidity α-sweep at L43 pre_tool in `mode='add'`.",
    "",
    "- Direction = v5_collapse_diff_of_means_L43_pre_tool (L2-normalized)",
    "- α ∈ {-100, -20, 0, +20, +100} — covers structural-rigidity regime (residual L2 ≈ 58)",
    "- Random direction control (norm-matched, seed=42) — tests specificity",
    "- 5 WANDERING (subset stratified by repo) × 5 α × 2 directions = 50 runs",
    "",
    "Decision matrix (at α=+100):",
    "- ≥30% finish_tool flip on probe direction, ≤10% on random → **CAUSAL** (lever)",
    "- 10-30% probe, ≤10% random → weak causal",
    "- ≤10% probe → **epiphenomenal** (joins Two Forms of Epiphenomenal Probes pattern)",
    "- Random ≥ probe → **CONFOUND** (non-specific perturbation)",
))

cells.append(code(
    "# 11) Exp C pilot — select 5 WANDERING stratified by repo",
    "from collections import defaultdict",
    "by_repo = defaultdict(list)",
    "for iid in wand_iids:",
    "    by_repo[assignment[iid]['repo']].append(iid)",
    "",
    "# Pick up to 2 per repo until we hit 5",
    "import random",
    "rng = random.Random(42)",
    "pilot_iids = []",
    "for repo, lst in by_repo.items():",
    "    rng.shuffle(lst)",
    "    pilot_iids.extend(lst[:2])",
    "    if len(pilot_iids) >= 5: break",
    "pilot_iids = pilot_iids[:5]",
    "",
    "print(f'Exp C pilot N={len(pilot_iids)} WANDERING:')",
    "for iid in pilot_iids:",
    "    print(f'  {iid[:50]}  repo={assignment[iid][\"repo\"]}')",
))

cells.append(code(
    "# 12) Exp C — α-sweep execution",
    "import json, time",
    "",
    "EXP_C_OUT = DRIVE_ROOT / 'exp_c_pilot'",
    "EXP_C_OUT.mkdir(parents=True, exist_ok=True)",
    "CKPT_C = EXP_C_OUT / 'results.json'",
    "",
    "results_c = json.loads(CKPT_C.read_text()) if CKPT_C.exists() else {}",
    "",
    "ALPHAS = [-100.0, -20.0, 0.0, +20.0, +100.0]",
    "v5_dir = v5dir['v5_collapse_diff_of_means_L43_pre_tool']  # normalized direction",
    "ref_l2 = v5_meta['reference_residual_l2_for_alpha_scaling']  # ~58",
    "",
    "# Random matched-norm direction (seed=42)",
    "random_dir = make_random_patch(int(v5_dir.shape[0]), seed=42,",
    "                                reference_tensor=v5_dir)",
    "print(f'v5 direction L2: {v5_dir.float().norm():.3f} (normalized)')",
    "print(f'random direction L2: {random_dir.float().norm():.3f} (matched to normalized v5 = 1.0)')",
    "print(f'reference residual L2 for scaling: {ref_l2:.1f}')",
    "print(f'α ∈ {ALPHAS} (effective patch L2 = α × 1.0)')",
    "print()",
    "",
    "DIRS = {'probe': v5_dir, 'random': random_dir}",
    "TOTAL_RUNS = len(pilot_iids) * len(ALPHAS) * len(DIRS)",
    "n_done = sum(1 for k in results_c if results_c[k].get('finished') is not None)",
    "print(f'Resuming Exp C: {n_done}/{TOTAL_RUNS} runs done')",
    "",
    "t_start = time.time()",
    "for i, iid in enumerate(pilot_iids):",
    "    for alpha in ALPHAS:",
    "        for dname, dir_tensor in DIRS.items():",
    "            # baseline (α=0) only needs to run once per direction since direction × 0 = same",
    "            if alpha == 0.0 and dname == 'random': continue  # skip duplicate baseline",
    "            run_key = f'{iid}__a{alpha:+.0f}__{dname}'",
    "            if run_key in results_c and results_c[run_key].get('finished') is not None:",
    "                continue",
    "            print(f'[{len([k for k in results_c if results_c[k].get(\"finished\") is not None]):3d}/{TOTAL_RUNS}] '",
    "                  f'RUN  {iid[:35]}  α={alpha:+.0f}  dir={dname}')",
    "            out = run_with_hook(by_iid[iid], layer=43, patch_tensor=dir_tensor,",
    "                                mode='add', alpha=alpha,",
    "                                run_tag=f'exp_c_pilot_a{alpha:+.0f}_{dname}')",
    "            results_c[run_key] = out",
    "            CKPT_C.write_text(json.dumps(results_c, indent=2, default=str))",
    "            elapsed = (time.time() - t_start) / 60",
    "            n_done2 = sum(1 for k in results_c if results_c[k].get('finished') is not None)",
    "            avg = elapsed / max(1, n_done2)",
    "            eta = avg * (TOTAL_RUNS - n_done2)",
    "            print(f'  wall={out[\"wall_seconds\"]/60:.1f}min  finish={out[\"finish_reason\"]:8s}  '",
    "                  f'turns={out.get(\"n_turns\",0):2d}  | elapsed {elapsed:.0f}m, ETA {eta:.0f}m')",
    "",
    "print(f'\\n=== Exp C Pilot COMPLETE ===')",
))

cells.append(code(
    "# 13) Exp C pilot verdict — flip rate per α per direction",
    "from collections import defaultdict",
    "results_c = json.loads(CKPT_C.read_text())",
    "",
    "# Aggregate: alpha × direction -> {n_total, n_finish_tool}",
    "agg = defaultdict(lambda: {'total': 0, 'finish_tool': 0, 'max_turns': 0, 'error': 0})",
    "for k, r in results_c.items():",
    "    if not isinstance(r, dict) or r.get('finished') is None: continue",
    "    parts = k.split('__')",
    "    if len(parts) < 3: continue",
    "    alpha_str = parts[1]   # 'a+100' or 'a-20'",
    "    dname = parts[2]       # 'probe' or 'random'",
    "    agg_key = (alpha_str, dname)",
    "    agg[agg_key]['total'] += 1",
    "    fr = r.get('finish_reason', 'error')",
    "    if fr == 'finish_tool': agg[agg_key]['finish_tool'] += 1",
    "    elif fr == 'max_turns': agg[agg_key]['max_turns'] += 1",
    "    else: agg[agg_key]['error'] += 1",
    "",
    "print(f'=== Exp C verdict ===')",
    "print(f'{\"α\":>6} {\"dir\":>7} {\"finish/N\":>10} {\"max_turns\":>10} {\"error\":>6}')",
    "for (a, d) in sorted(agg.keys(), key=lambda x: (float(x[0][1:]), x[1])):",
    "    r = agg[(a, d)]",
    "    print(f'  {a:>4s}  {d:>7s} {r[\"finish_tool\"]}/{r[\"total\"]:>2}        {r[\"max_turns\"]:>3d}      {r[\"error\"]:>3d}')",
    "",
    "# Decision",
    "ft_pos = agg[('a+100', 'probe')]['finish_tool'] / max(1, agg[('a+100', 'probe')]['total'])",
    "ft_neg = agg[('a-100', 'probe')]['finish_tool'] / max(1, agg[('a-100', 'probe')]['total'])",
    "rand_pos = agg[('a+100', 'random')]['finish_tool'] / max(1, agg[('a+100', 'random')]['total'])",
    "rand_neg = agg[('a-100', 'random')]['finish_tool'] / max(1, agg[('a-100', 'random')]['total'])",
    "print()",
    "print(f'At α=+100: probe flip {100*ft_pos:.0f}%, random flip {100*rand_pos:.0f}%')",
    "print(f'At α=-100: probe flip {100*ft_neg:.0f}%, random flip {100*rand_neg:.0f}%')",
    "if ft_pos >= 0.30 and rand_pos <= 0.10:",
    "    print('>>> CAUSAL — v5 direction is a lever')",
    "elif ft_pos >= 0.10 and rand_pos <= 0.10:",
    "    print('>>> Weak causal — direction has some specificity')",
    "elif ft_pos <= 0.10 and rand_pos <= 0.10:",
    "    print('>>> EPIPHENOMENAL — joins Two Forms of Epiphenomenal Probes pattern')",
    "elif rand_pos >= ft_pos:",
    "    print('>>> CONFOUND — random matches/exceeds probe; perturbation non-specific')",
))

# ---------------------------------------------------------------------------
# Save + summary
# ---------------------------------------------------------------------------
cells.append(md(
    "## Save & Next Steps",
    "",
    "All results are checkpointed to Drive in real-time. After completion:",
    "",
    "1. `swebench_exp_b_c/exp_b_primary/results.json` — 19 WANDERING × 1 condition (Exp B primary)",
    "2. `swebench_exp_b_c/exp_c_pilot/results.json` — 5 WANDERING × 5 α × 2 dirs (Exp C pilot)",
    "",
    "Locally: `python3 scripts/analyze_exp_b_c.py` will compute final verdicts, McNemar's test (Exp B), structural-rigidity diagnostic (Exp C), and write `paper/exp_b_results.md` + `paper/exp_c_results.md`.",
    "",
    "**Triple-source convergence** (per Caio's Phase 7-10 standard) needs Exp B+C verdicts plus the existing Exp D (HONEST NULL) to converge on the mechanism interpretation. The paper #2 narrative then becomes:",
    "",
    "- Exp B: tests \"persistent SUCCESS-residual injection at L55 biases WANDERING toward finish_tool\"",
    "- Exp C: tests \"v5 collapse direction at L43 is a CAUSAL lever, not just a CORRELATIONAL signature\"",
    "- Exp D: rules out \"WANDERING is silent success\" (HONEST NULL, ratio 1.37×, p=0.71)",
    "",
    "Any outcome (positive, weak, null, structurally-locked, confound) is publishable — null joins the Two Forms of Epiphenomenal Probes pattern.",
))

# Write notebook
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "machine_shape": "hm"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f'Wrote {OUT}')
print(f'Cells: {len(cells)}')
print(f'Size: {OUT.stat().st_size:,} bytes')
