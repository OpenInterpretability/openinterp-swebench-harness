"""Build Paper #4 "Modality Matters" notebook — behavioral vs residual intervention on WANDERING.

Tests the pre-registered design (paper/paper4/PREREG_modality_matters_2026-05-30.md):
  Same 20 WANDERING, live v5 tool-entropy trigger (entropy_last10 < TAU=0.503), 4 arms:
    0  = no intervention (paired baseline)
    A  = residual L11 SUCCESS-donor injection, gated at trigger (paper #2 lever, re-run in-session)
    B0 = neutral message at trigger (controls for "a message appeared / context refresh")
    B1 = forced re-plan message at trigger (the behavioral lever)
  Primary: paired McNemar (NO definitional 0/N baseline — the run-instability lesson).
    H1 B1 vs 0 ; H2 B1 vs A [modality matters] ; H3 B1 vs B0 [content not container].

Requires the agent/loop.py `_on_turn_start` hook on origin/main (self-test cell guards this).
Compute: ~80 runs (+smoke) ≈ ~13-16h RTX 6000 Pro Blackwell (~$30). Resume-safe per-run JSON.

Output: notebooks/nb_paper4_modality_matters.ipynb
"""
import json
from pathlib import Path

OUT = Path('/Volumes/SSD Major/fish/openinterp-swebench-harness/notebooks/nb_paper4_modality_matters.ipynb')


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}

def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": "\n".join(lines)}


cells = []

cells.append(md(
    "# Paper #4 — \"Modality Matters\": behavioral vs residual rescue of WANDERING",
    "",
    "Pre-registered: `paper/paper4/PREREG_modality_matters_2026-05-30.md`. Same 20 WANDERING,",
    "live v5 tool-entropy trigger, 4 arms. Tests whether a transient **behavioral** intervention",
    "rescues WANDERING where **residual** steering (paper #2 L11, null) does not.",
    "",
    "## Arms (all paired, same session, same seeds)",
    "- **0**  no intervention (paired baseline — expect ~7/20 flip from run-instability)",
    "- **A**  residual L11 SUCCESS-donor (α=0.70), gated at the trigger turn",
    "- **B0** neutral message at trigger (controls for context-refresh / a-message-appeared)",
    "- **B1** forced re-plan message at trigger (the behavioral lever)",
    "",
    "## Trigger (live, online — mimics deployment)",
    "v5 detector: fire ONCE at the first turn (>= MIN_TURN) where Shannon entropy of the last-10",
    "tool-call names < TAU (=0.503, the paper #1 operating point). Never-triggered instances run",
    "free and are reported separately.",
    "",
    "## Pre-registered gate (PAIRED McNemar — never a definitional 0/N baseline)",
    "- **H1** B1 vs 0: p<0.05 AND B1 helps",
    "- **H2** B1 vs A: B1 > A  ← MODALITY MATTERS (the headline)",
    "- **H3** B1 vs B0: B1 > B0 (content, not container)",
    "",
    "Compute ~80 runs ≈ 13-16h RTX 6000 Pro Blackwell (~$30). Resume-safe.",
))

# 0) GPU preflight (verbatim from paper #2)
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

# 1) install (verbatim)
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
    "# 1b) Log transformers commit (pin per run for behavioral A/B — feedback_transformers_main_version_pin)",
    "import transformers, subprocess, os",
    "site = os.path.dirname(transformers.__file__)",
    "h = subprocess.run(['git','-C',site,'rev-parse','HEAD'], capture_output=True, text=True).stdout.strip()",
    "print(f'transformers {transformers.__version__}, commit {h[:12] if h else \"(not git)\"}')",
))

# 2) Drive + paths (NEW dir for paper #4)
cells.append(code(
    "# 2) Drive + paths",
    "from google.colab import drive",
    "drive.mount('/content/drive')",
    "from pathlib import Path",
    "DRIVE_ROOT = Path('/content/drive/MyDrive/openinterp_runs/swebench_paper4_modality_matters')",
    "RUNS_DIR = DRIVE_ROOT / 'runs'",
    "RUNS_DIR.mkdir(parents=True, exist_ok=True)",
    "# the 20 WANDERING + their original no-hook traces (for the paired baseline + calibration)",
    "BASELINE_DIR = Path('/content/drive/MyDrive/openinterp_runs/swebench_exp_b_c/exp_b_baseline_nohook')",
    "print(f'Runs dir (resume-aware) -> {RUNS_DIR}')",
    "print(f'Existing run files: {len(list(RUNS_DIR.glob(\"*.json\")))}')",
))

# 3) pull harness (verbatim — MUST include the _on_turn_start hook on origin/main)
cells.append(code(
    "# 3) Pull harness (MUST have the agent/loop.py _on_turn_start hook — self-test in cell 8 verifies)",
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

# 3b) self-heal: ensure the _on_turn_start hook exists (no push dependency)
cells.append(code(
    "# 3b) Ensure agent/loop.py has the _on_turn_start hook (self-heal if not yet on origin/main)",
    "# Runs BEFORE any import of agent.loop, so a runtime patch is picked up cleanly.",
    "loop_path = Path(REPO_DIR) / 'agent' / 'loop.py'",
    "src = loop_path.read_text()",
    "if '_on_turn_start' not in src:",
    "    method = ('    def _on_turn_start(self, turn_idx, messages, result):\\n'",
    "              '        \\\"\\\"\\\"Hook at the start of each turn (no-op base); subclasses inject here.\\\"\\\"\\\"\\n'",
    "              '        pass\\n\\n')",
    "    src = src.replace('    def run(self, problem_message: str) -> AgentResult:',",
    "                      method + '    def run(self, problem_message: str) -> AgentResult:', 1)",
    "    src = src.replace('        for turn_idx in range(self.cfg.max_turns):\\n',",
    "                      '        for turn_idx in range(self.cfg.max_turns):\\n            self._on_turn_start(turn_idx, messages, result)\\n', 1)",
    "    loop_path.write_text(src)",
    "    print('Patched agent/loop.py with _on_turn_start hook (was missing on origin/main).')",
    "else:",
    "    print('agent/loop.py already has _on_turn_start hook.')",
))

# 4) load model (verbatim)
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

# 5) load L11 donors (for Arm A)
cells.append(code(
    "# 5) Load L11 donors (Arm A residual lever)",
    "import json, safetensors.torch as st",
    "ASSETS = Path(REPO_DIR) / 'paper_mega_v4_pairs'",
    "l11_donors = st.load_file(str(ASSETS / 'exp_l11_donors.safetensors'))",
    "print(f'Loaded {len(l11_donors)} donors: {list(l11_donors.keys())[:4]}...')",
))

# 6) the 20 WANDERING iids
cells.append(code(
    "# 6) The 20 WANDERING iids (same set as paper #2)",
    "baseline = json.load(open(BASELINE_DIR / 'results.json'))",
    "ALL_IIDS = list(baseline.keys()) if isinstance(baseline, dict) else [r['iid'] for r in baseline]",
    "print(f'WANDERING iids: {len(ALL_IIDS)}')",
    "def repo_of(iid):",
    "    if 'qutebrowser' in iid: return 'qutebrowser'",
    "    if 'openlibrary' in iid: return 'openlibrary'",
    "    if 'ansible' in iid: return 'ansible'",
    "    raise ValueError(iid)",
    "from collections import Counter",
    "print(f'Repo dist: {Counter(repo_of(i) for i in ALL_IIDS)}')",
))

# 7) Runner + dataset
cells.append(code(
    "# 7) Runner + dataset",
    "from runner import Runner",
    "from config import DEFAULT",
    "from dataclasses import replace",
    "from datasets import load_dataset",
    "ds = load_dataset('ScaleAI/SWE-bench_Pro', split='test')",
    "by_iid = {ex['instance_id']: ex for ex in ds}",
    "missing = [i for i in ALL_IIDS if i not in by_iid]",
    "assert not missing, f'Missing: {missing}'",
    "cfg = replace(DEFAULT, work_root=Path('/content/work_p4'),",
    "    capture_root=DRIVE_ROOT/'captures', trace_root=DRIVE_ROOT/'traces',",
    "    capture_layers=[], max_turns=50, bash_max_output_bytes=32_000)",
    "runner = Runner(model=model, tokenizer=tok, config=cfg)",
    "print('Runner ready (max_turns=50).')",
))

# 8) Behavioral loop + live v5 trigger + monkeypatch + self-test (THE CORE NEW CODE)
cells.append(code(
    "# 8) BehavioralAgentLoop — live v5 trigger + arm-dependent intervention (CORE)",
    "import numpy as np",
    "from collections import Counter",
    "from agent.loop import AgentLoop",
    "import runner as runner_mod",
    "from instrumentation.layerpatch import LayerPatch",
    "",
    "# --- live v5 tool-entropy (matches paper #1: WANDERING = LOW entropy collapsed loop) ---",
    "def _tool_names_per_turn(turns):",
    "    out = []",
    "    for t in turns:",
    "        tcs = getattr(t, 'tool_calls', None)",
    "        if tcs is None and isinstance(t, dict): tcs = t.get('tool_calls', [])",
    "        out.append([c.get('name','unknown') for c in (tcs or []) if isinstance(c, dict)])",
    "    return out",
    "",
    "def _shannon(items):",
    "    if not items: return 0.0",
    "    cnt = Counter(items); tot = sum(cnt.values())",
    "    return float(-sum((c/tot)*np.log2(c/tot) for c in cnt.values() if c))",
    "",
    "def live_v5_entropy(turns, last_n=10):",
    "    tc = _tool_names_per_turn(turns)",
    "    last = tc[-last_n:] if len(tc) >= last_n else tc",
    "    flat = [n for sub in last for n in sub]",
    "    return _shannon(flat), len(flat)",
    "",
    "# --- intervention config (module global; set by run_arm before each runner.run_one) ---",
    "INTERVENTION = {'arm':'0','tau':0.503,'min_turn':10,'last_n':10,'prompt':None,",
    "                'donor':None,'alpha':0.70,'layer':11}",
    "LAST_TRIGGER = {'fired':False,'turn':None,'entropy':None}",
    "ACTIVE_PATCH = {'patch':None}  # so run_arm can detach Arm-A patch after run_one returns",
    "",
    "REPLAN_PROMPT = ('You appear to be repeating similar tool calls without converging. '",
    "    'Stop and re-plan: in 2-3 sentences state what you have already established, then '",
    "    'either (a) call finish_tool with your best patch now, or (b) name the single concrete '",
    "    'blocker and the one next action to resolve it.')",
    "NEUTRAL_PROMPT = ('Here is a brief checkpoint note for your records. You may continue working '",
    "    'on the task as you see fit; no change of approach is required. Proceed with whatever next '",
    "    'step you judge appropriate.')",
    "",
    "class BehavioralAgentLoop(AgentLoop):",
    "    def __init__(self, *a, **k):",
    "        super().__init__(*a, **k)",
    "        self._fired = False",
    "        self._arm = INTERVENTION['arm']; self._tau = INTERVENTION['tau']",
    "        self._min_turn = INTERVENTION['min_turn']; self._last_n = INTERVENTION['last_n']",
    "        self._prompt = INTERVENTION['prompt']; self._donor = INTERVENTION['donor']",
    "        self._alpha = INTERVENTION['alpha']; self._layer = INTERVENTION['layer']",
    "",
    "    def _on_turn_start(self, turn_idx, messages, result):",
    "        if self._arm == '0' or self._fired or turn_idx < self._min_turn:",
    "            return",
    "        H, n_tools = live_v5_entropy(result.turns, last_n=self._last_n)",
    "        if n_tools < self._last_n or H >= self._tau:",
    "            return",
    "        # --- detector fired: apply the arm's intervention ONCE ---",
    "        self._fired = True",
    "        LAST_TRIGGER.update(fired=True, turn=turn_idx, entropy=H)",
    "        if self._arm in ('B0','B1'):",
    "            messages.append({'role':'user','content': self._prompt})",
    "        elif self._arm == 'A':",
    "            eff = self._donor.to(self.model.device).to(self.model.dtype) * float(self._alpha)",
    "            patch = LayerPatch(self.model, {self._layer: eff}, mode='add'); patch.attach()",
    "            ACTIVE_PATCH['patch'] = patch",
    "        # B2 (tool-space restriction) = stretch, intentionally not wired in v1",
    "",
    "# route the Runner through the behavioral loop (no edit to runner.py needed)",
    "runner_mod.AgentLoop = BehavioralAgentLoop",
    "",
    "# --- SELF-TEST: the base run() MUST call _on_turn_start, else interventions silently no-op ---",
    "import inspect",
    "_src = inspect.getsource(AgentLoop.run)",
    "assert '_on_turn_start' in _src, ('FATAL: base AgentLoop.run() does not call _on_turn_start. '",
    "    'The harness hook is missing on origin/main — push the agent/loop.py change before running.')",
    "print('Self-test OK: hook present in AgentLoop.run(); BehavioralAgentLoop wired into Runner.')",
))

# 9) prep_workdir + run_arm
cells.append(code(
    "# 9) prep_workdir + run_arm",
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
    "def run_arm(instance, *, arm, run_tag=''):",
    "    repo = repo_of(instance['instance_id'])",
    "    INTERVENTION.update(arm=arm, tau=TAU, min_turn=MIN_TURN, last_n=LAST_N,",
    "        prompt=(REPLAN_PROMPT if arm=='B1' else NEUTRAL_PROMPT if arm=='B0' else None),",
    "        donor=(l11_donors[f'L11_donor_success_{repo}'] if arm=='A' else None),",
    "        alpha=0.70, layer=11)",
    "    LAST_TRIGGER.update(fired=False, turn=None, entropy=None)",
    "    ACTIVE_PATCH['patch'] = None",
    "    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()",
    "    t0 = time.time()",
    "    try:",
    "        out = runner.run_one(instance, prepare_workdir_fn=prep_workdir)",
    "    except Exception as e:",
    "        out = {'instance_id': instance['instance_id'], 'finished': False,",
    "               'finish_reason':'error','error':f'{type(e).__name__}: {e}'}",
    "    finally:",
    "        if ACTIVE_PATCH['patch'] is not None:",
    "            try: ACTIVE_PATCH['patch'].detach()",
    "            except Exception: pass",
    "            ACTIVE_PATCH['patch'] = None",
    "    out['wall_seconds'] = time.time()-t0",
    "    out['arm'] = arm",
    "    out['intervened'] = LAST_TRIGGER['fired']",
    "    out['trigger_turn'] = LAST_TRIGGER['turn']",
    "    out['trigger_entropy'] = LAST_TRIGGER['entropy']",
    "    out['run_tag'] = run_tag",
    "    gc.collect(); torch.cuda.empty_cache()",
    "    return out",
))

# 10) CALIBRATION (CPU) — where does the live trigger fire? Tune before burning GPU.
cells.append(code(
    "# 10) CALIBRATION (CPU, no GPU) — live trigger fire-turn on the ORIGINAL no-hook traces.",
    "# Defines the pre-registered TRIGGER population. Tune TAU/MIN_TURN here, then run the arms.",
    "TAU = 0.503      # v5 operating threshold (paper #1 detector)",
    "MIN_TURN = 10    # require a full last-10 window before firing",
    "LAST_N = 10",
    "BASELINE_TRACES = BASELINE_DIR / 'traces'",
    "print(f'Calibration: TAU={TAU}, MIN_TURN={MIN_TURN}, LAST_N={LAST_N}')",
    "fire = {}",
    "for iid in ALL_IIDS:",
    "    tp = BASELINE_TRACES / f'{iid}.json'",
    "    if not tp.exists(): fire[iid] = None; continue",
    "    turns = json.load(open(tp)).get('turns', [])",
    "    ft = None",
    "    for t in range(MIN_TURN, len(turns)+1):",
    "        H, n = live_v5_entropy(turns[:t], last_n=LAST_N)",
    "        if n >= LAST_N and H < TAU: ft = t; break",
    "    fire[iid] = ft",
    "TRIGGER_POP = [i for i,v in fire.items() if v is not None]",
    "fired_turns = [v for v in fire.values() if v is not None]",
    "print(f'Trigger population: {len(TRIGGER_POP)}/{len(ALL_IIDS)} WANDERING would fire on original traces')",
    "if fired_turns:",
    "    print(f'  fire-turn: min={min(fired_turns)} median={int(np.median(fired_turns))} max={max(fired_turns)}')",
    "print('  -> if coverage too low, raise TAU (e.g. 0.55/0.60) or lower MIN_TURN, then re-run this cell.')",
    "print('  -> never-triggered instances run free and are reported separately (pre-reg §3).')",
))

# 11) SMOKE TEST (GPU) — 3 iids x {0, B1}, verify pipeline + trigger end-to-end before the full run
cells.append(code(
    "# 11) SMOKE TEST — 3 iids x {arm 0, arm B1}: verify trigger fires + injection works on GPU",
    "SMOKE_IIDS = (TRIGGER_POP[:3] if len(TRIGGER_POP) >= 3 else ALL_IIDS[:3])",
    "print(f'Smoke test on {SMOKE_IIDS}')",
    "for iid in SMOKE_IIDS:",
    "    for arm in ['0','B1']:",
    "        out = run_arm(by_iid[iid], arm=arm, run_tag=f'SMOKE_{iid}_{arm}')",
    "        print(f'  {iid[40:60]} arm={arm}: {out.get(\"finish_reason\")} T{out.get(\"n_turns\")} '",
    "              f'fired={out.get(\"intervened\")}@{out.get(\"trigger_turn\")} ({out[\"wall_seconds\"]:.0f}s)')",
    "print('Smoke OK if B1 shows fired=True on >=1 instance and runs complete without error.')",
))

# 12) MAIN — 20 iids x 4 arms, resume-aware
cells.append(code(
    "# 12) MAIN — 20 WANDERING x arms {0, A, B0, B1} (resume-safe per-run JSON)",
    "ARMS = ['0','A','B0','B1']   # B2 tool-restriction = stretch (add later)",
    "results = {}",
    "for iid in ALL_IIDS:",
    "    for arm in ARMS:",
    "        tag = f'{iid}__{arm}'",
    "        p = RUNS_DIR / f'{tag}.json'",
    "        if p.exists(): results[tag] = json.load(open(p)); continue",
    "        print(f'{repo_of(iid)[:4]} {iid[40:60]} arm={arm}: running...')",
    "        out = run_arm(by_iid[iid], arm=arm, run_tag=tag)",
    "        out['iid'] = iid",
    "        json.dump(out, open(p,'w'), indent=2); results[tag] = out",
    "        print(f'    {out.get(\"finish_reason\")} T{out.get(\"n_turns\")} '",
    "              f'fired={out.get(\"intervened\")}@{out.get(\"trigger_turn\")} ({out[\"wall_seconds\"]:.0f}s)')",
    "print(f'\\nMain done. {len(results)} runs in results.')",
))

# 13) PAIRED McNemar analysis (the honest test)
cells.append(code(
    "# 13) PAIRED ANALYSIS — McNemar (NO definitional 0/N baseline; the run-instability lesson)",
    "from scipy.stats import binomtest",
    "def load_arm(arm):",
    "    d = {}",
    "    for p in RUNS_DIR.glob(f'*__{arm}.json'):",
    "        r = json.load(open(p)); d[r['iid']] = r",
    "    return d",
    "arms = {a: load_arm(a) for a in ARMS}",
    "def fr(r): return r.get('finish_reason')",
    "",
    "print('=== Per-arm outcomes (all instances) ===')",
    "for a in ARMS:",
    "    outs = [fr(arms[a][i]) for i in arms[a]]",
    "    print(f'  arm {a:3s}: finish={outs.count(\"finish_tool\"):2d}  max_turns={outs.count(\"max_turns\"):2d}  '",
    "          f'invalid={outs.count(\"invalid_tools\"):2d}  (n={len(outs)})')",
    "",
    "# Pre-registered TRIGGER population (from calibration cell 10). Primary analysis restricts to it.",
    "pop = [i for i in TRIGGER_POP if all(i in arms[a] for a in ARMS)]",
    "print(f'\\n=== PRIMARY analysis on TRIGGER population (n={len(pop)}) ===')",
    "",
    "def mcnemar(a, b, label, ids):",
    "    fa = {i: fr(arms[a][i])=='finish_tool' for i in ids}",
    "    fb = {i: fr(arms[b][i])=='finish_tool' for i in ids}",
    "    a_only = sum(1 for i in ids if fa[i] and not fb[i])",
    "    b_only = sum(1 for i in ids if fb[i] and not fa[i])",
    "    disc = a_only + b_only",
    "    p = binomtest(min(a_only,b_only), disc, 0.5).pvalue if disc else 1.0",
    "    direction = 'B>A' if b_only > a_only else ('A>B' if a_only > b_only else 'tie')",
    "    print(f'  {label}: {a}={sum(fa.values())}/{len(ids)} vs {b}={sum(fb.values())}/{len(ids)} | '",
    "          f'discordant {a_only}({a}) vs {b_only}({b}) | McNemar p={p:.3f} | {direction}')",
    "    return {'p':float(p),'a_flips':int(sum(fa.values())),'b_flips':int(sum(fb.values())),",
    "            'a_only':a_only,'b_only':b_only,'direction':direction}",
    "",
    "print('  (each test: left arm vs right arm; B-only flips > A-only = the right arm helps)')",
    "h1 = mcnemar('0','B1','H1 (B1 vs baseline 0)        ', pop)",
    "h2 = mcnemar('A','B1','H2 (B1 vs residual A)[MODALITY]', pop)",
    "h3 = mcnemar('B0','B1','H3 (B1 vs neutral B0)        ', pop)",
    "",
    "print('\\n=== GATE ===')",
    "g1 = h1['p'] < 0.05 and h1['b_only'] > h1['a_only']",
    "g2 = h2['b_flips'] > h2['a_flips']",
    "g3 = h3['b_flips'] > h3['a_flips']",
    "print(f'  H1 (B1 rescues, p<0.05): {\"PASS\" if g1 else \"FAIL\"}  (p={h1[\"p\"]:.3f})')",
    "print(f'  H2 (B1 > A, modality):   {\"PASS\" if g2 else \"FAIL\"}')",
    "print(f'  H3 (B1 > B0, content):   {\"PASS\" if g3 else \"FAIL\"}')",
    "verdict = ('MODALITY MATTERS confirmed' if (g1 and g2 and g3) else",
    "           'context-refresh not content (B1~B0)' if (g1 and not g3) else",
    "           'behavioral ALSO null (deeper negative)' if not g1 else 'mixed — read per-hypothesis')",
    "print(f'  >>> {verdict}')",
    "",
    "# crash check (does the behavioral nudge destabilize, like residual high-alpha did?)",
    "print('\\n=== Safety: invalid_tools rate per arm (on trigger pop) ===')",
    "for a in ARMS:",
    "    it = sum(1 for i in pop if fr(arms[a][i])=='invalid_tools')",
    "    print(f'  arm {a:3s}: {it}/{len(pop)} invalid_tools')",
    "",
    "import numpy as np",
    "summary = {'n_trigger_pop':len(pop),'arms':{a:{'finish':sum(1 for i in arms[a] if fr(arms[a][i])=='finish_tool'),",
    "           'n':len(arms[a])} for a in ARMS},'H1':h1,'H2':h2,'H3':h3,'verdict':verdict,",
    "           'TAU':TAU,'MIN_TURN':MIN_TURN}",
    "json.dump(summary, open(DRIVE_ROOT/'paper4_summary.json','w'), indent=2)",
    "print(f'\\nSaved {DRIVE_ROOT}/paper4_summary.json')",
    "print('NOTE: secondary = Docker patch-pass on finish_tool runs (run locally, paper #2 eval harness).')",
))

# 14) shutdown (verbatim)
cells.append(code(
    "# 14) Save + shutdown",
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
    "## Outputs (in DRIVE_ROOT)",
    "- `paper4_summary.json` — per-arm finish rates, H1/H2/H3 McNemar, verdict",
    "- `runs/*.json` — per-run details (arm, finish_reason, intervened, trigger_turn)",
    "",
    "## Next (local)",
    "1. Sync `paper4_summary.json` + `runs/` to repo",
    "2. Docker patch-pass eval on finish_tool runs (secondary quality metric — paper #2 eval harness)",
    "3. Recompute paired analysis from raw JSON with `scripts/` (mirror paper2_l11_honest_paired_analysis)",
    "4. Paper #4 writeup — title depends on the gate:",
    "   - all PASS → 'Modality Matters: the WANDERING knowledge-action gap closes behaviorally'",
    "   - B1~B0 → 'Transient context-refresh, not content, partially rescues WANDERING'",
    "   - B1 null → 'Neither modality rescues: a deeper agent knowledge-action gap'",
    "5. Cross-architecture replication (Llama-70b) = pre-registered stretch.",
))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
      "language_info": {"name":"python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=2))
print(f"Wrote {OUT} ({len(cells)} cells)")
