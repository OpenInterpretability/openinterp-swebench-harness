"""Generate notebooks/nb_paper_mega_interchange.ipynb — paper-MEGA v4
Geiger-style interchange-intervention experiment.

16 cells. Designed for Modal/Runpod/Colab H100 80GB or A100 40GB fallback.
Validates/falsifies the 5-axis operational-constraints framework against
11 probes via causal-abstraction interchange interventions.

Phase A.2 deliverable per EXPERIMENTAL_PLAN_paper_mega_v4_interchange.md §10.
Run loops are stubbed with clear TODOs for cells that need pair files updated
post Phase B0 measurement.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_paper_mega_interchange.ipynb"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n", "outputs": [], "execution_count": None}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# Paper-MEGA v4 — Interchange-Intervention Experiment

Geiger-style causal-abstraction validation of the 5-axis operational-constraints
framework on Qwen3.6-27B. Tests whether each of the 11 probes from paper-MEGA
v3.4 (R1/R2/R3 positive + R4/R5 negative) preserves causal abstraction under
interchange-intervention protocol.

**Design doc:** EXPERIMENTAL_PLAN_paper_mega_v4_interchange.md (§1-15)
**Pair files:** paper_mega_v4_pairs/ (Phase A.1, commits 7f916d5..a699c42)

**Compute:** ~$50 + 2 wall-clock days on H100. Phase B minimum-viable
(R1 C1 layer factorial only) = $15-25 + 5-10 GPU-hr.

**Decision gate (after Cell 8):** if ST_L31 PASSes at L31 while ST_L11/L55
NULL → continue with R2/R3/R4/R5 defaults. If NOT → stop and reframe as
honest-negative paper.
"""),

    # -------- Cell 1: Setup --------
    code("""
# 1) Install — transformers from main (qwen3_5 model_type), accelerate, scipy
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q accelerate scipy safetensors huggingface_hub
!pip install -q flash-linear-attention causal-conv1d --no-build-isolation 2>&1 | tail -3 || true

import importlib
for pkg in ['transformers', 'accelerate', 'huggingface_hub']:
    m = importlib.import_module(pkg)
    print(f'  {pkg}: {getattr(m, "__version__", "?")}')
"""),

    # -------- Cell 2: GPU pre-flight + Drive --------
    code("""
# 2) GPU + Drive
import subprocess, sys, os
out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print(f'GPU: {out}')
mem_gb = int(out.split(',')[1].strip().split()[0]) / 1024
print(f'VRAM: {mem_gb:.1f} GB')
assert mem_gb >= 38, f'Need >=40GB. Got {mem_gb:.1f}GB.'

# Drive mount (Colab) or set DRIVE_ROOT for Modal/Runpod
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs/paper_mega_v4'
except ImportError:
    DRIVE_ROOT = os.path.expanduser('~/paper_mega_v4_runs')
os.makedirs(DRIVE_ROOT, exist_ok=True)
print(f'DRIVE_ROOT: {DRIVE_ROOT}')
"""),

    # -------- Cell 3: Model + tokenizer --------
    code("""
# 3) Load Qwen3.6-27B (bf16, device_map auto)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'Qwen/Qwen3-6-27B-Instruct'  # confirm exact HF ID at runtime
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)
model.eval()

# LAYERS shim — transformers main exposes either model.model.layers or .language_model.layers
LAYERS = (
    model.model.layers if hasattr(model.model, 'layers')
    else model.model.language_model.layers
)
print(f'Model loaded. N layers: {len(LAYERS)}, d_model: {LAYERS[0].self_attn.q_proj.in_features}')
"""),

    # -------- Cell 4: Pair files + probe weights --------
    code("""
# 4) Pull pair files + probe weights from HF dataset
from huggingface_hub import snapshot_download, hf_hub_download
import json

# 4a) Pair files (Phase A.1 deliverable in openinterp-swebench-harness GH)
PAIRS_DIR = '/content/openinterp-swebench-harness/paper_mega_v4_pairs'
if not os.path.exists(PAIRS_DIR):
    subprocess.run(['git', 'clone', '--depth=1',
                    'https://github.com/OpenInterpretability/openinterp-swebench-harness',
                    '/content/openinterp-swebench-harness'], check=True)

pair_files = {
    'st':  json.load(open(f'{PAIRS_DIR}/st_pairs.json')),
    'rg':  json.load(open(f'{PAIRS_DIR}/rg_l55_pairs.json')),
    'cap': json.load(open(f'{PAIRS_DIR}/cap_pairs.json')),
    'cot': json.load(open(f'{PAIRS_DIR}/cot_l55_pairs.json')),
}
for k, v in pair_files.items():
    print(f'  {k}: {v.get("n_pairs", "?")} pairs')

# 4b) Probe weights from HF paper-mega dataset
DATASET_PATH = snapshot_download(
    repo_id='caiovicentino1/openinterp-paper-mega-conditionally-causal',
    repo_type='dataset',
)
print(f'  Dataset cached at: {DATASET_PATH}')
"""),

    # -------- Cell 5: Build probe direction dict --------
    code("""
# 5) Construct probe_directions dict: probe_name -> (layer, position_role, v: tensor[d_model])
# Each probe direction sourced from Phase verdict JSONs in the HF dataset.
import torch
import numpy as np

probe_directions = {}

# ST_L31_gen, L11_gen, L55_gen — Ridge probe weights from subjective_time_probe_v1
# TODO: subjective_time_probe_v1_results.json has R²/ρ but not weights directly.
#       Real probe weights live in subjective_time_phase2a residuals_to_probe matrix.
#       Phase A.2 placeholder — load actual weight tensors here.
for L in [11, 31, 55]:
    probe_directions[f'ST_L{L}_gen'] = {
        'layer': L,
        'position_role': 'last_thinking_token',
        'v': torch.randn(LAYERS[0].self_attn.q_proj.in_features, dtype=torch.bfloat16, device='cuda'),  # TODO: load real
    }

# Capability probes — diff-means K=10 from phase11_verdict.json
# TODO: extract top-10 feature directions from phase11_verdict and rebuild as direction tensor
for L, pos in [(11, 'think_start'), (23, 'pre_tool'), (31, 'pre_tool'),
               (43, 'turn_end'), (43, 'think_start'), (55, 'pre_tool')]:
    name = f'Cap_L{L}_{pos}'
    probe_directions[name] = {
        'layer': L,
        'position_role': pos,
        'v': torch.randn(LAYERS[0].self_attn.q_proj.in_features, dtype=torch.bfloat16, device='cuda'),  # TODO
    }

# RG_L55_mid_think + FG_L31_pre_tool — from phase10_fg_rg_causality
# CoT_L55_mid_think — from phase8_causal_cot
# SWE_L43_pre_tool — from swebench_v6_phase6/phase7
# TODO: load real direction tensors

# Normalize all to unit norm
for name, p in probe_directions.items():
    p['v'] = p['v'] / p['v'].norm()

print(f'Loaded {len(probe_directions)} probe directions (PLACEHOLDERS — replace with real weights)')
"""),

    # -------- Cell 6: Residual capture utility --------
    code("""
# 6) Capture residual at (layer, position) during forward pass
# Position can be: integer (absolute token index), 'last' (last token), or callable

def capture_residual(model, tokenizer, prompt_text, layer_idx, position_resolver):
    \"\"\"Run forward on prompt_text, capture residual at (layer, position).

    Returns: (residual_tensor[d_model], full_tokens, position_idx_used)
    \"\"\"
    capture = {}

    def hook(module, inp, out):
        # out is typically (hidden, ...) — Qwen wraps as tuple
        h = out[0] if isinstance(out, tuple) else out
        capture['h'] = h.detach().clone()
        return out

    h_hook = LAYERS[layer_idx].register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt_text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            model(**inputs)
        full_residual = capture['h']  # shape (1, seq_len, d_model)
        seq_len = full_residual.shape[1]
        pos_idx = position_resolver(full_residual, inputs.input_ids[0], tokenizer)
        if pos_idx < 0:
            pos_idx = seq_len + pos_idx
        h_at_pos = full_residual[0, pos_idx, :]  # (d_model,)
        return h_at_pos, inputs.input_ids[0], pos_idx
    finally:
        h_hook.remove()


# Position resolvers
def last_token(h, ids, tok):
    return -1

def last_thinking_token(h, ids, tok):
    # Find last token before </think> close tag
    close_id = tok.convert_tokens_to_ids('</think>')
    arr = ids.tolist()
    if close_id in arr:
        return arr.index(close_id) - 1
    return -1

def pre_tool_position(h, ids, tok):
    # Last token before <tool_call> or similar tool-emit marker
    # Implementation-specific to Qwen3.6 chat template
    return -1  # TODO: refine for actual chat template

POSITION_RESOLVERS = {
    'last': last_token,
    'last_thinking_token': last_thinking_token,
    'pre_tool': pre_tool_position,
    'mid_think': lambda h, ids, tok: h.shape[1] // 2,  # rough midpoint
    'think_start': lambda h, ids, tok: 0,  # TODO: find first <think> token
    'turn_end': last_token,
}
"""),

    # -------- Cell 7: Interchange-intervention hook + generation --------
    code("""
# 7) Interchange-intervention: replace projection onto v at (L, pos) with α_counter

def interchange_generate(model, tokenizer, prompt_plus, layer_idx, position_resolver,
                         probe_v, h_counter, scale=1.0, max_new_tokens=512,
                         alpha_counter=None):
    \"\"\"Generate from P+ with projection onto probe_v at (L, pos) interchanged.

    h_counter: pre-captured residual from P- at the same (L, pos)
    scale: 1.0 = natural data-driven magnitude; vary for C3 magnitude factorial
    alpha_counter: optional pre-computed ⟨h_counter, v⟩; if None computed from h_counter
    \"\"\"
    v = probe_v / probe_v.norm()
    if alpha_counter is None:
        alpha_counter = (h_counter @ v).item()

    # State tracking for hook: only apply at first matching position during generation
    state = {'applied': False, 'pos_idx': None}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if state['applied']:
            return out
        # Resolve position lazily — depends on growing sequence
        try:
            input_ids = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else None
            if input_ids is None:
                return out
            pos = position_resolver(h, input_ids[0], tokenizer)
            if pos < 0:
                pos = h.shape[1] + pos
            h_orig = h[0, pos, :].clone()
            alpha_orig = (h_orig @ v).item()
            delta = scale * (alpha_counter - alpha_orig) * v
            new_h = h.clone()
            new_h[0, pos, :] = h_orig + delta
            state['applied'] = True
            state['pos_idx'] = pos
            return (new_h,) + (out[1:] if isinstance(out, tuple) else ())
        except Exception:
            return out

    handle = LAYERS[layer_idx].register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt_plus, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic for fair interchange
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)
        return {
            'text': gen_text,
            'applied': state['applied'],
            'pos_idx': state['pos_idx'],
            'alpha_counter': alpha_counter,
        }
    finally:
        handle.remove()
"""),

    # -------- Cell 8: Behavioral measurement utilities --------
    code("""
# 8) Per-probe behavioral measurement functions

import re

def measure_thinking_length(text):
    \"\"\"Count tokens (words proxy) inside <think>...</think> block.\"\"\"
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if not m:
        return 0
    return len(m.group(1).split())

def measure_thinking_emission(text):
    \"\"\"Binary: did <think> block appear at all?\"\"\"
    return '<think>' in text

def extract_final_answer(text):
    \"\"\"Heuristic: last number in generated text after </think> close.\"\"\"
    post_think = text.split('</think>')[-1] if '</think>' in text else text
    nums = re.findall(r'-?\\d+(?:,\\d{3})*(?:\\.\\d+)?', post_think)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(',', '').replace('$', ''))
    except ValueError:
        return None

def measure_answer_correctness(text, gold_answer):
    \"\"\"Compare extracted final answer to gold; tolerance 0.01.\"\"\"
    a = extract_final_answer(text)
    try:
        g = float(str(gold_answer).replace(',', '').replace('$', ''))
    except (ValueError, TypeError):
        return False
    return a is not None and abs(a - g) < 0.01

def measure_patch_emit(text):
    \"\"\"Binary: did generation contain a code patch (heuristic).\"\"\"
    return '```' in text or 'diff --git' in text or '<<<<<<< SEARCH' in text

def measure_tool_finish(text):
    \"\"\"Was 'finish' tool selected (vs other tool).\"\"\"
    # Match against Qwen3.6 tool call pattern
    return '<tool_call>' in text and '"name": "finish"' in text

PROBE_MEASUREMENT = {
    'ST_L31_gen':        ('thinking_length', measure_thinking_length),
    'ST_L11_gen':        ('thinking_length', measure_thinking_length),
    'ST_L55_gen':        ('thinking_length', measure_thinking_length),
    'RG_L55_mid_think':  ('answer_correctness', measure_answer_correctness),
    'CoT_L55_mid_think': ('thinking_emission', measure_thinking_emission),
    'Cap_L23_pre_tool':  ('patch_emit', measure_patch_emit),
    'Cap_L31_pre_tool':  ('patch_emit', measure_patch_emit),
    'Cap_L43_turn_end':  ('patch_emit', measure_patch_emit),
    'Cap_L55_pre_tool':  ('patch_emit', measure_patch_emit),
    'SWE_L43_pre_tool':  ('tool_finish', measure_tool_finish),
    'FG_L31_pre_tool':   ('answer_correctness', measure_answer_correctness),
}
"""),

    # -------- Cell 9: R1 ST_L31 — C1 layer factorial --------
    code("""
# 9) R1 (ST_L31_gen) — C1 layer factorial
# Tests: same probe direction, same prompts, apply interchange at 5 layers.
# Predicted: only L31 produces behavioral shift toward P-.

results_R1_C1 = {}
LAYER_GRID = [11, 23, 31, 43, 55]
N_PAIRS = 15  # use all 15 ST P+ candidates

for layer in LAYER_GRID:
    print(f'\\n=== R1 C1 layer={layer} ===')
    results_R1_C1[layer] = []
    for pair_idx in range(N_PAIRS):
        pair = pair_files['st']['pairs_P_plus_only'][pair_idx]
        # TODO Phase B0: pair P_minus_TODO needs actual easy GSM8K prompt
        p_plus_text = pair['P_plus']['question']
        # TODO: load matched P- from phase_b0_easy_gsm8k.json once available
        p_minus_text = 'TODO_PHASE_B0'  # placeholder

        if p_minus_text == 'TODO_PHASE_B0':
            print(f'  pair {pair_idx}: SKIP (P- pending Phase B0)')
            continue

        # Capture h_counter from P- at layer
        h_counter, _, _ = capture_residual(
            model, tok, p_minus_text, layer,
            POSITION_RESOLVERS['last_thinking_token']
        )

        # Run interchange on P+ at same layer
        v = probe_directions['ST_L31_gen']['v']  # always L31-trained direction
        interchange_out = interchange_generate(
            model, tok, p_plus_text, layer,
            POSITION_RESOLVERS['last_thinking_token'],
            probe_v=v, h_counter=h_counter, scale=1.0
        )

        # Random-direction matched control
        v_random = torch.randn_like(v)
        v_random = v_random / v_random.norm()
        random_out = interchange_generate(
            model, tok, p_plus_text, layer,
            POSITION_RESOLVERS['last_thinking_token'],
            probe_v=v_random, h_counter=h_counter, scale=1.0
        )

        # Measure
        _, fn = PROBE_MEASUREMENT['ST_L31_gen']
        baseline_len = pair['P_plus']['baseline_thinking_len']
        probe_len = fn(interchange_out['text'])
        random_len = fn(random_out['text'])

        results_R1_C1[layer].append({
            'pair_idx': pair_idx,
            'baseline_length': baseline_len,
            'probe_length': probe_len,
            'random_length': random_len,
            'shift_toward_P_minus_probe': baseline_len - probe_len,
            'shift_toward_P_minus_random': baseline_len - random_len,
        })
        print(f'  pair {pair_idx}: probe shift={baseline_len - probe_len:+.0f}, random shift={baseline_len - random_len:+.0f}')

    # Save intermediate per layer
    with open(f'{DRIVE_ROOT}/R1_C1_layer{layer}.json', 'w') as f:
        json.dump(results_R1_C1[layer], f, indent=2)
"""),

    # -------- Cell 10: R1 C2 trajectory factorial --------
    code("""
# 10) R1 (ST_L31_gen) — C2 trajectory factorial (onset timing)
# Tests: same probe direction at L31, but apply interchange at decode-step 1 vs 50 vs 200
# Predicted: only token-1 onset works; effect decays to 0 by step 200

results_R1_C2 = {}
ONSET_GRID = [1, 50, 200]  # decode step where interchange is first applied

# TODO Phase A.2 stub — implementation requires modifying interchange_generate
# to apply hook only after N generated tokens. Add `onset_step` parameter:
#   - At onset_step=1, hook fires immediately
#   - At onset_step=50, hook waits until 50 tokens have been generated then fires once
#
# Decision: defer implementation until Phase B for compute efficiency. Phase A.2
# Notebook ships with this cell as stub; Phase B fills implementation.

print('R1 C2 (trajectory) — stub for Phase B implementation')
print('  Decision: onset_step parameter requires generation-step-aware hook')
print('  Path forward: implement at Phase B by tracking generated_count in hook state')
"""),

    # -------- Cell 11: R1 C3 magnitude + C4 direction + C5 negative control --------
    code("""
# 11) R1 — C3 magnitude, C4 pair-direction asymmetry, C5 negative control

# C3: vary scale ∈ {0.5, 1.0, 2.0} — implementation = re-run Cell 9 with scale parameter
# C4: vary pair direction (P+→P- vs P-→P+) — swap which is captured vs steered
# C5: substitute CoT_L55 probe direction for ST_L31; expect NULL across all conditions

# Phase A.2 stub — implementations are minor variations on Cell 9.
# Phase B compute: run only at promising L31 + token-1 + scale=1.0 from C1+C2
# results, vary one constraint at a time.

print('R1 C3/C4/C5 — stubs for Phase B (minor variations on Cell 9)')
"""),

    # -------- Cell 12: R2 RG_L55_mid_think + R3 capability defaults --------
    code("""
# 12) R2 + R3 default conditions
# R2 RG_L55_mid_think — 25 length-proxy pairs (Phase A.1 weak proxy; v4.1 upgrade)
# R3 Cap × 4 sites — 10 emit-proxy pairs

results_R2 = []
print('\\n=== R2 RG_L55_mid_think default ===')
for pair_idx, pair in enumerate(pair_files['rg']['pairs']):
    p_plus_text = pair['P_plus']['question']
    p_minus_text = pair['P_minus']['question']
    v = probe_directions.get('RG_L55_mid_think')
    if v is None:
        print('  RG_L55 probe direction not loaded (Cell 5 TODO)')
        break
    h_counter, _, _ = capture_residual(model, tok, p_minus_text, 55, POSITION_RESOLVERS['mid_think'])
    out = interchange_generate(model, tok, p_plus_text, 55, POSITION_RESOLVERS['mid_think'],
                                probe_v=v['v'], h_counter=h_counter, scale=1.0)
    fn = PROBE_MEASUREMENT['RG_L55_mid_think'][1]
    correct = fn(out['text'], pair['P_plus'].get('gold_answer', ''))
    results_R2.append({'pair_idx': pair_idx, 'correct_under_interchange': correct})

with open(f'{DRIVE_ROOT}/R2_RG_L55.json', 'w') as f:
    json.dump(results_R2, f, indent=2)


results_R3 = {}
print('\\n=== R3 Capability × 4 sites ===')
for site in ['Cap_L23_pre_tool', 'Cap_L31_pre_tool', 'Cap_L43_turn_end', 'Cap_L55_pre_tool']:
    results_R3[site] = []
    p = probe_directions.get(site)
    if p is None:
        print(f'  {site}: probe direction not loaded')
        continue
    for pair_idx, pair in enumerate(pair_files['cap']['pairs']):
        # TODO: each Cap pair is an instance_id; need to load actual SWE prompt
        # for now stub with placeholder
        print(f'  {site} pair {pair_idx}: TODO load SWE prompt from instance_id')
        break  # stub
"""),

    # -------- Cell 13: R4 + R5 negative controls --------
    code("""
# 13) R4 + R5 negative controls

# CoT_L55 (R4 template-locked) — should NULL across all conditions
print('\\n=== R4 CoT_L55_mid_think (template control) ===')
results_R4_CoT = []
for pair_idx, pair in enumerate(pair_files['cot']['pairs'][:10]):  # subset for speed
    # P+ = enable_thinking=True, P- = enable_thinking=False
    # Apply chat template per pair config
    # TODO: implement Qwen3.6 chat template application with enable_thinking flag
    print(f'  pair {pair_idx}: TODO chat template apply')

# ST_L11 + ST_L55 (R4 wrong-layer controls) — already covered in Cell 9 C1 factorial
# SWE_L43 + FG_L31 (R5 epi controls) — TODO at Phase B

print('R4/R5 — partial coverage in Cell 9 (ST wrong-layer); rest at Phase B')
"""),

    # -------- Cell 14: Aggregation + per-probe verdicts --------
    code("""
# 14) Aggregate per-probe verdicts
# For each probe: compute mean behavioral shift under probe interchange vs random control
# Statistical test: paired Wilcoxon (probe shifts vs random shifts)
# Classify: PASS / NULL / FAIL based on signed shift and significance

import scipy.stats as stats
from collections import defaultdict

verdicts = {}

# R1 C1 verdict aggregation
print('=== R1 C1 layer factorial verdict ===')
for layer, runs in results_R1_C1.items():
    if not runs:
        continue
    probe_shifts = [r['shift_toward_P_minus_probe'] for r in runs]
    random_shifts = [r['shift_toward_P_minus_random'] for r in runs]
    if len(probe_shifts) >= 5:
        stat, pval = stats.wilcoxon(probe_shifts, random_shifts, alternative='greater')
        mean_gap = np.mean(probe_shifts) - np.mean(random_shifts)
        verdict = 'PASS' if (pval < 0.05 and mean_gap > 0) else ('NULL' if mean_gap < 10 else 'WEAK')
        verdicts[f'ST_L31_gen_at_L{layer}'] = {
            'n_pairs': len(probe_shifts),
            'mean_probe_shift': float(np.mean(probe_shifts)),
            'mean_random_shift': float(np.mean(random_shifts)),
            'gap': float(mean_gap),
            'wilcoxon_p': float(pval),
            'verdict': verdict,
        }
        print(f'  L{layer}: gap={mean_gap:+.1f}, p={pval:.4f} → {verdict}')

# Save aggregated verdicts
with open(f'{DRIVE_ROOT}/interchange_verdicts.json', 'w') as f:
    json.dump(verdicts, f, indent=2)

# Cross-reference with paper-MEGA Table 2 predictions
EXPECTED = {
    'ST_L31_gen_at_L31': 'PASS',
    'ST_L31_gen_at_L11': 'NULL',
    'ST_L31_gen_at_L55': 'NULL',
}
for k, expected in EXPECTED.items():
    actual = verdicts.get(k, {}).get('verdict', 'NOT_RUN')
    match = '✓' if actual == expected else '✗'
    print(f'  {match} {k}: predicted={expected}, observed={actual}')
"""),

    # -------- Cell 15: Save + HF upload --------
    code("""
# 15) Save complete results + upload to HF dataset

# Bundle all results
all_results = {
    'R1_C1_layer_factorial': results_R1_C1,
    'R2_RG': results_R2,
    'R3_Cap': results_R3,
    'verdicts': verdicts,
    'metadata': {
        'model': MODEL_ID,
        'experiment_date': '2026-05-17',  # update at run time
        'pair_files_commit': 'a699c42',  # paper_mega_v4_pairs commit
    },
}
with open(f'{DRIVE_ROOT}/paper_mega_v4_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Saved to {DRIVE_ROOT}/paper_mega_v4_results.json')

# Upload to HF dataset (append to paper-MEGA verification dataset)
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj=f'{DRIVE_ROOT}/paper_mega_v4_results.json',
    path_in_repo='paper_mega_v4/interchange_results.json',
    repo_id='caiovicentino1/openinterp-paper-mega-conditionally-causal',
    repo_type='dataset',
    commit_message='paper-MEGA v4: interchange-intervention results',
)
print('HF upload complete')
"""),

    # -------- Cell 16: Visualization --------
    code("""
# 16) Visualization: per-layer behavioral shift + verdict overlay

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

layers_run = sorted([L for L in results_R1_C1.keys() if results_R1_C1[L]])
gaps = [verdicts.get(f'ST_L31_gen_at_L{L}', {}).get('gap', 0) for L in layers_run]
colors = ['green' if verdicts.get(f'ST_L31_gen_at_L{L}', {}).get('verdict') == 'PASS' else 'gray'
          for L in layers_run]

ax.bar([str(L) for L in layers_run], gaps, color=colors)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Layer')
ax.set_ylabel('Behavioral shift (probe − random) toward P−')
ax.set_title('R1 ST_L31_gen interchange — C1 layer factorial\\n(green = PASS, gray = NULL)')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{DRIVE_ROOT}/R1_C1_layer_factorial.png', dpi=160)
plt.show()

print('Phase B minimum-viable result printed above.')
print('\\nDecision gate (per EXPERIMENTAL_PLAN §8):')
print('  - If L31 PASS and L11/L55 NULL → continue R2/R3/R4/R5 defaults')
print('  - Otherwise → stop and reframe as honest-negative paper')
"""),

    md("""
## Phase A.2 deliverable status

This notebook is the **skeleton** per EXPERIMENTAL_PLAN §10. Cells 1-8 implement
core utilities (model load, probe load, pair load, residual capture, interchange
hook, behavioral measurement). Cells 9-13 implement run loops; some are stubbed
with explicit TODOs for Phase B compute-time implementation:

**Ready to run as-is (Cells 1-8):** Setup, model load, utility functions.
**Phase B0 dependency:** Cell 9 (R1 C1 factorial) requires `phase_b0_easy_gsm8k.json`
  with verified short-think GSM8K pairs.
**TODOs in Cell 5 (probe weights):** placeholder random tensors — replace with
  real probe weights extracted from Phase 11/10/8 verdict JSONs.
**TODOs in Cell 10-13:** trajectory factorial (C2), magnitude/direction/locus
  (C3/C4/C5), R3 SWE prompt loading. All bounded scope, fillable at Phase B.

**Decision gate after Cell 14:** if ST_L31 at L31 PASSes while L11/L55 NULL,
proceed with full R2/R3/R4/R5 runs. Otherwise reframe.

## Cost estimate per cell (H100)

| Cell | Description | Compute |
|---|---|---|
| 1-3 | Setup + model load | ~3 min |
| 4-8 | Pair load + utility tests | <1 min |
| 9 | R1 C1 (5 layers × 15 pairs × 2) | ~3-5 hr |
| 10-11 | R1 C2-C5 factorial | ~3-5 hr |
| 12 | R2 + R3 defaults | ~1-2 hr |
| 13 | R4 + R5 negatives | ~1 hr |
| 14-16 | Aggregation + viz | <1 min |
| **Total minimum (1-9 + 14-16)** | **R1 C1 only** | **5-7 hr** |
| **Total full** | **All R1-R5** | **15-20 hr** |

## Next action

Run Phase B0 (`phase_b0_easy_gsm8k.json` measurement) → run minimum viable
(Cells 1-9, 14-16) → inspect decision gate → either continue full or reframe.
"""),
]


def main():
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Wrote {NB_PATH}")


if __name__ == "__main__":
    main()
