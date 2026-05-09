"""Build Track A Phase 16 V3 — Gemma-3-27B-L41 within-family scaling.

Produces: notebooks/nb_track_a_phase16_gemma27b_v3.py.ipynb

Replicates Phase 16 N=150 on `kitft/nla-gemma3-27b-L41` to test the
*decoupling magnification* sub-thesis from V2: as NLA training quality
improves, fve_nrm approaches its 1.0 ceiling while recall spread grows.

Predictions for V3 (Gemma-3-27B vs V2 Gemma-3-12B vs V1 Qwen-2.5-7B):
- fve_nrm overall: 0.880 → 0.992 → ?
- fve_nrm spread:  0.017 → 0.005 → ?
- recall spread:   0.490 → 0.649 → ?
- If recall spread continues to grow → magnification holds at 27B
- If recall spread saturates at ~0.65 → magnification has ceiling at moderate sizes

Constants change vs V2 (auto-loaded from sidecar):
- AV/AR: kitft/nla-gemma3-27b-L41-{av,ar}
- Target: google/gemma-3-27b-it (gated, ~54 GB bf16)
- Layer 41 (vs 32)
- d_model: probably 5376 (Gemma-3-27B standard)
- injection_scale: auto from sidecar
- injection_char: probably ㈜ U+321C (same as 12B)

VRAM budget on RTX 6000 96GB (sequential load/free):
- Target Gemma-3-27B-IT bf16: ~54 GB
- AV bf16: ~54 GB
- AR truncated 42/62 layers bf16: ~36 GB
- Peak: ~54 GB (target or AV resident, never both)

Final cell loads V1 (Qwen) + V2 (Gemma-12B) results from Drive and produces
three-way scaling comparison.

Run: python3 scripts/build_nb_track_a_phase16_gemma27b_v3.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "nb_track_a_phase16_gemma27b_v3.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True) or [text],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True) or [text],
    }


cells: list[dict] = []


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""# Track A Phase 16 V3 — Gemma-3-27B-L41 within-family scaling

**Goal**: Test the *decoupling magnification* sub-thesis from paper-7 V2 at
27B scale. V1 Qwen-7B and V2 Gemma-12B both show two-tier decoupling, and
spread *grew* from 0.490 → 0.649 from 7B to 12B. V3 tests whether this
magnification continues to 27B (same Gemma family, ~2× larger) or saturates.

**Predictions (decoupling magnification thesis)**:
- fve_nrm closer to 1.0 ceiling: 0.880 → 0.992 → ≥0.99
- fve_nrm spread shrinks: 0.017 → 0.005 → ≤0.005
- recall spread continues to grow: 0.490 → 0.649 → ≥0.65 (or saturates)
- Random Gaussian collapse sharper
- Permutation gap above floor grows

**Falsifications**:
- Recall spread saturates or shrinks → magnification has ceiling at moderate model sizes; refine thesis
- fve_nrm doesn't improve → Gemma-12B already saturates Tier 1; refine thesis
- Direction-injection drops below 3/4 → category-template separation degrades with scale; refine thesis

**Adaptations vs V2** (mostly identical, all read from `nla_meta.yaml` sidecar):
- AV/AR: `kitft/nla-gemma3-27b-L41-{av,ar}`
- Target: `google/gemma-3-27b-it` (gated, ~54 GB bf16)
- Extraction layer 41 (vs 32)
- d_model 5376 expected (vs 3840) — auto-detected
- injection_scale ~80000+ expected (Gemma √d scaling)

**Final cell**: three-way scaling comparison (V1 Qwen-7B vs V2 Gemma-12B vs V3 Gemma-27B).

**Estimated runtime**: ~50-65 min on RTX 6000 96GB (3 model loads, 200 generations,
larger model = slower per-token).

**License**: Apache-2.0 (notebook); kitft NLA pairs Apache-2.0;
Gemma-3-27B-IT under Google's Gemma Terms of Use.
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup"))

cells.append(code("""# Cell 1: install
!pip install -q transformers safetensors pyyaml huggingface_hub
"""))

cells.append(code("""# Cell 2: Drive mount + paths
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
OUT = Path('/content/drive/MyDrive/openinterp_runs/track_a_phase16_gemma27b')
OUT.mkdir(parents=True, exist_ok=True)
print(f'OUT = {OUT}')

QWEN_OUT = Path('/content/drive/MyDrive/openinterp_runs/track_a_phase16')
QWEN_OUT_LEGACY = Path('/content/drive/MyDrive/openinterp_runs/track_a_nla_qwen25_poc')
GEMMA12B_OUT = Path('/content/drive/MyDrive/openinterp_runs/track_a_phase16_gemma')
print(f'Reference V1 (Qwen): {QWEN_OUT} or legacy {QWEN_OUT_LEGACY}')
print(f'Reference V2 (Gemma-12B): {GEMMA12B_OUT}')
"""))

cells.append(code("""# Cell 3: HF login (Gemma-3 is gated)
import os
from huggingface_hub import login, whoami

if not os.environ.get('HF_TOKEN'):
    print('HF_TOKEN not set. Running interactive login...')
    from huggingface_hub import notebook_login
    notebook_login()
else:
    login(token=os.environ['HF_TOKEN'])

try:
    user = whoami()
    print(f'✓ Authenticated as {user[\"name\"]}')
except Exception as e:
    raise SystemExit(f'HF auth failed: {e}')
"""))

cells.append(code("""# Cell 4: imports + constants
import torch, torch.nn as nn
import json, re, time, gc, random, statistics
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import yaml
import numpy as np

random.seed(42)
torch.manual_seed(42)

# kitft NLA Gemma-3-27B L41 (Apache-2.0)
AV_REPO = 'kitft/nla-gemma3-27b-L41-av'
AR_REPO = 'kitft/nla-gemma3-27b-L41-ar'
TARGET_REPO = 'google/gemma-3-27b-it'
EXTRACTION_LAYER = 41
device = 'cuda'

assert torch.cuda.is_available(), 'GPU required'
gpu_gb = torch.cuda.get_device_properties(0).total_memory/1e9
print(f'GPU: {torch.cuda.get_device_name(0)}  ({gpu_gb:.0f} GB)')
if gpu_gb < 80:
    print('⚠ Warning: <80 GB VRAM may not fit Gemma-3-27B at bf16. Consider H100 or RTX 6000 96GB.')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 2. Download NLA pair + parse sidecars"))

cells.append(code("""# Cell 5: download AV + AR (~80 GB total — be patient)
print('Downloading AV (Gemma-3-27B-L41) — expect ~50 GB, ~5 min...')
av_path = snapshot_download(repo_id=AV_REPO, cache_dir='/content/hf_cache')
print(f'  → {av_path}')

print('Downloading AR (incl. value_head.safetensors) — expect ~35 GB...')
ar_path = snapshot_download(repo_id=AR_REPO, cache_dir='/content/hf_cache')
print(f'  → {ar_path}')
"""))

cells.append(code("""# Cell 6: parse nla_meta.yaml
av_meta = yaml.safe_load(open(f'{av_path}/nla_meta.yaml').read())
ar_meta = yaml.safe_load(open(f'{ar_path}/nla_meta.yaml').read())

INJECTION_CHAR = av_meta['tokens']['injection_char']
INJECTION_TOKEN_ID = av_meta['tokens']['injection_token_id']
INJECTION_SCALE = float(av_meta['extraction']['injection_scale'])
MSE_SCALE = float(ar_meta['extraction']['mse_scale'])
AV_TEMPLATE = av_meta['prompt_templates']['av']
AR_TEMPLATE = ar_meta['prompt_templates']['ar']
EXTRACTION_LAYER_INDEX = ar_meta.get('extraction_layer_index',
                                      ar_meta.get('critic', {}).get('extraction_layer_index', EXTRACTION_LAYER))
expected_d = av_meta['d_model']

assert EXTRACTION_LAYER_INDEX == EXTRACTION_LAYER

print(f'INJECTION_CHAR    = {INJECTION_CHAR!r}  (token id {INJECTION_TOKEN_ID})')
print(f'INJECTION_SCALE   = {INJECTION_SCALE}')
print(f'MSE_SCALE         = {MSE_SCALE:.6f}  (= √d_model = √{expected_d} = {np.sqrt(expected_d):.6f})')
print(f'EXTRACTION_LAYER  = {EXTRACTION_LAYER_INDEX}')
print(f'expected d_model  = {expected_d}')
print(f'AR_TEMPLATE       = {AR_TEMPLATE!r}')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Prompt corpus
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 3. 50-prompt corpus (identical to V1+V2)

Same 50 prompts. Apples-to-apples comparison across the three NLA pairs.
"""))

cells.append(code("""# Cell 7: define 50 prompts (verbatim from V1+V2)
PHASE_16_PROMPTS = {
    'chat': [
        'What is the capital of Brazil?',
        'Explain why the sky is blue.',
        \"What's the difference between a violin and a viola?\",
        \"Who wrote the novel '1984'?\",
        'Describe the taste of mango to someone who has never had one.',
        'What is the largest mammal on Earth?',
        'Why do leaves change color in autumn?',
        \"What does the term 'serendipity' mean?\",
        'Explain the concept of opportunity cost in one sentence.',
        'Who painted the Mona Lisa?',
        'What is the boiling point of water at sea level in Celsius?',
        'Describe the plot of Hamlet in two sentences.',
        'What is the role of mitochondria in a cell?',
    ],
    'code': [
        'Write a Python function that returns the nth Fibonacci number using memoization.',
        'Implement binary search on a sorted list in Python.',
        'Write a JavaScript function that flattens a nested array of arbitrary depth.',
        'Show how to read a JSON file in Python and parse it into a dictionary.',
        'Write a SQL query that finds the second-highest salary in an employees table.',
        'Implement quicksort in C without using recursion.',
        'Write a regex that matches valid IPv4 addresses.',
        'Show how to make a parallel HTTP request in Python using asyncio.',
        'Implement a simple LRU cache in Python with O(1) get and put.',
        'Write a Bash script that finds duplicate files by content hash.',
        'Show how to write a custom decorator in Python that logs function call times.',
        'Implement a thread-safe counter in Python.',
    ],
    'agent': [
        'Find all Python files modified in the last 24 hours and show their sizes.',
        'Locate the file containing the implementation of the user authentication flow.',
        'Run the existing test suite, identify failing tests, and propose fixes.',
        'Read README.md, then explore the src/ directory to understand the project layout.',
        'There is a memory leak in `processor.py`; instrument it with tracemalloc and report the top allocators.',
        'Implement a CLI flag `--verbose` in `main.py` that prints debug logs.',
        'Refactor the duplicated code in `utils/parser.py` and `utils/validator.py` into a shared helper.',
        'Find all calls to `requests.get` without a timeout parameter and add timeout=30.',
        'Add type hints to all functions in `models/database.py`.',
        'Investigate why `test_async_handler.py::test_concurrent_writes` is flaky and propose a fix.',
        'Bump the dependency `pydantic` to v2 and migrate all model definitions.',
        'Set up a pre-commit hook that runs ruff and mypy on staged Python files.',
        'Profile the slowest endpoint in the FastAPI app and identify the bottleneck.',
    ],
    'reasoning': [
        'If a clock loses 5 minutes every hour, how far behind will it be after 24 hours?',
        'Three friends split a $90 bill equally. The waiter gives them $5 back. They each take $1 and tip the waiter $2. Each friend paid $29, total $87 + $2 tip = $89. Where is the missing dollar?',
        'Prove that the square root of 2 is irrational.',
        'If you have 12 coins and one is heavier, how do you find it in 3 weighings on a balance scale?',
        'A train leaves NYC at 60 mph going to Chicago. Another leaves Chicago at 80 mph going to NYC. They meet 200 miles from NYC. How far apart are the cities?',
        'What is the next number in the sequence 1, 1, 2, 3, 5, 8, 13, ..., and why?',
        'If today is Wednesday, what day of the week was it 100 days ago?',
        'Prove that the sum of the first n odd integers equals n².',
        'A bat and ball cost $1.10 in total. The bat costs $1 more than the ball. How much does the ball cost?',
        'How many trailing zeros are in 100 factorial?',
        'If P(A) = 0.6 and P(B|A) = 0.4, what is P(A and B)?',
        'Explain why, in chess, a single bishop and a king cannot checkmate a lone king.',
    ],
}

total = sum(len(v) for v in PHASE_16_PROMPTS.values())
print(f'Total prompts: {total}')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Capture
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 4. Capture L41 activations from Gemma-3-27B-IT (target)"))

cells.append(code("""# Cell 8: load target Gemma-3-27B-IT (~54 GB bf16)
print(f'Loading {TARGET_REPO} (gated, ~54 GB) — this takes a few minutes...')
target_tok = AutoTokenizer.from_pretrained(TARGET_REPO)
target = AutoModelForCausalLM.from_pretrained(TARGET_REPO, dtype=torch.bfloat16, device_map='cuda')
target.eval()
print(f'Architecture: {type(target).__name__}')

# Multimodal Gemma3 — text config nested
text_cfg = target.config.text_config if hasattr(target.config, 'text_config') else target.config
print(f'num_hidden_layers: {text_cfg.num_hidden_layers}')
print(f'hidden_size: {text_cfg.hidden_size}')
print(f'VRAM after target load: {torch.cuda.memory_allocated()/1e9:.2f} GB')

assert text_cfg.hidden_size == expected_d, f'mismatch: {text_cfg.hidden_size} vs {expected_d}'
print(f'✓ d_model={text_cfg.hidden_size} matches sidecar')

# Find layers list robustly (Gemma-3-27B may have model.language_model.layers)
inner = target.model
while not hasattr(inner, 'layers'):
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    else:
        break
print(f'✓ layers found at: {type(inner).__name__}, total {len(inner.layers)} layers')
"""))

cells.append(code("""# Cell 9: capture all 50 acts at L41
captured = {}
def hook_fn(module, inputs, output):
    h = output[0] if isinstance(output, tuple) else output
    captured['h'] = h.detach()

# inner from cell 8 has the layers list
handle = inner.layers[EXTRACTION_LAYER].register_forward_hook(hook_fn)

phase16_acts = []
t0 = time.time()
with torch.no_grad():
    for cat, prompts in PHASE_16_PROMPTS.items():
        for j, prompt in enumerate(prompts):
            chat = target_tok.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            ids = target_tok(chat, return_tensors='pt').input_ids.to('cuda')
            captured.clear()
            _ = target(ids, use_cache=False)
            h = captured['h']
            last = h[0, -1].float().cpu().contiguous()
            phase16_acts.append({
                'category': cat,
                'idx_in_cat': j,
                'key': f'{cat}_{j}',
                'prompt': prompt,
                'seq_len': ids.shape[1],
                'act_l2': last.norm().item(),
                'act': last,
            })

handle.remove()
print(f'\\n✓ captured {len(phase16_acts)} acts in {time.time()-t0:.1f}s')

print('\\nL2 stats per category (V1 Qwen ~120; V2 Gemma-12B ~75k; V3 Gemma-27B ?):')
for cat in PHASE_16_PROMPTS:
    l2s = [a['act_l2'] for a in phase16_acts if a['category'] == cat]
    print(f'  {cat:10s}  L2 mean={statistics.mean(l2s):>8.1f}  std={statistics.stdev(l2s):>6.1f}  '
          f'min={min(l2s):>7.1f}  max={max(l2s):>7.1f}')

torch.save({'phase16_acts': phase16_acts, 'PHASE_16_PROMPTS': PHASE_16_PROMPTS}, OUT / 'phase16_acts.pt')
print(f'\\n✓ saved {OUT/\"phase16_acts.pt\"}')

del target, target_tok
gc.collect(); torch.cuda.empty_cache()
print(f'VRAM after free: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Verbalize
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 5. Verbalize 50 × K=3 = 150 with AV"))

cells.append(code("""# Cell 10: load AV + verbalize 150
K_SAMPLES = 3

print('Loading AV (Gemma-3-27B AV, ~54 GB)...')
av_tok = AutoTokenizer.from_pretrained(av_path)
av = AutoModelForCausalLM.from_pretrained(av_path, dtype=torch.bfloat16, device_map='cuda')
av.eval()
print(f'VRAM after AV load: {torch.cuda.memory_allocated()/1e9:.2f} GB')

@torch.no_grad()
def verbalize_one(raw_act, temperature=1.0, max_new_tokens=200):
    text = av_tok.apply_chat_template(
        [{'role': 'user', 'content': AV_TEMPLATE.format(injection_char=INJECTION_CHAR)}],
        tokenize=False, add_generation_prompt=True,
    )
    input_ids = av_tok(text, return_tensors='pt').input_ids.to('cuda')
    embeds = av.get_input_embeddings()(input_ids)
    inj_pos = (input_ids[0] == INJECTION_TOKEN_ID).nonzero(as_tuple=True)[0][0].item()
    v = raw_act.to('cuda').to(torch.bfloat16)
    v_scaled = v / v.norm().clamp_min(1e-12) * INJECTION_SCALE
    embeds[0, inj_pos] = v_scaled
    out = av.generate(
        inputs_embeds=embeds, attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens, do_sample=temperature > 0,
        temperature=temperature, pad_token_id=av_tok.eos_token_id,
    )
    text_out = av_tok.decode(out[0], skip_special_tokens=True)
    m = re.search(r'<explanation>\\s*(.*?)\\s*</explanation>', text_out, re.DOTALL)
    return m.group(1).strip() if m else text_out.strip()

phase16_explanations = []
t0 = time.time()
for i, item in enumerate(phase16_acts):
    for k in range(K_SAMPLES):
        exp = verbalize_one(item['act'], temperature=1.0)
        phase16_explanations.append({
            'key': item['key'], 'category': item['category'],
            'sample_idx': k, 'prompt': item['prompt'], 'explanation': exp,
        })
    if (i+1) % 5 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (len(phase16_acts) - i - 1)
        print(f'  [{i+1:2d}/{len(phase16_acts)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s')

print(f'\\n✓ {len(phase16_explanations)} explanations in {time.time()-t0:.0f}s')

torch.save({'phase16_explanations': phase16_explanations}, OUT / 'phase16_explanations.pt')
with open(OUT / 'phase16_explanations.json', 'w') as f:
    json.dump(phase16_explanations, f, indent=2)
print(f'✓ saved')

del av, av_tok
gc.collect(); torch.cuda.empty_cache()
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# AR canonical
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 6. AR canonical recipe"))

cells.append(code("""# Cell 11: load AR + canonical patches
print('Loading AR (Gemma-3-27B AR truncated to L41, ~37 GB)...')
ar_tok = AutoTokenizer.from_pretrained(ar_path)
ar = AutoModelForCausalLM.from_pretrained(ar_path, dtype=torch.bfloat16, device_map='cuda')
ar.eval()

# Strip lm_head — critic never emits logits
ar.lm_head = nn.Identity()

# Strip final layernorm — value_head sees raw post-block-K residual
inner_ar = ar.model if hasattr(ar, 'model') else ar
if hasattr(inner_ar, 'language_model'):
    inner_ar = inner_ar.language_model

FINAL_LN_ATTRS = ('norm', 'final_layernorm', 'ln_f', 'final_layer_norm')
norm_replaced = None
for attr in FINAL_LN_ATTRS:
    if hasattr(inner_ar, attr):
        setattr(inner_ar, attr, nn.Identity())
        norm_replaced = attr
        break
assert norm_replaced is not None, f'No final-LN found on {type(inner_ar).__name__}'
print(f'✓ replaced inner_model.{norm_replaced} with Identity')

# Load value_head
vh_state = load_file(f'{ar_path}/value_head.safetensors')
d_model = ar.config.hidden_size if hasattr(ar.config, 'hidden_size') else expected_d
value_head = nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16)
value_head.load_state_dict(vh_state)
value_head = value_head.to('cuda').eval()

print(f'✓ AR canonical: norm=Identity, lm_head=Identity, value_head Linear({d_model},{d_model})')
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"""))

cells.append(code("""# Cell 12: reconstruct() + score_canonical()
@torch.inference_mode()
def reconstruct(explanation):
    prompt = AR_TEMPLATE.format(explanation=explanation)
    ids = ar_tok(prompt, return_tensors='pt', add_special_tokens=True,
                 truncation=True, max_length=1024).input_ids.to('cuda')
    model_to_call = ar.model.language_model if hasattr(ar.model, 'language_model') else ar.model
    h = model_to_call(ids, use_cache=False).last_hidden_state[0, -1]
    return value_head(h).float().cpu()

def score_canonical(pred, gold):
    pred = pred.float().flatten()
    gold = gold.float().flatten()
    pred_n = pred / pred.norm().clamp_min(1e-12) * MSE_SCALE
    gold_n = gold / gold.norm().clamp_min(1e-12) * MSE_SCALE
    mse = ((pred_n - gold_n) ** 2).mean().item()
    var = ((gold_n - gold_n.mean()) ** 2).mean().item()
    cos = torch.nn.functional.cosine_similarity(pred.unsqueeze(0), gold.unsqueeze(0)).item()
    fve_nrm = 1.0 - mse / max(var, 1e-12)
    return mse, cos, fve_nrm

probe = ar_tok('x', add_special_tokens=True)['input_ids']
bos = ar_tok.bos_token_id
print(f'BOS check: bos_token_id={bos}, first probe id={probe[0]}, match={bos is None or probe[0] == bos}')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruct + per-category aggregate
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 7. Reconstruct all 150 + per-category aggregate"))

cells.append(code("""# Cell 13: reconstruct all 150
acts_by_key = {a['key']: a for a in phase16_acts}

phase16_results = []
t0 = time.time()
for ex in phase16_explanations:
    orig = acts_by_key[ex['key']]['act']
    rec = reconstruct(ex['explanation'])
    mse, cos, fve_nrm = score_canonical(rec, orig)
    phase16_results.append({
        'key': ex['key'], 'category': ex['category'],
        'sample_idx': ex['sample_idx'], 'prompt': ex['prompt'],
        'explanation': ex['explanation'],
        'cos': cos, 'mse': mse, 'fve_nrm': fve_nrm,
    })
print(f'✓ reconstructed {len(phase16_results)} in {time.time()-t0:.1f}s')
"""))

cells.append(code("""# Cell 14: per-category aggregate
print('=' * 75)
print(f'{\"category\":<12}{\"N\":>4}{\"cos\":>8}{\"fve_nrm\":>10}{\"fve_std\":>10}{\"within\":>10}')
print('-' * 75)
cat_summary = {}
for cat in ['chat', 'code', 'agent', 'reasoning']:
    cr = [r for r in phase16_results if r['category'] == cat]
    fves = [r['fve_nrm'] for r in cr]
    coss = [r['cos'] for r in cr]
    by_key = defaultdict(list)
    for r in cr:
        by_key[r['key']].append(r['fve_nrm'])
    within = [statistics.stdev(v) for v in by_key.values() if len(v) > 1]
    cat_summary[cat] = {
        'N': len(cr), 'cos': statistics.mean(coss),
        'fve_nrm': statistics.mean(fves), 'fve_std': statistics.stdev(fves),
        'within_prompt_std': statistics.mean(within) if within else 0.0,
    }
    s = cat_summary[cat]
    print(f'{cat:<12}{s[\"N\"]:>4}{s[\"cos\"]:>+8.3f}{s[\"fve_nrm\"]:>+10.3f}'
          f'{s[\"fve_std\"]:>+10.3f}{s[\"within_prompt_std\"]:>+10.3f}')

print('-' * 75)
all_fves = [r['fve_nrm'] for r in phase16_results]
print(f'{\"OVERALL\":<12}{len(phase16_results):>4}'
      f'{statistics.mean([r[\"cos\"] for r in phase16_results]):>+8.3f}'
      f'{statistics.mean(all_fves):>+10.3f}')

fve_spread = max(cat_summary[c]['fve_nrm'] for c in cat_summary) - min(cat_summary[c]['fve_nrm'] for c in cat_summary)
print(f'\\nfve_nrm category spread: {fve_spread:.3f}')
print(f'V1 Qwen-7B:    overall 0.880, spread 0.017')
print(f'V2 Gemma-12B:  overall 0.992, spread 0.005')
print(f'Magnification predicts: V3 Gemma-27B fve closer to 1.0, spread ≤ 0.005')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Topic-match
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 8. Topic-match analysis"))

cells.append(code("""# Cell 15: keyword recall
STOPWORDS = set('the a an and or but of to in on at for with from by about as is are was were be been being have has had do does did i you he she it we they this that these those what which who whom whose how why when where can could would should may might must shall will not no yes if then else than'.split())

def cwords(t, m=4):
    return set(x for x in re.findall(r'[a-zA-Z]+', t.lower()) if len(x) >= m and x not in STOPWORDS)

def recall_metric(prompt, explanation):
    pw = cwords(prompt); ew = cwords(explanation)
    return len(pw & ew) / len(pw) if pw else 0.0

print(f'{\"category\":<12}{\"mean recall\":>14}{\"std\":>8}{\"≥1 hit\":>10}')
print('-' * 50)
cat_topic = {}
for cat in ['chat', 'code', 'agent', 'reasoning']:
    cr = [r for r in phase16_results if r['category'] == cat]
    rs = [recall_metric(r['prompt'], r['explanation']) for r in cr]
    hits = [1 if x > 0 else 0 for x in rs]
    cat_topic[cat] = {
        'mean': statistics.mean(rs),
        'std': statistics.stdev(rs),
        'hit_rate': statistics.mean(hits),
    }
    t = cat_topic[cat]
    print(f'{cat:<12}{t[\"mean\"]:>14.3f}{t[\"std\"]:>8.3f}{t[\"hit_rate\"]:>10.3f}')

recall_spread = max(cat_topic[c]['mean'] for c in cat_topic) - min(cat_topic[c]['mean'] for c in cat_topic)
print(f'\\nrecall category spread: {recall_spread:.3f}')
print(f'V1 Qwen-7B:    spread 0.490 (chat 0.578 → agent 0.088)')
print(f'V2 Gemma-12B:  spread 0.649 (chat 0.782 → agent 0.133)')
print(f'Magnification predicts: V3 spread ≥ 0.65 (continues growing) OR ≈ 0.65 (saturates)')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Permutation
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 9. Control 1 — Permutation"))

cells.append(code("""# Cell 16: permutation control
real_recalls = [recall_metric(r['prompt'], r['explanation']) for r in phase16_results]
real_mean = statistics.mean(real_recalls)
print(f'  REAL pairings:        {real_mean:.3f}')

perm_within = []
for cat in ['chat', 'code', 'agent', 'reasoning']:
    cr = [r for r in phase16_results if r['category'] == cat]
    cps = [r['prompt'] for r in cr]
    ces = [r['explanation'] for r in cr]
    cps_sh = cps[:]; random.shuffle(cps_sh)
    perm_within.extend([recall_metric(p, e) for p, e in zip(cps_sh, ces)])
perm_within_mean = statistics.mean(perm_within)
print(f'  PERMUTED within-cat:  {perm_within_mean:.3f}  (gap: {real_mean-perm_within_mean:+.3f})')

shuf_idx = list(range(len(phase16_results))); random.shuffle(shuf_idx)
perm_cross = [recall_metric(phase16_results[shuf_idx[i]]['prompt'],
                             phase16_results[i]['explanation'])
              for i in range(len(phase16_results))]
perm_cross_mean = statistics.mean(perm_cross)
print(f'  PERMUTED cross-cat:   {perm_cross_mean:.3f}  (gap: {real_mean-perm_cross_mean:+.3f})')

print(f'\\nV1 Qwen:      gap +0.27')
print(f'V2 Gemma-12B: gap +0.38')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Random Gaussian
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 10. Control 2 — Random Gaussian baseline"))

cells.append(code("""# Cell 17: random Gaussian
mean_l2 = statistics.mean([a['act_l2'] for a in phase16_acts])
print(f'Mean L2 (V3 Gemma-27B): {mean_l2:.1f}')
print(f'  V1 Qwen-7B:   124.3')
print(f'  V2 Gemma-12B: 76020.5')

print('\\nReloading AV...')
av_tok = AutoTokenizer.from_pretrained(av_path)
av = AutoModelForCausalLM.from_pretrained(av_path, dtype=torch.bfloat16, device_map='cuda')
av.eval()

@torch.no_grad()
def verbalize_with_av(raw_act, temperature=1.0, max_new_tokens=200):
    text = av_tok.apply_chat_template(
        [{'role': 'user', 'content': AV_TEMPLATE.format(injection_char=INJECTION_CHAR)}],
        tokenize=False, add_generation_prompt=True,
    )
    input_ids = av_tok(text, return_tensors='pt').input_ids.to('cuda')
    embeds = av.get_input_embeddings()(input_ids)
    inj_pos = (input_ids[0] == INJECTION_TOKEN_ID).nonzero(as_tuple=True)[0][0].item()
    v = raw_act.to('cuda').to(torch.bfloat16)
    v_scaled = v / v.norm().clamp_min(1e-12) * INJECTION_SCALE
    embeds[0, inj_pos] = v_scaled
    out = av.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(input_ids),
                      max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                      temperature=temperature, pad_token_id=av_tok.eos_token_id)
    text_out = av_tok.decode(out[0], skip_special_tokens=True)
    m = re.search(r'<explanation>\\s*(.*?)\\s*</explanation>', text_out, re.DOTALL)
    return m.group(1).strip() if m else text_out.strip()

N_RANDOM = 30
random_acts_list = []
random_explanations = []
t0 = time.time()
for i in range(N_RANDOM):
    v = torch.randn(d_model)
    v = v / v.norm() * mean_l2
    random_acts_list.append(v)
    exp = verbalize_with_av(v, temperature=1.0)
    random_explanations.append(exp)
    if i < 5 or i == N_RANDOM-1:
        print(f'  [{i:2d}] {exp[:90]}...')
print(f'\\n✓ {N_RANDOM} random verbalizations in {time.time()-t0:.0f}s')

del av, av_tok
gc.collect(); torch.cuda.empty_cache()

random_results = []
for v, exp in zip(random_acts_list, random_explanations):
    rec = reconstruct(exp)
    mse, cos, fve_nrm = score_canonical(rec, v)
    random_results.append({'cos': cos, 'mse': mse, 'fve_nrm': fve_nrm,
                            'l2_orig': v.norm().item(), 'l2_rec': rec.norm().item(),
                            'explanation': exp})

rand_cos = statistics.mean([r['cos'] for r in random_results])
rand_fve = statistics.mean([r['fve_nrm'] for r in random_results])
print(f'\\nRandom Gaussian: cos={rand_cos:+.3f}  fve_nrm={rand_fve:+.3f}')
print(f'V1 Qwen:      random fve_nrm = -0.949')
print(f'V2 Gemma-12B: random fve_nrm = -0.992')

all_real_prompts = list({r['prompt'] for r in phase16_results})
random.shuffle(all_real_prompts)
rand_recalls = [recall_metric(all_real_prompts[i % len(all_real_prompts)], exp)
                for i, exp in enumerate(random_explanations)]
rand_recall_mean = statistics.mean(rand_recalls)
print(f'\\nRandom recall: {rand_recall_mean:.3f}')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Direction-injection
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 11. Control 3 — Direction-injection"))

cells.append(code("""# Cell 18: direction-injection
acts_by_cat_p = defaultdict(list)
for a in phase16_acts:
    acts_by_cat_p[a['category']].append(a['act'])

cat_means = {c: torch.stack(acts).mean(0) for c, acts in acts_by_cat_p.items()}
overall_mean_act = torch.stack(list(cat_means.values())).mean(0)

directions = {}
for cat in ['chat', 'code', 'agent', 'reasoning']:
    directions[f'{cat}_vs_other'] = cat_means[cat] - overall_mean_act
for cat in ['chat', 'code', 'agent', 'reasoning']:
    directions[f'NEG_{cat}_vs_other'] = -(cat_means[cat] - overall_mean_act)
directions['chat_minus_agent'] = cat_means['chat'] - cat_means['agent']
directions['agent_minus_chat'] = cat_means['agent'] - cat_means['chat']

print(f'Built {len(directions)} directions:')
for name, v in directions.items():
    print(f'  {name:<28}  L2={v.norm().item():>10.1f}')

print('\\nReloading AV...')
av_tok = AutoTokenizer.from_pretrained(av_path)
av = AutoModelForCausalLM.from_pretrained(av_path, dtype=torch.bfloat16, device_map='cuda')
av.eval()

@torch.no_grad()
def verbalize_direction(raw_act, temperature=1.0, max_new_tokens=200):
    text = av_tok.apply_chat_template(
        [{'role': 'user', 'content': AV_TEMPLATE.format(injection_char=INJECTION_CHAR)}],
        tokenize=False, add_generation_prompt=True,
    )
    input_ids = av_tok(text, return_tensors='pt').input_ids.to('cuda')
    embeds = av.get_input_embeddings()(input_ids)
    inj_pos = (input_ids[0] == INJECTION_TOKEN_ID).nonzero(as_tuple=True)[0][0].item()
    v = raw_act.to('cuda').to(torch.bfloat16)
    v_scaled = v / v.norm().clamp_min(1e-12) * INJECTION_SCALE
    embeds[0, inj_pos] = v_scaled
    out = av.generate(inputs_embeds=embeds, attention_mask=torch.ones_like(input_ids),
                      max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                      temperature=temperature, pad_token_id=av_tok.eos_token_id)
    text_out = av_tok.decode(out[0], skip_special_tokens=True)
    m = re.search(r'<explanation>\\s*(.*?)\\s*</explanation>', text_out, re.DOTALL)
    return m.group(1).strip() if m else text_out.strip()

K_DIR = 3
direction_results = {}
t0 = time.time()
for name, vec in directions.items():
    samples = [verbalize_direction(vec, temperature=1.0) for _ in range(K_DIR)]
    direction_results[name] = samples
    print(f'\\n--- {name}  (L2={vec.norm().item():.1f}) ---')
    for k, e in enumerate(samples):
        print(f'  s{k}: {e[:120]}')
print(f'\\n✓ {len(directions)*K_DIR} direction verbalizations in {time.time()-t0:.0f}s')

del av, av_tok
gc.collect(); torch.cuda.empty_cache()

expected_kw = {
    'chat':      {'wikipedia', 'article', 'encyclopedia', 'factual', 'biography', 'definition',
                  'history', 'wiki', 'informational', 'educational'},
    'code':      {'python', 'code', 'function', 'tutorial', 'script', 'programming', 'algorithm',
                  'def', 'syntax', 'snippet', 'documentation'},
    'agent':     {'technical', 'tool', 'command', 'documentation', 'configure', 'setup', 'cli',
                  'guide', 'shell', 'bash', 'instruction', 'devops'},
    'reasoning': {'math', 'proof', 'theorem', 'logic', 'puzzle', 'problem', 'calculation',
                  'formula', 'equation', 'reasoning', 'derivation', 'mathematical'},
}

def hit_count(text, kw_set):
    words = set(re.findall(r'[a-zA-Z]+', text.lower()))
    return len(kw_set & words)

print('\\n' + '=' * 88)
print('KEYWORD HIT MATRIX')
print('=' * 88)
print(f'{\"direction\":<28}{\"chat-kw\":>10}{\"code-kw\":>10}{\"agent-kw\":>10}{\"reason-kw\":>11}')
print('-' * 88)
dir_scores = {}
for name, exps in direction_results.items():
    hits = {cat: sum(hit_count(e, expected_kw[cat]) for e in exps) / K_DIR for cat in expected_kw}
    dir_scores[name] = hits
    print(f'{name:<28}{hits[\"chat\"]:>10.2f}{hits[\"code\"]:>10.2f}{hits[\"agent\"]:>10.2f}{hits[\"reasoning\"]:>11.2f}')

self_won_count = 0
for cat in ['chat', 'code', 'agent', 'reasoning']:
    name = f'{cat}_vs_other'
    h = dir_scores[name]
    if max(h, key=h.get) == cat:
        self_won_count += 1

print(f'\\nDirection→category alignment: {self_won_count}/4')
print(f'V1 Qwen:      4/4')
print(f'V2 Gemma-12B: 3/4 (agent collapses into code)')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 12. Save full results"))

cells.append(code("""# Cell 19: save
all_results = {
    'phase16_acts_meta': [{k: v for k, v in a.items() if k != 'act'} for a in phase16_acts],
    'phase16_explanations': phase16_explanations,
    'phase16_results': phase16_results,
    'cat_summary': cat_summary,
    'cat_topic': cat_topic,
    'controls': {
        'real_recall': real_mean,
        'perm_within_recall': perm_within_mean,
        'perm_cross_recall': perm_cross_mean,
        'random_gaussian_recall': rand_recall_mean,
        'random_gaussian_fve_nrm': rand_fve,
        'random_gaussian_cos': rand_cos,
    },
    'direction_interp': {
        'directions': {k: {'l2': v.norm().item()} for k, v in directions.items()},
        'direction_results': direction_results,
        'scores': dir_scores,
        'self_won_count': self_won_count,
    },
    'meta': {
        'AV_REPO': AV_REPO, 'AR_REPO': AR_REPO, 'TARGET_REPO': TARGET_REPO,
        'EXTRACTION_LAYER': EXTRACTION_LAYER, 'd_model': d_model,
        'INJECTION_SCALE': INJECTION_SCALE, 'INJECTION_CHAR': INJECTION_CHAR,
        'MSE_SCALE': MSE_SCALE, 'K_SAMPLES': K_SAMPLES, 'N_RANDOM': N_RANDOM,
    },
}
with open(OUT / 'phase16_full_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'✓ saved {OUT/\"phase16_full_results.json\"}')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Three-way scaling comparison
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 13. Three-way scaling comparison: Qwen-7B vs Gemma-12B vs Gemma-27B

If the decoupling-magnification thesis holds, V3 (Gemma-27B) should show:
- fve_nrm at or near ceiling (≥0.99)
- fve_nrm spread ≤0.005 (or shrinking)
- recall spread ≥0.65 (or growing)
- Random Gaussian collapse ≥−0.99
- Permutation gap ≥+0.38

If V3 saturates (similar to V2 Gemma-12B), magnification has a ceiling at
moderate model sizes — refines thesis, doesn't kill it.
"""))

cells.append(code("""# Cell 20: three-way scaling comparison
import os

def try_load(paths):
    for p in paths:
        if p.exists():
            try:
                return json.load(open(p)), p
            except Exception:
                continue
    return None, None

qwen_paths = [QWEN_OUT / 'phase16_full_results.json',
              QWEN_OUT_LEGACY / 'phase16_results_v2.json',
              QWEN_OUT_LEGACY / 'phase16_full_results.json']
gemma12_paths = [GEMMA12B_OUT / 'phase16_full_results.json']

qwen, qwen_p = try_load(qwen_paths)
gemma12, gemma12_p = try_load(gemma12_paths)

print(f'Qwen V1 path:     {qwen_p}')
print(f'Gemma-12B V2 path:{gemma12_p}')

if qwen is None and gemma12 is None:
    print('\\n⚠ V1 and V2 results not found in Drive. Skipping three-way comparison.')
else:
    print('\\n' + '=' * 95)
    print('THREE-WAY SCALING COMPARISON (Phase 16 V1 → V2 → V3)')
    print('=' * 95)
    print(f'{\"\":<28}{\"V1 Qwen-7B\":>15}{\"V2 Gemma-12B\":>17}{\"V3 Gemma-27B\":>17}{\"Δ V3-V2\":>10}')
    print('-' * 95)

    def get_summary(d):
        if d is None:
            return None
        cs = d['cat_summary']
        ct = d['cat_topic']
        return {
            'fve_overall': statistics.mean(cs[c]['fve_nrm'] for c in cs),
            'fve_spread': max(cs[c]['fve_nrm'] for c in cs) - min(cs[c]['fve_nrm'] for c in cs),
            'recall_overall': statistics.mean(ct[c]['mean'] for c in ct),
            'recall_spread': max(ct[c]['mean'] for c in ct) - min(ct[c]['mean'] for c in ct),
        }

    qsum = get_summary(qwen)
    g12sum = get_summary(gemma12)
    g27sum = {
        'fve_overall': statistics.mean(all_fves),
        'fve_spread': fve_spread,
        'recall_overall': statistics.mean([cat_topic[c]['mean'] for c in cat_topic]),
        'recall_spread': recall_spread,
    }

    def fmt(v):
        return f'{v:>+15.3f}' if v is not None else f'{\"-\":>15}'
    def fmt2(v):
        return f'{v:>+17.3f}' if v is not None else f'{\"-\":>17}'
    def delta(a, b):
        return f'{a-b:>+10.3f}' if a is not None and b is not None else f'{\"-\":>10}'

    rows = [
        ('Overall fve_nrm', 'fve_overall'),
        ('fve_nrm spread', 'fve_spread'),
        ('Overall recall', 'recall_overall'),
        ('Recall spread', 'recall_spread'),
    ]
    for label, key in rows:
        q = qsum[key] if qsum else None
        g12 = g12sum[key] if g12sum else None
        g27 = g27sum[key]
        print(f'{label:<28}{fmt(q)}{fmt2(g12)}{fmt2(g27)}{delta(g27, g12)}')

    print('\\nPer-category recall (V1 → V2 → V3):')
    for cat in ['chat', 'code', 'agent', 'reasoning']:
        q = qwen['cat_topic'][cat]['mean'] if qwen else None
        g12 = gemma12['cat_topic'][cat]['mean'] if gemma12 else None
        g27 = cat_topic[cat]['mean']
        print(f'  {cat:<10s}{fmt(q)}{fmt2(g12)}{fmt2(g27)}')

    if qwen and gemma12:
        qctrl = qwen.get('controls')
        g12ctrl = gemma12.get('controls')
        if qctrl and g12ctrl:
            print('\\nControls (V1 → V2 → V3):')
            print(f'  Permutation gap:    Qwen +{qctrl[\"real_recall\"]-qctrl[\"perm_cross_recall\"]:+.3f}  '
                  f'Gemma-12B +{g12ctrl[\"real_recall\"]-g12ctrl[\"perm_cross_recall\"]:+.3f}  '
                  f'Gemma-27B +{real_mean-perm_cross_mean:+.3f}')
            print(f'  Random fve_nrm:     Qwen {qctrl[\"random_gaussian_fve_nrm\"]:+.3f}  '
                  f'Gemma-12B {g12ctrl[\"random_gaussian_fve_nrm\"]:+.3f}  '
                  f'Gemma-27B {rand_fve:+.3f}')

    # Magnification verdict
    print('\\n' + '=' * 95)
    print('DECOUPLING MAGNIFICATION VERDICT')
    print('=' * 95)

    if qsum and g12sum:
        # Magnification: recall_spread should be increasing across the trajectory
        sp_traj = [qsum['recall_spread'], g12sum['recall_spread'], g27sum['recall_spread']]
        fve_traj = [qsum['fve_overall'], g12sum['fve_overall'], g27sum['fve_overall']]

        sp_growth_v2 = sp_traj[1] - sp_traj[0]
        sp_growth_v3 = sp_traj[2] - sp_traj[1]
        fve_growth_v2 = fve_traj[1] - fve_traj[0]
        fve_growth_v3 = fve_traj[2] - fve_traj[1]

        print(f'fve_nrm trajectory:  {fve_traj[0]:.3f} → {fve_traj[1]:.3f} → {fve_traj[2]:.3f}')
        print(f'  V1→V2: {fve_growth_v2:+.3f}  V2→V3: {fve_growth_v3:+.3f}')
        print(f'recall spread trajectory: {sp_traj[0]:.3f} → {sp_traj[1]:.3f} → {sp_traj[2]:.3f}')
        print(f'  V1→V2: {sp_growth_v2:+.3f}  V2→V3: {sp_growth_v3:+.3f}')

        if sp_growth_v3 > 0.02:
            print('\\n🎯 MAGNIFICATION CONTINUES at 27B — recall spread keeps growing with model quality.')
            print('   Paper-7 V3 thesis confirmed: better NLA training = sharper decoupling, no apparent ceiling.')
        elif sp_growth_v3 > -0.02:
            print('\\n🟡 MAGNIFICATION SATURATES at moderate model sizes.')
            print('   Refines thesis: decoupling magnification has a ceiling between 12B and 27B.')
            print('   Paper-7 V3 still confirms cross-model decoupling, with ceiling caveat.')
        else:
            print('\\n🟠 MAGNIFICATION REVERSES at 27B — recall spread shrinks.')
            print('   Refines thesis: at high training quality, NLA may begin capturing Tier 2 content.')
            print('   Paper-7 V3 reframes: magnification is a U-curve, not monotonic.')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 14. V3 verdict + paper update"))

cells.append(code("""# Cell 21: V3 verdict
print('=' * 65)
print('PHASE 16 V3 (Gemma-3-27B-L41) STANDALONE VERDICT')
print('=' * 65)

mean_fve = statistics.mean(all_fves)
mean_recall_overall = statistics.mean([cat_topic[c]['mean'] for c in cat_topic])

print(f'\\nReconstruction:        fve_nrm = {mean_fve:+.3f}')
print(f'Random Gaussian:       fve_nrm = {rand_fve:+.3f}')
print(f'Recall:                {real_mean:.3f}')
print(f'Permutation cross-cat: {perm_cross_mean:.3f}')
print(f'Direction-injection:   {self_won_count}/4 self-cat alignment')

print('\\nPer-category:')
for cat in ['chat', 'code', 'agent', 'reasoning']:
    print(f'  {cat:<10s}  fve_nrm={cat_summary[cat][\"fve_nrm\"]:+.3f}  recall={cat_topic[cat][\"mean\"]:.3f}')

print(f'\\nKey spreads:')
print(f'  fve_nrm:  {fve_spread:.3f}')
print(f'  recall:   {recall_spread:.3f}')

decoupling_confirmed = (
    fve_spread < 0.05 and recall_spread > 0.30
    and (real_mean - perm_cross_mean) > 0.10 and rand_fve < 0.5
)
if decoupling_confirmed:
    print('\\n🎯 V3 CONFIRMS two-tier decoupling — three-NLA-pair fortification.')
"""))

cells.append(md("""## 15. Summary — paper-7 V3 update

If V3 results align with V2 (uniform fve_nrm, large recall spread, format-prior
random Gaussian, mostly-self-cat direction-injection), paper-7 V3 abstract
upgrades:

> "We show across three NLA pairs from two model families spanning 7B → 12B →
> 27B parameters that..."

The magnification trajectory (V1 → V2 → V3) becomes the load-bearing observation:
either it continues monotonically (thesis confirmed, no ceiling), saturates at
~12B (thesis refined, ceiling at moderate sizes), or reverses (thesis reframed
as U-curve).

**Reproducibility**: this notebook + V1 + V2. Combined runtime ~120-180 min on
RTX 6000 96GB.

**License**: Apache-2.0; Gemma-3-27B-IT under Google's Gemma Terms of Use.
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Write
# ──────────────────────────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py", "mimetype": "text/x-python",
            "name": "python", "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3", "version": "3.11",
        },
        "colab": {"provenance": []},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
print(f"Wrote {NOTEBOOK_PATH}")
print(f"  Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, {sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
