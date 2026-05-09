"""Build Track A Phase 16 V2 — Gemma-3-12B-L32 cross-model fortification.

Produces: notebooks/nb_track_a_phase16_gemma_crossmodel.ipynb

Replicates the Phase 16 N=150 decoupling experiment on the second NLA pair from
the kitft release (Gemma-3-12B at L32) to test whether the two-tier verbalization
thesis is a property of NLA training generally or a Qwen2.5-7B artifact.

Differences from Phase 16 V1 (Qwen2.5-7B):
- Target: google/gemma-3-12b-it (gated, requires HF_TOKEN)
- AV/AR: kitft/nla-gemma3-12b-L32-{av,ar}
- Extraction: layer 32 (vs 20)
- d_model: 3840 (vs 3584) — auto-detected
- injection_scale: 80000 (vs 150) — due to Gemma's √d embed scaling
- injection_char: ㈜ U+321C (vs ㈎ U+320E)
- BOS prefix matters for AR tokenization (Qwen no-op; Gemma load-bearing)

Final cell loads Qwen Phase 16 V1 results (if present at
/content/drive/.../track_a_phase16/) and produces side-by-side comparison.

Run: python3 scripts/build_nb_track_a_phase16_gemma_crossmodel.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "nb_track_a_phase16_gemma_crossmodel.ipynb"


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
cells.append(md("""# Track A Phase 16 V2 — Cross-Model Fortification (Gemma-3-12B-L32)

**Goal**: Test whether the *two-tier verbalization* thesis from Phase 16 V1
(`nb_track_a_phase16_decoupling.ipynb`, on Qwen2.5-7B-NLA-L20) holds on a second
NLA pair from a different model family — `kitft/nla-gemma3-12b-L32`. If yes,
the decoupling between `fve_nrm` and semantic-content recall is a property of
NLA training generally, not a Qwen artifact.

**Phase 16 V1 (Qwen2.5-7B) found**:
- `fve_nrm` uniform at 0.880 (spread 0.017 across 4 categories)
- Keyword recall category-spread 0.490 (chat 0.578 → agent 0.088, 6.5×)
- Permutation gap +0.27 above floor (agent gap +0.045 floor-level)
- Random Gaussian: `fve_nrm` = −0.949, format-locked explanations
- Direction-injection: 4/4 self-cat alignment + 4/4 negation symmetry

**V2 prediction** (two-tier thesis, model-agnostic):
- Same uniform `fve_nrm` across categories
- Same category-spread in recall (chat ≫ agent)
- Same controls patterns

**V2 falsifications** (would refine thesis):
- (B) Recall uniform → Gemma's training mix covers content better → format-prior weaker
- (C) Random Gaussian doesn't collapse → AR less input-dependent
- (D) Direction-injection fails → format-prior is fixed template, not direction-modulated

**Key adaptations** vs V1 (all auto-loaded from `nla_meta.yaml`):
- Target: `google/gemma-3-12b-it` (gated — requires HF_TOKEN)
- AV/AR: `kitft/nla-gemma3-12b-L32-{av,ar}`
- L32 extraction (vs L20)
- d_model 3840 (vs 3584)
- injection_scale 80000 (vs 150) — Gemma's √d embed scaling
- injection_char ㈜ U+321C (vs ㈎ U+320E)
- BOS prefix critical for AR (Qwen had no BOS; Gemma loses 0.77→0.31 fve_nrm without)

**Final cell**: side-by-side comparison vs Phase 16 V1 (loads Qwen results
from Drive if present).

**Estimated runtime**: ~35-45 min on H100 (gated repo download + 3 model loads
+ 200 generations).

**License**: Apache-2.0 (notebook); kitft NLA pairs Apache-2.0;
Gemma-3-12B-IT under Google's Gemma Terms of Use.
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup"))

cells.append(code("""# Cell 1: install
!pip install -q transformers safetensors pyyaml huggingface_hub
"""))

cells.append(code("""# Cell 2: Drive mount + output dir
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
OUT = Path('/content/drive/MyDrive/openinterp_runs/track_a_phase16_gemma')
OUT.mkdir(parents=True, exist_ok=True)
print(f'OUT = {OUT}')

# Optional: also reference V1 (Qwen) results dir for cross-model comparison at end
QWEN_OUT = Path('/content/drive/MyDrive/openinterp_runs/track_a_phase16')
QWEN_OUT_LEGACY = Path('/content/drive/MyDrive/openinterp_runs/track_a_nla_qwen25_poc')
print(f'QWEN_OUT (V1, paths): {QWEN_OUT} (or legacy {QWEN_OUT_LEGACY})')
"""))

cells.append(code("""# Cell 3: HF login — Gemma-3 is GATED. Required before snapshot_download.
import os
from huggingface_hub import login, whoami

if not os.environ.get('HF_TOKEN'):
    print('HF_TOKEN not set in environment. Running interactive login...')
    from huggingface_hub import notebook_login
    notebook_login()
else:
    login(token=os.environ['HF_TOKEN'])

try:
    user = whoami()
    print(f'✓ Authenticated as {user[\"name\"]}')
except Exception as e:
    raise SystemExit(f'HF auth failed: {e}. Run notebook_login() and retry.')
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

# kitft NLA Gemma-3-12B L32 (Apache-2.0)
AV_REPO = 'kitft/nla-gemma3-12b-L32-av'
AR_REPO = 'kitft/nla-gemma3-12b-L32-ar'
TARGET_REPO = 'google/gemma-3-12b-it'  # gated
EXTRACTION_LAYER = 32  # residual stream output of block 32
device = 'cuda'

assert torch.cuda.is_available(), 'GPU required'
print(f'GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 2. Download NLA pair + parse sidecars"))

cells.append(code("""# Cell 5: download AV + AR
print('Downloading AV (Gemma-3-12B-L32)...')
av_path = snapshot_download(repo_id=AV_REPO, cache_dir='/content/hf_cache')
print(f'  → {av_path}')

print('Downloading AR (incl. value_head.safetensors)...')
ar_path = snapshot_download(repo_id=AR_REPO, cache_dir='/content/hf_cache')
print(f'  → {ar_path}')
"""))

cells.append(code("""# Cell 6: parse nla_meta.yaml from both AV and AR
av_meta = yaml.safe_load(open(f'{av_path}/nla_meta.yaml').read())
ar_meta = yaml.safe_load(open(f'{ar_path}/nla_meta.yaml').read())

INJECTION_CHAR = av_meta['tokens']['injection_char']
INJECTION_TOKEN_ID = av_meta['tokens']['injection_token_id']
INJECTION_SCALE = float(av_meta['extraction']['injection_scale'])  # Gemma: 80000
MSE_SCALE = float(ar_meta['extraction']['mse_scale'])  # √d_model
AV_TEMPLATE = av_meta['prompt_templates']['av']
AR_TEMPLATE = ar_meta['prompt_templates']['ar']
EXTRACTION_LAYER_INDEX = ar_meta.get('extraction_layer_index',
                                      ar_meta.get('critic', {}).get('extraction_layer_index', EXTRACTION_LAYER))

assert EXTRACTION_LAYER_INDEX == EXTRACTION_LAYER, (
    f'mismatch: meta={EXTRACTION_LAYER_INDEX} vs constant={EXTRACTION_LAYER}'
)

print(f'INJECTION_CHAR    = {INJECTION_CHAR!r}  (token id {INJECTION_TOKEN_ID})')
print(f'INJECTION_SCALE   = {INJECTION_SCALE}  (Gemma √d embed scaling — was 150 for Qwen)')
print(f'MSE_SCALE         = {MSE_SCALE:.6f}')
print(f'EXTRACTION_LAYER  = {EXTRACTION_LAYER_INDEX}')
print(f'AR_TEMPLATE       = {AR_TEMPLATE!r}')
print(f'\\nAV_TEMPLATE preview (first 300 chars):\\n{AV_TEMPLATE[:300]}...')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Prompt corpus
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 3. 50-prompt corpus (identical to V1 — apples-to-apples comparison)

Same 4-category × 50-prompt corpus as V1 Phase 16. Identical wording. The
*activations* will differ (Gemma residual stream at L32 ≠ Qwen residual
stream at L20) but the prompt distribution is held constant for direct
comparison.
"""))

cells.append(code("""# Cell 7: define 50 prompts (verbatim from V1)
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
for cat, prompts in PHASE_16_PROMPTS.items():
    print(f'  {cat:10s}: {len(prompts)} prompts')
assert total >= 50
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Capture
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 4. Capture L32 activations from Gemma-3-12B-IT (target)

Forward each prompt through Gemma's chat template, then through the model with
a forward hook on `model.layers[32]`. Take the last input token's residual.
"""))

cells.append(code("""# Cell 8: load target Gemma-3-12B-IT
print(f'Loading {TARGET_REPO} (gated)...')
target_tok = AutoTokenizer.from_pretrained(TARGET_REPO)
target = AutoModelForCausalLM.from_pretrained(TARGET_REPO, dtype=torch.bfloat16, device_map='cuda')
target.eval()
print(f'Architecture: {type(target).__name__}')
print(f'num_hidden_layers: {target.config.num_hidden_layers}')
print(f'hidden_size: {target.config.hidden_size}')
print(f'VRAM after target load: {torch.cuda.memory_allocated()/1e9:.2f} GB')

# Verify d_model matches sidecar's expected
expected_d = av_meta['d_model']
assert target.config.hidden_size == expected_d, (
    f'd_model mismatch: target={target.config.hidden_size} vs sidecar={expected_d}'
)
print(f'✓ d_model={target.config.hidden_size} matches sidecar')
"""))

cells.append(code("""# Cell 9: capture all 50 acts at L32 with forward hook
captured = {}
def hook_fn(module, inputs, output):
    h = output[0] if isinstance(output, tuple) else output
    captured['h'] = h.detach()

# Some Gemma variants nest inner model differently. Find layers list robustly.
inner_model = target.model if hasattr(target, 'model') else target
if hasattr(inner_model, 'language_model'):
    inner_model = inner_model.language_model
layers = inner_model.layers
print(f'Hooking layer {EXTRACTION_LAYER} of {len(layers)} total')

handle = layers[EXTRACTION_LAYER].register_forward_hook(hook_fn)

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

# L2 stats per category — Gemma residual L2 should be very different from Qwen
print('\\nL2 stats per category (compare to Qwen V1: ~120-128):')
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
cells.append(md("""## 5. Verbalize 50 × K=3 = 150 with AV (Gemma)

Each activation rescaled to L2 = `INJECTION_SCALE` (80000 for Gemma) before
embedding-injection.
"""))

cells.append(code("""# Cell 10: load AV + verbalize 150
K_SAMPLES = 3

print('Loading AV (Gemma-3-12B AV)...')
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
print(f'✓ saved {OUT/\"phase16_explanations.pt\"} + .json')

del av, av_tok
gc.collect(); torch.cuda.empty_cache()
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# AR canonical
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 6. AR canonical recipe (Gemma-3 specifics)

Same canonical recipe as V1, with one robustness adjustment: Gemma's final
layernorm attribute name may differ from Qwen's (`norm` vs `final_layernorm`).
We loop through known names per kitft NLACritic implementation.
"""))

cells.append(code("""# Cell 11: load AR + canonical patches
print('Loading AR (Gemma-3-12B AR truncated to L32)...')
ar_tok = AutoTokenizer.from_pretrained(ar_path)
ar = AutoModelForCausalLM.from_pretrained(ar_path, dtype=torch.bfloat16, device_map='cuda')
ar.eval()

# Strip lm_head — critic never emits logits
ar.lm_head = nn.Identity()

# Strip final layernorm — value_head sees raw post-block-K residual.
# Try multiple attribute names to handle Gemma vs Qwen architecture differences.
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
assert norm_replaced is not None, (
    f'No final-LN attribute found on {type(inner_ar).__name__}. '
    f'Tried: {FINAL_LN_ATTRS}. Add the right name to the loop.'
)
print(f'✓ replaced inner_model.{norm_replaced} with Identity (was final RMSNorm)')

# Load value_head from separate safetensors
vh_state = load_file(f'{ar_path}/value_head.safetensors')
d_model = ar.config.hidden_size
value_head = nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16)
value_head.load_state_dict(vh_state)
value_head = value_head.to('cuda').eval()

print(f'✓ AR canonical: {norm_replaced}=Identity, lm_head=Identity, value_head Linear({d_model},{d_model}) loaded')
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"""))

cells.append(code("""# Cell 12: reconstruct() + score_canonical()
@torch.inference_mode()
def reconstruct(explanation):
    \"\"\"Explanation text → reconstructed activation vector.\"\"\"
    prompt = AR_TEMPLATE.format(explanation=explanation)
    # add_special_tokens=True: BOS prefix is LOAD-BEARING for Gemma.
    # kitft README: dropping BOS tanks Gemma fve_nrm 0.77→0.31.
    ids = ar_tok(prompt, return_tensors='pt', add_special_tokens=True,
                 truncation=True, max_length=1024).input_ids.to('cuda')

    # Some Gemma3 variants nest model deeper
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

# Sanity: verify BOS handling
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
print(f'{\"category\":<12}{\"N\":>4}{\"cos\":>8}{\"fve_nrm\":>10}{\"fve_std\":>10}{\"within-prompt\":>15}')
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
          f'{s[\"fve_std\"]:>+10.3f}{s[\"within_prompt_std\"]:>+15.3f}')

print('-' * 75)
all_fves = [r['fve_nrm'] for r in phase16_results]
print(f'{\"OVERALL\":<12}{len(phase16_results):>4}'
      f'{statistics.mean([r[\"cos\"] for r in phase16_results]):>+8.3f}'
      f'{statistics.mean(all_fves):>+10.3f}')

fve_spread = max(cat_summary[c]['fve_nrm'] for c in cat_summary) - min(cat_summary[c]['fve_nrm'] for c in cat_summary)
print(f'\\nfve_nrm category spread: {fve_spread:.3f}')
print(f'Qwen V1 reference: spread 0.017')
print(f'Two-tier prediction: spread < 0.05 = uniform reconstruction')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Topic-match
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 8. Topic-match analysis (semantic recall)"))

cells.append(code("""# Cell 15: keyword recall metric
STOPWORDS = set('the a an and or but of to in on at for with from by about as is are was were be been being have has had do does did i you he she it we they this that these those what which who whom whose how why when where can could would should may might must shall will not no yes if then else than'.split())

def cwords(t, m=4):
    return set(x for x in re.findall(r'[a-zA-Z]+', t.lower())
               if len(x) >= m and x not in STOPWORDS)

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
print(f'Qwen V1 reference: spread 0.490 (chat 0.578 → agent 0.088)')
print(f'Two-tier prediction: spread > 0.30 = decoupled from fve_nrm')
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
    cps_sh = cps[:]
    random.shuffle(cps_sh)
    perm_within.extend([recall_metric(p, e) for p, e in zip(cps_sh, ces)])
perm_within_mean = statistics.mean(perm_within)
print(f'  PERMUTED within-cat:  {perm_within_mean:.3f}  (gap: {real_mean - perm_within_mean:+.3f})')

shuf_idx = list(range(len(phase16_results)))
random.shuffle(shuf_idx)
perm_cross = [recall_metric(phase16_results[shuf_idx[i]]['prompt'],
                             phase16_results[i]['explanation'])
              for i in range(len(phase16_results))]
perm_cross_mean = statistics.mean(perm_cross)
print(f'  PERMUTED cross-cat:   {perm_cross_mean:.3f}  (gap: {real_mean - perm_cross_mean:+.3f})')

print(f'\\nQwen V1 reference: real 0.329, perm-within 0.091, perm-cross 0.063 (gap +0.27)')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Random Gaussian
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 10. Control 2 — Random Gaussian baseline"))

cells.append(code("""# Cell 17: random Gaussian baseline
mean_l2 = statistics.mean([a['act_l2'] for a in phase16_acts])
print(f'Mean L2 in Phase 16: {mean_l2:.1f}  (Qwen V1: 124.3)')

# Reload AV
print('\\nReloading AV for random verbalization...')
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
    if i < 5 or i == N_RANDOM - 1:
        print(f'  [{i:2d}] {exp[:90]}...')
print(f'\\n✓ {N_RANDOM} random-act verbalizations in {time.time()-t0:.0f}s')

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
print(f'Real Gemma:      cos={statistics.mean([r[\"cos\"] for r in phase16_results]):+.3f}  fve_nrm={statistics.mean(all_fves):+.3f}')
print(f'\\nQwen V1 reference: random fve_nrm = -0.949 (collapse), real fve_nrm = +0.880')

all_real_prompts = list({r['prompt'] for r in phase16_results})
random.shuffle(all_real_prompts)
rand_recalls = [recall_metric(all_real_prompts[i % len(all_real_prompts)], exp)
                for i, exp in enumerate(random_explanations)]
rand_recall_mean = statistics.mean(rand_recalls)
print(f'\\nRecall (random-act exp ↔ random real prompt): {rand_recall_mean:.3f}')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Direction-injection
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 11. Control 3 — Direction-injection probe interp test"))

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

print(f'Built {len(directions)} directions (rescaled to {INJECTION_SCALE} before injection):')
for name, v in directions.items():
    print(f'  {name:<28}  L2={v.norm().item():>10.1f}')

# Reload AV
print('\\nReloading AV for direction injection...')
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
    samples = []
    for k in range(K_DIR):
        samples.append(verbalize_direction(vec, temperature=1.0))
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
print(f'Qwen V1 reference: 4/4 alignment')
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
# Cross-model comparison
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 13. Cross-model comparison: Gemma-3-12B-L32 vs Qwen2.5-7B-L20

If V1 (Qwen Phase 16) results are present in Drive, compare side-by-side. The
two-tier thesis predicts both models show the same qualitative pattern:
uniform `fve_nrm` across categories, large category spread in recall, and
matching control signatures.
"""))

cells.append(code("""# Cell 20: cross-model comparison vs Qwen V1
import os

# Try multiple V1 paths (current + legacy)
qwen_paths = [
    QWEN_OUT / 'phase16_full_results.json',
    QWEN_OUT_LEGACY / 'phase16_results_v2.json',  # if user ran in original POC dir
    QWEN_OUT_LEGACY / 'phase16_full_results.json',
]
qwen_results = None
qwen_controls = None
qwen_directions = None
qwen_path_used = None
for p in qwen_paths:
    if p.exists():
        try:
            d = json.load(open(p))
        except Exception:
            continue
        # Two file shapes are possible:
        #   - phase16_results_v2.json: {phase16_results, cat_summary, cat_topic}
        #   - phase16_full_results.json: same + controls + direction_interp
        qwen_results = d
        qwen_controls = d.get('controls')  # may be None for v2 file
        qwen_directions = d.get('direction_interp')
        qwen_path_used = p
        break

# If we have only the partial v2 file, try to also load companion controls/direction files
if qwen_results and qwen_controls is None:
    legacy = QWEN_OUT_LEGACY
    ctrl_p = legacy / 'phase16_controls.json'
    if ctrl_p.exists():
        c = json.load(open(ctrl_p))
        qwen_controls = {
            'real_recall': c.get('real_pairings_mean_recall'),
            'perm_within_recall': c.get('permuted_within_cat_recall'),
            'perm_cross_recall': c.get('permuted_cross_cat_recall'),
            'random_gaussian_recall': c.get('random_gaussian_recall'),
            'random_gaussian_fve_nrm': statistics.mean([r['fve_nrm'] for r in c['random_results']]) if 'random_results' in c else None,
            'random_gaussian_cos': statistics.mean([r['cos'] for r in c['random_results']]) if 'random_results' in c else None,
        }
    di_p = legacy / 'phase16_direction_interp.json'
    if di_p.exists():
        qwen_directions = json.load(open(di_p))

print(f'Qwen V1 results path: {qwen_path_used}')

if qwen_results is None:
    print('⚠ Qwen V1 results not found in Drive. Skipping comparison.')
    print('   To enable: run nb_track_a_phase16_decoupling.ipynb first; results saved to QWEN_OUT.')
else:
    qwen_cat_summary = qwen_results['cat_summary']
    qwen_cat_topic = qwen_results['cat_topic']
    qwen_fve_spread = (max(qwen_cat_summary[c]['fve_nrm'] for c in qwen_cat_summary) -
                       min(qwen_cat_summary[c]['fve_nrm'] for c in qwen_cat_summary))
    qwen_recall_spread = (max(qwen_cat_topic[c]['mean'] for c in qwen_cat_topic) -
                          min(qwen_cat_topic[c]['mean'] for c in qwen_cat_topic))

    gemma_overall_fve = statistics.mean(all_fves)
    qwen_overall_fve = statistics.mean(qwen_cat_summary[c]['fve_nrm'] for c in qwen_cat_summary)

    print('\\n' + '=' * 80)
    print('CROSS-MODEL COMPARISON (Phase 16 V1 Qwen vs V2 Gemma)')
    print('=' * 80)
    print(f'{\"\":<28}{\"Qwen-7B-L20\":>16}{\"Gemma-12B-L32\":>18}{\"Δ\":>10}')
    print('-' * 80)
    print(f'{\"Overall fve_nrm\":<28}{qwen_overall_fve:>+16.3f}{gemma_overall_fve:>+18.3f}{gemma_overall_fve - qwen_overall_fve:>+10.3f}')
    print(f'{\"fve_nrm category spread\":<28}{qwen_fve_spread:>16.3f}{fve_spread:>18.3f}{fve_spread - qwen_fve_spread:>+10.3f}')
    print()
    print(f'{\"recall (overall)\":<28}{statistics.mean(qwen_cat_topic[c][\"mean\"] for c in qwen_cat_topic):>+16.3f}{statistics.mean(cat_topic[c][\"mean\"] for c in cat_topic):>+18.3f}')
    print(f'{\"recall category spread\":<28}{qwen_recall_spread:>16.3f}{recall_spread:>18.3f}{recall_spread - qwen_recall_spread:>+10.3f}')
    print()
    print('Per-category recall:')
    for cat in ['chat', 'code', 'agent', 'reasoning']:
        q = qwen_cat_topic[cat]['mean']
        g = cat_topic[cat]['mean']
        print(f'  {cat:<10s}{q:>+16.3f}{g:>+18.3f}{g - q:>+10.3f}')

    if qwen_controls:
        print('\\nControls comparison:')
        print(f'  Random Gaussian fve_nrm:   Qwen={qwen_controls.get(\"random_gaussian_fve_nrm\"):+.3f}  Gemma={rand_fve:+.3f}')
        print(f'  Random Gaussian recall:    Qwen={qwen_controls.get(\"random_gaussian_recall\"):.3f}  Gemma={rand_recall_mean:.3f}')
        print(f'  Permutation real-cross:    Qwen={qwen_controls.get(\"real_recall\") - qwen_controls.get(\"perm_cross_recall\"):+.3f}  Gemma={real_mean - perm_cross_mean:+.3f}')

    if qwen_directions:
        print(f'\\nDirection-injection self-cat alignment:')
        print(f'  Qwen V1:  {qwen_directions.get(\"self_won_count\", qwen_directions.get(\"n_correct\")):>1}/4')
        print(f'  Gemma V2: {self_won_count}/4')

    # Two-tier thesis verdict
    print('\\n' + '=' * 80)
    print('TWO-TIER THESIS CROSS-MODEL VERDICT')
    print('=' * 80)
    gemma_thesis = (fve_spread < 0.05 and recall_spread > 0.30
                    and (real_mean - perm_cross_mean) > 0.10
                    and rand_fve < 0.5)
    qwen_thesis = qwen_fve_spread < 0.05 and qwen_recall_spread > 0.30
    if gemma_thesis and qwen_thesis:
        print('🎯 TWO-TIER THESIS REPLICATES across both NLA pairs.')
        print('   Format-vs-content decoupling is a property of NLA training generally,')
        print('   not a Qwen2.5-7B-L20 artifact. Paper-7 V2 fortification confirmed.')
    elif gemma_thesis:
        print('🟡 Gemma satisfies thesis but Qwen reference numbers are unusual — re-inspect.')
    elif qwen_thesis:
        print('🟡 Qwen replicates thesis but Gemma differs — refines thesis to model-specific.')
    else:
        print('🔴 Neither model fully satisfies thesis as initially stated — refine.')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Verdict + findings
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 14. V2 verdict"))

cells.append(code("""# Cell 21: V2 verdict (Gemma-only, before cross-model comparison context)
print('=' * 65)
print('PHASE 16 V2 (Gemma-3-12B-L32) VERDICT')
print('=' * 65)

mean_fve = statistics.mean(all_fves)
mean_recall_overall = statistics.mean([cat_topic[c]['mean'] for c in cat_topic])

print(f'\\nReconstruction (fve_nrm):')
print(f'  Real (Gemma Phase 16):  {mean_fve:+.3f}')
print(f'  Random Gaussian:        {rand_fve:+.3f}')

print(f'\\nKeyword recall:')
print(f'  Real:                   {real_mean:.3f}')
print(f'  Perm cross-cat:         {perm_cross_mean:.3f}')
print(f'  Random Gaussian:        {rand_recall_mean:.3f}')

print(f'\\nPer-category:')
for cat in ['chat', 'code', 'agent', 'reasoning']:
    print(f'  {cat:<10s}  fve_nrm={cat_summary[cat][\"fve_nrm\"]:+.3f}  recall={cat_topic[cat][\"mean\"]:.3f}')

print(f'\\nKey spreads:')
print(f'  fve_nrm:  {fve_spread:.3f}    (V1 Qwen: 0.017)')
print(f'  recall:   {recall_spread:.3f}    (V1 Qwen: 0.490)')

print(f'\\nDirection-injection: {self_won_count}/4 self-cat alignment  (V1 Qwen: 4/4)')

decoupling_confirmed = (
    fve_spread < 0.05 and recall_spread > 0.30
    and (real_mean - perm_cross_mean) > 0.10 and rand_fve < 0.5
)
if decoupling_confirmed:
    print('\\n🎯 V2 GEMMA: two-tier decoupling reproduces.')
    print('   Combined with V1 Qwen results = paper-7 thesis is cross-model fortified.')
else:
    print('\\n🟡 V2 GEMMA: at least one criterion fails. See cross-model cell for refinement.')
"""))

cells.append(md("""## 15. Summary — what V2 contributes to paper-7

If the cross-model comparison cell shows uniform `fve_nrm` and category-spread
recall for both Qwen2.5-7B-L20 and Gemma-3-12B-L32, the two-tier verbalization
thesis is no longer a single-model observation. The paper-7 abstract claim
upgrades from:

> "We show on `kitft/nla-qwen2.5-7b-L20` that..."

to:

> "We show across two NLA pairs (Qwen2.5-7B-L20 and Gemma-3-12B-L32) that...,
> ruling out the model-family/architecture as the source of decoupling."

If V2 Gemma gives a different signal — e.g., recall uniform, or random
Gaussian doesn't collapse — paper-7 V2 reframes from "thesis confirmed" to
"thesis refined: format-prior magnitude varies with NLA training distribution
diversity." Both outcomes inform the position.

**Reproducibility**: this notebook plus `nb_track_a_phase16_decoupling.ipynb`
(V1 Qwen) covers both models. Combined runtime: ~70 min on H100.

**License**: Apache-2.0. Gemma-3-12B-IT under Google's Gemma Terms of Use.
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Write notebook
# ──────────────────────────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11",
        },
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
print(f"Wrote {NOTEBOOK_PATH}")
print(f"  Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, {sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
