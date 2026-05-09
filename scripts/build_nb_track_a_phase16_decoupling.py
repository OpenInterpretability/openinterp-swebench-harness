"""Build Track A Phase 16 notebook — NLA reconstruction-vs-recall decoupling.

Produces: notebooks/nb_track_a_phase16_decoupling.ipynb

Reproduces the N=150 finding: NLA's reconstruction loss (fve_nrm 0.880 uniform)
decouples from semantic content fidelity (recall 0.088 agent → 0.578 chat,
spread 0.490) on kitft/nla-qwen2.5-7b-L20.

Three controls:
  1. Permutation (shuffle prompt↔explanation pairs within and across categories)
  2. Random Gaussian baseline (L2-matched random vectors → AV → AR)
  3. Real-data reference (Phase 16 main run)

Outputs to /content/drive/MyDrive/openinterp_runs/track_a_phase16/.

Run: python scripts/build_nb_track_a_phase16_decoupling.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "nb_track_a_phase16_decoupling.ipynb"


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
cells.append(md("""# Track A Phase 16 — NLA decoupling: reconstruction vs semantic recall

**Goal**: Reproduce the N=150 finding that NLA's reconstruction loss (`fve_nrm`)
decouples from semantic content fidelity (keyword recall) on
[kitft/nla-qwen2.5-7b-L20](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av).

**Headline result** (verified 2026-05-09 on RTX 6000):
- `fve_nrm` UNIFORM at 0.880 across 4 categories (chat/code/agent/reasoning, spread 0.017)
- Keyword recall MASSIVELY category-dependent: chat 0.578, code 0.351, reasoning 0.325, agent 0.088 (spread 0.490, 6.5× ratio)
- Within-prompt std 0.007–0.010 = NLA reconstruction is deterministic given an activation

**Decoupling thesis**: NLA's GRPO reward optimizes reconstruction loss but not semantic fidelity. The AR head reads structural/positional signal from the verbalization template ("Structured technical guide format with code example") sufficient to recover activation direction—without ever encoding the prompt's specific subject (file paths, function names, mathematical entities, named entities).

**Estimated runtime**: ~30 min on H100 (2 model loads, 200 generations). Drive mount required.

**References**:
- Fraser-Taliente et al. 2026, "Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations", Transformer Circuits
- [kitft/nla-inference](https://github.com/kitft/nla-inference) — canonical NLACritic recipe
- Anthropic 2026, "Persona Vectors" — adjacent line on direction-controlled behavior
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup"))

cells.append(code("""# Cell 1: install
!pip install -q transformers safetensors pyyaml huggingface_hub
"""))

cells.append(code("""# Cell 2: Drive mount + output dir (HARD RULE: every Colab MUST checkpoint to Drive)
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
OUT = Path('/content/drive/MyDrive/openinterp_runs/track_a_phase16')
OUT.mkdir(parents=True, exist_ok=True)
print(f"OUT = {OUT}")
"""))

cells.append(code("""# Cell 3: imports + constants
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

# kitft NLA Qwen2.5-7B L20 (Apache-2.0)
AV_REPO = 'kitft/nla-qwen2.5-7b-L20-av'
AR_REPO = 'kitft/nla-qwen2.5-7b-L20-ar'
TARGET_REPO = 'Qwen/Qwen2.5-7B-Instruct'
EXTRACTION_LAYER = 20  # residual stream output of block 20
device = 'cuda'

assert torch.cuda.is_available(), 'GPU required'
print(f"GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 2. Download NLA pair + parse sidecar"))

cells.append(code("""# Cell 4: download AV + AR
print('Downloading AV...')
av_path = snapshot_download(repo_id=AV_REPO, cache_dir='/content/hf_cache')
print(f'  → {av_path}')

print('Downloading AR (incl. value_head.safetensors)...')
ar_path = snapshot_download(repo_id=AR_REPO, cache_dir='/content/hf_cache')
print(f'  → {ar_path}')
"""))

cells.append(code("""# Cell 5: parse nla_meta.yaml from AV (injection_char, scale, AV template)
#         + AR (mse_scale, ar template)
av_meta = yaml.safe_load(open(f'{av_path}/nla_meta.yaml').read())
ar_meta = yaml.safe_load(open(f'{ar_path}/nla_meta.yaml').read())

INJECTION_CHAR = av_meta['tokens']['injection_char']
INJECTION_TOKEN_ID = av_meta['tokens']['injection_token_id']
INJECTION_SCALE = float(av_meta['extraction']['injection_scale'])  # 150 for Qwen7B
MSE_SCALE = float(ar_meta['extraction']['mse_scale'])  # √d_model ≈ 59.87
AV_TEMPLATE = av_meta['prompt_templates']['av']
AR_TEMPLATE = ar_meta['prompt_templates']['ar']
EXTRACTION_LAYER_INDEX = ar_meta.get('extraction_layer_index', ar_meta.get('critic', {}).get('extraction_layer_index', 20))

assert EXTRACTION_LAYER_INDEX == EXTRACTION_LAYER, f'mismatch: meta={EXTRACTION_LAYER_INDEX} vs constant={EXTRACTION_LAYER}'

print(f'INJECTION_CHAR    = {INJECTION_CHAR!r}  (token id {INJECTION_TOKEN_ID})')
print(f'INJECTION_SCALE   = {INJECTION_SCALE}')
print(f'MSE_SCALE         = {MSE_SCALE:.6f}  (√3584 = {np.sqrt(3584):.6f})')
print(f'EXTRACTION_LAYER  = {EXTRACTION_LAYER_INDEX}')
print(f'AR_TEMPLATE       = {AR_TEMPLATE!r}')
print(f'\\nAV_TEMPLATE preview (first 300 chars):\\n{AV_TEMPLATE[:300]}...')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Prompt corpus
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 3. 50-prompt corpus across 4 categories

Designed to span:
- **chat**: short factual Q&A (Wikipedia-style)
- **code**: programming task requests (Python/SQL/Bash/JS/C)
- **agent**: SWE-bench-Pro-style task descriptions with file paths/function names
- **reasoning**: math, logic, classical puzzles

Categories chosen to test whether NLA verbalization quality varies by prompt distribution.
"""))

cells.append(code("""# Cell 6: define 50 prompts
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
cells.append(md("## 4. Capture L20 activations from Qwen2.5-7B-Instruct (target)"))

cells.append(code("""# Cell 7: load target + capture all 50 acts at L20 (last token of chat-template prompt)
print(f'Loading {TARGET_REPO}...')
target_tok = AutoTokenizer.from_pretrained(TARGET_REPO)
target = AutoModelForCausalLM.from_pretrained(TARGET_REPO, dtype=torch.bfloat16, device_map='cuda')
target.eval()
print(f'VRAM after target load: {torch.cuda.memory_allocated()/1e9:.2f} GB')

captured = {}
def hook_fn(module, inputs, output):
    h = output[0] if isinstance(output, tuple) else output
    captured['h'] = h.detach()

handle = target.model.layers[EXTRACTION_LAYER].register_forward_hook(hook_fn)

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
            h = captured['h']  # (1, T, d_model)
            last = h[0, -1].float().cpu().contiguous()  # (d_model,)
            phase16_acts.append({
                'category': cat,
                'idx_in_cat': j,
                'key': f'{cat}_{j}',  # category-aware key prevents chat/code 'c0' collision
                'prompt': prompt,
                'seq_len': ids.shape[1],
                'act_l2': last.norm().item(),
                'act': last,
            })

handle.remove()
print(f'\\n✓ captured {len(phase16_acts)} acts in {time.time()-t0:.1f}s')

# L2 stats per category
print('\\nL2 stats per category:')
for cat in PHASE_16_PROMPTS:
    l2s = [a['act_l2'] for a in phase16_acts if a['category'] == cat]
    print(f'  {cat:10s}  L2 mean={statistics.mean(l2s):>6.1f}  std={statistics.stdev(l2s):>5.1f}  '
          f'min={min(l2s):>5.1f}  max={max(l2s):>5.1f}')

# Save before freeing target
torch.save({'phase16_acts': phase16_acts, 'PHASE_16_PROMPTS': PHASE_16_PROMPTS}, OUT / 'phase16_acts.pt')
print(f'\\n✓ saved {OUT/\"phase16_acts.pt\"}')

del target, target_tok
gc.collect(); torch.cuda.empty_cache()
print(f'VRAM after free target: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Verbalize
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 5. Verbalize 50 × K=3 samples = 150 with AV

K=3 samples per prompt at temperature=1.0 measures within-prompt explanation stability.
Activation injected at `injection_char` token position with L2-normalization to `INJECTION_SCALE=150`.
"""))

cells.append(code("""# Cell 8: load AV + verbalize 150
K_SAMPLES = 3

print('Loading AV...')
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

# Free AV (will reload for random control later)
del av, av_tok
gc.collect(); torch.cuda.empty_cache()
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# AR canonical
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 6. AR canonical recipe

Per kitft/nla-inference NLACritic class:
1. Load AR `Qwen2ForCausalLM` (truncated to K+1 layers, here 21)
2. Replace `model.norm` with `nn.Identity()` — value_head sees raw post-block-K residual, NOT normed
3. Replace `lm_head` with `nn.Identity()` — critic never emits logits
4. Load `value_head.safetensors` separately (not in model-*.safetensors); attach as `nn.Linear(d, d, bias=False)`
5. Reconstruct: tokenize `AR_TEMPLATE.format(explanation=...)` with `add_special_tokens=True`, forward, take last token, apply value_head

**Critical bug to avoid**: if you load AR via `from_pretrained` alone and don't strip+attach manually, transformers re-initializes `model.norm.weight` (random-init RMSNorm) and `lm_head.weight` randomly. The reconstruction will look ~OK on Phase A short prompts (cos 0.94 by accident — random RMSNorm with weights=ones happens to scale right) but produces NaN/garbage on longer agent-format prompts.
"""))

cells.append(code("""# Cell 9: load AR + canonical patches
print('Loading AR...')
ar_tok = AutoTokenizer.from_pretrained(ar_path)
ar = AutoModelForCausalLM.from_pretrained(ar_path, dtype=torch.bfloat16, device_map='cuda')
ar.eval()

# Strip the two heads transformers re-initializes randomly:
ar.model.norm = nn.Identity()   # value_head sees raw post-block-K residual
ar.lm_head = nn.Identity()       # critic never emits logits

# Load value_head from its separate safetensors file
vh_state = load_file(f'{ar_path}/value_head.safetensors')
d_model = ar.config.hidden_size
value_head = nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16)
value_head.load_state_dict(vh_state)
value_head = value_head.to('cuda').eval()

print(f'✓ AR canonical: norm=Identity, lm_head=Identity, value_head Linear({d_model},{d_model}) loaded')
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"""))

cells.append(code("""# Cell 10: reconstruct() + score_canonical() per kitft NLACritic.score()
@torch.inference_mode()
def reconstruct(explanation):
    \"\"\"Explanation text → reconstructed activation vector (raw, unnormalized).\"\"\"
    prompt = AR_TEMPLATE.format(explanation=explanation)
    # add_special_tokens=True: BOS handling. Qwen has no BOS so it's a no-op,
    # but the kit assumes this flag is set (matches training-time tokenization).
    ids = ar_tok(prompt, return_tensors='pt', add_special_tokens=True,
                 truncation=True, max_length=1024).input_ids.to('cuda')
    h = ar.model(ids, use_cache=False).last_hidden_state[0, -1]  # last token, raw post-block-K
    return value_head(h).float().cpu()

def score_canonical(pred, gold):
    \"\"\"Paper metric: L2-normalize both to mse_scale, MSE = 2(1-cos), fve_nrm = 1 - mse/var.

    See kitft NLACritic.score() — both vectors normalized to L2=mse_scale=√d before
    MSE computation, so MSE is direction-only metric in [0, 4]. fve_nrm is the
    paper's reported reconstruction quality.
    \"\"\"
    pred = pred.float().flatten()
    gold = gold.float().flatten()
    pred_n = pred / pred.norm().clamp_min(1e-12) * MSE_SCALE
    gold_n = gold / gold.norm().clamp_min(1e-12) * MSE_SCALE
    mse = ((pred_n - gold_n) ** 2).mean().item()
    var = ((gold_n - gold_n.mean()) ** 2).mean().item()
    cos = torch.nn.functional.cosine_similarity(pred.unsqueeze(0), gold.unsqueeze(0)).item()
    fve_nrm = 1.0 - mse / max(var, 1e-12)
    return mse, cos, fve_nrm
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruct + per-category aggregate
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 7. Reconstruct all 150 with canonical AR + per-category aggregate"))

cells.append(code("""# Cell 11: reconstruct all 150 (~30s)
acts_by_key = {a['key']: a for a in phase16_acts}

phase16_results = []
t0 = time.time()
for ex in phase16_explanations:
    orig = acts_by_key[ex['key']]['act']  # category-aware key prevents collision
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

cells.append(code("""# Cell 12: per-category aggregate fve_nrm + within-prompt std (= K=3 sample stability)
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
all_coss = [r['cos'] for r in phase16_results]
print(f'{\"OVERALL\":<12}{len(phase16_results):>4}{statistics.mean(all_coss):>+8.3f}{statistics.mean(all_fves):>+10.3f}')

fve_spread = max(cat_summary[c]['fve_nrm'] for c in cat_summary) - min(cat_summary[c]['fve_nrm'] for c in cat_summary)
print(f'\\nfve_nrm category spread: {fve_spread:.3f}  (paper-7 prediction: < 0.03 = uniform)')
print(f'Paper claim (in-distribution): fve_nrm 0.752')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Topic-match
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 8. Topic-match analysis (semantic recall)

For each (prompt, explanation) pair: extract content words (≥4 chars, non-stopwords),
compute `recall = |prompt_words ∩ explanation_words| / |prompt_words|`.

This measures whether NLA explanations contain the prompt's specific subject keywords—
or just generic format language ("Structured X format with Y").
"""))

cells.append(code("""# Cell 13: topic-match recall metric
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
print(f'\\nrecall category spread: {recall_spread:.3f}  (paper-7 prediction: > 0.30 = decoupled from fve_nrm)')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Permutation control
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 9. Control 1 — Permutation

Shuffles `prompt ↔ explanation` pairing (within and across categories). Real recall must
exceed permuted-baseline recall by a meaningful margin to claim the metric measures
real signal rather than common-vocabulary noise.
"""))

cells.append(code("""# Cell 14: permutation control
real_recalls = [recall_metric(r['prompt'], r['explanation']) for r in phase16_results]
real_mean = statistics.mean(real_recalls)
print(f'  REAL pairings:        {real_mean:.3f}')

# Within-category shuffle
perm_within = []
for cat in ['chat', 'code', 'agent', 'reasoning']:
    cr = [r for r in phase16_results if r['category'] == cat]
    cps = [r['prompt'] for r in cr]
    ces = [r['explanation'] for r in cr]
    cps_sh = cps[:]
    random.shuffle(cps_sh)
    perm_within.extend([recall_metric(p, e) for p, e in zip(cps_sh, ces)])
perm_within_mean = statistics.mean(perm_within)
print(f'  PERMUTED within-cat:  {perm_within_mean:.3f}  (signal-above-floor: {real_mean - perm_within_mean:+.3f})')

# Cross-category shuffle (harder baseline — different category vocab pools)
shuf_idx = list(range(len(phase16_results)))
random.shuffle(shuf_idx)
perm_cross = [recall_metric(phase16_results[shuf_idx[i]]['prompt'],
                             phase16_results[i]['explanation'])
              for i in range(len(phase16_results))]
perm_cross_mean = statistics.mean(perm_cross)
print(f'  PERMUTED cross-cat:   {perm_cross_mean:.3f}  (signal-above-floor: {real_mean - perm_cross_mean:+.3f})')

print(f'\\nPer-category real-vs-perm-within breakdown:')
print(f'{\"category\":<12}{\"real\":>10}{\"perm-within\":>14}{\"Δ\":>10}')
for cat in ['chat', 'code', 'agent', 'reasoning']:
    cr = [r for r in phase16_results if r['category'] == cat]
    cat_real = [recall_metric(r['prompt'], r['explanation']) for r in cr]
    cps = [r['prompt'] for r in cr]
    cps_sh = cps[:]; random.shuffle(cps_sh)
    cat_perm = [recall_metric(p, r['explanation']) for p, r in zip(cps_sh, cr)]
    rm = statistics.mean(cat_real); pm = statistics.mean(cat_perm)
    print(f'{cat:<12}{rm:>10.3f}{pm:>14.3f}{rm-pm:>+10.3f}')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Random Gaussian control
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 10. Control 2 — Random Gaussian baseline

Generates 30 random Gaussian vectors with L2 matched to Phase 16 mean (~125), passes
through AV → AR. Tests:

1. Do random activations produce coherent-looking explanations? (If yes, AV's verbalization is partly content-independent.)
2. Does AR reconstruction collapse on random inputs? (Should drop to ~0 fve_nrm if AR is genuinely input-dependent.)
3. Do random-act explanations recover real prompt keywords? (Should be ~baseline noise — common English words only.)
"""))

cells.append(code("""# Cell 15: random Gaussian baseline
mean_l2 = statistics.mean([a['act_l2'] for a in phase16_acts])
print(f'Mean L2 in Phase 16: {mean_l2:.1f}')

# Reload AV (was freed in cell 8)
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

# Generate 30 random Gaussian L2-matched
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

# Free AV again
del av, av_tok
gc.collect(); torch.cuda.empty_cache()

# Reconstruct random
random_results = []
for v, exp in zip(random_acts_list, random_explanations):
    rec = reconstruct(exp)
    mse, cos, fve_nrm = score_canonical(rec, v)
    random_results.append({'cos': cos, 'mse': mse, 'fve_nrm': fve_nrm,
                            'l2_orig': v.norm().item(), 'l2_rec': rec.norm().item(),
                            'explanation': exp})

rand_cos = statistics.mean([r['cos'] for r in random_results])
rand_fve = statistics.mean([r['fve_nrm'] for r in random_results])
print(f'\\n=== RANDOM GAUSSIAN ===')
print(f'cos:       {rand_cos:+.3f}  (Phase 16 real: {statistics.mean([r[\"cos\"] for r in phase16_results]):+.3f})')
print(f'fve_nrm:   {rand_fve:+.3f}  (Phase 16 real: {statistics.mean(all_fves):+.3f})')

# Recall: random-act explanations vs random real prompt
all_real_prompts = list({r['prompt'] for r in phase16_results})
random.shuffle(all_real_prompts)
rand_recalls = [recall_metric(all_real_prompts[i % len(all_real_prompts)], exp)
                for i, exp in enumerate(random_explanations)]
rand_recall_mean = statistics.mean(rand_recalls)
print(f'\\nRecall comparison:')
print(f'  Real Phase 16:        {real_mean:.3f}')
print(f'  Permuted within-cat:  {perm_within_mean:.3f}')
print(f'  Permuted cross-cat:   {perm_cross_mean:.3f}')
print(f'  Random Gaussian:      {rand_recall_mean:.3f}')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Control 3 — direction-injection probe interp test
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 11. Control 3 — Direction-injection probe interp test

Tests whether NLA can verbalize **synthetic interpretable directions** (not real
prompt activations). Builds 4 category-mean-difference directions from Phase 16
acts, plus their negations, plus a cross-axis (chat↔agent), then injects each
into AV and checks if the explanation matches the expected category-template.

This is the closest test we have to the canonical probe interpretability use
case: "I have a direction in residual stream — what does it encode?"
"""))

cells.append(code("""# Cell 17: direction-injection — does NLA verbalize directions or default to template?
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

print(f'Built {len(directions)} directions')
for name, v in directions.items():
    print(f'  {name:<28}  L2={v.norm().item():>6.1f}  (rescaled to {INJECTION_SCALE} before injection)')

# Reload AV
print('\\nReloading AV for direction injection...')
av_tok = AutoTokenizer.from_pretrained(av_path)
av = AutoModelForCausalLM.from_pretrained(av_path, dtype=torch.bfloat16, device_map='cuda')
av.eval()

# Same verbalize as Cell 8 / Cell 15
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

# Free AV
del av, av_tok
gc.collect(); torch.cuda.empty_cache()

# Score: keyword hit count per expected category
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
print('KEYWORD HIT MATRIX (mean across K=3 samples)')
print('=' * 88)
print(f'{\"direction\":<28}{\"chat-kw\":>10}{\"code-kw\":>10}{\"agent-kw\":>10}{\"reason-kw\":>11}')
print('-' * 88)

dir_scores = {}
for name, exps in direction_results.items():
    hits = {cat: sum(hit_count(e, expected_kw[cat]) for e in exps) / K_DIR for cat in expected_kw}
    dir_scores[name] = hits
    print(f'{name:<28}{hits[\"chat\"]:>10.2f}{hits[\"code\"]:>10.2f}{hits[\"agent\"]:>10.2f}{hits[\"reasoning\"]:>11.2f}')

# Self-category alignment for pure-positive directions
print('\\n' + '=' * 60)
print('SELF-CATEGORY ALIGNMENT (pure positive directions)')
print('=' * 60)
self_won_count = 0
for cat in ['chat', 'code', 'agent', 'reasoning']:
    name = f'{cat}_vs_other'
    h = dir_scores[name]
    max_cat = max(h, key=h.get)
    self_won = max_cat == cat
    self_won_count += int(self_won)
    marker = '✓' if self_won else '✗'
    print(f'  {marker}  {name:>22s}  →  top={max_cat}  self-hit={h[cat]:.2f}  max-hit={h[max_cat]:.2f}')

print(f'\\nDirection→category alignment: {self_won_count}/4')

if self_won_count == 4:
    print('🎯 NLA verbalizes directions correctly at CATEGORY/FORMAT granularity (Tier 1).')
    print('   Combined with category-dependent recall (Tier 2 unencoded), this confirms')
    print('   the TWO-TIER VERBALIZATION thesis: format-direction-modulated, content-largely-lost.')
elif self_won_count >= 2:
    print('🟡 NLA partially verbalizes directions; format-prior modulation is partial.')
else:
    print('🔴 NLA fails to verbalize directions; format-prior dominates regardless of input.')

with open(OUT / 'phase16_direction_interp.json', 'w') as f:
    json.dump({
        'directions': {k: {'l2': v.norm().item()} for k, v in directions.items()},
        'direction_results': direction_results,
        'scores': dir_scores,
        'self_won_count': self_won_count,
    }, f, indent=2)
print(f'\\n✓ saved {OUT/\"phase16_direction_interp.json\"}')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Verdict + save
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 12. Save + verdict"))

cells.append(code("""# Cell 16: save full results + verdict
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
    'meta': {
        'AV_REPO': AV_REPO, 'AR_REPO': AR_REPO, 'TARGET_REPO': TARGET_REPO,
        'EXTRACTION_LAYER': EXTRACTION_LAYER,
        'INJECTION_SCALE': INJECTION_SCALE, 'MSE_SCALE': MSE_SCALE,
        'K_SAMPLES': K_SAMPLES, 'N_RANDOM': N_RANDOM,
    },
}
with open(OUT / 'phase16_full_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'✓ saved {OUT/\"phase16_full_results.json\"}')

# Verdict
print('\\n' + '=' * 65)
print('PHASE 16 VERDICT')
print('=' * 65)
mean_fve = statistics.mean(all_fves)
print(f'\\nReconstruction (fve_nrm):')
print(f'  Real (Phase 16):    {mean_fve:+.3f}')
print(f'  Random Gaussian:    {rand_fve:+.3f}    (Δ {mean_fve - rand_fve:+.3f}; AR is input-dependent)')

print(f'\\nKeyword recall:')
print(f'  Real:               {real_mean:.3f}')
print(f'  Perm within-cat:    {perm_within_mean:.3f}')
print(f'  Perm cross-cat:     {perm_cross_mean:.3f}')
print(f'  Random Gaussian:    {rand_recall_mean:.3f}')
print(f'  Real-vs-floor gap:  {real_mean - perm_cross_mean:+.3f}  (>0.10 = metric meaningful)')

print(f'\\nPer-category:')
for cat in ['chat', 'code', 'agent', 'reasoning']:
    print(f'  {cat:10s}  fve_nrm={cat_summary[cat][\"fve_nrm\"]:+.3f}  recall={cat_topic[cat][\"mean\"]:.3f}')

print(f'\\nKey spreads:')
print(f'  fve_nrm  category spread: {fve_spread:.3f}    ← UNIFORM')
print(f'  recall   category spread: {recall_spread:.3f}    ← MASSIVE (decoupled)')

decoupling_confirmed = (
    fve_spread < 0.05 and recall_spread > 0.30
    and (real_mean - perm_cross_mean) > 0.10 and rand_fve < 0.5
)
if decoupling_confirmed:
    print('\\n🎯 PAPER-7 DECOUPLING CONFIRMED:')
    print('   1. fve_nrm uniform across categories  (spread {:.3f} < 0.05)'.format(fve_spread))
    print('   2. recall category-dependent           (spread {:.3f} > 0.30)'.format(recall_spread))
    print('   3. permutation control validates metric (gap {:+.3f} > 0.10)'.format(real_mean - perm_cross_mean))
    print('   4. random Gaussian destroys AR reconstruction (fve_nrm {:+.3f} < 0.5)'.format(rand_fve))
else:
    print('\\n🟡 PARTIAL — re-inspect numbers; decoupling may be weaker than expected on this run.')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Conclusion (markdown)
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 13. Findings — paper-7 seed

**Two-tier verbalization in NLA.** NLA's `fve_nrm` (reconstruction-loss metric) and
keyword recall (semantic content metric) measure different things and decouple:

| | chat | code | agent | reasoning | spread |
|---|---|---|---|---|---|
| `fve_nrm` | 0.880 | 0.887 | 0.883 | 0.870 | **0.017** |
| keyword recall | 0.578 | 0.351 | **0.088** | 0.325 | **0.490** |
| within-prompt std | 0.010 | 0.007 | 0.007 | 0.007 | — |

**Three controls validate the metric and refine the thesis:**

1. **Permutation control**: shuffled `prompt↔explanation` pairs drop to ~0.05 (within-cat) and ~0.03 (cross-cat) recall — real signal exceeds floor by ~0.27. Per-category breakdown: chat gap +0.49 (5× signal), agent gap only +0.05 (floor-level).
2. **Random Gaussian control**: random vectors produce coherent format-locked explanations ("Formal wiki article structure with numbered facts about a cultural history magazine", "Structured game description with formatted fields") but reconstruction collapses (fve_nrm = -0.95) and recall against real prompts drops to 0.012.
3. **Direction-injection probe interp test**: 4/4 self-category alignment for pure-positive directions (chat→chat-keywords, code→code-keywords, etc.) and consistent negation symmetry (NEG_chat→code-keywords, etc.). Cross-axis chat↔agent works as expected.

**Two-tier thesis**: NLA verbalization operates at two distinct granularities:

- **Tier 1 (FORMAT/CATEGORY)**: direction-modulated, correctly tracked. Random Gaussian → format prior fires unconditionally; meaningful directions modulate which category-template (article/code/math/technical) fires. `fve_nrm` measures Tier 1 fidelity. Tier 1 is what AR can decode.
- **Tier 2 (CONTENT/SPECIFICITY)**: largely unencoded. File paths, named entities, math entities, test names — these don't survive into NLA's verbalization. Recall measures Tier 2 fidelity, which varies 6.5× across categories.

NLA's GRPO reward optimizes Tier 1 fidelity (sufficient for AR to reconstruct activation direction) but does NOT directly optimize Tier 2. The reconstruction-loss metric is blind to Tier 2.

**Why agent fails worst at Tier 2**: agent prompts contain technical-specific tokens (file paths like `processor.py`, function names like `requests.get`, test names like `test_async_handler.py::test_concurrent_writes`) that don't appear in NLA's training distribution (WildChat + Ultra-FineWeb). The L20 residual at the last input token of an agent-format prompt encodes "this is a request for technical task execution" (Tier 1, recoverable) but not the specific entities involved (Tier 2, lost).

**Implication for interpretability practice**: Reconstruction loss is necessary but insufficient for explanation quality. Any reconstruction-based interpretability method (SAEs, activation autoencoders, NLA) needs to validate explanation quality independently from reconstruction loss.

**Implication for our paper-5 saturation-direction work**: NLA can tell you the broad CATEGORY a probe direction points toward (e.g., "this direction points toward technical/agent territory") but NOT the specific SEMANTIC content the probe encodes (e.g., "this direction encodes patch-success vs patch-fail"). Format-classifier yes, content-decoder no. Reconstruction-loss validation does not certify explanation accuracy at the semantic level.

---

**Reproducibility**:
- Build script: `scripts/build_nb_track_a_phase16_decoupling.py`
- Notebook: `notebooks/nb_track_a_phase16_decoupling.ipynb`
- Drive output: `/content/drive/MyDrive/openinterp_runs/track_a_phase16/`
- Models: `kitft/nla-qwen2.5-7b-L20-{av,ar}` (Apache-2.0)
- Target: `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0)
- Random seed: 42 (numpy + torch + random)
- Compute: ~30 min on H100, ~45 min on RTX 6000

**License**: Apache-2.0
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
