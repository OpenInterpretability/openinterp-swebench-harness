"""Build Track A Phase 17 — AV Residual Decode Pilot (Phase A0 of paper-7 V4 mech).

Produces: notebooks/nb_track_a_phase17_av_decode_pilot.ipynb

Cheap (~25-35 min on H100, ~$3) pilot that decides whether SAE training on
the AV residual stream is warranted for paper-7 V4 mechanistic.

Pipeline (per NLA pair, dual-model: Qwen2.5-7B + Gemma-3-12B):
  1. Capture target activations at L_extract for 200 paper-7 prompts
     (50 × 4 categories — chat/code/agent/reasoning).
  2. Free target, load AV.
  3. For each prompt: single forward pass through AV with injection-token
     position embedding replaced by L2-scaled activation. Capture residual
     stream AT injection_char position at multiple layers via forward hooks.
  4. For each captured layer: train sklearn LogisticRegression with 4-class
     softmax + 5-fold CV → report macro AUROC.
  5. Compare AUROC × layer × model. Decision rule:
       AUROC > 0.9        → category linearly decodable, SAE training justified
       0.7 < AUROC < 0.9  → marginal, SAE with smaller expansion (8k)
       < 0.7              → reformulate (try other positions: last input token,
                            first generation token; or escalate to nonlinear)

Layers swept:
  - Qwen2.5-7B AV (28 layers): [10, 14, 18, 22, 26]
  - Gemma-3-12B AV (48 layers): [16, 24, 32, 40, 44]

Estimated cost: ~$3 (≈30 min H100). No SAE training, no generation.

Run: python3 scripts/build_nb_track_a_phase17_av_decode_pilot.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "nb_track_a_phase17_av_decode_pilot.ipynb"


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
cells.append(md("""# Track A Phase 17 — AV Residual Decode Pilot

**Paper-7 V4 Phase A0**: cheap pilot deciding whether SAE training on the
NLA AV residual stream is warranted for the V4 mechanistic extension.

## The question

Phase 16 (V1+V2+V3, paper-7 published) showed `fve_nrm` decouples from
keyword recall: `fve_nrm` is uniform across categories at high level, while
recall varies 6.5–8.8× (chat ≫ agent). Two-tier thesis: Tier 1 (format /
category) is direction-modulated and what `fve_nrm` measures; Tier 2
(specific content) is largely unencoded.

V4 mechanistic asks: **where in the AV residual stream is Tier 1 located?**

The Anthropic NLA paper (transformer-circuits.pub/2026/nla) explicitly lists
this gap as a limitation:
> "Lack of mechanistic grounding: NLAs are blackboxes by construction; we
> cannot determine which aspects of an activation drove a given component
> of an explanation."

## Phase A0 design (this notebook)

Before spending $30-50 training an SAE on AV residual, run a $3 pilot that
linearly decodes category from raw AV residual. If category is linearly
decodable at high AUROC at some layer, SAE training will recover clean
categorical features. If not, reformulate.

**Pipeline** (per NLA pair):
1. Capture target activations at L_extract on the 200 paper-7 prompts
2. Free target, load AV
3. Forward each prompt through AV with embedding injection (scaled to
   `INJECTION_SCALE`); capture residual at multiple AV layers at the
   `injection_char` token position (one forward pass per prompt — no
   generation needed)
4. For each layer: 5-fold CV `LogisticRegression` (4-class softmax) →
   macro AUROC + confusion matrix
5. Compare layers × models, decide V4 path

**Decision rule**:

| AUROC at best layer | V4 implication |
|---|---|
| > 0.9 | Category linearly decodable → SAE training recovers clean features → **proceed Phase A** |
| 0.7–0.9 | Marginal → SAE with smaller expansion (8k not 16k) |
| < 0.7 | Reformulate: try last-input-token / first-generation-token position; or escalate to nonlinear methods |

## Scope (matches paper-7 V1+V2 rigor)

- Qwen2.5-7B-L20 (`kitft/nla-qwen2.5-7b-L20-{av,ar}`): 28-layer AV. Sweep AV layers [10, 14, 18, 22, 26].
- Gemma-3-12B-L32 (`kitft/nla-gemma3-12b-L32-{av,ar}`): 48-layer AV. Sweep AV layers [16, 24, 32, 40, 44].

Skipping Gemma-3-27B-L41 for Phase A0 (cost outweighs marginal information at
pilot stage; if Phase A confirms signal, V4 paper repeats on 27B).

## Greenfield context

No public general-purpose SAE on Gemma-3-12B exists (only one domain-scoped
to coding). Our V4 SAE would be the **first general-purpose Gemma-3-12B SAE
AND the first SAE on any NLA component** — two artifact firsts.

## License

Apache-2.0. kitft NLA pairs Apache-2.0. Qwen2.5-7B-Instruct Apache-2.0.
Gemma-3-12B-IT under Google's Gemma Terms of Use.

## Estimated runtime

~30 min on H100 (target download cached from V3; AV download ~5 min;
target capture 200 prompts ~3 min; AV forward 200 prompts × 5 layers ~15 min;
probes train in seconds).
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup"))

cells.append(code("""# Cell 1: install
!pip install -q transformers safetensors pyyaml huggingface_hub scikit-learn
"""))

cells.append(code("""# Cell 2: Drive mount + output dirs
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
OUT = Path('/content/drive/MyDrive/openinterp_runs/track_a_phase17_av_decode')
OUT.mkdir(parents=True, exist_ok=True)
print(f'OUT = {OUT}')
"""))

cells.append(code("""# Cell 3: HF login (Gemma is gated)
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

cells.append(code("""# Cell 4: imports + seeds
import torch, torch.nn as nn
import json, re, time, gc, random, statistics
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import yaml
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

device = 'cuda'
assert torch.cuda.is_available(), 'GPU required'
print(f'GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Per-pair config
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("## 2. Dual-pair config"))

cells.append(code("""# Cell 5: NLA pair configs
NLA_PAIRS = {
    'qwen25_7b': {
        'av_repo': 'kitft/nla-qwen2.5-7b-L20-av',
        'ar_repo': 'kitft/nla-qwen2.5-7b-L20-ar',
        'target_repo': 'Qwen/Qwen2.5-7B-Instruct',
        'extraction_layer': 20,
        # AV is 28-layer Qwen2.5-7B; sweep middle/late layers
        'av_probe_layers': [10, 14, 18, 22, 26],
        'd_model_expected': 3584,
        'add_special_tokens_av': False,  # Qwen no BOS
    },
    'gemma3_12b': {
        'av_repo': 'kitft/nla-gemma3-12b-L32-av',
        'ar_repo': 'kitft/nla-gemma3-12b-L32-ar',
        'target_repo': 'google/gemma-3-12b-it',
        'extraction_layer': 32,
        # AV is 48-layer Gemma-3-12B; sweep middle/late layers
        'av_probe_layers': [16, 24, 32, 40, 44],
        'd_model_expected': 3840,
        'add_special_tokens_av': True,  # Gemma BOS load-bearing
    },
}

print('NLA pairs to pilot:')
for k, v in NLA_PAIRS.items():
    print(f'  {k}: target {v[\"target_repo\"]}, L{v[\"extraction_layer\"]}, '
          f'd_model {v[\"d_model_expected\"]}, AV probe layers {v[\"av_probe_layers\"]}')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Prompt corpus
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 3. 50-prompt corpus (verbatim from Phase 16)

Same corpus as paper-7 V1+V2+V3 — apples-to-apples comparison.
"""))

cells.append(code("""# Cell 6: prompt corpus
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
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 4. Helpers — capture target acts, AV multi-layer hooks, probe

These functions are pair-agnostic. The Run section calls them per NLA pair.
"""))

cells.append(code("""# Cell 7: capture_target_acts
def capture_target_acts(target_repo: str, extraction_layer: int):
    \"\"\"Load target, capture L_extract residual at last input token for each prompt, free.\"\"\"
    print(f'Loading target {target_repo} ...')
    target_tok = AutoTokenizer.from_pretrained(target_repo)
    target = AutoModelForCausalLM.from_pretrained(target_repo, dtype=torch.bfloat16, device_map='cuda')
    target.eval()
    print(f'  arch: {type(target).__name__}, n_layers: {target.config.num_hidden_layers if not hasattr(target.config, \"text_config\") else target.config.text_config.num_hidden_layers}')

    captured = {}
    def hook_fn(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured['h'] = h.detach()

    inner_model = target.model if hasattr(target, 'model') else target
    if hasattr(inner_model, 'language_model'):
        inner_model = inner_model.language_model
    layers = inner_model.layers
    handle = layers[extraction_layer].register_forward_hook(hook_fn)

    acts = []
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
                acts.append({
                    'category': cat,
                    'idx_in_cat': j,
                    'key': f'{cat}_{j}',
                    'prompt': prompt,
                    'act': last,
                    'act_l2': last.norm().item(),
                })
    handle.remove()
    print(f'  captured {len(acts)} target acts in {time.time()-t0:.1f}s')

    del target, target_tok, inner_model, layers
    gc.collect(); torch.cuda.empty_cache()
    print(f'  VRAM after free: {torch.cuda.memory_allocated()/1e9:.2f} GB')
    return acts
"""))

cells.append(code("""# Cell 8: capture_av_residuals
def capture_av_residuals(av_path: str, target_acts: list, probe_layers: list,
                         d_model_expected: int, add_special_tokens: bool):
    \"\"\"Load AV, run each act through AV (single forward pass with injection), capture
    residuals at probe_layers at the injection_char token position. Returns dict
    {layer: np.ndarray (N, d_model)} + labels list.\"\"\"
    print(f'Loading AV from {av_path} ...')
    av_tok = AutoTokenizer.from_pretrained(av_path)
    av = AutoModelForCausalLM.from_pretrained(av_path, dtype=torch.bfloat16, device_map='cuda')
    av.eval()
    print(f'  AV n_layers: {av.config.num_hidden_layers if not hasattr(av.config, \"text_config\") else av.config.text_config.num_hidden_layers}')
    print(f'  AV hidden_size: {av.config.hidden_size if not hasattr(av.config, \"text_config\") else av.config.text_config.hidden_size}')

    # Parse sidecar for INJECTION values
    av_meta = yaml.safe_load(open(f'{av_path}/nla_meta.yaml').read())
    INJECTION_CHAR = av_meta['tokens']['injection_char']
    INJECTION_TOKEN_ID = av_meta['tokens']['injection_token_id']
    INJECTION_SCALE = float(av_meta['extraction']['injection_scale'])
    AV_TEMPLATE = av_meta['prompt_templates']['av']
    print(f'  INJECTION_CHAR={INJECTION_CHAR!r} (id {INJECTION_TOKEN_ID})')
    print(f'  INJECTION_SCALE={INJECTION_SCALE}')

    # Hooks at probe_layers
    inner = av.model if hasattr(av, 'model') else av
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    layers_attr = inner.layers

    captured_per_layer = {L: None for L in probe_layers}
    def make_hook(L):
        def hook_fn(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured_per_layer[L] = h.detach()
        return hook_fn
    handles = [layers_attr[L].register_forward_hook(make_hook(L)) for L in probe_layers]

    residuals = {L: [] for L in probe_layers}
    labels = []
    keys = []
    t0 = time.time()
    with torch.no_grad():
        for i, item in enumerate(target_acts):
            text = av_tok.apply_chat_template(
                [{'role': 'user', 'content': AV_TEMPLATE.format(injection_char=INJECTION_CHAR)}],
                tokenize=False, add_generation_prompt=True,
            )
            input_ids = av_tok(text, return_tensors='pt', add_special_tokens=add_special_tokens).input_ids.to('cuda')
            embeds = av.get_input_embeddings()(input_ids)
            inj_pos_t = (input_ids[0] == INJECTION_TOKEN_ID).nonzero(as_tuple=True)[0]
            if inj_pos_t.numel() == 0:
                raise RuntimeError(f'Injection token {INJECTION_TOKEN_ID!r} not found in input_ids for prompt {i}')
            inj_pos = inj_pos_t[0].item()

            v = item['act'].to('cuda').to(torch.bfloat16)
            v_scaled = v / v.norm().clamp_min(1e-12) * INJECTION_SCALE
            embeds[0, inj_pos] = v_scaled

            # Single forward pass — no generation needed for probe
            for L in probe_layers:
                captured_per_layer[L] = None
            _ = av(inputs_embeds=embeds, attention_mask=torch.ones_like(input_ids), use_cache=False)
            for L in probe_layers:
                h = captured_per_layer[L]
                # Take residual at injection_pos (the token that received the activation embedding)
                r = h[0, inj_pos].float().cpu().numpy()
                residuals[L].append(r)
            labels.append(item['category'])
            keys.append(item['key'])
            if (i+1) % 25 == 0 or (i+1) == len(target_acts):
                elapsed = time.time() - t0
                eta = elapsed / (i+1) * (len(target_acts) - i - 1)
                print(f'  [{i+1:3d}/{len(target_acts)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s')

    for h in handles:
        h.remove()
    print(f'  ✓ captured residuals for {len(target_acts)} prompts × {len(probe_layers)} layers in {time.time()-t0:.0f}s')

    del av, av_tok, inner, layers_attr
    gc.collect(); torch.cuda.empty_cache()
    print(f'  VRAM after free: {torch.cuda.memory_allocated()/1e9:.2f} GB')

    out = {L: np.stack(residuals[L]) for L in probe_layers}
    return out, labels, keys, {
        'INJECTION_CHAR': INJECTION_CHAR,
        'INJECTION_TOKEN_ID': INJECTION_TOKEN_ID,
        'INJECTION_SCALE': INJECTION_SCALE,
    }
"""))

cells.append(code("""# Cell 9: probe_categorical
CATEGORIES = ['chat', 'code', 'agent', 'reasoning']
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}

def probe_categorical(X: np.ndarray, labels_str: list, n_splits: int = 5):
    \"\"\"5-fold CV LogisticRegression macro AUROC + per-class one-vs-rest AUROC.\"\"\"
    y = np.array([CAT_TO_IDX[c] for c in labels_str])
    n_classes = len(CATEGORIES)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aurocs = []
    fold_per_class = {c: [] for c in CATEGORIES}
    fold_acc = []

    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        clf = LogisticRegression(
            max_iter=2000, C=1.0, solver='lbfgs',
            multi_class='multinomial', class_weight='balanced',
            random_state=42,
        )
        clf.fit(Xtr_s, ytr)
        proba = clf.predict_proba(Xte_s)
        # macro AUROC ovr
        auc = roc_auc_score(yte, proba, multi_class='ovr', average='macro', labels=list(range(n_classes)))
        fold_aurocs.append(auc)
        for ci, cn in enumerate(CATEGORIES):
            yte_bin = (yte == ci).astype(int)
            if yte_bin.sum() > 0 and yte_bin.sum() < len(yte_bin):
                fold_per_class[cn].append(roc_auc_score(yte_bin, proba[:, ci]))
        preds = proba.argmax(axis=1)
        fold_acc.append((preds == yte).mean())

    return {
        'macro_auroc_mean': float(np.mean(fold_aurocs)),
        'macro_auroc_std': float(np.std(fold_aurocs)),
        'per_class_auroc': {c: float(np.mean(fold_per_class[c])) for c in CATEGORIES if fold_per_class[c]},
        'accuracy_mean': float(np.mean(fold_acc)),
        'n_folds': n_splits,
    }
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Run Qwen
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 5. Run pilot for Qwen2.5-7B (smaller, run first)"""))

cells.append(code("""# Cell 10: download Qwen NLA pair
PAIR_KEY = 'qwen25_7b'
PCFG = NLA_PAIRS[PAIR_KEY]

print(f'Downloading {PCFG[\"av_repo\"]} ...')
qwen_av_path = snapshot_download(repo_id=PCFG['av_repo'], cache_dir='/content/hf_cache')
print(f'  → {qwen_av_path}')
"""))

cells.append(code("""# Cell 11: capture target acts
qwen_target_acts = capture_target_acts(PCFG['target_repo'], PCFG['extraction_layer'])
"""))

cells.append(code("""# Cell 12: capture AV residuals across probe_layers
qwen_residuals, qwen_labels, qwen_keys, qwen_inj_meta = capture_av_residuals(
    qwen_av_path, qwen_target_acts, PCFG['av_probe_layers'],
    PCFG['d_model_expected'], PCFG['add_special_tokens_av'],
)
print(f'qwen_residuals shapes:')
for L, X in qwen_residuals.items():
    print(f'  L{L}: {X.shape}')
"""))

cells.append(code("""# Cell 13: probe each layer
qwen_probe_results = {}
print(f'\\n=== Qwen2.5-7B AV residual category-decode ===')
print(f'{\"layer\":>6} {\"macro AUROC\":>14} {\"acc\":>8}  per-class AUROC')
for L in PCFG['av_probe_layers']:
    res = probe_categorical(qwen_residuals[L], qwen_labels)
    qwen_probe_results[L] = res
    pc = '  '.join(f'{c[:3]}={res[\"per_class_auroc\"][c]:.3f}' for c in CATEGORIES if c in res['per_class_auroc'])
    print(f'L{L:>4} {res[\"macro_auroc_mean\"]:>9.3f}±{res[\"macro_auroc_std\"]:.3f}  {res[\"accuracy_mean\"]:>6.3f}  {pc}')

best_layer_qwen = max(qwen_probe_results, key=lambda L: qwen_probe_results[L]['macro_auroc_mean'])
print(f'\\n→ Best Qwen AV layer: L{best_layer_qwen} '
      f'(AUROC {qwen_probe_results[best_layer_qwen][\"macro_auroc_mean\"]:.3f})')
"""))

cells.append(code("""# Cell 14: save Qwen results
qwen_dump = {
    'pair': PAIR_KEY,
    'config': {k: v for k, v in PCFG.items() if not callable(v)},
    'inj_meta': qwen_inj_meta,
    'probe_results': {f'L{L}': qwen_probe_results[L] for L in PCFG['av_probe_layers']},
    'best_layer': best_layer_qwen,
    'best_auroc': qwen_probe_results[best_layer_qwen]['macro_auroc_mean'],
    'n_prompts': len(qwen_target_acts),
    'categories': CATEGORIES,
}
with open(OUT / f'phase17_{PAIR_KEY}_results.json', 'w') as f:
    json.dump(qwen_dump, f, indent=2)
# Save residuals (one file per layer to keep size manageable)
for L in PCFG['av_probe_layers']:
    np.save(OUT / f'phase17_{PAIR_KEY}_residuals_L{L}.npy', qwen_residuals[L])
np.save(OUT / f'phase17_{PAIR_KEY}_labels.npy', np.array(qwen_labels))
print(f'✓ saved Qwen results + residuals to {OUT}')

# Free memory
del qwen_residuals, qwen_target_acts
gc.collect(); torch.cuda.empty_cache()
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Run Gemma
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 6. Run pilot for Gemma-3-12B (larger; kernel restart recommended if VRAM tight)

**Note**: Gemma-3-12B target + AV both load ~24 GB each in bf16. If VRAM
issues appear, restart kernel and rerun cells 1-9 + this section.
"""))

cells.append(code("""# Cell 15: download Gemma NLA pair
PAIR_KEY = 'gemma3_12b'
PCFG = NLA_PAIRS[PAIR_KEY]

print(f'Downloading {PCFG[\"av_repo\"]} ...')
gemma_av_path = snapshot_download(repo_id=PCFG['av_repo'], cache_dir='/content/hf_cache')
print(f'  → {gemma_av_path}')
"""))

cells.append(code("""# Cell 16: capture target acts (Gemma-3-12B-IT, gated)
gemma_target_acts = capture_target_acts(PCFG['target_repo'], PCFG['extraction_layer'])
"""))

cells.append(code("""# Cell 17: capture AV residuals across probe_layers
gemma_residuals, gemma_labels, gemma_keys, gemma_inj_meta = capture_av_residuals(
    gemma_av_path, gemma_target_acts, PCFG['av_probe_layers'],
    PCFG['d_model_expected'], PCFG['add_special_tokens_av'],
)
print(f'gemma_residuals shapes:')
for L, X in gemma_residuals.items():
    print(f'  L{L}: {X.shape}')
"""))

cells.append(code("""# Cell 18: probe each layer
gemma_probe_results = {}
print(f'\\n=== Gemma-3-12B AV residual category-decode ===')
print(f'{\"layer\":>6} {\"macro AUROC\":>14} {\"acc\":>8}  per-class AUROC')
for L in PCFG['av_probe_layers']:
    res = probe_categorical(gemma_residuals[L], gemma_labels)
    gemma_probe_results[L] = res
    pc = '  '.join(f'{c[:3]}={res[\"per_class_auroc\"][c]:.3f}' for c in CATEGORIES if c in res['per_class_auroc'])
    print(f'L{L:>4} {res[\"macro_auroc_mean\"]:>9.3f}±{res[\"macro_auroc_std\"]:.3f}  {res[\"accuracy_mean\"]:>6.3f}  {pc}')

best_layer_gemma = max(gemma_probe_results, key=lambda L: gemma_probe_results[L]['macro_auroc_mean'])
print(f'\\n→ Best Gemma AV layer: L{best_layer_gemma} '
      f'(AUROC {gemma_probe_results[best_layer_gemma][\"macro_auroc_mean\"]:.3f})')
"""))

cells.append(code("""# Cell 19: save Gemma results
gemma_dump = {
    'pair': PAIR_KEY,
    'config': {k: v for k, v in PCFG.items() if not callable(v)},
    'inj_meta': gemma_inj_meta,
    'probe_results': {f'L{L}': gemma_probe_results[L] for L in PCFG['av_probe_layers']},
    'best_layer': best_layer_gemma,
    'best_auroc': gemma_probe_results[best_layer_gemma]['macro_auroc_mean'],
    'n_prompts': len(gemma_target_acts),
    'categories': CATEGORIES,
}
with open(OUT / f'phase17_{PAIR_KEY}_results.json', 'w') as f:
    json.dump(gemma_dump, f, indent=2)
for L in PCFG['av_probe_layers']:
    np.save(OUT / f'phase17_{PAIR_KEY}_residuals_L{L}.npy', gemma_residuals[L])
np.save(OUT / f'phase17_{PAIR_KEY}_labels.npy', np.array(gemma_labels))
print(f'✓ saved Gemma results + residuals to {OUT}')

del gemma_residuals, gemma_target_acts
gc.collect(); torch.cuda.empty_cache()
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 7. Verdict — decision rule for V4 Phase A

Apply decision rule to best AUROC across both models:

| Best AUROC | V4 implication |
|---|---|
| > 0.9 | 🟢 SAE training justified — proceed Phase A with d_sae=16384 |
| 0.7-0.9 | 🟡 Marginal — SAE with d_sae=8192 (smaller expansion) |
| < 0.7 | 🔴 Reformulate — try other positions or escalate to nonlinear |

Cross-model agreement matters: if Qwen >0.9 but Gemma <0.7, the format-prior
locus is model-specific — V4 paper would have to address this asymmetry.
"""))

cells.append(code("""# Cell 20: aggregate verdict
import json as _json

# Reload (so this cell is independently runnable)
qwen_dump = _json.load(open(OUT / 'phase17_qwen25_7b_results.json'))
gemma_dump = _json.load(open(OUT / 'phase17_gemma3_12b_results.json'))

print('=' * 78)
print('PHASE 17 VERDICT — AV residual category-decode pilot')
print('=' * 78)

def color(auroc):
    if auroc > 0.90: return '🟢'
    if auroc > 0.70: return '🟡'
    return '🔴'

for model_name, dump in [('Qwen2.5-7B AV', qwen_dump), ('Gemma-3-12B AV', gemma_dump)]:
    print(f'\\n## {model_name}')
    print(f'{\"layer\":>8} {\"AUROC\":>10}  status')
    for L_str, res in dump['probe_results'].items():
        auc = res['macro_auroc_mean']
        print(f'{L_str:>8} {auc:>10.3f}  {color(auc)}')
    print(f'\\n→ Best: {dump[\"best_layer\"] if isinstance(dump[\"best_layer\"], str) else \"L\"+str(dump[\"best_layer\"])}, '
          f'AUROC {dump[\"best_auroc\"]:.3f} {color(dump[\"best_auroc\"])}')

# Cross-model decision
both_best = min(qwen_dump['best_auroc'], gemma_dump['best_auroc'])
print(f'\\n=== Cross-model conservative best: AUROC {both_best:.3f} ===')
if both_best > 0.90:
    verdict = '🟢 STRONG_SIGNAL'
    rec = 'Proceed Phase A: train SAE d_sae=16384 (4x expansion) on AV residual at best layer.'
elif both_best > 0.70:
    verdict = '🟡 MARGINAL_SIGNAL'
    rec = 'Phase A with smaller SAE: d_sae=8192 (2x expansion). Discuss in paper-7 V4 §6.1.'
else:
    verdict = '🔴 INSUFFICIENT_SIGNAL'
    rec = 'Reformulate before Phase A: (a) try residual at last input token instead of injection_pos; (b) try residual at first generation token; (c) escalate to attribution patching from output→residual.'

print(f'\\nVERDICT: {verdict}')
print(f'\\nRECOMMENDATION: {rec}')

# Save verdict file
verdict_doc = {
    'verdict': verdict,
    'recommendation': rec,
    'qwen': {
        'best_layer': qwen_dump['best_layer'],
        'best_auroc': qwen_dump['best_auroc'],
        'all_layers': {k: v['macro_auroc_mean'] for k, v in qwen_dump['probe_results'].items()},
    },
    'gemma': {
        'best_layer': gemma_dump['best_layer'],
        'best_auroc': gemma_dump['best_auroc'],
        'all_layers': {k: v['macro_auroc_mean'] for k, v in gemma_dump['probe_results'].items()},
    },
    'cross_model_conservative_auroc': both_best,
}
with open(OUT / 'phase17_verdict.json', 'w') as f:
    json.dump(verdict_doc, f, indent=2)
print(f'\\n✓ verdict saved to {OUT / \"phase17_verdict.json\"}')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 8. Sanity checks (random-feature baseline + content-confound check)

**Why run these?**

1. **Random-feature baseline**: with d_model=3584/3840 and N=200, the linear
   probe is overparameterized. The decision-relevant question is whether
   AUROC is significantly above what a random direction in residual-space
   would give. Random K-matched control is the standard methodology
   contribution from paper-6 (two-forms-epiphenomenal-probes).

2. **Content-confound**: the 4 categories have very different surface vocab
   (code keywords vs reasoning words). A naive bag-of-words on the prompt
   might already hit AUROC 0.95, in which case the AV residual signal could
   be reading prompt text instead of activation content. Comparison
   establishes whether AV adds anything.

If random-baseline AUROC is close to actual probe AUROC at the same K, the
probe is reading noise. If bag-of-words AUROC ≥ AV-residual AUROC, the AV's
contribution is upper-bounded by what the prompt text already gives.
"""))

cells.append(code("""# Cell 21: random-feature baseline
def random_baseline_auroc(N: int, d: int, labels: list, n_seeds: int = 10):
    aurocs = []
    rng = np.random.default_rng(0)
    for seed in range(n_seeds):
        rng2 = np.random.default_rng(seed)
        Xrand = rng2.standard_normal((N, d)).astype(np.float32)
        res = probe_categorical(Xrand, labels)
        aurocs.append(res['macro_auroc_mean'])
    return float(np.mean(aurocs)), float(np.std(aurocs))

print('Random Gaussian features, K-matched (same N, same d_model):')
for pair_key in ['qwen25_7b', 'gemma3_12b']:
    PCFG = NLA_PAIRS[pair_key]
    d = PCFG['d_model_expected']
    labels = list(np.load(OUT / f'phase17_{pair_key}_labels.npy'))
    mean_auc, std_auc = random_baseline_auroc(len(labels), d, labels)
    print(f'  {pair_key}: AUROC {mean_auc:.3f} ± {std_auc:.3f}  (chance 0.500)')
"""))

cells.append(code("""# Cell 22: bag-of-words baseline (prompt-text upper bound)
from sklearn.feature_extraction.text import TfidfVectorizer

print('Bag-of-words (TF-IDF) on prompt text:')
prompts_flat = []
labels_flat = []
for cat, prompts in PHASE_16_PROMPTS.items():
    for p in prompts:
        prompts_flat.append(p)
        labels_flat.append(cat)

# TF-IDF char n-grams
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
Xtfidf = vec.fit_transform(prompts_flat).toarray().astype(np.float32)
print(f'  TF-IDF dim: {Xtfidf.shape[1]}')
res_tfidf = probe_categorical(Xtfidf, labels_flat)
print(f'  TF-IDF AUROC: {res_tfidf[\"macro_auroc_mean\"]:.3f} ± {res_tfidf[\"macro_auroc_std\"]:.3f}  (this is what you can get from prompt text alone)')

print()
print('Interpretation:')
print(f'  Qwen AV  best AUROC: {qwen_dump[\"best_auroc\"]:.3f}')
print(f'  Gemma AV best AUROC: {gemma_dump[\"best_auroc\"]:.3f}')
print(f'  TF-IDF prompt    : {res_tfidf[\"macro_auroc_mean\"]:.3f}')
print(f'  If AV best > TF-IDF prompt: AV residual carries activation-derived category info beyond prompt text.')
print(f'  If AV best ≈ TF-IDF prompt: indistinguishable — V4 needs to decompose by entity/format axes (Tier 1 vs Tier 2).')
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""## 9. Next steps

If verdict is 🟢 (AUROC > 0.9):
- **Phase A** notebook: `nb_track_a_phase18_av_sae_train.ipynb` — TopK SAE
  d_sae=16384, k=64, AuxK on AV residual at best layer. Train on 5M tokens
  mix (NLA-native + FineWeb-Edu). ~$30, ~10 hr H100.
- **Phase B**: discovery — top-K categorical features per category, auto-interp
  via Claude Opus.
- **Phase C** (killer): random-Gaussian + categorical-feature steering test.
  Predict shift_rate ≥ 75% per category.

If verdict is 🟡 (AUROC 0.7-0.9):
- Phase A with d_sae=8192. V4 paper §6.1 acknowledges Tier 1 is partially
  distributed.

If verdict is 🔴 (AUROC < 0.7):
- Pilot extension `nb_track_a_phase17b_av_decode_position_sweep.ipynb`:
  test residual at last_input_token, first_generation_token, mid-generation.
  If still null, V4 pivots to attribution patching from output→residual.

---

**Artifacts saved**:
- `phase17_qwen25_7b_results.json` (per-layer AUROC + per-class)
- `phase17_gemma3_12b_results.json`
- `phase17_qwen25_7b_residuals_L*.npy` (raw residuals, ~6 MB/layer × 5 layers)
- `phase17_gemma3_12b_residuals_L*.npy` (~6 MB/layer × 5 layers)
- `phase17_*_labels.npy`
- `phase17_verdict.json`

Total Drive footprint: ~70 MB.
"""))


# ──────────────────────────────────────────────────────────────────────────────
# Write notebook
# ──────────────────────────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NOTEBOOK_PATH, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Wrote {NOTEBOOK_PATH} ({len(cells)} cells)")
