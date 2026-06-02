#!/usr/bin/env python3
"""Builds notebooks/nb_verdict_circuit_stage1.ipynb — Verdict-Circuit Stage-1 (model-in-the-loop).

Pre-registered in paper/verdict_circuit/PREREG_stage1.md. Runs on Colab/RTX with Qwen3.6-27B.
Job#1 (interpretation gate H1) is fully self-contained (model+SAE+public corpus). Jobs H2-H4
need the SWE-bench trajectory transcripts (one harness-specific adapter, clearly marked).

Cell-source literals use ''' delimiters because cell code contains \"\"\" docstrings.
"""
import json
from pathlib import Path

def md(src):   return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}

cells = []

cells.append(md(r'''# Verdict Circuit — Stage-1 (interpret the feature, test the readout)

**Pre-registered** in `paper/verdict_circuit/PREREG_stage1.md`. Gated by Stage-0's PRELIMINARY GO
(a SUCCESS-selected SAE feature — L23 #22358 — fires at WANDERING-final but not LOCKED-final).

**The whole point of Stage-1 is to try to FALSIFY the Stage-0 caveat:** is feature #22358 really a
*"task-done" verdict*, or just a generic *"active-at-the-end"* feature? Run **H1 first** — it gates everything.

| Step | Question | Gate |
|---|---|---|
| **H1** interpretation | Are #22358's top-activating contexts *completion/conclusion* language? | ≥12/20 completion-related → continue; else **STOP** (honest negative) |
| **H2** readout sanity | Is P(finish) high at SUCCESS-final, low at WANDERING-final? | sanity |
| **H3** DLA | Does W_dec[#22358] vote "finish" in the logits? | direction |
| **H4** causal | Clamp #22358 to SUCCESS-level at WANDERING-final → ΔP(finish)? | ≈0 → §13 confirmed (verdict present, readout broken); ≫0 → surgical lever (extraordinary, high bar) |

Honesty rules carried in: paired tests (WANDERING not run-stable at temp 1.0), random-feature clamp null,
replicate at L43 #908, behavioral confirmation before any "lever" claim. Reported whichever way it falls.'''))

cells.append(md("## 1. Install (Qwen3.6-27B needs `transformers` from main — `model_type=qwen3_5`)"))
cells.append(code(r'''
!pip -q install "git+https://github.com/huggingface/transformers.git" safetensors huggingface_hub datasets scipy accelerate
import transformers, subprocess
# Log the commit hash — pin it for any behavioral A/B (see feedback_transformers_main_version_pin).
sha = subprocess.run(["pip","show","transformers"], capture_output=True, text=True).stdout
print("transformers", transformers.__version__)
print([l for l in sha.splitlines() if l.lower().startswith("version")])
'''))

cells.append(md("## 2. Config — set HF token + Drive paths"))
cells.append(code(r'''
import os, torch, numpy as np
try:
    from google.colab import drive; drive.mount('/content/drive'); COLAB=True
except Exception:
    COLAB=False
os.environ.setdefault("HF_TOKEN", "")          # <-- paste an HF read token (or huggingface-cli login)
MODEL_ID = "Qwen/Qwen3.6-27B"                   # from config.py; confirm exact HF id at runtime
LAYER    = 23                                    # SAE ∩ captures = {L23 (strong), L43}
FEATURE  = 22358                                 # L23 primary done-feature from Stage-0
SAE_REPO = "caiovicentino1/qwen36-27b-sae-fullstack"
DRIVE    = "/content/drive/MyDrive/openinterp_runs/swebench_v6_phase6"   # captures + transcripts live here
OUT      = "/content/drive/MyDrive/openinterp_runs/verdict_circuit_stage1"; os.makedirs(OUT, exist_ok=True)
TOOL_NAMES = ["bash", "str_replace_editor", "finish"]   # agent/tools.py TOOL_NAMES
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| layer:", LAYER, "| feature:", FEATURE)
'''))

cells.append(md("## 3. Load model + tokenizer + SAE; helpers (robust layer finder for `model.language_model.layers`)"))
cells.append(code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors import safe_open

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                             device_map="auto", trust_remote_code=True).eval()

def get_layer_module(m, idx):
    # SAE hook_name = 'model.language_model.layers.{L}' -> include that path FIRST.
    for path in ("model.language_model.layers", "language_model.layers",
                 "model.model.layers", "model.layers", "transformer.h"):
        cur = m; ok = True
        for part in path.split("."):
            if not hasattr(cur, part): ok=False; break
            cur = getattr(cur, part)
        if ok:
            try: return cur[idx]
            except Exception: continue
    raise RuntimeError("could not locate layer module")

LAYER_MOD = get_layer_module(model, LAYER)
print("layer module:", type(LAYER_MOD).__name__)

# --- SAE (TopK k=128) on GPU ---
def load_sae(layer):
    p = hf_hub_download(SAE_REPO, f"sae_L{layer}_latest.safetensors")
    s = {}
    with safe_open(p, "pt") as f:
        for k in f.keys(): s[k] = f.get_tensor(k).to(DEVICE, torch.float32)
    return s
SAE = load_sae(LAYER)
K_TOPK = 128

def sae_encode(x):  # x: [..., 5120] -> z: [..., 40960] TopK-sparse
    z = (x.float() - SAE["b_dec"]) @ SAE["W_enc"] + SAE["b_enc"]
    z = torch.relu(z)
    if z.shape[-1] > K_TOPK:
        thresh = z.topk(K_TOPK, dim=-1).values[..., -1:].clamp_min(1e-9)
        z = torch.where(z >= thresh, z, torch.zeros_like(z))
    return z

# --- full-sequence residual capture hook at LAYER ---
_CAP = {}
def _cap_hook(m, i, o):
    _CAP["hs"] = (o[0] if isinstance(o, tuple) else o).detach()
def forward_capture(input_ids):
    h = LAYER_MOD.register_forward_hook(_cap_hook)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids.to(model.device))
    finally:
        h.remove()
    return out.logits, _CAP["hs"]   # logits [1,seq,V], hs [1,seq,5120] at LAYER
print("ready.")
'''))

cells.append(md('''## 4. Action tokens — identify the `finish` / `bash` / `str_replace_editor` choice token
At `<tool_call>{"name": "` the model picks a tool. We read P(finish) as the next-token mass on the
first token of each tool name in that context.'''))
cells.append(code(r'''
PREFIX = '<tool_call>{"name": "'    # the action-selection point (agent/parser.py format)
def first_tok_after(prefix, name):
    a = tok(prefix, add_special_tokens=False).input_ids
    b = tok(prefix + name, add_special_tokens=False).input_ids
    return b[len(a)]                 # the token that begins `name` in this context
ACTION_TOK = {n: first_tok_after(PREFIX, n) for n in TOOL_NAMES}
print("action first-tokens:", {n: (t, repr(tok.decode([t]))) for n,t in ACTION_TOK.items()})

def p_finish_from_logits(next_logits):
    # next_logits: [V] logits for the token right after PREFIX. Softmax over the 3 tools.
    ids = [ACTION_TOK[n] for n in TOOL_NAMES]
    p = torch.softmax(next_logits[ids].float(), dim=-1)
    return {n: float(p[i]) for i, n in enumerate(TOOL_NAMES)}
'''))

cells.append(md('''## 5. **H1 — INTERPRETATION (the gate).** Top-activating contexts of feature #22358.
Self-contained: needs only model + SAE + a public corpus. (5b additionally runs on trajectory
transcripts if `DRIVE` has them.)'''))
cells.append(code(r'''
from datasets import load_dataset
import heapq

def top_activations(texts, feature=FEATURE, max_tokens=64, keep=20, ctx=12):
    # Return top-`keep` (activation, decoded-window, peak-token) for `feature`.
    heap = []; store = []
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids
        if ids.shape[1] < 3: continue
        _, hs = forward_capture(ids)               # [1,seq,5120]
        z = sae_encode(hs[0])                       # [seq,40960]
        acts = z[:, feature]                        # [seq]
        j = int(acts.argmax()); a = float(acts[j])
        if a <= 0: continue
        toks = ids[0].tolist()
        lo, hi = max(0, j-ctx), min(len(toks), j+ctx+1)
        window = "".join(("[[" + tok.decode([toks[k]]) + "]]") if k==j else tok.decode([toks[k]])
                          for k in range(lo, hi))
        store.append((a, window, repr(tok.decode([toks[j]]))))
        if len(heap) < keep: heapq.heappush(heap, (a, len(store)-1))
        elif a > heap[0][0]: heapq.heapreplace(heap, (a, len(store)-1))
    return sorted((store[i] for _, i in heap), key=lambda x: -x[0])

# general corpus (interp standard)
corpus = load_dataset("NeelNanda/pile-10k", split="train").select(range(400))["text"]
top_general = top_activations(corpus)
print("=== TOP-ACTIVATING CONTEXTS (general corpus) — feature", FEATURE, "===")
for a, w, pk in top_general:
    print(f"[{a:6.2f}] peak={pk:>14} … {w[:160]!r}")
'''))

cells.append(code(r'''
# 5b (in-domain): top-activating contexts INSIDE trajectory transcripts, if available.
# Adapter: load whatever text you have per instance. Looks for *.json under DRIVE.
import glob, json as _json
def load_trajectory_texts(drive=DRIVE, limit=60):
    texts = []
    for p in sorted(glob.glob(os.path.join(drive, "**", "*.json"), recursive=True)):
        try: obj = _json.load(open(p))
        except Exception: continue
        msgs = obj.get("messages") or obj.get("trajectory") or obj
        if isinstance(msgs, list):
            t = "\n".join(str(m.get("content","")) for m in msgs if isinstance(m, dict))
            if len(t) > 200: texts.append(t[:8000])
        if len(texts) >= limit: break
    return texts
traj_texts = load_trajectory_texts()
print(f"loaded {len(traj_texts)} trajectory texts from {DRIVE}")
if traj_texts:
    top_traj = top_activations(traj_texts, max_tokens=128)
    print("=== TOP-ACTIVATING CONTEXTS (trajectories) — feature", FEATURE, "===")
    for a, w, pk in top_traj:
        print(f"[{a:6.2f}] peak={pk:>14} … {w[:160]!r}")
else:
    top_traj = []
    print("No transcripts found — H1 still decided by the general corpus above. "
          "Point DRIVE at the Phase-6 run dir to add the in-domain view.")
'''))

cells.append(md(r'''### H1 VERDICT (fill in before proceeding)
Look at the top-20 contexts above. **PASS iff ≥12/20 are completion/conclusion-related**
(e.g. "done", "complete", "finished", summaries, closings, the `finish` tool, end-of-patch),
**not** generic code-editing / arbitrary late tokens.

- **PASS** → feature is a *task-done* feature → run H2–H4.
- **FAIL** → it is "active-at-the-end", not "done" → **STOP and write the honest negative**
  ("WANDERING looks active-late, not done; §13's 'verdict consolidated' is not a single-feature fact").

Optional automated judge below (still confirm by eye).'''))
cells.append(code(r'''
# OPTIONAL auto-judge (heuristic keyword vote — replace with an LLM judge if you prefer).
DONE_KW = ["done","complete","finish","finished","resolved","fixed","submit","conclusion",
           "summary","that's all","</patch>","successfully","verified"]
def judge(top):
    hits = sum(any(k in (w.lower()+pk.lower()) for k in DONE_KW) for _,w,pk in top)
    return hits, len(top)
h,n = judge(top_general)
print(f"general corpus: {h}/{n} completion-related (heuristic; PASS if >= {int(0.6*n)})")
if top_traj:
    ht,nt = judge(top_traj); print(f"trajectories:   {ht}/{nt} completion-related")
print("\n>>> CONFIRM BY EYE. Set H1_PASS below and only then run H2-H4.")
H1_PASS = None   # <-- set True / False after inspecting the contexts
'''))

cells.append(md(r'''## 6. **H2 / H3 — readout** (run only if `H1_PASS`).
Needs `pre_tool` prompts per class: the trajectory text truncated at the final-turn
`<tool_call>{"name": "` choice point. **This is the one harness-specific adapter — confirm it
renders the same way your Phase-6 run did** (chat template + tools), so the L23 residual matches.'''))
cells.append(code(r'''
assert H1_PASS, "Set H1_PASS=True only if the contexts are completion-related (else STOP)."

# ADAPTER (confirm against your transcript schema): build (iid, sub_class, prompt_ending_in_PREFIX)
# from the FINAL turn of each trajectory. Default best-effort — adapt to your saved messages.
def build_pretool_prompts(drive=DRIVE, per_class=12):
    import glob, json as _json, csv
    lab = {}
    lp = os.path.join(drive, "..", "inflection_turn_out", "features_n99.csv")
    if os.path.exists(lp):
        for r in csv.DictReader(open(lp)): lab[r["iid"]] = r["sub_class"]
    prompts = {"success": [], "wandering": [], "locked": []}
    for p in sorted(glob.glob(os.path.join(drive, "**", "*.json"), recursive=True)):
        iid = Path(p).stem
        cls = lab.get(iid) or next((lab[k] for k in lab if iid.startswith(k) or k.startswith(iid)), None)
        if cls not in prompts or len(prompts[cls]) >= per_class: continue
        try: obj = _json.load(open(p))
        except Exception: continue
        msgs = obj.get("messages") or obj.get("trajectory")
        if not isinstance(msgs, list): continue
        try:
            convo = msgs[:-1] if msgs and msgs[-1].get("role")=="assistant" else msgs
            text = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            prompts[cls].append((iid, text + PREFIX))
        except Exception:
            continue
    return prompts

from pathlib import Path
PROMPTS = build_pretool_prompts()
print({k: len(v) for k,v in PROMPTS.items()})

def p_finish_for(prompt):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids
    logits, hs = forward_capture(ids)
    pf = p_finish_from_logits(logits[0, -1])
    zf = float(sae_encode(hs[0, -1:])[0, FEATURE])    # feature activation at the choice point
    return pf, zf

rows = {c: [p_finish_for(pr) for _, pr in PROMPTS[c]] for c in PROMPTS}
for c in PROMPTS:
    if rows[c]:
        pf = np.mean([r[0]["finish"] for r in rows[c]]); zf = np.mean([r[1] for r in rows[c]])
        print(f"{c:10s} P(finish)={pf:.3f}  feature#{FEATURE}_act={zf:.3f}  (n={len(rows[c])})")
'''))
cells.append(code(r'''
# H3 — Direct Logit Attribution of W_dec[FEATURE] -> tool logits (direct L23->logits path).
with torch.no_grad():
    d = SAE["W_dec"][FEATURE].to(model.device, torch.float32)        # [5120] L23 residual direction
    norm_w = None
    for nm in ["model.language_model.norm","model.model.norm","model.norm"]:
        cur=model; ok=True
        for part in nm.split("."):
            if not hasattr(cur,part): ok=False;break
            cur=getattr(cur,part)
        if ok and hasattr(cur,"weight"): norm_w=cur.weight.float(); break
    dn = d * (norm_w if norm_w is not None else 1.0)
    WU = model.get_output_embeddings().weight.float()               # [V,5120]
    contrib = dn @ WU.T                                             # [V]
    print("DLA of feature direction -> tool logits:",
          {n: round(float(contrib[ACTION_TOK[n]]),3) for n in TOOL_NAMES})
'''))

cells.append(md(r'''## 7. **H4 — the causal §13 test** (run only if H1_PASS).
Clamp feature #22358 to the SUCCESS-final level at the WANDERING-final choice point and read ΔP(finish).
Controls: a random *active* feature clamp of equal magnitude (null); paired per-trajectory.'''))
cells.append(code(r'''
def make_clamp_hook(f, target, last_only=True):
    wdec = SAE["W_dec"][f]
    def hook(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o
        x = hs[:, -1:, :] if last_only else hs
        cur = sae_encode(x.float())[..., f:f+1]                     # current activation
        delta = (target - cur) * wdec                              # move to target along decoder dir
        x.add_(delta.to(hs.dtype))
        return (hs, *o[1:]) if isinstance(o, tuple) else hs
    return hook

def p_finish_clamped(prompt, f, target):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(model.device)
    h = LAYER_MOD.register_forward_hook(make_clamp_hook(f, target))
    try:
        with torch.no_grad(): logits = model(input_ids=ids).logits
    finally: h.remove()
    return p_finish_from_logits(logits[0, -1])["finish"]

succ_target = float(np.mean([r[1] for r in rows["success"]])) if rows["success"] else 2.0
print("clamp target (SUCCESS-final feature level):", round(succ_target,3))

# random ACTIVE feature for the null (fires in a SUCCESS-final prompt)
import random
_, hs0 = forward_capture(tok(PROMPTS["success"][0][1], return_tensors="pt",
                             truncation=True, max_length=4096).input_ids)
active = (sae_encode(hs0[0, -1:])[0] > 0).nonzero().flatten().tolist()
rand_feat = random.choice([a for a in active if a != FEATURE])

res = {"clamp_feature": [], "clamp_random": [], "ablate_success": []}
for iid, pr in PROMPTS["wandering"]:
    base = p_finish_for(pr)[0]["finish"]
    res["clamp_feature"].append(p_finish_clamped(pr, FEATURE, succ_target) - base)
    res["clamp_random"].append(p_finish_clamped(pr, rand_feat, succ_target) - base)
for iid, pr in PROMPTS["success"]:
    base = p_finish_for(pr)[0]["finish"]
    res["ablate_success"].append(p_finish_clamped(pr, FEATURE, 0.0) - base)

from scipy.stats import wilcoxon
def summ(name, arr):
    arr=np.array(arr)
    try: p=float(wilcoxon(arr).pvalue)
    except Exception: p=None
    print(f"{name:16s} mean dP(finish)={arr.mean():+.3f}  median={np.median(arr):+.3f}  p={p}  n={len(arr)}")
print("=== H4 CAUSAL ===")
summ("clamp #22358", res["clamp_feature"]); summ("clamp random", res["clamp_random"]); summ("ablate(succ)", res["ablate_success"])
print("READOUT (pre-registered):")
print("  clamp ~0 AND > clamp-random ~0 -> §13 CONFIRMED: verdict present, readout broken (detection!=control).")
print("  clamp >> 0 AND > clamp-random  -> surgical lever (EXTRAORDINARY): need L43 replication + behavioral confirm.")
print("  ablate(succ) < 0               -> necessity of the verdict for finishing in SUCCESS.")
'''))

cells.append(md("## 8. Save results to Drive"))
cells.append(code(r'''
import json
summary = {"layer": LAYER, "feature": FEATURE,
           "H1_top_general": [(a, w, pk) for a,w,pk in top_general],
           "H1_PASS": H1_PASS,
           "H2_P_finish": {c: float(np.mean([r[0]["finish"] for r in rows[c]])) for c in rows if rows[c]},
           "H2_feature_act": {c: float(np.mean([r[1] for r in rows[c]])) for c in rows if rows[c]},
           "H4": {k: list(map(float, v)) for k,v in res.items()}}
open(os.path.join(OUT, "stage1_results.json"), "w").write(json.dumps(summary, indent=2))
print("saved ->", os.path.join(OUT, "stage1_results.json"))
'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parent.parent / "notebooks" / "nb_verdict_circuit_stage1.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")
