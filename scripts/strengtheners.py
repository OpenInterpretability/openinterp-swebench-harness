#!/usr/bin/env python3
"""Strengtheners #2 (valid-call re-parse) and #3 (h2_bash_baseline) for paper #7.

Reuses the cached deterministic decision points + residuals (`results/commit_lever_state.pt` on HF) — so this
is generation-only, fast, one short Colab session. Retires two EVAL caveats:
  #3 h2_bash_baseline: bash-emission rate at EDIT points with NO patch, so the under-brake bash rate (0.60)
     can be compared to baseline → states whether the brake *redirects* to exploration.
  #2 valid-call: re-run the 4 headline conditions with MAXNEW=48, report onset rate (prefix, the paper's
     metric) AND valid-call rate (full tool name 'str_replace_editor' present), retiring the intent-only caveat.
"""
import os, json, time
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download, upload_file

MODEL_ID = "Qwen/Qwen3.6-27B"; REPO = "caiovicentino1/swebench-phase6-verdict-circuit"
TOOL_NAMES = ["bash", "str_replace_editor", "finish"]
ACTION_TOK = {"finish": 28, "bash": 21402, "str_replace_editor": 15462}
GEN_LAYERS = [55, 59]; OUT = "/content/streng_out"; os.makedirs(OUT, exist_ok=True)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

log("loading", MODEL_ID)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
st = torch.load(hf_hub_download(REPO, "results/commit_lever_state.pt", repo_type="dataset", token=os.environ.get("HF_TOKEN")),
                map_location="cpu", weights_only=False)
SUBN = int(os.environ.get("SUBN", "30"))   # subsample to fit the VM-death window
EDIT, BASH, E_res, B_res = st["EDIT"][:SUBN], st["BASH"][:SUBN], st["E_res"], st["B_res"]
log(f"state: edit {len(EDIT)} bash {len(BASH)} (SUBN={SUBN})")

# resume from HF + checkpoint after every piece (the VM dies ~25-40min; keep-alive 403 bug)
res = {}
try:
    res = json.load(open(hf_hub_download(REPO, "results/strengtheners.json", repo_type="dataset",
                                         token=os.environ.get("HF_TOKEN"), force_download=True)))
    log("resumed:", list(res.keys()), list((res.get("validcall") or {}).keys()))
except Exception:
    log("no prior strengtheners.json — fresh")
def save():
    json.dump(res, open(os.path.join(OUT, "strengtheners.json"), "w"), indent=1)
    try:
        upload_file(path_or_fileobj=os.path.join(OUT, "strengtheners.json"),
                    path_in_repo="results/strengtheners.json", repo_id=REPO, repo_type="dataset",
                    token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        log("save upload skip:", str(e)[:60])

def layer(idx):
    for p in ("model.language_model.layers", "model.model.layers", "model.layers"):
        cur = model; ok = True
        for q in p.split("."):
            if not hasattr(cur, q): ok = False; break
            cur = getattr(cur, q)
        if ok:
            try: return cur[idx]
            except Exception: continue
    raise RuntimeError("layer")
def patch_hook(d):
    def h(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o; hs = hs.clone(); hs[:, -1, :] = d.to(hs.dtype).to(hs.device)
        return (hs, *o[1:]) if isinstance(o, tuple) else hs
    return h
def gen(ids, L=None, donor=None, maxnew=48):
    hh = None
    if L is not None and donor is not None:
        def ph(m, i, o):
            hs = o[0] if isinstance(o, tuple) else o
            if hs.shape[1] > 1:
                hs = hs.clone(); hs[:, -1, :] = donor.to(hs.dtype).to(hs.device)
                return (hs, *o[1:]) if isinstance(o, tuple) else hs
            return o
        hh = layer(L).register_forward_hook(ph)
    try:
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(model.device), max_new_tokens=maxnew, do_sample=False,
                                 use_cache=True, pad_token_id=tok.eos_token_id, attention_mask=torch.ones_like(ids).to(model.device))
    finally:
        if hh is not None: hh.remove()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
def onset(cont, name):
    s = cont.lstrip()
    if s.startswith("="): s = s[1:]
    return s.lstrip().lower().startswith(name.lower()[:6]) or (name.lower()[:6] in cont[:24].lower())
def valid(cont, name): return name.lower() in cont.lower()  # full tool name present
def edit_donor(brow, i, L):
    same = [j for j, e in enumerate(EDIT) if e["repo"] == brow["repo"]]; j = same[0] if same else i % len(EDIT)
    return E_res[L][j]
def bash_donor(erow, i, L):
    same = [j for j, b in enumerate(BASH) if b["repo"] == erow["repo"]]; j = same[0] if same else i % len(BASH)
    return B_res[L][j]

# ---- #3 h2_bash_baseline: bash-emission at EDIT points, NO patch
if "h2_bash_baseline_rate" not in res:
    hits = 0
    for r in EDIT:
        hits += int(onset(gen(r["ids"], maxnew=16), "bash")); torch.cuda.empty_cache()
    res["h2_bash_baseline_rate"] = hits / len(EDIT); save()
    log(f"#3 h2_bash_baseline = {res['h2_bash_baseline_rate']:.2f}  (under-brake bash was 0.60) [saved]")
else:
    log(f"#3 skip (done {res['h2_bash_baseline_rate']:.2f})")

# ---- #2 valid-call re-parse on the 4 headline conditions (checkpoint each)
def rate(rows, name, L=None, donor_fn=None, maxnew=40):
    o = v = 0
    for i, r in enumerate(rows):
        d = None if donor_fn is None else donor_fn(r, i, L)
        c = gen(r["ids"], L=(None if d is None else L), donor=d, maxnew=maxnew)
        o += int(onset(c, name)); v += int(valid(c, name)); torch.cuda.empty_cache()
    return o / len(rows), v / len(rows)
conds = [
    ("h1_baseline", BASH, "str_replace_editor", None, None),
    ("h1_editdonor_L59", BASH, "str_replace_editor", 59, edit_donor),
    ("h2_baseline", EDIT, "str_replace_editor", None, None),
    ("h2_suppress_L55", EDIT, "str_replace_editor", 55, bash_donor),
]
res.setdefault("validcall", {})
for nm, rows, name, L, fn in conds:
    if nm in res["validcall"]:
        log(f"#2 {nm} skip (done)"); continue
    o, v = rate(rows, name, L, fn)
    res["validcall"][nm] = {"onset": o, "valid": v}; save()
    log(f"#2 {nm}: onset {o:.2f} | valid-call {v:.2f} [saved]")

print("OILAB_JSON_BEGIN"); print(json.dumps(res)); print("OILAB_JSON_END", flush=True)
log("STRENGTHENERS_DONE")
