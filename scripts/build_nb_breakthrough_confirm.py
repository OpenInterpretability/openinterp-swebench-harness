#!/usr/bin/env python3
"""Builds notebooks/nb_breakthrough_confirm.ipynb — the HIGH-BAR confirmation of the action-channel positive.

The breakthrough run found a PRELIMINARY positive: injecting the SUCCESS late-block state (L51-L59) into
WANDERING raises P(finish) donor-specifically (+0.06..+0.15), while mid-layer/verdict interventions are inert.
PREREG_action_channel.md §2 holds a positive to an EXTRAORDINARY bar — it must be confirmed by:
  (1) a real GENERATION that actually emits the `finish` tool call (not a one-token probability bump),
  (2) held-out replication (donor from a disjoint SUCCESS split),
  (3) a per-pair / per-trajectory donor (not the coarse SUCCESS-mean) to rule out the averaged-vector confound.

This notebook reuses the verified H4/breakthrough reconstruction infra (faithful decision-point prompts from
the HF data bundle, no Drive). Core test = GENERATION emission-rate of `finish` under late-layer patching.

Cell-source literals use ''' (cells contain \"\"\").
"""
import json
from pathlib import Path

def md(s):   return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)}
C=[]

C.append(md(r'''# Breakthrough CONFIRMATION — does the late-block lever survive a real generation?

The breakthrough run found a **preliminary positive**: injecting the SUCCESS late-block residual (L51–L59) into
WANDERING raises P(finish) donor-specifically, while every mid-layer/verdict intervention is inert. That was a
**one-token probability bump**. `PREREG_action_channel.md` §2 holds a positive to an extraordinary bar. This
notebook runs the three confirmation gates:

1. **GENERATION** — patch the SUCCESS late-block state during autoregressive decoding at the WANDERING
   decision point, and check the model **actually emits `<function=finish`** (not just a ΔP bump).
   Gate: success-donor finish-rate ≫ no-patch finish-rate AND ≫ locked-donor finish-rate.
2. **HELD-OUT** — recompute the donor from a **disjoint SUCCESS split (B)** and confirm the effect persists.
3. **PER-PAIR donor** — replace the coarse SUCCESS-mean with a **single per-trajectory SUCCESS donor**
   (repo/task-matched when possible) to rule out the averaged-vector confound.

Reuses the verified reconstruction infra; data from the HF bundle (no Drive).'''))

C.append(md("## 1. Install + clone harness"))
C.append(code(r'''
!pip -q install "git+https://github.com/huggingface/transformers.git" safetensors huggingface_hub datasets scipy accelerate
import os, sys, subprocess, math
REPO_DIR="/content/openinterp-swebench-harness"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git","clone","-q","https://github.com/OpenInterpretability/openinterp-swebench-harness",REPO_DIR], check=True)
sys.path.insert(0, REPO_DIR)
import transformers; print("transformers", transformers.__version__)
'''))

C.append(md("## 2. Config (focus on the LATE block — where the lever appeared)"))
C.append(code(r'''
import torch, numpy as np
os.environ.setdefault("HF_TOKEN","")          # HF read token (model + SAE)
MODEL_ID="Qwen/Qwen3.6-27B"
DATA_REPO="caiovicentino1/swebench-phase6-verdict-circuit"
OUT="/content/confirm_out"; os.makedirs(OUT, exist_ok=True)
TOOL_NAMES=["bash","str_replace_editor","finish"]
PREFIX="<function"
ACTION_TOK={"finish":28,"bash":21402,"str_replace_editor":15462}   # verified token ids (=/=b/=str)
MAXLEN=4000
N_W=20            # use ALL available WANDERING (held-out beyond the breakthrough's n=12)
N_S=20            # SUCCESS pool (split A=donor / B=held-out donor)
DENSE=[48,50,52,54,56,58,60,62]      # dense late one-token sweep (confirm shape)
GEN_LAYERS=[55,59]                    # layers to run the GENERATION test at
MAXNEW=16        # generated tokens at the decision point
DEVICE="cuda" if torch.cuda.is_available() else "cpu"; print("device",DEVICE)
'''))

C.append(md("## 3. Load model + hooks + patched-generation helper"))
C.append(code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
                                           trust_remote_code=True).eval()
def get_layer_module(idx):
    for path in ("model.language_model.layers","language_model.layers","model.model.layers","model.layers"):
        cur=model; ok=True
        for p in path.split("."):
            if not hasattr(cur,p): ok=False; break
            cur=getattr(cur,p)
        if ok:
            try: return cur[idx]
            except Exception: continue
    raise RuntimeError("layer not found")
def p_finish(nl):
    ids=[ACTION_TOK[n] for n in TOOL_NAMES]; p=torch.softmax(nl[ids].float(),-1)
    return float(p[TOOL_NAMES.index("finish")])
def fwd_layers(ids, layers):
    caps={}; handles=[]
    def mk(L):
        def h(m,i,o): caps[L]=(o[0] if isinstance(o,tuple) else o)[:,-1:,:].detach()
        return h
    for L in layers: handles.append(get_layer_module(L).register_forward_hook(mk(L)))
    try:
        with torch.no_grad(): out=model(input_ids=ids.to(model.device), use_cache=False, logits_to_keep=1)
    finally:
        for h in handles: h.remove()
    return out.logits, caps
def _patch_hook(donor):
    def h(m,i,o):
        hs=o[0] if isinstance(o,tuple) else o; hs=hs.clone(); hs[:,-1,:]=donor.to(hs.dtype)
        return (hs,*o[1:]) if isinstance(o,tuple) else hs
    return h
def patch_one_token(ids,L,donor):   # one-token ΔP(finish) under a layer-L last-pos patch
    hh=get_layer_module(L).register_forward_hook(_patch_hook(donor))
    try:
        with torch.no_grad(): lg=model(input_ids=ids.to(model.device),use_cache=False,logits_to_keep=1).logits
    finally: hh.remove()
    r=p_finish(lg[0,-1]); del lg; return r
def gen_after_function(ids, L=None, donor=None, maxnew=MAXNEW):
    """Greedy-generate from the choice point (prompt ends at '<function'). If L/donor given, the layer-L
    last-position residual is replaced by `donor` at EVERY decode step. Returns the continuation string."""
    hh=get_layer_module(L).register_forward_hook(_patch_hook(donor)) if (L is not None and donor is not None) else None
    try:
        with torch.no_grad():
            out=model.generate(input_ids=ids.to(model.device), max_new_tokens=maxnew, do_sample=False,
                               use_cache=True, pad_token_id=tok.eos_token_id)
    finally:
        if hh is not None: hh.remove()
    cont=tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    del out; return cont
def emits_finish(cont):
    s=cont.lstrip()
    if s.startswith("="): s=s[1:]                 # '<function' + '=' + name
    return s.lstrip().lower().startswith("finish") or ("finish" in cont[:24].lower())
print("ready. action tokens:", {n:(t,repr(tok.decode([t]))) for n,t in ACTION_TOK.items()})
'''))

C.append(md("## 4. Harness assembly + HF data bundle (no Drive)"))
C.append(code(r'''
from agent.prompts import SYSTEM_PROMPT, render_problem
from agent.tools import TOOLS
from agent.parser import _strip_think
from datasets import load_dataset
from pathlib import Path
import csv, glob, json as J, tarfile
os.makedirs("/content/vc_data",exist_ok=True)
from huggingface_hub import hf_hub_download
with tarfile.open(hf_hub_download(DATA_REPO,"traces.tar.gz",repo_type="dataset")) as t: t.extractall("/content/vc_data")
TRACES={Path(p).stem:p for p in glob.glob("/content/vc_data/traces/instance_*.json")}
LAB={r["iid"]:r["sub_class"] for r in csv.DictReader(open(hf_hub_download(DATA_REPO,"features_n99.csv",repo_type="dataset")))}
ds=load_dataset("ScaleAI/SWE-bench_Pro", split="test")
by_iid={r["instance_id"]:r for r in ds if r.get("instance_id")}
def instance_for(iid):
    if iid in by_iid: return by_iid[iid]
    return next((r for k,r in by_iid.items() if k in iid or iid in k), None)
def repo_of(iid):  # task-similarity proxy for per-pair donor matching
    return iid.split("__")[0] if "__" in iid else iid.split("-")[0]
print("traces",len(TRACES),"| labels",len(LAB),"| dataset",len(ds))
'''))

C.append(md("## 5. Reconstruct decision-point prompts (faithful) + behavioral-fidelity gate"))
C.append(code(r'''
import gc
def build_messages_upto_final(trace, instance):
    msgs=[{"role":"system","content":SYSTEM_PROMPT}]
    msgs.append({"role":"user","content":render_problem({**instance,"__workdir__":f"/content/work_p6/{trace['instance_id']}"})})
    turns=trace["turns"]
    for tn in turns[:-1]:
        _,body=_strip_think(tn.get("raw_response") or "")
        msgs.append({"role":"assistant","content":body})
        for tr_ in (tn.get("tool_results") or []):
            msgs.append({"role":"tool","content":J.dumps(tr_.get("result"),ensure_ascii=False)[:32000]})
        if not (tn.get("tool_calls")):
            msgs.append({"role":"user","content":"You did not call a tool. Use bash, str_replace_editor, or finish."})
    return msgs, turns[-1]
def choice_point_ids(trace, instance):
    msgs, final = build_messages_upto_final(trace, instance)
    prefix = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False)
    raw = final.get("raw_response") or ""
    cut = raw.find(PREFIX)
    if cut < 0: return None
    ids = tok(prefix + raw[:cut+len(PREFIX)], add_special_tokens=False).input_ids
    if len(ids) > MAXLEN: ids = ids[-MAXLEN:]
    return torch.tensor([ids])
def collect(cls, n):
    out=[]
    for iid,tp in TRACES.items():
        lab=LAB.get(iid) if iid in LAB else next((LAB[k] for k in LAB if iid.startswith(k) or k.startswith(iid)),None)
        if lab!=cls: continue
        tr=J.load(open(tp)); inst=instance_for(iid)
        if inst is None: continue
        ids=choice_point_ids(tr,inst)
        if ids is None: continue
        lg,_=fwd_layers(ids,[])
        out.append({"iid":iid,"repo":repo_of(iid),"ids":ids.cpu(),"baseP":p_finish(lg[0,-1])})
        gc.collect(); torch.cuda.empty_cache()
        if len(out)>=n: break
    return out
PR={"success":collect("success",N_S),"wandering":collect("wandering",N_W),"locked":collect("locked",N_S)}
for c in PR: print(f"{c:10} n={len(PR[c])} baseP(finish)={np.mean([r['baseP'] for r in PR[c]]) if PR[c] else float('nan'):.3f}")
print(">>> FIDELITY GATE: proceed only if SUCCESS baseP(finish) >> WANDERING.")
'''))

C.append(md("## 6. Donors — SUCCESS split A/B (held-out), LOCKED null, per-trajectory residuals"))
C.append(code(r'''
LAYERS_NEEDED=sorted(set(DENSE+GEN_LAYERS))
def per_traj_resids(rows):    # {layer: [vec per row]} at the last position
    acc={L:[] for L in LAYERS_NEEDED}; iids=[]
    for r in rows:
        _,caps=fwd_layers(r["ids"],LAYERS_NEEDED)
        for L in LAYERS_NEEDED: acc[L].append(caps[L][0,0].float().cpu())
        iids.append(r); torch.cuda.empty_cache()
    return acc, iids
S_res, S_rows = per_traj_resids(PR["success"])
L_res, _      = per_traj_resids(PR["locked"])
# split A (donor) / B (held-out donor)
nA=len(S_rows)//2
def mean_over(idx,res): return {L:torch.stack([res[L][i] for i in idx]).mean(0) for L in LAYERS_NEEDED}
succ_A=mean_over(range(0,nA),S_res); succ_B=mean_over(range(nA,len(S_rows)),S_res)
lock_mean={L:torch.stack(L_res[L]).mean(0) for L in LAYERS_NEEDED}
# per-pair: assign each WANDERING a single SUCCESS donor, repo-matched when possible (else round-robin)
def pick_pair(wrow,i):
    same=[j for j,sr in enumerate(S_rows) if sr["repo"]==wrow["repo"]]
    j=(same[0] if same else i%len(S_rows))
    return {L:S_res[L][j] for L in LAYERS_NEEDED}, S_rows[j]["iid"]
print("donors ready | succ split A n=",nA,"B n=",len(S_rows)-nA,"| per-pair repo-matched where available")
'''))

C.append(md("## 7. GENERATION test (the decisive gate) — does the agent actually emit `finish`?"))
C.append(code(r'''
def finish_rate(rows, L=None, donor_fn=None):
    hits=0; examples=[]
    for i,r in enumerate(rows):
        donor=None if donor_fn is None else donor_fn(r,i,L)
        cont=gen_after_function(r["ids"], L=(None if donor is None else L), donor=donor)
        ok=emits_finish(cont); hits+=int(ok)
        if len(examples)<3: examples.append((ok, cont[:40].replace(chr(10)," ")))
        gc.collect(); torch.cuda.empty_cache()
    return hits/len(rows), examples
W=PR["wandering"]
print("=== GENERATION emission-rate of `finish` at the WANDERING decision point (greedy) ===")
rate_base,ex_base = finish_rate(W, L=None, donor_fn=None)
print(f"  NO-PATCH (baseline)            finish-rate = {rate_base:.2f}   eg:",ex_base)
GEN={"baseline":rate_base}
for L in GEN_LAYERS:
    rs,exs = finish_rate(W, L=L, donor_fn=lambda r,i,LL=L: succ_A[LL])
    rl,_   = finish_rate(W, L=L, donor_fn=lambda r,i,LL=L: lock_mean[LL])
    GEN[f"succ_A_L{L}"]=rs; GEN[f"lock_L{L}"]=rl
    print(f"  SUCCESS-donor @L{L} (split A)   finish-rate = {rs:.2f}   eg:",exs)
    print(f"  LOCKED-donor  @L{L} (null)      finish-rate = {rl:.2f}")
print(">>> GATE: succ-donor finish-rate >> baseline AND >> locked-donor  ->  the lever survives a REAL generation.")
'''))

C.append(md("## 8. HELD-OUT donor (split B) + PER-PAIR donor (rule out the averaged-vector confound)"))
C.append(code(r'''
print("=== confirmation: held-out (split-B) donor + per-pair single-trajectory donor ===")
for L in GEN_LAYERS:
    rB,_  = finish_rate(W, L=L, donor_fn=lambda r,i,LL=L: succ_B[LL])
    rP,exP= finish_rate(W, L=L, donor_fn=lambda r,i,LL=L: pick_pair(r,i)[0][LL])
    GEN[f"succ_B_L{L}"]=rB; GEN[f"perpair_L{L}"]=rP
    print(f"  L{L}: held-out(split B) finish-rate = {rB:.2f}  |  per-pair finish-rate = {rP:.2f}   eg:",exP)
print(">>> if both stay >> baseline/locked -> NOT a coarse-mean / split artifact. The late-block lever is real.")
# dense one-token ΔP shape (sanity, cheap) with split-A donor
print("\n=== dense late one-token ΔP(finish) (success-A donor vs locked null) ===")
dPs={}; dPl={}
for L in DENSE:
    s=np.mean([patch_one_token(r["ids"],L,succ_A[L])-r["baseP"] for r in W])
    l=np.mean([patch_one_token(r["ids"],L,lock_mean[L])-r["baseP"] for r in W])
    dPs[L]=float(s); dPl[L]=float(l); print(f"  L{L:2d}: succ {s:+.3f}   locked {l:+.3f}")
    gc.collect(); torch.cuda.empty_cache()
'''))

C.append(md("## 9. Save + verdict"))
C.append(code(r'''
import json
out={"fidelity":{c:float(np.mean([r['baseP'] for r in PR[c]])) for c in PR if PR[c]},
     "generation_finish_rate":GEN,
     "dense_one_token":{"succ_A":dPs,"locked":dPl},
     "config":{"N_W":len(W),"N_S":len(S_rows),"GEN_LAYERS":GEN_LAYERS,"MAXNEW":MAXNEW,"MAXLEN":MAXLEN}}
json.dump(out, open(os.path.join(OUT,"confirm_results.json"),"w"), indent=1)
print("saved", os.path.join(OUT,"confirm_results.json"))
print("\n--- VERDICT CHECKLIST ---")
print("[generation]  succ-donor finish-rate >> baseline & locked ?  -> real lever, not a 1-token bump")
print("[held-out]    split-B donor reproduces ?                     -> not a donor-split artifact")
print("[per-pair]    per-trajectory donor reproduces ?              -> not the coarse-mean")
print("All three GO  ->  BREAKTHROUGH CONFIRMED: the termination lever is the late action-commitment block.")
'''))

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},
    "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).resolve().parent.parent/"notebooks"/"nb_breakthrough_confirm.ipynb"
out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(nb,indent=1))
print("wrote",out,"with",len(C),"cells")
