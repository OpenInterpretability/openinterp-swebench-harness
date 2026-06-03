#!/usr/bin/env python3
"""Builds notebooks/nb_breakthrough_action_channel.ipynb — the breakthrough experiment (H1 + H2), GPU/Colab.

Tests the candidate LAW: for long-horizon agents, the control surface of a decision is the ACTION/token
channel, not the internal representation. Reuses the verified H4 reconstruction infra (faithful decision-
point prompts from the HF data bundle — account-independent, sidesteps the Drive). Pre-registered in
paper/breakthrough/PREREG_action_channel.md ("characterize the boundary" — both outcomes are positive).

Obs : logit-lens across layers — where does the finish decision become readable?
H1  : layer-sweep activation patching (success-donor vs locked-donor) — is there an internal lever at ANY layer?
H2  : output-targeted feature steering vs the naive #22358 clamp — was the clamp the wrong method?

Cell-source literals use ''' (cells contain \"\"\").
"""
import json
from pathlib import Path

def md(s):   return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)}
C=[]

C.append(md(r'''# Breakthrough — Is the agent's control surface the ACTION channel, not the representation?

Pre-registered in `paper/breakthrough/PREREG_action_channel.md`. The candidate **LAW**: detection lives in the
representation (probes/features predict `finish` at AUROC 0.81–0.91) but every representational intervention
fails (3 residual nulls + the #22358 clamp ΔP=−0.001), while the behavioral interruption rescues 30→70%
(paper #4). So control may live in the **action/token channel**, not the representation.

**Framing (both outcomes positive):** (a) if patching/steering at the right locus DOES cause `finish` → we
found the first internal lever → *closed the knowledge-action gap on agents*. (b) if NOTHING internal works
at any layer/feature → the gap is *irreducible internally; the action channel is the unique control surface*
— a law, not a null.

- **Obs:** logit-lens across layers (where is the decision readable?)
- **H1:** layer-sweep activation patching — success-donor vs locked-donor null.
- **H2:** output-targeted feature steering vs the naive clamp.

Reuses the H4 faithful-reconstruction infra; data from the HF bundle (no Drive needed).'''))

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

C.append(md("## 2. Config"))
C.append(code(r'''
import torch, numpy as np
os.environ.setdefault("HF_TOKEN","")          # HF read token (model + SAE)
MODEL_ID="Qwen/Qwen3.6-27B"; LAYER=23; FEATURE=22358
SAE_REPO="caiovicentino1/qwen36-27b-sae-fullstack"
DATA_REPO="caiovicentino1/swebench-phase6-verdict-circuit"
OUT="/content/breakthrough_out"; os.makedirs(OUT, exist_ok=True)
TOOL_NAMES=["bash","str_replace_editor","finish"]
PREFIX="<function"
ACTION_TOK={"finish":28,"bash":21402,"str_replace_editor":15462}   # verified token ids (=/=b/=str)
MAXLEN=4000; N_PER_CLASS=12
SWEEP=list(range(3,64,4))   # L3,7,...,63 layer sweep
DEVICE="cuda" if torch.cuda.is_available() else "cpu"; print("device",DEVICE)
'''))

C.append(md("## 3. Load model + SAE + hooks (memory-lean; multi-layer capture)"))
C.append(code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors import safe_open
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
LAYER_MOD=get_layer_module(LAYER)
SAE={}
with safe_open(hf_hub_download(SAE_REPO,f"sae_L{LAYER}_latest.safetensors"),"pt") as f:
    for k in f.keys(): SAE[k]=f.get_tensor(k).to(DEVICE,torch.float32)
K_TOPK=128
def sae_encode(x):
    z=(x.float()-SAE["b_dec"])@SAE["W_enc"]+SAE["b_enc"]; z=torch.relu(z)
    thr=z.topk(K_TOPK,dim=-1).values[...,-1:].clamp_min(1e-9); return torch.where(z>=thr,z,torch.zeros_like(z))
# final norm weight + unembedding (for logit-lens / DLA)
NORM_W=None
for nm in ["model.language_model.norm","model.model.norm","model.norm"]:
    cur=model; ok=True
    for p in nm.split("."):
        if not hasattr(cur,p): ok=False;break
        cur=getattr(cur,p)
    if ok and hasattr(cur,"weight"): NORM_W=cur.weight.detach().float(); break
WU=model.get_output_embeddings().weight.detach()
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
with tarfile.open(hf_hub_download(DATA_REPO,"traces.tar.gz",repo_type="dataset")) as t: t.extractall("/content/vc_data")
TRACES={Path(p).stem:p for p in glob.glob("/content/vc_data/traces/instance_*.json")}
LAB={r["iid"]:r["sub_class"] for r in csv.DictReader(open(hf_hub_download(DATA_REPO,"features_n99.csv",repo_type="dataset")))}
ds=load_dataset("ScaleAI/SWE-bench_Pro", split="test")
by_iid={r["instance_id"]:r for r in ds if r.get("instance_id")}
def instance_for(iid):
    if iid in by_iid: return by_iid[iid]
    return next((r for k,r in by_iid.items() if k in iid or iid in k), None)
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
        lg,_=fwd_layers(ids,[]);
        out.append({"iid":iid,"ids":ids.cpu(),"baseP":p_finish(lg[0,-1])})
        gc.collect(); torch.cuda.empty_cache()
        if len(out)>=n: break
    return out
PR={c:collect(c,N_PER_CLASS) for c in ["success","wandering","locked"]}
for c in PR: print(f"{c:10} n={len(PR[c])} baseP(finish)={np.mean([r['baseP'] for r in PR[c]]) if PR[c] else float('nan'):.3f}")
print(">>> FIDELITY GATE: proceed only if SUCCESS baseP(finish) >> WANDERING (model recovers the real decision).")
'''))

C.append(md("## 6. OBS — logit-lens across layers: where does the finish decision become readable?"))
C.append(code(r'''
def logit_lens(resid):  # resid [d] at some layer -> finish-minus-others logit (RMSNorm-approx logit lens)
    x=resid.float(); x=x/ (x.norm()/math.sqrt(x.numel()) + 1e-6)
    if NORM_W is not None: x=x*NORM_W
    lg=x@WU.float().T
    return float(lg[ACTION_TOK["finish"]] - 0.5*(lg[ACTION_TOK["bash"]]+lg[ACTION_TOK["str_replace_editor"]]))
def class_layer_lens(cls,n):
    acc={L:[] for L in SWEEP}
    for r in PR[cls][:n]:
        _,caps=fwd_layers(r["ids"],SWEEP)
        for L in SWEEP: acc[L].append(logit_lens(caps[L][0,0]))
        torch.cuda.empty_cache()
    return {L:float(np.mean(acc[L])) for L in SWEEP}
lensS=class_layer_lens("success",N_PER_CLASS); lensW=class_layer_lens("wandering",N_PER_CLASS)
print("layer :  SUCCESS  WANDERING   gap(S-W)  <- where the finish-direction emerges")
for L in SWEEP:
    print(f"L{L:2d}   : {lensS[L]:+7.2f}  {lensW[L]:+8.2f}   {lensS[L]-lensW[L]:+7.2f}")
print(">>> READ: the layer where gap(S-W) jumps = where the decision becomes readable. Late jump supports the action-channel thesis.")
'''))

C.append(md("## 7. H1 — layer-sweep activation patching (the causal core)"))
C.append(code(r'''
def class_layer_means(cls,n):
    acc={L:[] for L in SWEEP}
    for r in PR[cls][:n]:
        _,caps=fwd_layers(r["ids"],SWEEP)
        for L in SWEEP: acc[L].append(caps[L][0,0].float())
        torch.cuda.empty_cache()
    return {L:torch.stack(acc[L]).mean(0) for L in SWEEP}
succ_mean=class_layer_means("success",N_PER_CLASS); lock_mean=class_layer_means("locked",N_PER_CLASS)
def patch_layer(ids,L,donor):
    def h(m,i,o):
        hs=o[0] if isinstance(o,tuple) else o; hs=hs.clone(); hs[:,-1,:]=donor.to(hs.dtype)
        return (hs,*o[1:]) if isinstance(o,tuple) else hs
    hh=get_layer_module(L).register_forward_hook(h)
    try:
        with torch.no_grad(): lg=model(input_ids=ids.to(model.device),use_cache=False,logits_to_keep=1).logits
    finally: hh.remove()
    r=p_finish(lg[0,-1]); del lg; return r
dP_s={L:[] for L in SWEEP}; dP_l={L:[] for L in SWEEP}
for r in PR["wandering"][:N_PER_CLASS]:
    base=r["baseP"]
    for L in SWEEP:
        dP_s[L].append(patch_layer(r["ids"],L,succ_mean[L])-base)
        dP_l[L].append(patch_layer(r["ids"],L,lock_mean[L])-base)
    gc.collect(); torch.cuda.empty_cache()
print("=== H1: inject SUCCESS state at layer L into WANDERING -> ΔP(finish) (vs LOCKED-donor null) ===")
print("layer :  success-donor   locked-donor(null)")
best=None
for L in SWEEP:
    s=np.mean(dP_s[L]); l=np.mean(dP_l[L]); print(f"L{L:2d}   :   {s:+.3f}          {l:+.3f}")
    if best is None or s>best[1]: best=(L,s)
print(f">>> best success-donor layer: L{best[0]} ΔP={best[1]:+.3f}")
print(">>> GATE: if some layer's success-donor ΔP >> locked-donor and > ~0.1 -> INTERNAL LEVER EXISTS (gap closable).")
print(">>>       if all ≈ locked null -> the LAW: no internal lever at the decision residual; control is in the action channel.")
'''))

C.append(md("## 8. H2 — output-targeted feature steering vs the naive #22358 clamp"))
C.append(code(r'''
with torch.no_grad():
    dfin=(WU[ACTION_TOK["finish"]].float() - 0.5*(WU[ACTION_TOK["bash"]].float()+WU[ACTION_TOK["str_replace_editor"]].float()))
    if NORM_W is not None: dfin=dfin*NORM_W
    dla=(SAE["W_dec"].float()@dfin)                 # each L23 feature's direct finish-logit attribution
    topfeat=torch.topk(dla,5).indices.tolist()
    rank22358=int((dla>dla[FEATURE]).sum())
print("top finish-promoting L23 features by DLA:",topfeat,"| #22358 DLA rank:",rank22358,"/",dla.numel())
def steer_feat(ids,f,val):
    wd=SAE["W_dec"][f]
    def h(m,i,o):
        hs=o[0] if isinstance(o,tuple) else o; hs=hs.clone()
        cur=sae_encode(hs[:,-1:,:].float())[...,f:f+1]
        hs[:,-1:,:]=hs[:,-1:,:]+((val-cur)*wd).to(hs.dtype)
        return (hs,*o[1:]) if isinstance(o,tuple) else hs
    hh=LAYER_MOD.register_forward_hook(h)
    try:
        with torch.no_grad(): lg=model(input_ids=ids.to(model.device),use_cache=False,logits_to_keep=1).logits
    finally: hh.remove()
    r=p_finish(lg[0,-1]); del lg; return r
import random
STEER=3.0  # steer target activation (≈ SUCCESS-level scale)
res={"top_dla_feat":[],"clamp_22358":[],"random_feat":[]}
# a random active feature as null
_,caps0=fwd_layers(PR["wandering"][0]["ids"],[]);
act=(sae_encode(fwd_layers(PR['success'][0]['ids'],[LAYER])[1][LAYER][0,-1:])[0]>0).nonzero().flatten().tolist()
randf=random.choice([a for a in act if a not in topfeat and a!=FEATURE])
for r in PR["wandering"][:N_PER_CLASS]:
    base=r["baseP"]
    res["top_dla_feat"].append(steer_feat(r["ids"],topfeat[0],STEER)-base)
    res["clamp_22358"].append(steer_feat(r["ids"],FEATURE,STEER)-base)
    res["random_feat"].append(steer_feat(r["ids"],randf,STEER)-base)
    gc.collect(); torch.cuda.empty_cache()
print("=== H2: steer a feature UP at L23 decision point -> ΔP(finish) ===")
for k,v in res.items(): print(f"  {k:14} mean ΔP={np.mean(v):+.3f}  median={np.median(v):+.3f}  n={len(v)}")
print(">>> if top-DLA feature steering >> random -> an internal FEATURE lever exists (the #22358 clamp null was a method/feature artifact).")
'''))

C.append(md("## 9. Save"))
C.append(code(r'''
import json
out={"obs_logitlens":{"success":lensS,"wandering":lensW,"sweep":SWEEP},
     "H1_patch":{"success_donor":{int(L):float(np.mean(dP_s[L])) for L in SWEEP},
                 "locked_donor":{int(L):float(np.mean(dP_l[L])) for L in SWEEP},
                 "best_layer":int(best[0]),"best_dP":float(best[1])},
     "H2_steer":{k:list(map(float,v)) for k,v in res.items()},
     "H2_top_dla_features":topfeat,"feat22358_dla_rank":rank22358,
     "fidelity":{c:float(np.mean([r['baseP'] for r in PR[c]])) for c in PR if PR[c]}}
json.dump(out, open(os.path.join(OUT,"breakthrough_results.json"),"w"), indent=1)
print("saved", os.path.join(OUT,"breakthrough_results.json"))
'''))

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},
    "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).resolve().parent.parent/"notebooks"/"nb_breakthrough_action_channel.ipynb"
out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(nb,indent=1))
print("wrote",out,"with",len(C),"cells")
