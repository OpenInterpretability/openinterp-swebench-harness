#!/usr/bin/env python3
"""Builds notebooks/nb_verdict_circuit_h4.ipynb — Verdict-Circuit Stage-1 H4 (causal, GPU/Colab).

The ONLY remaining piece of the verdict-circuit program. Everything else (Stage-0, structural,
H1 interpretation, observational E/E2) was done on CPU. This runs the model to test:
clamp feature #22358 up at a WANDERING decision point -> does P(finish) rise?

Design: clone the harness repo and REUSE its own prompt assembly (agent.prompts.SYSTEM_PROMPT /
render_problem, agent.tools.TOOLS, agent.parser._strip_think) so the reconstructed prompt is
IDENTICAL to the original Phase-6 run. A FIDELITY GATE (cosine of the reconstructed L23 residual
vs the saved capture) runs first and must pass before any H4 number is trusted.

Cell-source literals use ''' (cells contain \"\"\").
"""
import json
from pathlib import Path

def md(s):   return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)}
C=[]

C.append(md(r'''# Verdict Circuit — Stage-1 **H4 (causal)**, the only GPU step

Pre-registered in `paper/verdict_circuit/PREREG_stage1.md`. Stage-0 + H1 + the observational test are
**done on CPU** and committed; this notebook runs the model for the one causal question.

**Feature #22358 @ L23 is a "subtask completed & verified" signal** (H1 PASS) that predicts the `finish`
action (AUROC 0.91) — but WANDERING forms it repeatedly and never terminates. **H4 asks: is the verdict
*sufficient* for the action?** Clamp #22358 to the SUCCESS-final level at a WANDERING decision point and
read ΔP(finish):
- **ΔP(finish) ≈ 0** (and ≈ random-feature clamp) → verdict present but **readout broken** → §13 confirmed (detection ≠ control even at the right feature).
- **ΔP(finish) ≫ 0** (and > random-feature clamp) → verdict was the missing piece → **surgical lever** (extraordinary; then requires L43 #908 replication + a real generation that actually emits finish, never the definitional baseline).

**Honesty gates baked in:** a fidelity check (reconstructed residual must match the saved capture) runs
FIRST; controls = random-active-feature clamp + ablate-in-SUCCESS (necessity); paired per-trajectory.'''))

C.append(md("## 1. Install + clone harness (reuse its exact prompt assembly) + mount Drive"))
C.append(code(r'''
!pip -q install "git+https://github.com/huggingface/transformers.git" safetensors huggingface_hub datasets scipy accelerate
import os, sys, subprocess
REPO_DIR="/content/openinterp-swebench-harness"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git","clone","-q","https://github.com/OpenInterpretability/openinterp-swebench-harness",REPO_DIR], check=True)
sys.path.insert(0, REPO_DIR)
import transformers; print("transformers", transformers.__version__)
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
'''))

C.append(md("## 2. Config"))
C.append(code(r'''
import torch, numpy as np
os.environ.setdefault("HF_TOKEN","")          # HF read token (model + SAE)
MODEL_ID="Qwen/Qwen3.6-27B"; LAYER=23; FEATURE=22358
SAE_REPO="caiovicentino1/qwen36-27b-sae-fullstack"
DRIVE="/content/drive/MyDrive/openinterp_runs/swebench_v6_phase6"   # has captures/ and traces/
OUT="/content/drive/MyDrive/openinterp_runs/verdict_circuit_stage1"; os.makedirs(OUT, exist_ok=True)
TOOL_NAMES=["bash","str_replace_editor","finish"]
PREFIX='<function'   # cut before '='; first divergent token after <function picks the tool
FIDELITY_COS=0.60      # keep trajectories whose reconstructed L23 residual matches the saved capture
N_PER_CLASS=15
DEVICE="cuda" if torch.cuda.is_available() else "cpu"; print("device",DEVICE)
'''))

C.append(md("## 3. Load model + SAE + hooks"))
C.append(code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors import safe_open
tok=AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
                                           trust_remote_code=True).eval()
def get_layer_module(m, idx):
    for path in ("model.language_model.layers","language_model.layers","model.model.layers","model.layers"):
        cur=m; ok=True
        for p in path.split("."):
            if not hasattr(cur,p): ok=False; break
            cur=getattr(cur,p)
        if ok:
            try: return cur[idx]
            except Exception: continue
    raise RuntimeError("layer not found")
LAYER_MOD=get_layer_module(model,LAYER)
SAE={}
with safe_open(hf_hub_download(SAE_REPO,f"sae_L{LAYER}_latest.safetensors"),"pt") as f:
    for k in f.keys(): SAE[k]=f.get_tensor(k).to(DEVICE,torch.float32)
K_TOPK=128
def sae_encode(x):
    z=(x.float()-SAE["b_dec"])@SAE["W_enc"]+SAE["b_enc"]; z=torch.relu(z)
    if z.shape[-1]>K_TOPK:
        thr=z.topk(K_TOPK,dim=-1).values[...,-1:].clamp_min(1e-9); z=torch.where(z>=thr,z,torch.zeros_like(z))
    return z
_CAP={}
def _hook(m,i,o): _CAP["hs"]=(o[0] if isinstance(o,tuple) else o).detach()
def fwd(ids):
    h=LAYER_MOD.register_forward_hook(_hook)
    try:
        with torch.no_grad(): out=model(input_ids=ids.to(model.device))
    finally: h.remove()
    return out.logits, _CAP["hs"]
def first_tok_after(prefix,name):
    a=tok(prefix,add_special_tokens=False).input_ids; b=tok(prefix+name,add_special_tokens=False).input_ids
    return b[len(a)]
ACTION_TOK={"finish":28,"bash":21402,"str_replace_editor":15462}   # verified token ids ( =/=b/=str )
def p_finish(nl):
    ids=[ACTION_TOK[n] for n in TOOL_NAMES]; p=torch.softmax(nl[ids].float(),-1)
    return float(p[TOOL_NAMES.index("finish")])
print("action tokens:",{n:(t,repr(tok.decode([t]))) for n,t in ACTION_TOK.items()})
'''))

C.append(md("## 4. Harness assembly (reused) + SWE-bench Pro task text + labels"))
C.append(code(r'''
from agent.prompts import SYSTEM_PROMPT, render_problem
from agent.tools import TOOLS
from agent.parser import _strip_think
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from pathlib import Path
import csv, glob, json as J, tarfile
# Account-independent: traces + labels + fidelity residuals bundled on HF (the Colab Drive account
# need not hold the Phase-6 data).
DATA_REPO="caiovicentino1/swebench-phase6-verdict-circuit"
os.makedirs("/content/vc_data",exist_ok=True)
with tarfile.open(hf_hub_download(DATA_REPO,"traces.tar.gz",repo_type="dataset")) as t: t.extractall("/content/vc_data")
TRACES={Path(p).stem:p for p in glob.glob("/content/vc_data/traces/instance_*.json")}
LAB={r["iid"]:r["sub_class"] for r in csv.DictReader(open(hf_hub_download(DATA_REPO,"features_n99.csv",repo_type="dataset")))}
FIDR=dict(np.load(hf_hub_download(DATA_REPO,"final_pretool_L23.npz",repo_type="dataset")))
ds=load_dataset("ScaleAI/SWE-bench_Pro", split="test")
by_iid={r["instance_id"]:r for r in ds if r.get("instance_id")}
def instance_for(iid):
    if iid in by_iid: return by_iid[iid]
    return next((r for k,r in by_iid.items() if k in iid or iid in k), None)
print("traces",len(TRACES),"| labels",len(LAB),"| fidresid",len(FIDR),"| dataset",len(ds))
'''))

C.append(md('''## 5. Reconstruct the WANDERING/SUCCESS **final-turn decision point** (reuse harness assembly) + FIDELITY GATE
Build messages exactly as `agent/loop.py` did (system + render_problem + per-turn assistant `body_with_tools`
+ tool results), then teacher-force the final turn up to `<tool_call>{"name": "`. The fidelity gate compares
the reconstructed L23 residual to the **saved capture** — only matching trajectories are used for H4.'''))
C.append(code(r'''
def build_messages_upto_final(trace, instance):
    msgs=[{"role":"system","content":SYSTEM_PROMPT}]
    wd=f"/content/work_p6/{trace['instance_id']}"
    msgs.append({"role":"user","content":render_problem({**instance,"__workdir__":wd})})
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
    full = prefix + raw[:cut+len(PREFIX)]            # ends exactly at the tool-name choice point
    return tok(full, return_tensors="pt", truncation=True, max_length=16000).input_ids

def saved_pretool_residual(iid):
    v=FIDR.get(iid) or next((FIDR[k] for k in FIDR if iid.startswith(k) or k.startswith(iid)),None)
    return torch.tensor(np.asarray(v),device=DEVICE) if v is not None else None

def collect(cls, n):
    out=[]
    for iid,tp in TRACES.items():
        lab=LAB.get(iid) or next((LAB[k] for k in LAB if iid.startswith(k) or k.startswith(iid)),None)
        if lab!=cls: continue
        tr=J.load(open(tp)); inst=instance_for(iid)
        if inst is None: continue
        ids=choice_point_ids(tr, inst)
        if ids is None: continue
        logit,hs=fwd(ids)
        rec=hs[0,-1]                                  # reconstructed choice-point residual
        sav=saved_pretool_residual(iid)
        cos=float(torch.cosine_similarity(rec.float(), sav.float(), dim=0)) if sav is not None else float("nan")
        out.append({"iid":iid,"ids":ids,"baseP":p_finish(logit[0,-1]),
                    "feat":float(sae_encode(rec[None])[0,FEATURE]),"fid_cos":cos})
        if len(out)>=n: break
    return out

print("Reconstructing (fidelity check)…")
W=collect("wandering",N_PER_CLASS); S=collect("success",N_PER_CLASS)
import numpy as np
print(f"WANDERING n={len(W)} fidelity cos median={np.nanmedian([r['fid_cos'] for r in W]):.3f}  baseP(finish)={np.mean([r['baseP'] for r in W]):.3f}")
print(f"SUCCESS   n={len(S)} fidelity cos median={np.nanmedian([r['fid_cos'] for r in S]):.3f}  baseP(finish)={np.mean([r['baseP'] for r in S]):.3f}")
print(">>> If fidelity cos is low (<0.9), the reconstruction is off — debug assembly BEFORE trusting H4.")
'''))

C.append(md("## 6. **H4 causal** (only on fidelity-passing trajectories)"))
C.append(code(r'''
import random
from scipy.stats import wilcoxon
Wok=[r for r in W if r["fid_cos"]>=FIDELITY_COS]; Sok=[r for r in S if r["fid_cos"]>=FIDELITY_COS]
print(f"fidelity-passing: WANDERING {len(Wok)}/{len(W)}  SUCCESS {len(Sok)}/{len(S)}")
target=float(np.mean([r["feat"] for r in Sok])) if Sok else 2.0   # SUCCESS-final feature level
print("clamp target (SUCCESS feature level):",round(target,3))

def clamp_hook(f,val):
    wdec=SAE["W_dec"][f]
    def h(m,i,o):
        hs=o[0] if isinstance(o,tuple) else o
        x=hs[:,-1:,:]; cur=sae_encode(x.float())[...,f:f+1]
        x.add_(((val-cur)*wdec).to(hs.dtype))
        return (hs,*o[1:]) if isinstance(o,tuple) else hs
    return h
def p_clamped(ids,f,val):
    h=LAYER_MOD.register_forward_hook(clamp_hook(f,val))
    try:
        with torch.no_grad(): lo=model(input_ids=ids.to(model.device)).logits
    finally: h.remove()
    return p_finish(lo[0,-1])

# random ACTIVE feature for the null
act=(sae_encode(fwd(Sok[0]["ids"])[1][0,-1:])[0]>0).nonzero().flatten().tolist()
randf=random.choice([a for a in act if a!=FEATURE])
res={"clamp_feature":[],"clamp_random":[],"ablate_success":[]}
for r in Wok:
    res["clamp_feature"].append(p_clamped(r["ids"],FEATURE,target)-r["baseP"])
    res["clamp_random"].append(p_clamped(r["ids"],randf,target)-r["baseP"])
for r in Sok:
    res["ablate_success"].append(p_clamped(r["ids"],FEATURE,0.0)-r["baseP"])
def summ(n,a):
    a=np.array(a);
    try: p=float(wilcoxon(a).pvalue)
    except Exception: p=None
    print(f"{n:16} mean ΔP(finish)={a.mean():+.3f} median={np.median(a):+.3f} p={p} n={len(a)}")
print("=== H4 ==="); summ("clamp #22358",res["clamp_feature"]); summ("clamp random",res["clamp_random"]); summ("ablate(success)",res["ablate_success"])
print("READOUT: clamp~0 & ≈random -> §13 CONFIRMED (verdict present, readout broken).")
print("         clamp>>0 & >random -> surgical lever (EXTRAORDINARY -> needs L43 #908 + real-generation finish).")
'''))

C.append(md("## 7. Save"))
C.append(code(r'''
J.dump({"feature":FEATURE,"layer":LAYER,"fidelity_cos_threshold":FIDELITY_COS,
        "n_wander_ok":len(Wok),"n_success_ok":len(Sok),"clamp_target":target,
        "H4":{k:list(map(float,v)) for k,v in res.items()},
        "fidelity_cos":{ "wander":[r["fid_cos"] for r in W], "success":[r["fid_cos"] for r in S]}},
       open(os.path.join(OUT,"stage1_h4.json"),"w"), indent=1)
print("saved",os.path.join(OUT,"stage1_h4.json"))
'''))

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},
    "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).resolve().parent.parent/"notebooks"/"nb_verdict_circuit_h4.ipynb"
out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(nb,indent=1))
print("wrote",out,"with",len(C),"cells")
