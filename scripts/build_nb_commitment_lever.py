#!/usr/bin/env python3
"""Builds notebooks/nb_commitment_lever.ipynb — Tier-1 of the circuit-breaker program.

Tests whether the LATE commitment lever (paper #6: control of `finish` lives in L51-63, ~30 layers downstream
of the verdict) GENERALIZES to a second, distinct action-commitment decision: `str_replace_editor` (commit a
state-mutating file edit) vs `bash` (reversible exploration). Pre-registered in
paper/circuit_breaker/PREREG_commitment_lever.md (H1 elicit, H2 SUPPRESS=the brake, H3 lever-layer transfer).

Reuses the verified #6 reconstruction infra (faithful decision-point prompts from the HF bundle, no Drive) and
`decision-locator`'s logit (locate) + patch (sweep_patch / steer_generate) primitives. The ONE generalization:
decision points are taken at ANY turn whose chosen tool is the target (not only the final turn), so we have many
edit-decision and bash-decision points with their turn indices (for the positional confound control).

Cell-source literals use ''' (cells contain \"\"\").
"""
import json
from pathlib import Path

def md(s):   return {"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)}
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(keepends=True)}
C=[]

C.append(md(r'''# Commitment-lever generalization — does the late lever control a SECOND action (`edit`), not just `finish`?

Paper #6 found the termination lever is **late** (L51-63) and task-matched. This notebook (Tier-1 of the
mechanistic-circuit-breaker program, `PREREG_commitment_lever.md`) tests whether that is a **general property of
the action-commitment channel** or specific to `finish`, using a second, distinct decision available in the
existing data: **commit a file edit (`str_replace_editor`) vs explore (`bash`)**.

- **LOCATE** — where does the edit decision become readable (logit-lens sweep)? Late like `finish`, or mid?
- **H1 ELICIT** — at a *bash* (explore) decision point, patch a task-matched **edit-donor** into the late block:
  does P(edit) rise and does the model actually emit `<function=str_replace_editor`? Controls: bash-donor (null)
  + cross-task edit-donor (task-specificity).
- **H2 SUPPRESS (the brake)** — at an *edit* decision point, patch a **bash-donor**: does the commit get
  blocked (P(edit) down, model explores instead)? This is the circuit-breaker direction.
- **H3 TRANSFER** — does the edit lever-layer coincide with `finish`'s L51-63?

Generalization vs another null is the gate: lever generalizes -> general action-channel law + circuit-breaker
greenlit; lever is finish-specific -> termination is mechanistically special (also publishable, saves the
crypto build). Reuses #6 reconstruction infra; data from the HF bundle (no Drive).'''))

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

C.append(md("## 2. Config — target action = `str_replace_editor` (commit), alt = `bash` (explore)"))
C.append(code(r'''
import torch, numpy as np
os.environ.setdefault("HF_TOKEN","")          # HF read token (model)
MODEL_ID="Qwen/Qwen3.6-27B"
DATA_REPO="caiovicentino1/swebench-phase6-verdict-circuit"
OUT="/content/commit_lever_out"; os.makedirs(OUT, exist_ok=True)
TOOL_NAMES=["bash","str_replace_editor","finish"]
ACTION_TOK={"finish":28,"bash":21402,"str_replace_editor":15462}   # verified token ids (=/=b/=str)
PREFIX="<function"
TARGET="str_replace_editor"      # the committal/semi-irreversible action under test
ALT="bash"                       # the reversible-exploration alternative
MAXLEN=4000
N_EDIT=60        # edit decision points (target) to collect
N_BASH=60        # bash decision points (alt) to collect
SWEEP=list(range(3,64,4))            # FULL sweep for LOCATE (does edit emerge late like finish?)
PATCH_LATE=[48,52,55,59,63]          # late block where #6's lever lived
PATCH_MID=[23,31]                    # mid controls (verdict / consolidation) — should be inert if law holds
GEN_LAYERS=[55,59]                   # layers to run the GENERATION test at
MAXNEW=16
DEVICE="cuda" if torch.cuda.is_available() else "cpu"; print("device",DEVICE)
'''))

C.append(md("## 3. Load model + hooks + per-action prob + patched-generation helpers"))
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
def p_action(nl, name=TARGET):
    ids=[ACTION_TOK[n] for n in TOOL_NAMES]; p=torch.softmax(nl[ids].float(),-1)
    return float(p[TOOL_NAMES.index(name)])
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
def patch_one_token(ids,L,donor,name=TARGET):     # one-token ΔP(action) under a layer-L last-pos patch
    hh=get_layer_module(L).register_forward_hook(_patch_hook(donor))
    try:
        with torch.no_grad(): lg=model(input_ids=ids.to(model.device),use_cache=False,logits_to_keep=1).logits
    finally: hh.remove()
    r=p_action(lg[0,-1],name); del lg; return r
def gen_after_function(ids, L=None, donor=None, maxnew=MAXNEW):
    """PREFILL-ONLY patch (decision-locator canonical method): inject `donor` at the decision position on
    the prefill pass only, then decode freely. Patching EVERY step degenerates into repetition (=str=str...)."""
    hh=None
    if L is not None and donor is not None:
        def _ph(m,i,o):
            hs=o[0] if isinstance(o,tuple) else o
            if hs.shape[1] > 1:                       # prefill only; skip single-token decode steps
                hs=hs.clone(); hs[:,-1,:]=donor.to(hs.dtype)
                return (hs,*o[1:]) if isinstance(o,tuple) else hs
            return o
        hh=get_layer_module(L).register_forward_hook(_ph)
    try:
        with torch.no_grad():
            out=model.generate(input_ids=ids.to(model.device), max_new_tokens=maxnew, do_sample=False,
                               use_cache=True, pad_token_id=tok.eos_token_id,
                               attention_mask=torch.ones_like(ids).to(model.device))
    finally:
        if hh is not None: hh.remove()
    cont=tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False); del out; return cont
def emits(cont, name):
    s=cont.lstrip()
    if s.startswith("="): s=s[1:]
    s=s.lstrip().lower()
    return s.startswith(name.lower()[:6]) or (name.lower()[:6] in cont[:24].lower())
# logit-lens (decision-locator LOCATE primitive): project a layer residual through final norm + unembed
def final_norm():
    for path in ("model.language_model.norm","language_model.norm","model.model.norm","model.norm"):
        cur=model; ok=True
        for p in path.split("."):
            if not hasattr(cur,p): ok=False; break
            cur=getattr(cur,p)
        if ok: return cur
    return None
NORM=final_norm(); LMH=model.get_output_embeddings()
def lens_gap(resid_vec, name=TARGET):
    with torch.no_grad():
        h=resid_vec.to(model.device).to(next(model.parameters()).dtype).unsqueeze(0)
        logits=LMH(NORM(h))[0].float()
    tid=ACTION_TOK[name]; alts=[ACTION_TOK[n] for n in TOOL_NAMES if n!=name]
    return float(logits[tid]-logits[alts].mean())
print("ready. tokens:", {n:(t,repr(tok.decode([t]))) for n,t in ACTION_TOK.items()})
'''))

C.append(md("## 4. Harness assembly + HF data bundle (no Drive)"))
C.append(code(r'''
from agent.prompts import SYSTEM_PROMPT, render_problem
from agent.tools import TOOLS
from agent.parser import _strip_think
from datasets import load_dataset
from pathlib import Path
import csv, glob, json as J, tarfile, gc
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
def repo_of(iid): return iid.split("__")[0] if "__" in iid else iid.split("-")[0]
print("traces",len(TRACES),"| labels",len(LAB),"| dataset",len(ds))
'''))

C.append(md(r'''## 5. Generalized decision-point reconstruction (ANY turn) + collect edit/bash points

The #6 infra cut at the FINAL turn's `<function`. Here `choice_point_at_turn(trace,inst,k)` rebuilds messages
up to turn k and cuts turn k's raw_response at `<function`, so the model is poised exactly before choosing the
tool it actually chose at turn k. We collect points where the chosen tool was the TARGET (edit) and where it was
the ALT (bash), recording the **turn index** for the positional-confound control.'''))
C.append(code(r'''
def chosen_tool(turn):
    tcs=turn.get("tool_calls") or []
    return (tcs[0].get("name") if tcs else None)
def build_messages_upto(trace, instance, k):
    msgs=[{"role":"system","content":SYSTEM_PROMPT},
          {"role":"user","content":render_problem({**instance,"__workdir__":f"/content/work_p6/{trace['instance_id']}"})}]
    for tn in trace["turns"][:k]:
        _,body=_strip_think(tn.get("raw_response") or "")
        msgs.append({"role":"assistant","content":body})
        for tr_ in (tn.get("tool_results") or []):
            msgs.append({"role":"tool","content":J.dumps(tr_.get("result"),ensure_ascii=False)[:32000]})
        if not tn.get("tool_calls"):
            msgs.append({"role":"user","content":"You did not call a tool. Use bash, str_replace_editor, or finish."})
    return msgs
def choice_point_at_turn(trace, instance, k):
    msgs=build_messages_upto(trace, instance, k)
    prefix=tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False)
    raw=trace["turns"][k].get("raw_response") or ""
    cut=raw.find(PREFIX)
    if cut<0: return None
    ids=tok(prefix+raw[:cut+len(PREFIX)], add_special_tokens=False).input_ids
    if len(ids)>MAXLEN: ids=ids[-MAXLEN:]
    return torch.tensor([ids])
def collect_points(target_name, n, max_per_traj=3):
    out=[]
    for iid,tp in TRACES.items():
        tr=J.load(open(tp)); inst=instance_for(iid)
        if inst is None: continue
        taken=0
        for k,turn in enumerate(tr["turns"]):
            if k==0: continue
            if chosen_tool(turn)!=target_name: continue
            ids=choice_point_at_turn(tr,inst,k)
            if ids is None: continue
            lg,_=fwd_layers(ids,[])
            out.append({"iid":iid,"repo":repo_of(iid),"turn":k,"nturns":len(tr["turns"]),
                        "ids":ids.cpu(),"baseP_edit":p_action(lg[0,-1],TARGET),"baseP_alt":p_action(lg[0,-1],ALT)})
            taken+=1; gc.collect(); torch.cuda.empty_cache()
            if taken>=max_per_traj: break
            if len(out)>=n: break
        if len(out)>=n: break
    return out
EDIT=collect_points(TARGET, N_EDIT)
BASH=collect_points(ALT,    N_BASH)
print(f"edit points n={len(EDIT)}  baseP(edit)={np.mean([r['baseP_edit'] for r in EDIT]):.3f}  turn={np.mean([r['turn'] for r in EDIT]):.1f}")
print(f"bash points n={len(BASH)}  baseP(edit)={np.mean([r['baseP_edit'] for r in BASH]):.3f}  turn={np.mean([r['turn'] for r in BASH]):.1f}")
print(">>> FIDELITY GATE: proceed only if baseP(edit) at EDIT points >> at BASH points (states reproduce the real propensity).")
'''))

C.append(md("## 6. LOCATE — where the edit decision becomes readable (logit-lens sweep). Late like `finish`, or mid?"))
C.append(code(r'''
def lens_profile(rows, layers):
    prof={L:[] for L in layers}
    for r in rows:
        _,caps=fwd_layers(r["ids"],layers)
        for L in layers: prof[L].append(lens_gap(caps[L][0,0], TARGET))
        gc.collect(); torch.cuda.empty_cache()
    return {L:float(np.mean(v)) for L,v in prof.items()}
edit_lens=lens_profile(EDIT, SWEEP)
bash_lens=lens_profile(BASH, SWEEP)
print("L : edit_pts_gap   bash_pts_gap   (gap = logit[edit] - mean logit[bash,finish])")
for L in SWEEP: print(f"L{L:2d}: {edit_lens[L]:+7.3f}      {bash_lens[L]:+7.3f}")
print(">>> if the edit gap is flat early and rises in the LATE block (L48+), the geometry matches `finish` (H3 toward general law).")
'''))

C.append(md("## 7. Donors — per-trajectory edit/bash residuals; task-matched + cross-task + neutral controls"))
C.append(code(r'''
LAYERS_NEEDED=sorted(set(PATCH_LATE+PATCH_MID+GEN_LAYERS))
def per_point_resids(rows):
    acc={L:[] for L in LAYERS_NEEDED}
    for r in rows:
        _,caps=fwd_layers(r["ids"],LAYERS_NEEDED)
        for L in LAYERS_NEEDED: acc[L].append(caps[L][0,0].float().cpu())
        gc.collect(); torch.cuda.empty_cache()
    return acc
E_res=per_point_resids(EDIT); B_res=per_point_resids(BASH)
def edit_donor_for(brow, i):           # task-matched: an edit point from the SAME repo if possible
    same=[j for j,er in enumerate(EDIT) if er["repo"]==brow["repo"]]
    j=(same[0] if same else i%len(EDIT))
    return {L:E_res[L][j] for L in LAYERS_NEEDED}, EDIT[j]["iid"]
def crosstask_edit_donor_for(brow, i): # cross-task: an edit point from a DIFFERENT repo (specificity control)
    diff=[j for j,er in enumerate(EDIT) if er["repo"]!=brow["repo"]]
    j=(diff[i%len(diff)] if diff else i%len(EDIT))
    return {L:E_res[L][j] for L in LAYERS_NEEDED}
def bash_donor_for(erow, i):           # neutral/explore donor (same position-type) — isolates edit CONTENT vs position
    same=[j for j,br in enumerate(BASH) if br["repo"]==erow["repo"]]
    j=(same[0] if same else i%len(BASH))
    return {L:B_res[L][j] for L in LAYERS_NEEDED}
print("donors ready | edit per-pair repo-matched, cross-task control, bash neutral control")
'''))

C.append(md(r'''## 8. H1 ELICIT — patch an edit-donor at a BASH point: does the commit get elicited?

Gate: edit-donor finish... er, edit-emission rate >> baseline AND >> bash-donor null AND >> cross-task donor.
The **bash-donor control shares turn-index and task** with nothing changed but the donor identity -> it absorbs
the positional confound: if only the edit-donor flips the choice, the effect is the edit content, not lateness.'''))
C.append(code(r'''
def emit_rate(rows, name, L=None, donor_fn=None):
    hits=0; ex=[]
    for i,r in enumerate(rows):
        donor=None if donor_fn is None else donor_fn(r,i,L)
        cont=gen_after_function(r["ids"], L=(None if donor is None else L), donor=donor)
        ok=emits(cont,name); hits+=int(ok)
        if len(ex)<3: ex.append((ok,cont[:36].replace(chr(10)," ")))
        gc.collect(); torch.cuda.empty_cache()
    return hits/len(rows), ex
print("=== H1 ELICIT: at BASH (explore) decision points, can we elicit a real `edit` emission? ===")
H1={}
r0,e0=emit_rate(BASH,TARGET,L=None,donor_fn=None); H1["baseline_edit_rate"]=r0
print(f"  NO-PATCH baseline edit-rate = {r0:.2f}  eg:",e0)
for L in GEN_LAYERS:
    rs,es=emit_rate(BASH,TARGET,L=L,donor_fn=lambda r,i,LL=L: edit_donor_for(r,i)[0][LL])
    rn,_ =emit_rate(BASH,TARGET,L=L,donor_fn=lambda r,i,LL=L: bash_donor_for(r,i)[LL])
    rc,_ =emit_rate(BASH,TARGET,L=L,donor_fn=lambda r,i,LL=L: crosstask_edit_donor_for(r,i)[LL])
    H1[f"editdonor_L{L}"]=rs; H1[f"bashnull_L{L}"]=rn; H1[f"crosstask_L{L}"]=rc
    print(f"  L{L}: edit-donor {rs:.2f}  | bash-null {rn:.2f}  | cross-task {rc:.2f}   eg:",es)
print(">>> GATE: edit-donor >> baseline & bash-null & cross-task  ->  the late lever elicits a SECOND action (generalizes).")
'''))

C.append(md(r'''## 9. H2 SUPPRESS — the brake: patch a bash-donor at an EDIT point, does the commit get BLOCKED?

This is the circuit-breaker direction. Gate (pre-set): edit-emission DROPS by a meaningful margin (target the
agent instead emits bash/explore), above the neutral (edit-donor self) control.'''))
C.append(code(r'''
print("=== H2 SUPPRESS: at EDIT (commit) decision points, can we BLOCK the commit (the brake)? ===")
H2={}
rE,exE=emit_rate(EDIT,TARGET,L=None,donor_fn=None); H2["baseline_edit_rate"]=rE
print(f"  NO-PATCH baseline edit-rate = {rE:.2f}  eg:",exE)
for L in GEN_LAYERS:
    # suppress: inject a task-matched BASH(explore) donor -> hope edit-rate drops, bash-rate rises
    rsup,esup=emit_rate(EDIT,TARGET,L=L,donor_fn=lambda r,i,LL=L: bash_donor_for(r,i)[LL])
    rbash,_  =emit_rate(EDIT,ALT,   L=L,donor_fn=lambda r,i,LL=L: bash_donor_for(r,i)[LL])
    # control: inject a task-matched EDIT donor (same-class) -> should NOT suppress
    rctl,_   =emit_rate(EDIT,TARGET,L=L,donor_fn=lambda r,i,LL=L: edit_donor_for(r,i)[0][LL])
    H2[f"suppress_editrate_L{L}"]=rsup; H2[f"suppress_bashrate_L{L}"]=rbash; H2[f"ctl_editrate_L{L}"]=rctl
    print(f"  L{L}: edit-rate {rE:.2f}->{rsup:.2f} (brake)  | bash-rate now {rbash:.2f}  | edit-donor ctl {rctl:.2f}")
print(">>> GATE: bash-donor drops edit-rate well below baseline & below edit-donor ctl, and raises bash-rate -> a usable BRAKE.")
'''))

C.append(md("## 10. H3 + dense ΔP shape + save + verdict"))
C.append(code(r'''
print("=== dense one-token ΔP(edit) at BASH points: edit-donor vs bash-null (lever-layer profile) ===")
dPe={}; dPn={}
ALLP=sorted(set(PATCH_MID+PATCH_LATE))
for L in ALLP:
    e=np.mean([patch_one_token(r["ids"],L,edit_donor_for(r,i)[0][L])-r["baseP_edit"] for i,r in enumerate(BASH)])
    n=np.mean([patch_one_token(r["ids"],L,bash_donor_for(r,i)[L])     -r["baseP_edit"] for i,r in enumerate(BASH)])
    dPe[L]=float(e); dPn[L]=float(n); print(f"  L{L:2d}: edit-donor {e:+.3f}  bash-null {n:+.3f}")
    gc.collect(); torch.cuda.empty_cache()
lever_layer=max(ALLP, key=lambda L: dPe[L]-dPn[L])
print(f"\n  edit lever-layer (max donor-specific ΔP) = L{lever_layer}   |  finish lever (paper #6) = L51-63")
import json
out={"fidelity":{"edit_pts_baseP_edit":float(np.mean([r['baseP_edit'] for r in EDIT])),
                 "bash_pts_baseP_edit":float(np.mean([r['baseP_edit'] for r in BASH])),
                 "edit_pts_meanturn":float(np.mean([r['turn'] for r in EDIT])),
                 "bash_pts_meanturn":float(np.mean([r['turn'] for r in BASH]))},
     "locate":{"edit":edit_lens,"bash":bash_lens},
     "H1_elicit":H1,"H2_suppress":H2,
     "dense_dP":{"edit_donor":dPe,"bash_null":dPn},"edit_lever_layer":lever_layer,
     "config":{"N_EDIT":len(EDIT),"N_BASH":len(BASH),"GEN_LAYERS":GEN_LAYERS,"MAXLEN":MAXLEN,"MAXNEW":MAXNEW}}
json.dump(out, open(os.path.join(OUT,"commit_lever_results.json"),"w"), indent=1)
print("saved", os.path.join(OUT,"commit_lever_results.json"))
# DURABILITY: /content is ephemeral — echo the full result so the verdict survives in cell output, and
# try a Drive backup if mounted. ALWAYS copy this block out of the notebook before the machine recycles.
print("\n===== FULL RESULTS JSON (copy this — survives a machine stop) =====")
print(json.dumps(out, indent=1))
try:
    if os.path.isdir("/content/drive/MyDrive"):
        import shutil; shutil.copy(os.path.join(OUT,"commit_lever_results.json"),
                                   "/content/drive/MyDrive/commit_lever_results.json")
        print("also backed up to Drive: /content/drive/MyDrive/commit_lever_results.json")
except Exception as e:
    print("drive backup skipped:", str(e)[:80])
print("\n--- VERDICT CHECKLIST (PREREG_commitment_lever.md) ---")
print("[fidelity] baseP(edit) edit-pts >> bash-pts ?              -> states reproduce real propensity")
print("[LOCATE]   edit gap flat early, rises in late block ?      -> geometry matches finish")
print("[H1]       edit-donor elicits real edit >> bash-null & XT ?-> the late lever GENERALIZES (general law)")
print("[H2]       bash-donor BLOCKS the commit (brake) ?          -> the circuit-breaker mechanism works")
print("[H3]       edit lever-layer ~ L51-63 ?                     -> single action-commitment surface")
print("ALL GO -> circuit-breaker greenlit (Tier-2 crypto). H1/H2 null -> termination is mechanistically special.")
'''))

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","name":"python3"},
    "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out=Path(__file__).resolve().parent.parent/"notebooks"/"nb_commitment_lever.ipynb"
out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(nb,indent=1))
print("wrote",out,"with",len(C),"cells")
