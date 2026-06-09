#!/usr/bin/env python3
"""Standalone, connection-drop-proof runner for the commitment-lever Tier-1 experiment.

Same experiment as notebooks/nb_commitment_lever.ipynb (PREREG_commitment_lever.md), but as a
single server-side process meant to be launched with nohup on a Colab VM:

    nohup python run_commitment_lever.py > /content/run.log 2>&1 &

Durability: progress prints flush to the log; the final JSON is saved to /content AND uploaded
to the HF dataset repo (results/commit_lever_results.json) so a VM death cannot eat the result.
Requires HF_TOKEN in the environment (for the upload; model+data are public).
"""
import os, sys, json, csv, glob, math, gc, tarfile, time

sys.path.insert(0, "/content/openinterp-swebench-harness")
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from pathlib import Path
from agent.prompts import SYSTEM_PROMPT, render_problem
from agent.tools import TOOLS
from agent.parser import _strip_think

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

MODEL_ID = "Qwen/Qwen3.6-27B"
DATA_REPO = "caiovicentino1/swebench-phase6-verdict-circuit"
OUT = "/content/commit_lever_out"; os.makedirs(OUT, exist_ok=True)
TOOL_NAMES = ["bash", "str_replace_editor", "finish"]
ACTION_TOK = {"finish": 28, "bash": 21402, "str_replace_editor": 15462}
PREFIX = "<function"
TARGET, ALT = "str_replace_editor", "bash"
MAXLEN = 4000
N_EDIT = N_BASH = 60
SWEEP = list(range(3, 64, 4))
PATCH_LATE = [48, 52, 55, 59, 63]
PATCH_MID = [23, 31]
GEN_LAYERS = [55, 59]
MAXNEW = 16

log("loading model", MODEL_ID)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto",
                                             trust_remote_code=True).eval()
log("model loaded")

def get_layer_module(idx):
    for path in ("model.language_model.layers", "language_model.layers", "model.model.layers", "model.layers"):
        cur = model; ok = True
        for p in path.split("."):
            if not hasattr(cur, p): ok = False; break
            cur = getattr(cur, p)
        if ok:
            try: return cur[idx]
            except Exception: continue
    raise RuntimeError("layer not found")

def p_action(nl, name=TARGET):
    ids = [ACTION_TOK[n] for n in TOOL_NAMES]; p = torch.softmax(nl[ids].float(), -1)
    return float(p[TOOL_NAMES.index(name)])

def fwd_layers(ids, layers):
    caps = {}; handles = []
    def mk(L):
        def h(m, i, o): caps[L] = (o[0] if isinstance(o, tuple) else o)[:, -1:, :].detach()
        return h
    for L in layers: handles.append(get_layer_module(L).register_forward_hook(mk(L)))
    try:
        with torch.no_grad():
            out = model(input_ids=ids.to(model.device), use_cache=False, logits_to_keep=1)
    finally:
        for h in handles: h.remove()
    return out.logits, caps

def _patch_hook(donor):
    def h(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o; hs = hs.clone(); hs[:, -1, :] = donor.to(hs.dtype)
        return (hs, *o[1:]) if isinstance(o, tuple) else hs
    return h

def patch_one_token(ids, L, donor, name=TARGET):
    hh = get_layer_module(L).register_forward_hook(_patch_hook(donor))
    try:
        with torch.no_grad():
            lg = model(input_ids=ids.to(model.device), use_cache=False, logits_to_keep=1).logits
    finally:
        hh.remove()
    r = p_action(lg[0, -1], name); del lg; return r

def gen_after_function(ids, L=None, donor=None, maxnew=MAXNEW):
    """PREFILL-ONLY patch (decision-locator canonical): inject at the decision position on the
    prefill pass only, then decode free. Every-step patching degenerates (=str=str...)."""
    hh = None
    if L is not None and donor is not None:
        def _ph(m, i, o):
            hs = o[0] if isinstance(o, tuple) else o
            if hs.shape[1] > 1:
                hs = hs.clone(); hs[:, -1, :] = donor.to(hs.dtype)
                return (hs, *o[1:]) if isinstance(o, tuple) else hs
            return o
        hh = get_layer_module(L).register_forward_hook(_ph)
    try:
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(model.device), max_new_tokens=maxnew, do_sample=False,
                                 use_cache=True, pad_token_id=tok.eos_token_id,
                                 attention_mask=torch.ones_like(ids).to(model.device))
    finally:
        if hh is not None: hh.remove()
    cont = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False); del out; return cont

def emits(cont, name):
    s = cont.lstrip()
    if s.startswith("="): s = s[1:]
    s = s.lstrip().lower()
    return s.startswith(name.lower()[:6]) or (name.lower()[:6] in cont[:24].lower())

def final_norm():
    for path in ("model.language_model.norm", "language_model.norm", "model.model.norm", "model.norm"):
        cur = model; ok = True
        for p in path.split("."):
            if not hasattr(cur, p): ok = False; break
            cur = getattr(cur, p)
        if ok: return cur
    return None

NORM = final_norm(); LMH = model.get_output_embeddings()

def lens_gap(resid_vec, name=TARGET):
    with torch.no_grad():
        h = resid_vec.to(model.device).to(next(model.parameters()).dtype).unsqueeze(0)
        logits = LMH(NORM(h))[0].float()
    tid = ACTION_TOK[name]; alts = [ACTION_TOK[n] for n in TOOL_NAMES if n != name]
    return float(logits[tid] - logits[alts].mean())

# ---------------- data
log("downloading data bundle")
os.makedirs("/content/vc_data", exist_ok=True)
with tarfile.open(hf_hub_download(DATA_REPO, "traces.tar.gz", repo_type="dataset")) as t:
    t.extractall("/content/vc_data")
TRACES = {Path(p).stem: p for p in glob.glob("/content/vc_data/traces/instance_*.json")}
LAB = {r["iid"]: r["sub_class"] for r in csv.DictReader(open(hf_hub_download(DATA_REPO, "features_n99.csv", repo_type="dataset")))}
ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
by_iid = {r["instance_id"]: r for r in ds if r.get("instance_id")}
log(f"traces {len(TRACES)} labels {len(LAB)} dataset {len(ds)}")

def instance_for(iid):
    if iid in by_iid: return by_iid[iid]
    return next((r for k, r in by_iid.items() if k in iid or iid in k), None)

def repo_of(iid): return iid.split("__")[0] if "__" in iid else iid.split("-")[0]
def chosen_tool(turn):
    tcs = turn.get("tool_calls") or []
    return (tcs[0].get("name") if tcs else None)

def build_messages_upto(trace, instance, k):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_problem({**instance, "__workdir__": f"/content/work_p6/{trace['instance_id']}"})}]
    for tn in trace["turns"][:k]:
        _, body = _strip_think(tn.get("raw_response") or "")
        msgs.append({"role": "assistant", "content": body})
        for tr_ in (tn.get("tool_results") or []):
            msgs.append({"role": "tool", "content": json.dumps(tr_.get("result"), ensure_ascii=False)[:32000]})
        if not tn.get("tool_calls"):
            msgs.append({"role": "user", "content": "You did not call a tool. Use bash, str_replace_editor, or finish."})
    return msgs

def choice_point_at_turn(trace, instance, k):
    msgs = build_messages_upto(trace, instance, k)
    prefix = tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False)
    raw = trace["turns"][k].get("raw_response") or ""
    cut = raw.find(PREFIX)
    if cut < 0: return None
    ids = tok(prefix + raw[:cut + len(PREFIX)], add_special_tokens=False).input_ids
    if len(ids) > MAXLEN: ids = ids[-MAXLEN:]
    return torch.tensor([ids])

def collect_points(target_name, n, max_per_traj=3):
    out = []
    for iid, tp in TRACES.items():
        tr = json.load(open(tp)); inst = instance_for(iid)
        if inst is None: continue
        taken = 0
        for k, turn in enumerate(tr["turns"]):
            if k == 0: continue
            if chosen_tool(turn) != target_name: continue
            ids = choice_point_at_turn(tr, inst, k)
            if ids is None: continue
            lg, _ = fwd_layers(ids, [])
            out.append({"iid": iid, "repo": repo_of(iid), "turn": k, "nturns": len(tr["turns"]),
                        "ids": ids.cpu(), "baseP_edit": p_action(lg[0, -1], TARGET), "baseP_alt": p_action(lg[0, -1], ALT)})
            taken += 1; gc.collect(); torch.cuda.empty_cache()
            if len(out) % 10 == 0: log(f"  {target_name} points: {len(out)}")
            if taken >= max_per_traj: break
            if len(out) >= n: break
        if len(out) >= n: break
    return out

log("collecting EDIT decision points")
EDIT = collect_points(TARGET, N_EDIT)
log("collecting BASH decision points")
BASH = collect_points(ALT, N_BASH)
fid = {"edit_pts_baseP_edit": float(np.mean([r["baseP_edit"] for r in EDIT])),
       "bash_pts_baseP_edit": float(np.mean([r["baseP_edit"] for r in BASH])),
       "edit_pts_meanturn": float(np.mean([r["turn"] for r in EDIT])),
       "bash_pts_meanturn": float(np.mean([r["turn"] for r in BASH]))}
log("FIDELITY:", json.dumps(fid))

log("LOCATE sweep")
def lens_profile(rows, layers):
    prof = {L: [] for L in layers}
    for r in rows:
        _, caps = fwd_layers(r["ids"], layers)
        for L in layers: prof[L].append(lens_gap(caps[L][0, 0], TARGET))
        gc.collect(); torch.cuda.empty_cache()
    return {L: float(np.mean(v)) for L, v in prof.items()}
edit_lens = lens_profile(EDIT, SWEEP)
bash_lens = lens_profile(BASH, SWEEP)
for L in SWEEP: log(f"  L{L:2d}: edit {edit_lens[L]:+.3f}  bash {bash_lens[L]:+.3f}")

log("building donors")
LAYERS_NEEDED = sorted(set(PATCH_LATE + PATCH_MID + GEN_LAYERS))
def per_point_resids(rows):
    acc = {L: [] for L in LAYERS_NEEDED}
    for r in rows:
        _, caps = fwd_layers(r["ids"], LAYERS_NEEDED)
        for L in LAYERS_NEEDED: acc[L].append(caps[L][0, 0].float().cpu())
        gc.collect(); torch.cuda.empty_cache()
    return acc
E_res = per_point_resids(EDIT); B_res = per_point_resids(BASH)
def edit_donor_for(brow, i):
    same = [j for j, er in enumerate(EDIT) if er["repo"] == brow["repo"]]
    j = (same[0] if same else i % len(EDIT))
    return {L: E_res[L][j] for L in LAYERS_NEEDED}
def crosstask_edit_donor_for(brow, i):
    diff = [j for j, er in enumerate(EDIT) if er["repo"] != brow["repo"]]
    j = (diff[i % len(diff)] if diff else i % len(EDIT))
    return {L: E_res[L][j] for L in LAYERS_NEEDED}
def bash_donor_for(erow, i):
    same = [j for j, br in enumerate(BASH) if br["repo"] == erow["repo"]]
    j = (same[0] if same else i % len(BASH))
    return {L: B_res[L][j] for L in LAYERS_NEEDED}

def emit_rate(rows, name, L=None, donor_fn=None, tag=""):
    hits = 0; ex = []
    for i, r in enumerate(rows):
        donor = None if donor_fn is None else donor_fn(r, i, L)
        cont = gen_after_function(r["ids"], L=(None if donor is None else L), donor=donor)
        ok = emits(cont, name); hits += int(ok)
        if len(ex) < 3: ex.append((ok, cont[:36].replace(chr(10), " ")))
        gc.collect(); torch.cuda.empty_cache()
        if (i + 1) % 20 == 0: log(f"    {tag} {i+1}/{len(rows)} rate so far {hits/(i+1):.2f}")
    return hits / len(rows), ex

log("H1 ELICIT (at BASH points)")
H1 = {}
r0, e0 = emit_rate(BASH, TARGET, tag="baseline"); H1["baseline_edit_rate"] = r0
log(f"  baseline edit-rate {r0:.2f} eg {e0}")
for L in GEN_LAYERS:
    rs, es = emit_rate(BASH, TARGET, L=L, donor_fn=lambda r, i, LL=L: edit_donor_for(r, i)[LL], tag=f"edit-donor L{L}")
    rn, _ = emit_rate(BASH, TARGET, L=L, donor_fn=lambda r, i, LL=L: bash_donor_for(r, i)[LL], tag=f"bash-null L{L}")
    rc, _ = emit_rate(BASH, TARGET, L=L, donor_fn=lambda r, i, LL=L: crosstask_edit_donor_for(r, i)[LL], tag=f"cross-task L{L}")
    H1[f"editdonor_L{L}"] = rs; H1[f"bashnull_L{L}"] = rn; H1[f"crosstask_L{L}"] = rc
    log(f"  L{L}: edit-donor {rs:.2f} | bash-null {rn:.2f} | cross-task {rc:.2f} eg {es}")

log("H2 SUPPRESS (at EDIT points) — the brake")
H2 = {}
rE, exE = emit_rate(EDIT, TARGET, tag="baseline-edit"); H2["baseline_edit_rate"] = rE
log(f"  baseline edit-rate {rE:.2f} eg {exE}")
for L in GEN_LAYERS:
    rsup, _ = emit_rate(EDIT, TARGET, L=L, donor_fn=lambda r, i, LL=L: bash_donor_for(r, i)[LL], tag=f"suppress L{L}")
    rbash, _ = emit_rate(EDIT, ALT, L=L, donor_fn=lambda r, i, LL=L: bash_donor_for(r, i)[LL], tag=f"suppress-bashrate L{L}")
    rctl, _ = emit_rate(EDIT, TARGET, L=L, donor_fn=lambda r, i, LL=L: edit_donor_for(r, i)[LL], tag=f"ctl L{L}")
    H2[f"suppress_editrate_L{L}"] = rsup; H2[f"suppress_bashrate_L{L}"] = rbash; H2[f"ctl_editrate_L{L}"] = rctl
    log(f"  L{L}: edit-rate {rE:.2f}->{rsup:.2f} (brake) | bash-rate {rbash:.2f} | edit-donor ctl {rctl:.2f}")

log("dense one-token profile")
dPe = {}; dPn = {}
ALLP = sorted(set(PATCH_MID + PATCH_LATE))
for L in ALLP:
    e = np.mean([patch_one_token(r["ids"], L, edit_donor_for(r, i)[L]) - r["baseP_edit"] for i, r in enumerate(BASH)])
    n = np.mean([patch_one_token(r["ids"], L, bash_donor_for(r, i)[L]) - r["baseP_edit"] for i, r in enumerate(BASH)])
    dPe[L] = float(e); dPn[L] = float(n); log(f"  L{L:2d}: edit-donor {e:+.3f}  bash-null {n:+.3f}")
    gc.collect(); torch.cuda.empty_cache()
lever_layer = max(ALLP, key=lambda L: dPe[L] - dPn[L])

out = {"fidelity": fid, "locate": {"edit": edit_lens, "bash": bash_lens},
       "H1_elicit": H1, "H2_suppress": H2,
       "dense_dP": {"edit_donor": dPe, "bash_null": dPn}, "edit_lever_layer": lever_layer,
       "config": {"N_EDIT": len(EDIT), "N_BASH": len(BASH), "GEN_LAYERS": GEN_LAYERS, "MAXLEN": MAXLEN, "MAXNEW": MAXNEW}}
path = os.path.join(OUT, "commit_lever_results.json")
json.dump(out, open(path, "w"), indent=1)
log("saved", path)
print("OILAB_JSON_BEGIN"); print(json.dumps(out)); print("OILAB_JSON_END", flush=True)

# durable off-VM copy
try:
    from huggingface_hub import upload_file
    upload_file(path_or_fileobj=path, path_in_repo="results/commit_lever_results.json",
                repo_id=DATA_REPO, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    log("uploaded results to HF:", DATA_REPO, "results/commit_lever_results.json")
except Exception as e:
    log("HF upload failed (results still in log above):", str(e)[:200])
log("RUN_DONE")
