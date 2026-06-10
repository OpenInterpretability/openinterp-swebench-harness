"""Stage library for the commitment-lever Tier-1 — cell-by-cell, verifiable, RESUMABLE.

Driven one short exec at a time (never fire-and-forget):

    import sys; sys.path.insert(0,'/content/openinterp-swebench-harness/scripts')
    import commit_lever_stages as S
    S.setup()                  # load model (~13 min on G4)
    S.capture()                # points + fidelity + residuals -> uploads state to HF (or resumes)
    S.block('h1_baseline')     # ONE emit-rate block (~6 min) -> checkpoint to HF
    S.block('h1_editdonor_L55') ... S.dense() ... S.finalize()

Every boundary persists to the HF dataset repo, so a VM death resumes instead of restarts:
  results/commit_lever_state.pt      (decision-point ids + residuals — capture stage)
  results/commit_lever_partial.json  (fidelity/locate/blocks ledger)
Deterministic: trace files are SORTED, so decision points are identical across VMs.
"""
import os, json, csv, glob, gc, tarfile, time, io
import torch, numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download, upload_file

DATA_REPO = "caiovicentino1/swebench-phase6-verdict-circuit"
MODEL_ID = "Qwen/Qwen3.6-27B"
TOOL_NAMES = ["bash", "str_replace_editor", "finish"]
ACTION_TOK = {"finish": 28, "bash": 21402, "str_replace_editor": 15462}
PREFIX = "<function"; TARGET, ALT = "str_replace_editor", "bash"
MAXLEN = 4000; N_EDIT = N_BASH = 60; MAXNEW = 16
GEN_LAYERS = [55, 59]; PATCH_MID = [23, 31]; PATCH_LATE = [48, 52, 55, 59, 63]
LAYERS_NEEDED = sorted(set(PATCH_LATE + PATCH_MID + GEN_LAYERS))

S = {}   # session state: model, tok, EDIT, BASH, E_res, B_res
P = {}   # partial results ledger (mirrors HF results/commit_lever_partial.json)

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def _tok(): return os.environ.get("HF_TOKEN") or None

def save_partial():
    json.dump(P, open("/content/clp.json", "w"), indent=1)
    upload_file(path_or_fileobj="/content/clp.json", path_in_repo="results/commit_lever_partial.json",
                repo_id=DATA_REPO, repo_type="dataset", token=_tok())

def load_partial():
    global P
    try:
        p = hf_hub_download(DATA_REPO, "results/commit_lever_partial.json", repo_type="dataset",
                            token=_tok(), force_download=True)
        P = json.load(open(p)); log("resumed partial ledger:", sorted(P.keys()))
    except Exception:
        P = {}; log("no partial ledger on HF — fresh start")

# ---------------------------------------------------------------- setup (model)
def setup():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("loading", MODEL_ID)
    S["tok"] = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    S["model"] = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                      device_map="auto", trust_remote_code=True).eval()
    load_partial()
    log("SETUP_OK")

def _layer(idx):
    m = S["model"]
    for path in ("model.language_model.layers", "language_model.layers", "model.model.layers", "model.layers"):
        cur = m; ok = True
        for p in path.split("."):
            if not hasattr(cur, p): ok = False; break
            cur = getattr(cur, p)
        if ok:
            try: return cur[idx]
            except Exception: continue
    raise RuntimeError("layer not found")

def _p_action(nl, name=TARGET):
    ids = [ACTION_TOK[n] for n in TOOL_NAMES]; p = torch.softmax(nl[ids].float(), -1)
    return float(p[TOOL_NAMES.index(name)])

def _fwd(ids, layers):
    caps = {}; hs = []
    def mk(L):
        def h(m, i, o): caps[L] = (o[0] if isinstance(o, tuple) else o)[:, -1:, :].detach()
        return h
    for L in layers: hs.append(_layer(L).register_forward_hook(mk(L)))
    try:
        with torch.no_grad():
            out = S["model"](input_ids=ids.to(S["model"].device), use_cache=False, logits_to_keep=1)
    finally:
        for h in hs: h.remove()
    return out.logits, caps

def _gen(ids, L=None, donor=None):
    tok, model = S["tok"], S["model"]
    hh = None
    if L is not None and donor is not None:
        def _ph(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            if h.shape[1] > 1:
                h = h.clone(); h[:, -1, :] = donor.to(h.dtype)
                return (h, *o[1:]) if isinstance(o, tuple) else h
            return o
        hh = _layer(L).register_forward_hook(_ph)
    try:
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(model.device), max_new_tokens=MAXNEW, do_sample=False,
                                 use_cache=True, pad_token_id=tok.eos_token_id,
                                 attention_mask=torch.ones_like(ids).to(model.device))
    finally:
        if hh is not None: hh.remove()
    cont = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False); del out; return cont

def _emits(cont, name):
    s = cont.lstrip()
    if s.startswith("="): s = s[1:]
    return s.lstrip().lower().startswith(name.lower()[:6]) or (name.lower()[:6] in cont[:24].lower())

# ---------------------------------------------------------------- capture (resumable)
def capture():
    tok = S["tok"]
    # resume path: state already on HF?
    try:
        p = hf_hub_download(DATA_REPO, "results/commit_lever_state.pt", repo_type="dataset",
                            token=_tok(), force_download=True)
        st = torch.load(p, map_location="cpu", weights_only=False)
        S.update(EDIT=st["EDIT"], BASH=st["BASH"], E_res=st["E_res"], B_res=st["B_res"])
        log(f"CAPTURE_RESUMED from HF: edit={len(S['EDIT'])} bash={len(S['BASH'])}")
        return
    except Exception:
        log("no state on HF — capturing fresh (deterministic order)")
    import sys
    sys.path.insert(0, "/content/openinterp-swebench-harness")
    from agent.prompts import SYSTEM_PROMPT, render_problem
    from agent.tools import TOOLS
    from agent.parser import _strip_think
    from datasets import load_dataset
    os.makedirs("/content/vc_data", exist_ok=True)
    with tarfile.open(hf_hub_download(DATA_REPO, "traces.tar.gz", repo_type="dataset")) as t:
        t.extractall("/content/vc_data")
    traces = sorted(glob.glob("/content/vc_data/traces/instance_*.json"))   # SORTED = deterministic
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    by_iid = {r["instance_id"]: r for r in ds if r.get("instance_id")}
    def inst(iid):
        if iid in by_iid: return by_iid[iid]
        return next((r for k, r in by_iid.items() if k in iid or iid in k), None)
    def msgs_upto(tr, instance, k):
        ms = [{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": render_problem({**instance, "__workdir__": f"/content/work_p6/{tr['instance_id']}"})}]
        for tn in tr["turns"][:k]:
            _, body = _strip_think(tn.get("raw_response") or "")
            ms.append({"role": "assistant", "content": body})
            for r_ in (tn.get("tool_results") or []):
                ms.append({"role": "tool", "content": json.dumps(r_.get("result"), ensure_ascii=False)[:32000]})
            if not tn.get("tool_calls"):
                ms.append({"role": "user", "content": "You did not call a tool. Use bash, str_replace_editor, or finish."})
        return ms
    def point(tr, instance, k):
        pre = tok.apply_chat_template(msgs_upto(tr, instance, k), tools=TOOLS, add_generation_prompt=True, tokenize=False)
        raw = tr["turns"][k].get("raw_response") or ""
        cut = raw.find(PREFIX)
        if cut < 0: return None
        ids = tok(pre + raw[:cut + len(PREFIX)], add_special_tokens=False).input_ids
        return torch.tensor([ids[-MAXLEN:]])
    def collect(name, n, mpt=3):
        out = []
        for tp in traces:
            tr = json.load(open(tp)); ii = inst(tr.get("instance_id") or Path(tp).stem)
            if ii is None: continue
            taken = 0
            for k, turn in enumerate(tr["turns"]):
                if k == 0: continue
                tcs = turn.get("tool_calls") or []
                if not tcs or tcs[0].get("name") != name: continue
                ids = point(tr, ii, k)
                if ids is None: continue
                lg, _ = _fwd(ids, [])
                iid = tr.get("instance_id") or Path(tp).stem
                out.append({"iid": iid, "repo": iid.split("__")[0] if "__" in iid else iid.split("-")[0],
                            "turn": k, "ids": ids.cpu(), "baseP_edit": _p_action(lg[0, -1], TARGET)})
                taken += 1; gc.collect(); torch.cuda.empty_cache()
                if len(out) % 20 == 0: log(f"  {name}: {len(out)}")
                if taken >= mpt or len(out) >= n: break
            if len(out) >= n: break
        return out
    S["EDIT"] = collect(TARGET, N_EDIT); S["BASH"] = collect(ALT, N_BASH)
    fid = {"edit_pts_baseP_edit": float(np.mean([r["baseP_edit"] for r in S["EDIT"]])),
           "bash_pts_baseP_edit": float(np.mean([r["baseP_edit"] for r in S["BASH"]])),
           "edit_meanturn": float(np.mean([r["turn"] for r in S["EDIT"]])),
           "bash_meanturn": float(np.mean([r["turn"] for r in S["BASH"]]))}
    P["fidelity"] = fid; log("FIDELITY:", json.dumps(fid))
    def resids(rows):
        acc = {L: [] for L in LAYERS_NEEDED}
        for r in rows:
            _, caps = _fwd(r["ids"], LAYERS_NEEDED)
            for L in LAYERS_NEEDED: acc[L].append(caps[L][0, 0].float().cpu())
            gc.collect(); torch.cuda.empty_cache()
        return acc
    S["E_res"] = resids(S["EDIT"]); S["B_res"] = resids(S["BASH"])
    torch.save({"EDIT": S["EDIT"], "BASH": S["BASH"], "E_res": S["E_res"], "B_res": S["B_res"]},
               "/content/state.pt")
    upload_file(path_or_fileobj="/content/state.pt", path_in_repo="results/commit_lever_state.pt",
                repo_id=DATA_REPO, repo_type="dataset", token=_tok())
    save_partial()
    log("CAPTURE_OK — state uploaded to HF")

# ---------------------------------------------------------------- donors
def _edit_donor(brow, i, L):
    same = [j for j, er in enumerate(S["EDIT"]) if er["repo"] == brow["repo"]]
    j = (same[0] if same else i % len(S["EDIT"])); return S["E_res"][L][j]
def _xt_donor(brow, i, L):
    diff = [j for j, er in enumerate(S["EDIT"]) if er["repo"] != brow["repo"]]
    j = (diff[i % len(diff)] if diff else i % len(S["EDIT"])); return S["E_res"][L][j]
def _bash_donor(erow, i, L):
    same = [j for j, br in enumerate(S["BASH"]) if br["repo"] == erow["repo"]]
    j = (same[0] if same else i % len(S["BASH"])); return S["B_res"][L][j]

BLOCKS = {
    # name: (rows, emitted-action, layer, donor_fn or None)
    "h1_baseline":        ("BASH", TARGET, None, None),
    "h1_editdonor_L55":   ("BASH", TARGET, 55, _edit_donor),
    "h1_bashnull_L55":    ("BASH", TARGET, 55, _bash_donor),
    "h1_crosstask_L55":   ("BASH", TARGET, 55, _xt_donor),
    "h1_editdonor_L59":   ("BASH", TARGET, 59, _edit_donor),
    "h1_bashnull_L59":    ("BASH", TARGET, 59, _bash_donor),
    "h1_crosstask_L59":   ("BASH", TARGET, 59, _xt_donor),
    "h2_baseline":        ("EDIT", TARGET, None, None),
    "h2_suppress_L55":    ("EDIT", TARGET, 55, _bash_donor),
    "h2_suppress_bashrate_L55": ("EDIT", ALT, 55, _bash_donor),
    "h2_ctl_L55":         ("EDIT", TARGET, 55, _edit_donor),
    "h2_suppress_L59":    ("EDIT", TARGET, 59, _bash_donor),
    "h2_suppress_bashrate_L59": ("EDIT", ALT, 59, _bash_donor),
    "h2_ctl_L59":         ("EDIT", TARGET, 59, _edit_donor),
}

def block(name):
    if P.get("blocks", {}).get(name) is not None:
        log(f"BLOCK_SKIP {name} (already done: {P['blocks'][name]['rate']:.2f})"); return
    rows_key, action, L, donor_fn = BLOCKS[name]
    rows = S[rows_key]; hits = 0; ex = []
    for i, r in enumerate(rows):
        donor = None if donor_fn is None else donor_fn(r, i, L)
        cont = _gen(r["ids"], L=(None if donor is None else L), donor=donor)
        ok = _emits(cont, action); hits += int(ok)
        if len(ex) < 2: ex.append(cont[:30].replace(chr(10), " "))
        gc.collect(); torch.cuda.empty_cache()
        if (i + 1) % 20 == 0: log(f"  {name} {i+1}/{len(rows)} rate {hits/(i+1):.2f}")
    rate = hits / len(rows)
    P.setdefault("blocks", {})[name] = {"rate": rate, "n": len(rows), "eg": ex}
    save_partial()
    log(f"BLOCK_OK {name}: rate={rate:.2f} (n={len(rows)}) — checkpointed to HF")

def dense():
    if P.get("dense_dP"): log("DENSE_SKIP (done)"); return
    dPe, dPn = {}, {}
    for L in sorted(set(PATCH_MID + PATCH_LATE)):
        e, n = [], []
        for i, r in enumerate(S["BASH"]):
            hh = _layer(L).register_forward_hook(_patch := _mkpatch(_edit_donor(r, i, L)))
            try:
                with torch.no_grad():
                    lg = S["model"](input_ids=r["ids"].to(S["model"].device), use_cache=False, logits_to_keep=1).logits
            finally: hh.remove()
            e.append(_p_action(lg[0, -1]) - r["baseP_edit"])
            hh = _layer(L).register_forward_hook(_mkpatch(_bash_donor(r, i, L)))
            try:
                with torch.no_grad():
                    lg = S["model"](input_ids=r["ids"].to(S["model"].device), use_cache=False, logits_to_keep=1).logits
            finally: hh.remove()
            n.append(_p_action(lg[0, -1]) - r["baseP_edit"])
            gc.collect(); torch.cuda.empty_cache()
        dPe[L] = float(np.mean(e)); dPn[L] = float(np.mean(n))
        log(f"  L{L:2d}: edit-donor {dPe[L]:+.3f}  bash-null {dPn[L]:+.3f}")
    P["dense_dP"] = {"edit_donor": dPe, "bash_null": dPn}
    save_partial(); log("DENSE_OK")

def _mkpatch(donor):
    def h(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o; hs = hs.clone(); hs[:, -1, :] = donor.to(hs.dtype)
        return (hs, *o[1:]) if isinstance(o, tuple) else hs
    return h

def finalize():
    upload_file(path_or_fileobj="/content/clp.json", path_in_repo="results/commit_lever_results.json",
                repo_id=DATA_REPO, repo_type="dataset", token=_tok())
    log("FINALIZED — results/commit_lever_results.json on HF")
    print("OILAB_JSON_BEGIN"); print(json.dumps(P)); print("OILAB_JSON_END", flush=True)
