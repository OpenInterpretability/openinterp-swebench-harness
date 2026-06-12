"""Attn-vs-MLP decomposition of the late commitment-lever (paper #7 follow-up). RESUMABLE.

Reuses scripts/commit_lever_stages.py verbatim (model, EDIT/BASH points, donors, P(action) metric),
so every number is comparable to the published H1/H2/dense panels. The published lever overwrites the
last-token residual OUTPUT y_L of a layer L with a donor's y_L. Since (pre-norm block)
    y_L = x_L + a_L + m_L      (x=input residual, a=attn write, m=mlp write)
the published delta Δ_full = y_donor - y_point = Δx + Δa + Δm is EXACTLY additive. We re-inject each
piece alone with an ADDITIVE hook on layer-L output (y += δ, last token, prefill only) and measure which
sublayer reproduces the elicit (0.23->0.77) and the brake (0.48->0.02).

Drive ONE exec at a time (survives VM death; every block checkpoints to HF):

    import sys; sys.path.insert(0,'/content/openinterp-swebench-harness/scripts')
    import commit_lever_stages as S, commit_lever_decomp as D
    S.setup(); S.capture()        # base state (model + EDIT/BASH points + y-residuals) — reuses #7 HF state
    D.load_d()                    # resume decomp ledger
    D.capture_sublayers()         # x_L, a_L, m_L per row at L in {55,59}  (~3 min) -> HF
    D.run_dp()                    # all ΔP blocks (resumable, per-block HF checkpoint)
    D.run_emit()                  # emit-rate confirm at headline layers
    D.finalize()                  # prints OILAB_JSON
"""
import os, json, time
import torch, numpy as np
from huggingface_hub import hf_hub_download, upload_file
import commit_lever_stages as S

Ls = [55, 59]
TARGET, ALT = S.TARGET, S.ALT
DREPO = S.DATA_REPO
DEC_FILE = "results/commit_lever_decomp.json"
SUB_FILE = "results/commit_lever_sublayers.pt"
D = {}

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def _tok(): return os.environ.get("HF_TOKEN") or None

def save_d():
    json.dump(D, open("/content/cld.json", "w"))
    upload_file(path_or_fileobj="/content/cld.json", path_in_repo=DEC_FILE,
                repo_id=DREPO, repo_type="dataset", token=_tok())

def load_d():
    global D
    try:
        p = hf_hub_download(DREPO, DEC_FILE, repo_type="dataset", token=_tok(), force_download=True)
        D = json.load(open(p)); log("resumed decomp ledger:", sorted(D.keys()))
    except Exception:
        D = {}; log("no decomp ledger on HF — fresh")

# ----------------------------------------------------- sublayer capture (x,a,m)
def capture_sublayers():
    if "E_res" not in S.S:
        raise RuntimeError("call S.capture() first (need base state)")
    try:
        p = hf_hub_download(DREPO, SUB_FILE, repo_type="dataset", token=_tok(), force_download=True)
        st = torch.load(p, map_location="cpu", weights_only=False)
        S.S.update(st); log("SUBLAYERS_RESUMED from HF"); return
    except Exception:
        log("capturing sublayers fresh")
    dev = S.S["model"].device

    def cap(rows, tag):
        X = {L: [] for L in Ls}; A = {L: [] for L in Ls}; M = {L: [] for L in Ls}
        for n, r in enumerate(rows):
            box = {}; handles = []
            for L in Ls:
                lyr = S._layer(L)
                def mkpre(L):
                    def pre(m, args, kwargs):
                        h = kwargs.get("hidden_states", args[0] if args else None)
                        box[("x", L)] = h[:, -1, :].detach().float().cpu()[0]
                    return pre
                def mka(L):
                    def hk(m, i, o):
                        h = o[0] if isinstance(o, tuple) else o
                        box[("a", L)] = h[:, -1, :].detach().float().cpu()[0]
                    return hk
                def mkm(L):
                    def hk(m, i, o):
                        h = o[0] if isinstance(o, tuple) else o
                        box[("m", L)] = h[:, -1, :].detach().float().cpu()[0]
                    return hk
                handles.append(lyr.register_forward_pre_hook(mkpre(L), with_kwargs=True))
                handles.append(lyr.self_attn.register_forward_hook(mka(L)))
                handles.append(lyr.mlp.register_forward_hook(mkm(L)))
            with torch.no_grad():
                S.S["model"](input_ids=r["ids"].to(dev), use_cache=False, logits_to_keep=1)
            for h in handles: h.remove()
            for L in Ls:
                X[L].append(box[("x", L)]); A[L].append(box[("a", L)]); M[L].append(box[("m", L)])
            if (n + 1) % 20 == 0: log(f"  sublayers {tag}: {n+1}/{len(rows)}")
        return X, A, M

    S.S["E_x"], S.S["E_a"], S.S["E_m"] = cap(S.S["EDIT"], "EDIT")
    S.S["B_x"], S.S["B_a"], S.S["B_m"] = cap(S.S["BASH"], "BASH")
    # GATE 1: residual reconstruction y ~= x+a+m
    rec = {}
    for L in Ls:
        errs = []
        for j in range(min(8, len(S.S["EDIT"]))):
            y = S.S["E_res"][L][j].float()
            r = S.S["E_x"][L][j] + S.S["E_a"][L][j] + S.S["E_m"][L][j]
            errs.append(float((y - r).norm() / (y.norm() + 1e-6)))
        rec[L] = float(np.mean(errs))
    D["recon_relerr"] = rec; log("GATE recon relerr (want <1e-2):", rec)
    torch.save({k: S.S[k] for k in ("E_x", "E_a", "E_m", "B_x", "B_a", "B_m")}, "/content/sub.pt")
    upload_file(path_or_fileobj="/content/sub.pt", path_in_repo=SUB_FILE,
                repo_id=DREPO, repo_type="dataset", token=_tok())
    save_d(); log("SUBLAYERS_OK — uploaded to HF")

# ----------------------------------------------------- patch machinery (additive)
def _mkadd(delta):
    def h(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o
        hs = hs.clone(); hs[:, -1, :] = hs[:, -1, :] + delta.to(hs.device, hs.dtype)
        return (hs, *o[1:]) if isinstance(o, tuple) else hs
    return h

def _dp(ids, L, delta):
    hh = S._layer(L).register_forward_hook(_mkadd(delta))
    try:
        with torch.no_grad():
            lg = S.S["model"](input_ids=ids.to(S.S["model"].device), use_cache=False, logits_to_keep=1).logits
    finally:
        hh.remove()
    return S._p_action(lg[0, -1], TARGET)

def _gen_add(ids, L, delta):
    model, tok = S.S["model"], S.S["tok"]
    def ph(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if h.shape[1] > 1:
            h = h.clone(); h[:, -1, :] = h[:, -1, :] + delta.to(h.device, h.dtype)
            return (h, *o[1:]) if isinstance(o, tuple) else h
        return o
    hh = S._layer(L).register_forward_hook(ph)
    try:
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(model.device), max_new_tokens=S.MAXNEW, do_sample=False,
                                 use_cache=True, pad_token_id=tok.eos_token_id,
                                 attention_mask=torch.ones_like(ids).to(model.device))
    finally:
        hh.remove()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)

# ----------------------------------------------------- donor index selection (mirrors #7)
def _edit_idx(brow, i):
    same = [j for j, er in enumerate(S.S["EDIT"]) if er["repo"] == brow["repo"]]
    return same[0] if same else i % len(S.S["EDIT"])
def _xt_edit_idx(brow, i):
    diff = [j for j, er in enumerate(S.S["EDIT"]) if er["repo"] != brow["repo"]]
    return diff[i % len(diff)] if diff else i % len(S.S["EDIT"])
def _bash_idx(erow, i):
    same = [j for j, br in enumerate(S.S["BASH"]) if br["repo"] == erow["repo"]]
    return same[0] if same else i % len(S.S["BASH"])
def _xt_bash_idx(erow, i):
    diff = [j for j, br in enumerate(S.S["BASH"]) if br["repo"] != erow["repo"]]
    return diff[i % len(diff)] if diff else i % len(S.S["BASH"])

def _delta(comp, Dpk, Ppk, L, dj, i):
    Dx, Da, Dm, Dr = Dpk; Px, Pa, Pm, Pr = Ppk
    if comp == "inp":     return Dx[L][dj] - Px[L][i]
    if comp == "attn":    return Da[L][dj] - Pa[L][i]
    if comp == "mlp":     return Dm[L][dj] - Pm[L][i]
    if comp == "full":    return Dr[L][dj] - Pr[L][i]
    if comp == "attnmlp": return (Da[L][dj] - Pa[L][i]) + (Dm[L][dj] - Pm[L][i])
    raise ValueError(comp)

def _packs(kind):  # kind in {"edit","bash"} -> (x,a,m,res) for that row-set
    if kind == "edit": return (S.S["E_x"], S.S["E_a"], S.S["E_m"], S.S["E_res"])
    return (S.S["B_x"], S.S["B_a"], S.S["B_m"], S.S["B_res"])

# direction: elicit = BASH points get EDIT donor; brake = EDIT points get BASH donor
SPECS = {
    "elicit": dict(rows="BASH", pt="bash", dn="edit", idx=_edit_idx,  ridx=_xt_edit_idx),
    "brake":  dict(rows="EDIT", pt="edit", dn="bash", idx=_bash_idx,  ridx=_xt_bash_idx),
}
COMPS = ["full", "inp", "attn", "mlp", "attnmlp"]

def _dp_block(direction, L, comp, rand=False):
    sp = SPECS[direction]
    name = f"dp_{direction}_L{L}_{comp}" + ("_rand" if rand else "")
    if name in D.get("dp", {}): log("skip", name); return
    Ppk = _packs(sp["pt"]); Dpk = _packs(sp["dn"]); rows = S.S[sp["rows"]]
    idxf = sp["ridx"] if rand else sp["idx"]
    dps = []
    for i, r in enumerate(rows):
        dj = idxf(r, i)
        dps.append(_dp(r["ids"], L, _delta(comp, Dpk, Ppk, L, dj, i)) - r["baseP_edit"])
    D.setdefault("dp", {})[name] = {"mean": float(np.mean(dps)), "per": [float(x) for x in dps]}
    save_d(); log(f"{name}: meanΔP {np.mean(dps):+.3f}")

def run_dp():
    for direction in ("elicit", "brake"):
        for L in Ls:
            for comp in COMPS:
                _dp_block(direction, L, comp)
            _dp_block(direction, L, "attn", rand=True)  # direction control on the attn channel
    log("RUN_DP_OK")

def _emit_block(direction, L, comp):
    sp = SPECS[direction]
    name = f"emit_{direction}_L{L}_{comp}"
    if name in D.get("emit", {}): log("skip", name); return
    Ppk = _packs(sp["pt"]); Dpk = _packs(sp["dn"]); rows = S.S[sp["rows"]]
    hits = 0; per = []
    for i, r in enumerate(rows):
        dj = sp["idx"](r, i)
        cont = _gen_add(r["ids"], L, _delta(comp, Dpk, Ppk, L, dj, i))
        ok = int(S._emits(cont, TARGET)); hits += ok; per.append(ok)
    D.setdefault("emit", {})[name] = {"rate": hits / len(rows), "per": per}
    save_d(); log(f"{name}: edit-rate {hits/len(rows):.2f}")

def _emit_baseline(direction):
    sp = SPECS[direction]; name = f"emit_{direction}_baseline"
    if name in D.get("emit", {}): log("skip", name); return
    rows = S.S[sp["rows"]]; hits = 0; per = []
    for r in rows:
        cont = S._gen(r["ids"]); ok = int(S._emits(cont, TARGET)); hits += ok; per.append(ok)
    D.setdefault("emit", {})[name] = {"rate": hits / len(rows), "per": per}
    save_d(); log(f"{name}: edit-rate {hits/len(rows):.2f}")

def run_emit():
    # behavioral confirm at the headline layers: elicit L59, brake L55
    for direction, L in (("elicit", 59), ("brake", 55)):
        _emit_baseline(direction)
        for comp in ("full", "attn", "mlp"):
            _emit_block(direction, L, comp)
    log("RUN_EMIT_OK")

def finalize():
    upload_file(path_or_fileobj="/content/cld.json", path_in_repo=DEC_FILE,
                repo_id=DREPO, repo_type="dataset", token=_tok())
    log("FINALIZED:", DEC_FILE)
    print("OILAB_JSON_BEGIN"); print(json.dumps(D)); print("OILAB_JSON_END", flush=True)
