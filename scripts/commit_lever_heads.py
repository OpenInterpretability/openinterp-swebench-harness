"""Per-HEAD decomposition of the L59 attention write of the commitment-lever (paper #7 follow-up). RESUMABLE.

The attn-vs-MLP decomposition (commit_lever_decomp) showed the ELICIT lever is written by the L59 attention
sublayer (r_attn=0.43, MLP null, direction-specific). L59 is a FULL-attention layer in Qwen3.6-27B's hybrid
linear/full stack (full_attention_interval=4): 24 query heads, head_dim 256, o_proj has no bias.

The attention output is an EXACT sum over heads:  attn_out_L = Σ_h c_h,  c_h = z[h_slice] @ W_O[:, h_slice]^T,
where z = the o_proj INPUT (concat of head outputs). So the per-head re-injection delta for the donor→point
swap is  Δc_h = W_O[:, h_slice] @ (z_donor[h_slice] − z_point[h_slice]),  and Σ_h Δc_h = the full attention
delta (== the +0.223 ΔP attn-channel effect). We re-inject each head's Δc_h alone at the L59 output and ask
which head(s) carry the commit.

Reuses commit_lever_stages (base state) + commit_lever_decomp (donors, _mkadd, _dp, _gen_add). Drive one exec
at a time; every block checkpoints to HF.
    import sys; sys.path.insert(0,'/content/openinterp-swebench-harness/scripts')
    import commit_lever_stages as S, commit_lever_decomp as D, commit_lever_heads as H
    S.setup(); S.capture(); D.load_d(); D.capture_sublayers()   # base + a/m/x (a[59] used to validate)
    H.load_h(); H.capture_z(); H.geom(); H.run_heads(); H.run_cumulative(); H.run_emit(); H.finalize()
"""
import os, json, time
import torch, numpy as np
from huggingface_hub import hf_hub_download, upload_file
import commit_lever_stages as S, commit_lever_decomp as D

LH = 59
HREPO = S.DATA_REPO
HFILE = "results/commit_lever_heads.json"
ZFILE = "results/commit_lever_z59.pt"
TARGET = S.TARGET
H = {}

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def _tok(): return os.environ.get("HF_TOKEN") or None

def save_h():
    json.dump(H, open("/content/clh.json", "w"))
    upload_file(path_or_fileobj="/content/clh.json", path_in_repo=HFILE, repo_id=HREPO,
                repo_type="dataset", token=_tok())

def load_h():
    global H
    try:
        p = hf_hub_download(HREPO, HFILE, repo_type="dataset", token=_tok(), force_download=True)
        H = json.load(open(p)); log("resumed heads ledger:", sorted(H.keys()))
    except Exception:
        H = {}; log("fresh heads ledger")

def _oproj():
    return S._layer(LH).self_attn.o_proj

def _cfg():
    c = S.S["model"].config
    tc = getattr(c, "text_config", c)
    nh = tc.num_attention_heads; hd = getattr(tc, "head_dim", None) or tc.hidden_size // nh
    return nh, hd

# ----------------------------------------------------- capture o_proj input z (per row)
def capture_z():
    if "E_res" not in S.S:
        raise RuntimeError("call S.capture() + D.capture_sublayers() first")
    nh, hd = _cfg(); H["n_heads"] = nh; H["head_dim"] = hd
    try:
        p = hf_hub_download(HREPO, ZFILE, repo_type="dataset", token=_tok(), force_download=True)
        st = torch.load(p, map_location="cpu", weights_only=False)
        S.S["E_z"] = st["E_z"]; S.S["B_z"] = st["B_z"]; log("Z_RESUMED from HF"); return
    except Exception:
        log("capturing o_proj input z fresh")
    op = _oproj(); dev = S.S["model"].device
    def cap(rows, tag):
        zs = []
        for n, r in enumerate(rows):
            box = {}
            h = op.register_forward_pre_hook(lambda m, a: box.__setitem__("z", a[0][:, -1, :].detach().float().cpu()[0]))
            try:
                with torch.no_grad():
                    S.S["model"](input_ids=r["ids"].to(dev), use_cache=False, logits_to_keep=1)
            finally:
                h.remove()
            zs.append(box["z"])
            if (n + 1) % 20 == 0: log(f"  z {tag}: {n+1}/{len(rows)}")
        return zs
    S.S["E_z"] = cap(S.S["EDIT"], "EDIT"); S.S["B_z"] = cap(S.S["BASH"], "BASH")
    # GATE: Σ_h c_h == attn_out (validate the head split against the captured self_attn output)
    WO = op.weight.detach().float().cpu()           # [hidden, nh*hd]
    rec = WO @ S.S["E_z"][0]                          # should == E_a[LH][0]
    rel = float((rec - S.S["E_a"][LH][0]).norm() / (S.S["E_a"][LH][0].norm() + 1e-6))
    H["headsplit_relerr"] = rel; H["oproj_in"] = int(WO.shape[1]); H["has_bias"] = op.bias is not None
    log(f"GATE head-split relerr {rel:.4f} (want <1e-2) | nh={nh} hd={hd} oproj_in={WO.shape[1]} bias={op.bias is not None}")
    torch.save({"E_z": S.S["E_z"], "B_z": S.S["B_z"]}, "/content/z59.pt")
    upload_file(path_or_fileobj="/content/z59.pt", path_in_repo=ZFILE, repo_id=HREPO,
                repo_type="dataset", token=_tok())
    save_h(); log("Z_OK")

def _WO():
    return _oproj().weight.detach().float().cpu()

# donor=same-repo edit (elicit); point=bash
def _meanΔz():
    dz = []
    for i, r in enumerate(S.S["BASH"]):
        j = D._edit_idx(r, i); dz.append(S.S["E_z"][j] - S.S["B_z"][i])
    return torch.stack(dz).mean(0)

# ----------------------------------------------------- free geometric attribution
def geom():
    if "geom" in H: log("geom skip"); return
    nh, hd = H["n_heads"], H["head_dim"]; WO = _WO()
    mdz = _meanΔz()                                   # [nh*hd]
    dattn = WO @ mdz; u = dattn / (dattn.norm() + 1e-8)
    rows = []
    for h in range(nh):
        sl = slice(h * hd, (h + 1) * hd)
        ch = WO[:, sl] @ mdz[sl]
        rows.append({"head": h, "proj": float(ch @ u), "norm": float(ch.norm())})
    H["geom"] = {"per_head": rows, "dattn_norm": float(dattn.norm())}
    save_h()
    top = sorted(rows, key=lambda x: -x["proj"])[:6]
    log("GEOM top heads by proj:", [(r["head"], round(r["proj"], 3)) for r in top])

def _head_delta(WO, hd, h, j, i):
    sl = slice(h * hd, (h + 1) * hd)
    return WO[:, sl] @ (S.S["E_z"][j][sl] - S.S["B_z"][i][sl])

# ----------------------------------------------------- causal ΔP per head (elicit, all heads)
def run_heads():
    nh, hd = H["n_heads"], H["head_dim"]; WO = _WO()
    H.setdefault("dp_head", {})
    for h in range(nh):
        key = str(h)
        if key in H["dp_head"]: log(f"head {h} skip"); continue
        dps = []
        for i, r in enumerate(S.S["BASH"]):
            j = D._edit_idx(r, i)
            dps.append(D._dp(r["ids"], LH, _head_delta(WO, hd, h, j, i)) - r["baseP_edit"])
        H["dp_head"][key] = {"mean": float(np.mean(dps)), "per": [float(x) for x in dps]}
        save_h(); log(f"head {h:2d}: meanΔP {np.mean(dps):+.3f}")
    log("RUN_HEADS_OK")

def _cum_delta(WO, hd, heads, j, i):
    d = torch.zeros(WO.shape[0])
    for h in heads:
        sl = slice(h * hd, (h + 1) * hd)
        d = d + WO[:, sl] @ (S.S["E_z"][j][sl] - S.S["B_z"][i][sl])
    return d

def _ranked_heads():
    return [int(k) for k, _ in sorted(H["dp_head"].items(), key=lambda kv: -kv[1]["mean"])]

# ----------------------------------------------------- cumulative top-k + all (sanity) + random ctl
def run_cumulative():
    nh, hd = H["n_heads"], H["head_dim"]; WO = _WO()
    order = _ranked_heads()
    H.setdefault("dp_cum", {})
    sets = {f"top{k}": order[:k] for k in (1, 2, 3, 5, 8)}
    sets["all"] = list(range(nh))
    for name, heads in sets.items():
        if name in H["dp_cum"]: log(f"cum {name} skip"); continue
        dps = []
        for i, r in enumerate(S.S["BASH"]):
            j = D._edit_idx(r, i)
            dps.append(D._dp(r["ids"], LH, _cum_delta(WO, hd, heads, j, i)) - r["baseP_edit"])
        H["dp_cum"][name] = {"mean": float(np.mean(dps)), "per": [float(x) for x in dps], "heads": heads}
        save_h(); log(f"cum {name} ({len(heads)} heads): meanΔP {np.mean(dps):+.3f}")
    # direction control: top head with cross-repo donor
    if "topctl" not in H["dp_cum"]:
        htop = order[0]; dps = []
        for i, r in enumerate(S.S["BASH"]):
            j = D._xt_edit_idx(r, i)
            dps.append(D._dp(r["ids"], LH, _head_delta(WO, hd, htop, j, i)) - r["baseP_edit"])
        H["dp_cum"]["topctl"] = {"mean": float(np.mean(dps)), "per": [float(x) for x in dps], "head": htop}
        save_h(); log(f"cum topctl (head {htop}, cross-repo donor): meanΔP {np.mean(dps):+.3f}")
    log("RUN_CUM_OK")

def run_emit():
    nh, hd = H["n_heads"], H["head_dim"]; WO = _WO()
    order = _ranked_heads()
    H.setdefault("emit", {})
    for name, heads in (("top3", order[:3]), ("top5", order[:5]), ("all", list(range(nh)))):
        if name in H["emit"]: log(f"emit {name} skip"); continue
        hits = 0; per = []
        for i, r in enumerate(S.S["BASH"]):
            j = D._edit_idx(r, i)
            cont = D._gen_add(r["ids"], LH, _cum_delta(WO, hd, heads, j, i))
            ok = int(S._emits(cont, TARGET)); hits += ok; per.append(ok)
        H["emit"][name] = {"rate": hits / len(S.S["BASH"]), "per": per, "heads": heads}
        save_h(); log(f"emit {name}: rate {hits/len(S.S['BASH']):.2f}")
    log("RUN_EMIT_OK")

def finalize():
    upload_file(path_or_fileobj="/content/clh.json", path_in_repo=HFILE, repo_id=HREPO,
                repo_type="dataset", token=_tok())
    log("FINALIZED:", HFILE)
    print("OILAB_JSON_BEGIN"); print(json.dumps(H)); print("OILAB_JSON_END", flush=True)
