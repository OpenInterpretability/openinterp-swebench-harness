"""Attention KNOCKOUT — do commit heads 8/6/3 CAUSALLY copy the tool name? (paper #7 follow-up). RESUMABLE.

The attention analysis showed heads 8/6/3 at L59 attend to the trajectory's tool-call tokens
(`str_replace_editor` / `bash`). Attention ≠ causation. Decisive test: on EDIT points (where the agent
naturally tends to emit `edit`), zero each commit head's attention from the decision token to the
`str_replace_editor` key positions and renormalize, then measure the drop in P(edit) / edit-emit.

Mechanics (exact): post-softmax zero+renorm == pre-softmax -inf. For head h, o_h = a_h @ V_h (a_h = L59
last-row attention from output_attentions; V_h = v_proj output, GQA group h//(nq/nkv)). Counterfactual
o_h'' = renorm(a_h with targets=0) @ V_h. We inject Δc = Σ_h (o_h''−o_h) mapped through o_proj (offload-robust
module call) at the L59 output and read the logits. Reuses commit_lever_decomp/_heads machinery.

Conditions (EDIT points, n=60):
  ko_edit      : heads {8,6,3}, targets = str_replace_editor positions   -> expect P(edit) DOWN
  ko_bash      : heads {8,6,3}, targets = bash positions                 -> distinct effect (mass to edit?)
  ko_rand      : heads {8,6,3}, targets = random positions (count-matched)-> control, expect ~0
  ko_edit_ctl  : head {12} (non-commit), targets = str_replace_editor    -> head-specificity control

    import sys; sys.path.insert(0,'/content/openinterp-swebench-harness/scripts')
    import commit_lever_stages as S, commit_lever_decomp as D, commit_lever_heads as Hm, commit_lever_knockout as K
    K.setup_eager(); S.capture(); D.load_d(); D.capture_sublayers(); Hm.load_h(); Hm.capture_z()
    K.load_k(); K.run(); K.finalize()
"""
import os, json, time, random
import torch, numpy as np
from huggingface_hub import hf_hub_download, upload_file
import commit_lever_stages as S, commit_lever_decomp as D, commit_lever_heads as Hm

LH = 59
COMMIT = [8, 6, 3]; CTL_HEAD = 12
TARGET, ALT = S.TARGET, S.ALT
KREPO = S.DATA_REPO; KFILE = "results/commit_lever_knockout.json"
K = {}

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def _tok(): return os.environ.get("HF_TOKEN") or None

def save_k():
    json.dump(K, open("/content/clk.json", "w"))
    upload_file(path_or_fileobj="/content/clk.json", path_in_repo=KFILE, repo_id=KREPO,
                repo_type="dataset", token=_tok())

def load_k():
    global K
    try:
        p = hf_hub_download(KREPO, KFILE, repo_type="dataset", token=_tok(), force_download=True)
        K = json.load(open(p)); log("resumed knockout ledger rows:", len(K.get("rows", {})))
    except Exception:
        K = {"rows": {}}; log("fresh knockout ledger")

def setup_eager():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("loading (eager)", S.MODEL_ID)
    S.S["tok"] = AutoTokenizer.from_pretrained(S.MODEL_ID, trust_remote_code=True)
    S.S["model"] = AutoModelForCausalLM.from_pretrained(
        S.MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        attn_implementation="eager").eval()
    S.load_partial(); log("SETUP_EAGER_OK")

def _cfg():
    c = S.S["model"].config; tc = getattr(c, "text_config", c)
    nq = tc.num_attention_heads; nkv = tc.num_key_value_heads
    hd = getattr(tc, "head_dim", None) or tc.hidden_size // nq
    lt = list(tc.layer_types); full = [i for i, t in enumerate(lt) if t == "full_attention"]
    return nq, nkv, hd, full.index(LH), len(lt)

def _tool_positions(ids_row, name):
    toks = [S.S["tok"].decode([i]) for i in ids_row]
    s = ""; spans = []
    for t in toks:
        spans.append((len(s), len(s) + len(t))); s += t
    pos = set(); start = 0
    while True:
        j = s.find(name, start)
        if j < 0: break
        a, b = j, j + len(name)
        for k, (cs, ce) in enumerate(spans):
            if cs < b and ce > a: pos.add(k)
        start = j + 1
    return sorted(pos)

def _capture(ids):
    """one eager forward -> (a[nq,k], V_full[k, nkv*hd], z[nq*hd])."""
    op = Hm._oproj(); box = {}
    h1 = S._layer(LH).self_attn.v_proj.register_forward_hook(
        lambda m, i, o: box.__setitem__("v", (o[0] if isinstance(o, tuple) else o)[0].detach().float().cpu()))
    h2 = op.register_forward_pre_hook(lambda m, a: box.__setitem__("z", a[0][0, -1, :].detach().float().cpu()))
    try:
        with torch.no_grad():
            out = S.S["model"](input_ids=ids.to(S.S["model"].device), use_cache=False, output_attentions=True)
    finally:
        h1.remove(); h2.remove()
    fidx, nall = _cfg()[3], _cfg()[4]
    att = out.attentions[LH] if len(out.attentions) == nall else out.attentions[fidx]
    a = att[0, :, -1, :].detach().float().cpu()
    del out; torch.cuda.empty_cache()
    return a, box["v"], box["z"]

def _oh(a_h, V, h, nq, nkv, hd):
    g = h // (nq // nkv)
    Vh = V[:, g * hd:(g + 1) * hd]                 # [k, hd]
    return a_h @ Vh                                # [hd]

def _knock(a_h, targets):
    a2 = a_h.clone(); a2[targets] = 0.0; s = float(a2.sum())
    return a_h if s < 1e-9 else a2 / s

def _delta(a, V, heads, targets, nq, nkv, hd):
    dz = torch.zeros(nq * hd)
    for h in heads:
        oh = _oh(a[h], V, h, nq, nkv, hd)
        ohk = _oh(_knock(a[h], targets), V, h, nq, nkv, hd)
        dz[h * hd:(h + 1) * hd] = (ohk - oh)
    return Hm._oproj_apply(dz)

def _p2(ids, delta):
    hh = S._layer(LH).register_forward_hook(D._mkadd(delta))
    try:
        with torch.no_grad():
            lg = S.S["model"](input_ids=ids.to(S.S["model"].device), use_cache=False, logits_to_keep=1).logits
    finally:
        hh.remove()
    nl = lg[0, -1]
    return S._p_action(nl, TARGET), S._p_action(nl, ALT)

def run():
    nq, nkv, hd = _cfg()[:3]
    tc = getattr(S.S["model"].config, "text_config", S.S["model"].config); HID = tc.hidden_size
    rows = S.S["EDIT"]; rng = random.Random(0)
    K.setdefault("rows", {})
    for i, r in enumerate(rows):
        if str(i) in K["rows"]: continue
        ids = r["ids"]; ids_row = ids[0].tolist(); k = len(ids_row)
        a, V, z = _capture(ids)
        # gate: o_h(a@V) reconstructs captured z[h_slice]
        if "recon" not in K:
            errs = [float((_oh(a[h], V, h, nq, nkv, hd) - z[h * hd:(h + 1) * hd]).norm() /
                          (z[h * hd:(h + 1) * hd].norm() + 1e-6)) for h in COMMIT]
            K["recon"] = errs; log("GATE o_h=a@V recon relerr (want <1e-2):", [round(e, 4) for e in errs])
        ep = _tool_positions(ids_row, "str_replace_editor")
        bp = _tool_positions(ids_row, "bash")
        cand = [p for p in range(k - 1) if p not in set(ep) | set(bp)]
        rp = sorted(rng.sample(cand, min(len(ep), len(cand)))) if ep else []
        base_e, base_b = _p2(ids, torch.zeros(HID))
        rec = {"n_edit_pos": len(ep), "n_bash_pos": len(bp), "base_e": base_e, "base_b": base_b}
        for name, heads, targets in (("ko_edit", COMMIT, ep), ("ko_bash", COMMIT, bp),
                                     ("ko_rand", COMMIT, rp), ("ko_edit_ctl", [CTL_HEAD], ep)):
            if not targets:
                rec[name] = None; continue
            d = _delta(a, V, heads, targets, nq, nkv, hd)
            pe, pb = _p2(ids, d); rec[name] = {"pe": pe, "pb": pb}
        # emit for the decisive pair
        for name, heads, targets in (("ko_edit", COMMIT, ep), ("ko_rand", COMMIT, rp)):
            if not targets: rec[name + "_emit"] = None; continue
            d = _delta(a, V, heads, targets, nq, nkv, hd)
            cont = D._gen_add(ids, LH, d); rec[name + "_emit"] = int(S._emits(cont, TARGET))
        K["rows"][str(i)] = rec
        if (i + 1) % 6 == 0:
            save_k(); log(f"  knockout {i+1}/{len(rows)}  ko_edit pe {rec.get('ko_edit',{}) and rec['ko_edit']['pe']:.3f} (base {base_e:.3f})")
    save_k(); log("RUN_OK")

def finalize():
    R = K["rows"]; idx = sorted(R, key=int)
    def col(name, key):
        return [R[i][name][key] for i in idx if R[i].get(name)]
    base = [R[i]["base_e"] for i in idx]
    log("=== KNOCKOUT (EDIT points, n=%d) ===" % len(idx))
    log(f"  baseline P(edit) {np.mean(base):.3f}")
    for name in ("ko_edit", "ko_bash", "ko_rand", "ko_edit_ctl"):
        pe = col(name, "pe"); pb = col(name, "pb")
        if pe:
            dpe = np.mean(pe) - np.mean([R[i]["base_e"] for i in idx if R[i].get(name)])
            dpb = np.mean(pb) - np.mean([R[i]["base_b"] for i in idx if R[i].get(name)])
            log(f"  {name:12s} P(edit) {np.mean(pe):.3f} (ΔP {dpe:+.3f}) | P(bash) {np.mean(pb):.3f} (ΔP {dpb:+.3f})")
    for name in ("ko_edit_emit", "ko_rand_emit"):
        em = [R[i][name] for i in idx if R[i].get(name) is not None]
        if em: log(f"  {name}: edit-rate {np.mean(em):.2f}  (EDIT baseline 0.48)")
    upload_file(path_or_fileobj="/content/clk.json", path_in_repo=KFILE, repo_id=KREPO,
                repo_type="dataset", token=_tok())
    print("OILAB_JSON_BEGIN"); print(json.dumps({k: K[k] for k in K if k != "rows"})); print("OILAB_JSON_END", flush=True)
