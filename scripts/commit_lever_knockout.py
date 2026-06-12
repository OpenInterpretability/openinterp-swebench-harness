"""Source-token KNOCKOUT — is reading the `str_replace_editor` tokens CAUSALLY necessary for `edit`?
(paper #7 follow-up). RESUMABLE.

The attention analysis showed the L59 commit heads attend to the trajectory's tool-call tokens. Attention ≠
causation. Gate-agnostic causal test (Qwen3.6 full-attn uses output-gated attention, so reconstructing o_h=a@V
fails — we instead ablate the SOURCE CONTENT): on EDIT points (agent naturally emits `edit` ~0.48), mean-ablate
the L59 attention's view of the `str_replace_editor` key positions by replacing k_proj & v_proj outputs at
those positions with the per-channel mean over positions. The decision token can no longer read
"str_replace_editor" there. Measure the drop in P(edit) / edit-emit.

This ablates the SOURCE the heads read (surgical: only L59's K/V, not the residual stream or the gate). Head
specificity is inherited from the head decomposition (8/6/3 are the readers); here we test source necessity.

Conditions (EDIT points, n=60):
  ko_edit : ablate str_replace_editor key positions  -> expect P(edit) DOWN if the heads copy the tool name
  ko_bash : ablate bash key positions                -> distinct effect (mass shifts off bash)
  ko_rand : ablate random key positions (count-matched) -> control, expect ~0

    import sys; sys.path.insert(0,'/content/openinterp-swebench-harness/scripts')
    import commit_lever_stages as S, commit_lever_decomp as D, commit_lever_knockout as K
    S.setup(); S.capture(); K.load_k(); K.run(); K.finalize()
"""
import os, json, time, random
import torch, numpy as np
from huggingface_hub import hf_hub_download, upload_file
import commit_lever_stages as S, commit_lever_decomp as D

LH = 59
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

def _patch_hooks(targets):
    """Mean-ablate k_proj & v_proj outputs at `targets` (prefill only) — removes those positions' content."""
    sa = S._layer(LH).self_attn
    tgt = torch.tensor(targets, dtype=torch.long)
    hs = []
    def mk():
        def h(m, i, o):
            x = o[0] if isinstance(o, tuple) else o
            if x.shape[1] > 1 and len(targets):
                x = x.clone(); mean = x[0].mean(0)
                x[0, tgt, :] = mean.to(x.dtype)
                return (x, *o[1:]) if isinstance(o, tuple) else x
            return o
        return h
    hs.append(sa.k_proj.register_forward_hook(mk()))
    hs.append(sa.v_proj.register_forward_hook(mk()))
    return hs

def _p2(ids, targets):
    hs = _patch_hooks(targets)
    try:
        with torch.no_grad():
            lg = S.S["model"](input_ids=ids.to(S.S["model"].device), use_cache=False, logits_to_keep=1).logits
    finally:
        for h in hs: h.remove()
    nl = lg[0, -1]
    return S._p_action(nl, TARGET), S._p_action(nl, ALT)

def _emit(ids, targets):
    model, tok = S.S["model"], S.S["tok"]
    hs = _patch_hooks(targets)
    try:
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(model.device), max_new_tokens=S.MAXNEW, do_sample=False,
                                 use_cache=True, pad_token_id=tok.eos_token_id,
                                 attention_mask=torch.ones_like(ids).to(model.device))
    finally:
        for h in hs: h.remove()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)

def run():
    rows = S.S["EDIT"]; rng = random.Random(0)
    K.setdefault("rows", {})
    for i, r in enumerate(rows):
        if str(i) in K["rows"]: continue
        ids = r["ids"]; ids_row = ids[0].tolist(); k = len(ids_row)
        ep = _tool_positions(ids_row, "str_replace_editor")
        bp = _tool_positions(ids_row, "bash")
        cand = [p for p in range(k - 1) if p not in set(ep) | set(bp)]
        rp = sorted(rng.sample(cand, min(len(ep), len(cand)))) if ep else []
        be, bb = _p2(ids, [])
        rec = {"n_edit_pos": len(ep), "n_bash_pos": len(bp), "base_e": be, "base_b": bb}
        for name, targets in (("ko_edit", ep), ("ko_bash", bp), ("ko_rand", rp)):
            if not targets: rec[name] = None; continue
            pe, pb = _p2(ids, targets); rec[name] = {"pe": pe, "pb": pb}
        for name, targets in (("ko_edit", ep), ("ko_rand", rp)):
            rec[name + "_emit"] = (int(S._emits(_emit(ids, targets), TARGET)) if targets else None)
        rec["base_emit"] = int(S._emits(_emit(ids, []), TARGET))
        K["rows"][str(i)] = rec
        if (i + 1) % 6 == 0:
            save_k()
            ke = rec.get("ko_edit"); log(f"  knockout {i+1}/{len(rows)}  base_e {be:.3f}" +
                                         (f" ko_edit_pe {ke['pe']:.3f}" if ke else " (no edit-pos)"))
    save_k(); log("RUN_OK")

def finalize():
    R = K["rows"]; idx = sorted(R, key=int)
    def pair(name):
        sub = [i for i in idx if R[i].get(name)]
        pe = np.mean([R[i][name]["pe"] for i in sub]); pb = np.mean([R[i][name]["pb"] for i in sub])
        be = np.mean([R[i]["base_e"] for i in sub]); bb = np.mean([R[i]["base_b"] for i in sub])
        return pe, pe - be, pb, pb - bb, len(sub)
    log("=== SOURCE-KNOCKOUT (EDIT points) ===")
    log(f"  baseline  P(edit) {np.mean([R[i]['base_e'] for i in idx]):.3f}  P(bash) {np.mean([R[i]['base_b'] for i in idx]):.3f}  (n={len(idx)})")
    for name in ("ko_edit", "ko_bash", "ko_rand"):
        pe, dpe, pb, dpb, n = pair(name)
        log(f"  {name:8s} P(edit) {pe:.3f} (ΔP {dpe:+.3f}) | P(bash) {pb:.3f} (ΔP {dpb:+.3f})  [n={n}]")
    be = [R[i]["base_emit"] for i in idx if R[i].get("base_emit") is not None]
    log(f"  emit baseline {np.mean(be):.2f}")
    for name in ("ko_edit_emit", "ko_rand_emit"):
        em = [R[i][name] for i in idx if R[i].get(name) is not None]
        if em: log(f"  {name}: edit-rate {np.mean(em):.2f}")
    upload_file(path_or_fileobj="/content/clk.json", path_in_repo=KFILE, repo_id=KREPO,
                repo_type="dataset", token=_tok())
    print("OILAB_JSON_BEGIN"); print(json.dumps({k: K[k] for k in K if k != "rows"})); print("OILAB_JSON_END", flush=True)
