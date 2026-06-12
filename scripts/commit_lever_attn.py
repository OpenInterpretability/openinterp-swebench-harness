"""What do the commit heads (8,6,3) ATTEND TO at the decision point? (paper #7 follow-up). RESUMABLE.

The head decomposition (commit_lever_heads) found a sparse push circuit: heads 8/6/3 at L59 write the elicit,
opposed by 16/19/21. Here we ask what those heads READ. At the decision token (`<function`, last position) we
capture the L59 attention distribution over all key positions for the commit heads vs the opponent heads, in
the EDIT (committed) state and the BASH (wandering) state, and characterize:
  * attention entropy (focused vs diffuse)
  * locality (mass on the last 32/128 tokens vs global)
  * the most-attended decoded tokens (does it read the verdict / diff / tool-result / scaffold?)

Model is loaded with attn_implementation="eager" so out.attentions exposes the weights (only the 16 full-attn
layers of the hybrid stack materialize; L59 is one). Processes one row at a time, frees per row.

    import sys; sys.path.insert(0,'/content/openinterp-swebench-harness/scripts')
    import commit_lever_stages as S, commit_lever_attn as A
    A.setup_eager(); S.capture(); A.load_a(); A.smoke(); A.analyze(); A.finalize()
"""
import os, json, time, math
from collections import Counter
import torch, numpy as np
from huggingface_hub import hf_hub_download, upload_file
import commit_lever_stages as S

LH = 59
COMMIT = [8, 6, 3]; OPP = [21, 16, 19]; RAND = [12]
HEADS = COMMIT + OPP + RAND
AREPO = S.DATA_REPO
AFILE = "results/commit_lever_attn.json"
A = {}

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def _tok(): return os.environ.get("HF_TOKEN") or None

def save_a():
    json.dump(A, open("/content/cla.json", "w"))
    upload_file(path_or_fileobj="/content/cla.json", path_in_repo=AFILE, repo_id=AREPO,
                repo_type="dataset", token=_tok())

def load_a():
    global A
    try:
        p = hf_hub_download(AREPO, AFILE, repo_type="dataset", token=_tok(), force_download=True)
        A = json.load(open(p)); log("resumed attn ledger:", sorted(A.keys()))
    except Exception:
        A = {}; log("fresh attn ledger")

def setup_eager():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("loading (eager)", S.MODEL_ID)
    S.S["tok"] = AutoTokenizer.from_pretrained(S.MODEL_ID, trust_remote_code=True)
    S.S["model"] = AutoModelForCausalLM.from_pretrained(
        S.MODEL_ID, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        attn_implementation="eager").eval()
    S.load_partial(); log("SETUP_EAGER_OK")

def _attn_idx():
    """Index of L59 within out.attentions — the hybrid stack only emits attentions for full-attention layers."""
    tc = getattr(S.S["model"].config, "text_config", S.S["model"].config)
    lt = list(tc.layer_types)
    full = [i for i, t in enumerate(lt) if t == "full_attention"]
    A["_layer_types_len"] = len(lt); A["_n_full"] = len(full); A["_LH_full_idx"] = full.index(LH)
    return full.index(LH), len(lt), len(full)

def _attn_last(ids):
    """Return L59 attention of the last query row: [n_heads, k_len] (float cpu)."""
    fidx, n_all, n_full = _attn_idx()
    with torch.no_grad():
        out = S.S["model"](input_ids=ids.to(S.S["model"].device), use_cache=False, output_attentions=True)
    natt = len(out.attentions)
    att = out.attentions[LH] if natt == n_all else out.attentions[fidx]   # full-list vs compact-list
    if att is None:
        raise RuntimeError(f"attention for L{LH} is None — eager/output_attentions not exposing weights")
    a = att[0, :, -1, :].detach().float().cpu()
    del out
    torch.cuda.empty_cache()
    return a

def smoke():
    r = S.S["EDIT"][0]
    a = _attn_last(r["ids"])
    A["smoke"] = {"shape": list(a.shape), "row_sums_ok": bool(torch.allclose(a.sum(-1), torch.ones(a.shape[0]), atol=1e-2))}
    log("SMOKE attn shape", list(a.shape), "rowsum~1:", A["smoke"]["row_sums_ok"])
    save_a()

def _row_stats(a_h, ids_row):
    p = a_h.numpy().astype(np.float64); p = p / (p.sum() + 1e-12)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    k = len(p)
    rec32 = float(p[-32:].sum()); rec128 = float(p[-128:].sum())
    top = np.argsort(-p)[:8]
    toks = [(int(t), float(p[t]), S.S["tok"].decode([int(ids_row[t])])) for t in top]
    return ent, rec32, rec128, toks

def analyze():
    if A.get("done"): log("analyze skip (done)"); return
    part = A.get("acc", {})
    for state, rows_key in (("edit", "EDIT"), ("bash", "BASH")):
        rows = S.S[rows_key]
        st = part.get(state, {"i": 0, "heads": {str(h): {"ent": [], "rec32": [], "rec128": [], "tokmass": {}}
                                               for h in HEADS}})
        start = st["i"]
        for i in range(start, len(rows)):
            r = rows[i]; ids_row = r["ids"][0].tolist()
            a = _attn_last(r["ids"])                     # [nh, k]
            for h in HEADS:
                ent, rec32, rec128, toks = _row_stats(a[h], ids_row)
                hs = st["heads"][str(h)]
                hs["ent"].append(ent); hs["rec32"].append(rec32); hs["rec128"].append(rec128)
                for _, w, tokstr in toks:
                    hs["tokmass"][tokstr] = hs["tokmass"].get(tokstr, 0.0) + w
            st["i"] = i + 1
            if (i + 1) % 20 == 0:
                part[state] = st; A["acc"] = part; save_a(); log(f"  attn {state}: {i+1}/{len(rows)}")
        part[state] = st; A["acc"] = part; save_a(); log(f"attn {state} done ({len(rows)})")
    # summarize
    summ = {}
    for state in ("edit", "bash"):
        summ[state] = {}
        for h in HEADS:
            hs = A["acc"][state]["heads"][str(h)]
            top = sorted(hs["tokmass"].items(), key=lambda kv: -kv[1])[:20]
            summ[state][str(h)] = {"ent": float(np.mean(hs["ent"])), "rec32": float(np.mean(hs["rec32"])),
                                   "rec128": float(np.mean(hs["rec128"])),
                                   "top_tokens": [[t, round(w / len(S.S["EDIT" if state == "edit" else "BASH"]), 4)] for t, w in top]}
    A["summary"] = summ; A["done"] = True; save_a()
    log("ANALYZE_OK")
    for h in COMMIT:
        s = summ["edit"][str(h)]
        log(f"  commit head {h}: ent {s['ent']:.2f} rec128 {s['rec128']:.2f} top:", [t[0] for t in s["top_tokens"][:6]])
    for h in OPP:
        s = summ["edit"][str(h)]
        log(f"  opp head {h}:    ent {s['ent']:.2f} rec128 {s['rec128']:.2f} top:", [t[0] for t in s["top_tokens"][:6]])

def finalize():
    upload_file(path_or_fileobj="/content/cla.json", path_in_repo=AFILE, repo_id=AREPO,
                repo_type="dataset", token=_tok())
    log("FINALIZED:", AFILE)
    print("OILAB_JSON_BEGIN"); print(json.dumps({k: A[k] for k in A if k != "acc"})); print("OILAB_JSON_END", flush=True)
