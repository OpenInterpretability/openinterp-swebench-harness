"""Hybridization-selection audit ("HydraHead B+") — PREREG: paper/circuit_breaker/PREREG_hybridization_audit.md

Stage 1: HydraHead-style receiver IE (arXiv:2606.20097 Eq. 9-10) for ALL 384 FA heads of Qwen3.6-27B
(full_attention_interval=4 -> FA layers 3,7,...,63; 24 heads each) on NIAH counterfactual probes.
H1/H2: are the commit writers {8,6,3}@L59 / opposers {16,19,11,4,21,23}@L59 in the retained-FA set
at K=25% (96) and K=12.5% (48)?
H3 (PRIMARY, two-sided causal): under the SAME ablation of non-retained heads, measure (a) the commit
circuit at the #7 decision points (P(edit) + emit) and (b) NIAH readout. Reviewable positive =
"benchmark passes while the commitment circuit degrades/mis-calibrates".

Ablation proxy for "head converted away": replace the head's o_proj-input slice at EVERY position with
its own per-row sequence mean (removes position-specific content, keeps average drive; zero-ablation as
robustness). Honest scope: upper bound of damage BEFORE distillation recovery.

Driven cell-by-cell (never fire-and-forget), resumable via HF ledger:
    import sys; sys.path.insert(0,'/content/openinterp-swebench-harness/scripts')
    import commit_lever_stages as S, hydra_audit as A
    S.setup()
    A.load_a(); A.build_probes(); A.bases()          # clean/corr runs + caches + G1 denominators
    A.ie_layer(3) ... A.ie_layer(63)  (or A.ie_all())  # per-layer checkpoint to HF
    A.rank()                                          # ranking, cutoffs, H1/H2, split-half stability
    S.capture()                                       # (H3 only) the #7 deterministic decision points
    A.h3_block('base'); A.h3_block('drop_K25') ...    # two-sided blocks, checkpointed
    A.finalize()
"""
import os, json, gc, time, random
import torch, numpy as np
from huggingface_hub import hf_hub_download, upload_file
import commit_lever_stages as S

FA_LAYERS = list(range(3, 64, 4))          # 3,7,...,63 (16 layers)
NH, HD = 24, 256                            # verified in RESULTS_heads_L59.md (GQA 24:4, o_proj no bias)
L59 = 59
WRITERS = [(L59, h) for h in (8, 6, 3)]
OPPOSERS = [(L59, h) for h in (16, 19, 11, 4, 21, 23)]
LAM = 0.9                                   # HydraHead span-readout decay
IE_MIN = 0.01                               # their criticality threshold
K_CUTS = {"K25": 96, "K125": 48}
N_PAIRS = 6                                 # per probe (their C.1: ranking stable ~6)
DENOM_MIN = 0.5                             # skip pair if corruption barely moves the readout
CTX_TOK_BUDGET = 3600                       # + answer stays under harness MAXLEN=4000
AFILE = "results/hydra_audit_ie.json"

A = {}   # ledger (mirrors HF)

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def _tok(): return os.environ.get("HF_TOKEN") or None

def save_a():
    json.dump(A, open("/content/ha.json", "w"))
    upload_file(path_or_fileobj="/content/ha.json", path_in_repo=AFILE,
                repo_id=S.DATA_REPO, repo_type="dataset", token=_tok())

def load_a():
    global A
    try:
        p = hf_hub_download(S.DATA_REPO, AFILE, repo_type="dataset", token=_tok(), force_download=True)
        A = json.load(open(p)); log("resumed hydra ledger:", sorted(A.keys()))
    except Exception:
        A = {}; log("fresh hydra ledger")

def _attn_mod(L):
    """The attention sublayer of layer L (works for both full and linear layers of the hybrid)."""
    lay = S._layer(L)
    for name in ("self_attn", "linear_attn", "attn"):
        if hasattr(lay, name): return getattr(lay, name)
    raise RuntimeError(f"no attn module at L{L}")

def _oproj(L):
    m = _attn_mod(L)
    for name in ("o_proj", "out_proj"):
        if hasattr(m, name): return getattr(m, name)
    raise RuntimeError(f"no o_proj at L{L}")

# ------------------------------------------------------------------ NIAH probes (deterministic)
FILLER = ("The sky was clear over the valley and the grass kept growing in silence. "
          "People walked to the market and back, carrying baskets of fruit and bread. "
          "A river ran past the old mill, turning slowly through the seasons. ")
KEYS = ["golden-lock", "silver-gate", "crimson-door", "ivory-tower", "amber-bridge", "cobalt-vault"]

def _num(rng): return "".join(rng.choice("0123456789") for _ in range(7))

def _corrupt(num, rng):
    while True:
        c = _num(rng)
        if c != num: return c

def _needle(key, num): return f" The special magic number for {key} is {num}. "

def _mk_prompt(needles, qkey):
    """needles: list[(key, num, depth 0..1)] -> (prompt_text, question_prefix)."""
    tok = S["tok"] if isinstance(S, dict) else S.S["tok"]
    base = FILLER * 200
    ids = tok(base, add_special_tokens=False).input_ids[:CTX_TOK_BUDGET]
    text = tok.decode(ids)
    for key, num, depth in sorted(needles, key=lambda x: x[2]):
        pos = int(len(text) * depth)
        cut = text.rfind(". ", 0, pos)
        cut = cut + 2 if cut > 0 else pos
        text = text[:cut] + _needle(key, num) + text[cut:]
    q = f"\n\nQuestion: What is the special magic number for {qkey}?\nAnswer: The special magic number for {qkey} is "
    return text + q

def build_probes():
    """12 counterfactual pairs: 6 single-key + 6 multi-key(3 needles, query 1). Deterministic seed."""
    if A.get("probes"): log("PROBES_SKIP (in ledger)"); return
    rng = random.Random(59)
    probes = []
    for i in range(N_PAIRS):                                   # single-key
        key = KEYS[i % len(KEYS)]; num = _num(rng); corr = _corrupt(num, rng)
        depth = [0.2, 0.35, 0.5, 0.65, 0.8, 0.44][i]
        probes.append({"probe": "single", "pair": i, "qkey": key, "ans": num, "ans_corr": corr,
                       "needles": [[key, num, depth]],
                       "needles_corr": [[key, corr, depth]]})
    for i in range(N_PAIRS):                                   # multi-key MK1
        ks = [KEYS[(i + j) % len(KEYS)] for j in range(3)]
        ns = [_num(rng) for _ in range(3)]; qi = i % 3
        corr = _corrupt(ns[qi], rng)
        ds = [[0.25, 0.5, 0.75], [0.3, 0.55, 0.7], [0.2, 0.45, 0.8]][i % 3]
        needles = [[ks[j], ns[j], ds[j]] for j in range(3)]
        nc = [list(x) for x in needles]; nc[qi][1] = corr
        probes.append({"probe": "mk1", "pair": i, "qkey": ks[qi], "ans": ns[qi], "ans_corr": corr,
                       "needles": needles, "needles_corr": nc})
    A["probes"] = probes; save_a(); log(f"PROBES_OK n={len(probes)}")

# ------------------------------------------------------------------ span readout m(x) (Eq. 9)
def _ans_ids(num):
    tok = S.S["tok"]
    return tok(num, add_special_tokens=False).input_ids

def _readout(ids_full, n_ans, aplus, aminus):
    """Teacher-forced logits at the n_ans answer positions -> lambda-weighted logit diff."""
    with torch.no_grad():
        lg = S.S["model"](input_ids=ids_full.to(S.S["model"].device), use_cache=False).logits[0].float()
    # logits predicting answer token j live at position (len - n_ans - 1 + j)
    T = ids_full.shape[1]; m, Z = 0.0, 0.0
    for j in range(n_ans):
        z = lg[T - n_ans - 1 + j]
        w = LAM ** j
        m += w * float(z[aplus[j]] - z[aminus[j]]); Z += w
    del lg
    return m / Z

def _pair_ids(pr, corrupted):
    tok = S.S["tok"]
    needles = pr["needles_corr"] if corrupted else pr["needles"]
    txt = _mk_prompt([tuple(n) for n in needles], pr["qkey"])
    ap = _ans_ids(pr["ans"])                                    # teacher-force the CLEAN answer always
    ids = tok(txt, add_special_tokens=False).input_ids + ap
    return torch.tensor([ids]), len(ap), ap, _ans_ids(pr["ans_corr"])

def bases():
    """Clean/corrupted m per pair + caches needed for patching. G1 denominator check."""
    if A.get("bases"): log("BASES_SKIP"); return
    out = {}
    for k, pr in enumerate(A["probes"]):
        idc, na, ap, am = _pair_ids(pr, corrupted=False)
        idx, _, _, _ = _pair_ids(pr, corrupted=True)
        if len(ap) != len(am):                                  # rare tokenizer length mismatch: resample noted
            out[str(k)] = {"skip": "anslen"}; continue
        m_clean = _readout(idc, na, ap, am)
        m_corr = _readout(idx, na, ap, am)
        out[str(k)] = {"m_clean": m_clean, "m_corr": m_corr, "denom": m_clean - m_corr}
        log(f"  pair {k} ({pr['probe']}): clean {m_clean:+.2f} corr {m_corr:+.2f}")
        gc.collect(); torch.cuda.empty_cache()
    A["bases"] = out
    ok = [k for k, v in out.items() if v.get("denom", 0) and v["denom"] > DENOM_MIN]
    A["pairs_ok"] = ok; save_a()
    log(f"BASES_OK usable pairs {len(ok)}/{len(out)} (G1 needs most; corruption must flip the readout)")

# ------------------------------------------------------------------ per-head receiver IE
def _cache_z_corr(pr):
    """Corrupted-run o_proj inputs at all FA layers, all positions (cpu fp16)."""
    idx, na, ap, am = _pair_ids(pr, corrupted=True)
    caches, hs = {}, []
    for L in FA_LAYERS:
        hs.append(_oproj(L).register_forward_pre_hook(
            (lambda LL: lambda m, a: caches.__setitem__(LL, a[0].detach().half().cpu()))(L)))
    try:
        with torch.no_grad():
            S.S["model"](input_ids=idx.to(S.S["model"].device), use_cache=False)
    finally:
        for h in hs: h.remove()
    return caches

def _cache_attn_clean(idc):
    """Clean-run attention-sublayer outputs at ALL layers (for the downstream freeze), cpu fp16."""
    n_layers = 64
    caches, hs = {}, []
    for L in range(n_layers):
        try: mod = _attn_mod(L)
        except RuntimeError: continue
        def mk(LL):
            def h(m, i, o):
                y = o[0] if isinstance(o, tuple) else o
                caches[LL] = y.detach().half().cpu()
            return h
        hs.append(mod.register_forward_hook(mk(L)))
    try:
        with torch.no_grad():
            S.S["model"](input_ids=idc.to(S.S["model"].device), use_cache=False)
    finally:
        for h in hs: h.remove()
    return caches

def _patched_m(idc, na, ap, am, L, head, z_corr_L, attn_clean):
    """m(x; O_{L,h} <- corrupted), downstream attention frozen to clean (direct effect, their Eq. 10)."""
    hs = []
    sl = slice(head * HD, (head + 1) * HD)
    def swap(m, a):
        z = a[0].clone()
        z[:, :, sl] = z_corr_L.to(z.device, z.dtype)[:, :z.shape[1], sl]
        return (z,) + a[1:]
    hs.append(_oproj(L).register_forward_pre_hook(swap))
    for Ld in range(L + 1, 64):
        try: mod = _attn_mod(Ld)
        except RuntimeError: continue
        if Ld not in attn_clean: continue
        def mkfreeze(LL):
            def h(m, i, o):
                y = attn_clean[LL].to(S.S["model"].device)
                if isinstance(o, tuple): return (y.to(o[0].dtype),) + o[1:]
                return y.to(o.dtype)
            return h
        hs.append(mod.register_forward_hook(mkfreeze(Ld)))
    try:
        return _readout(idc, na, ap, am)
    finally:
        for h in hs: h.remove()

def ie_layer(L):
    """Receiver IE for all 24 heads of FA layer L over usable pairs. Checkpoints per layer."""
    key = f"ie_L{L}"
    if A.get(key): log(f"IE_SKIP L{L}"); return
    res = {str(h): [] for h in range(NH)}
    for k in A["pairs_ok"]:
        pr = A["probes"][int(k)]; b = A["bases"][k]
        idc, na, ap, am = _pair_ids(pr, corrupted=False)
        zc = _cache_z_corr(pr)[L]
        ac = _cache_attn_clean(idc)
        for h in range(NH):
            mp = _patched_m(idc, na, ap, am, L, h, zc, ac)
            ie = max(0.0, min(1.0, (b["m_clean"] - mp) / b["denom"]))
            res[str(h)].append({"pair": int(k), "probe": pr["probe"], "ie": ie})
            gc.collect(); torch.cuda.empty_cache()
        del zc, ac; gc.collect()
        log(f"  L{L} pair {k} done")
    A[key] = res; save_a(); log(f"IE_OK L{L}")

def ie_all():
    for L in FA_LAYERS: ie_layer(L)

# ------------------------------------------------------------------ ranking + H1/H2
def rank():
    rows = []
    for L in FA_LAYERS:
        res = A.get(f"ie_L{L}")
        if not res: raise RuntimeError(f"ie_L{L} missing")
        for h in range(NH):
            recs = res[str(h)]
            by_probe = {}
            for r in recs: by_probe.setdefault(r["probe"], []).append(r["ie"])
            ie_mean = float(np.mean([r["ie"] for r in recs]))
            kappa = float(np.mean([1.0 if max(v) >= IE_MIN else 0.0 for v in by_probe.values()]))
            # split-half stability inputs
            e_half = float(np.mean([r["ie"] for r in recs[0::2]])); o_half = float(np.mean([r["ie"] for r in recs[1::2]]))
            rows.append({"L": L, "h": h, "ie": ie_mean, "kappa": kappa, "s": ie_mean * kappa,
                         "half_a": e_half, "half_b": o_half})
    rows.sort(key=lambda r: -r["s"])
    for i, r in enumerate(rows): r["rank"] = i + 1
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr([r["half_a"] for r in rows], [r["half_b"] for r in rows]).statistic)
    except Exception:
        a = np.argsort(np.argsort([r["half_a"] for r in rows])); b = np.argsort(np.argsort([r["half_b"] for r in rows]))
        rho = float(np.corrcoef(a, b)[0, 1])
    # zero-inflation diagnostic (G1 companion): top-20 overlap between halves
    ta = {(r["L"], r["h"]) for r in sorted(rows, key=lambda r: -r["half_a"])[:20]}
    tb = {(r["L"], r["h"]) for r in sorted(rows, key=lambda r: -r["half_b"])[:20]}
    top20_overlap = len(ta & tb)
    def _find(L, h): return next(r for r in rows if r["L"] == L and r["h"] == h)
    verdict = {"split_half_rho": rho, "G1_pass": rho >= 0.8, "top20_overlap": top20_overlap,
               "writers": {f"L{L}h{h}": _find(L, h) for (L, h) in WRITERS},
               "opposers": {f"L{L}h{h}": _find(L, h) for (L, h) in OPPOSERS}}
    for name, K in K_CUTS.items():
        keep = {(r["L"], r["h"]) for r in rows[:K]}
        verdict[name] = {"writers_kept": [f"L{L}h{h}" for (L, h) in WRITERS if (L, h) in keep],
                         "opposers_kept": [f"L{L}h{h}" for (L, h) in OPPOSERS if (L, h) in keep],
                         "retained": sorted([f"L{r['L']}h{r['h']}" for r in rows[:K]])}
    A["ranking"] = rows; A["verdict_stage1"] = verdict; save_a()
    log("RANK_OK", json.dumps({k: v for k, v in verdict.items() if k != "writers"}, default=str)[:400])
    log("H1 writers:", {k: (v["rank"], round(v["s"], 4)) for k, v in verdict["writers"].items()})
    log("H2 opposers:", {k: (v["rank"], round(v["s"], 4)) for k, v in verdict["opposers"].items()})

# ------------------------------------------------------------------ H3: two-sided causal test
def _headset(cond):
    """PREREG primary scope = L59 (the named circuit's layer); late-band secondary."""
    rows = A["ranking"]
    keep25 = {(r["L"], r["h"]) for r in rows[:K_CUTS["K25"]]}
    l59_all = [(L59, h) for h in range(NH)]
    l59_kept = [x for x in l59_all if x in keep25]
    l59_drop = [x for x in l59_all if x not in keep25]
    late = [(L, h) for L in (51, 55, 59, 63) for h in range(NH)]
    rng = random.Random(42)
    sets = {
        "base": [],
        "drop_nonret_L59": l59_drop,                                  # PRIMARY: LA-ify non-retained @L59
        "drop_rand_L59": rng.sample(l59_all, len(l59_drop)),          # null: same size, random
        "drop_ret_L59": l59_kept,                                     # positive control: NIAH must be able to fail
        "drop_writers": list(WRITERS),                                # behavioral anchor
        "drop_opposers": list(OPPOSERS),                              # H2 over-commit branch
        "drop_nonret_late": [x for x in late if x not in keep25],     # secondary severity
        "drop_rand_late": rng.sample(late, len([x for x in late if x not in keep25])),  # size-matched null (random 74)
        "keep_writers_late": [x for x in late if x not in keep25 and x not in set(WRITERS)],  # whitelist-rescue test
    }
    nonret_late = [x for x in late if x not in keep25]
    # M3: extra random draws (order-independent seeds)
    for i, seed in enumerate((43, 44, 45, 46), start=2):
        sets[f"drop_rand_late_s{i}"] = random.Random(seed).sample(late, len(nonret_late))
    # M1: severity control — ablate nonret minus 2 random NON-writer heads (72, size-matched to keep_writers)
    pool = [x for x in nonret_late if x not in set(WRITERS)]
    for i, seed in enumerate((101, 102, 103), start=1):
        keep2 = set(random.Random(seed).sample(pool, 2))
        sets[f"keep2rand_late_s{i}"] = [x for x in nonret_late if x not in keep2]
    return sets[cond]

def _mk_abl(heads_by_L, mode="mean"):
    """Pre-hooks on o_proj: replace each ablated head's input slice with its per-row sequence mean
    (mode='mean') or zeros (mode='zero'), at every position. Returns handles."""
    hs = []
    for L, heads in heads_by_L.items():
        sls = [slice(h * HD, (h + 1) * HD) for h in heads]
        def mk(slices):
            def pre(m, a):
                z = a[0].clone()
                for sl in slices:
                    if mode == "mean":
                        z[:, :, sl] = z[:, :, sl].mean(dim=1, keepdim=True)
                    else:
                        z[:, :, sl] = 0
                return (z,) + a[1:]
            return pre
        hs.append(_oproj(L).register_forward_pre_hook(mk(sls)))
    return hs

def _group(headlist):
    by = {}
    for L, h in headlist: by.setdefault(L, []).append(h)
    return by

def h3_block(cond, mode="mean"):
    """Two-sided: (a) circuit side on the #7 points (P(edit) on BASH+EDIT pts, emit on both),
    (b) benchmark side (NIAH m>0 on usable pairs). Same ablation for both."""
    key = f"h3_{cond}_{mode}"
    if A.get(key): log(f"H3_SKIP {key}"); return
    heads = _group(_headset(cond))
    out = {"cond": cond, "mode": mode, "n_heads": sum(len(v) for v in heads.values())}
    # (b) benchmark side
    bench = []
    for k in A["pairs_ok"]:
        pr = A["probes"][int(k)]
        idc, na, ap, am = _pair_ids(pr, corrupted=False)
        hs = _mk_abl(heads, mode)
        try: m = _readout(idc, na, ap, am)
        finally:
            for h in hs: h.remove()
        bench.append({"pair": int(k), "m": m, "pass": bool(m > 0)})
        gc.collect(); torch.cuda.empty_cache()
    out["niah_pass_rate"] = float(np.mean([b["pass"] for b in bench])); out["niah"] = bench
    # (a) circuit side — P(edit) at last position + greedy emits, EDIT and BASH points
    for side in ("EDIT", "BASH"):
        ps, emits = [], []
        for r in S.S[side]:
            hs = _mk_abl(heads, mode)
            try:
                lg, _ = S._fwd(r["ids"], [])
                ps.append(S._p_action(lg[0, -1], S.TARGET))
                cont = S._gen(r["ids"])
                emits.append(int(S._emits(cont, S.TARGET)))
            finally:
                for h in hs: h.remove()
            gc.collect(); torch.cuda.empty_cache()
        out[f"{side.lower()}_P_edit"] = float(np.mean(ps))
        out[f"{side.lower()}_emit"] = float(np.mean(emits))
        out[f"{side.lower()}_emit_per"] = emits
    A[key] = out; save_a()
    log(f"H3_OK {key}: NIAH {out['niah_pass_rate']:.2f} | edit_emit {out['edit_emit']:.2f} "
        f"bash_emit {out['bash_emit']:.2f} edit_P {out['edit_P_edit']:.3f}")

# ------------------------------------------------------------------ G1 remedy: targeted pair expansion
def build_probes_v2():
    """+6 single +6 mk1 pairs (seed 60), appended as pair ids 12-23 under key probes_v2."""
    if A.get("probes_v2"): log("PROBES_V2_SKIP"); return
    rng = random.Random(60)
    probes = []
    for i in range(N_PAIRS):
        key = KEYS[(i + 3) % len(KEYS)]; num = _num(rng); corr = _corrupt(num, rng)
        depth = [0.28, 0.42, 0.58, 0.7, 0.33, 0.62][i]
        probes.append({"probe": "single", "pair": 12 + i, "qkey": key, "ans": num, "ans_corr": corr,
                       "needles": [[key, num, depth]], "needles_corr": [[key, corr, depth]]})
    for i in range(N_PAIRS):
        ks = [KEYS[(i + 2 + j) % len(KEYS)] for j in range(3)]
        ns = [_num(rng) for _ in range(3)]; qi = (i + 1) % 3
        corr = _corrupt(ns[qi], rng)
        ds = [[0.22, 0.48, 0.72], [0.35, 0.6, 0.78], [0.27, 0.52, 0.66]][i % 3]
        needles = [[ks[j], ns[j], ds[j]] for j in range(3)]
        nc = [list(x) for x in needles]; nc[qi][1] = corr
        probes.append({"probe": "mk1", "pair": 18 + i, "qkey": ks[qi], "ans": ns[qi], "ans_corr": corr,
                       "needles": needles, "needles_corr": nc})
    A["probes_v2"] = probes; save_a(); log(f"PROBES_V2_OK n={len(probes)}")

def bases_v2():
    if A.get("bases_v2"): log("BASES_V2_SKIP"); return
    out = {}
    for k, pr in enumerate(A["probes_v2"]):
        idc, na, ap, am = _pair_ids(pr, corrupted=False)
        idx, _, _, _ = _pair_ids(pr, corrupted=True)
        if len(ap) != len(_ans_ids(pr["ans_corr"])):
            out[str(k)] = {"skip": "anslen"}; continue
        m_clean = _readout(idc, na, ap, am); m_corr = _readout(idx, na, ap, am)
        out[str(k)] = {"m_clean": m_clean, "m_corr": m_corr, "denom": m_clean - m_corr}
        gc.collect(); torch.cuda.empty_cache()
    A["bases_v2"] = out
    A["pairs_ok_v2"] = [k for k, v in out.items() if v.get("denom", 0) and v["denom"] > DENOM_MIN]
    save_a(); log(f"BASES_V2_OK usable {len(A['pairs_ok_v2'])}/{len(out)}")

def _targets():
    rows = A["ranking"]
    t = {(r["L"], r["h"]) for r in rows if r["s"] > 0} | set(WRITERS) | set(OPPOSERS)
    return sorted(t)

def ie_targets():
    """Receiver IE on the v2 pairs, only for decision-relevant heads. Checkpoints per pair."""
    done = A.get("ie_v2", {})
    targets = _targets()
    by_layer = {}
    for (L, h) in targets: by_layer.setdefault(L, []).append(h)
    for k in A["pairs_ok_v2"]:
        if k in done.get("_pairs_done", []): continue
        pr = A["probes_v2"][int(k)]; b = A["bases_v2"][k]
        idc, na, ap, am = _pair_ids(pr, corrupted=False)
        zc_all = _cache_z_corr(pr)
        ac = _cache_attn_clean(idc)
        for L, heads in sorted(by_layer.items()):
            for h in heads:
                mp = _patched_m(idc, na, ap, am, L, h, zc_all[L], ac)
                ie = max(0.0, min(1.0, (b["m_clean"] - mp) / b["denom"]))
                done.setdefault(f"L{L}h{h}", []).append({"pair": int(k) + 100, "probe": pr["probe"], "ie": ie})
                gc.collect(); torch.cuda.empty_cache()
        done.setdefault("_pairs_done", []).append(k)
        A["ie_v2"] = done; save_a(); log(f"IE_V2 pair {k} done ({len(targets)} heads)")
    log("IE_V2_OK")

def stability_v2():
    """Combined v1+v2 means for targets; verify the Stage-1 verdict is unchanged."""
    out = {}
    for (L, h) in _targets():
        v1 = [r["ie"] for r in A[f"ie_L{L}"][str(h)]]
        v2 = [r["ie"] for r in A.get("ie_v2", {}).get(f"L{L}h{h}", [])]
        out[f"L{L}h{h}"] = {"v1": float(np.mean(v1)), "v2": float(np.mean(v2)) if v2 else None,
                            "combined": float(np.mean(v1 + v2)) if v2 else float(np.mean(v1)), "n": len(v1) + len(v2)}
    A["stability_v2"] = out; save_a()
    for name in ("L59h8", "L59h6", "L59h3", "L59h21", "L59h19"):
        log(f"  {name}: v1 {out[name]['v1']:.4f} | v2 {out[name]['v2']:.4f} | combined {out[name]['combined']:.4f}")
    log("STABILITY_V2_OK")

def finalize():
    save_a()
    log("FINALIZED —", AFILE, "on HF")
    print("HYDRA_JSON_BEGIN"); print(json.dumps(A.get("verdict_stage1", {}))); print("HYDRA_JSON_END", flush=True)
