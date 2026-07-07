"""Is the agent's action in the workspace? (J-lens x commitment) — beat-11.
PREREG: paper/circuit_breaker/PREREG_workspace_action.md

Tests whether an agent's action commitment passes through the verbalizable global
workspace (J-space, Anthropic transformer-circuits.pub/2026/workspace) or bypasses it,
entering verbalizable coordinates only in the motor regime.

Estimator (scalar-row J-lens, pre-declared): for token v and layer L, the J-lens row is
    g_{v,L,t} = d[ sum_{t'} logit_v(t') ] / d h_{L,t}
computed by ONE backward per (v, prompt). The causal mask makes t' >= t automatic; we
normalize each position t by its future-count (T - t) and average over prompts. Params are
frozen and the embedding output is made the sole grad leaf, so a single backward yields the
row at every layer with NO parameter gradients (memory-light) — gradient checkpointing on.

Driven cell-by-cell, resumable via HF ledger results/jspace_action.json:
    import sys; sys.path.insert(0,'/content/openinterp-swebench-harness/scripts')
    import commit_lever_stages as S, jspace_action as J
    S.setup(); S.capture()                 # model + 120 decision points + residuals
    J.load_a(); J.prep()                   # freeze, grad-checkpoint, choose control tokens
    J.est_set()                            # build estimation prompts (agent + generic)
    J.jlens('actions'); J.jlens('answers') # estimate g vectors -> HF .pt
    J.g0()                                 # harness fidelity + estimator sanity + null-swap no-op
    J.g0prime()                            # THE gate: 2-hop answer-swap positive control (>=25% flip)
    J.capture_resids()                     # decision-point residuals at all readout layers
    J.h1()                                 # readout: action-token signal by depth (AUROC)
    J.h2('midW'); J.h2('tail'); J.h2('motor')   # causal swaps in 3 fixed bands
    J.h3()                                 # top-10 J-space ablation at decision points
    J.finalize()
"""
import os, json, gc, time, random, math
import torch, numpy as np
from huggingface_hub import hf_hub_download, upload_file
import commit_lever_stages as S

AFILE = "results/jspace_action.json"
VFILE = "results/jspace_vectors.pt"                 # estimated g vectors (HF-mirrored)
NL = 64                                             # Qwen3.6-27B layers; depth% = L/NL*100

# readout depths (~16) span sensory->motor
READOUT_L = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
# fixed bands (PREREG, depth arithmetic source-verified)
BANDS = {"midW": [27, 31, 35, 39, 43],             # 42-67%
         "tail": [47, 51, 55],                      # 73-86%
         "motor": [59, 63]}                         # 92-98%
WS_BAND = [27, 31, 35, 39, 43, 47, 51, 55]          # workspace band (38-92%) for G0' + H3
JLENS_L = sorted(set(READOUT_L) | set(WS_BAND) | {59, 63})

ACTIONS = {"str_replace_editor": 15462, "bash": 21402, "finish": 28}   # verified single-token
# candidate control words (single-token verified at runtime in prep())
CTRL_WORDS = [" file", " data", " value", " number", " result", " system",
              " code", " error", " path", " name", " test", " line"]
EST_CTX = 1536                                       # estimation context cap (OOM fallback 1024)
N_AGENT = 150; N_GENERIC = 50
LAM_ID = "sigma"                                     # swap tag
TOPK_ABL = 10                                        # their top-k J-space ablation

A = {}                                               # ledger (mirrors HF AFILE)
G = {}                                               # in-mem g vectors {group: {tok: {L: tensor}}}

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def _tok(): return os.environ.get("HF_TOKEN") or None
def _dev(): return S.S["model"].device

def save_a():
    json.dump(A, open("/content/ja.json", "w"), indent=1)
    upload_file(path_or_fileobj="/content/ja.json", path_in_repo=AFILE,
                repo_id=S.DATA_REPO, repo_type="dataset", token=_tok())

def load_a():
    global A
    try:
        p = hf_hub_download(S.DATA_REPO, AFILE, repo_type="dataset", token=_tok(), force_download=True)
        A = json.load(open(p)); log("resumed jspace ledger:", sorted(A.keys()))
    except Exception:
        A = {}; log("fresh jspace ledger")

# --------------------------------------------------------------- vector persistence
def _save_vecs():
    torch.save(G, "/content/jvec.pt")
    upload_file(path_or_fileobj="/content/jvec.pt", path_in_repo=VFILE,
                repo_id=S.DATA_REPO, repo_type="dataset", token=_tok())
    log("vectors saved to HF")

def load_vecs():
    global G
    try:
        p = hf_hub_download(S.DATA_REPO, VFILE, repo_type="dataset", token=_tok(), force_download=True)
        G = torch.load(p, map_location="cpu"); log("resumed vectors:", list(G.keys()))
    except Exception:
        G = {}; log("no vectors on HF")

# --------------------------------------------------------------- prep (freeze, grad-checkpoint, tokens)
def _emb():
    m = S.S["model"]
    for path in ("model.language_model.embed_tokens", "language_model.embed_tokens",
                 "model.model.embed_tokens", "model.embed_tokens"):
        cur = m; ok = True
        for p in path.split("."):
            if not hasattr(cur, p): ok = False; break
            cur = getattr(cur, p)
        if ok: return cur
    raise RuntimeError("embed_tokens not found")

def prep():
    m = S.S["model"]; tok = S.S["tok"]
    m.requires_grad_(False)
    try:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.config.use_cache = False
    except Exception as e:
        log("grad-checkpoint enable warn:", e)
        try: m.gradient_checkpointing_enable(); m.config.use_cache = False
        except Exception as e2: log("grad-checkpoint 2nd warn:", e2)
    # choose 6 single-token controls deterministically
    if "ctrl" in A:
        log("ctrl tokens (ledger):", A["ctrl"])
    else:
        ctrl = {}
        for w in CTRL_WORDS:
            ids = tok(w, add_special_tokens=False).input_ids
            if len(ids) == 1 and ids[0] not in ACTIONS.values():
                ctrl[w.strip()] = ids[0]
            if len(ctrl) >= 6: break
        A["ctrl"] = ctrl; save_a(); log("ctrl tokens:", ctrl)
    # verify action tokens still single-token on THIS tokenizer
    ver = {k: tok(k, add_special_tokens=False).input_ids for k in ACTIONS}
    A["action_tok_check"] = {k: (ACTIONS[k], ver[k]) for k in ACTIONS}
    save_a(); log("action tok check:", A["action_tok_check"])
    log("PREP_OK")

def prep_no_ckpt():
    """Fallback if reentrant checkpointing swallowed grads: disable checkpointing."""
    m = S.S["model"]
    try: m.gradient_checkpointing_disable()
    except Exception as e: log("disable warn:", e)
    m.config.use_cache = False
    log("PREP_NO_CKPT_OK (checkpointing off)")

# --------------------------------------------------------------- estimation set
def est_set():
    """150 agent-distribution windows (from captured decision points, <=EST_CTX) + 50 generic."""
    if "est_n" in A: log("EST_SKIP", A["est_n"]); return
    agent = []
    pts = list(S.S["EDIT"]) + list(S.S["BASH"])           # 120 on-distribution contexts
    for r in pts:
        ids = r["ids"][0].tolist()
        agent.append(ids[-EST_CTX:])
    # expand to 150: add earlier windows (first EST_CTX) of the longest contexts
    longs = sorted(pts, key=lambda r: r["ids"].shape[1], reverse=True)
    for r in longs:
        if len(agent) >= N_AGENT: break
        ids = r["ids"][0].tolist()
        if len(ids) > EST_CTX: agent.append(ids[:EST_CTX])
    agent = agent[:N_AGENT]
    # generic text windows (deterministic filler sentences, tokenized)
    tok = S.S["tok"]
    gen_src = (S.FILLER if hasattr(S, "FILLER") else
               "The history of science is a history of careful measurement and honest error. "
               "A river runs through the valley where the old mill turns slowly with the seasons. "
               "In mathematics a proof is a chain of statements each following from the last. ")
    generic = []
    base = tok((gen_src * 60), add_special_tokens=False).input_ids
    step = max(1, (len(base) - 256) // N_GENERIC)
    for i in range(N_GENERIC):
        s = i * step
        generic.append(base[s:s + 256])
    A["est_n"] = {"agent": len(agent), "generic": len(generic), "ctx": EST_CTX}
    torch.save({"agent": agent, "generic": generic}, "/content/est.pt")
    upload_file(path_or_fileobj="/content/est.pt", path_in_repo="results/jspace_est.pt",
                repo_id=S.DATA_REPO, repo_type="dataset", token=_tok())
    save_a(); log("EST_OK", A["est_n"])

def _load_est():
    p = hf_hub_download(S.DATA_REPO, "results/jspace_est.pt", repo_type="dataset", token=_tok(), force_download=True)
    return torch.load(p)

# --------------------------------------------------------------- J-lens estimator (scalar-row)
def _grad_rows(ids_list, vocab_ids, layers):
    """One backward per (prompt, v): returns {v: {L: mean-over-t normalized row}} averaged over prompts."""
    m = S.S["model"]; emb = _emb()
    acc = {v: {L: torch.zeros(m.config.hidden_size, dtype=torch.float32) for L in layers} for v in vocab_ids}
    cnt = 0
    for pi, ids in enumerate(ids_list):
        x = torch.tensor([ids], device=_dev())
        for v in vocab_ids:
            store = {}
            heh = emb.register_forward_hook(lambda mod, i, o: o.requires_grad_(True))
            hs = []
            def mk(L):
                def h(mod, i, o):
                    t = o[0] if isinstance(o, tuple) else o
                    t.retain_grad(); store[L] = t
                return h
            for L in layers: hs.append(S._layer(L).register_forward_hook(mk(L)))
            try:
                out = m(input_ids=x, use_cache=False)
                lg = out.logits[0].float()                         # [T, V]
                T = lg.shape[0]
                loss = lg[:, v].sum()                              # sum over positions of logit_v
                m.zero_grad(set_to_none=True)
                loss.backward()
                for L in layers:
                    if store[L].grad is None:
                        raise RuntimeError(f"grad is None at L{L} — reentrant gradient-checkpointing "
                                           "swallowed retained grads; call prep_no_ckpt() and lower EST_CTX")
                    g = store[L].grad[0].float()                  # [T, d]
                    fut = torch.arange(T, 0, -1, dtype=torch.float32, device=g.device).clamp(min=1).unsqueeze(1)
                    row = (g / fut).mean(0).cpu()                  # per-t normalize by future-count, mean over t
                    acc[v][L] += row
            finally:
                heh.remove()
                for h in hs: h.remove()
                for L in list(store): store[L] = None
                del out, lg
                gc.collect(); torch.cuda.empty_cache()
        cnt += 1
        if (pi + 1) % 20 == 0: log(f"  est {pi+1}/{len(ids_list)}")
    for v in vocab_ids:
        for L in layers: acc[v][L] /= max(cnt, 1)
    return acc

def jlens(group):
    """group='actions' -> action+control tokens over agent set; 'answers' -> QA-pool tokens over generic set."""
    load_vecs()
    if group in G: log(f"JLENS_SKIP {group}"); return
    est = _load_est()
    if group == "actions":
        vocab = {**{k: v for k, v in ACTIONS.items()}, **A["ctrl"]}
        ids_list = est["agent"]
    elif group == "answers":
        vocab = _qa_pool()[0]                                     # {word: tokid}
        ids_list = est["generic"]
    else:
        raise ValueError(group)
    inv = {tid: name for name, tid in vocab.items()}
    rows = _grad_rows(ids_list, list(vocab.values()), JLENS_L)
    G[group] = {inv[tid]: {L: rows[tid][L] for L in JLENS_L} for tid in vocab.values()}
    _save_vecs()
    A.setdefault("jlens_done", []).append(group); save_a()
    log(f"JLENS_OK {group} tokens={list(vocab)}")

# --------------------------------------------------------------- 2-hop QA positive control (G0')
def _qa_pool():
    """Deterministic single-token answer pool + 20 two-hop items. Returns (pool{word:tokid}, items)."""
    tok = S.S["tok"]
    cand = [" red", " blue", " green", " gold", " black", " white", " north", " south",
            " east", " west", " king", " queen", " sun", " moon", " fire", " water"]
    pool = {}
    for w in cand:
        ids = tok(w, add_special_tokens=False).input_ids
        if len(ids) == 1: pool[w.strip()] = ids[0]
    words = list(pool)                                           # single-token answers
    rng = random.Random(11)
    syms = ["A", "B", "C", "D", "E", "F", "G", "H"]
    items = []
    for i in range(20):
        w = rng.sample(words, 4)                                 # 4 leaf answers
        mids = rng.sample(["K", "M", "P", "Q"], 4)
        # chain: leftsym -> mid -> answer  (two hops)
        lefts = syms[:4]; rng.shuffle(lefts)
        rules_mid = [f"{lefts[j]} maps to {mids[j]}." for j in range(4)]
        rules_ans = [f"{mids[j]} maps to {w[j]}." for j in range(4)]
        rng.shuffle(rules_mid); rng.shuffle(rules_ans)
        qi = rng.randrange(4)
        correct = w[qi]; alt = w[(qi + 1) % 4]
        prompt = ("Rules: " + " ".join(rules_mid) + " " + " ".join(rules_ans) +
                  f" Question: In two steps, {lefts[qi]} maps to? Answer:")
        items.append({"prompt": prompt, "correct": correct, "alt": alt})
    return pool, items

# --------------------------------------------------------------- swap machinery
def _swap_hooks(pairs, layers, sigma=True):
    """pairs: {L: (V[d,2])}; swap the two coordinates at last position. sigma=False -> identity (no-op)."""
    hs = []
    def mk(L):
        V = pairs[L].to(_dev())
        Vp = torch.linalg.pinv(V.float())                        # [2,d]
        def h(mod, i, o):
            t = o[0] if isinstance(o, tuple) else o
            t = t.clone()
            h_last = t[:, -1, :].float()                         # [B,d]
            c = (Vp @ h_last.T).T                                # [B,2]
            c2 = c.flip(1) if sigma else c                       # coordinate exchange
            delta = ((c2 - c) @ V.float().T)                     # [B,d]
            t[:, -1, :] = (h_last + delta).to(t.dtype)
            return (t, *o[1:]) if isinstance(o, tuple) else t
        return h
    for L in layers: hs.append(S._layer(L).register_forward_hook(mk(L)))
    return hs

def _logits_last(ids):
    with torch.no_grad():
        lg = S.S["model"](input_ids=ids.to(_dev()), use_cache=False, logits_to_keep=1).logits
    return lg[0, -1].float()

def g0():
    """Harness fidelity + estimator sanity (upcoming-token rank vs logit-lens) + null-swap no-op."""
    if "g0" in A: log("G0_SKIP", A["g0"]); return
    load_vecs()
    fid = {"edit_baseP": float(np.mean([r["baseP_edit"] for r in S.S["EDIT"]])),
           "bash_baseP": float(np.mean([r["baseP_edit"] for r in S.S["BASH"]]))}
    # estimator sanity: on generic windows, does g_v . h_L rank the ACTUAL next token above controls at late L?
    est = _load_est(); tok = S.S["tok"]
    # null-swap no-op: sigma=identity must not change logits
    r0 = S.S["EDIT"][0]["ids"]
    base = _logits_last(r0)
    Vpair = {L: torch.stack([G["actions"]["str_replace_editor"][L], G["actions"]["bash"][L]], 1).float()
             for L in BANDS["motor"]}
    hs = _swap_hooks(Vpair, BANDS["motor"], sigma=False)
    try: noop = _logits_last(r0)
    finally:
        for h in hs: h.remove()
    d_noop = float((noop - base).abs().max())
    A["g0"] = {"fidelity": fid, "null_swap_maxabs_logit": d_noop,
               "null_swap_ok": d_noop < 1e-2}
    save_a(); log("G0:", json.dumps(A["g0"]))

def g0prime():
    """THE gate: workspace-band J-lens answer-swap flips >=25% of 20 two-hop answers top-1."""
    if "g0prime" in A: log("G0'_SKIP", A["g0prime"]); return
    load_vecs()
    pool, items = _qa_pool(); tok = S.S["tok"]
    flips, correct_base, n = 0, 0, 0
    detail = []
    for it in items:
        ids = torch.tensor([tok(it["prompt"], add_special_tokens=False).input_ids])
        base = _logits_last(ids)
        top_base = int(base.argmax())
        cid, aid = pool[it["correct"]], pool[it["alt"]]
        base_correct = (top_base == cid)
        Vpair = {L: torch.stack([G["answers"][it["correct"]][L], G["answers"][it["alt"]][L]], 1).float()
                 for L in WS_BAND}
        hs = _swap_hooks(Vpair, WS_BAND, sigma=True)
        try: sw = _logits_last(ids)
        finally:
            for h in hs: h.remove()
        top_sw = int(sw.argmax())
        flipped = base_correct and (top_sw == aid)               # correct->alt under swap
        flips += int(flipped); correct_base += int(base_correct); n += 1
        detail.append({"corr": it["correct"], "alt": it["alt"],
                       "base_correct": base_correct, "flipped_to_alt": flipped})
        gc.collect(); torch.cuda.empty_cache()
    rate = flips / max(correct_base, 1)
    A["g0prime"] = {"n": n, "base_correct": correct_base, "flips_corr_to_alt": flips,
                    "flip_rate_of_correct": rate, "pass": rate >= 0.25, "detail": detail}
    save_a()
    log(f"G0': base_correct={correct_base}/{n} flips={flips} rate={rate:.2f} pass={rate>=0.25}")

# --------------------------------------------------------------- decision-point residuals (readout)
def capture_resids():
    if "resids" in A: log("RESIDS_SKIP"); return
    def rr(rows):
        out = {L: [] for L in JLENS_L}
        for r in rows:
            _, caps = S._fwd(r["ids"], JLENS_L)
            for L in JLENS_L: out[L].append(caps[L][0, 0].float().cpu())
            gc.collect(); torch.cuda.empty_cache()
        return {L: torch.stack(out[L]) for L in JLENS_L}
    E = rr(S.S["EDIT"]); B = rr(S.S["BASH"])
    torch.save({"E": E, "B": B}, "/content/jresid.pt")
    upload_file(path_or_fileobj="/content/jresid.pt", path_in_repo="results/jspace_resid.pt",
                repo_id=S.DATA_REPO, repo_type="dataset", token=_tok())
    A["resids"] = {L: True for L in JLENS_L}; save_a(); log("RESIDS_OK")

def _load_resid():
    p = hf_hub_download(S.DATA_REPO, "results/jspace_resid.pt", repo_type="dataset", token=_tok(), force_download=True)
    return torch.load(p)

def _auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))

def h1():
    """Readout: at each depth, does g_action . h separate edit-points from bash-points? (AUROC)"""
    if "h1" in A: log("H1_SKIP"); return
    load_vecs(); R = _load_resid()
    ge = G["actions"]["str_replace_editor"]; gb = G["actions"]["bash"]
    res = {}
    for L in READOUT_L:
        E, B = R["E"][L], R["B"][L]                              # [n,d]
        # score = margin g_edit.h - g_bash.h ; edit-pts should score higher
        se_E = (E @ ge[L]) - (E @ gb[L]); se_B = (B @ ge[L]) - (B @ gb[L])
        au = _auroc(se_E.numpy(), se_B.numpy())
        # split-half stability
        n = len(se_E); half = n // 2
        au1 = _auroc(se_E[:half].numpy(), se_B[:half].numpy())
        au2 = _auroc(se_E[half:].numpy(), se_B[half:].numpy())
        res[L] = {"depth_pct": round(100 * L / NL, 1), "auroc": au,
                  "auroc_half1": au1, "auroc_half2": au2,
                  "edit_margin_mean": float(se_E.mean()), "bash_margin_mean": float(se_B.mean())}
        log(f"  H1 L{L:2d} ({res[L]['depth_pct']}%) AUROC={au:.3f}  half=({au1:.2f},{au2:.2f})")
    A["h1"] = res; save_a(); log("H1_OK")

# --------------------------------------------------------------- H2 causal swaps
def h2(band):
    key = f"h2_{band}"
    if key in A: log(f"{key}_SKIP"); return
    load_vecs()
    layers = BANDS[band]
    ge = G["actions"]["str_replace_editor"]; gb = G["actions"]["bash"]
    Vpair = {L: torch.stack([ge[L], gb[L]], 1).float() for L in layers}
    def run(rows, sigma):
        pe, emit = [], []
        for i, r in enumerate(rows):
            hs = _swap_hooks(Vpair, layers, sigma=sigma)      # P(edit) under swap
            try: pe.append(S._p_action(_logits_last(r["ids"]), S.TARGET))
            finally:
                for h in hs: h.remove()
            cont = _gen_swapped(r["ids"], Vpair, layers, sigma)   # emit under swap (own hooks)
            emit.append(int(S._emits(cont, S.TARGET)))
            gc.collect(); torch.cuda.empty_cache()
        return pe, emit
    # swap bash<->edit on EDIT points (should it flip to bash?) and BASH points
    pe_E, em_E = run(S.S["EDIT"], True)
    pe_B, em_B = run(S.S["BASH"], True)
    # null: norm-matched random-direction swap
    rng = torch.Generator().manual_seed(59)
    Vrand = {L: _norm_matched_random(Vpair[L], rng) for L in layers}
    def run_rand(rows):
        pe = []
        for r in rows:
            hs = _swap_hooks(Vrand, layers, sigma=True)
            try: pe.append(S._p_action(_logits_last(r["ids"]), S.TARGET))
            finally:
                for h in hs: h.remove()
            gc.collect(); torch.cuda.empty_cache()
        return pe
    pe_E_rand = run_rand(S.S["EDIT"])
    A[key] = {"layers": layers,
              "edit_pts": {"P_edit_swapped": float(np.mean(pe_E)), "emit_edit": float(np.mean(em_E)),
                           "per_emit": em_E},
              "bash_pts": {"P_edit_swapped": float(np.mean(pe_B)), "emit_edit": float(np.mean(em_B)),
                           "per_emit": em_B},
              "edit_pts_randnull_P_edit": float(np.mean(pe_E_rand))}
    save_a(); log(f"{key}_OK P_edit(EDIT)={np.mean(pe_E):.3f} emit(EDIT)={np.mean(em_E):.2f} "
                  f"P_edit(BASH)={np.mean(pe_B):.3f} rand-null={np.mean(pe_E_rand):.3f}")

def _norm_matched_random(V, gen):
    d = V.shape[0]
    R = torch.randn(d, 2, generator=gen)
    R = R / R.norm(dim=0, keepdim=True) * V.norm(dim=0, keepdim=True)
    return R

def _gen_swapped(ids, Vpair, layers, sigma):
    hs = _swap_hooks(Vpair, layers, sigma=sigma)
    try:
        tok, model = S.S["tok"], S.S["model"]
        with torch.no_grad():
            out = model.generate(input_ids=ids.to(_dev()), max_new_tokens=S.MAXNEW, do_sample=False,
                                 use_cache=True, pad_token_id=tok.eos_token_id,
                                 attention_mask=torch.ones_like(ids).to(_dev()))
        cont = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
    finally:
        for h in hs: h.remove()
    return cont

# --------------------------------------------------------------- H3 top-10 workspace ablation
def h3():
    if "h3" in A: log("H3_SKIP"); return
    load_vecs()
    # top-10 J-space directions by lens activation on the agent distribution, per layer of WS_BAND.
    # Approximation (pre-declared row-restricted variant): use the estimated action+control rows +
    # answer-pool rows as the verbalizable J-space basis; take the k=10 with largest mean |g.h| on
    # captured decision points as the workspace directions to project out.
    R = _load_resid()
    allrows = {}
    for grp in ("actions", "answers"):
        for name, perL in G.get(grp, {}).items():
            allrows[f"{grp}:{name}"] = perL
    res = {}
    for L in WS_BAND:
        H = torch.cat([R["E"][L], R["B"][L]], 0)                  # [n,d]
        acts = {nm: float((H @ perL[L]).abs().mean()) for nm, perL in allrows.items()}
        top = sorted(acts, key=acts.get, reverse=True)[:TOPK_ABL]
        res[L] = {"top_dirs": top}
    # project out top-k at every WS layer, measure emit differential at decision points
    def run(rows):
        pe = []
        for r in rows:
            hs = []
            for L in WS_BAND:
                V = torch.stack([allrows[nm][L] for nm in res[L]["top_dirs"]], 1).float()  # [d,k]
                Q, _ = torch.linalg.qr(V)
                hs.append(S._layer(L).register_forward_hook(_abl_hook(Q)))
            try: pe.append(S._p_action(_logits_last(r["ids"]), S.TARGET))
            finally:
                for h in hs: h.remove()
            gc.collect(); torch.cuda.empty_cache()
        return pe
    pe_E = run(S.S["EDIT"]); pe_B = run(S.S["BASH"])
    base_E = float(np.mean([r["baseP_edit"] for r in S.S["EDIT"]]))
    base_B = float(np.mean([r["baseP_edit"] for r in S.S["BASH"]]))
    A["h3"] = {"topdirs": {str(L): res[L]["top_dirs"] for L in WS_BAND},
               "P_edit_ablated": {"edit_pts": float(np.mean(pe_E)), "bash_pts": float(np.mean(pe_B))},
               "P_edit_base": {"edit_pts": base_E, "bash_pts": base_B},
               "diff_survives": (np.mean(pe_E) - np.mean(pe_B))}
    save_a(); log(f"H3_OK ablated diff={np.mean(pe_E)-np.mean(pe_B):+.3f} "
                  f"(base {base_E-base_B:+.3f}); E {np.mean(pe_E):.3f} B {np.mean(pe_B):.3f}")

def _abl_hook(Q):
    def h(mod, i, o):
        t = o[0] if isinstance(o, tuple) else o
        t = t.clone(); x = t[:, -1, :].float()
        x = x - (x @ Q) @ Q.T                                    # remove top-k subspace
        t[:, -1, :] = x.to(t.dtype)
        return (t, *o[1:]) if isinstance(o, tuple) else t
    return h

def finalize():
    upload_file(path_or_fileobj="/content/ja.json", path_in_repo="results/jspace_results.json",
                repo_id=S.DATA_REPO, repo_type="dataset", token=_tok())
    log("FINALIZED — results/jspace_results.json on HF")
