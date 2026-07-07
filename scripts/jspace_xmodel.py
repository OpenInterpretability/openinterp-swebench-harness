"""Cross-model replication of the beat-11 dissociation (#1): does the ANSWER steer flip at
mid-workspace while the ACTION does not, on a SECOND model? Arch-agnostic; default gpt-oss-20b.

Reuses the generic real-decision-point construction of cross_model_realpts.py (SYS+FORCE JSON tool
choice from released SWE-bench trajectories) so the target model genuinely decides the next tool LATE.
Same scalar-row J-lens estimator (final-position diagonal) + validated steer; bands mapped by depth%.

Focused on the DECISIVE dissociation, not the full H1/H3 curve:
  setup(); build_points(); est_set()
  jlens('actions'); jlens('answers')      # chunk-resumable
  spec()                                  # instrument validity on THIS model
  dissoc()                                # answer & action steer per band + random control
  finalize()
Run cell-by-cell (driver), MODEL env picks the model. Ledger results/jspace_xmodel_<model>.json.
"""
import os, json, gc, glob, tarfile, time, random
import torch, numpy as np
from huggingface_hub import hf_hub_download, upload_file

MODEL_ID = os.environ.get("MODEL", "openai/gpt-oss-20b")
SAFE = MODEL_ID.split("/")[-1]
REPO = "caiovicentino1/swebench-phase6-verdict-circuit"
AFILE = f"results/jspace_xmodel_{SAFE}.json"
VFILE = f"results/jspace_xmodel_vec_{SAFE}.pt"
PFILE = f"results/jspace_xmodel_pts_{SAFE}.pt"
NEDIT = NBASH = 40
EST_CTX = 1024; N_GEN = 30; CHUNK = 12
SYS = ("You are an autonomous coding agent solving a software issue in a repository. Tools: 'bash' (run a shell "
       "command — reversible exploration), 'edit' (modify a file — COMMITS a change), 'finish' (submit). Decide "
       'the single next tool. Respond ONLY with JSON: {"tool": "<bash|edit|finish>", "args": {...}}')
FORCE = '{"tool": "'
CAND = [" red", " blue", " green", " gold", " black", " white", " north", " south",
        " east", " west", " king", " queen", " sun", " moon", " fire", " water"]

M = {}; A = {}; G = {}
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def _tok(): return os.environ.get("HF_TOKEN") or None
def _dev(): return M["model"].device

def save_a():
    json.dump(A, open("/content/xa.json", "w"), indent=1)
    upload_file(path_or_fileobj="/content/xa.json", path_in_repo=AFILE, repo_id=REPO, repo_type="dataset", token=_tok())
def load_a():
    global A
    try: A = json.load(open(hf_hub_download(REPO, AFILE, repo_type="dataset", token=_tok(), force_download=True))); log("resumed", sorted(A))
    except Exception: A = {}; log("fresh ledger")

def setup():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("loading", MODEL_ID)
    M["tok"] = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    M["model"] = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                                      device_map="auto", trust_remote_code=True).eval()
    cfg = getattr(M["model"].config, "text_config", M["model"].config)
    M["NL"] = cfg.num_hidden_layers; M["H"] = cfg.hidden_size
    M["model"].requires_grad_(False)
    try: M["model"].gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as e: log("gc warn", e)
    load_a()
    tokd = M["tok"]
    def ft(s): return tokd(s, add_special_tokens=False).input_ids
    M["TOOL"] = {"edit": ft("edit")[0], "bash": ft("bash")[0], "finish": ft("finish")[0]}
    M["POOL"] = {w.strip(): ft(w)[0] for w in CAND if len(ft(w)) == 1}
    # depth bands
    NL = M["NL"]
    def band(a, b): return [L for L in range(NL) if a <= L / NL < b] or [int((a + b) / 2 * NL)]
    M["BANDS"] = {"midW": band(0.42, 0.68), "tail": band(0.72, 0.87), "motor": band(0.90, 1.0)}
    M["READOUT_L"] = sorted(set(sum(M["BANDS"].values(), [])) | set(range(0, NL, max(1, NL // 12))))
    A.setdefault("meta", {"model": MODEL_ID, "NL": NL, "bands": M["BANDS"], "tool_tok": M["TOOL"], "pool": M["POOL"]})
    save_a(); log(f"SETUP_OK NL={NL} bands={M['BANDS']} tools={M['TOOL']}")

def _layer(L):
    m = M["model"]
    for path in ("model.layers", "model.model.layers", "model.language_model.layers"):
        cur = m; ok = True
        for p in path.split("."):
            if not hasattr(cur, p): ok = False; break
            cur = getattr(cur, p)
        if ok:
            try: return cur[L]
            except Exception: continue
    raise RuntimeError("no layers")

def _emb():
    m = M["model"]
    for path in ("model.embed_tokens", "model.model.embed_tokens", "model.language_model.embed_tokens"):
        cur = m; ok = True
        for p in path.split("."):
            if not hasattr(cur, p): ok = False; break
            cur = getattr(cur, p)
        if ok: return cur
    raise RuntimeError("no embed")

def _p_edit(lg):
    ids = [M["TOOL"]["bash"], M["TOOL"]["edit"], M["TOOL"]["finish"]]
    return float(torch.softmax(lg[ids].float(), -1)[1])
def _logits_last(ids):
    with torch.no_grad():
        return M["model"](input_ids=ids.to(_dev()), use_cache=False).logits[0, -1].float()

# ---------------- real decision points (generic format)
def build_points():
    if A.get("fidelity"):
        st = torch.load(hf_hub_download(REPO, PFILE, repo_type="dataset", token=_tok(), force_download=True))
        M["EDIT"], M["BASH"] = st["EDIT"], st["BASH"]; log("points resumed", len(M["EDIT"]), len(M["BASH"])); return
    from datasets import load_dataset
    os.makedirs("/content/rp", exist_ok=True)
    with tarfile.open(hf_hub_download(REPO, "traces.tar.gz", repo_type="dataset", token=_tok())) as t: t.extractall("/content/rp")
    traces = sorted(glob.glob("/content/rp/traces/instance_*.json"))
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test"); by = {r["instance_id"]: r for r in ds if r.get("instance_id")}
    tok = M["tok"]
    def prob(iid):
        r = by.get(iid) or next((v for k, v in by.items() if k in iid or iid in k), None)
        if not r: return None
        for k in ("problem_statement", "issue_text", "text"):
            if r.get(k): return str(r[k])
    def msgs(tr, problem, k):
        ms = [{"role": "system", "content": SYS}, {"role": "user", "content": problem[:5000]}]
        for tn in tr["turns"][:k]:
            tcs = tn.get("tool_calls") or []
            if tcs:
                nm = "edit" if tcs[0].get("name") == "str_replace_editor" else tcs[0].get("name")
                ms.append({"role": "assistant", "content": json.dumps({"tool": nm, "args": tcs[0].get("args", {})})[:1200]})
            for r_ in (tn.get("tool_results") or []):
                ms.append({"role": "user", "content": "[tool result] " + json.dumps(r_.get("result"), ensure_ascii=False)[:1200]})
        return ms
    def ids_at(tr, problem, k):
        try: pre = tok.apply_chat_template(msgs(tr, problem, k), add_generation_prompt=True, tokenize=False)
        except Exception: pre = SYS + "\n" + problem[:5000] + "\nAssistant: "
        return torch.tensor([tok(pre + FORCE, add_special_tokens=False).input_ids[-3500:]])
    E, B = [], []
    for tp in traces:
        tr = json.load(open(tp)); iid = tr.get("instance_id") or os.path.basename(tp); problem = prob(iid)
        if not problem: continue
        te = tb = 0
        for k, turn in enumerate(tr["turns"]):
            if k == 0: continue
            tcs = turn.get("tool_calls") or []
            if not tcs: continue
            nm = tcs[0].get("name")
            if nm == "str_replace_editor" and len(E) < NEDIT and te < 3:
                ids = ids_at(tr, problem, k); E.append({"ids": ids, "pe": _p_edit(_logits_last(ids))}); te += 1
            elif nm == "bash" and len(B) < NBASH and tb < 3:
                ids = ids_at(tr, problem, k); B.append({"ids": ids, "pe": _p_edit(_logits_last(ids))}); tb += 1
        if len(E) >= NEDIT and len(B) >= NBASH: break
    M["EDIT"], M["BASH"] = E, B
    A["fidelity"] = {"edit_pts": float(np.mean([r["pe"] for r in E])), "bash_pts": float(np.mean([r["pe"] for r in B])),
                     "nE": len(E), "nB": len(B)}
    torch.save({"EDIT": E, "BASH": B}, "/content/xpts.pt")
    upload_file(path_or_fileobj="/content/xpts.pt", path_in_repo=PFILE, repo_id=REPO, repo_type="dataset", token=_tok())
    save_a(); log("POINTS_OK", A["fidelity"])

def est_set():
    if "est_n" in A: log("EST_SKIP"); return
    agent = [r["ids"][0].tolist()[-EST_CTX:] for r in (M["EDIT"] + M["BASH"])]
    tok = M["tok"]
    gsrc = "The history of science is a history of careful measurement and honest error. A river runs through the valley. In mathematics a proof is a chain of statements each following from the last. "
    base = tok(gsrc * 60, add_special_tokens=False).input_ids; step = max(1, (len(base) - 256) // N_GEN)
    generic = [base[i * step:i * step + 256] for i in range(N_GEN)]
    torch.save({"agent": agent, "generic": generic}, "/content/xest.pt")
    upload_file(path_or_fileobj="/content/xest.pt", path_in_repo=f"results/jspace_xmodel_est_{SAFE}.pt", repo_id=REPO, repo_type="dataset", token=_tok())
    A["est_n"] = {"agent": len(agent), "generic": len(generic)}; save_a(); log("EST_OK", A["est_n"])
def _load_est():
    return torch.load(hf_hub_download(REPO, f"results/jspace_xmodel_est_{SAFE}.pt", repo_type="dataset", token=_tok(), force_download=True))

# ---------------- J-lens estimator (final-position diagonal, chunk-resumable)
def _grad_one(ids, vids, layers):
    m = M["model"]; emb = _emb(); x = torch.tensor([ids], device=_dev()); out = {v: {} for v in vids}
    m.train()
    for v in vids:
        store = {}; heh = emb.register_forward_hook(lambda mo, i, o: o.requires_grad_(True)); hs = []
        def mk(L):
            def h(mo, i, o):
                t = o[0] if isinstance(o, tuple) else o; t.retain_grad(); store[L] = t
            return h
        for L in layers: hs.append(_layer(L).register_forward_hook(mk(L)))
        try:
            lg = m(input_ids=x, use_cache=False).logits[0].float(); loss = lg[-1, v]
            m.zero_grad(set_to_none=True); loss.backward()
            for L in layers:
                if store[L].grad is None: raise RuntimeError(f"grad None L{L}")
                out[v][L] = store[L].grad[0][-1].float().cpu()
        finally:
            heh.remove()
            for h in hs: h.remove()
            gc.collect(); torch.cuda.empty_cache()
    m.eval(); return out

def _pf(group): return f"results/jspace_xmodel_vp_{SAFE}_{group}.pt"
def jlens(group):
    global G
    try: G = torch.load(hf_hub_download(REPO, VFILE, repo_type="dataset", token=_tok(), force_download=True), map_location="cpu")
    except Exception: G = {}
    if group in G and group in A.get("jlens_done", []): log("JLENS_SKIP", group); return
    est = _load_est()
    if group == "actions": vocab = dict(M["TOOL"]); ids_list = est["agent"][:40]
    else: vocab = dict(M["POOL"]); ids_list = est["generic"]
    vids = list(vocab.values()); inv = {t: n for n, t in vocab.items()}
    layers = M["READOUT_L"]; d = M["H"]
    try:
        part = torch.load(hf_hub_download(REPO, _pf(group), repo_type="dataset", token=_tok(), force_download=True))
        summ = part["sum"]; done = part["done"]; log("JLENS_RESUME", group, done)
    except Exception:
        summ = {v: {L: torch.zeros(d) for L in layers} for v in vids}; done = 0
    for pi in range(done, len(ids_list)):
        row = _grad_one(ids_list[pi], vids, layers)
        for v in vids:
            for L in layers: summ[v][L] += row[v][L]
        if (pi + 1) % CHUNK == 0 or pi + 1 == len(ids_list):
            torch.save({"sum": summ, "done": pi + 1}, "/content/xvp.pt")
            upload_file(path_or_fileobj="/content/xvp.pt", path_in_repo=_pf(group), repo_id=REPO, repo_type="dataset", token=_tok())
            log(f"  JLENS {group} {pi+1}/{len(ids_list)}")
    n = len(ids_list)
    G[group] = {inv[t]: {L: summ[t][L] / max(n, 1) for L in layers} for t in vids}
    torch.save(G, "/content/xvec.pt")
    upload_file(path_or_fileobj="/content/xvec.pt", path_in_repo=VFILE, repo_id=REPO, repo_type="dataset", token=_tok())
    A.setdefault("jlens_done", []).append(group); save_a(); log("JLENS_OK", group)

def _load_vec():
    global G
    G = torch.load(hf_hub_download(REPO, VFILE, repo_type="dataset", token=_tok(), force_download=True), map_location="cpu")

def _steer_last(ids, dirs, band, alpha):
    hs = []
    for L in band:
        u = dirs[L].to(_dev()).float(); u = u / (u.norm() + 1e-9)
        def mk(u):
            def h(m, i, o):
                t = o[0] if isinstance(o, tuple) else o; t = t.clone(); hl = t[:, -1, :].float()
                t[:, -1, :] = (hl + alpha * hl.norm(dim=-1, keepdim=True) * u).to(t.dtype)
                return (t, *o[1:]) if isinstance(o, tuple) else t
            return h
        hs.append(_layer(L).register_forward_hook(mk(u)))
    try:
        with torch.no_grad(): lg = M["model"](input_ids=ids.to(_dev()), use_cache=False).logits[0, -1].float()
    finally:
        for h in hs: h.remove()
    return lg

# ---------------- specificity control (instrument validity on this model)
def _qa_items():
    tok = M["tok"]; words = list(M["POOL"]); rng = random.Random(11)
    syms = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "L", "R", "S"]; mids = ["K", "M", "P", "Q", "T", "V", "W", "Z"]
    def chain(a, m, leaf): return f"{a} -> {m}. {m} -> {leaf}."
    items = []
    for i in range(20):
        w = rng.sample(words, 6); s = rng.sample(syms, 6); md = rng.sample(mids, 6)
        d1 = f"{chain(s[0], md[0], w[0])} Two hops: {s[0]} -> {w[0]}."
        d2 = f"{chain(s[1], md[1], w[1])} Two hops: {s[1]} -> {w[1]}."
        rules = [chain(s[2], md[2], w[2]), chain(s[3], md[3], w[3])]; rng.shuffle(rules)
        items.append({"prompt": f"Follow two hops. {d1}\n{d2}\n{' '.join(rules)} Two hops: {s[2]} ->",
                      "correct": w[2], "alt": w[3]})
    return items

def spec():
    if "spec" in A: log("SPEC_SKIP"); return
    _load_vec(); tok = M["tok"]; items = _qa_items(); band = M["BANDS"]["midW"]
    gA = G["answers"]; gen = torch.Generator().manual_seed(7)
    base_ok = con = rnd = 0
    for it in items:
        ids = torch.tensor([tok(it["prompt"], add_special_tokens=False).input_ids])
        cid, aid = M["POOL"][it["correct"]], M["POOL"][it["alt"]]
        base_ok += int(int(_logits_last(ids).argmax()) == cid)
        d1 = {L: (gA[it["alt"]][L] - gA[it["correct"]][L]) for L in band}
        con += int(int(_steer_last(ids, d1, band, 0.5).argmax()) == aid)
        d3 = {L: torch.randn(gA[it["alt"]][L].shape, generator=gen) for L in band}
        rnd += int(int(_steer_last(ids, d3, band, 0.5).argmax()) == aid)
        gc.collect(); torch.cuda.empty_cache()
    A["spec"] = {"base_correct": base_ok, "contrast_to_alt": con, "random_to_alt": rnd, "n": len(items)}
    save_a(); log(f"SPEC base={base_ok}/20 contrast={con} random={rnd}")

# ---------------- the dissociation (answer & action steer per band)
def dissoc():
    if "dissoc" in A: log("DISSOC_SKIP"); return
    _load_vec(); tok = M["tok"]; items = _qa_items()
    gA = G["answers"]; ge = G["actions"]["edit"]; gb = G["actions"]["bash"]; bash_id = M["TOOL"]["bash"]
    gen = torch.Generator().manual_seed(31); out = {"answer": {}, "action": {}}
    for bn, band in M["BANDS"].items():
        ah = ar = 0
        for it in items:
            ids = torch.tensor([tok(it["prompt"], add_special_tokens=False).input_ids]); aid = M["POOL"][it["alt"]]
            d = {L: (gA[it["alt"]][L] - gA[it["correct"]][L]) for L in band}
            ah += int(int(_steer_last(ids, d, band, 0.5).argmax()) == aid)
            dr = {L: torch.randn(gA[it["alt"]][L].shape, generator=gen) for L in band}
            ar += int(int(_steer_last(ids, dr, band, 0.5).argmax()) == aid)
            gc.collect(); torch.cuda.empty_cache()
        out["answer"][bn] = {"contrast_to_alt": ah, "random": ar, "n": len(items)}
        ch = cr = 0
        for r in M["EDIT"]:
            d = {L: (gb[L] - ge[L]) for L in band}
            ch += int(int(_steer_last(r["ids"], d, band, 0.5).argmax()) == bash_id)
            dr = {L: torch.randn(ge[L].shape, generator=gen) for L in band}
            cr += int(int(_steer_last(r["ids"], dr, band, 0.5).argmax()) == bash_id)
            gc.collect(); torch.cuda.empty_cache()
        out["action"][bn] = {"edit_to_bash": ch, "random": cr, "n": len(M["EDIT"])}
        log(f"  DISSOC {bn}: answer {ah}/{len(items)}(r{ar}) | action {ch}/{len(M['EDIT'])}(r{cr})")
    A["dissoc"] = out; save_a(); log("DISSOC_OK")

def finalize():
    A["finalized"] = True; save_a(); log("FINALIZED")
