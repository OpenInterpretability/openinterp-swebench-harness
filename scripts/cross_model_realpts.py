#!/usr/bin/env python3
"""Cross-model circuit on REAL agent decision points (default gpt-oss-20b).

The synthetic single-turn probe could not reach the late-decision regime in some models. Here we reconstruct
REAL SWE-bench-Pro tool-decision points from the released agent trajectories (the agent's real tool-call
history) and re-render them in a generic chat format, so the target model genuinely decides the next tool LATE
(a long real transcript). Then the same architecture-agnostic decomposition: swap the o_proj INPUT (attention
channel) / mlp output (mlp channel) from a task-matched edit-point donor into a bash-point, and per-head.

Run:  HF_TOKEN=... MODEL=openai/gpt-oss-20b python cross_model_realpts.py
"""
import os, json, time, glob, tarfile
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import hf_hub_download, upload_file

MODEL_ID = os.environ.get("MODEL", "openai/gpt-oss-20b")
NEDIT = int(os.environ.get("NEDIT", "20")); NBASH = int(os.environ.get("NBASH", "20"))
SAFE = MODEL_ID.split("/")[-1]; REPO = "caiovicentino1/swebench-phase6-verdict-circuit"
RFILE = f"results/cross_realpts_{SAFE}.json"
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def _tok(): return os.environ.get("HF_TOKEN") or None
R = {}
def save():
    json.dump(R, open("/content/crp.json", "w"))
    upload_file(path_or_fileobj="/content/crp.json", path_in_repo=RFILE, repo_id=REPO, repo_type="dataset", token=_tok())
def load():
    global R
    try:
        p = hf_hub_download(REPO, RFILE, repo_type="dataset", token=_tok(), force_download=True); R = json.load(open(p)); log("resumed:", sorted(R))
    except Exception: R = {}

SYS = ("You are an autonomous coding agent solving a software issue in a repository. Tools: 'bash' (run a shell "
       "command — reversible exploration), 'edit' (modify a file — COMMITS a change), 'finish' (submit). Decide "
       'the single next tool. Respond ONLY with JSON: {"tool": "<bash|edit|finish>", "args": {...}}')
FORCE = '{"tool": "'

log("loading", MODEL_ID)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
cfg = getattr(model.config, "text_config", model.config)
NL = cfg.num_hidden_layers; NH = cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
log(f"loaded; {NL} layers, {NH} heads, hd {HD}")
def first_tok(s): return tok(s, add_special_tokens=False).input_ids[0]
TOK = {"edit": first_tok("edit"), "bash": first_tok("bash"), "finish": first_tok("finish")}
log("tool tokens:", {k: repr(tok.decode([v])) for k, v in TOK.items()})

def _resolve(paths):
    for path in paths:
        cur = model; ok = True
        for p in path.split("."):
            if not hasattr(cur, p): ok = False; break
            cur = getattr(cur, p)
        if ok: return cur
LAYERS = _resolve(("model.layers", "model.model.layers", "model.language_model.layers"))
def lyr(L): return LAYERS[L]
def oproj(L): return lyr(L).self_attn.o_proj
def mlpmod(L): return lyr(L).mlp
def p_edit(lg): ids = [TOK["bash"], TOK["edit"], TOK["finish"]]; return float(torch.softmax(lg[ids].float(), -1)[1])
def fwd_logits(ids):
    with torch.no_grad(): return model(input_ids=ids.to(model.device), use_cache=False).logits[0, -1].float()
def cap_zmy(ids, L):
    box = {}
    h1 = lyr(L).register_forward_hook(lambda m, i, o: box.__setitem__("y", (o[0] if isinstance(o, tuple) else o)[0, -1, :].detach().float().cpu()))
    h2 = oproj(L).register_forward_pre_hook(lambda m, a: box.__setitem__("z", a[0][0, -1, :].detach().float().cpu()))
    h3 = mlpmod(L).register_forward_hook(lambda m, i, o: box.__setitem__("m", (o[0] if isinstance(o, tuple) else o)[0, -1, :].detach().float().cpu()))
    try: fwd_logits(ids)
    finally:
        for h in (h1, h2, h3): h.remove()
    return box["z"], box["m"], box["y"]
def _set_layer_out(L, v):
    def h(m, i, o):
        x = o[0] if isinstance(o, tuple) else o; x = x.clone(); x[0, -1, :] = v.to(x.device, x.dtype); return (x, *o[1:]) if isinstance(o, tuple) else x
    return lyr(L).register_forward_hook(h)
def _set_oproj_in(L, v):
    def pre(m, a):
        x = a[0].clone(); x[0, -1, :] = v.to(x.device, x.dtype); return (x, *a[1:])
    return oproj(L).register_forward_pre_hook(pre)
def _set_mlp_out(L, v):
    def h(m, i, o):
        x = o[0] if isinstance(o, tuple) else o; x = x.clone(); x[0, -1, :] = v.to(x.device, x.dtype); return (x, *o[1:]) if isinstance(o, tuple) else x
    return mlpmod(L).register_forward_hook(h)
def dP(ids, base, hfn):
    hh = hfn()
    try: pe = p_edit(fwd_logits(ids))
    finally: hh.remove()
    return pe - base

# ---------------- REAL decision points from released trajectories
def build_points():
    os.makedirs("/content/rp", exist_ok=True)
    with tarfile.open(hf_hub_download(REPO, "traces.tar.gz", repo_type="dataset")) as t: t.extractall("/content/rp")
    traces = sorted(glob.glob("/content/rp/traces/instance_*.json"))
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    by = {r["instance_id"]: r for r in ds if r.get("instance_id")}
    def prob(iid):
        r = by.get(iid) or next((v for k, v in by.items() if k in iid or iid in k), None)
        if not r: return None
        for k in ("problem_statement", "issue_text", "text"):
            if r.get(k): return str(r[k])
        return None
    def msgs_upto(tr, problem, k):
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
        try: pre = tok.apply_chat_template(msgs_upto(tr, problem, k), add_generation_prompt=True, tokenize=False)
        except Exception: pre = SYS + "\n" + problem[:5000] + "\nAssistant: "
        i = tok(pre + FORCE, add_special_tokens=False).input_ids
        return torch.tensor([i[-3500:]])
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
                ids = ids_at(tr, problem, k); E.append({"ids": ids, "pe": p_edit(fwd_logits(ids))}); te += 1
            elif nm == "bash" and len(B) < NBASH and tb < 3:
                ids = ids_at(tr, problem, k); B.append({"ids": ids, "pe": p_edit(fwd_logits(ids))}); tb += 1
            torch.cuda.empty_cache()
        if len(E) >= NEDIT and len(B) >= NBASH: break
    return E, B

load()
log("building REAL decision points ...")
E, B = build_points()
N = min(len(E), len(B)); E, B = E[:N], B[:N]
R["fidelity"] = {"edit_pts": float(np.mean([r["pe"] for r in E])), "bash_pts": float(np.mean([r["pe"] for r in B])), "n": N}
log("REAL fidelity P(edit): edit-pts %.3f bash-pts %.3f (n=%d)" % (R["fidelity"]["edit_pts"], R["fidelity"]["bash_pts"], N)); save()

# locate by attn-channel
late = [L for L in range(int(NL * 0.55), NL) if L % max(1, (NL - int(NL * 0.55)) // 7) == 0]
locf, loca = {}, {}
for L in late:
    caps = [cap_zmy(E[j]["ids"], L) for j in range(N)]
    locf[L] = float(np.mean([dP(B[j]["ids"], B[j]["pe"], (lambda L=L, y=caps[j][2]: lambda: _set_layer_out(L, y))()) for j in range(N)]))
    loca[L] = float(np.mean([dP(B[j]["ids"], B[j]["pe"], (lambda L=L, z=caps[j][0]: lambda: _set_oproj_in(L, z))()) for j in range(N)]))
    log(f"  L{L}: full {locf[L]:+.3f} | attn {loca[L]:+.3f}")
R["locate"] = {"full": locf, "attn": loca}; Ls = int(max(loca, key=loca.get)); R["L_star"] = Ls; save(); log("L* =", Ls)

ZE = [None] * N; ME = [None] * N; YE = [None] * N; ZB = [None] * N
for j in range(N):
    ZE[j], ME[j], YE[j] = cap_zmy(E[j]["ids"], Ls); ZB[j], _, _ = cap_zmy(B[j]["ids"], Ls)
R["channels"] = {
    "attn": float(np.mean([dP(B[j]["ids"], B[j]["pe"], (lambda j=j: lambda: _set_oproj_in(Ls, ZE[j]))()) for j in range(N)])),
    "mlp": float(np.mean([dP(B[j]["ids"], B[j]["pe"], (lambda j=j: lambda: _set_mlp_out(Ls, ME[j]))()) for j in range(N)])),
    "full": float(np.mean([dP(B[j]["ids"], B[j]["pe"], (lambda j=j: lambda: _set_layer_out(Ls, YE[j]))()) for j in range(N)]))}
log("channels -> attn %.3f | mlp %.3f | full %.3f" % (R["channels"]["attn"], R["channels"]["mlp"], R["channels"]["full"])); save()

R.setdefault("head_dP", {})
for h in range(NH):
    if str(h) in R["head_dP"]: continue
    sl = slice(h * HD, (h + 1) * HD); dv = []
    for j in range(N):
        z = ZB[j].clone(); z[sl] = ZE[j][sl]; dv.append(dP(B[j]["ids"], B[j]["pe"], (lambda z=z: lambda: _set_oproj_in(Ls, z))()))
    R["head_dP"][str(h)] = float(np.mean(dv))
    if (h + 1) % 8 == 0: save(); log(f"  head {h+1}/{NH}")
order = sorted(range(NH), key=lambda h: -R["head_dP"][str(h)]); cum = {}
for k in (1, 2, 3, 5, 8):
    hs = order[:k]; dv = []
    for j in range(N):
        z = ZB[j].clone()
        for h in hs: z[h * HD:(h + 1) * HD] = ZE[j][h * HD:(h + 1) * HD]
        dv.append(dP(B[j]["ids"], B[j]["pe"], (lambda z=z: lambda: _set_oproj_in(Ls, z))()))
    cum[f"top{k}"] = float(np.mean(dv)); log(f"  cum top{k} {hs}: {cum[f'top{k}']:+.3f}")
R["cum"] = cum; R["order_top8"] = order[:8]
R["meta"] = {"model": MODEL_ID, "NL": NL, "NH": NH, "HD": HD, "L_star": Ls, "n": N}
save()
log("DONE. fidelity %.3f/%.3f | attn %.3f mlp %.3f | top3 %.3f (%.0f%% of attn)" % (
    R["fidelity"]["edit_pts"], R["fidelity"]["bash_pts"], R["channels"]["attn"], R["channels"]["mlp"],
    cum["top3"], 100 * cum["top3"] / R["channels"]["attn"] if abs(R["channels"]["attn"]) > 1e-6 else 0))
print("OILAB_JSON_BEGIN"); print(json.dumps(R)); print("OILAB_JSON_END", flush=True)
