#!/usr/bin/env python3
"""Cross-model CIRCUIT: does the sparse attention-head commitment circuit generalize beyond Qwen3.6-27B?

Extends cross_model_lite (synthetic edit/explore tool-decision contexts, forced-JSON next-tool token) with a
mechanistic decomposition of the late lever, done in an ARCHITECTURE-AGNOSTIC way: instead of an exact additive
residual split (breaks under Gemma sandwich-norm / Qwen output-gating), we SWAP the o_proj input (attention
channel) and the mlp output (mlp channel) from the explore-context point to the task-matched edit-context donor
and let the rest of the block run. This is robust to post-norms, gating, soft-capping, MoE.

Measures at the located late layer L*:
  - attn-channel swap (all heads' o_proj input) vs mlp-channel swap vs full residual swap (the lever) -> dP(edit)
  - per-HEAD swap (swap one head's o_proj-input slice) -> dP(edit); rank + cumulative -> SPARSITY
Q: is the elicit write (a) attention-carried and (b) sparse, across architectures?

Run:  HF_TOKEN=... MODEL=meta-llama/Llama-3.1-8B-Instruct python cross_model_circuit.py
"""
import os, json, time
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import hf_hub_download, upload_file

MODEL_ID = os.environ.get("MODEL", "mistralai/Mistral-Small-24B-Instruct-2501")
N = int(os.environ.get("N", "20"))
SAFE = MODEL_ID.split("/")[-1]
REPO = "caiovicentino1/swebench-phase6-verdict-circuit"
RFILE = f"results/cross_circuit_v3_{SAFE}.json" if "gemma" in MODEL_ID.lower() else f"results/cross_circuit_v2_{SAFE}.json"
IS_GEMMA = "gemma" in MODEL_ID.lower()
TOOLS_SPEC = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command (reversible exploration).",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Modify a file. This COMMITS a change.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "finish", "description": "Submit the solution.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]
GEMMA_FORCE = "<|tool_call>call:"   # Gemma 4 native tool-call format (after the empty thought channel)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
def _tok(): return os.environ.get("HF_TOKEN") or None

R = {}
def save():
    json.dump(R, open("/content/ccx.json", "w"))
    upload_file(path_or_fileobj="/content/ccx.json", path_in_repo=RFILE, repo_id=REPO,
                repo_type="dataset", token=_tok())
def load():
    global R
    try:
        p = hf_hub_download(REPO, RFILE, repo_type="dataset", token=_tok(), force_download=True)
        R = json.load(open(p)); log("resumed:", sorted(R.keys()))
    except Exception:
        R = {}

SYS = ("You are an autonomous coding agent solving a software issue in a repository. You have exactly three "
       "tools: 'bash' (run a shell command — reversible exploration), 'edit' (modify a file — this COMMITS a "
       "change), 'finish' (submit your solution). Decide the single next tool to call. "
       'Respond ONLY with JSON: {"tool": "<bash|edit|finish>", "args": {...}}')
FORCE = '{"tool": "'

log("loading", MODEL_ID)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
                                             trust_remote_code=True).eval()
cfg = model.config
tcfg = getattr(cfg, "text_config", cfg)
NL = tcfg.num_hidden_layers
NH = tcfg.num_attention_heads
HD = getattr(tcfg, "head_dim", None) or tcfg.hidden_size // NH
log(f"loaded; {NL} layers, {NH} heads, head_dim {HD}")

def first_tok(s): return tok(s, add_special_tokens=False).input_ids[0]
TOK = {"edit": first_tok("edit"), "bash": first_tok("bash"), "finish": first_tok("finish")}
log("tool first-tokens:", {k: repr(tok.decode([v])) for k, v in TOK.items()})

def _resolve(paths):
    for path in paths:
        cur = model; ok = True
        for p in path.split("."):
            if not hasattr(cur, p): ok = False; break
            cur = getattr(cur, p)
        if ok: return cur
    return None
LAYERS = _resolve(("model.layers", "model.model.layers", "model.language_model.layers"))
def lyr(L): return LAYERS[L]
def oproj(L): return lyr(L).self_attn.o_proj
def mlpmod(L): return lyr(L).mlp

def decision_ids(t, mode):
    if IS_GEMMA:
        sysmsg = "You are an autonomous coding agent solving a software issue. Decide the single next tool to call."
        if mode == "bash":
            msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": t[:5500]},
                    {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"command": "grep -rn failing_symbol src/"}}}]},
                    {"role": "tool", "name": "bash", "content": "The failing symbol appears in 3 files; a root cause is plausible but NOT yet confirmed. You could edit a likely spot now, or run one more check first."}]
        else:
            msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": t[:5500]},
                    {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"command": "grep -rn failing_symbol; sed -n 1,80p src/file.py"}}}]},
                    {"role": "tool", "name": "bash", "content": "Reproduced the failure and located the root cause: " + t[:200] + " ... The fix is clear; nothing left to explore."}]
        try:
            prefix = tok.apply_chat_template(msgs, tools=TOOLS_SPEC, add_generation_prompt=True, tokenize=False)
        except Exception:
            prefix = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(prefix + GEMMA_FORCE, add_special_tokens=False).input_ids
        return torch.tensor([ids[-3500:]])
    if mode == "bash":
        # on-the-fence: model has explored a bit, has a plausible-but-unconfirmed cause -> genuinely deciding LATE
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": t[:5500]},
                {"role": "assistant", "content": '{"tool": "bash", "args": {"command": "grep -rn failing_symbol src/"}}'},
                {"role": "user", "content": "[tool result] The failing symbol appears in 3 files; a root cause is plausible but NOT yet confirmed. You could edit a likely spot now, or run one more check first."}]
    else:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": t[:5500]},
                {"role": "assistant", "content": '{"tool": "bash", "args": {"command": "grep -rn failing_symbol; sed -n 1,80p src/file.py"}}'},
                {"role": "user", "content": "[tool result] Reproduced the failure and located the root cause: " + t[:260] + " ... The fix is clear; nothing left to explore."}]
    try:
        prefix = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        prefix = SYS + "\n\nTask:\n" + t[:5500] + "\n\nAssistant: "
    ids = tok(prefix + FORCE, add_special_tokens=False).input_ids
    return torch.tensor([ids[-3500:]])

def p_edit(logits):
    ids = [TOK["bash"], TOK["edit"], TOK["finish"]]
    return float(torch.softmax(logits[ids].float(), -1)[1])

def fwd_logits(ids):
    with torch.no_grad():
        return model(input_ids=ids.to(model.device), use_cache=False).logits[0, -1].float()

# capture y(residual out), z(o_proj input), m(mlp out) at layer L, last token
def cap_zmy(ids, L):
    box = {}
    h1 = lyr(L).register_forward_hook(lambda m, i, o: box.__setitem__("y", (o[0] if isinstance(o, tuple) else o)[0, -1, :].detach().float().cpu()))
    h2 = oproj(L).register_forward_pre_hook(lambda m, a: box.__setitem__("z", a[0][0, -1, :].detach().float().cpu()))
    h3 = mlpmod(L).register_forward_hook(lambda m, i, o: box.__setitem__("m", (o[0] if isinstance(o, tuple) else o)[0, -1, :].detach().float().cpu()))
    try: fwd_logits(ids)
    finally:
        for h in (h1, h2, h3): h.remove()
    return box["z"], box["m"], box["y"]

def _set_layer_out(L, yvec):
    def h(m, i, o):
        x = o[0] if isinstance(o, tuple) else o; x = x.clone(); x[0, -1, :] = yvec.to(x.device, x.dtype)
        return (x, *o[1:]) if isinstance(o, tuple) else x
    return lyr(L).register_forward_hook(h)
def _set_oproj_in(L, zvec):
    def pre(m, a):
        x = a[0].clone(); x[0, -1, :] = zvec.to(x.device, x.dtype); return (x, *a[1:])
    return oproj(L).register_forward_pre_hook(pre)
def _set_mlp_out(L, mvec):
    def h(m, i, o):
        x = o[0] if isinstance(o, tuple) else o; x = x.clone(); x[0, -1, :] = mvec.to(x.device, x.dtype)
        return (x, *o[1:]) if isinstance(o, tuple) else x
    return mlpmod(L).register_forward_hook(h)

def dP_with(ids, base_pe, handle_fn):
    hh = handle_fn()
    try: pe = p_edit(fwd_logits(ids))
    finally: hh.remove()
    return pe - base_pe

# ---------------- build contexts
log("loading SWE-bench Pro tasks")
ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
def ttext(r):
    for k in ("problem_statement", "issue_text", "text", "instance_id"):
        if r.get(k): return str(r[k])
    return json.dumps(r)[:4000]
tasks = [ttext(r) for r in list(ds)[:N * 2]][:N]
load()

if "points" not in R:
    E, B = [], []
    for i, t in enumerate(tasks):
        eb = decision_ids(t, "edit"); bb = decision_ids(t, "bash")
        E.append({"ids": eb, "pe": p_edit(fwd_logits(eb))})
        B.append({"ids": bb, "pe": p_edit(fwd_logits(bb))})
        if (i + 1) % 10 == 0: log(f"  contexts {i+1}/{N}")
    R["points"] = True
    R["fidelity"] = {"edit_ctx": float(np.mean([r["pe"] for r in E])), "bash_ctx": float(np.mean([r["pe"] for r in B]))}
    log("fidelity P(edit): edit-ctx %.3f bash-ctx %.3f" % (R["fidelity"]["edit_ctx"], R["fidelity"]["bash_ctx"]))
    # keep ids in memory only (not serialized)
    globals()["_E"], globals()["_B"] = E, B
    save()
else:
    # rebuild ids deterministically (same tasks/order)
    _E = [{"ids": decision_ids(t, "edit"), "pe": None} for t in tasks]
    _B = [{"ids": decision_ids(t, "bash"), "pe": None} for t in tasks]
    for i, t in enumerate(tasks):
        _E[i]["pe"] = p_edit(fwd_logits(_E[i]["ids"])); _B[i]["pe"] = p_edit(fwd_logits(_B[i]["ids"]))
E, B = globals()["_E"], globals()["_B"]

# ---------------- LOCATE late layer L* by residual-donor writability
late = [L for L in range(int(NL * 0.66), NL) if L % max(1, (NL - int(NL * 0.66)) // 6) == 0]
if "locate" not in R:
    loc_full, loc_attn = {}, {}
    for L in late:
        caps = [cap_zmy(E[j]["ids"], L) for j in range(N)]  # (z,m,y) edit donors
        df = [dP_with(B[j]["ids"], B[j]["pe"], (lambda L=L, y=caps[j][2]: lambda: _set_layer_out(L, y))()) for j in range(N)]
        da = [dP_with(B[j]["ids"], B[j]["pe"], (lambda L=L, z=caps[j][0]: lambda: _set_oproj_in(L, z))()) for j in range(N)]
        loc_full[L] = float(np.mean(df)); loc_attn[L] = float(np.mean(da))
        log(f"  locate L{L}: full {loc_full[L]:+.3f} | attn-channel {loc_attn[L]:+.3f}")
    R["locate"] = {"full": loc_full, "attn": loc_attn}
    R["L_star"] = int(max(loc_attn, key=loc_attn.get)); save()   # where ATTENTION writes, not where full saturates
Ls = R["L_star"]; log("L* =", Ls)

# capture z,m,y at L* for all rows
ZE = [None] * N; ME = [None] * N; YE = [None] * N; ZB = [None] * N
for j in range(N):
    ZE[j], ME[j], YE[j] = cap_zmy(E[j]["ids"], Ls)
    ZB[j], _, _ = cap_zmy(B[j]["ids"], Ls)

# ---------------- channel swaps (attn vs mlp vs full)
if "channels" not in R:
    fa = [dP_with(B[j]["ids"], B[j]["pe"], (lambda j=j: lambda: _set_oproj_in(Ls, ZE[j]))()) for j in range(N)]
    fm = [dP_with(B[j]["ids"], B[j]["pe"], (lambda j=j: lambda: _set_mlp_out(Ls, ME[j]))()) for j in range(N)]
    ff = [dP_with(B[j]["ids"], B[j]["pe"], (lambda j=j: lambda: _set_layer_out(Ls, YE[j]))()) for j in range(N)]
    R["channels"] = {"attn": float(np.mean(fa)), "mlp": float(np.mean(fm)), "full": float(np.mean(ff))}
    log("channels dP -> attn %.3f | mlp %.3f | full %.3f" % (R["channels"]["attn"], R["channels"]["mlp"], R["channels"]["full"]))
    save()

# ---------------- per-head swap (o_proj-input slice)
R.setdefault("head_dP", {})
for h in range(NH):
    if str(h) in R["head_dP"]: continue
    sl = slice(h * HD, (h + 1) * HD)
    dvals = []
    for j in range(N):
        zsw = ZB[j].clone(); zsw[sl] = ZE[j][sl]
        dvals.append(dP_with(B[j]["ids"], B[j]["pe"], (lambda z=zsw: lambda: _set_oproj_in(Ls, z))()))
    R["head_dP"][str(h)] = float(np.mean(dvals))
    if (h + 1) % 8 == 0: save(); log(f"  head {h+1}/{NH}")
save()

# ---------------- cumulative top-k sparsity
order = sorted(range(NH), key=lambda h: -R["head_dP"][str(h)])
cum = {}
for k in (1, 2, 3, 5, 8):
    heads = order[:k]
    dvals = []
    for j in range(N):
        zsw = ZB[j].clone()
        for h in heads:
            sl = slice(h * HD, (h + 1) * HD); zsw[sl] = ZE[j][sl]
        dvals.append(dP_with(B[j]["ids"], B[j]["pe"], (lambda z=zsw: lambda: _set_oproj_in(Ls, z))()))
    cum[f"top{k}"] = float(np.mean(dvals)); log(f"  cum top{k} ({heads}): dP {cum[f'top{k}']:+.3f}")
R["cum"] = cum; R["order_top8"] = order[:8]
# sparsity: heads needed to reach 80% of all-heads (attn channel)
allh = R["channels"]["attn"]
R["sparsity_top3_frac"] = cum["top3"] / allh if abs(allh) > 1e-6 else None
R["meta"] = {"model": MODEL_ID, "NL": NL, "NH": NH, "HD": HD, "N": N, "L_star": Ls}
save()
log("DONE. attn-channel %.3f | top3 heads %.3f (%.0f%% of attn) | mlp %.3f" % (
    allh, cum["top3"], 100 * (R["sparsity_top3_frac"] or 0), R["channels"]["mlp"]))
print("OILAB_JSON_BEGIN"); print(json.dumps({k: R[k] for k in R if k != "points"})); print("OILAB_JSON_END", flush=True)
