#!/usr/bin/env python3
"""Cross-model-lite: does the late commitment-lever GEOMETRY generalize beyond Qwen3.6-27B?

Pre-reg: paper/circuit_breaker/PREREG_cross_model.md. LOCATE (logit-lens late-emergence) + one-token ΔP
on a SECOND-family model's tool-COMMIT decision, over real SWE-bench Pro tasks. No agent harness / Docker /
generation — single-turn forced-JSON tool decision. Fast (7B), completes in one Colab session.

Run:  HF_TOKEN=... python cross_model_lite.py            (default Mistral-7B-Instruct-v0.3)
      MODEL=mistralai/Mistral-Small-24B-Instruct-2501 python cross_model_lite.py
"""
import os, sys, json, time
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = os.environ.get("MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
N = int(os.environ.get("N", "30"))
OUT = "/content/cross_out"; os.makedirs(OUT, exist_ok=True)
SAFE = MODEL_ID.split("/")[-1]

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

SYS = ("You are an autonomous coding agent solving a software issue in a repository. You have exactly three "
       "tools: 'bash' (run a shell command — reversible exploration), 'edit' (modify a file — this COMMITS a "
       "change), 'finish' (submit your solution). Decide the single next tool to call. "
       'Respond ONLY with JSON: {"tool": "<bash|edit|finish>", "args": {...}}')
FORCE = '{"tool": "'   # forces the next token to be the tool name

log("loading", MODEL_ID)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto").eval()
NL = model.config.num_hidden_layers
log(f"loaded; {NL} layers")

def first_tok(s):
    ids = tok(s, add_special_tokens=False).input_ids
    return ids[0]
TOK = {"edit": first_tok("edit"), "bash": first_tok("bash"), "finish": first_tok("finish")}
log("tool first-tokens:", {k: (v, repr(tok.decode([v]))) for k, v in TOK.items()})

def layers():
    for path in ("model.layers", "model.model.layers"):
        cur = model; ok = True
        for p in path.split("."):
            if not hasattr(cur, p): ok = False; break
            cur = getattr(cur, p)
        if ok: return cur
    raise RuntimeError("layers not found")
LAYERS = layers()
def norm():
    for path in ("model.norm", "model.model.norm"):
        cur = model; ok = True
        for p in path.split("."):
            if not hasattr(cur, p): ok = False; break
            cur = getattr(cur, p)
        if ok: return cur
NORM = norm(); LMH = model.get_output_embeddings()
SWEEP = list(range(1, NL, max(1, NL // 16)))  # ~16 layers spread

def decision_ids(task_text, mode):
    """mode='bash' -> first-action explore context (P(edit) low); mode='edit' -> bug-located context (P(edit) high)."""
    if mode == "bash":
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": task_text[:5500] +
                 "\n\n[This is your FIRST action. You have NOT explored the repository or reproduced the failure yet.]"}]
    else:  # edit
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": task_text[:5500]},
                {"role": "assistant", "content": '{"tool": "bash", "args": {"command": "grep -rn failing_symbol; sed -n 1,80p src/file.py"}}'},
                {"role": "user", "content": "[tool result] Reproduced the failure for this task and located the root cause: " + task_text[:260] + " ... The fix is clear; nothing left to explore."}]
    try:
        prefix = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        prefix = SYS + "\n\nTask:\n" + task_text[:5500] + "\n\nAssistant: "
    ids = tok(prefix + FORCE, add_special_tokens=False).input_ids
    return torch.tensor([ids[-3500:]])

def caps_and_logits(ids, layers_):
    caps = {}; hs = []
    def mk(L):
        def h(m, i, o): caps[L] = (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach()
        return h
    for L in layers_: hs.append(LAYERS[L].register_forward_hook(mk(L)))
    try:
        with torch.no_grad():
            out = model(input_ids=ids.to(model.device), use_cache=False)
    finally:
        for h in hs: h.remove()
    return out.logits[0, -1].float(), caps

def lens_gap(vec, name="edit"):
    with torch.no_grad():
        h = vec.reshape(1, -1).to(model.device).to(next(model.parameters()).dtype)
        lg = LMH(NORM(h)).reshape(-1).float()   # [vocab]
    alts = [TOK[n] for n in TOK if n != name]
    return float(lg[TOK[name]] - lg[alts].mean())

def p_tool(logits, name="edit"):
    ids = [TOK["bash"], TOK["edit"], TOK["finish"]]
    p = torch.softmax(logits[ids].float(), -1)
    return float(p[["bash", "edit", "finish"].index(name)])

log("loading SWE-bench Pro tasks")
ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
def task_text(r):
    for k in ("problem_statement", "issue_text", "text", "instance_id"):
        if r.get(k): return str(r[k])
    return json.dumps(r)[:4000]
tasks = [task_text(r) for r in list(ds)[:N * 2]][:N]

# ---- collect edit-context + bash-context decision points (task-matched)
all_layers = sorted(set(SWEEP) | {NL - 1})
late = [L for L in all_layers if L >= int(NL * 0.72)]
mid = [L for L in all_layers if int(NL * 0.25) <= L <= int(NL * 0.5)][:3]
PATCH = sorted(set(mid + late))
E, B = [], []   # edit-context rows, bash-context rows (index-aligned by task)
for i, t in enumerate(tasks):
    for mode, store in (("edit", E), ("bash", B)):
        ids = decision_ids(t, mode)
        logits, caps = caps_and_logits(ids, all_layers)
        store.append({"ids": ids.cpu(), "p_edit": p_tool(logits, "edit"),
                      "prof": {L: lens_gap(caps[L], "edit") for L in all_layers},
                      "caps_last": {L: caps[L][0].float().cpu() for L in PATCH}})
        del caps, logits; torch.cuda.empty_cache()
    if (i + 1) % 10 == 0: log(f"  collected {i+1}/{len(tasks)}")
log(f"patch layers: mid {mid} | late {late}")

# H1 LOCATE — does the edit gap emerge LATE? (edit-context)
mean_prof = {L: float(np.mean([r["prof"][L] for r in E])) for L in all_layers}
log("LOCATE edit-gap per layer (edit-context; flat early -> rises late = generalizes):")
for L in all_layers: log(f"  L{L:2d}: {mean_prof[L]:+.3f}")

# H2 fidelity — P(edit) higher in edit-context than bash-context
pe = float(np.mean([r["p_edit"] for r in E])); pb = float(np.mean([r["p_edit"] for r in B]))
log(f"fidelity P(edit): edit-context {pe:.3f}  bash-context {pb:.3f}  (gate: edit >> bash)")

# H3 writability — inject the task-matched edit-context donor into the bash-context decision, late layers
def _patch_hook(donor):
    def h(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o
        hs = hs.clone(); hs[:, -1, :] = donor.to(hs.dtype).to(hs.device)
        return (hs, *o[1:]) if isinstance(o, tuple) else hs
    return h

dP_edit, dP_neutral = {}, {}   # edit-donor vs same-norm shuffled-neutral, per patch layer (mid + late)
g = torch.Generator().manual_seed(0)
for L in PATCH:
    de, dn = [], []
    for j, rb in enumerate(B):
        donor = E[j]["caps_last"][L]                                  # task-matched edit donor
        neutral = donor[torch.randperm(donor.numel(), generator=g)]  # same-norm random direction
        for store, d in ((de, donor), (dn, neutral)):
            hh = LAYERS[L].register_forward_hook(_patch_hook(d))
            try:
                with torch.no_grad():
                    lg = model(input_ids=rb["ids"].to(model.device), use_cache=False).logits[0, -1]
            finally:
                hh.remove()
            store.append(p_tool(lg, "edit") - rb["p_edit"]); torch.cuda.empty_cache()
    dP_edit[L] = float(np.mean(de)); dP_neutral[L] = float(np.mean(dn))
log("H3 ΔP(edit) injecting into bash-context — edit-donor | shuffled-neutral (mid should be ~0, late should be +):")
for L in PATCH: log(f"  L{L:2d}{'(mid)' if L in mid else '(late)'}: {dP_edit[L]:+.3f}  |  {dP_neutral[L]:+.3f}")

out = {"model": MODEL_ID, "n": len(E), "num_layers": NL, "mid_layers": mid, "late_layers": late,
       "locate_mean_prof": mean_prof,
       "fidelity_p_edit": {"edit_context": pe, "bash_context": pb},
       "writability_dP": {"edit_donor": dP_edit, "neutral_ctrl": dP_neutral}}
path = os.path.join(OUT, f"cross_model_{SAFE}.json")
json.dump(out, open(path, "w"), indent=1)
log("saved", path)
try:
    from huggingface_hub import upload_file
    upload_file(path_or_fileobj=path, path_in_repo=f"results/cross_model_{SAFE}.json",
                repo_id="caiovicentino1/swebench-phase6-verdict-circuit", repo_type="dataset",
                token=os.environ.get("HF_TOKEN"))
    log("uploaded to HF")
except Exception as e:
    log("HF upload skipped:", str(e)[:80])
print("OILAB_JSON_BEGIN"); print(json.dumps(out)); print("OILAB_JSON_END", flush=True)
log("CROSS_MODEL_DONE")
