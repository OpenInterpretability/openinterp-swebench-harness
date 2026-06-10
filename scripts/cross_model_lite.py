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

def decision_ids(task_text):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": task_text[:6000]}]
    try:
        prefix = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        prefix = SYS + "\n\nTask:\n" + task_text[:6000] + "\n\nAssistant: "
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
        h = vec.to(model.device).to(next(model.parameters()).dtype).unsqueeze(0)
        lg = LMH(NORM(h))[0].float()
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

# ---- LOCATE + fidelity
rows = []
all_layers = sorted(set(SWEEP) | {NL - 1})
for i, t in enumerate(tasks):
    ids = decision_ids(t)
    logits, caps = caps_and_logits(ids, all_layers)
    choice = max(("bash", "edit", "finish"), key=lambda n: float(logits[TOK[n]]))
    prof = {L: lens_gap(caps[L], "edit") for L in all_layers}
    rows.append({"ids": ids.cpu(), "choice": choice, "prof": prof,
                 "p_edit": p_tool(logits, "edit"), "caps_last": {L: caps[L][0].float().cpu() for L in all_layers}})
    del caps, logits; torch.cuda.empty_cache()
    if (i + 1) % 10 == 0: log(f"  located {i+1}/{len(tasks)}")

choices = [r["choice"] for r in rows]
log("model tool-choice distribution:", {c: choices.count(c) for c in set(choices)})
mean_prof = {L: float(np.mean([r["prof"][L] for r in rows])) for L in all_layers}
log("LOCATE edit-gap per layer (flat early -> rises late = generalizes):")
for L in all_layers: log(f"  L{L:2d}: {mean_prof[L]:+.3f}")
# fidelity: edit-gap on edit-choosers vs bash-choosers (late layers)
late = [L for L in all_layers if L >= int(NL * 0.75)]
def gap_late(sub):
    if not sub: return float("nan")
    return float(np.mean([np.mean([r["prof"][L] for L in late]) for r in sub]))
edit_rows = [r for r in rows if r["choice"] == "edit"]
bash_rows = [r for r in rows if r["choice"] == "bash"]
log(f"fidelity late-gap: edit-choosers {gap_late(edit_rows):+.3f}  bash-choosers {gap_late(bash_rows):+.3f}")

# ---- one-token ΔP: inject an edit-donor into a bash decision at late layers
def _patch_hook(donor):
    def h(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o
        hs = hs.clone(); hs[:, -1, :] = donor.to(hs.dtype).to(hs.device)
        return (hs, *o[1:]) if isinstance(o, tuple) else hs
    return h

dP = {}
if edit_rows and bash_rows:
    def patch_dP(L):
        donor = edit_rows[0]["caps_last"][L]  # task-matched-ish single edit donor
        ds_ = []
        for r in bash_rows:
            hh = LAYERS[L].register_forward_hook(_patch_hook(donor))
            try:
                with torch.no_grad():
                    lg = model(input_ids=r["ids"].to(model.device), use_cache=False).logits[0, -1]
            finally:
                hh.remove()
            ds_.append(p_tool(lg, "edit") - r["p_edit"]); torch.cuda.empty_cache()
        return float(np.mean(ds_))
    for L in late: dP[L] = patch_dP(L)
    log("one-token ΔP(edit) injecting edit-donor at bash decisions, late layers:")
    for L in late: log(f"  L{L:2d}: {dP[L]:+.3f}")

out = {"model": MODEL_ID, "n": len(rows), "num_layers": NL,
       "choice_dist": {c: choices.count(c) for c in set(choices)},
       "locate_mean_prof": mean_prof,
       "fidelity_late_gap": {"edit_choosers": gap_late(edit_rows), "bash_choosers": gap_late(bash_rows)},
       "one_token_dP_late": dP}
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
