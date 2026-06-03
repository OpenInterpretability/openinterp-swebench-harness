# decision-locator

**Find and steer the layer where a tool-calling agent commits a decision.**

A small, dependency-light (`torch` + `transformers` only), model-agnostic tool that locates *where inside a
language model* a discrete action decision is computed — and then steers it. It is the method behind the
WANDERING arc's first positive result ([paper #6, "The Lever Is Late"](../../paper/breakthrough/verdict_lever_paper.pdf)):
for a long-horizon agent, the control surface of a decision (e.g. *emit `finish`* vs *keep working*) is **not**
the mid-layer representation that *predicts* it, but a **late action-commitment block** — and steering that
late block with a **task-matched** donor makes the agent actually emit the action.

## Why

The interpretability field repeatedly finds that internal states which *predict* a behavior do not *control*
it (the knowledge–action gap). This tool turns that into something actionable for agents: it answers, for a
specific decision, **(1) where is the decision readable, (2) where is it writable, and (3) does writing it
produce a real action emission** — not just a probability bump.

## Install

No packaging needed — copy the folder, or:

```python
import sys; sys.path.insert(0, "path/to/tools")
from decision_locator import DecisionLocator
```

Requires `torch` and `transformers`. Verify it works on your box (tiny random model, CPU, no download):

```bash
python tools/decision_locator/smoke_test.py     # -> ALL SMOKE CHECKS PASSED
```

## The three primitives

A **decision point** is a prompt whose final token leaves the model poised to choose an action. For a
tool-calling agent, that means the prompt ends right before the function-name token (e.g. just after
`<function`). You supply the target action's first token id and the alternative action token ids.

```python
from decision_locator import DecisionLocator, commitment_layer

loc = DecisionLocator(model, tokenizer)        # auto-resolves decoder layers + final norm

# 1) LOCATE — where does the decision become readable? (logit-lens gap per layer)
gaps = loc.locate(ids, target_id=FINISH_TOK, alt_ids=[BASH_TOK, EDIT_TOK])

# 2) SWEEP_PATCH — where is it writable? (ΔP(target) when injecting a donor state per layer)
donor_by_layer = {L: loc.donor_state(success_prompt_ids_list, L) for L in LATE_LAYERS}
dP = loc.sweep_patch(ids, donor_by_layer, LATE_LAYERS, FINISH_TOK, [BASH_TOK, EDIT_TOK])
lever = commitment_layer(dP)                    # layer with the largest causal effect

# 3) STEER_GENERATE — does writing it emit a REAL action? (patch the decision position, decode freely)
donor = loc.donor_state([task_matched_success_ids], lever)   # task-matched beats a class mean
continuation = loc.steer_generate(ids, layer=lever, donor=donor, max_new_tokens=24)
print(continuation)   # e.g. "=finish> <parameter=output> ..."  -> the agent actually finished
```

**Key finding baked into the API:** a *task-matched single donor* steers far better than a class mean (paper
#6: 42% real-`finish` flip with a task-matched donor, `p=0.031`, vs 25% n.s. for the class mean). Prefer a
one-element `donor_state([same_task_success_ids], L)` over the full class.

## Worked example — Qwen3.6-27B `finish` decision (the paper's setup)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from decision_locator import DecisionLocator

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", torch_dtype=torch.bfloat16,
                                             device_map="auto", trust_remote_code=True)
loc = DecisionLocator(model, tok)

# action-selector token ids right after "<function": finish='=' bash='=b' editor='=str'
FINISH, BASH, EDIT = 28, 21402, 15462
ids = build_decision_point(wandering_trajectory)   # prompt ending at "<function"

gaps  = loc.locate(ids, FINISH, [BASH, EDIT])                       # flat <L31, explodes L51-63
donor = loc.donor_state([task_matched_success_ids], layer=55)      # task-matched late donor
print(loc.steer_generate(ids, layer=55, donor=donor))              # -> emits "=finish> ..."
```

## Supported architectures

Auto-resolves decoder layers and the final norm for Qwen, Llama, Mistral, GPT-2, GPT-NeoX, and similar HF
causal LMs. For anything else, pass `DecisionLocator(model, tok, layer_modules=list(model.<...>.layers))`.

## Caveats (from the paper)

- The steered effect is **partial** (a fraction of decision points flip) and **task-conditioned** — there is
  no single generic "finish direction". Use a task-matched donor.
- `steer_generate` patches the decision position only (prefill); patching *every* decode step degenerates
  into repetition. Don't do that.
- A positive ΔP is necessary but not sufficient — always confirm with a real generation that emits the action.

## Cite

> C. Vicentino. *The Lever Is Late: Causal Control of Long-Horizon Agent Termination Lives in a Task-Matched,
> Late Action-Commitment Block.* OpenInterpretability, 2026. WANDERING arc paper #6.

Part of the [WANDERING arc](../../paper/) on long-horizon agent failure.
