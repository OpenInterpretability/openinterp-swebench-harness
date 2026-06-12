# RESULTS — What do the commit heads (8,6,3) attend to?

**Run:** 2026-06-12 · Qwen/Qwen3.6-27B (loaded `attn_implementation="eager"`) · n=60 EDIT + 60 BASH ·
Colab G4. **Code:** `scripts/commit_lever_attn.py`. **Ledger:** HF `…:results/commit_lever_attn.json`.
Follows `RESULTS_heads_L59.md` (the elicit lever = a sparse ~3-head push circuit at L59: heads 8/6/3 vs an
opposing set 16/19/21).

## Method
Loaded eager so `out.attentions` exposes weights. The hybrid stack emits attentions only for its 16
full-attention layers; L59 is the 15th → compact index 14 (validated: smoke shape [24, k], rows sum to 1).
At the decision token (`<function`, last query position) we read the L59 attention distribution over all keys
for heads {8,6,3} (commit), {21,16,19} (opponents), {12} (random), in both the EDIT (committed) and BASH
(wandering) state, and summarize attention entropy, locality (mass on last 32/128 keys), and the highest-mass
decoded key tokens.

## What the commit heads read (EDIT state, mass on a token aggregated over rows)
| head | top attended tokens (mass) | entropy | rec128 (locality) |
|---|---|---|---|
| **8** | `ash` .26 · `bash` .08 · `<think>` .07 · `=str` .06 · `str` .05 · `}}}` .02 | 3.13 | 0.10 |
| **6** | `ash` .08 · `=str` .07 · `<think>` .05 · `str` .04 · `\n` .04 · `}}}` .03 · `_replace` .01 | 3.99 | 0.12 |
| **3** | `ash` .16 · `=str` .13 · `str` .07 · `function` .05 · `":` .05 · `}}}` .04 · `_replace` .01 | 2.97 | 0.16 |
| opp 21 | `=b` .24 · `=str` .21 · `ash` .14 · `str` .06 | 2.29 | 0.08 |
| opp 19 | `=` .21 · `=b` .16 · `\n` .07 · `>` .05 · `_editor` · `view` | 3.22 | 0.10 |

## Findings (honest)
1. **The commit heads read the trajectory's TOOL-CALL HISTORY, not a semantic verdict.** Their mass lands on
   the tokens that spell prior tool invocations — the tool *names* `bash` (`ash`/`bash`) and
   `str_replace_editor` (`=str`/`str`/`_replace`/`_editor`) and the function-call JSON scaffold
   (`function`, `}}}`, `":`). My prior hypothesis ("they read the consolidated 'all tests pass' verdict") is
   **not supported** — there is no semantic success-token in the top mass. The mechanism is more mundane and
   concrete: **the decision attends to which tools were used.**
2. **Global, not local.** Locality is low (rec128 ≈ 0.10–0.16 in the committed state) — the heads reach back
   across the whole context to the tool-call tokens, not the recent window. This is consistent with an
   **induction / copy-style** head that promotes the next action token from the trajectory's tool-call pattern.
3. **Commit vs opponent is real but graded.** Head 3 concentrates on the `str_replace_editor` tokens
   (`=str`/`str`/`_replace`) and `bash`; the opponents read slightly different targets — head 21 splits its
   mass between `=bash` and `=str` (≈0.24/0.21, more balanced → no net push), head 19 reads syntactic glue
   (`=`, `>`, `\n`, `_editor`/`view`). So the push heads emphasize the *edit-tool name* while the opponents
   either balance the two tools or read editor syntax — a plausible basis for the push–pull we measured.
4. **EDIT vs BASH.** Top tokens are similar across states (the heads always read tool-call tokens), but in the
   EDIT (donor) state locality drops further (more global) and the commit heads weight the
   `str_replace_editor` tokens a touch more — the attention difference that the donor swap injects.

## Caveat
Many of these are frequent structural tokens, so the commit-vs-opponent differentiation is graded, not a clean
dichotomy. This is a top-token *aggregate*, not a causal attention-knockout. The decisive next test (not run):
ablate each commit head's attention specifically to the `str_replace_editor` key tokens and check the elicit
dies — that would prove the heads *copy the tool name* rather than merely co-occur with it. Single model,
single decision position, n=60.

## One-line mechanism (three levels deep, honest)
The agent's late commitment to an action is written by **~3 induction-style attention heads at L59 that read
the trajectory's tool-call history and promote a tool name**, partially opposed by a counter-set — not by an
MLP feature and not by attending to a semantic "task done" verdict.
