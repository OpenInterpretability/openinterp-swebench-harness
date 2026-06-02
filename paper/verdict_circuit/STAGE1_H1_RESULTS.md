# Stage-1 H1 RESULTS — feature interpretation — **PASS (CPU, in-domain)**

**Run:** 2026-06-02, **CPU only** (no model load, no GPU). Used: SAE L23 + existing Phase-6 captures + the trajectory transcripts found in `swebench_v6_phase6/traces/*.json` (schema: per-turn `raw_response`/`thinking`/`content`/`tool_calls`/`finish_reason`). **Pre-registered** in `PREREG_stage1.md` (H1 is the gate).

## Verdict: **H1 PASS.** Feature L23 #22358 is a **"subtask-completed-and-verified"** feature — not a generic boundary/active-late feature (caveat #3 from Stage-0 is **falsified in the right direction**).

## Method (no model needed)
We already have #22358's activation at every (trajectory, turn, `pre_tool`) from the captures. With the transcripts now located, we read **what the agent is actually saying/doing** at the turns where #22358 fires hardest. This is the in-domain top-activations interpretation, done on CPU from existing artifacts.

## What it fires on (top `pre_tool` activations)
| act | class | turn | tool | agent text (tail) |
|---|---|---|---|---|
| 3.60 | success | 30/31 | **finish** | "…removed, restoring the default caching behavior." |
| 2.27 | success | 25/26 | **finish** | "Updated get_from_archive_bulk: Changed…" |
| 2.06 | success | 31/33 | bash | "**All 157 tests pass.** Let me view the final state…" |
| 2.05 | success | 22/35 | bash | "**All tests pass.** Now let me remove the complexity suppression…" |
| 1.97 | success | 43/46 | bash | "**The implementation is complete and tested.** Let me verify…" |
| 1.95 | success | 19/44 | editor | "**All 30 tests pass.** Now let me view the final state…" |
| 1.79 | success | 34/35 | **finish** | "**The refactoring is complete.** Let me provide a summary…" |
| 1.83 | wandering | 20/50 | editor | "The function **works as expected**. Now let me add tests…" |

~18–20 of the top-25 are explicit completion/verification statements ("All tests pass", "implementation complete", "final state", "provide a summary", "restoring", "properly defined"). The general-corpus logit-lens looked like generic punctuation/boundary tokens — a **domain-mismatch artifact**; the feature is domain-specific and crisply interpretable in-domain.

## Quantitative
- `finish` tool is **18% of the top-60 activations vs 0.9% overall → ~20× enriched** when #22358 is high.
- But most high-activations are bash/editor at *mid-trajectory subtask completions* (final-turn frac 0.23 in top-60): the feature marks **"a unit of work is complete/verified"**, not specifically "the whole task is done." 

## The refined WANDERING mechanism (better than §13's original wording)
WANDERING agents at high-#22358 are saying *"works as expected / my fix is appropriate / final state"* — they **reach the completion-verification state repeatedly** but then look for "one more thing" (add tests, verify another file) instead of terminating. Combined with the Stage-0 structural profile (in SUCCESS the feature consolidates *through* the `finish` action to turn_end 1.77; in WANDERING it is present at the pre-action decision point 0.70 but **collapses** to 0.11 after the non-finish action):

> **WANDERING does not fail to *form* the completion verdict — it forms it repeatedly. It fails to *treat it as terminal*.** This sharpens §13 ("the verdict is consolidated but the edge fails to emit"): the verdict is a *graded, repeatable subtask-completion* signal, and WANDERING's pathology is not verdict-absence but failure-to-terminate-on-verdict.

## Honest caveats
- "Completion/verification" not literally "whole-task-done": it fires at subtask completions too (the ~20× finish-enrichment is the link to termination).
- This is interpretation by reading transcript text at capture positions (turn-granularity, not per-token); a full per-token top-activations run (model on text) would refine it but is not needed to settle H1.
- The §13 **second half** (does forcing the verdict cause termination? — H4 causal clamp) and H2 (the model's internal P(finish) per class) still require a model forward pass = GPU. Transcripts are now available (`traces/`), so the only remaining blocker is a GPU backend.

## Status
H1 (the gate) = **PASS, done on CPU.** Proceed to H2/H4 when a GPU backend is up (openinterp MCP or any A100/H100). The notebook's transcript adapter must use the `traces/` `turns` schema (not `messages`).
