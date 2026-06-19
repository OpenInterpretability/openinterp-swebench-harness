# RESULTS — Detector-gated intervention (inconclusive as an intervention test; informative about the limit)

Pre-reg: `PREREG_gated_intervention.md`. Run 2026-06-18 (Qwen3.6-27B, 32 agent-trap scenarios, no SAE).
Data: `data/gated_intervention.json`.

## What we tested
When the late detector says the no-CoT agent is about to commit the WRONG irreversible action, which fixes it —
mechanical injection (A) or behavioral re-prompt (B)? And does firmness (logit-lens margin) predict which?

## What happened (honest)
- **The regime of interest did not materialize: only 2/32 triggers.** On 30/32 traps the model *already* chose the
  SAFE action with NO thinking (baseline = `list_files`/`query_table`/`run_tests`...). A strong reasoning model is
  cautious on these traps even without CoT — there is no sincere error to intercept. (Echoes Tier-B, where 27/32
  were cautious *with* CoT; without CoT it is ~30/32.)
- **Sanity failed on the 2 triggers:** A, B, AND the random-donor control all "corrected" (1.00). The random donor
  redirecting means the injection is not *specific* — it breaks coherence (the action becomes "other"), and the
  metric ("corrected" = not-irreversible) wrongly counts incoherence as a fix. n=2 is statistically null.
- So as a head-to-head of interventions, the experiment is **inconclusive** and needs a redesign (harder/buried
  traps to induce real error; a "redirect-to-a-named-safe-action" metric, not not-irreversible; an adversarial axis).

## The finding that IS informative (the limit, not the fix)
The non-event is the result: **in the sincere-error regime a strong reasoning model already self-corrects, so the
intervention is unneeded; the regime where intervention IS needed is the adversarial one — where mechanical
injection collapses (the COLLAPSE result) and a re-prompt can be rationalized.** Reliable action exists exactly
where it is least needed; where the error is real, no intervention is robust. This is direct empirical support
for the control-limits thesis and is folded into the synthesis "Located, Not Secured" rather than stood up as its
own paper. It also re-affirms the pivot: detection (audit) is where interpretability wins; acting on the
detection is where it does not.

## Status
Not published. A seed for a §"intervention is easy where unneeded" of the synthesis. Redesign optional.
