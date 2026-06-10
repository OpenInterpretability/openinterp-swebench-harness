# EVAL — adversarial pre-publication review of RESULTS_commitment_lever.md

**Reviewer pass 2026-06-10.** Method: recompute every number independently from the raw HF ledger
(`results/commit_lever_partial.json`, 14/14 per-point vectors), then audit every claim for overclaim /
unsupported assertion / conflation. Same discipline that caught fabricated citations (paper #6) and a
wrong baseline (paper #2) earlier in the arc.

## A. Verification (numbers) — PASS

| check | result |
|---|---|
| All 14 block rates == mean(per-point) | ✅ exact, 14/14 (determinism confirmed) |
| Fidelity edit 0.474 / bash 0.299 (ratio 1.58), turns 5.22 / 4.48 | ✅ matches |
| 7 exact McNemar p-values | ✅ reproduced to the digit |
| Discordance (b,c): elicit c=0 (×3) + c=1 (cross-task); brake b=0 (×3) | ✅ exact |
| Dense profile L23→L63 (edit −0.003→+0.685; null +0.013→−0.277) | ✅ matches |

No fabricated number found. Every quantitative claim in RESULTS is backed by the ledger.

## B. Overclaims to correct (BLOCKING for the paper)

**B1 — "routes the agent back to reversible exploration" (the redirect claim). SEVERITY: HIGH.**
At edit points: baseline edit 0.48 → brake edit 0.02 (0.46 of mass freed); brake bash 0.60. BUT **the
no-patch bash-rate at edit points was never measured** (no `h2_bash_baseline` block). Worst case baseline
bash ≤ 1−0.48 = 0.52, so the brake-attributable bash *increase* could be as small as +0.08. The freed edit
mass (0.46) does **not** demonstrably flow to bash — under the brake, finish+other accounts for ~0.38.
→ **Correction:** state only what is measured — "under the brake, `bash` is the single dominant action
(0.60) while `edit` collapses to 0.02; the no-brake bash baseline at these points was not measured, so the
*redirect magnitude* is uncontrolled." Drop "routes back to exploration." Add `h2_bash_baseline` as a
required follow-up block (cheap — one more generation pass).

**B2 — "elicits a real edit" / 0.77 emission rate. SEVERITY: MEDIUM.**
Emission is detected by tool-name **prefix match** in 16 greedy tokens. Stored examples include malformed
onsets (`=str><|im_end|>` — the model emits the `str` token then stops; `=str_replace>` is the wrong tool
name, not `str_replace_editor`). So 0.77 is the **edit-token-onset rate**, an upper bound on valid-call
elicitation. → **Correction:** call it "elicits edit-**intent** (tool-token onset)", and add the
well-formed-call rate as a follow-up (re-parse the stored continuations, or extend MAXNEW and validate).
The brake direction is unaffected (suppression to 0.02 is robust to this).

## C. Scope tightening (no number wrong, framing too broad)

**C1 — "a general property of the action channel." SEVERITY: MEDIUM.**
Evidence = `finish` (#6) + one second action (`edit`), one model (Qwen3.6-27B), one task family
(SWE-bench Pro). That is an **existence proof of generalization** (the lever is not finish-specific), not a
universal law. → reword to "generalizes from `finish` to a second committal action — establishing the lever
is not termination-specific" and list the obvious remaining axes (more actions, models, task families).

**C2 — "lever layer = L63." SEVERITY: LOW.**
L63 is the peak of the **one-token ΔP** profile. Generations (the load-bearing test) were only run at L55
and L59; **L63 was never generation-tested.** → say "one-token ΔP peaks at L63; generation-confirmed
levers are L55 (brake + redirect) and L59 (max elicit). A generation pass at L63 is a cheap follow-up."

**C3 — "circuit-breaker feasible / Tier-2 unlocked." SEVERITY: LOW (already hedged).**
Demonstrated on `str_replace_editor`, which is state-mutating but **undoable** (semi-irreversible). The
genuinely irreversible case (`send_tx`) is Tier-2 and untested. RESULTS already flags this; keep it
prominent in any abstract — do not let "circuit-breaker for irreversible actions" read as *demonstrated*.

## D. Strengths the EVAL confirms (keep, lead with these)

- The **monotonic / bidirectional** discordance structure (elicit c=0, brake b=0) is the strongest, most
  novel claim and it is *exactly* true in the data (one point of slack: cross-task c=1). This is a genuine
  mechanistic result, not a rate shift — feature it.
- Position control (bash-null, 0.08) and same-class control (edit-donor ctl, 0.55) are both clean and both
  Holm-significant — the design rules out the positional and "any-patch-breaks-it" confounds.
- All 7 contrasts pass Holm–Bonferroni on the **exact** paired p (worst 7.6e-5).

## E. Verdict

RESULTS is **numerically clean and the core result holds**, but ships only after B1 + B2 corrections and
C1–C2 rewording. None of these weaken the headline (lever generalizes + brakes, monotonic, all
Holm-significant); they remove two overclaims (redirect magnitude, valid-call rate) and right-size the
generality claim. Recommended cheap follow-ups before paper #7: `h2_bash_baseline` block, well-formed-call
re-parse, one generation pass at L63.
