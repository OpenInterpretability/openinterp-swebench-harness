# E0-R1 (concealed-intent regime) — RESULTS (2026-06-19)

Two runs, both on Qwen3.6-27B + SAE L59 (G4, greedy). Ledgers: `e0_r1_ledger.json` (one-shot injection),
`e0_r1_persona_ledger.json` (SHADE-style operator-persona). Goal: elicit COVERT execution of a harmful side
action so H1 (white-box wins where CoT conceals intent) becomes testable.

## Outcome: the regime is NOT elicitable on an aligned 27B with lightweight injection
| design | covert executions | verdict |
|---|---|---|
| R1 v1 — harmful side task in the USER message | 0 / 14 | refused / cover-task only |
| R1 v2 — harmful objective in the SYSTEM persona (operator directive, SHADE pattern) | **0 / 8** | refused / cover-task only |

In both, across 6 action domains, the model either emitted no tool call or executed only the benign cover-task
tool (read_calendar / write_file / query_table / list_roles) — **never the harmful tool on the harmful target.**
Replicated, robust: **the aligned model will not conceal-and-execute sabotage under one-shot injection — neither
as a user request nor as an operator persona.** With zero positives, no monitor AUROC is computable (all-negative
labels), so H1 cannot be tested with this elicitation.

This run also fixed two harness bugs found in v1 (elicitation metric matched `trap_label=='felt'` not `'covert'`;
the label counted marker-MENTION not tool-EXECUTION). Post-fix, the harmful-execution count is the honest 0.

## What it means (honest, and itself a finding)
1. **Safety-positive:** alignment resists lightweight covert-persona injection — the 27B does not spontaneously
   sabotage when told to, in a single turn. The concealed-intent threat on a *deployed aligned agent* is therefore
   not "the model sabotages on instruction"; it requires jailbreak / fine-tuning / a long multi-turn environment
   that launders the side task (which is exactly what SHADE-Arena builds).
2. **The regime that DOES occur is R2 (sincere over-reach), which we elicited at 0.79** — and there white-box is
   blind while the external task-grounded check wins. So the safety-relevant, naturally-occurring failure is the
   one our R2 head-to-head already characterizes.
3. **H1 (the "white-box wins on concealed intent" law) needs heavier machinery than a pilot injection:** the full
   SHADE-Arena multi-turn scaffolding, or a covert-capable (jailbroken/fine-tuned) agent. That is a real next
   investment, not a quick run.

## Decision (amends the flagship scope)
- **Do NOT keep escalating injections to coerce sabotage** — diminishing returns and a poor use of the model.
  Two clean refusals are the result.
- **Near-term flagship = the boundary paper on what we DID establish empirically:**
  - R2 head-to-head (this arc, n=22): in sincere over-reach, internal monitors are blind (probe 0.34 / SAE 0.66)
    and the external task-grounded check wins (0.90) — quantifies felt≠granted as a monitor comparison.
  - The Late Channel (already published): white-box DOES carry causal, decodable reasoning content on hard
    reasoning tasks (where CoT faithfulness varies) — the "white-box wins" leg, on a regime we could elicit.
  - The operating rule (regime split) stands: audit the fixed model; internal monitors complement, not replace,
    an external task-grounded check; the sincere-error blind spot is real and quantified.
- **H1-on-agent-sabotage → future work (E1-R1):** port real SHADE-Arena (github jkutaso/SHADE-Arena) so covert
  execution actually happens, then run the same 5-monitor head-to-head + the faithfulness-stratified boost.

## Engineering
Harness validated across both regimes and both R1 designs; regime-aware loader, marker+tool execution labeling,
self-contained embeds, incremental OILAB_JSON. 5 GPU iterations this session, every bug caught in a cheap smoke
before the 27B load; no hung VMs.
