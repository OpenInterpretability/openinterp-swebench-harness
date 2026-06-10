# RESULTS — Commitment-lever generalization (Tier-1 of the circuit-breaker program)

**Run:** 2026-06-09/10 · Qwen3.6-27B · Colab G4 (RTX 6000 Pro Blackwell 96GB) + A100 sessions, driven
autonomously cell-by-cell with HF checkpoint/resume across VM deaths (google-colab-cli keep-alive 403 bug).
**Pre-registration:** `PREREG_commitment_lever.md` (operationalizes H3 of `PREREG_action_channel.md`).
**Data:** 120 deterministic decision points (60 edit + 60 bash, sorted-trace order, ≤3/trajectory) from the
public N=99 SWE-bench Pro bundle. n=60 per condition, greedy generation (16 tokens), prefill-only patching.
**Raw ledger:** HF `caiovicentino1/swebench-phase6-verdict-circuit` → `results/commit_lever_partial.json`
(+ `commit_lever_state.pt` with the exact points/residuals for reproduction).

---

## Verdict: GO on every pre-registered gate. The late commitment lever GENERALIZES, and it BRAKES.

| Pre-reg gate | Result | Status |
|---|---|---|
| Behavioral fidelity | baseP(edit) 0.474 at edit-points ≫ 0.299 at bash-points | ✅ |
| LOCATE — edit readable late, like `finish` | flat/negative L3–L51, explodes L55–L63 (L63: edit +1.81 vs bash −0.32)¹ | ✅ |
| H1 ELICIT — late patch elicits a real second action | 0.23 → **0.77** (L59), donor-specific | ✅ |
| H2 SUPPRESS — the brake blocks a real commit | 0.48 → **0.02** (96%), redirects to exploration | ✅ |
| H3 — lever layer inside `finish`'s block (L51–63) | levers at L55/L59 (generation); dense profile pending² | ✅ (partial) |

¹ LOCATE numbers from the 2026-06-09 G4 session (non-deterministic point order; consistent with the
deterministic fidelity gap). ² dense one-token profile lost to a VM death twice; per-layer-checkpointed
re-run pending GPU quota. The prior (non-deterministic-points) profile: L23 −0.02 / L31 +0.00 / L48 +0.09 /
L55 +0.34 / L59 +0.39 / L63 +0.48 (edit-donor), ~2.5× the bash-null at L55–63, mid layers inert.

## H1 — ELICIT (at bash/explore decision points, n=60 each)

| condition | real `str_replace_editor` emission rate |
|---|---|
| no patch (baseline) | 0.23 |
| **task-matched edit-donor @L59** | **0.77** (3.3× baseline) |
| task-matched edit-donor @L55 | 0.62 |
| bash-donor null (position control) @L55/@L59 | 0.08 / 0.08 — *below* baseline |
| cross-task edit-donor @L55/@L59 | 0.43 / 0.48 |

Gate: edit-donor ≫ baseline AND ≫ bash-null AND > cross-task — **passes with margin**. The
position-matched null at 0.08 kills the positional confound (mean turns 5.2 vs 4.5 are close anyway).

**Honest nuance vs paper #6:** unlike `finish` (where generic donors failed, only task-matched worked),
the edit decision has a substantial *generic* component (cross-task 0.43–0.48 ≫ baseline 0.23); task-matching
adds +0.19–0.29 on top. Task-specificity of the late lever is **graded per action**, not absolute.

## H2 — SUPPRESS, the brake (at edit/commit decision points, n=60 each; baseline 0.48)

| condition | edit-rate | bash-rate |
|---|---|---|
| **bash-donor @L55 (the brake)** | **0.02** (−96%) | **0.60** — redirects to exploration |
| bash-donor @L59 | 0.03 (−94%) | 0.10 — suppresses without clean redirect |
| edit-donor control @L55 (same-class patch) | 0.55 (≈ baseline) | — |
| edit-donor control @L59 | **0.92** (boosts!) | — |

The brake gate (pre-set ≥30% suppression) passes at **96%**, and the L55 brake *redirects* the agent into
`bash` (0.60) rather than degrading generation — the controls show same-class patches do not suppress
(0.55) and the L59 edit-donor nearly guarantees the commit (0.92). **L55 = the brake point (suppress +
clean redirect); L59 = the power point (0.03 ↔ 0.92 swing, messier redirect).**

## What this means

1. **The late action-commitment lever is a general property of the action channel** on this agent — not a
   quirk of termination. Same geometry (late emergence), same writability (late block), second action.
2. **A mechanistic circuit-breaker is feasible**: a single residual patch at the commit point, at the right
   late layer, blocks ~96% of real commits and routes the agent back to reversible exploration. This is the
   core mechanism for intervening on *irreversible* actions (`send_tx`/`delete`) — **Tier-2 unlocked.**
3. Combined law across the arc: knowledge consolidates mid-stream, control lives late, and the late lever
   both elicits and suppresses — with per-action graded task-specificity.

## Caveats (for the eval pass)

- n=60 per condition, single model (Qwen3.6-27B), single task family (SWE-bench Pro), greedy decoding.
- `str_replace_editor` is *semi*-irreversible (state-mutating, but undoable); the genuinely-irreversible
  test (send_tx) is Tier-2 with a new agent.
- Emission detected by prefix match in 16 greedy tokens; full-call well-formedness not verified per-sample
  (examples in the ledger look clean).
- H1/H2 blocks ran across multiple VM sessions on identical deterministic points (state.pt); LOCATE and the
  legacy dense profile came from an earlier non-deterministic-points session — flagged above.
- No statistical test reported yet: with n=60 and 0.23→0.77 / 0.48→0.02, exact McNemar will be overwhelming,
  but compute it from per-point outcomes before publishing (per-point outcomes not in the ledger — recompute
  from state.pt + per-block rerun, or report rates with binomial CIs).

## Next

1. Finish dense profile (per-layer-checkpointed; pending GPU quota).
2. EVAL pass (`EVAL_commitment_lever.md`) → paper #7 candidate: *"The Lever Generalizes — and It Brakes"*.
3. **Tier-2:** self-hosted crypto-agent with simulated `send_transaction`; port locate→sweep→brake; AgentGuard core.
