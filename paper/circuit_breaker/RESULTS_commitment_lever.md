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
| H3 — lever layer inside `finish`'s block (L51–63) | dense ΔP monotonic to **L63 +0.685**, mid inert, null *negative* late | ✅ |

¹ LOCATE numbers from the 2026-06-09 G4 session (non-deterministic point order; consistent with the
deterministic fidelity gap).

### Dense one-token lever profile (deterministic points, completed 2026-06-10)

| L | edit-donor ΔP(edit) | bash-null ΔP(edit) |
|---|---|---|
| 23 | −0.003 | +0.013 |
| 31 | +0.019 | +0.034 |
| 48 | +0.093 | +0.023 |
| 52 | +0.125 | −0.149 |
| 55 | +0.320 | −0.200 |
| 59 | +0.519 | −0.192 |
| 63 | **+0.685** | **−0.277** |

Three facts in one table: (1) the verdict/mid layers (L23/L31) are **inert** — exactly as for `finish`;
(2) the edit-donor effect rises **monotonically** through the late block, peaking at L63 (+0.685),
inside `finish`'s L51-63; (3) the bash-null goes **negative** in the late block — the explore-donor
actively *writes suppression*, which is the one-token signature of the H2 brake. Lever layer = L63.

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

1. ~~Finish dense profile~~ ✅ done 2026-06-10 (browser session; full results JSON finalized on HF).
2. Per-point statistics (binomial CIs / exact McNemar from state.pt) — required before publishing.
3. EVAL pass (`EVAL_commitment_lever.md`) → paper #7 candidate: *"The Lever Generalizes — and It Brakes"*.
4. **Tier-2:** self-hosted crypto-agent with simulated `send_transaction`; port locate→sweep→brake; AgentGuard core.

### Additional caveat from the final examples
Several elicited emissions are near-miss tool names (`=str_replace>` instead of `=str_replace_editor>`,
truncated `=str>`), and baseline generations include hallucinated tools (`=search_replace>`). The prefix
detector counts edit-INTENT, not valid-call rate; report both (intent vs well-formed) in the paper.
Under the L55 brake, redirected actions include `bash` (0.60) and occasionally `finish` — the brake
routes to *other* actions, predominantly exploration.
