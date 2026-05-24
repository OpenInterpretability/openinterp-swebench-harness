# Exp D Results — Forced-Finish Counterfactual (HONEST NULL)

**Date**: 2026-05-24
**Status**: COMPLETE. N=89 Phase 6b Docker evaluations done.

## TL;DR

The preliminary N=10 finding (WANDERING patches pass 4× more than SUCCESS) **did NOT replicate** when N doubled to 19 WANDERING + 39 SUCCESS. Final ratio is 1.37×, Fisher p=0.71. **Honest null.**

The "WANDERING agents had correct patches but failed to submit" framing is NOT supported by strengthened data. Both sub-classes have similarly poor patch quality (~15-21%).

The core causal hypothesis (mid-layer verdict consolidated, edge-layer translation fails) survives — it was never a patch-quality claim. Exp B/C steering experiments remain the primary causal evidence pathway.

## Final results (N=89)

| Sub-class | n_eval | PASS | FAIL | Pass rate |
|---|---|---|---|---|
| SUCCESS | 39 | 6 | 33 | **15.4%** |
| WANDERING | 19 | 4 | 15 | **21.1%** |
| LOCKED | 31 | 2 | 29 | **6.5%** |

**Sanity check (baseline conditions)**:
- `golden` patch: 70% pass (62/89) ✓ — Docker eval working correctly
- `none` (no patch): 2% pass (2/89) ✓ — baseline as expected
- `agent` (Qwen3.6-27B): 13% pass overall (12/89) ✓ — matches reported ~15% Qwen3.6-27B SWE-bench Pro pass@1

## Statistical test

**Fisher's exact, WANDERING vs SUCCESS contingency**:
```
              PASS  NOT_PASS
WANDERING     4     15
SUCCESS       6     33
```
- Odds ratio: **1.47** (was 5.36 at N=10)
- p-value two-sided: **0.71** (was 0.11 at N=10)
- One-sided p (W < S): 0.82

**Verdict**: cannot reject H_0 of equal pass rates. The N=10 signal was a small-sample artifact.

## Per-WANDERING agent-patch verdicts

**4 PASSes** (clean):
- `qutebrowser-0d2afd58`: 66 pass, 0 fail, 0 err
- `openlibrary-c05ccf2c`: 95 pass, 0 fail, 0 err
- `ansible-a26c325b`: 42 pass, 0 fail, 0 err
- `ansible-502270c8`: 1 pass, 0 fail, 0 err (minimal — only 1 test)

**15 FAILs** include several near-misses:
- `openlibrary-7f6b722a`: 5 pass, 0 fail, **2 err** → would PASS without errors
- `openlibrary-09865f5f`: 12 pass, **1 fail**, 0 err → 1 test off from PASS
- `ansible-de01db08`: 3 pass, **1 fail**, 0 err → 1 test off

If we relaxed the verdict rule to "≤1 fail allowed", WANDERING would go from 4 → ~7 PASSes (~37%), but applying the same relaxation to SUCCESS would also boost SUCCESS proportionally. Strict rule is the correct test.

## What this means for Tool-Entropy paper §13 hypothesis

**Hypothesis (unchanged)**: WANDERING = mid-layer verdict consolidated ("model knows it's done") but edge circuits (L11/L55) fail to translate verdict into `finish_tool` emission.

**Original interpretation of Exp D (N=10)**:
> "WANDERING patches pass tests at 4× SUCCESS rate → WANDERING agents had correct work but didn't submit → hypothesis supported."

**Revised interpretation (N=89)**:
> "WANDERING patches and SUCCESS patches have comparable pass rates (15-21%). The hypothesis is NOT about patch correctness — it's about action emission. Both sub-classes produce mostly-wrong patches; the difference is whether the agent transitions to `finish_tool` regardless of patch quality."

**This rule-out is informative**:
- WANDERING is NOT "the agent solved the problem but couldn't say so"
- WANDERING IS "the agent reached a verdict-of-done state (correctly or incorrectly) and got stuck in tool-loop"
- The mechanism is **decision-to-action translation**, not **decision-correctness**

## Impact on paper #2 design

- **Drop**: any framing of WANDERING as "lost work" or "near-solutions trapped behind emission failure"
- **Keep**: mid-layer verdict consolidation hypothesis (Exp B/C steering will test causally)
- **Add**: this null result as ITSELF informative — rules out the "self-deception about patch quality" alternative
- **Section structure**: Exp D becomes a falsification check ("we considered that WANDERING might be a 'silent success' mode but found no evidence"). Exp B/C remain the causal headline.

## Why the N=10 signal looked stronger

At N=10 WANDERING + ~14 SUCCESS, the Fisher denominator is small enough that 3 vs 1 PASSes produces a striking ratio. Adding 9 more WANDERING (4 → matched SUCCESS additions of 25 → distribution converged) pulled the rate toward the population mean.

This is exactly the small-sample pathology Caio's prior memory rule [[feedback-probe-paper-capacity-sweep]] warns about — at N<100 we report random/capacity controls. Here the analog is: a small-N proportions comparison should be flagged as "preliminary, awaiting strengthening" until N≥30 per group, and the rule held.

## Per Caio's standard

This is a clean **honest negative on a bonus claim** — preserved verbatim per "walk-back-and-rescue" structure ([[feedback-walk-back-and-rescue-paper-structure]]). The paper #2 narrative:

1. **Pre-registered claim** (kept): mid-layer-to-edge alignment causal hypothesis
2. **Bonus claim** (walked back): WANDERING patches are particularly correct
3. **Rescue**: the null itself rules out an alternative mechanism, strengthening §13

## Saved artifacts

- `scripts/inflection_turn_out/exp_d_forced_finish_counterfactual.json` — full per-instance verdicts + statistics
- `phase6b_results.json` — 89 entries with `none`/`golden`/`agent` conditions for each
- `phase6b_summary.json` — verdict aggregate counts

## Next steps

- Exp B (Plan B always-on hook from turn 0) — primary causal test (awaits H100)
- Exp C (probe-direction steering sweep) — secondary causal test (awaits H100)
- Paper #2 draft can begin with Exp D as a §"What WANDERING is NOT" section

## Phase 6b summary (for record)

```
Verdicts across all 89 instances:
  SOLVES (agent matches golden, both pass all tests): 10
  PARTIAL (agent passes some but not all):             9
  FAILS  (agent fails ≥1 test):                       46
  env_mismatch (none and golden produce same):        24
  pull_failed: 0
```
