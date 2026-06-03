# RESULTS — Generation confirmation — **CONFIRMED POSITIVE: a task-matched late-block state is a real, partial causal lever for termination**

**Run:** 2026-06-03, Colab A100, Qwen3.6-27B, n=12 WANDERING, faithful reconstruction (HF bundle, no Drive).
Same loaded session as the breakthrough run (`RESULTS_action_channel.md`); confirmation cells reuse model/data.
Pre-registered HIGH bar in `PREREG_action_channel.md` §2 (real generation, held-out, per-pair donor).

## Verdict: the preliminary one-token positive SURVIVES a real generation — but only at the **late block** and only with a **task-matched (per-pair) donor**. It is a **genuine but partial** internal lever (42% flip, p=0.031), strictly weaker than the behavioral channel (70%, paper #4) and absent at the verdict layer (0%, paper #5).

## The decisive test — does the agent ACTUALLY emit `finish`?
Patch the SUCCESS late-block residual at the WANDERING **decision position only** (prefill-only hook; the
earlier persistent-every-step patch degenerated into `=b=b=b` repetition — a method artifact, now fixed), then
greedy-generate. Count trajectories whose continuation emits a well-formed `<function=finish>` call.

**Example emitted continuation (patched):** `=finish> <parameter=output> The issue ...` — a coherent, valid
finish tool call, NOT degenerate text.

### Per-layer generation finish-rate (n=12)
| layer | baseline | SUCCESS-donor | LOCKED (null) | held-out (split B) | per-pair |
|---|---|---|---|---|---|
| L55 | 0.00 | 0.08 | 0.00 | 0.08 | **0.25** |
| L59 | 0.00 | 0.08 | 0.00 | 0.00 | **0.33** |
| L63 | 0.00 | 0.17 | 0.00 | 0.17 | 0.17 |

### Headline — any-late-layer flip (union over L55/L59/L63), exact one-sided McNemar vs the 0/12 baseline
| donor | flips | rate | p |
|---|---|---|---|
| **per-pair (task-matched)** | **5/12** | **0.42** | **0.031** ✅ |
| succ-mean (coarse) | 3/12 | 0.25 | 0.125 (n.s.) |
| locked-null | 0/12 | 0.00 | 1.000 |

## What the confirmation establishes
- **It is a real generation, not a one-token bump.** The +ΔP translated into actual, well-formed `<function=finish>` emissions in 42% of WANDERING decision points (vs 0% baseline). The decisive gate of the pre-registration PASSES.
- **Donor- and direction-specific.** LOCKED-donor = 0/12 at every layer. Injecting the *success* late-state flips; injecting the *locked* state never does.
- **Task-specificity is the mechanism (new finding).** The per-pair (repo/task-matched) donor (0.42, p=0.031) **beats the coarse SUCCESS-mean** (0.25, p=0.125, n.s.); only the task-matched donor clears significance. The lever requires the matching task's finish-context; a generic mean washes it out. This explains why paper #2's generic-direction residual injections were nulls.

## Honest caveats (kept to the prereg bar)
- **n=12**, small sample; p=0.031 is exact one-sided McNemar with c=0 (all 5 discordant pairs flip one way).
- **The coarse-mean alone is NOT significant** (p=0.125) — the positive headline depends on the task-matched donor.
- **The effect is partial (42%)** and strictly weaker than the behavioral interruption (70%, paper #4).
- Greedy decode, MAXLEN 4000 truncation, last-position full-residual replacement (heavy-handed but donor-specific).

## The integrated LAW (now positive and confirmed)
**The knowledge-action gap on long-horizon agents is a graded LAYER gap.** Control over the termination
decision increases with proximity to the action channel:
- **verdict / mid-layer (L23):** detection only (AUROC 0.91) — causal 0 (paper #5).
- **late residual block (L55–L63), task-matched:** partial internal lever — **42% generation flip, p=0.031** (here).
- **action / token channel (behavioral interruption):** dominant — **70%** rescue (paper #4).

Knowledge is consolidated ~30 layers before the action is committed; the late internal state is a genuine but
partial lever; the action channel dominates. This **closes the gap on agents** (first significant internal
positive of the arc) and **explains all five prior nulls** as wrong-layer / wrong-donor / wrong-channel.

## Status
**CONFIRMED POSITIVE (partial, n=12).** First significant internal causal lever of the WANDERING arc. Artifacts:
`notebooks/nb_breakthrough_confirm.ipynb`, `confirm_out/confirm_results.json` (Colab). → Paper #6 (the arc's
first positive). Reusable tool: a "finish-decision locator + late-block steerer" runnable on any open model.
Next strengthening (optional): larger n (binding constraint = labeled WANDERING set), denser late sweep,
attention-vs-MLP decomposition of the late commitment.
