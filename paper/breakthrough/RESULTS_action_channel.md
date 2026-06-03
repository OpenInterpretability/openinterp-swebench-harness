# RESULTS — The action-channel experiment — **PRELIMINARY POSITIVE: the termination lever is LATE, not at the verdict**

**Run:** 2026-06-03, Colab A100, Qwen3.6-27B, n=12/class, faithful reconstruction (HF bundle, no Drive). Behavioral fidelity PASSED with a clean 3-way ordering: **SUCCESS P(finish) 0.594 ≫ WANDERING 0.068 ≫ LOCKED 0.005.** Pre-registered in `PREREG_action_channel.md` (the "characterize the boundary" frame).

## Verdict: the agent DOES have an internal lever for termination — but it lives in the **late block (L51–L63), not the mid-layer verdict (L23)**. This is the gap-closing (positive) outcome, held to the pre-registered HIGH bar (still needs generation-confirmation + held-out replication). It explains all 5 arc nulls in one stroke: every prior intervention was at the wrong layer (or too weak a method).

## Obs — where the finish decision becomes readable (logit-lens across layers)
The SUCCESS−WANDERING gap in the `finish`-direction is **flat (~0) through L31**, then ramps and **explodes in the last layers**: L35 +0.38 · L43 +0.20 · L47 +0.36 · **L51 +1.01 · L55 +2.10 · L59 +2.73 · L63 +3.57**. At the verdict layer L23 the gap is **−0.02** (zero). → The finish decision is computed ~30 layers *downstream* of the verdict.

## H1 — layer-sweep activation patching (inject SUCCESS state at layer L into WANDERING)
| layer | success-donor ΔP(finish) | locked-donor (null) |
|---|---|---|
| L3–L47 | ≈0 (±0.016) | ≈0 |
| L51 | **+0.059** | −0.039 |
| L55 | **+0.134** | −0.063 |
| L59 | **+0.153** | −0.067 |
| L63 | **+0.747** | −0.067 |

Injecting the SUCCESS late-block state into WANDERING **raises P(finish)** (donor-specific: the LOCKED-donor does the opposite); injecting at the verdict/mid layers does nothing. The lever ramps from L51 and peaks at L63. **The internal lever exists — late.**

## H2 — no L23 feature is a lever (and the verdict literally doesn't write finish)
Steering at L23: top-finish-DLA feature ΔP −0.001, #22358 clamp −0.006, random +0.002 — all ≈0. **The verdict feature #22358's direct-logit-attribution to `finish` ranks 40950 / 40960 (near the very bottom).** The feature that *predicts* finish at AUROC 0.91 is among those that *least* write the finish token → it is correlational/upstream, causally disconnected from the action output. This converges with Obs + H1: the decision is late, not at the feature.

## The integrated finding (the candidate LAW, with a positive twist)
**The knowledge-action gap on long-horizon agents is a LAYER gap.** Knowledge (the "task-done" verdict) is consolidated mid-stream (L23, AUROC 0.91 detection) ~30 layers *before* the termination action is committed (the late block L51–L63). Intervening at the knowledge locus — any L23 feature, the residual nulls of paper #2 — fails; injecting the success state at the late action-commitment block works (and the behavioral interruption of paper #4 works because it feeds that late decision through the token stream). This **closes the gap on agents** (the first positive) and **explains the arc's five nulls** as wrong-layer/wrong-method.

## Honest caveats (the positive is EXTRAORDINARY → high bar, per prereg)
- **L63 is near-output** (last layer → norm → logits), so its +0.747 is partly trivial (you nearly overwrite the decision). The load-bearing evidence is **L51–L59 (+0.06 to +0.15)** — deep enough to be real computation, donor-specific, beating the null.
- **n=12, donor = SUCCESS-mean** (a coarse, task-unmatched donor), MAXLEN 4000 truncation.
- **The +ΔP is a one-token probability bump.** Not yet confirmed by a real GENERATION emitting `finish` + a valid action.
- **Needed to confirm the breakthrough:** (1) held-out replication; (2) a generation test — patch L55/L59 during decoding and check the agent actually emits the finish tool call; (3) per-pair (task-matched) donor patching to rule out the coarse-mean confound; (4) ideally a denser layer sweep around L48–L63 + attention-vs-MLP decomposition of the late decision.

## Status
PRELIMINARY POSITIVE — the first internal lever in the arc, located late. Next = the generation-confirmation + held-out experiment (the high-bar gate that turns this into the breakthrough). Artifacts: `notebooks/nb_breakthrough_action_channel.ipynb`, `breakthrough_out/breakthrough_results.json` (Colab). Reusable tool: a "finish-decision locator" (the layer-sweep patcher) runnable on any open model.
