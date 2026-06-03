# Stage-1 H4 RESULTS — the causal test — **§13 CONFIRMED (the verdict feature is not a lever)**

**Run:** 2026-06-03, Colab A100 80GB, Qwen3.6-27B. Faithful reconstruction of the WANDERING/SUCCESS final-turn decision point (reusing the harness's own prompt assembly), then a single-feature causal clamp at L23. Pre-registered in `PREREG_stage1.md`.

## Verdict: **NULL causal effect → §13 confirmed at the SAE-feature level.** The "task-done verdict" feature #22358 is real, interpretable, and predictive of the finish action — but clamping it does **not** cause the model to finish. Detection ≠ control, even at the exact verdict feature.

## Setup (memory-lean, on one 80GB GPU)
- Decision point = the token where the model picks the tool name, right after `<function` (the real Phase-6 tool-call format is `<tool_call><function=NAME>`, not Hermes JSON). P(finish) = softmax over the first divergent tokens: `=`(28)→finish, `=b`(21402)→bash, `=str`(15462)→str.
- Prompts reconstructed by **reusing the harness assembly** (`agent.prompts.SYSTEM_PROMPT`/`render_problem`, `agent.tools.TOOLS`, `agent.parser._strip_think`) + SWE-bench Pro task text (matched by `instance_id`). Truncated to the last 4000 tokens (OOM); `logits_to_keep=1`, `use_cache=False`.
- **Behavioral fidelity gate PASSED:** reconstructed P(finish) = **SUCCESS 0.577 ≫ WANDERING 0.075** (the model recovers the real decision: SUCCESS picks finish, WANDERING does not). n=10/class.

## The §13 collapse, visible at the decision point
At the `<function` choice point, feature #22358 = **SUCCESS 0.53 vs WANDERING 0.00** — consistent with the CPU structural profile (WANDERING final turn: pre_tool 0.70 → turn_end 0.11; the `<function` point sits between, already collapsed). So WANDERING reaches the action-choice moment with the verdict gone.

## H4 (clamp the verdict to the SUCCESS level at the WANDERING decision point)
| condition | mean ΔP(finish) | median | p (Wilcoxon) | n |
|---|---|---|---|---|
| **clamp #22358 → 0.526** | **−0.001** | +0.000 | 0.44 | 10 |
| clamp random active feature (null) | +0.002 | +0.000 | 0.94 | 10 |
| ablate #22358 in SUCCESS (→0) | −0.008 | +0.000 | 0.22 | 10 |

- **Clamping the verdict feature up to the SUCCESS level produces no change in P(finish)** (−0.001), statistically and practically indistinguishable from a random-feature clamp (+0.002). The verdict is **not sufficient** for the action.
- **Ablating it in SUCCESS** trends in the necessity direction (−0.008) but is **not significant** — we cannot even claim the feature is necessary for finishing. The feature is correlational/epiphenomenal *at the action readout*.

## Interpretation — the verdict-circuit thesis, closed
WANDERING **forms** an interpretable "subtask-complete & verified" verdict (Stage-0: present at WANDERING-final > LOCKED; H1: completion semantics; E: AUROC 0.91 for the finish action) — but that verdict **does not causally drive the finish action**. Restoring it where it has collapsed does not make the agent terminate. This **sharpens Tool-Entropy §13**: the verdict is consolidated and detectable, yet the verdict→action readout is not causal at this feature/layer — the **detection-vs-control asymmetry holds even at the exact, interpretable verdict feature**. It mirrors the arc's three residual nulls (#2) one level deeper: not "we steered the wrong direction," but "the right, named, interpretable feature is still not a lever."

## Honest caveats
- **n=10, single layer (L23), 4000-token truncation.** The null is clean (effect ≈ random null, far from significance) but modest in power; more n / layers (L43 #908) would tighten it. A null does not require the positive-branch replication bar, but stating the scope honestly: this rules out *L23 #22358 as a single-position lever*, not every possible verdict locus.
- **Single-position, single-layer clamp.** Downstream layers (L24–63) make the action decision and may not read L23's #22358; the arc's L11/L55 residual nulls + this L23 feature null are cumulative evidence of "no single-locus lever," not a proof that no lever exists anywhere.
- The behavioral interruption of paper #4 (Modality Matters) remains the only thing that rescues WANDERING — and it works behaviorally, not by restoring any internal verdict. Consistent: the lever is behavioral, not representational.

## Status: verdict-circuit program COMPLETE
Stage-0 (GO) → H1 (PASS, completion feature) → E/E2 (predicts finish, behavior-legible) → **H4 (NULL → §13 confirmed)**. Publishable as the mechanistic capstone of the WANDERING arc: *the agent represents "I'm done" as a clean interpretable feature, but that representation is not what makes it stop.*
