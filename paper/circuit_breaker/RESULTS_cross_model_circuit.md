# RESULTS — Does the sparse attention-head commitment circuit generalize across architectures?

**Run:** 2026-06-12 · Colab G4 · N=20 synthetic edit/explore tool-decision contexts per model. **Code:**
`scripts/cross_model_circuit.py`. **Ledgers:** HF `…:results/cross_circuit_v2_<model>.json`. Follows the
Qwen3.6-27B head circuit (`RESULTS_heads_L59.md`) and the lite cross-model lever replication of paper #7.

## Method & its honest limit
We reuse the paper-#7 lite protocol (synthetic single-turn forced-JSON next-tool decision; no agent runs) and
add an architecture-agnostic decomposition: swap the o_proj **input** (attention channel) and the **mlp output**
from the explore-context point to the task-matched edit-context donor, and let the rest of the block run
(robust to gating / sandwich-norm / MoE). We locate L\* by the **attention-channel** peak (not full-swap, which
saturates early), then per-head swap. **Key limit discovered on the first run:** synthetic contexts often bake
the decision **upstream** — the full-residual swap saturates by mid-depth, leaving no late attention write to
decompose. The Qwen circuit lived in **real on-the-fence trajectory decision points** where the action is
computed late. So the test only has power where a model's synthetic decision is **genuinely late** (full-swap
does not saturate).

## Results (attention-channel ΔP, mlp-channel ΔP, and head sparsity at L\*)
| model | fidelity edit/explore | full-swap | **attn** | **mlp** | sparse heads | reading |
|---|---|---|---|---|---|---|
| Qwen3.6-27B (real pts, ref) | 0.47 / 0.30 | 0.77 (rises late) | **+0.223** | ~0 | **3 heads = 0.262 (>all)** | clear circuit |
| **gpt-oss-20b** | 0.86 / 0.08 | 0.79 (late, unsaturated) | **+0.218** | +0.050 | ~8/64 heads = 0.228≈all | **clear, attn≫mlp 4:1** |
| Llama-3.1-8B | 1.00 / 0.04 | 0.96 (saturates) | +0.033 | +0.000 | head27=0.047>all; top3=0.074 | faint, structure-consistent |
| Mistral-Small-24B | 0.95 / 0.01 | 0.95 (saturates early) | +0.003 | +0.001 | all ~0 | nil (decision upstream) |
| gemma-4-31B-it | **0.00 / 0.00** | 0.00 | 0.00 | 0.00 | — | always-`bash` (no contrast) |

## Findings (honest)
1. **The LEVER replicates broadly.** In Mistral, Llama, and gpt-oss the edit/explore fidelity gate passes and
   the full-residual swap drives P(edit) (0.79–0.96) — the late commitment representation is writable across
   four families, consistent with paper #7's geometry replication.
2. **The CIRCUIT replicates where the decision is genuinely late.** Only gpt-oss has a non-saturating full-swap
   (the late block still matters), and there the structure matches Qwen cleanly: the late write is
   **attention-carried** (attn +0.218 vs mlp +0.050, a 4:1 margin) and **sparse** (~8 of 64 heads reproduce the
   whole attention channel). Llama shows the same structure **faintly** (attention ≫ mlp; a single head, 27,
   overshoots the net; push–pull). This is cross-architecture corroboration of *attention-carried, MLP-
   subordinate, sparse, push–pull* — the Qwen circuit's signature — in 2 further families.
3. **Where the synthetic decision is baked upstream, the circuit is untestable.** Mistral's full-swap saturates
   by 65% depth → the attention channel is ~0 at every late layer (nothing left to write). This is a protocol
   ceiling, not evidence the model lacks a circuit.
4. **gpt-oss bonus.** A 64-head MoE model (OpenAI's open weights) shows the sparse attention-head commitment
   write — first-on-model.

## Honest verdict
**Partial, structure-consistent cross-model corroboration, not a clean universal law.** The lever generalizes
to 4 families (already published); the sparse attention-head *circuit* replicates clearly in gpt-oss and faintly
in Llama, is untestable in Mistral (synthetic decision upstream), and **could not be measured in Gemma 4**. For Gemma 4
we adapted the harness to its native reasoning format (`tools=` template + `<|tool_call>call:` force; after the
fix `bash` is the clean top tool token, so the *format* probe works), but Gemma 4 predicts **`bash` in both the
edit- and explore-context** — it is exploration-conservative and does not commit an edit in the synthetic
single-turn setup, so there is no edit/explore contrast to decompose (a behavioral, not format, limitation).
A full cross-model circuit claim
needs **real per-model agent-trajectory decision points** (the late-decision regime), not synthetic contexts —
the named next step.

## Real-decision-points attempt (gpt-oss-20b) — honest null
We tried to give gpt-oss **real** decision points by reconstructing the released Qwen SWE-bench trajectories
(real tool-call history) and re-rendering them in gpt-oss's context (`scripts/cross_model_realpts.py`). The
**fidelity gate failed and reversed**: edit-points P(edit) 0.132 < bash-points 0.169 (both low). Reason: the
edit/bash *labels* come from **Qwen's** choices, but gpt-oss, reading the same transcript, is bash-prone and
does not reproduce that split — **decision-point labels do not transfer across models**. So there is no clean
edit/explore contrast and the donor-swap is invalid (a stray +0.087 attn-channel with sparse heads appears but
on a broken contrast → not counted). **Doing real points right requires the target model's OWN agent
trajectories** (run gpt-oss as the agent on SWE-bench), so the labels reflect its decisions — the expensive
path. The *synthetic* gpt-oss result above (fidelity 0.86/0.08, attn 0.218) remains the cleaner evidence,
because there the contrast was deliberately induced.

## Caveats
N=20 synthetic contexts/model; within-model donors; the head-sparsity is read off the o_proj-input swap (not a
generation-confirmed emit like Qwen); Gemma 4 needs per-model format adaptation. Ledgers + script released.
