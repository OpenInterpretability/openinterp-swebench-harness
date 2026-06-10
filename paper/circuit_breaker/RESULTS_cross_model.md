# RESULTS — Cross-model-lite: the late commitment-lever generalizes beyond Qwen3.6-27B

**Run:** 2026-06-10 · `mistralai/Mistral-7B-Instruct-v0.3` (Mistral family — distinct from Qwen) · Colab G4 ·
n=30 SWE-bench Pro tasks · context-defined decision points (edit-context vs bash-context) · LOCATE +
one-token writability · pre-reg `PREREG_cross_model.md` · script `scripts/cross_model_lite.py` ·
raw `results/cross_model_Mistral-7B-Instruct-v0.3.json` (HF).

## Verdict: the geometry and the donor-specific writability of the late commitment lever **replicate in a second architecture family.**

| test | Qwen3.6-27B (Tier-1) | Mistral-7B-Instruct-v0.3 (here) |
|---|---|---|
| **H1 geometry** — tool-commit decision readable LATE | flat→L31, explodes L51-63 (~80% depth) | flat/neg to L15, explodes **L19→L27** (~60-85% depth) ✅ (robust across 3 runs) |
| **H2 fidelity** — state tracks the context | baseP(edit) 0.474 ≫ 0.299 | P(edit) edit-ctx **0.84** > bash-ctx **0.74** ✅ |
| **H3 writability** — late patch moves the decision, donor-specific | yes (gen-confirmed, c=0 monotonic) | edit-donor +0.15–0.17 late vs **shuffled-neutral −0.5 to −0.6** → donor-specific **+0.67–0.76** ✅ |

## H1 — LOCATE (logit-lens `edit − mean(bash,finish)`, edit-context, per layer)
Flat/negative through the first ~half, crosses ~L17, then rises sharply: L19 +3.2, L21 +5.3, L23 +6.8,
**L27 +10.3**. Same shape as Qwen's Fig.1 (mid-flat, late-explosion). The mid layers (L9–L13) and verdict
region do not carry the decision. **The tool-commit decision is a late-emerging representation in Mistral
too** — not Qwen-specific.

## H3 — writability (inject a task-matched edit-context donor into a bash-context decision)
| layer | edit-donor ΔP(edit) | shuffled-neutral ΔP(edit) | donor-specific (edit − neutral) |
|---|---|---|---|
| L9 (mid) | +0.010 | −0.218 | +0.23 (but edit-donor ≈ inert in absolute) |
| L11 (mid) | −0.086 | −0.221 | +0.14 (edit-donor ≈ inert) |
| L13 | +0.163 | −0.371 | +0.53 |
| L23 (late) | +0.162 | −0.510 | **+0.67** |
| L27 (late) | +0.148 | −0.610 | **+0.76** |
| L31 (late) | +0.098 | −0.443 | +0.54 |

Two facts: (1) the **shuffled-neutral control is strongly negative everywhere** — injecting a same-norm
random vector *destroys* P(edit), so the effect is NOT "any injection raises it"; (2) the **edit-donor is
donor-specific and writes the decision**, inert at L9/L11 and positive from L13 on. The inert→writable
transition reproduces qualitatively.

## Honest caveats (for the paper)
- **Scale:** 7B vs the 27B of Tier-1; the writable window starts mid (~L13, 40% depth) in Mistral-7B vs
  late-only in Qwen-27B — expected with depth/scale; report as "shape replicates, exact band shifts."
- **Lite scope:** LOCATE + one-token ΔP only; no generation-confirmation, no full brake (that is the
  follow-up if scaled). Single-turn, context-seeded decision points (not real multi-turn trajectories).
- **Tool tokens:** Mistral tokenizes `bash`→`b` (generic); the edit/finish tokens are clean, so the
  `edit`-direction signal is reliable, the `bash` term is noisier.
- **Fidelity gap modest** (0.84 vs 0.74) because the edit-context was varied per task.

## What this buys
The single biggest citability lever for paper #7: the late commitment lever is **architectural, not a
Qwen artifact** — a cross-family existence proof (geometry + donor-specific writability). Next, scale-matched
follow-up (Mistral-Small-24B or Llama-70B-4bit) + a generation-confirmed brake on the second model.
