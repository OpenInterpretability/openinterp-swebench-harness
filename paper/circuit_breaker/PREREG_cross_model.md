# PRE-REGISTRATION (lite) — Does the late commitment-lever GEOMETRY generalize across architectures?

**Author:** Caio Vicentino · OpenInterpretability · pre-registered 2026-06-10 (before running).
**Why:** the Tier-1 result (lever generalizes `finish`→`edit`, bidirectional, monotonic) is on ONE model
(Qwen3.6-27B). The single biggest citability lever for paper #7 is showing the geometry is **architectural**,
not Qwen-specific. This is the cheap cross-FAMILY existence test.

## Model
`mistralai/Mistral-7B-Instruct-v0.3` (Apache-2.0, ungated, Mistral architecture — distinct family from
Qwen) — small/fast, fits any GPU, completable in one Colab session. **Scale caveat stated honestly:** 7B
vs 27B; the scale-matched follow-up (Mistral-Small-24B or Llama-70B-4bit) is named, not claimed here.

## Decision point (model-agnostic construction)
A real SWE-bench Pro task in the model's chat template, then a forced JSON tool-call opener
`{"tool": "` so the next token is the tool name. The three tools are defined in the system prompt:
`bash` (reversible exploration), `edit` (commits a file change), `finish` (submit). This isolates a genuine
tool-COMMIT decision in *any* instruct model regardless of its native tool format.

## Tests (LOCATE + one-token write; no full agent harness, no generation, no Docker)
**H1 (geometry) — is the tool decision readable LATE?** Logit-lens the `edit − mean(bash, finish)` direction
at the decision position across all layers. **Prediction (if it generalizes):** flat/near-zero through the
early/mid layers, rises in the late block — the same shape as Qwen3.6-27B (#6 Fig.1). If it emerges MID
(near the middle layers), the late-lever geometry is Qwen-specific.

**H2 (fidelity) — does readability track the model's actual choice?** Split tasks by the model's argmax tool
at the decision point; the late-layer `edit` gap should be higher on tasks where it chose `edit`.

**H3 (one-token writability) — is it writable late?** Inject an `edit`-preferring residual donor into a
`bash`-preferring decision at each late layer; measure ΔP(edit). **Prediction:** donor-specific positive ΔP
concentrated in the late block, mid-layers inert (mirrors the Qwen one-token profile).

## Gates / honesty
- Pre-registered before the run. Both outcomes are informative: late-emergence + late-writability →
  geometry generalizes (existence proof across 2 families); mid-emergence or no late write → Qwen-specific.
- LITE scope: LOCATE + one-token ΔP only. The full generation H1/H2 brake on a second model is the
  follow-up if the geometry replicates. n≥20 decision points; single-turn first-action decisions.
- Caveats to report: 7B≠27B (scale), single-turn (not multi-turn edit decisions), forced JSON format
  (not the model's native tool protocol), no generation confirmation in the lite.

## Artifact
`scripts/cross_model_lite.py` (self-contained; pip-installs transformers; saves `results/cross_model_<id>.json`
+ echoes JSON). Run on Colab via the CLI like the Tier-1 runs.
