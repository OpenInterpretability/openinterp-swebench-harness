# EVAL — pre-mint adversarial review of paper #7 (`lever_generalizes.tex`)

**Reviewer pass 2026-06-10.** The arc's pre-publication gate: recompute every number against released data,
verify every citation exists (the fabrication check that caught Hong→Du / "Cohen"→Fernando in earlier
papers), hunt overclaims, check figures and internal consistency. Compiles clean (6pp, 3 figs, 11 refs, no
undefined refs, no overfull boxes).

## A. Numbers — VERIFIED against HF (`results/commit_lever_partial.json`, `cross_model_*.json`)
Every quantitative claim in the `.tex` was recomputed from the released per-point data and matched:
- Fidelity 0.474 / 0.299 ✓; elicit baseline 0.23, L59 0.77, L55 0.62, position 0.08, cross-task 0.48 ✓;
  brake baseline 0.48, L55 0.02, L59 0.03, same-class ctrl 0.55, opposite 0.92 ✓;
  dense L23 −0.003, L63 +0.685, bash-null late −0.28 ✓.
- **Exact McNemar re-derived from the per-point vectors**: L59 4.7e-10 (b=32,c=0), L55 2.4e-7 (b=23,c=0),
  position 9.1e-13, cross-task 7.6e-5 (b=18,c=1), brake L55 7.5e-9 (b=0,c=28), brake L59 1.5e-8 (b=0,c=27)
  — all match the paper to the digit. Holm m=7 worst 7.6e-5 ✓.
- Cross-model: L19 +3.2 / L23 +6.8 / L27 +10.3; writability edit +0.15 vs neutral −0.61 @L27 — all match.
**No fabricated or mistranscribed number.**

## B. Citations — VERIFIED, none fabricated
| ref | check |
|---|---|
| Turner et al., Activation Addition, arXiv:2308.10248 | ✅ web-verified |
| Zou et al., Representation Engineering, arXiv:2310.01405 | ✅ web-verified |
| Jimenez et al., SWE-bench, ICLR 2024, arXiv:2310.06770 | ✅ web-verified |
| 5× WANDERING-arc self-cites (Lever Is Late / Tool-Entropy / Right-Locus / Modality / Verdict) | ✅ DOI-verified (own records) |
| nostalgebraist, logit lens, LessWrong 2020 | ✅ canonical |
| McNemar 1947 (Psychometrika); Holm 1979 (Scand. J. Stat.) | ✅ classic, exact venue/year |

**Deliberately OMITTED:** the "knowledge–action gap" Basu reference (memory had an arXiv 2026 id I could not
verify) — phrased generally instead, to avoid a fabricated citation. **No unverifiable citation included.**

## C. Overclaims — the EVAL_commitment_lever corrections are PRESENT, not reintroduced
- ✅ "general action-channel property" → "generalizes… existence proof", not universal law (abstract,
  contributions, §7).
- ✅ "routes to exploration" → §7 explicitly: "bash is the dominant action (0.60), but we did not measure the
  no-brake bash baseline… we report the brake-state composition, not 'routes to exploration'."
- ✅ emission = tool-token onset; 0.77 is "an upper bound on valid-call elicitation" (§7).
- ✅ semi-irreversible proxy; irreversible `send_transaction` named as untested next step (abstract, §7).
- ✅ cross-model honest deltas stated (7B scale; writable window starts mid; lite, no generation).

## D. Figures — data-grounded, 3
1. Qwen one-token ΔP profile (edit-donor writes late, brake goes negative) — from `dense_dP`.
2. Cross-family LOCATE (Mistral late-emergence) — from `cross_model_*.json`.
3. Bidirectional elicit/brake bars with controls — from the block rates.
All generated programmatically from the released JSON; no hand-drawn numbers.

## E. Consistency / residual nits
- Abstract says brake "96\%"; §4 0.48→0.02 = 95.8% — consistent (rounded).
- L51–L63 (Qwen, 64 layers) vs L55/L59 generation layers — stated correctly (one-token profile vs
  generation-tested layers).
- Cross-model uses "bash"→"b" generic token (noted in RESULTS, not load-bearing); the paper's cross-model
  claim rests on the edit-direction and the neutral control, both clean.

## F. Verdict
**Numerically clean, citations all verified, overclaims controlled, figures data-grounded — paper #7 is
mint-ready as a workshop draft.** Recommended before submission (not blocking the draft): (1) the cheap GPU
follow-ups from `EVAL_commitment_lever.md` (h2_bash_baseline, valid-call re-parse) to retire the two
remaining caveats; (2) a scale-matched cross-model; (3) choose an OpenReview MI workshop (the indexing path).
Optional: a fourth figure for the cross-model writability bars if a venue allows more.
