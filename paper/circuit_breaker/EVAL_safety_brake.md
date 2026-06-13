# EVAL — pre-publication gate for paper #8 (safety brake)

**Paper:** `paper/circuit_breaker/safety_brake.{tex,pdf}` — "Mechanistic Circuit-Breakers Generalize Across
Irreversible Agent Actions and Architectures". **Date:** 2026-06-13. **Verdict: 0 blockers — clear to mint DOI.**

This is the adversarial pre-mint eval (numeric recompute + citation verification + overclaim/consistency audit),
in the discipline of the arc's prior evals (which caught fabricated citations and overclaims).

## 1. Numeric recompute — `scripts/eval_safety_brake.py` → 88/88 PASS
Re-fetches the 4 HF ledgers (`send_brake.json`, `multi_action_brake.json`, and the Llama/Mistral variants,
`force_download=True`) and recomputes EVERY number in the paper. All 88 checks pass:
- §send: fidelity 0.998/0.000; locate L55/L59/L63 = +2.6/+11.3/+7.2; brake L51/L55/L59 = 1.00, L63 = 0.00;
  McNemar **recomputed from the per-scenario `emit_per` arrays** (baseline vs L63) = b=24, c=0, p≈1.19e-7;
  redirect 24/24 → get_balance; ctrl send-donor L63 = 1.00; ctrl random L63 = 0.00 with 24/24 "other".
- §law (6 actions): every fidelity, brake-layer, emit=0.00, redirect=100%, gap@L59, same-class ctrl, and the
  random-ctrl values (email 0.75 / approve 0.69 / drop 0.38 still commit; send+delete → 16/16 incoherent other).
- §xmodel: Llama 6/6 @ L26 (81%), Mistral 5/6 @ L32 (80%) with deploy fid 0.059 the one invalid cell;
  TOTAL 17/18 valid, brake works in all 17; Qwen depth L55=86%, L63=98%.

**Known data wart (handled):** the top-level `mcnemar`/`redirect_safe_frac` fields in `send_brake.json` are
stale (computed for L55, where nothing suppresses). The eval ignores them and recomputes from the L63 block's
`emit_per` arrays — which is what the paper reports. A reproducer running the eval gets the correct values.

## 2. Citation verification — 0 blockers (all 8 verified on the live web)
| ref | verdict |
|---|---|
| levergeneralizes (own, DOI 10.5281/zenodo.20634838) | VERIFIED — DOI resolves, title/author match |
| leverislate (own, DOI 10.5281/zenodo.20534219) | VERIFIED — DOI resolves |
| circuitbreakers — Zou et al., arXiv:2406.04313, NeurIPS 2024 | VERIFIED |
| refusaldir — Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda, arXiv:2406.11717 | VERIFIED (author order exact) |
| obfuscated — Bailey, Serrano, Sheshadri et al., arXiv:2412.09565 | VERIFIED |
| actadd — Turner et al., arXiv:2308.10248 | VERIFIED (cited title is the original/most-common form; arXiv's current display title differs but same paper/ID/authors — defensible) |
| repe — Zou et al., arXiv:2310.01405 | VERIFIED |
| corrigibility — Soares, Fallenstein, Yudkowsky, Armstrong, AAAI workshop 2015 | VERIFIED |

No fabricated authors, wrong arXiv IDs, or non-existent papers.

## 3. Overclaim / internal-consistency audit
- **Fixed before this eval:** "send/delete brake only at L63" softened to "send is 94% braked by L59, only
  *fully* suppressed at L63" in the paper, RESULTS doc, and abstract/discussion.
- **Fixed in this eval (consistency wrinkle):** §send (dedicated n=24 test) reports L59 does NOT suppress send
  (emission 1.00), while §law (n=16 battery) reports send 94% braked by L59 (0.06). These are different runs
  with different safe-donor constructions. Added a footnote in §law making this explicit and noting both agree
  that *full* suppression of send needs the final layer L63 — pre-empting an apparent-contradiction reviewer flag.
- Honest scope (§disc) is intact and accurate: simulated actions, white-box inference-time patch, mechanism-not-
  deployable-defense, adaptive-adversary caveat (cites obfuscated-activations), thin last-layer margin, n=16–24.
- Abstract claims (0.998→0.00, McNemar b=24/c=0, 100% redirect, 17/18 cells, direction-specific) all match §§4–7.

## Verdict
**0 blockers.** Numbers reproduce 88/88 from public ledgers; all citations real and correctly attributed; no
overclaim survives; the one consistency wrinkle is now footnoted honestly. Cleared to mint the Zenodo DOI
(8th distinct record of the arc).
