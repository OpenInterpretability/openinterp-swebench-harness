# Pre-mint EVAL — located_not_secured.tex (2026-06-19)

Synthesis paper: most numbers are CITED from already-published papers, so the eval verifies each against the
authoritative source (the published PDF on Zenodo), recomputes what is local, checks DOIs, and audits overclaims.

## 1. Recompute (this session's numbers, from local data)
- gated self-correct: **30/32** without CoT (recomputed from `paper/faithfulness/data/gated_intervention.json`) — PASS.
- COLLAPSE 0→1.0 @ budget-4: from this session's adaptive-attack run (`PREREG_adaptive_attack.md` + ledger).

## 2. Cross-check — every cited number vs the SOURCE published PDF (not memory)
Downloaded each cited paper's PDF from Zenodo, `pdftotext`, grepped the number. The motivation: a number from
memory can be wrong (the #1 Tool-Entropy DOI was 20368601 in stale scripts; the real one is 20368807). Result —
**all present in the source PDF**:
| Number (synthesis) | Source paper (DOI) | In PDF? |
|---|---|---|
| AUROC 0.91 ; ΔP −0.001 | #5 Verdict (20532769) | ✓ 0.91, 0.001 |
| 42% ; p=0.031 | #6 Lever Is Late (20534219) | ✓ 42, 0.031 |
| send 0.998→0.00 | #8 Circuit-Breakers (20679287) | ✓ 0.998 |
| allows 21/21 | #10 Felt (20685264) | ✓ 21/21 |
| 0.838 → 0.08 | #11 Form (20724650) | ✓ 0.838, 0.08 |

## 3. Citations / DOIs
8 DOIs cited; all 8 present in the verified live Zenodo account (`GET /api/deposit/depositions`). PASS.

## 4. Adversarial overclaim audit
- **🔴 FIXED — figure semantics.** The original Fig 1 plotted "Felt" with the favorable bar = the EXTERNAL check
  (1.0), inconsistent with the other pairs where the favorable bar = the interpretability metric. A reviewer would
  read it as "interp delivers 1.0 on felt", which is false (the internal monitor is 0; the external check is 1.0).
  Removed Felt from the plot → 3 pairs with identical semantics (Detect, Form, Brake); felt & gated stated as
  qualitative in text/table. Caption updated to "three of the five limits".
- **Gated used honestly:** the synthesis cites only the robust 30/32 self-correct number, NOT the inconclusive
  A-vs-B "corrected" rates (which failed the random-donor sanity). OK.
- **"every control promise collapses"** (fig title) now scopes to the 3 plotted pairs, all of which do collapse. OK.
- **Threat-model honesty:** Limitations state the continuous-embedding attack's deployment realism is contested,
  simulated actions, white-box, single model family, modest n. OK.
- No "universal" / "robust defense" wording; the positive (located lever) flagged as smallest-n.

## Verdict
**0 blockers.** Cited numbers verified against source PDFs (not memory); local number recomputed; DOIs verified;
the one real overclaim (figure semantics) fixed. Mint held for the author's GO.
