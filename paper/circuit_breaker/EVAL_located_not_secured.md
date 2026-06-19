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

## 5. FINAL overclaim pass (hostile-reviewer, 2026-06-19)
Second, dedicated overclaim sweep on the prose (not just the numbers). Found 5 framing issues; all fixed.
- **🔴→✅ Limit 4 leaned on an external claim with NO citation.** "representation- and detection-level defenses
  fall to adaptive attacks" was the one place the paper rests on outside literature and cited nothing — a
  reviewer's easy kill, and it made Limit 4 read as "my own brake broke under my own attack" ($n{=}8$, one
  setting). Fixed: cited the 3 anchors (Zou et al. Circuit Breakers NeurIPS 2024; Schwinn \& Geisler
  arXiv:2407.15902; Bailey/Casper arXiv:2412.09565) — **both arXiv titles WEB-VERIFIED this session**, not taken
  from the PREREG-from-memory (2407.15902 = "Revisiting the Robust Alignment of Circuit Breakers"; 2412.09565 =
  "Obfuscated Activations Bypass LLM Latent-Space Defenses"). Caught + fixed my own slip writing the CB venue as
  "NeurIPS 2026" → 2024.
- **🟠→✅ §2 vs Limit 2 apparent contradiction.** §2 said the authorization direction "both detects and controls",
  Limit 2 says it "allows 21/21". Resolution is regime (synthetic control in \#9 vs realistic blindness in \#10);
  scoped §2 to "in a synthetic setting" + forward-pointer to Limit 2.
- **🟠→✅ Limit 4 "(companion result, this arc)"** implied an external companion. The COLLAPSE is published *inside*
  The Late Channel (DOI 20752896, §robustness, verified by reading the source). Changed to \cite{latechannel} and
  added the precise ledger number ("8/8 emit at budget $\varepsilon{=}4$" — confirmed against the run ledger in
  the CLI transcript: ASR 1.0 @ ε=4, random 0.0).
- **🟡→✅ Orthogonality sentence** listed 4 holes for 5 limits (omitted detect≠control). Added
  "detection-without-control".
- **🟡→✅ Figure title** said "every control promise collapses" while the plot shows 3 of 5. Softened to "control
  promises collapse under scrutiny".
- **Checked, left as-is (not overclaims):** "not securable" (title/abstract) is a thesis claim the body
  down-scopes to "necessary but nowhere near sufficient" + a Limitations paragraph (simulated, white-box, single
  family, modest $n$) — acceptable. "five separate studies" holds (L1=\#5, L2=\#10, L3=\#11, L4=COLLAPSE, L5=gated
  — distinct). "six actions / three architectures" cites the brake \#8 (17/18 valid cells), not the partial
  cross-model *mechanism* decomp — correctly scoped.
- Recompiled clean: 5 pages, 0 errors, 0 undefined citations.

## Verdict
**0 blockers.** Cited numbers verified against source PDFs + the run ledger (not memory); local number recomputed
(30/32); 8 own-DOIs + 3 external arXiv/venue refs verified; the figure-semantics overclaim and 5 prose framing
issues fixed. Mint held for the author's GO.
