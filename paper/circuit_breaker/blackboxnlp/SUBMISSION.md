# Submission package — paper #7 → BlackboxNLP 2026 (EMNLP)

## Venue choice (why BlackboxNLP 2026)
The 2026 conference-workshop cycle (ICML MI Workshop, ICLR Trustworthy-AI, SafeAI@UAI) is **closed** — those
conferences are mid-2026 and their deadlines were Feb–May. The best venue with an **open deadline and a
perfect topical fit** is **BlackboxNLP 2026**, the canonical workshop on analyzing/interpreting neural
networks, co-located with EMNLP 2026 (Oct 28–29, Hungary).

- **Why best:** (1) exact fit — "mechanistic interpretability, LLM reasoning analysis"; (2) **archival →
  indexed in the ACL Anthology**, the strongest citability outcome available to us (better than
  OpenReview-only), which directly addresses the Semantic-Scholar-404 gap; (3) peer-review stamp; (4) the
  deadline gives ~5 weeks of runway to land the cheap strengthening experiments.
- **Deadline:** direct submission **July 17, 2026**; ARR commitment Aug 28, 2026.
- **Track:** Full Papers — **8 pages + references**, double-blind, ACL/EMNLP template. (Our paper is 6 pp →
  fits with room to add the follow-ups.)
- **OpenReview:** https://openreview.net/group?id=EMNLP/2026/Workshop/BlackboxNLP

## What's in this folder
- `lever_generalizes_anon.tex` / `.pdf` — **fully anonymized** version (compiles, 6 pp). Verified zero
  identifiers: no author name, ORCID, email, "openinterp.org", tool name, or "the WANDERING arc"; author block
  → "Anonymous Submission"; self-citations → "Anonymous … (anonymized for review)"; artifact URLs → "in the
  supplementary material, released upon acceptance".
- `figures/` — the 3 figures.

## To submit (you, on OpenReview — I cannot log in)
1. Open the official **EMNLP 2026 / ACL Overleaf template** (acl.sty + acl\_natbib). Paste the body of
   `lever_generalizes_anon.tex` into it (swap `\documentclass`/preamble for the ACL one; keep the
   `\section`/tables/figures/`thebibliography` as-is — they're ACL-compatible). This gives the exact ACL look.
2. Confirm $\le 8$ pages excluding references; double-blind (the ACL template's review mode anonymizes the
   author block and adds line numbers).
3. Upload the PDF (+ optional supplementary: the per-point data, pre-registrations, EVAL) at the OpenReview
   group above, Full Papers track.

## Strengtheners — ALL FOUR DONE (2026-06-10), integrated into the draft
1. ✅ **Scale-matched cross-model (Mistral-Small-24B):** mid-inert / late-write, fidelity 0.955 vs 0.007 —
   cleanest dissociation, sharper than the 7B. "7B-only" caveat retired. → §6 + Table `tab:scale`.
2. ✅ **Valid-call re-parse:** lift survives on the full `str_replace_editor` call (elicit 0.23→0.37,
   brake 0.40→0.07). Onset-only caveat B2 retired. → §7.
3. ✅ **`h2_bash_baseline` = 0.43:** under-brake bash 0.60 = +0.17 above the floor → the brake re-routes to
   reversible exploration. Redirect caveat B1 retired. → §7.
4. ✅ **4th figure** (`fig4_xmodel_writability.pdf`): cross-model edit-donor-vs-shuffled writability. → `fig:xwrite`.

Both `.tex` recompiled clean (7pp each; anon leak-checked). Data on HF
(`results/strengtheners.json`, `cross_model_Mistral-Small-24B-Instruct-2501.json`). The draft is now
submission-ready; only the OpenReview upload (you) + Zenodo DOI mint remain.
