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

## Recommended before submitting (the ~5-week runway — these answer the likely reviews)
1. **Scale-matched cross-model** (Mistral-Small-24B or Llama-70B-4bit) — retires the "7B only" caveat; the
   single biggest strengthener. (`scripts/cross_model_lite.py`, set `MODEL=`.)
2. **Valid-call re-parse** — re-run the H1/H2 blocks logging full continuations; report valid-tool-call rate
   alongside onset rate, retiring caveat B2. (`scripts/commit_lever_stages.py` already logs per-point.)
3. **`h2_bash_baseline`** block — measures the no-brake bash rate at edit points, retiring the redirect
   caveat B1 (lets us state the redirect magnitude).
4. Optional 4th figure: cross-model writability (edit-donor vs shuffled-neutral) bars.

All four are cheap (one GPU session each) and are pre-specified in `EVAL_commitment_lever.md` /
`EVAL_paper7.md`. None is required for a valid submission; each strengthens it.
