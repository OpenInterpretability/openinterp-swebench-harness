# BlackboxNLP 2026 submission — beat 10 (ACL port)

**Paper:** *The Criterion Cannot See What It Does Not Measure: Auditing Capability-Guided
Attention Hybridization Against a Named Agent-Commitment Circuit*

**Source:** `paper/circuit_breaker/hybridization_audit.tex` (single-column article, published
Zenodo concept DOI 10.5281/zenodo.21175758). This directory is the **ACL-template, double-blind**
port for the BlackboxNLP OpenReview submission.

## Files
- `hydra_audit_acl.tex` — ACL 2023 template (`\usepackage[review]{acl}`), anonymized author
  (`Anonymous ACL submission`). Compiles to **6 pages** (archival main-track limit) + references.
- `acl.sty`, `acl_natbib.bst` — template assets (from `blackboxnlp/acltmpl/`).
- `figures/` — the 3 figures (`fig1_antialign_L59`, `fig2_twosided`, `fig3_rescue`), regenerable
  from the public HF ledger via `scripts/make_figs_hydra.py`.
- `hydra_audit_acl.pdf` — compiled proof (6 pp, no overfull > 50pt, no undefined refs/cites).

## Build
```
pdflatex hydra_audit_acl && pdflatex hydra_audit_acl   # inline thebibliography, no bibtex needed
```

## Anonymization status (double-blind)
- Author line: `Anonymous ACL submission`. No ORCID / affiliation / handle in the body.
- Self-citations to the arc (Vicentino 2026a/b/c = lever / brake / located) are kept in **third
  person** (`\citet`/`\citep`), which ACL policy permits for published prior work. Verified: **no
  first-person self-reference** ("our prior work", "we previously") anywhere in the body.
- The released tool/ledger names (`openinterp-swebench-harness`, HF dataset) appear in the
  reproducibility paragraph. **Before submission, a human should decide** whether to (a) redact
  these to an anonymized artifact link for review and restore camera-ready, or (b) rely on the
  ACL exception for anonymized-but-linkable artifacts. Currently left visible — flag for Caio.

## Track choice
- **Archival (6 pp):** this file, as-is. Goes to ACL Anthology.
- **Non-archival (2 pp extended abstract):** if preferred, cut Method + Stages 3–4 to a 2-page
  summary pointing at the Zenodo paper. Not yet produced — say the word.
- **Reproducibility Special Track:** strong fit (recompute-every-number eval `eval_hydra_audit.py`,
  59/59 from public ledger). Consider selecting it in the OpenReview submission form.

## Human steps (cannot be automated)
1. Confirm BlackboxNLP 2026 OpenReview is open and the July 17 deadline still holds.
2. Decide archival vs non-archival + whether to select the Reproducibility track.
3. Decide the artifact-anonymization question above.
4. Upload `hydra_audit_acl.pdf` (+ optional supplementary) to OpenReview under the account.

Preprint policy: the CFP allows posting preprints anytime ("no restrictions on when posted"), so
the public Zenodo DOI does not block submission.
