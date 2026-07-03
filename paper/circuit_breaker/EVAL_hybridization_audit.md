# EVAL — hybridization_audit ("HydraHead B+", paper #16) — pre-mint gate record

**Date:** 2026-07-03 · **Verdict: 0 blockers remaining. Mint-ready pending Caio's OK.**

## Layers run (chronological)

1. **G-1 lit-check (pre-GPU, agent):** conditional GO. Killed the "capability-set blind-spot framing is
   new" clause (arXiv:2510.08525 owns it for reasoning); raised the causal bar (two-sided H3 promoted to
   primary deliverable). 10 must-cites mapped.
2. **Numeric eval (`scripts/eval_hydra_audit.py`):** recomputes EVERY claimed number from the public HF
   ledger — **59/59 PASS**. Caught 3 prose errors pre-report (set sizes 84/82 → real 74/72; late-band
   retention 22 not 12 — code was right, prose was wrong).
3. **Citation verification round 1 (agent, 11 IDs):** 11/11 exist, correct titles/first authors.
   2 soft flags resolved (HydraHead 25%/RULER/patching body-verified by direct full-PDF read; AAPP
   descriptor corrected to abstract's own wording).
4. **Adversarial red-team (agent):** 2 blockers + 5 majors + 7 minors — ALL incorporated:
   - B1 (set-size counts) — already caught by layer 2.
   - B2 — capability side rewritten as UNTESTED (positive control failed; teacher-forced NIAH margin
     RISES under ablation, 10.56 vs 9.39 base; probe has no sensitivity). Headline no longer claims
     "capability preserved".
   - M1 — severity confound killed with 3 GPU controls (keep-2-random at identical severity stays
     collapsed 0.15–0.20 vs keep-writers 0.483; paired 17/0 p=1.5e-5).
   - M2 — P(edit) (fully-ablated, use_cache=False) foregrounded as primary readout; emission disclosed
     as prefill-only ablation.
   - M3 — "worse than random" re-based on 5 seeds (capability below all 5; paired significant 3/5;
     direct paired test 10/1 p=0.012) instead of one draw.
   - M4 — global ranking instability owned (ρ=0.46 stands; top-20 overlap is diagnostic, not gate pass).
   - M5 — pre-registered primary (L59) null disclosed; band result labeled secondary/exploratory.
   - minors m1–m7 (κ gate coarseness, draw composition, tied ranks, expansion wording, multiple
     comparisons, brake-signature softening, "fluent generation" removed) — all applied.
5. **Post-result lit-scan (agent, 7 angles incl. May–Jul 2026):** no direct hit on the package; zero
   published follow-ups to HydraHead (scoop intact). Mechanisms of findings #2/#4/#5 repositioned as
   INSTANCES of known phenomena (Wei 2402.05162 · CGC 2603.16440 · AnchorKV 2606.17872 / AAPP);
   load-bearing novelty = #1 anti-alignment + #3 nominal 2-head rescue + first executed instance of the
   transformation-audit agenda (Oxford AIGI 2026 · Lan 2606.00033). Related work + conclusion rewritten;
   +7 citations.
6. **Citation verification round 2 (the 7 NEW cites, direct WebFetch):** caught **2 fabricated author
   names I introduced** (bibitem "Yao"→real first author Chhabra 2504.04215; "Yang"→Benfeghoul
   2510.05901), 2 shortened titles, **1 WRONG-PAPER citation removed** (2512.06341 "Interpretive
   Efficiency" is information-geometric data-usefulness — the lit-scan agent's description did not match;
   dropped, CGC alone carries the point). AnchorKV author (Ni), Make-MI-Auditable full title + author
   (Lan), Oxford AIGI URL all fixed.
7. **Figures added** (3, generated from the ledger): fig1 anti-alignment at L59 · fig2 two-sided audit ·
   fig3 rescue specificity. Final abstract precision fix (paired p attributed to its seed, not the mean).

## Final state

- `hybridization_audit.pdf` — 8 pages, 3 figures, 2 tables, 21 references, zero undefined citations.
- `eval_hydra_audit.py` — 59/59 PASS against `results/hydra_audit_ie.json` (public HF ledger).
- PREREG → RESULTS → paper chain consistent; all drift disclosed in-paper.
- NOT committed, NOT published. Zenodo mint + any submission are human-gated (Caio).
