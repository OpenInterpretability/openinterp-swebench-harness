# Pre-mint eval — Paper #6 "The Lever Is Late" (2026-06-03)

Full adversarial eval before any Zenodo mint, per the arc's discipline (the companion-note eval caught 2
fabricated citations; this gate is real). **Verdict: PASS, 0 blockers, after 3 overclaim fixes applied.**

## 1. Citation verification (web-verified, not carried-forward)
| cite | claimed | verified | verdict |
|---|---|---|---|
| Basu et al. 2603.18353 | "Interpretability without actionability…" | REAL — Sanjay Basu et al.; clinical-triage QA, probe 98.2% vs output 45.1%, steering corrects ≤24% | ✅ exact, supports KAG claim; scope "single-turn QA" accurate |
| Bhalla et al. 2411.04430 | "Towards Unifying Interpretability and Control…" | REAL — Usha Bhalla et al. | ✅ exact; softened our phrasing (no crisp predict≠steer "theorem") |
| Wang ASA 2602.04935 | added | REAL — "ASA: Training-Free Representation Engineering for Tool-Calling Agents", Youjin Wang; mid-layer "lazy agent" steering | ✅ added — closest neighbor, mid-vs-late contrast |
| Sun 2605.09252 | added | REAL — "LLM Agents Already Know When to Call Tools", Chung-En Sun; AUROC 0.89–0.96 single-turn | ✅ added — detection counterpart |
| Meng ROME 2202.05262 | added | canonical | ✅ methods cite (activation patching) |
| Vig 2004.12265 | added | canonical | ✅ methods cite (causal mediation) |

**No fabricated citations.** Scooping scan: NOT scooped — no prior work does activation-patching + logit-lens
cross-layer localization of an agent's termination/finish commitment.

## 2. Statistical recompute (independent)
- per-pair 5/12: exact one-sided sign/McNemar (c=0) p = 0.5^5 = **0.03125 → 0.031** ✅ (Fisher cross-check 0.0186 — our McNemar is the *conservative* choice).
- succ-mean 3/12: p = 0.5^3 = **0.125** ✅ (Fisher 0.109).
- locked 0/12: p = 1.0 ✅.

## 3. Ground-truth number check (paper vs RESULTS files)
Fidelity 0.594/0.068/0.005 ✅ · H1 patch L55 +0.134/−0.063, L59 +0.153/−0.067 ✅ · logit-lens L23 −0.02 …
L63 +3.57 ✅ · generation table (all cells) ✅ · union 5/12 0.42 / 3/12 0.25 / 0/12 ✅ · 6 arc DOIs ✅. All figures
match their tables.

## 4. Overclaim audit — 3 fixes applied
1. **"Action channel dominates (70%)"** → corrected. By *lift* the late lever (+42pp from 0%) and behavioral
   (+40pp from 30%) are comparable; reframed the headline as a **layer gap** (verdict-null → late-lever, the
   *controlled* same-experiment contrast). Fig 3 redrawn to show the behavioral baseline + lift honestly.
2. **"Task-matched" confound** → added explicit caveat: per-pair differs from the mean in two ways (single
   coherent state *and* repo-matched-where-available); its edge may be task-conditioning *or* single-vs-averaged
   geometry; we do not separate these.
3. **Bhalla over-attribution** → softened from "features that predict are not those that steer" to "methods
   that read a behavior do not reliably steer it".

## 5. Honest residual limitations (in §6, not blockers)
n=12; coarse-mean n.s. (positive depends on task-matched donor); L63 near-output; greedy decode; 4000-tok
truncation; the three-site ordering is synthesis, only the verdict→late jump is controlled.

**Compile:** 8 pp, 0 overfull, 0 undefined refs/citations, all 6 cites used in body. Ready to mint on Caio's go.
