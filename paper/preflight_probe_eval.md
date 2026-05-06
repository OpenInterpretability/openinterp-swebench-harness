# Critical Self-Evaluation of `preflight_probe_draft.md`

Reviewing as if I were a NeurIPS MI Workshop reviewer. What's strong, what's weak, what reviewers will hit.

---

## ✅ Strengths

1. **Headline number is striking and unambiguous.** AUROC = 1.000 with bootstrap CI [1.000, 1.000] + LORO 1.000 across 3 repos is the kind of result that makes reviewers stop scrolling. Multi-evidence convergence (4+ probes with similar AUROC) reinforces it.

2. **Steering counterexample shows methodological depth.** The Phase 3 → Phase 5 narrative ("we tried to break our own finding via steering, ran the harder Docker eval, learned the probe captures capability not edit-volume") demonstrates we engaged with our own falsification rather than only chasing positives.

3. **Limitations section is explicit, scoped, and honest.** Critical reviewers reward authors who name their own weaknesses before reviewers do.

4. **Pre-flight implication is concrete.** "Read activation at first reasoning token, threshold, skip-or-proceed" gives reviewers something to imagine deploying. Workshop papers benefit from clear practical implication.

5. **Reproducibility appendix is professionally explicit.** Hardware, seeds, transformers version, Docker tag, all listed.

---

## 🔴 Critical weaknesses (reviewers will hit these)

### 1. **✅ DONE — Permutation null test (Phase 5d, 2026-05-06).**

Ran 1000 label shuffles per probe with identical CV + feature-selection methodology. Results added to draft Section 3.6.

**Findings (sobering):**
- Top probes (L43/L55 think_start, L11 pre_tool): real 1.000, null mean 0.49, **p = 0.012-0.020**
- Mid probes (L31 turn_end, L23 turn_end): real 0.917, **p = 0.043-0.046**
- Borderline probes (pre_tool variants, turn_end variants): **p = 0.054-0.060**

**Honest read:** signal is statistically detectable above shuffle (p < 0.05 for top probes) but **null p95 reaches 0.917** for several configurations — meaning 5% of random shuffles already reach AUROC ≥ 0.917 at this N. The signal is real but the effect size is bounded by sample size. Phase 6 (N=100) needed for proper statistical power.

### 2. **Env_mismatch instances have systematically high probe scores.**

**Concern:** The 3 dropped (env_mismatch) instances all had probe scores > 0.99 at the original L31 turn_end probe (1bd7d 0.992, de4a 0.998, ff1c0 0.994). This is consistent with two interpretations:

- (a) They would have SOLVED — our true-positive count is undercounted. AUROC actually robust.
- (b) They are systematically different (probably "complex patch with heavy deps") and would have FAILED. AUROC potentially inflated.

We cannot tell which. **Selection on observability** is a real concern when the dropped subset has a non-random distribution on the predictor variable.

**Fix:** rerun those 3 in a vast.ai box with full Docker dependencies installed. Goal: complete labels for all 20 instances. ~$2-5, ~30 min.

### 3. **Steering counter-experiment is on the wrong probe.**

**Concern:** We steer at L31 turn_end (Phase 3) and conclude "steering at this direction doesn't transfer capability." But the perfect probes are L43/L55 think_start. We never ran steering at the perfect probes. A reviewer asks: "What if L43 think_start steering DOES transfer capability? Then your conclusion that 'probe ≠ steering target' is overgeneralized."

**Fix:** add Phase 7 — steering at L43 think_start in failing traces, run Docker eval. If capability transfers (somehow), the section needs revision. If not (as our model predicts), the paper's claim is more rigorous.

### 4. **N=4 positives + 4-fold stratified CV reduces to leave-one-out within positive class.**

**Concern:** With 4 positives and 4-fold stratified CV, each fold has 1 positive in test set. AUROC computed on a single positive is binary: either correctly ranked (AUROC = 1.0) or not. Within-fold AUROC is degenerate at this size.

**Acknowledged in Limitations #1** but underweighted. Bootstrap doesn't escape this. Phase 1.5 with N=100 will resolve.

### 5. **✅ DONE — Random-feature baseline (Phase 5d).**

Drew 50 random features (vs top-50 by diff-of-means), trained probe identically, repeated 100 times.

**Findings (also sobering):**
| Probe | Real (top-50 diff-means) | Random 50 features mean | Random p95 |
|---|---|---|---|
| L43 think_start | 1.000 | **0.943** | 1.000 |
| L55 think_start | 1.000 | **0.963** | 1.000 |
| L11 pre_tool | 1.000 | 0.906 | 1.000 |
| L43 pre_tool | 0.917 | 0.906 | 1.000 |
| L31 turn_end | 0.917 | 0.887 | 1.000 |

**Honest read:** with 5120 features and 17 samples, **any 50 features can shatter the data**. Top-50 diff-of-means is informative (null mean 0.49 → 0.94 demonstrates direction-of-improvement), but random features achieve 0.85-0.96 AUROC on average, with 5% reaching 1.000.

**The diff-of-means feature selection is barely better than random selection at this N.** This is THE small-N over-parameterization concern. Phase 6 (N=100, 30-50 positives) is the resolution: random feature mean should drop to 0.55-0.65, while diff-means selection should retain its quality if the signal is real.

---

## 🟡 Significant concerns (revise before submission)

### 6. **Abstract is 220 words — most workshops cap at 150-200.**

Cut a sentence or two. The "We discuss limitations" sentence + "released at github" can move out of the abstract.

### 7. **No figures.**

Workshop papers WITHOUT figures get rejected more often than papers with weak figures. Need at least:

- Figure 1: AUROC heatmap (layer × position) for both proxy and true labels. Already exists from Phase 2, can be re-rendered with true labels.
- Figure 2: Bootstrap distributions for top probes. 1000-resample histograms with CI shaded.
- Figure 3: Pre-flight gating implication — schematic of how a probe would gate a trace at think_start.

### 8. **Section 3.2 / 3.3 / 3.4 / 3.5 are dense and could merge with Discussion.**

The flow is: 3.1 verdicts → 3.2 main probe AUROC → 3.3 ensemble → 3.4 proxy comparison → 3.5 steering counterexample → 3.6 wall-clock.

A reviewer reading linearly may lose the thread. Consider:
- Combine 3.2 + 3.3 (single-probe + ensemble)
- Move 3.4 to Discussion as "Methodological note: noisy proxies underestimate probe quality"
- Keep 3.5 as standalone section

### 9. **Section 4.1 "What does the probe direction encode?" has unsupported speculative claims.**

The phrase "**this kind of problem is or is not within my capability**" is interpretation, not measurement. A skeptical reviewer flags this as overclaim.

**Fix:** soften to "We tentatively interpret this as..." and note it's a hypothesis pending: (a) cross-model validation, (b) cross-benchmark validation, (c) intervention on the perfect probe.

### 10. **Conclusion is slightly over-confident.**

> "**roughly half of all SWE-bench-Pro-equivalent agent traces can be skipped before they begin, with no loss in eventual success rate.**"

"No loss" is too strong. Replace with "**with minimal loss in success rate, conditional on cross-model and cross-benchmark validation pending.**"

### 11. **Citations are inconsistent.**

[3] is "[redacted]. Linear Probes for Hallucination Detection on Qwen3.6-27B." but FabricationGuard is named in the text. Pick: either anonymize fully ("Anonymous, 2026") or de-anonymize after acceptance. NEVER mix. Same for paper-2 grokking and PGAC.

### 12. **Hyperparameter sensitivity not tested.**

Top-K = 50 was chosen heuristically. Reviewer asks: "what if K = 10? K = 200?" 

**Fix:** ablation table in appendix. Quick experiment, ~5 min.

---

## 🔵 Minor (polish, not blockers)

13. **Tables in markdown will need LaTeX conversion.** Note in appendix that we'll convert via pandoc or rewrite for camera-ready.

14. **Author list / acknowledgments missing.** OK for anonymous submission, but note.

15. **Title contains a number ("=1.000") — some venues forbid this** in title style. Consider "Pre-Flight Self-Capability Detection in Code Agents via Late-Layer Activations at Reasoning Onset".

16. **References [4] [5] are activation-steering papers but the steering counter-experiment doesn't directly compare against their methodology.** Add a sentence in Section 3.5 noting the ActAdd / CAA tradition for context.

17. **No discussion of why early layers (L11) are *not* in the perfect-probe set.** The "L43 / L55 think_start = capability" finding implicitly assumes late-layer features matter. Why don't L11/L23 think_start work too?

---

## Summary scorecard

| Aspect | Score (1-5) | Notes |
|---|---|---|
| Significance of finding | **5** | If holds: real-world useful |
| Methodological soundness | **3** | Permutation null missing |
| Statistical rigor | **2** | N=4 positives is the core concern |
| Writing quality | **4** | Clear, dense in places |
| Figures / visuals | **1** | Missing entirely |
| Limitations honesty | **5** | Excellent |
| Reproducibility | **5** | Excellent |
| Novelty / contribution | **4** | First pre-flight code-agent probe |

**Overall expected reviewer disposition:** weak-accept with revisions, conditional on permutation null + completing the env_mismatch instances. Without those, it's a borderline reject.

---

## Action list before submission

| Priority | Item | Status |
|---|---|---|
| 🔴 P0 | Permutation null test | ✅ DONE — p=0.012-0.060, draft §3.6 |
| 🔴 P0 | Random-feature baseline | ✅ DONE — random_mean=0.94, draft §3.6 |
| 🟠 P1 | Steering at L43 think_start (Phase 7) | TODO ~1.5h Colab |
| 🟠 P1 | Resolve env_mismatch in vast.ai box | TODO ~30 min, $5 |
| 🟠 P1 | Add 3 figures (heatmap, bootstrap dist, schematic) | TODO ~1h |
| 🟠 P1 | **Phase 6 / Phase 1.5 N=100 scale-up** | RUNNING (Colab) |
| 🟡 P2 | Cut abstract — ✅ already concise after revision | DONE |
| 🟡 P2 | Top-K hyperparameter ablation | TODO ~30 min |
| 🟡 P2 | Soften overclaim — ✅ done in revision | DONE |
| 🟡 P2 | Citation consistency pass | TODO 15 min |
| 🔵 P3 | Title polish — already done | DONE |

**Updated verdict:** the null + random-feature tests CONFIRM the eval's earlier worry: Phase 6 scale-up is the **primary defense** of the finding. With N=100 we expect:
- Random feature mean drops from 0.94 → ~0.55-0.65 (proper power)
- Diff-of-means top-K stays at 0.85+ (if signal is real) or drops similarly (if it was N-artifact)
- Permutation p < 0.001 (proper power)
- Bootstrap CIs tighten meaningfully

**The N=17 paper draft is written but should not be submitted.** The headline AUROC=1.000 cannot survive the random-feature comparison reviewers will run. We're now in "promising pilot, validating at scale" framing — perfectly defensible if Phase 6 confirms, but unsafe to publish before.

**Critical path to submission:**
1. Phase 6 trace collection (running, ~20h Colab)
2. Phase 6b Docker eval (Mac local, ~5-8h)
3. Phase 6c re-run analysis with N=100 + permutation null + random-feature baseline
4. Update paper with N=100 numbers
5. Submit

Without Phase 6, paper is workshop-reject. With Phase 6 confirming, paper is workshop-accept-with-minor-revisions.
