# Pre-flight Probe — Complete Eval v3 (post Phase 6 partial)

Updates `preflight_probe_eval_v2_complete.md` with the first Phase 6 evidence at N=12 (Phase 1's 11 + Phase 6's 8 of 99). The picture has shifted materially. This doc is the honest current-state assessment for paper, SDK, and product positioning.

**TL;DR**: the AUROC=1.000 signal at L43 think_start collapses to 0.500 (random) when we add just 4 more qutebrowser negatives. Only L55 think_start partially survives (0.875). This is the over-parameterization regime the v2 random-feature baseline already warned us about — now confirmed empirically. The paper's "pilot finding" framing was correct in spirit but the *direction of the finding may be wrong*: not "probe works at small N", more like "probe trained on small N is not the probe we'll find at scale, may not exist at all in the form we hypothesized."

Action: continue Phase 6 to N=99 to get the truth. Don't submit anything until the verdict is in. Update SDK plan to gate on AUROC ≥ 0.75 (relaxed from ≥ 0.85).

---

## A. New evidence (Phase 6 partial)

### A.1 Trace collection
- 9 of 99 problems run on Colab RTX 6000 Pro Blackwell (GDN fast path, all params on cuda, no offload)
- Wall time per problem: median ~14 min, range 10-24 min
- 8 of 9 generated patches (1 hit max_turns with no diff)
- All 9 are from `qutebrowser/qutebrowser` (stratification iterates groups in random order — 33 per repo expected, this batch happens to be all qutebrowser first)

### A.2 Docker eval (Phase 6b)
- 8 patched instances evaluated locally on Mac (Docker Desktop, /Volumes/SSD Major storage)
- Wall: ~3 min total (small Python repos, fast pytest)

| Verdict | Count | Detail |
|---|---|---|
| **SOLVES** | **0** | none |
| FAILS | 4 | agent_n_pass = none_n_pass in 2/4 (patches are no-ops on test surface) |
| env_mismatch | 4 | golden_n_pass = none_n_pass = 0 (test_patch doesn't activate failing tests) |

### A.3 Preview AUROC (N=12 merged, 4 strict positives, 8 strict negatives)

| Probe | Phase 5d (N=17) | Phase 6 preview (N=12) | Change |
|---|---|---|---|
| L43 think_start | **1.000** | **0.500** | −0.500 (collapse to random) |
| L55 think_start | **1.000** | **0.875** | −0.125 (partially robust) |
| L11 pre_tool | 1.000 | 0.750 | −0.250 |
| L43 pre_tool | 0.917 | 0.750 | −0.167 |
| L31 turn_end | 0.917 | 0.375 | −0.542 (worse than random) |
| L23 turn_end | 0.917 | 0.500 | −0.417 |

CV inside fold (correct, no leakage). Top-50 features by |diff-of-means| chosen on training fold only. 4-fold stratified CV.

Bootstrap CIs at this N are degenerate (200 iter, often (0.625, 1.000) — too small to interpret).

---

## B. Updated statistical truth ranking

### Tier 1 — strong (corroborated)
1. **L55 think_start has the most robust signal.** AUROC stayed at 0.875 with 4 added negatives. Of all probes tested, only this one survived the perturbation cleanly.
2. **The agent's patches frequently no-op the test surface.** 2/4 fails had `agent_n_pass = none_n_pass`. The model writes 870-3520 bytes of diff that never intersects the failing tests. This is qualitatively interesting independent of probe AUROC.
3. **Phase 5d's "perfect" L43 think_start was an N-artifact.** Random-feature baseline already said 0.94 mean → top-50 feature selection was barely better than random at N=17. Phase 6 partial confirmed: tiny perturbation in sample → random AUROC.

### Tier 2 — weak (suggestive but unstable)
1. **Pre_tool position may carry signal.** L11 pre_tool 0.750 and L43 pre_tool 0.750 both survived better than think_start except for L55.
2. **Repo confound is real.** All Phase 6 negatives so far are qutebrowser. Phase 1's 4 positives included other repos. We cannot yet rule out "probe memorizes qutebrowser failure mode" as the finding.

### Tier 3 — overturned (was claimed, now refuted)
1. ~~"Probe AUROC = 1.000 detects pre-flight capability."~~ → currently 0.500-0.875 across probes at N=12.
2. ~~"L43 think_start is the perfect probe."~~ → it's the WORST of the think_start probes after one perturbation. L55 think_start is the candidate now.
3. ~~"Multi-evidence convergence (4+ probes with similar AUROC)."~~ → spread now ranges 0.375 to 0.875. No convergence.

---

## C. Why the AUROC collapsed (statistical mechanics)

Three layered explanations, in increasing severity:

### C.1 Over-parameterization (likely main cause)
With 5120 residual stream dims and ~12 samples, top-50 feature selection lets the LR fit any binary labeling. Phase 5d random-feature baseline showed 0.94 AUROC mean — meaning the gap between "informed" and "random" feature selection is tiny at this N. Adding 4 samples shifted the diff-of-means rankings, picked different features, and the LR fit to a different (weaker) decision boundary.

**Test**: at N=99 with 30-40 positives, random-feature mean should drop to ~0.55-0.60. If diff-means top-50 stays substantially higher, the signal is real. If it tracks random, signal was an artifact.

### C.2 Repo confounding (likely contributing cause)
Phase 1 sampled 20 across 3 repos (qutebrowser, openlibrary, ansible). Phase 6 partial = 8 qutebrowser only. The probe at N=17 may have been picking up "this is qutebrowser-style problem" rather than "model can solve this." LORO across 3 repos at N=17 still showed 1.0 — but only with 1-2 positives per repo, also degenerate.

**Test**: when Phase 6 reaches problems #34+ (openlibrary/ansible), re-run preview. If L55 think_start AUROC stays ≥ 0.80 with multi-repo data, repo confound rejected. If it drops further, repo confound confirmed.

### C.3 The signal genuinely doesn't exist (worst case)
Possible: the model's internal state at first reasoning token does NOT encode capability prediction in a linearly-decodable way. The N=17 pattern was coincidence — 4 random positives happened to align with 4 random feature directions. At larger N, no consistent direction holds.

**Test**: same as C.2. If at N=99 with proper power, no probe ≥ 0.65, accept the null.

These three explanations have different paper implications:
- C.1 → "probe needs more training data, methodology was right" — paper survives at honest-positive
- C.2 → "probe is repo-specific, not generalizable" — paper becomes negative result with caveat
- C.3 → "no pre-flight capability signal exists in this layer/position" — paper becomes pure null result

---

## D. Updated reviewer simulation

Imagining 3 reviewers reading the paper if we submitted today with current evidence:

### Reviewer A — strict statistician
> "Authors report AUROC = 1.000 at N=17, then preview at N=12 (?) shows collapse to 0.500. The honest read is that the original number was N-artifact — Phase 5d random-feature baseline (0.94) already said this. The paper has no submittable finding without the full N=99 results.
> 
> **Verdict**: Reject. Wait for Phase 6 N=99."

### Reviewer B — methodology-focused
> "The Phase 5d eval (permutation null + random-feature baseline) was excellent. Authors caught their own over-parameterization issue. Phase 6 preview (N=12) confirms it. The paper at N=17 is unsubmittable; at N=99 with a clear positive verdict it would be a strong contribution; with a clear negative verdict it would still be publishable as an honest negative result with the methodology contribution.
> 
> **Verdict**: Conditional accept pending N=99 results, regardless of which way the verdict swings. The methodology is the contribution."

### Reviewer C — practical / engineering
> "What matters: does the SDK actually save compute in production? At AUROC 0.875 (current best, L55 think_start) with N=12, threshold-calibrated decisions are still meaningful — but you can't claim 30% compute savings at < 5% loss. You'd be looking at 10-15% savings at 5-10% loss, which is marginal.
> 
> **Verdict**: Workshop accept if N=99 confirms ≥ 0.75 AUROC on best probe. Otherwise reject — no useful application."

**Net**: same as before, *worse* — borderline reject pre-Phase-6, conditional accept pending N=99 in best case, reject in worst case.

---

## E. Updated decision tree (post Phase 6 verdicts at N=99)

| Outcome | L55 think_start AUROC at N=99 | Probability (updated) | Paper | SDK | Frame |
|---|---|---|---|---|---|
| **A: Strong** | ≥ 0.85 with random gap ≥ 0.20 | 25% (was 50%) | NeurIPS workshop accept | ship v0.1 | "Pre-flight probe validated" |
| **B: Moderate** | 0.70-0.85 | 35% (was 25%) | NeurIPS workshop accept with revisions | ship v0.1 with conservative threshold | "Honest signal, narrow scope" |
| **C: Weak** | 0.55-0.70 | 25% (was 25%) | NeurIPS reject; ICLR honest-negative | no SDK ship | "Methodology paper — what doesn't work and why" |
| **D: Null** | ≤ 0.55 | 15% (new) | NeurIPS reject; ICML neg-result track | no SDK ship | "Mech interp negative result on agent capability" |

**Probability shift rationale**: Phase 6 partial showed L43 think_start collapsed and L55 only at 0.875. Adjusting outcome A down (less probable that the strongest probe holds) and outcome D up (now non-zero given the empirical collapse). EV of the project still positive even in C and D given the methodology contribution.

EV calculation:
- Paper-publishable probability (any frame): 25 + 35 + 25 = 85% (was 100%)
- SDK-shippable probability: 25 + 35 = 60% (was 75%)
- Methodology contribution (anti-N=17 trap): 100%

The methodology contribution is the only sure thing now.

---

## F. Updated threat model

| Risk | v2 status | v3 status | Notes |
|---|---|---|---|
| N=17 over-parameterization | identified, untested | **CONFIRMED** by Phase 6 partial | L43 collapsed |
| Repo confounding | identified, untested | partial evidence (qutebrowser-only sample) | retest at problem #34+ |
| Selection bias on env_mismatch | identified, untested | now 4/8 = 50% in Phase 6 — much higher than Phase 1's 3/20 = 15%. WORSE. | rerun env_mismatch with vast.ai container if too high at N=99 |
| Steering doesn't transfer | identified | unchanged | Phase 7 still TODO |
| Cross-model generalization | identified | unchanged | post-MVP |
| Cross-benchmark generalization | identified | unchanged | post-MVP |

The 50% env_mismatch rate in Phase 6 is concerning. If that holds at N=99, we have only ~50 valid instances, with ~10-15 positives — barely better than Phase 1's effective N. Need to investigate why qutebrowser specifically has so many env_mismatch in Phase 6 sample but not Phase 1.

Hypothesis: Phase 6 stratified by `problem_statement` length, which correlates with bug complexity, which correlates with `selected_test_files` not exercising the patch's affected paths. Phase 1 was random sample. Need to verify.

---

## G. Updated submission scorecard

| Aspect | v1 (5d) | v2 (5d done) | v3 (post-Phase-6 partial) | v3 + N=99 OutcomeA target |
|---|---|---|---|---|
| Significance of finding | 5/5 | 5/5 | **3/5** (collapse seen) | 5/5 |
| Methodological soundness | 3/5 | 5/5 | 5/5 | 5/5 |
| Statistical rigor | 2/5 | 4/5 | **3/5** (still N-too-small) | 5/5 |
| Writing quality | 4/5 | 4/5 | 4/5 | 4/5 |
| Figures / visuals | 1/5 | 1/5 | 1/5 | 4/5 |
| Limitations honesty | 5/5 | 5/5 | 5/5 | 5/5 |
| Reproducibility | 5/5 | 5/5 | 5/5 | 5/5 |
| Novelty / contribution | 4/5 | 4/5 | **3/5** if signal weak | 5/5 if signal strong |
| **Total / 40** | 24 | 30 | **27** | 37+ |

v3 dropped 3 points vs v2 due to the partial evidence weakening the headline finding. Still salvageable if N=99 swings to outcome A; recoverable to publishable-negative if outcome C or D, but loses the "real signal" claim.

---

## H. Honest call

### What I would do today
**Don't submit anything.** Wait for Phase 6 N=99 + Phase 6b labels + final 6c analysis. The eval v2 said this; the v3 evidence reinforces it harder.

If I had to bet on the final outcome: **outcome B (moderate, 0.70-0.85)** is the most likely state at N=99. L55 think_start partially survived, ensemble of L11 pre_tool + L43 pre_tool + L55 think_start may push above 0.80, and that's a publishable + shippable position.

### What changes vs v2
- **Lower SDK ship gate**: from AUROC ≥ 0.85 to ≥ 0.75. Below that, no SDK. Above that, ship v0.1 with conservative threshold (skip_below=0.10, escalate_below=0.50, default=balanced+).
- **Pivot probe target**: L55 think_start is now the lead, not L43. Update paper draft to lead with L55. L43 demoted to comparison probe.
- **Add ensemble baseline**: 3-probe ensemble (L11 pre_tool + L43 pre_tool + L55 think_start) — if ensemble AUROC is consistently 0.05+ above any single probe, lead with that.

### What does NOT change
- Phase 6 trace collection continues (Colab RTX 6000, ~22h remaining)
- Phase 6b incremental as instances complete (Mac, ~3-5 min per batch of 8-10)
- Phase 6c re-run after each repo's worth of data lands
- Paper draft stays in "pilot finding" framing — don't oversell, don't undersell

### What the user should be aware of
The original AUROC=1.000 result was, in retrospect, **predictably an artifact of N=17**. The Phase 5d random-feature baseline (0.94) said this. We continued anyway because (a) signal might have been real and small N just amplified it, (b) Phase 6 was already planned. Phase 6 partial is now showing the predicted regression toward random. This is the methodology working — *we caught it ourselves before submitting* — but the project's most likely outcome has shifted from "publishable positive" to "publishable moderate or honest negative."

---

## I. Recommended actions in the next 24 hours

1. **Continue Phase 6 trace collection** (Colab) — autonomous, no intervention needed
2. **Run Phase 6c preview after every batch of ~10 new labeled instances** — track AUROC trajectory across N
3. **At problem #34** (first openlibrary/ansible coming in), run preview specifically with --merge-phase1 to see if multi-repo evidence stabilizes L55 think_start
4. **Investigate env_mismatch 50% rate** — read the 4 env_mismatch instances' `selected_test_files`, see if there's a pattern (e.g., test_files is `tests/test_helpers/*` but test_patch is in `tests/unit/*`)
5. **Add ensemble baseline to phase6c_preview_analysis.py** — average probe scores across L11 pre_tool, L43 pre_tool, L55 think_start
6. **Do NOT touch the paper draft** until N≥40 with positives from at least 2 repos

## J. Pivot options if final number is < 0.70

If at N=99 best probe AUROC < 0.70, reframe the paper as:

**"What pre-flight signals do NOT work for code agents — a negative result with methodology"**

Contributions:
1. We tested 8 probes × 4 positions × 5 layers (40 candidates) and none hit AUROC ≥ 0.70 at N=99
2. We caught our own over-parameterization at N=17 (eval methodology)
3. We show the agent's patches frequently no-op the test surface (separate qualitative finding)
4. We propose what *should* work: training the probe on much larger N (1000+ instances), testing positions in middle-of-thinking rather than think onset, etc.

This is publishable at NeurIPS workshop honest-negative track. Lower citation impact, but real contribution to the field. Plus the harness + dataset are public, so others can build on it.

For the SDK: don't ship. Frame openinterp's contribution as the harness + dataset, not the probe-as-product. agent-probe-guard becomes a research tool, not a deployable product. Update task #114 accordingly if outcome is C/D.

---

## K. The core lesson (for memory)

**Random-feature baseline at small N is the canary.** When `random_top50_AUROC_mean / real_AUROC > 0.85`, the signal is most likely an N-artifact. Phase 5d already gave us 0.94/1.000 = 0.94 ratio — a 6 percentage point gap with random — and that's exactly what collapsed under perturbation. Going forward: any probe paper at N < 50 should report this ratio prominently and treat it as a hard ceiling on the claim.

This is itself a methodology contribution worth a paragraph in the paper, regardless of the final probe verdict.
