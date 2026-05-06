# Complete Evaluation — Preflight Probe Paper Draft v2

This is a comprehensive evaluation of the revised draft `preflight_probe_draft.md` (commit 869f3db, post-null-test integration). Covers: technical rigor, submission readiness, reviewer simulation, decision tree, and honest threat model.

---

## Section A: What changed between v1 and v2

| Element | v1 (initial draft) | v2 (post-null-test) |
|---|---|---|
| Title | "Predict SWE-bench Pro Success with **AUROC=1.000**" | "A Pre-Flight Capability Signal — A **Pilot Finding** at N=17" |
| Headline | "AUROC=1.000 with bootstrap CI [1.0, 1.0]" | "AUROC=1.000, permutation p=0.012-0.020, random baseline 0.94" |
| Conclusion | "agent traces can be skipped...with no loss" | "**we do not claim a deployable detector at this stage**" |
| Reviewer disposition | Likely reject (overclaim, no null) | Pilot-grade, reviewer waits for Phase 6 |
| Honesty score | 3/5 (omits null + random baseline) | 5/5 (explicitly states all caveats) |

The revision reduced the paper's *headline strength* but increased its *defensibility*. This is the right tradeoff if we want the paper to survive peer review when Phase 6 lands.

---

## Section B: Statistical rigor — what the data ACTUALLY supports

### B.1 Three findings ranked by strength

**🟢 Finding 1 (medium-strong): Diff-of-means feature selection IS informative.**
- Null shuffled-label mean AUROC: **0.49**
- Random feature selection mean AUROC: **0.94**
- Selection direction matters: 0.94 vs 0.49 is meaningful

**🟡 Finding 2 (weak): The probes exceed permutation null.**
- Top probes p = 0.012-0.020 — significant but small effect
- Mid probes p = 0.043-0.060 — borderline
- Several would not survive Bonferroni correction (8 probes × 0.05 = 0.006 threshold; only 1 survives)

**🔴 Finding 3 (very weak): The 1.000 ceiling is real.**
- Random feature p95 = 1.000 — 5% of random selections also reach perfect
- Cannot distinguish "true 1.000" from "lucky 1.000" at this N
- Bootstrap CI [1.0, 1.0] is statistically meaningless because resamples preserve the same N=4 positives

### B.2 What's NOT supported

- ❌ "Pre-flight gate" claim — needs deployment validation
- ❌ "Capability axis" interpretation — needs cross-model + intervention on perfect probe
- ❌ Cross-benchmark generalization — only SWE-bench Pro tested
- ❌ Publication-grade significance — Bonferroni-corrected p < 0.006 only achievable for 1 probe

### B.3 Honest read

The paper has **a methodological contribution** (Docker-validated probe pipeline) and **a directional finding** (signal exists at p < 0.05). It does NOT have a publication-strength quantitative result. Phase 6 is the test that produces such a result.

---

## Section C: Reviewer simulation

### C.1 Reviewer A (statistician — friendly but rigorous)

> *"The methodology is sound. The Docker eval pipeline is a clear contribution. However, the headline AUROC=1.000 is meaningless at N=17, 4 positives — the random-feature baseline of 0.94 makes this clear. The authors honestly report this. My recommendation: ask the authors to scale to N≥50 before resubmission."*
>
> Score: **Weak Reject**, with clear path to weak accept

### C.2 Reviewer B (mech interp practitioner — sympathetic)

> *"Pre-flight probing is a great research direction. The fact that p < 0.05 holds for L43/L55 think_start across 1000 shuffles is encouraging. The steering counter-experiment is methodologically nice. However, I want to see whether the L43/L55 think_start signal survives at proper N before recommending acceptance."*
>
> Score: **Weak Accept** if Phase 6 confirms; **Reject** if author resubmits without N=100

### C.3 Reviewer C (skeptic — pattern-matches to "small N over-claim")

> *"Random feature baseline mean = 0.94 indicates the probes are essentially picking up structure that ANY 50 features in a 5120-d space can express at this N. The authors' own analysis acknowledges this. There's no demonstrable signal beyond what over-parameterization predicts. I'm unconvinced this paper has a finding."*
>
> Score: **Strong Reject**

### C.4 Most likely outcome at NeurIPS MI Workshop

Two of three reviewers like the methodology, one rejects on small-N. **Net: borderline reject.** The honest framing helps (authors aren't making the mistake reviewers are looking for) but the data doesn't support a strong claim yet.

---

## Section D: What's still missing (P1 list)

### D.1 Methodological gaps

1. **Steering on the perfect probe.** Phase 3 steered L31 turn_end (proxy AUROC 0.79). The "perfect" probes are L43/L55 think_start. Did NOT test steering there. Reviewer asks: "what if those direction transfer capability when steered?" — we'd argue they wouldn't (think_start fires before any output) but we haven't measured.

2. **Env_mismatch resolution.** 3 instances dropped (1bd7d, de4a1, ff1c0) — all had probe scores > 0.99 on the proxy task. We don't know whether they would have been SOLVES or FAILS. Selection bias in our 17. **Vast.ai box can resolve in 30 min, $5.**

3. **Cross-model.** Same probe on Llama 3.x or DeepSeek? Untested. Could be Qwen-specific direction.

4. **Cross-benchmark.** Pre-flight probe on AIME, HumanEval, BigCodeBench? Untested. Could be SWE-bench-specific.

5. **Hyperparameter sensitivity.** K=50 features chosen heuristically. K∈{10, 20, 100, 200}? Untested. Probably stable but worth confirming.

### D.2 Presentation gaps

6. **No figures.** Workshop papers benefit from 2-3 figures: AUROC heatmap, permutation null distribution, schematic of pre-flight gating.

7. **Citations.** Mix of named and "[redacted]" — needs consistency pass before submission.

8. **Anonymization sweep.** GitHub URL embedded in artifact section — needs anonymization or post-acceptance reveal.

### D.3 Strategic gaps

9. **No comparison to prior pre-flight literature.** Are there other papers attempting agent-trace prediction at task time? Need a related-work pass.

10. **No deployment cost analysis.** Workshop papers benefit from "X compute saved" estimates. Conditional on validation, what's the engineering payoff?

---

## Section E: Decision tree — what to do post-Phase 6

We will get one of three outcomes from N=100 trace collection + Docker eval + analysis:

### E.1 Outcome A: Phase 6 confirms (random_mean drops to 0.55-0.65, real probe stays 0.85+)

- **Probability**: ~50% (best-case if signal is real)
- **Action**: revise paper with N=100 numbers. AUROC + bootstrap will be tight. Permutation p < 0.001 expected.
- **Submission**: NeurIPS MI Workshop 2026, expected weak-accept → accept
- **Side benefit**: deployable probe, ship `agent-probe-guard` package
- **Timeline**: paper submit + ship product within 4-5 weeks of Phase 6 start

### E.2 Outcome B: Phase 6 partial (random_mean ~0.65-0.75, real probe ~0.70-0.80)

- **Probability**: ~25%
- **Action**: paper still publishable but reframed: "moderate signal, useful as ensemble component, not standalone"
- **Submission**: NeurIPS MI Workshop or ICLR Tiny Papers
- **No SDK ship** — probe too weak for production
- **Timeline**: paper-only, ~5 weeks

### E.3 Outcome C: Phase 6 falls apart (random_mean ≈ real_mean ≈ 0.55-0.65)

- **Probability**: ~25%
- **Action**: **honest negative paper**: "pre-flight probing was a small-N artifact in initial pilot; at N=100 the signal disappears"
- **Submission**: still publishable as cautionary methodology paper
- **Pivot**: try different layer/position, different label, different model
- **Timeline**: ~3 weeks

### E.4 Probability-weighted expected value

| Outcome | P | Paper outcome | Ship outcome |
|---|---|---|---|
| A (signal holds) | 0.50 | accept @ MI Workshop | SDK ship |
| B (moderate) | 0.25 | accept @ workshop or Tiny | no SDK |
| C (collapses) | 0.25 | honest-negative paper | no SDK |

EV: ~75% chance of any paper, ~50% chance of usable SDK. Phase 6 cost: ~$0 (already paid Colab + local). Worth it.

---

## Section F: Threat model — what could go wrong

### F.1 Phase 6 trace collection fails

- **Risk:** Colab session disconnects; checkpoint resume works but adds days
- **Likelihood:** medium
- **Mitigation:** notebook has per-problem Drive checkpoint. Script can resume from any point.

### F.2 Phase 6b Docker eval fails

- **Risk:** Some Python environments don't install on local Mac (Phase 5a had 3/11 env_mismatch)
- **Likelihood:** medium-high (~30% per repo)
- **Mitigation:** vast.ai box for env_mismatch instances. ~$10 for 100 evals.

### F.3 Phase 6c shows no signal

- **Risk:** Random_mean ≈ real_mean. Phase 1 was N-artifact.
- **Likelihood:** ~25% (per Outcome C above)
- **Mitigation:** honest-negative paper still publishable. Pivot to different layer/position.

### F.4 Reviewer hostility (probability of rejection)

- **Even with Phase 6 success:** still small N=100, single model, single benchmark. Could get reviewer C-style rejection.
- **Mitigation:** add cross-model probe (Llama or DeepSeek) as P1 work. Builds robustness story.

### F.5 Scoop risk

- **Risk:** Someone else publishes pre-flight probe finding first
- **Likelihood:** low (this is unusual angle; agent reasoning is mostly behaviorally studied)
- **Mitigation:** open-source release builds priority claim

### F.6 Ethical / safety concerns

- **Risk:** Pre-flight gating mis-routes hard problems away from larger models, reducing user experience
- **Mitigation:** paper should include false-negative analysis (what % of TRUE solves get classified as fails?)
- Status in current draft: not addressed. **TODO.**

---

## Section G: Submission readiness scorecard

| Aspect | v1 score | v2 score | Phase 6 will move to |
|---|---|---|---|
| Title clarity | 3/5 | 5/5 | 5/5 |
| Abstract honesty | 2/5 | 5/5 | 5/5 |
| Statistical rigor | 2/5 | 4/5 | 5/5 |
| Statistical strength | 2/5 | 2/5 | 4-5/5 |
| Limitations honesty | 5/5 | 5/5 | 5/5 |
| Figures | 1/5 | 1/5 | 4/5 (need to add) |
| Reproducibility | 5/5 | 5/5 | 5/5 |
| Novelty | 4/5 | 3/5 | 4/5 |
| **OVERALL** | 24/40 | 30/40 | 37/40 (target) |

**v2 is honest enough to NOT damage credibility if released, but is at submission floor (30/40) — borderline workshop reject.**

**Phase 6 confirms → 37/40, comfortable workshop accept.**

---

## Section H: What I'd do as the author

### H.1 Submit v2 to a workshop now?

**No.** Three reasons:
1. Random feature baseline 0.94 is brutal in review. Every reviewer who runs the test sees this.
2. We're 4-5 weeks from Phase 6 results. Better to wait.
3. Submitting and getting rejected wastes review-budget capital with the venue.

### H.2 Submit v2 as preprint to arXiv now?

**Maybe.** Preprints don't have peer review. Could establish priority on the methodology + pilot finding. But also:
- Reviewers later see "they wrote a preprint claiming X, then walked it back at review" → bad look
- Better: hold preprint until Phase 6 lands, post BOTH together (pilot + scaled validation)

### H.3 Write up methodology paper now?

**Yes.** A "Docker-validated probe pipeline for code agent traces" methodology paper is publishable independently of the specific finding strength. Reframe v2 as:

> "We present a methodology for evaluating pre-flight probes on code-generating agents using full Docker test execution. We illustrate its use with a pilot study (N=17) finding suggestive but underpowered evidence of a self-capability signal at L43/L55 think_start. Methodology is general; downstream studies (such as ours at N=100, in progress) will reach proper power."

This reframes the paper from "we found X" to "we built tools to find X-class results" — much safer.

### H.4 Ship anything now?

**Probably not.** N=17 with 0.94 random baseline is too thin to defend in public. Wait for Phase 6.

**Exception**: open-source the methodology (harness, Docker eval scripts, probe code). Without claims about specific numbers. This is the OpenInterpretability hygiene move.

---

## Section I: Final recommendation

### Critical path (4-5 weeks)

```
Week 1 (NOW)        Phase 6 trace collection running on Colab (4 batches × 25)
                    ✅ Phase 6b Docker eval script ready (Mac local)
                    ✅ Permutation null + random baseline integrated into draft

Week 2              Phase 6 finishes
                    Phase 6b runs on Mac (~5-8h)
                    Phase 6c re-runs probe analysis with N=100

Week 3              Update paper with N=100 numbers
                    Run permutation null + random baseline on N=100
                    Add figures (heatmap, null dist, schematic)
                    Cross-model probe on Llama/DeepSeek (1-2 instances each)

Week 4              Submit to NeurIPS MI Workshop 2026
                    Open-source release (harness, probes, replication scripts)
                    Twitter / HN post

Week 5              (if Phase 6 confirms) Build agent-probe-guard SDK
                    Landing page openinterp.org/products/agent-probe-guard
                    Public ship + outreach
```

### Risk mitigation in parallel

- Run Phase 6b/6c immediately after Phase 6 — same week
- vast.ai box for env_mismatch resolution ~$10
- Cross-model spot check on Llama 3.x think_start activation
- Steering on perfect probe (1.5h Colab) — addresses Reviewer A concern

### Bottom line

**The current draft (v2) is honest and not embarrassing — but it's not the paper to submit.** It's the paper to **hold** until Phase 6 lands, then revise with proper numbers.

The strategic move: **patience + Phase 6 = workshop accept + SDK ship + Anthropic-application-strengthening artifact.** Premature submission burns the chance.
