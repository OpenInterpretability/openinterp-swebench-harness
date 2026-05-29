# SELF_REVIEW v0.1 — paper2 (Two Honest Nulls)

Internal critique by Claude, simulating a tough peer reviewer in the Tool-Entropy research program's discipline. Calibrated against `feedback_math_validation_review_technique` + `feedback_walk_back_and_rescue_paper_structure` + standard MI workshop expectations.

## CRITICAL (must fix in v0.2)

### C1. McNemar p value is wrong

Paper writes: "McNemar (3 vs 2): z ≈ 0.45, p ≈ 0.65"

The z-approximation is not valid at n_discordant=5. Exact McNemar (binomial):
- Under H_0, discordant pair direction is 50/50
- P(X ≥ 3 | n=5, p=0.5) = (C(5,3) + C(5,4) + C(5,5)) / 32 = 16/32 = 0.5
- Two-tailed exact p = 2 × min(P(X≥3), P(X≤2)) = 1.0 (saturates)

**Fix**: replace "z ≈ 0.45, p ≈ 0.65" with "exact McNemar two-tailed p = 1.00".

### C2. Title overclaims with "Refute"

Title currently: "...Pre-registered Causal Tests **Refute** Simple Mid-Layer-to-Edge Rescue"

"Refute" implies the hypothesis is killed. Our actual claim: hypothesis not supported by this test at this magnitude. Softer phrasing preserves accuracy.

**Fix**: change subtitle to "Pre-registered Causal Tests **Refine** the Mid-Layer-to-Edge Hypothesis".

### C3. Confidence intervals missing for all rates

Standard practice in workshop papers: report 95% CI alongside point estimates. We give rates without CIs throughout. Wilson 95% CIs (more reliable than normal approx at our n):
- Exp D WANDERING (4/19): 21.1% [8.51%, 43.34%]
- Exp D SUCCESS (6/39): 15.4% [7.18%, 29.69%]
- Exp B Baseline (7/20): 35.0% [17.74%, 57.42%]
- Exp B Primary (6/20): 30.0% [14.55%, 51.92%]

**Fix**: add Wilson CIs in Tables 1 and 2.

## IMPORTANT (should fix)

### I1. No figure

Workshop papers need at least one figure. Table 3 (per-instance paired comparison) is data-dense but visually inert. A 2×2 confusion matrix (baseline outcome × primary outcome) makes the paired structure immediate.

**Fix**: add Figure 1 — Exp B paired confusion matrix with 12/3/2/3 counts visible.

### I2. Stable-WANDERING subset analysis buried in §10

The strongest positive signal in the paper — restricting to the 13/20 trajectories that consistently exhaust across the original run and an independent re-run (same RTX 6000) gives 4/13 = 31% hook vs 0/13 = 0% baseline (highly significant) — is mentioned only in §10 Future Work as "post-hoc, requires pre-registration". This buries a finding that could substantially change the reader's impression.

**Fix**: surface this in §6 (Run-Stability Discovery) with explicit numbers and the "post-hoc" caveat, so the reader sees both (a) the null at full N=20 and (b) the positive trend at the stable subset, with proper hedging.

### I3. McNemar discordant pair count: 3 vs 2 is wrong direction sign discussion

Paper says "Primary-only flips (hook rescued): 2 instances" — but checking the data:
- Baseline finish_tool, Primary max_turns: 3 instances (9b71c1, bb152d23, 3db08adb) — baseline got the flip
- Baseline max_turns, Primary finish_tool: 2 instances (7f6b722, cf06f4e3) — primary got the flip

Phrasing is correct, but reader could read "hook rescued 2" as net-positive. The 3 vs 2 asymmetry means hook NET blocked 1 more natural finish than it rescued. The "rescued" word is mildly misleading without context.

**Fix**: minor phrasing tightening in §5.2 ("...rescued 2; *the slight directional asymmetry favours hook blocking over rescuing*").

### I4. Citations sparse

Only 5 bibliography entries. Workshop reviewers expect 10-20 contextual refs for a paper that engages multiple sub-literatures.

**Fix**: add 4-5 references:
- Räuker et al. 2023 (probe limitations / linear-probe critiques)
- Anthropic 2024 sleeper-agent probes (relevant probe-detection prior)
- Apollo deception probes (cited in companion paper)
- METR MALT (cross-task agent failure)
- Tool-entropy reward-shaping prior (cited in companion paper)

### I5. Abstract lacks the Run-Stability finding's punchline number

Abstract mentions "single-run classification captures a temperature-stochastic phenotype that varies run-to-run on identical hardware (RTX 6000; no H100)" without the punchline number (35% flip in an independent no-hook re-run). The methodological finding is one of three contributions; the number should appear.

**Fix**: add "...an independent no-hook re-run shows 7/20 = 35% finish_tool flip rate (same RTX 6000 Pro Blackwell), indicating run-to-run stochasticity under temperature=1.0, not hardware determinism..." to abstract.

## NICE-TO-HAVE (optional in v0.2)

### N1. §2 Related Work could expand

Currently 4 paragraphs. Could expand with WANDERING-adjacent work (sandbagging detection, deception probes more broadly).

### N2. §7 third interpretation is speculative

"Attention-pattern lock-in" is named but not grounded. Either drop or add a sentence justifying why attention pattern is a plausible alternative mechanism.

### N3. §10 Future Work prioritization

List of 5 future experiments. Reader could benefit from a one-line "if I could run only one next, I would run X" priority signal.

### N4. Tables could use consistent number formatting

Some tables use percentage signs, some don't. Some use ratios in fraction form, some decimal.

### N5. Add `\paragraph{}` headers within long sections

Sections §4-5 are dense paragraphs. Sub-paragraph headers would improve readability.

## NON-ISSUES (looked at, decided OK as-is)

- N=20 sample size disclosed in §9 — adequate honesty
- mode=replace → mode=add pivot disclosed in both §3 and Appendix D — adequate transparency
- Walk-back of N=10 preliminary disclosed in §4.4 — adequate per [[feedback-walk-back-and-rescue-paper-structure]]
- Title length is fine for workshop (subtitle is informative not just descriptive)
- No figures-of-results in main text is OK if Table 2 + Figure 1 (to add) cover the visual story

## FIX PLAN

1. Fix C1 (McNemar p value): single Edit in §5.2
2. Fix C2 (title): single Edit in \title{}
3. Fix C3 (Wilson CIs): expand Tables 1 and 2
4. Fix I1 (figure): generate matplotlib confusion matrix, save to figures/, add to §5
5. Fix I2 (stable subset): add subsection to §6 with explicit numbers
6. Fix I3 (phrasing): minor Edit in §5.2
7. Fix I4 (citations): expand references.bib + cite in §2 and §8
8. Fix I5 (abstract): minor Edit in abstract

Total time estimate: ~30-45 min.
