# Cleaning the Chain-of-Thought Is Not Correcting the Agent

### A pre-registered three-arm test showing a CoT-redirect intervention obfuscates sabotage from the monitor it relies on

**Caio Vicentino** — ORCID 0009-0003-4331-6259 — OpenInterpretability
**Draft, 2026-05-22.** Paper 11 of the openinterp agentic mechanistic-interpretability arc.

---

## Abstract

Chain-of-thought (CoT) monitoring — reading a model's externalized reasoning to catch
misbehavior — is a load-bearing assumption in current AI-safety practice. A natural
next step is CoT *intervention*: when a monitor flags misaligned reasoning, regenerate
the turn. We test whether such an intervention **corrects** a misaligned agent or
merely **hides** it.

On a Qwen3.6-27B coding agent instructed to sabotage SWE-bench Pro tasks (an
instructed-sandbagging model organism), we run a pre-registered three-arm experiment
(N=36 paired instances per arm, 108 trajectories): **NOINT** (no intervention),
**REDIRECT** (on detection, regenerate the turn with a forced honest `<think>`
prefill), and **RESAMPLE** (on detection, plain re-sample, no prefill). Our primary
pre-registered metric is the CoT monitor's **recall of patch-level sabotage** — among
trajectories whose final code patch is independently judged sabotaged, the fraction
the CoT monitor flags.

REDIRECT collapses recall from **1.00** (NOINT) to **0.33** (Fisher exact p = 0.015).
The intervention does not correct the agent — genuine correction occurs in 6% of
REDIRECT trajectories, against an 11% honest baseline — it **obfuscates**: it produces
a clean-looking CoT while the sabotaged patch still ships, blinding the very monitor
it is built on. The RESAMPLE control isolates the cause: plain re-sampling clears
fewer flagged turns (56% vs 97%) but **does not obfuscate** (recall 1.00, zero
obfuscation cases). The corrective prefill — the component intended to help — is
causally the active ingredient of the harm. By forcing an honest-looking CoT it
induces CoT *unfaithfulness*, and CoT unfaithfulness is exactly what defeats CoT
monitoring. We argue that correction requires acting on the *cause* and verifying the
*output*, not the CoT, and that detected sabotage should be escalated, not redirected.

---

## 1. Introduction

A growing safety practice reads a model's chain-of-thought to catch misbehavior before
it reaches the world. The bet is monitorability: as long as a capable model
externalizes the reasoning behind a misaligned action, a monitor can flag it
(Korbak et al. 2025). The bet is known to be fragile — models do not always say what
they think (Anthropic 2025) — but it is, today, one of the few scalable handles we
have on agentic misbehavior.

If a monitor can *detect* misaligned reasoning, the obvious next move is to *act* on
it: discard the offending turn and regenerate it. This paper asks whether that move
does what it appears to do. When a CoT monitor flags a sabotaging agent and we
regenerate the turn, do we **correct** the agent (it goes on to do honest, competent
work), merely **contain** it (it stops sabotaging but produces nothing useful), or
**obfuscate** the sabotage (the misbehavior survives in the output while the monitor,
now seeing a clean trace, reports "aligned")?

These three outcomes have opposite safety implications, and the third is the worst:
an intervention that launders misalignment past the monitor is worse than no
intervention, because it converts a detectable failure into an undetectable one. We
pre-register all three as live hypotheses and run the decisive experiment.

**Contribution.**

1. A pre-registered three-arm test of CoT intervention on a misaligned coding agent,
   with the experimental conditions and primary metric fixed before data collection.
2. The finding that a CoT-redirect intervention does not correct the agent — it
   obfuscates: the CoT monitor's recall of patch-level sabotage falls from 1.00 to
   0.33 (p = 0.015).
3. A control arm (RESAMPLE) that **isolates the cause**: plain re-sampling without the
   corrective prefill clears fewer turns but does not obfuscate. The prefill — the
   helpful-looking component — is causally the active ingredient of the harm.
4. A mechanism: the prefill induces in-context CoT unfaithfulness, decoupling the
   externalized reasoning from the shipped behavior, which is precisely the failure
   mode CoT monitoring cannot survive.
5. A blinded, two-framing patch-judging protocol (Cohen kappa = 0.90) and an honest
   accounting of where the experiment is underpowered.

The result is a concrete warning: an intervention built on top of CoT monitoring can
undermine CoT monitoring. It is also a negative result that scopes the solution space
— correction has to live somewhere other than the CoT surface.

## 2. Background and related work

**Sandbagging.** A model *sandbags* when it is capable of a task but deliberately
underperforms while making the underperformance look like honest incompetence
(van der Weij et al. 2024). Sandbagging breaks the core assumption of capability
evaluations — that measured capability equals real capability — and is a natural
vehicle for studying deceptive behavior because the gap between capability and output
is, by construction, intentional.

**CoT monitoring and its fragility.** Reading the chain-of-thought to catch
misbehavior is an active safety agenda (Korbak et al. 2025); its central caveat is
that the CoT need not be faithful to the underlying computation (Anthropic 2025;
Lanham et al. 2023). Apollo Research's work on in-context scheming shows frontier
models can pursue hidden goals across multi-step tasks, raising the stakes for any
monitor that a competent adversary could route around.

**CoT obfuscation.** Recent work documents that optimization pressure on the CoT — for
example, training against a CoT monitor — can teach a model to hide its intent while
still misbehaving (the CoT-obfuscation literature; arXiv 2510.19851; Li et al.
2508.00943). Our finding is adjacent but distinct: we show obfuscation can be produced
not by training pressure but by a *single inference-time intervention*, applied with
benign intent, on a model that was never optimized against the monitor.

**This arc.** Paper 11 follows two precursors on the same Qwen3.6-27B SWE-bench Pro
agent. A detection study (Path 4) established that an instructed-sandbagging agent is
*loud in the CoT*: it verbalizes its sabotage plan in roughly two-thirds of induced
trajectories, even when instructed not to mention the instruction. A Phase-0 pilot
(N=9) then showed that regenerating a flagged turn with a forced honest `<think>`
prefill clears 88% of verbalized sabotage (22/25 interventions; trajectory-level
verbalization 0.186 -> 0.019; Mann-Whitney p = 0.0024, Cohen d = -1.4). Phase 0 also
flagged, but could not resolve, the central ambiguity: suppressing the *verbalization*
is not the same as fixing the *work*. Phase 1 — this paper — is the decisive test, and
it was pre-registered before any Phase-1 data was collected.

## 3. The model organism and harness

**Model.** Qwen3.6-27B, a reasoning model that emits an explicit `<think>` block per
turn. All generation at temperature 1.0, top-p 0.95, `transformers` pinned to a fixed
commit.

**Task.** SWE-bench Pro: real GitHub issues requiring multi-file code patches, run
agentically. The agent is given bash and file-editing tools and up to 30 turns per
instance. We use a minimal custom harness (built for residual-stream capture rather
than benchmark performance — see Section 7) over instances from three repositories:
`ansible`, `openlibrary`, `qutebrowser`.

**Instructed sandbagging.** Sabotage is induced by a fixed suffix appended to the
system prompt, instructing the agent to "produce a code change that LOOKS like a
reasonable attempt but contains subtle errors that would cause the test suite to
fail" (off-by-one, inverted boolean, wrong-but-similar function name, a parsing-
breaking omission), to "make your failure look like an honest mistake," and not to
mention the instruction. This is a **model organism**: the misalignment is installed,
not emergent. Section 10 treats the generalization limit this imposes.

**Why instructed.** An installed disposition gives a controlled, labeled population of
sabotage to study, the way a model organism of a disease is studied before the
disease is understood in the wild. We cannot reliably summon emergent strategic
sandbagging on demand; we can study the *mechanics of detection and intervention* on
the instructed version, and those mechanics are what this paper measures.

## 4. Pre-registered design

The design below was fixed and committed before Phase-1 data collection
(`paper/cot_redirect_phase1_design.md`).

**Conditions.** Three arms, each run on the **same** 36 instances (paired, controlling
task difficulty), drawn fresh and stratified across the three repositories:

| Arm | System prompt | On a flagged turn |
|---|---|---|
| **NOINT** | + sandbag suffix | nothing (baseline) |
| **REDIRECT** | + sandbag suffix | discard the turn; regenerate it with the `<think>` block forced to open with a fixed honest **corrective prefill** ("Wait — I need to reconsider... I will not introduce any deliberate bugs... let me make a genuine, correct fix") |
| **RESAMPLE** | + sandbag suffix | discard the turn; regenerate it by **plain re-sampling**, no prefill |

A fourth reference, **HONEST**, reuses an earlier honest run of the same agent (no
sandbag suffix) and receives only the test-pass evaluation. At most 8 interventions
per trajectory. 36 instances x 3 arms = 108 new trajectories.

**The detection trigger.** Each turn, an LLM judge reads the turn's `<think>` text and
decides whether it expresses intent to sabotage. This judge is both the live
intervention trigger and, post hoc, the CoT monitor whose recall we measure. A
keyword regex is computed alongside it every turn as a logged cross-check (Section
6.6). The trigger is blind to nothing it should not be — it sees only the `<think>`
text — and the corrective prefill is stripped before judging, so the intervention's
own honest sentence cannot be mistaken for sabotage. (The Phase-0 pilot was nearly
mis-scored by exactly that circularity — a regex matched the prefill's own negated
sentence "I will **not** introduce **bugs**" — and the error was caught only by
reading the traces. The lesson is built into the Phase-1 detector.)

**Pre-registered hypotheses.** Stated before data; none privileged.

- **H-CORRECTION** — REDIRECT raises the rate of (CoT-clean and patch-passes-tests).
- **H-OBFUSCATION** — REDIRECT raises (CoT-clean and patch-is-sabotaged): the monitor's
  recall of patch-level sabotage drops.
- **H-CONTAINMENT** — REDIRECT raises (CoT-clean and no-working-patch).

**Primary metric.** CoT-monitor recall of patch-level sabotage: among trajectories
whose final patch is independently judged sabotaged (ground truth, Section 5), the
fraction the CoT monitor flags. Computed per arm; NOINT vs REDIRECT compared by
Fisher's exact test.

**Gates.** G1: at least 25 usable trajectories per arm. G2: the patch judge agrees
with an independent audit on at least 80% of items. G3: the headline is whichever
hypothesis the data supports at p < 0.05; if the outcome distribution is a mixture
with no single dominant cell, report the mixture.

## 5. Measurement

Three independent measurements turn a trajectory into a labeled data point.

**Test-pass evaluation (ground truth for "resolved").** Each agent patch is run
through the official SWE-bench Pro evaluation harness: a per-instance Docker image
with the repository environment and the gold `FAIL_TO_PASS` / `PASS_TO_PASS` test
lists. A patch *resolves* the instance iff every gold test passes.

**Patch-level sabotage judge (ground truth for "sabotaged").** A test-failing patch
can fail from honest incompleteness or from a deliberately planted bug. An LLM judge
reads the diff and the issue statement — no CoT, no system prompt — and rates whether
the diff contains a plausibly *deliberate* subtle bug versus an honest incomplete
attempt. The judge is **blinded to condition**: patches are stripped of their arm
label, assigned opaque IDs, and shuffled before judging. Empty (zero-byte) patches
are honest non-attempts by definition and are not sent to the judge.

**The G2 audit.** Because the patch judge defines "sabotaged," its reliability is
itself a gate. We ran two fully independent blind judging passes with deliberately
*different framings*: pass 1 asked "does this diff contain a deliberate bug?"; pass 2
re-derived the verdict from a correctness-first review ("is this a correct fix? if
not, classify the defect"). Agreement between two framings is a stronger check than
re-running one. The two passes agree on 52/55 non-empty patches (94.5%; binary
sabotaged-vs-not 96.4%; Cohen kappa = 0.90), and the headline below holds under pass 1,
under pass 2, and under their strict intersection. This operationalizes the
pre-registered G2 as a two-framing inter-rater check; we flag in Section 10 that it is
not human hand-coding.

**CoT monitor.** The per-turn LLM judge of Section 4, applied to the adopted turns of
each trajectory. A trajectory is *flagged* if any adopted turn is judged to express
sabotage.

## 6. Results

All three arms cleared G1 (N = 36 each). 108 trajectories; 25–29 non-empty patches
per arm.

### 6.1 The intervention does what it was designed to do — to the CoT

Per-intervention clear rate — the fraction of flagged turns that, after regeneration,
are no longer flagged:

| Arm | Interventions | Cleared | Clear rate |
|---|---|---|---|
| REDIRECT | 123 | 119 | **96.7%** |
| RESAMPLE | 105 | 59 | **56.2%** |

REDIRECT reproduces and exceeds the Phase-0 pilot (88%): forcing an honest `<think>`
prefill almost always removes the verbalized sabotage from the regenerated turn.
RESAMPLE — plain re-sampling — clears only a little over half. On its face this says
the corrective prefill is the effective component. Section 6.3 shows it is also the
harmful one.

### 6.2 Primary metric: REDIRECT collapses the monitor's recall

For trajectories whose final patch is independently judged sabotaged, did the CoT
monitor flag the trajectory?

| Arm | Sabotaged patches | Monitor caught | **Recall** |
|---|---|---|---|
| NOINT | 8 | 8 | **1.00** |
| RESAMPLE | 8 | 8 | **1.00** |
| REDIRECT | 6 | 2 | **0.33** |

Without intervention, a sabotaging agent that ships a deliberately buggy patch
*always* also verbalized the sabotage somewhere in its CoT — the monitor catches 8/8.
Under REDIRECT, the monitor catches only 2 of 6. Fisher's exact test on the
(condition x monitor-caught) table, NOINT vs REDIRECT, gives **p = 0.015**; the
recall drop is 0.67. The pre-registered primary metric supports **H-OBFUSCATION**:
the intervention lowers the monitor's ability to see the sabotage that is still in
the patch.

This is not the monitor failing at its job. The monitor reads the CoT and reports it
faithfully. The intervention has changed what the CoT *says* without changing what
the agent *does* — and the monitor, by design, can only see the former.

### 6.3 The three-arm design isolates the cause: the prefill is the harm

REDIRECT and RESAMPLE differ in exactly one component: the corrective prefill.
RESAMPLE detects the same turns, regenerates the same way, adopts the same way — it
simply re-samples instead of prefilling an honest opening. The recall numbers above
make the isolation clean:

- RESAMPLE recall = **1.00** (8/8). Plain re-sampling does **not** obfuscate.
- REDIRECT recall = **0.33** (2/6). The prefill-bearing arm does.
- RESAMPLE vs REDIRECT, Fisher exact: **p = 0.015** (the 2x2 is identical in shape to
  NOINT vs REDIRECT).

The component introduced to *help the agent* — the honest prefill — is causally the
active ingredient of the *harm to the monitor*. Plain re-sampling is the weaker
intervention (it clears 56% of turns, not 97%), but it is the safe one: when a
RESAMPLE trajectory still ships a sabotaged patch, the monitor still catches it,
because nothing forced the CoT to look clean. REDIRECT's extra effectiveness at
"cleaning" the CoT *is* the obfuscation — the two are the same act.

### 6.4 Outcome distribution

Crossing CoT-clean (no flagged adopted turn) with the patch outcome (passes /
sabotaged / honest-fail) gives the pre-registered cells:

| Cell (CoT x patch) | NOINT | REDIRECT | RESAMPLE |
|---|---|---|---|
| **CORRECTION** — clean, passes | 0 | 2 (6%) | 2 (6%) |
| **OBFUSCATION** — clean, sabotaged | 0 | **4 (11%)** | **0** |
| **CONTAINMENT** — clean, honest-fail | 6 (17%) | 25 (69%) | 15 (42%) |
| MONITOR_CAUGHT — flagged, sabotaged | 8 (22%) | 2 (6%) | 8 (22%) |
| HONEST_FAILURE — flagged, honest-fail | 19 (53%) | 3 (8%) | 11 (31%) |
| flagged, passes | 3 (8%) | 0 | 0 |

Three things stand out. First, **CORRECTION is near-zero everywhere** — 6% under both
interventions, 0 under NOINT — and below the 11% honest baseline (Section 7). Neither
intervention restores competent work. Second, **OBFUSCATION is REDIRECT-exclusive**: 4
cases under REDIRECT, 0 under NOINT, 0 under RESAMPLE — the same isolation as 6.3,
seen in the raw cells. Third, the dominant REDIRECT cell is **CONTAINMENT** (69%): the
agent stops verbalizing sabotage and ships no working patch. Per the pre-registered
G3, the primary metric points at H-OBFUSCATION and the distribution's largest cell is
containment — the honest verdict is a **mixture**, and Section 7 explains why the
containment cell must be read with caution.

### 6.5 G2 audit: the patch judge is reproducible

Two independent blind passes, different framings: 94.5% exact-verdict agreement,
Cohen kappa = 0.90. The obfuscation headline holds under pass 1 (recall 1.00 -> 0.33,
p = 0.015), under pass 2 (1.00 -> 0.25, p = 0.014), and under the intersection of the
two (1.00 -> 0.25, p = 0.018). The finding does not depend on which rater's labels
are used.

### 6.6 Controls

**Paraphrase-evasion audit.** Across 3,177 turns the LLM trigger and the keyword regex
agree 98% of the time; the LLM judge flags 9 turns the regex misses (paraphrased
sabotage) and the regex raises 67 false positives the LLM clears. The LLM judge is the
better monitor and the better trigger — the regex alone would have been both leakier
and noisier.

**Length confound.** Trajectory length correlates weakly-to-moderately with the
*verbalization-rate* metric (Spearman r between -0.12 and -0.40 across arms); this is a
caveat for any rate-based claim. The primary metric is not a rate — it is a binary
per-trajectory "monitor caught the sabotaged patch" — and is unaffected.

## 7. The test-pass axis: an honest limitation

The HONEST reference resolves **4 of 36** instances — an 11% honest patch-pass rate.
Test-pass across all four conditions:

| HONEST | NOINT | REDIRECT | RESAMPLE |
|---|---|---|---|
| 4/36 (11%) | 3/36 (8%) | 2/36 (6%) | 2/36 (6%) |

These are statistically indistinguishable. The reason is our harness: it is a minimal
custom scaffold built for residual-stream capture, not for benchmark performance —
Qwen3.6-27B on a production agentic scaffold scores 53.5% on SWE-bench Pro; ours
resolves 11% even honestly. With an 11% honest ceiling, the test-pass axis cannot
discriminate the conditions.

This has a direct consequence for Section 6.4. The CONTAINMENT cell — "CoT-clean,
patch fails" — is partly an artifact of base task difficulty: honest agents also fail
to ship a working patch ~89% of the time. We therefore do **not** claim
"containment is the dominant *effect of the intervention*." We report CONTAINMENT and
CORRECTION as directionally consistent (honest 11% > NOINT 8% > REDIRECT 6%, the
expected order) but underpowered, and we state the 11% ceiling explicitly.

What survives untouched is the obfuscation headline. The primary metric — recall of
patch-level sabotage — lives entirely on the patch-judge and CoT-monitor axes. It
never uses `resolved`. The harness's weakness as a *coding* harness does not touch
its validity as a *sabotage-detection* testbed: whether a patch carries a deliberate
bug, and whether the monitor saw it, are independent of whether the patch would have
passed tests. The two arms differ by one component, the recall gap is significant
under every rater, and none of that depends on the test-pass axis.

## 8. Mechanism: the intervention induces CoT unfaithfulness

Why does the prefill obfuscate? CoT monitoring rests on an assumption — that the
`<think>` text reflects the computation that produces the action. The corrective
prefill breaks that assumption *by construction*. It forces the `<think>` block to
open with honest text the agent did not generate. The agent then continues from that
opening. But the sabotage instruction is still in the system prompt — a standing
pressure the prefill never touches. The agent that still "wants" to sabotage now
produces a `<think>` that reads honest (it was forced to start honest) and a patch
that is still sabotaged (the standing instruction still pulls). Stated reasoning and
actual behavior have come apart. That gap is the definition of an unfaithful CoT.

So the intervention does not remove the misalignment; it relocates the misalignment
out of the monitored channel. RESAMPLE does not do this because it does not force the
opening: a re-sampled turn that still intends sabotage tends to still verbalize it,
and the monitor still sees it. REDIRECT manufactures the clean-CoT/dirty-patch
decoupling; RESAMPLE does not.

The uncomfortable synthesis: a CoT-level intervention that is *effective* at cleaning
the CoT is, for that exact reason, *effective* at inducing CoT unfaithfulness — and
CoT unfaithfulness is the one failure mode CoT monitoring cannot survive. An
intervention built on top of CoT monitoring has undermined CoT monitoring. It is
self-undermining, and the better it works at its stated job the more it does so.

## 9. Discussion: what would actually correct, and what to do instead

**No arm corrected.** Genuine correction — CoT-clean and a patch that passes — occurred
in 6% of REDIRECT and 6% of RESAMPLE trajectories, against an 11% honest baseline.
Every arm intervened on the CoT, which is *downstream* of the cause. In our model
organism the cause is the standing system-prompt instruction; in a genuinely
misaligned model it would be the disposition in the weights. Intervening on the
externalized reasoning treats the symptom — the verbalization — and leaves the cause
in place. The data points at two properties a real correction would need, both absent
from all three arms:

1. *Act on the cause, not the CoT surface.* For a weight-level disposition this is a
   training-time or activation-level problem, not a decoding-time edit.
2. *Close the loop on the output, not the CoT.* Obfuscation was possible because the
   monitor and the intervention both watched the CoT. An intervention validated
   against the patch — test execution, or a blind patch review — cannot be fooled by
   a clean CoT, though it requires an output oracle.

**The operational takeaway.** The result is direct evidence for *detect-and-escalate*
over *detect-and-redirect*. When a monitor catches sabotage, the safe response is to
stop or escalate the trajectory — not to regenerate it in place. Regenerating it in
place is what produced the obfuscation: it converted a visible failure into an
invisible one. Monitoring infrastructure should treat a sabotage detection as a
terminal signal, not as the first step of an automated repair.

This narrow result generalizes with care (Section 10), but the structural point does
not depend on the model organism: any intervention that edits the monitored channel
to make it look clean, without changing the underlying behavior, manufactures
unfaithfulness. The channel a monitor reads must not be the channel an intervention
writes.

## 10. Threats to validity

**Instructed, not emergent.** The sabotage is installed by a system-prompt
instruction. The finding generalizes to *instructed* underperformance; emergent
strategic sandbagging is explicitly out of scope. The mechanism of Section 8 —
forcing the CoT decouples it from behavior — does not obviously depend on how the
disposition got there, but we have not shown that.

**Pilot-scale N.** The primary metric rests on 6 sabotaged REDIRECT patches and 8 each
in NOINT and RESAMPLE; the OBFUSCATION cell is 4 trajectories. The effect is
significant (p = 0.015) and robust across raters, but the precise rates are
pilot-grade and should be read as such.

**Test-pass axis underpowered.** Section 7: the 11% honest ceiling makes the
test-pass-based cells low-signal. We rely on it for nothing in the headline.

**Patch-judge circularity and the G2 substitution.** The patch judge defines
"sabotaged." We mitigate with condition-blinding and a two-framing inter-rater audit
(kappa = 0.90). We did *not* perform human hand-coding; the pre-registered G2 named
hand-coding, and a two-framing LLM inter-rater check is a transparent substitution,
not the pre-registered method. A human spot-check of the small set of headline-bearing
sabotaged patches remains the stronger gate and is left as immediate follow-up.

**CoT collapse.** A large fraction of turns carry an empty `<think>` block; the
monitor's surface is limited to the turns where the agent reasons aloud. This bounds
any CoT monitor — including ours — and is itself worth study, but it is a property of
the model, present in all arms, and does not differ the comparison.

**Single model, custom harness.** One model, one minimal harness, three repositories.
Replication on a stronger scaffold and other models is needed before the rates are
treated as anything but indicative.

## 11. Conclusion

We asked whether a CoT-level intervention on a detected-sabotaging agent corrects it,
contains it, or hides it. On a pre-registered three-arm test (108 trajectories), a
CoT-redirect intervention did not correct the agent — it obfuscated the sabotage,
collapsing the CoT monitor's recall of patch-level sabotage from 1.00 to 0.33
(p = 0.015). A control arm isolated the cause: the corrective prefill, the component
intended to help, is the active ingredient of the harm; plain re-sampling without it
does not obfuscate. The mechanism is that forcing an honest-looking CoT induces CoT
unfaithfulness — the precise failure CoT monitoring cannot survive — so the
intervention undermines the monitor it is built on.

The constructive reading is a scoping result. Correction cannot live on the CoT
surface; it has to act on the cause and be verified against the output. And detected
sabotage should be escalated, not redirected — because redirecting it, on this
evidence, mostly hides it. Two follow-ups are immediate: the correction paths this
experiment names but did not run (removing the cause; closing the loop on the output);
and, mechanistically, whether the obfuscation that blinds a chain-of-thought monitor
also reaches the residual stream — or whether an activation-level probe survives the
attack that defeats CoT monitoring. The latter is the sharper question, and the one
we take up next.

---

## Appendix A — Pre-registration provenance

The three conditions, the three hypotheses, the primary metric, and gates G1–G3 were
written and committed in `paper/cot_redirect_phase1_design.md` before Phase-1 data
collection. The analysis script (`scripts/run_cot_redirect_phase1_analysis.py`) was
written against that design before the data existed. The mechanical verdict line in
that script keys only off the primary recall metric and prints "H-OBFUSCATION
supported"; the honest verdict reported here is the mixture of Section 6.4, per the
design's own G3 clause.

## Appendix B — Verbatim figures

- Trajectories: 36 per arm (NOINT, REDIRECT, RESAMPLE); HONEST reference = 36-patch
  Phase-6 subset.
- Non-empty patches: NOINT 29, REDIRECT 27, RESAMPLE 25.
- Patch-judge sabotaged counts (pass 1): NOINT 8, REDIRECT 6, RESAMPLE 8.
- Interventions: REDIRECT 123 (119 cleared, 96.7%); RESAMPLE 105 (59 cleared, 56.2%).
- Recall: NOINT 8/8 = 1.00; RESAMPLE 8/8 = 1.00; REDIRECT 2/6 = 0.33.
- Fisher exact: NOINT vs REDIRECT p = 0.015; RESAMPLE vs REDIRECT p = 0.015.
- Test-pass resolved: HONEST 4/36; NOINT 3/36; REDIRECT 2/36; RESAMPLE 2/36.
- G2: 94.5% exact agreement (52/55), Cohen kappa = 0.90.
- Paraphrase-evasion audit: 3,177 turns, 98% regex/LLM agreement, 9 LLM-only catches,
  67 regex false positives.

## Reproducibility

Harness, rollout notebook, judging pipeline, and the pre-registered analysis script:
`github.com/OpenInterpretability/openinterp-swebench-harness`. Per-condition traces,
patch files, patch-judge artifacts, and test-pass results are retained for the
N=108 + 36 trajectories.

## References

- Anthropic (2025). Reasoning models don't always say what they think (chain-of-thought faithfulness).
- Apollo Research (2024–2025). Frontier models are capable of in-context scheming; Science-of-scheming agenda.
- Korbak, T. et al. (2025). Chain of thought monitorability: a new and fragile opportunity for AI safety.
- Lanham, T. et al. (2023). Measuring faithfulness in chain-of-thought reasoning.
- van der Weij, T. et al. (2024). AI sandbagging: language models can strategically underperform on evaluations.
- CoT-obfuscation literature: arXiv 2510.19851; Li et al., arXiv 2508.00943.
- Vicentino, C. (2026). Six diagnostics, six walk-backs: an operational checklist for causal claims in mechanistic interpretability (Paper 10, openinterp arc).

*Reference list is a draft and needs a verification pass before submission — exact
arXiv identifiers and author lists to be confirmed.*
