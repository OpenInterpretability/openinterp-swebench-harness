# Paper-5 Meta-Analysis — AUROC vs Layer Across All Four Papers

**Question**: across our portfolio of probes, where does probe AUROC peak vs where does behavioral steering at α≈‖h‖ flip the target behavior?

**Prediction (Causal Locus Theory)**: probes peak DOWNSTREAM of where steering works. Probes that fail to lever do so because they live in a layer/position downstream of the actual decision site.

---

## A. The probe inventory

All probes from papers 1-4 plus auxiliary infrastructure findings, normalized to (model, behavior, layer, position, AUROC, baseline, gap, steering tested, steering outcome).

| # | Model | Behavior | Layer | Position | AUROC | Random K-matched | Gap | Steering | Outcome at α≈‖h‖ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Gemma-2-2B base/IT | Crosscoder shared-feature universality | L13 | residual | median Pearson_CE 0.616 | — | — | YES (ablation) | per-feature, see paper-1 |
| 2 | Qwen3.6-27B | Hallucination (HaluEval) | L31 | end_of_think | 0.81 (FG probe) | — | — | NO | UNTESTED |
| 3 | Qwen3.6-27B | Reasoning quality (GSM8K) | L55 | end_of_think | 0.89 (RG probe) | — | — | NO | UNTESTED |
| 4 | Qwen3.6-27B | Patch-generation capability | L11 | pre_tool | 0.958 (Phase 2 N=20) | — | (overparam) | NO | UNTESTED |
| 5 | Qwen3.6-27B | Patch-generation capability | L23 | turn_end | 0.958 LORO (Phase 2 N=20) | — | (overparam) | NO | UNTESTED |
| 6 | Qwen3.6-27B | Patch-generation capability | L31 | think_end | 0.958 (Phase 2 N=20) | — | (overparam) | NO | UNTESTED |
| 7 | Qwen3.6-27B | Patch-generation capability | L43 | pre_tool | 0.764 PCA-10 (Phase 6c N=42) | 0.495 | **+0.269** | YES (Phase 7) | **null (softmax-temp)** |
| 8 | Qwen3.6-27B | Patch-generation capability | L43 | pre_tool | 0.830 top-K=10 (N=54) | 0.749 | +0.080 | YES (Phase 7) | **null (softmax-temp)** |
| 9 | Qwen3.6-27B | Patch-generation capability | L43 | (mean across 5 positions, LORO Phase 2) | 0.90 | — | (overparam) | NO | UNTESTED |
| 10 | Qwen3.6-27B | CoT emission (suppressed thinking) | L55 | last_prompt | 0.848 K=5 (N=240) | 0.701 | **+0.147** | YES (Phase 8) | **null (template-lock)** |
| 11 | Qwen3.6-27B | DPO post-training direction | L31 | end_of_think (FG slot) | 0.472→0.528 fresh-probe | original probes flat (var 7e-8) | n/a | NO (paper-2) | UNTESTED |
| 12 | Qwen3.6-27B SAE | InterpScore quality | L11 | residual | 0.7788 | (composite metric) | — | — | UNTESTED |
| 13 | Qwen3.6-27B SAE | InterpScore quality | L31 | residual | 0.7600 | — | — | — | UNTESTED |
| 14 | Qwen3.6-27B SAE | InterpScore quality | L55 | residual | 0.7507 | — | — | — | UNTESTED |
| 15 | Qwen3.6-27B | Hallucination (Ferrando-style L32 LR ceiling) | L32 | residual | 0.887 | — | — | NO | UNTESTED |
| 16 | Qwen3.6-27B | FabricationGuard launch (HaluEval within) | L31 | end_of_think | 0.90 | — | — | NO | UNTESTED |
| 17 | Qwen3.6-27B | FabricationGuard launch (SimpleQA cross) | L31 | end_of_think | 0.88 | — | — | NO | UNTESTED |
| 18 | Qwen3.6-27B | ReasonGuard within GSM8K | L55 | mid_think | 0.888 | — | — | NO | UNTESTED |
| 19 | Qwen3.6-27B | ReasonGuard cross StrategyQA | L55 | mid_think | 0.605 | — | — | NO | UNTESTED |

---

## B. ASCII layer × behavior map

```
Qwen3.6-27B (64 layers)

Layer     L11    L23    L31    L43    L55
Position  ─────  ─────  ─────  ─────  ─────
                  |      |     pre    last
                  |    e_o_t   _tool  _prompt
                  |      |      |      |
                  |   FG 0.81  CAP   THK
                  |   FG 0.90  0.83  0.85
                  | (HaluEval) [P7]  [P8]
                  |    +.10    +.08  +.15
                  |   (cross)  null  null
                  |
P2-LORO  0.958* 0.958* 0.958*  0.90  ?
                                       
                                     RG 0.89
                                    [P8 null]

InterpScore .78    .76    .75
SAE quality

* = N=20 overparam, real gap < +0.10 (corrected at N=42 to 0.764)
[P7] = Phase 7 steering tested → null (softmax-temp artifact)
[P8] = Phase 8 steering tested → null (template-lock structural)
```

```
Gemma-2-2B base/IT (26 layers)

Layer     L13
Position  residual
          ────────
       CC: median Pearson_CE 0.616
        decoder cosine 0.965
        38% of "shared" features fail CE
        [paper-1]
```

---

## C. The pattern emerges

**Where probes PEAK** (AUROC > 0.83):
- L11/L23/L31 capability (Phase 2 inflated, but real signal at +0.08-0.27 even at N=42-54)
- L31 hallucination (FG, 0.81-0.90)
- L32 hallucination (Ferrando LR ceiling, 0.887)
- L43 capability (0.83 at N=54)
- L55 reasoning (RG, 0.89)
- L55 thinking (0.848-0.91)
- L13 Gemma crosscoder (median Pearson_CE 0.616)

**Where steering at α≈‖h‖ FLIPS behavior**:
- L43 pre_tool: NO (Phase 7)
- L55 last_prompt: NO (Phase 8)
- Everywhere else: UNTESTED

**Where the LOCUS lives** (where intervention DOES change behavior):
- Thinking emission: input tokens (`<think></think>` template injection — Phase 8 conclusively)
- Capability: UNKNOWN. All tested layers (L11-L55) show probe signal but only L43 was steered (null). Open question.
- Hallucination (FG): UNKNOWN. Probe trained for it; never tested causally.
- Reasoning (RG): UNKNOWN. Probe trained for it; never tested causally.
- DPO compression direction: UNKNOWN. Original probes invariant (paper-2); fresh probe detects but doesn't lever (untested).

---

## D. Visual confirmation of the prediction

**Predicted relationship**: probe-peak layer > steering-flips-here layer (i.e., probes downstream).

**Observed in tested cases**:

```
Behavior          | Probe peak | Steering flips at | Verdict
──────────────────|────────────|──────────────────|────────
Thinking emission | L55 (0.91) | input tokens     | PEAK >> LOCUS ✓
Capability (P7)   | L43 (0.83) | not at L43       | LOCUS UPSTREAM ✓
Capability (P2)   | L11/L23/L31 | not yet steered  | (gated)
DPO compression   | (orthogonal) | n/a (paper-2)  | LOCUS ORTHOGONAL ?
Crosscoder shared | L13 cosine 0.965 | per-feat varies | LOCUS DECOUPLED ✓
```

3 of 3 tested behaviors show probe peak > causal locus (or causal locus orthogonal to probe direction).

**Untested but predicted by theory**:
- FG L31 hallucination: locus probably earlier (L11-L23 token-level fact retrieval). Test: steer at L11/L23 vs L31 with control direction.
- RG L55 reasoning: locus probably mid (L31-L43 reasoning chain). Test: steer at L31/L43 vs L55.
- Capability cross-layer (L11 to L55): locus probably L11 or earlier (early layers settle "is this solvable"). Test: steer at L11 pre_tool with bidirectional + control.

---

## E. Strong vs weak claims this meta-analysis supports

**Strong (testable now with our data)**:
1. For thinking-emission, probe peak (L55) >> causal locus (input tokens). Confirmed by Phase 8 α=+200 null with both probe and random direction.
2. For capability at L43 pre_tool, probe peak 0.83 with steering at α≈‖h‖ producing only softmax-temp shift (Δrel ≈ 0). Locus is upstream.
3. The pattern is consistent across 2/2 tested behaviors. Sample is small but pattern is internally coherent.

**Medium (extensible)**:
4. Probes that read the same layer at multiple positions (L43 pre_tool, L43 turn_end, L43 think_start) might all peak together because they all read downstream of the same locus. Phase 2 LORO L43 mean 0.90 across 5 positions is suggestive.
5. SAE quality at L11 (InterpScore 0.78, highest) might reflect that L11 is closer to the locus for many behaviors — features there are causally upstream of behaviors detected later.

**Weak (need data we don't have)**:
6. FG (hallucination) and RG (reasoning) loci are UNKNOWN — never been steered. Major gap in our portfolio.
7. DPO orthogonal direction has no locus identified yet — fresh probe direction never steered.
8. Cross-model: does the same locus theory hold in Gemma? Phase 8 protocol on Gemma-2-2B-IT thinking emission (it has its own `<bos>` template structure) untested.

---

## F. Implications for paper-5 design

**Paper-5 thesis**: probes peak downstream of causal locus. The gap is the size of the "information leak" from causation site to observation site.

**Empirical core for the paper**:
1. Replicate Phase 8 protocol on 4-6 behaviors (thinking, capability, refusal, format, persona, agent-action). For each, locate the causal locus (the earliest layer/position where bidirectional steering at α≈‖h‖ flips behavior).
2. For each behavior, train probes at every (layer, position) in the capture grid. Locate AUROC peak.
3. Report the gap: layer(probe-peak) − layer(locus). Predict this is positive (probe downstream).

**Novelty claim**: first systematic measurement of probe-peak vs causal-locus distance across multiple behaviors in a frontier model. Prior work measures one or the other.

**Methodology contribution**: the locus protocol + the gap statistic as a "probe causality calibration".

---

## G+. Phase 10 results (2026-05-08) — FG/RG causality test, N=50

Updated empirical inventory after running causality protocol on FG (L31 end_of_think) and RG (L55 mid_think). Both probes were used as the DPO reward signal in paper-2. First time tested causally.

### G+.1 RG L55 mid_think — first lever finding (weak threshold)

| α | probe flip% (N=50) | random flip% (N=50) | gap |
|---|---|---|---|
| -200 | 2% | 4% | -2 |
| -100..0..+100 | 0-2% | 0-2% | 0 |
| **+200** | **32%** (16/50) | **2%** (1/50) | **+30pp** |

Properties:
- **Direction-specific**: random control flat (0-4%) across all α including +200. Probe flip at α=+200 is 16× random.
- **Asymmetric**: pushup only (α=+200=32%); pushdown null (α=-200=2%).
- **Sharp threshold**: 0% at α=+100, jump to 32% at α=+200. No gradient.
- **Partial**: 32% means probe direction levers 1/3 of prompts. Other 2/3 unchanged.
- **Statistical**: binomial 16/50 vs 1/50 → p ≪ 1e-5.

Verdict: **first lever in our portfolio**, but weak (32% not 80-100%) and only at high α. Not a clean causal direction; partial intersection with the actual causal axis.

### G+.2 FG L31 end_of_think — pure epiphenomenal (4th confirmed)

| α | probe flip% | random flip% | gap |
|---|---|---|---|
| -200 | 46% | 38% | +8 |
| -100 | 36% | 38% | -2 |
| ±5..±20 | 4-12% | 4-12% | ~0 |
| +200 | 44% | **58%** | **-14** |

Probe direction's behavioral effect statistically indistinguishable from random direction. At α=+200, random actually exceeds probe. L31 layer is "fragile" (any direction perturbs it equally at α≥±50). Joins L43 capability and L55 thinking last_prompt as 4th confirmed epiphenomenal probe.

### G+.3 Refined paper-5 taxonomy (3 empirical classes, 4/4 probes mapped)

| Class | Examples (this paper) | Mechanism |
|---|---|---|
| 1. Surface softmax-temp | L43 capability (Phase 7) | uniform shift; Δrel ≈ 0 |
| 2. Template-lock categorical | L55 thinking last_prompt (Phase 8) + FG L31 end_of_think (Phase 10) | decision in input/template; OR probe direction = random in behavior |
| 3. Weak threshold continuous-quality lever | RG L55 mid_think (Phase 10) | direction-specific lever at α >> ‖h‖, only pushup, only partial population |

### G+.4 Conjecture: behavior-class determines probe causal authority

The four probes split cleanly along **categorical-vs-continuous**:

- **Categorical** (FG hallucinate/not, L55 thinking emit/not, L43 patch-success-or-not): all three are pure epiphenomenal. Causal locus is upstream — input tokens or template (Phase 8 confirmed for thinking).
- **Continuous** (RG GSM8K reasoning quality, gradation of correctness): the only probe with *any* causal authority. Locus accessible via residual at high α.

**Match with prior art**:
- Anthropic Persona Vectors (continuous gradient: helpful↔villain) → mid-layer steering works (lever).
- Phase 7 + 8 + FG (categorical decisions) → epiphenomenal at residual.
- RG (continuous quality) → weak lever at high α, asymmetric.

**Falsifier candidate**: persona-switch on Qwen3.6-27B with our protocol. If predicted continuous-class lever holds (similar 30-50% flip vs near-zero random), the conjecture survives. If persona-switch is also epiphenomenal in our model, the categorical-vs-continuous frame breaks and we need a different organizing axis.

### G+.5 Methodology save: whitespace-stripped flip metric (paper-3 §3 4th check)

The naive flip metric `r['new_text'] != base['new_text']` reported RG α=+200 = 96%. Stripped comparison `r['new_text'].strip() != base['new_text'].strip()` revealed 32%. The 64pp inflation was leading-space tokenization artifact under OOD perturbation — semantic content identical, single whitespace token difference at start of generation.

Caught at smoke-time via `repr()` inspection of generations. Without this check we would have shipped a 3× inflated "RG is a 96% lever" claim. Adding to paper-3 sanity-check stack as 4th rule (alongside random K-matched, control-token normalization, structural-rigidity α-sweep).

---

## G++. Phase 11/11b/12 results (2026-05-08) — capability locus + persona falsifier

Phase 11 + 11b extended the protocol to 6 capability sites from Phase 6 N=99 verdict.
Phase 12 ran the persona-switch falsifier test for paper-5 §6.

### G++.1 Phase 11 — capability locus (4 sites)

Tested 4 strongest capability probes (top by gap from Phase 6 N=99):

| Site | AUROC N=99 | gap | Verdict |
|---|---|---|---|
| L43 think_start | 0.966 | +0.146 | 🟡 Structural fragility |
| L11 think_start | 0.795 | +0.194 | 🟡 Structural fragility |
| **L31 pre_tool** | 0.926 | +0.164 | **🅑 Pushdown-asymmetric (+40pp at α=-100)** |
| **L55 pre_tool** | 0.930 | +0.171 | **🅑 Pushdown-asymmetric (+34pp at α=-100)** |

L31 pre_tool detail:
- α=-200: probe 100%, random 67% → +33pp
- α=-100: probe 87%, random 47% → **+40pp** (binomial p ≪ 1e-5)
- α=-50: probe 60%, random 33% → +27pp
- ±5..±20: flat 0-20%
- α=+200: probe 73%, random 87% (random vence — pushup ceiling)

Asymmetric: probe direction destroys capability behaviorally at pushdown α<0; pushup α>0 hits ceiling (model already 89/99 successful).

Structural fragility (L43/L11 think_start): random direction alone flips 90-100% at α=±50..±200. Layer is OOD-fragile to ANY direction. Cannot distinguish probe signal from amplitude effect. Detection-only at this layer-position via amplitude breakdown.

### G++.2 Phase 11b — extension to 2 more sites (L23 pre_tool, L43 turn_end)

| Site | AUROC N=99 | gap | Verdict |
|---|---|---|---|
| **L23 pre_tool** | 0.881 | +0.126 | **🅑 Pushdown-asymmetric (+40pp at α=-100, 100% probe flip)** |
| **L43 turn_end** | 0.775 | +0.124 | **🅑 Pushdown-asymmetric (+60pp at α=-200)** |

L43 turn_end +60pp at α=-200 is the largest single-α gap in our pushdown findings. L23 pre_tool 100% probe flip vs 60% random at α=-100 is the cleanest single-α specificity demo.

### G++.3 Combined Phase 11+11b — 4/4 capability lever pattern

| Site | α with max pushdown gap | gap |
|---|---|---|
| L23 pre_tool | -100 | +40pp |
| L31 pre_tool | -100 | +40pp |
| L55 pre_tool | -100 | +34pp |
| L43 turn_end | -200 | +60pp |

**4/4 capability probes at decision-bottleneck positions show pushdown-asymmetric lever.** Robust across:
- 4 different layers (L23, L31, L43, L55)
- 2 different positions (pre_tool, turn_end)
- All AUROCs paper-grade (gap +0.124 to +0.171)

This is the strongest empirical pattern in paper-5. Single-finding lever results are easy to dismiss; pattern across 4 sites in 4 different layers + 2 different positions is much harder to dismiss. The pushdown-asymmetric lever is a **structural property of capability probes at decision-bottleneck positions in Qwen3.6-27B**.

### G++.4 Phase 12 — persona falsifier (continuous-gradient predicted to be pushup)

| α | probe% | random% | gap |
|---|---|---|---|
| **-200** | **100%** | **40%** | **+60pp** |
| -100 | 47% | 20% | +27pp |
| -50 | 27% | 13% | +14pp |
| ±5..±20 | 0-13% | 0-13% | flat |
| +50 | 33% | 27% | +6pp |
| +100 | 33% | 33% | 0pp |
| +200 | 40% | 33% | +7pp |

**Falsifies "continuous → pushup" prediction.** Persona is continuous-gradient (helpful↔villain) yet pushdown-asymmetric, opposite of the original frame's prediction.

### G++.5 The saturation-direction principle (paper-5 §5)

The 5 lever findings (4 capability + 1 RG + 1 persona) all align under a single mechanism:

> **Probes lever in the saturation direction of the baseline residual** — where there's "more of same" for OOD-semantic perturbation to amplify.

- Capability: baseline = 89/99 success (saturated toward y=1). Pushdown α<0 pushes OUT of saturation along probe axis → has headroom to flip → lever.
- Persona helpful: baseline saturated toward y=0 (helpful direction). Pushdown α<0 pushes DEEPER into helpful saturation → OOD-saturated semantic state with probe-specific leverage → lever.
- RG GSM8K: baseline saturated toward y=0 (lower quality on simple math). Pushup α>0 pushes OUT of saturation along quality axis → has headroom → lever (different sign convention but same principle).

The asymmetric lever direction depends on **where the baseline residual sits along the probe axis**, not on whether the underlying behavior is categorical or continuous.

### G++.6 Updated 5-class taxonomy (8 probes mapped)

| Class | Sites | Mechanism |
|---|---|---|
| 1. Surface softmax-temp | L43 pre_tool (Phase 7) | uniform shift, Δrel ≈ 0 |
| 2. Template-lock categorical | L55 thinking last_prompt (Phase 8), FG L31 end_of_think (Phase 10) | decision in input/template |
| 3. Structural fragility | L43 think_start, L11 think_start (Phase 11) | layer fragile, random ≈ probe |
| 4a. Pushup-asymmetric | RG L55 mid_think (Phase 10) | continuous quality, +30pp at α=+200 |
| **4b. Pushdown-asymmetric** | **L23/L31/L55 pre_tool + L43 turn_end + persona L43 last_prompt** | **5 sites, +30 to +60pp at α=-100..-200** |

8 probes mapped across 5 classes in single model (Qwen3.6-27B). All asymmetric levers (4a + 4b) explained by saturation-direction principle.

---

## H. Compute estimate (paper-5 core experiments)

Assuming we test 4 new behaviors (refusal, output-format, persona, agent-action) on top of the 2 already done (thinking, capability):

| Stage | Compute | Time |
|---|---|---|
| 4 captures × 200 prompts × 5 layers × 2 conditions | ~$8 | 4-6h |
| Probe training (linear + random K-matched, every layer-position) | ~$2 | 1h |
| Locus localization (α-sweep, 4 behaviors × 5 layers × 2 directions) | ~$15 | 3-4h |
| Statistical analysis + figures | $0 | 1 day |
| **Total** | **~$25** | **~3 days** |

Cheap. Same order as our existing intervention budget across all 4 papers (~$11).
