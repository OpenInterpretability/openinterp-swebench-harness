# Cross-Probe Coherence Buildup in Successful LLM Agent Trajectories

**Author:** Caio Vicentino (ORCID: 0009-0003-4331-6259), OpenInterpretability  
**Target venue:** NeurIPS MI Workshop 2026 (Sep submission window) OR ICLR 2027 main  
**Draft started:** 2026-05-18  
**Status:** OUTLINE — empirical foundation locked, writeup pending

---

## Abstract (~150 words)

Multi-probe ensembles for LLM monitoring are usually evaluated probe-by-probe. We propose a meta-signal — cross-probe coherence κ_t — measured as the mean absolute pairwise correlation across N concurrent behavioral probes within a moving window of agent turns. On 99 SWE-bench Pro traces from Qwen3.6-27B (3872 turns, 5 per-turn probes at L43 paired residual: tool-class × thinking-length × tool-success × repo-identity), we find: (1) per-trace mean κ_t separates successful from failed trajectories at AUROC 0.677, Mann-Whitney p=0.003; (2) κ_t SLOPE over a trajectory is strongly positive in successful traces (+0.00368/turn) and flat in failed traces (+0.00029, p=0.0003). This pattern is the inverse of cardiac uncoupling (cardiology: cross-vital decorrelation anticipates decompensation). For LLM agents, successful reasoning is characterized by progressive cross-axis coherence buildup; failure is a deficit, not a collapse. We discuss implications for agent monitoring and the limits of biological metaphors in interpretability.

---

## 1. Introduction

### 1.1 Motivation

LLM agent monitoring (Apollo Watcher, Anthropic deception probes, OpenAI internal monitoring) overwhelmingly evaluates single behavioral probes in isolation. With N≥4 probes available (we have 5+ in agent-probe-guard v0.1), a natural question: does the joint correlation structure across probes carry meta-signal beyond individual probe outputs?

### 1.2 Biological inspiration

20 years of ICU literature (since Morris 2006) shows that healthy multi-vital signs (HR ↔ BP ↔ RR) are highly cross-correlated, and decompensating patients show **cross-vital uncoupling** (collapse of correlation) HOURS before crisis. This pattern is robust across thousands of patients and is the basis of validated Early Warning Scores.

We test whether an analogous pattern exists in LLM reasoning agents.

### 1.3 Contribution

1. Define cross-probe coherence κ_t for N concurrent behavioral probes in agent loops
2. Show κ_t separates success/failure trajectories with AUROC 0.677 (gap +0.176 vs shuffled, p=0.003)
3. Discover that κ_t SLOPE (not absolute value) is the strongest signal: successful traces show +0.00368/turn coherence buildup vs +0.00029 in failures (p=0.0003)
4. Demonstrate the pattern is INVERSE to cardiac uncoupling: success = coherence buildup, failure = absence of buildup, not collapse
5. Validate via pre-registered 4-gate protocol (3/4 pass) that ruled out 5 prior single-probe hypotheses

### 1.4 Methodological context

This finding emerged after rigorous walk-back of 5 single-probe candidates (silent-refusal, tool-doubt, tool-doubt position-sweep, Inner-Outer paired, κ_t v1 with trace-level labels). Pre-registered gates (AUROC > 0.65, baseline < 0.55, lift > 0.05, std < 0.05) acted as vaccine against lexical leak and label confounds. We argue cross-probe meta-signals are a distinct research direction that complements (rather than competes with) single-probe development.

---

## 2. Method

### 2.1 Cross-probe coherence κ_t

For N probes P_1, ..., P_N applied per agent turn, let s_i(t) be probe i's output at turn t for a given trace. Define:

κ_t(trace, t; W) = mean over i≠j of |corr(s_i, s_j) | within window [t-W, t+W]

For trace-level summary: κ̄(trace) = mean of κ_t over all valid turns.

### 2.2 Probe stack (5 per-turn axes)

Trained on SWE-bench Phase 6 L43 paired residual (think_end + turn_end concat, 10240 dim), GroupKFold by instance_id:

| Probe | Per-turn label | AUROC |
|---|---|---|
| A_tool_finish | turn calls finish? | 1.000 |
| B_tool_bash | turn calls bash? | 1.000 |
| C_long_thinking | thinking_chars > trace median? | 0.990 |
| D_tool_ok | tool result this turn .ok=True? | 0.998 |
| E_repo_ansible | trace in ansible repo? | 0.970 |

Pairwise probe-score correlations: B-D anti-correlated (-0.83, bash fails often); A-D weak (+0.33); C, E orthogonal to others. Effective independent axes: ~3-4.

### 2.3 Pre-registered gate protocol

| Gate | Threshold | Rationale |
|---|---|---|
| G1 | κ̄ AUROC > 0.65 | discriminates outcomes |
| G2 | gap vs shuffled-κ̄ > 0.10 | not artifact |
| G3 | Mann-Whitney p < 0.05 | distributions differ |
| G4 | EARLY κ_t (first 5 turns) AUROC > 0.60 | true anticipation |

Bonus: κ_t SLOPE over trace × outcome correlation (post-hoc, motivated by Test 1 result).

---

## 3. Results

### 3.1 Trace-level κ̄ separates outcomes

- Mean κ̄ in success traces: 0.371 ± 0.069
- Mean κ̄ in failed traces:  0.350 ± 0.071
- Mann-Whitney p = 0.003
- AUROC of (−κ̄) → failure: 0.677 (vs shuffled baseline 0.501, gap +0.176)

### 3.2 κ_t SLOPE is the headline finding

For each trace, fit linear slope of κ_t(t) over turns t:

| | Success | Failed |
|---|---|---|
| Mean slope | **+0.00368** | +0.00029 |
| Std | 0.0021 | 0.0019 |
| N | 40 | 59 |
| Mann-Whitney p | **0.0003** | |

**Successful traces show clear positive κ_t slope = progressive coherence buildup over turns.**  
**Failed traces show essentially flat κ_t = no buildup.**

### 3.3 EARLY κ_t (anticipation) only weakly significant

EARLY κ_t (mean over first 5 turns of trace) → failure: AUROC 0.572 (just below G4 threshold 0.60). Slope test (above) is stronger anticipation evidence than absolute early value.

### 3.4 Inverse of cardiac uncoupling

Comparison with ICU literature pattern:

| | Cardiology (Morris 2006+) | LLM agents (this work) |
|---|---|---|
| Healthy state | Cross-vital correlated | Cross-probe coherence increasing |
| Failure state | Decorrelation collapse | Absence of coherence buildup |
| Mechanism | Decompensation event | Reasoning deficit |
| Predictive window | Hours before crisis | Buildup detectable from mid-trace |

Same structural pattern (cross-channel correlation as health proxy) but **opposite directionality**: medical failure = collapse from baseline, LLM failure = failure to build up.

### 3.5 Robustness

- GroupKFold by instance_id prevents trace-level leakage
- 3 seed replicates (std < 0.001 across)
- D1 random-feature baseline at κ_t-shuffled level
- Effective 3-4 independent axes (B-D and A-D collinearities checked)

### 3.6 Internal replication v3 (9 axes)

Same-day re-test with expanded probe set: added 4 axes (response_long, early_trace, late_trace, slow_turn) to original 5. Total 9 axes on same residuals.

| Metric | v2 (5 axes) | v3 (9 axes) |
|---|---|---|
| κ_t AUROC | 0.677 | **0.697** |
| Gap vs shuffled | +0.176 | **+0.194** |
| Mann-Whitney p | 0.003 | **0.0009** |
| Slope p | 0.0003 | 0.0033 |
| Success slope | +0.00368 | +0.00251 |
| Failed slope | +0.00029 | +0.00016 |
| Effect ratio (succ/fail) | 12.7× | **15.7×** |

**Pattern replicates. Trace-level discrimination strengthens; slope p weakens (more variance) but effect ratio grows. Coherence buildup is not axis-set-dependent.**

---

## 4. Discussion

### 4.1 Why "buildup not collapse"?

Plausible mechanism: in early turns, model has not yet committed to specific action plan; probes (tool-choice, thinking-length, tool-success) are mostly independent. As trace progresses successfully, model converges on coherent strategy → tool choices align with reasoning length align with successful execution → probes become coupled. Failed traces lack this convergence: model thrashes between strategies, probe outputs remain weakly coupled.

### 4.2 Implications for biological metaphors in interpretability

Importing patterns from biology (cardiac uncoupling, hormone feedback, immune response) is increasingly common in mech-interp. This work cautions: **transfer the structural insight (cross-channel correlation matters), not the directionality** (which depends on the specific failure mode of each system).

### 4.3 Implications for agent monitoring

κ_t can be computed online during agent execution (each new turn updates window). Operationally:
- Initial high κ_t may indicate over-rigid commitment (worth investigating)
- Sustained low κ_t in mid-trace → flag for human review
- κ_t slope > +0.003/turn → high success likelihood (early termination opportunity)

For agent-probe-guard SDK v0.2: ship κ_t as a meta-signal alongside individual probe outputs.

### 4.4 Limitations

1. **Single model (Qwen3.6-27B):** general-phenomenon claim requires multi-model validation
2. **Single benchmark (SWE-bench Pro):** behavior on other agent tasks (web nav, retrieval QA) untested
3. **Probe collinearity:** B-D at -0.83 reduces effective independence; ideal probe set is more orthogonal
4. **N=99 traces:** slope p-value (0.0003) is robust but tighter intervals require scale-up
5. **G4 marginal (0.572 < 0.60):** EARLY anticipation is weak; SLOPE test is post-hoc and needs pre-registered replication

---

## 5. Related Work

### 5.1 Multi-probe ensembles
- Anthropic Cheap Monitors (Apr 2026): 3 variants of single axis
- Apollo Watcher (Feb-May 2026): single LLM-judge
- OpenAI internal (Mar 2026): 99.9% coverage, single classifiers
- Hua et al "Optimally Combining Probe + BB Monitors" (2507.15886): N=2-4 same axis

**None compute cross-probe correlation as meta-signal.**

### 5.2 Cardiac uncoupling
- Morris (2006): cross-vital correlation breakdown predicts ICU decompensation
- 20yr EWS literature: validated mortality predictor in tens of thousands of patients

### 5.3 LLM reasoning quality
- Boppana Reasoning Theater (2603.05488): single-phase reasoning quality probe 87% AUROC
- Chen 2025 (2505.05410): inner-outer divergence documentation, not probed
- Young (2603.26410): thinking vs answer divergence at 55.4%

### 5.4 Our prior work
- agent-probe-guard v0.1: 2-probe ensemble (thinking + capability)
- Paper-5 saturation-direction: epiphenomenal modes for single probes
- Two-Forms paper: distinct epiphenomenal mechanisms

---

## 6. Conclusion

We demonstrate that cross-probe coherence κ_t carries meta-signal about LLM agent reasoning quality. The headline finding — successful traces show progressive κ_t buildup absent in failures (p=0.0003) — establishes a new direction in agent monitoring: meta-signals over independent behavioral probes. The pattern is inverse to its biological inspiration (cardiac uncoupling), highlighting that structural insights transfer better than directional ones across domains.

---

## Appendix A. Reproducibility

- Local runner: `/tmp/run_kappa_t_v2.py` (will move to repo)
- Results: `kappa_t_v2_results.json`
- Phase 6 captures: `openinterp-swebench-harness/swebench_v6_phase6/captures/*.safetensors`
- Phase 6 traces: `openinterp-swebench-harness/swebench_v6_phase6/traces/instance_*.json`
- transformers commit at Phase 6 capture: see Phase 6 metadata

## Appendix B. Walk-back history

5 walked-back probes preceded this finding. Pre-registered gates caught:
1. silent-refusal (non-reproducible across transformers commits, P=3.4e-6)
2. tool-doubt (lexical leak via ok=false strings)
3. tool-doubt position sweep (no semantic signal at any layer/position)
4. Inner-Outer paired (capability redux, paired adds nothing over single)
5. κ_t v1 with trace-level labels (no within-trace dynamics, G4 failed at 0.458)

This v2 was the 6th probe attempt. Pre-registration carried credibility.

---

## TODO before submission

- [ ] Pre-registered slope test on independent dataset (Phase 6c traces?)
- [ ] Multi-model validation (Qwen2.5-7B, ideally Llama-3.1-8B)
- [ ] Layer/position sweep (test κ_t at L31, L55 + alternative positions)
- [ ] Probe orthogonalization (replace B-D collinear pair with truly independent axes)
- [ ] Causal test (does steering κ_t change outcome?) — Phase 2 work
- [ ] Compare with naive ensemble (mean of 5 probes vs κ_t) — does meta-signal beat aggregation?
- [ ] Figures: κ_t trajectory plot (success vs fail), slope distribution, gate diagram
- [ ] Cite Anthropic Cheap Monitors, Apollo Watcher, Hua et al (scoop defense)
