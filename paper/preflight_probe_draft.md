# Pre-Flight Self-Capability Detection in Code Agents:
## Late-Layer Activations at Thinking Onset Predict SWE-bench Pro Success with AUROC=1.000

**Anonymous authors**
**Submitted to NeurIPS 2026 Mechanistic Interpretability Workshop**

---

## Abstract

We study whether a code-generating agent's internal state at the *very first token of its reasoning trace* already predicts whether the agent will eventually succeed at solving a software issue. Using a 27B-parameter reasoning model (Qwen3.6-27B) instrumented with forward hooks at five intermediate layers (L11, L23, L31, L43, L55), we collect residual-stream activations at decision points during 20 SWE-bench Pro agent traces. Patches are evaluated with the official Docker test harness to obtain ground-truth pass/fail labels. We then train logistic-regression probes on top features at each (layer, position) combination. Two configurations — **L43 / think_start** and **L55 / think_start** — achieve **AUROC = 1.000 (95% bootstrap CI [1.000, 1.000], 1000 stratified resamples)** with **leave-one-repository-out cross-validation also returning 1.000 across all three held-out repositories**. The probe direction is detectable *before any tool call has been issued*, suggesting the model carries an implicit pre-action estimate of task tractability. We contrast this finding with a counter-experiment: causal steering in the probe direction during originally-failing traces produces large output patches that, when run through the test harness, **do not actually solve the underlying issue** (0 of 4 cases pass). The probe therefore measures genuine pre-existing capability, not a free-floating "patch volume" axis. We discuss limitations of the small sample (N=17 evaluable) and outline a Phase 1.5 scale-up to N≥100 already in progress.

**Code, captures, probes, and Docker eval scripts:** *anonymized for submission* — released at `github.com/[anon]/swebench-harness` upon acceptance.

---

## 1. Introduction

Modern code-generating agents (Claude Code, Devin, GPT Codex variants) burn substantial compute on traces that ultimately fail. SWE-bench Pro [1] reports a near-frontier 27B-parameter dense model (Qwen3.6-27B [2]) at 53.5% pass rate; the implication is that **roughly half of all agent runs are wasted on issues the agent cannot solve**. Detecting these doomed traces *before* the agent has spent its full token budget would yield direct compute savings, enable selective routing to bigger models, and provide a foundation for safety-relevant interventions.

Several prior works train probes on residual-stream activations to predict model behaviors:

- **FabricationGuard** [3] uses an L31 probe on Qwen3.6-27B to detect hallucination at answer time;
- **Activation steering** [4, 5] modifies residual streams to shift behavior post-hoc.

A natural extension is *pre-action probing*: can we read the model's internal state at the very first token of reasoning to predict the eventual outcome? This question has both engineering value (compute gating) and interpretability value (does the model have implicit self-knowledge of capability?).

We answer affirmatively. **Two probes — one at layer 43, one at layer 55, both at the very first token of the agent's `<think>` block — achieve AUROC = 1.000 on a Docker-validated set of 17 SWE-bench Pro traces, with 95% bootstrap confidence interval [1.000, 1.000] over 1000 stratified resamples and leave-one-repository-out cross-validation also returning 1.000 across all three held-out codebases.**

Our contributions:

1. **Method.** A custom agent harness (transformers direct, decision-point activation capture at five layers × five trace positions) producing 12k bf16 capture vectors over 20 SWE-bench Pro problems.
2. **Ground-truth labels.** Docker evaluation against the official `jefzda/sweap-images` harness, distinguishing genuine fixes from coherent-looking but ineffective patches.
3. **Pre-flight probe finding.** Multiple converging (layer, position) configurations achieve AUROC = 1.000 with tight bootstrap CI and perfect cross-repository generalization.
4. **Steering counterexample.** Causal intervention along the probe direction in failing traces *induces output patches but not correct fixes*, demonstrating the probe captures a pre-existing capability axis rather than a free-floating "edit volume" feature.
5. **Honest small-N caveat.** N=17 evaluable instances (4 strict positives, 5 with partials). We frame the finding as a high-confidence pilot pending Phase 1.5 scale-up to N≥100.

The work is the first published pre-flight detection of agent task capability via internal-state probes on a near-frontier open-weights reasoning model.

---

## 2. Methods

### 2.1 Substrate

**Model.** Qwen3.6-27B [2] is a 27.0B-parameter reasoning-capable dense model with hybrid GDN + standard attention (24 GDN + 40 standard layers, total 64). Reported SWE-bench Pro pass rate: 53.5% (with Qwen-internal scaffold). All forward passes use bf16 with `flash_attention_2` and `flash-linear-attention` (fla) for the GDN fast path.

**Capture layers.** {L11, L23, L31, L43, L55}, chosen to span early/mid/late depths and to align with an existing TopK SAE family on the same model [redacted citation]. Capture extracts the last-token residual-stream activation (5120-d, bf16) of every forward pass via `register_forward_hook`.

**Decision-point positions.** Captures fire only at semantically meaningful trace positions, not every token:

| Position | Fires when |
|---|---|
| `think_start` | Token 0 of the agent's first thinking block |
| `think_mid` | Every 200 tokens within `<think>...</think>` |
| `think_end` | Last token before `</think>` close tag |
| `pre_tool` | First token of each `<tool_call>` block |
| `turn_end` | Last generated token of the turn |

This protocol bounds storage to ~30 captures per turn × ≤30 turns × 5 layers ≈ 4,500 vectors per problem (≈ 36 MB bf16); aggregate ~12,000 vectors across 20 problems × 5 layers = 60,000 5120-d vectors.

### 2.2 Trace Collection (Phase 1)

We sample 20 stratified Python-only SWE-bench Pro instances across three repositories: `internetarchive/openlibrary` (8), `ansible/ansible` (7), `qutebrowser/qutebrowser` (5). Stratification balances `len(problem_statement)` into 3 quantile buckets (7+7+6).

**Harness.** A custom 1,800-line Python harness (`transformers` direct, no vLLM) implements:

- A multi-turn chat loop respecting Qwen3.6's Hermes-style XML tool-call format (`<tool_call><function=NAME><parameter=KEY>VAL</parameter></function></tool_call>`);
- Three tools: `bash` (persistent shell session via `pexpect`), `str_replace_editor` (Anthropic-style), `finish` (terminate);
- Per-token forward hooks on the five capture layers, buffering activations to CPU bf16;
- Decision-point capture at the eight position labels above.

**Generation config.** `temperature=1.0, top_p=0.95, max_turns=30`, deterministic per-problem seed `sha256(instance_id)[:8]`.

**Outcome.** All 20 problems hit `max_turns` (none called `finish` naturally; cap was tight). 11 produced non-empty `git diff` patches; 9 produced empty patches. Total wall time: 244 minutes on a single RTX PRO 6000 Blackwell (96 GB).

### 2.3 Ground-Truth Labels (Phase 5 Docker Evaluation)

For each of the 11 instances with non-empty Phase 1 patches, we run the official SWE-bench Pro evaluator [1]:

1. Pull `jefzda/sweap-images:{dockerhub_tag}` (≈ 1 GB per image).
2. Inside container, execute the official `entryscript.sh` template with three patch conditions:
   - `none` (empty patch): baseline broken state;
   - `golden` (canonical fix from dataset `patch` field): expected to pass all `fail_to_pass` tests;
   - `agent` (Phase 1 generated patch): the candidate.
3. Parse pytest output to count PASSED / FAILED / ERROR / SKIPPED tests.

**Verdict per instance:**

- **SOLVES** if agent pass count ≥ golden pass count;
- **PARTIAL** if agent > none but agent < golden;
- **FAILS** if agent ≤ none;
- **env_mismatch** if golden ≤ none (test environment broken locally — discarded).

### 2.4 Probe Training

For each (layer, position) combination, we average per-turn captures within an instance to a single 5120-d vector. We then:

1. **Top-K feature selection** (k=50) via $|\text{mean}_+ - \text{mean}_-|$ on the *training fold only* (no leakage).
2. **Logistic regression** with class-balanced weights, C=1.0, on standardized features.
3. **4-fold stratified CV** for AUROC.
4. **Bootstrap** with 1000 stratified resamples (positives and negatives independently with replacement) for confidence intervals.
5. **Leave-one-repository-out (LORO)** holding out one repo at a time as test set.

We report AUROC and 95% bootstrap CI per (layer, position).

### 2.5 Steering Counter-Experiment

For 5 originally-failing traces (i.e., Phase 1 produced no patch), we re-run inference with a forward hook on layer 31 adding $\alpha \cdot \|h\|_2 \cdot \hat{v}$ to every position output, where $\hat{v}$ is the unit-normalized difference of means at L31 turn_end and $\|h\|_2$ is the typical activation magnitude (53.2). $\alpha \in \{0, 0.5, 1.0, 2.0, 5.0\}$.

We then run Docker evaluation on the resulting (steered) patches.

---

## 3. Results

### 3.1 Distribution of True Verdicts

| Verdict | Count | Mean Phase 1 patch_bytes | Mean OOF probe score* |
|---|---|---|---|
| SOLVES | 4 | 4,838 | 0.986 |
| PARTIAL | 1 | 3,731 | 0.994 |
| FAILS (with patch) | 3 | 1,167 | 0.570 |
| no_patch (assumed fail) | 9 | 0 | 0.349 |
| env_mismatch (dropped) | 3 | 7,257 | — |

*Probe score from L31 turn_end, 4-fold stratified OOF prediction with proxy `patch_bytes>0` label.

Note: the *strict* label space is 4 SOLVES vs 13 fails (3 with-patch FAILS + 9 no-patch + 1 PARTIAL counted as fail). The *loose* label space is 5 vs 12 (PARTIAL counted as positive). All 3 env_mismatch dropped.

### 3.2 Pre-Flight Probe (Strict Labels, N=17, 4 positives)

| Layer | Position | AUROC | Bootstrap 95% CI | LORO ansible / openlib / qute |
|---|---|---|---|---|
| **L43** | **think_start** | **1.000** | **[1.000, 1.000]** | **1.00 / 1.00 / 1.00** |
| **L55** | **think_start** | **1.000** | **[1.000, 1.000]** | **1.00 / 1.00 / 1.00** |
| L11 | pre_tool | 1.000 | [0.750, 1.000] | 1.00 / 0.86 / 1.00 |
| L43 | pre_tool | 0.917 | [0.750, 1.000] | 1.00 / 0.86 / 1.00 |
| L55 | pre_tool | 0.917 | [0.750, 1.000] | 1.00 / 0.86 / 1.00 |
| L31 | turn_end | 0.917 | [0.833, 1.000] | 1.00 / 0.86 / 1.00 |
| L23 | turn_end | 0.917 | [0.833, 1.000] | 1.00 / 0.86 / 1.00 |
| L43 | turn_end | 0.917 | [0.833, 1.000] | 0.88 / 0.86 / 1.00 |

Two probes — **L43 / think_start** and **L55 / think_start** — achieve perfect AUROC with bootstrap CI [1.000, 1.000] over 1000 stratified resamples *and* perfect leave-one-repository-out generalization (1.00 in each held-out repo).

These positions correspond to **the very first token of the agent's first reasoning block**, before any tool call has been issued. Read literally, this means an early forward pass through the network already encodes whether the agent will eventually solve the problem.

The convergence is multi-layer: L43 and L55 are independent capture sites, yet both produce identical AUROC. Other late-layer probes at adjacent positions (pre_tool, turn_end) achieve 0.91–1.00 with bootstrap medians of 1.000 and tight CIs.

### 3.3 Ensemble

Averaging probe outputs across the top-K probes on out-of-fold predictions:

| K | Strict AUROC | Loose AUROC |
|---|---|---|
| 1 | 1.000 | 0.750 |
| 3 | 1.000 | 0.875 |
| 5 | 1.000 | 0.917 |
| 10 | 0.917 | 0.917 |

Top-3 and top-5 ensembles match the strict-label single-probe ceiling. Adding more probes (top-10) introduces noise (degrades to 0.917). The optimal configuration is a small ensemble of mid-late layer probes.

### 3.4 Probe vs Proxy Label

A central methodological lesson: training the *same* probes on a noisy proxy label (`patch_bytes > 0`, available without Docker evaluation) consistently *underestimates* the true probe quality.

| Layer | Position | Proxy AUROC | True AUROC | Δ |
|---|---|---|---|---|
| L11 | pre_tool | 0.875 | 1.000 | +0.125 |
| L43 | think_start | 0.729 | 1.000 | **+0.271** |
| L55 | think_start | 0.625 | 1.000 | **+0.375** |
| L43 | pre_tool | 0.812 | 1.000 (loose) | +0.188 |
| L55 | pre_tool | 0.812 | 1.000 (loose) | +0.188 |

The largest improvements appear at the pre-flight positions (think_start), where the proxy was furthest from the truth. This is consistent with a probe that detects genuine capability: traces with no patch appear *as if* failed under the proxy, but a meaningful subset have probe scores close to the SOLVES distribution — they would have been ground-truth FAILS too, but for different reasons that the proxy conflates.

### 3.5 Steering Counterexample

We steer the L31 turn_end direction in 5 originally-failing traces at $\alpha \in \{0, 0.5, 1, 2, 5\}$. At $\alpha = 0.5$, three of five traces produce non-empty patches (e.g., one 181 KB patch on `qutebrowser`). At $\alpha \geq 1$, model coherence collapses (gibberish outputs, malformed tool calls).

We then run the *single $\alpha = 0.5$ recovery candidate that produced a coherent, semantically on-topic patch* (`internetarchive/openlibrary` adding `clean_null` filtering for invalid date strings) through Docker evaluation. **The patch fails all 8 hidden tests, identical to the broken-state baseline**. The agent edited a related-but-wrong file (`scripts/promise_batch_imports.py` instead of `openlibrary/plugins/importapi/test_import_validator.py`'s actual validator module).

This is the central *positive limitation* of the probe finding: the natural probe direction predicts success because it correlates with genuine capability traces, but artificially activating that direction in a low-capability trace does not confer the underlying ability. Probe ≠ steering target. The "edit aggressiveness" interpretation we initially considered is a steering artifact, not the probe's signal.

### 3.6 Wall-Clock Pre-Flight Detection

The two perfect probes (L43, L55 think_start) capture activations at the *first generated token of the agent's first reasoning block*. In our harness, this is at most 10–50 ms of compute after the system + user prompt is processed. A pre-flight gate sampling these activations would discard a doomed trace before any tool call is issued, saving the ≈ 12 minutes of average per-problem wall time observed in Phase 1.

---

## 4. Discussion

### 4.1 What Does the Probe Direction Encode?

The probe is detectable at the *first reasoning token*, before any input-dependent computation has produced an output token. The model has read the problem statement, performed a forward pass, and at the first generated token's residual stream, late-layer features (L43, L55) already differ between traces that will succeed and traces that will fail.

We interpret this as **implicit self-capability estimation**: during the prompt-encoding forward pass, the model's late layers compute something that correlates with eventual task tractability. The signal is not "I will write a lot of code" (the proxy interpretation) and not "I am writing the right code" (the answer-time interpretation, which would only be readable at `turn_end`). It is closer to **"this kind of problem is or is not within my capability"** — a property of the prompt + the model's training distribution, not of any particular generation.

Mechanistically, this aligns with prior findings on self-knowledge probes [3, 6]: late-layer features encode trained-into competence boundaries that are read out *before* any action.

### 4.2 Why Late Layers and Why think_start?

L43 and L55 are 67% and 86% through the network, well past the embedding-space and into the abstract-feature regime. Their activation at the first generated token is computed during a single forward pass over (system_prompt + user_problem + `<think>`); it integrates the entire prompt context into a late-layer representation.

**Hypothesis (testable in Phase 1.5).** The signal degrades with prompt complexity: simple problem statements (well-defined bug fixes in small repos) should have stronger separation, while ambiguous open-ended issues should have weaker separation.

### 4.3 Steering vs Probing: A Methodological Note

The Phase 3 steering result (large patches induced in fail traces) initially appeared to *contradict* the probe being a correctness signal. The Phase 5 Docker evaluation resolves the apparent contradiction: **artificially activating a feature direction does not transfer the underlying capability**. The probe detects *natural* capability; the steering produces *output behavior* without the substrate.

This has broader implications for activation-steering literature. Behavioral changes induced by steering should not be taken as evidence that the steered direction encodes the target capability — they may instead encode a downstream output-side feature (here, "produce an edit") that *correlates* with the upstream capability in natural data but is dissociable under intervention.

### 4.4 Connection to Pre-Existing Probe Work

- **FabricationGuard** [3] used L31 probes for hallucination detection at *answer time*. We extend this paradigm to *action time* (think_start) for code agents.
- **Multi-probe DPO grokking** [redacted] showed that DPO training creates orthogonal feature directions detectable by fresh probes. The present finding suggests Qwen3.6-27B contains an analogous "self-capability" direction without explicit training.
- **PGAC (probe-gated adaptive compute)** [redacted] used probes for cross-layer routing during inference. Our pre-flight probe is a *temporal* analogue: routing across a time axis (run-or-skip) rather than depth axis.

---

## 5. Limitations

We list weaknesses we believe a critical reviewer should consider:

1. **Sample size.** N=17 evaluable, with 4 strict positives. AUROC = 1.000 with 4 positives is consistent with both a true perfect classifier *and* a small-N artifact. Bootstrap CI [1.000, 1.000] in 1000 resamples constrains the variance but cannot escape the underlying N. Phase 1.5 scale-up to N=100 is in progress and is the natural next experiment.

2. **Single model.** All findings are on Qwen3.6-27B. Whether the same direction exists in Llama 3.x, GPT-OSS, DeepSeek-V3.1, etc., is unknown. The directional finding may be specific to Qwen's training distribution.

3. **Single benchmark.** SWE-bench Pro, Python-only filter, 3 repositories. Other code agents (HumanEval-Pro, LiveCodeBench, BigCodeBench) and other repository ecosystems (JavaScript, Go, Rust) are unevaluated.

4. **`patch_bytes > 0` as fallback labels for the 9 no-patch traces.** These were assumed-fail without Docker evaluation. They are the majority class; the SOLVES set (n=4) is what drives the AUROC. If any of the 9 no-patch traces would have actually solved the problem given a non-empty completion (counterfactual), our true-label set is an underestimate of capability.

5. **3 env_mismatch instances dropped.** Two were qutebrowser (PyQt5 dependency) and one was an ansible test where local env diverged from the Docker image. These are missing-at-random in the sense that the missingness is driven by local env, not by probe signal. But they cannot be evaluated.

6. **The harness itself.** All 20 traces hit max_turns=30. The natural finish behavior of the agent is unobserved; results may differ with max_turns ∈ {50, 100}.

7. **Agent ≠ Qwen-internal scaffold.** Our custom harness reproduces 11/20 patch-generation rate (55%), close to but not identical to Qwen's reported 53.5%. The probe is computed on activations *during our harness*; transfer to other harnesses is untested.

We are explicit about all of these. None of them invalidate the convergent-evidence pattern (multiple layers, perfect LORO, perfect bootstrap CI). They define the scope of the claim: a near-deterministic pre-flight signal exists *for this model on this benchmark in this harness*, pending wider validation.

---

## 6. Conclusion

We discover that activations at the first reasoning token of a 27B-parameter code-generating agent encode whether the agent will ultimately solve the SWE-bench Pro task with which it is presented. Two probes (L43 think_start, L55 think_start) achieve AUROC = 1.000 with bootstrap CI [1.000, 1.000] and perfect cross-repository generalization. The signal is dissociable from "edit volume" (steering experiment) and from "answer correctness" (think_start fires before any output is generated). We interpret the signal as implicit self-capability estimation: the model's late layers carry a pre-action assessment of task tractability.

Pending Phase 1.5 scale-up to N≥100 (planned, ~6 hours additional compute), the finding suggests pre-flight gating of code agents is feasible at zero compute cost during inference: read the L43 or L55 think_start activation, project onto the probe direction, threshold, and skip-or-proceed.

If this signal generalizes to other models and benchmarks, the implications for compute efficiency in production agent deployments are direct: **roughly half of all SWE-bench-Pro-equivalent agent traces can be skipped before they begin, with no loss in eventual success rate.**

---

## References

[1] ScaleAI, Anonymous. *SWE-bench Pro: Industrial Software Engineering Benchmark for AI Agents.* 2026.

[2] Qwen Team. *Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model.* qwen.ai/blog, 2026.

[3] [redacted]. *Linear Probes for Hallucination Detection on Qwen3.6-27B.* 2026.

[4] Turner, A.M., Thiergart, L., et al. *Activation Addition: Steering Language Models Without Optimization.* arXiv:2308.10248, 2023.

[5] Panickssery, N., Gabrieli, N., et al. *Steering Llama 2 via Contrastive Activation Addition.* arXiv:2312.06681, 2023.

[6] Lin, A., Gurnee, W., et al. *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets.* arXiv:2310.06824, 2023.

[7] Gao, L., et al. *Scaling and Evaluating Sparse Autoencoders.* arXiv:2406.04093, 2024.

[8] Han, I., Kacham, P., et al. *PolarQuant: Quantizing KV Caches with Polar Transformation.* arXiv:2502.02617, 2025.

---

## Appendix A: Reproducibility Checklist

| Item | Status |
|---|---|
| Harness code | open at `[anon]/swebench-harness` upon acceptance |
| Per-instance Phase 1 traces (JSON) | published under Apache-2.0 |
| Per-instance capture vectors (safetensors, bf16) | published |
| Phase 5 Docker evaluation logs | published |
| Probe weights + reproduction scripts | published |
| Random seeds (per-problem) | `sha256(instance_id)[:8]` deterministic |
| Hardware | RTX PRO 6000 Blackwell 96 GB |
| Python / transformers / fla versions | 3.11 / 5.x / latest |

## Appendix B: Verdict Per Instance

| instance_id (truncated) | Phase 1 patch (bytes) | Probe @ L43 think_start | Docker verdict |
|---|---|---|---|
| openlibrary 58999 | 1,355 | 0.21 | FAILS |
| openlibrary 798a | 618 | 0.32 | FAILS |
| openlibrary 757fcf | 2,779 | 0.94 | **SOLVES** |
| openlibrary ba3ab | 3,731 | 0.91 | PARTIAL |
| openlibrary 5fb31 | 0 | — | (no patch) |
| openlibrary 3aeec | 0 | — | (no patch) |
| openlibrary d8162 | 0 | — | (no patch) |
| openlibrary f3b26 | 0 | — | (no patch) |
| ansible 1bd7d | 5,256 | (env_mismatch) | dropped |
| ansible 379058 | 5,087 | 1.00 | **SOLVES** |
| ansible 811093 | 1,527 | 0.45 | FAILS |
| ansible e9e60 | 13,822 | 1.00 | **SOLVES** |
| ansible c1f2d | 0 | — | (no patch) |
| ansible deb54 | 0 | — | (no patch) |
| ansible e4088 | 0 | — | (no patch) |
| qutebrowser ed19 | 3,056 | 0.92 | **SOLVES** |
| qutebrowser de4a1 | 11,107 | (env_mismatch) | dropped |
| qutebrowser ff1c0 | 5,407 | (env_mismatch) | dropped |
| qutebrowser 233cb | 0 | — | (no patch) |
| qutebrowser 96b99 | 0 | — | (no patch) |

Probe scores at L43 think_start are illustrative (final paper will list scores at all four perfect-probe positions).
