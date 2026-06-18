# PRE-REGISTRATION — Mechanistic CoT Faithfulness on a 27B Reasoning Model: Causal or Performative?

**Author:** Caio Vicentino · OpenInterpretability · pre-registered 2026-06-18 (before running).
**Lineage / pivot:** First experiment of the post-control agenda. The arc proved interpretability-based *control* is not adversarially robust (synthesis: "Located, Not Secured"). This pivots to the **non-adversarial auditing regime**, where the verified literature says interp WINS (Mirtaheri & Belkin 2603.17199: internal-state probes beat full-trace CoT monitors). The single open, moat-aligned gap (deep-research, 25/25 claims confirmed): **SAE-feature-based CAUSAL CoT-faithfulness on a large open reasoning model.**
**Model/data:** Qwen3.6-27B (reasoning) + the full-stack SAE (`caiovicentino1/qwen36-27b-sae-fullstack`, 11 layers, d_sae=40960, k=128). One GPU (Colab G4 / RTX6000-Pro 96GB).

**External motivation (2026-06-18):** Google DeepMind, *"Securing the Future of AI Agents"* (blog, Jun 2026), states the frontier position directly: monitoring the *visible* chain-of-thought "will not be enough — models learn to hide their reasoning," and that securing agents "will need to ... potentially **inspect the model's inner workings**" (a future, not-yet-built direction in their black-box→white-box roadmap), with defense escalated by **action severity/reversibility**. This experiment builds exactly the missing piece (mechanistic, SAE-based, causal faithfulness/monitoring of a reasoning model) that their roadmap names as necessary but does not yet provide. The honest scope: map WHERE on the adversarial↔auditing spectrum mechanistic monitoring holds (Mirtaheri & Belkin 2603.17199: internals beat CoT-text monitors in the non-adversarial regime) vs breaks (our COLLAPSE result + obfuscated activations, the insider-threat regime DeepMind flags).

---

## 0. The question (positive-capable, both outcomes informative)
When a reasoning model "thinks" then answers, are the SAE features active **during the CoT** *causal* for the answer (**faithful** reasoning), or is the CoT a post-hoc narrative whose features don't drive the result (**performative** reasoning)? And does the "lever is late" structure mean the causal features concentrate in **late** layers?

**Why it's open (not retrabalho — verified):**
- The ONLY SAE+activation-patching causal faithfulness test (Chen et al. **2507.22928**) used **Pythia-70M/2.8B base** models and found a **scale threshold** (null@70M; CoT-feature swap raises answer log-prob 1.2→4.3 @2.8B). A 27B *native reasoning* model is outside their scope. Their method *is* our causality-protocol (SAE + patching + random-feature baseline).
- Behavioral faithfulness is saturated and *behavioral-only by admission* (Turpin 2305.04388; Anthropic 2505.05410; the 40-author monitorability position paper 2507.11473 explicitly defers mechanistic interp as "requiring further breakthroughs").
- A textual baseline exists on the **same model family**: Young (2603.26410) measures "thinking-answer divergence" on Qwen3.5-27B (19.6%) — but via keyword matching, "not a direct readout of latent computation."

**Both outcomes are a field-level claim:**
- **CAUSAL/FAITHFUL** → first demonstration that CoT features are causal at 27B-reasoning scale; extends the Chen scale curve; the reasoning trace is mechanistically load-bearing in this regime.
- **POST-HOC/PERFORMATIVE** → the CoT is theater here too, with a *mechanistic* signature (the answer is decided by features NOT active during the stated reasoning) — the strong-form version of "performative reasoning," now causal not correlational.

## 1. Design (Tier-A)
Dataset: N reasoning items with a verifiable answer where CoT changes the answer (GSM8K-style / multi-step; filter to items where greedy answer-with-CoT ≠ answer-no-CoT, so the CoT is doing work). Per item:
- **Run-CoT:** model thinks → answer. Capture residual states at the SAE layers across the *thinking* tokens; SAE-encode to features.
- **Run-noCoT:** suppress thinking (force the answer turn directly). Measure logP(correct answer token(s)).
- **Causal patch:** inject the CoT-derived features (decoded back to residual: `xhat = z@W_dec + b_dec`) into the run-noCoT residual at layer L (last pre-answer position). Measure ΔlogP(correct answer).

## 2. Controls (the causality-protocol — the through-line)
- **Random-feature baseline:** patch the same number/magnitude of RANDOM SAE features → ΔlogP_random. Causal only if real ≫ random.
- **Per-layer sweep (the lever-is-late prediction):** run the patch at every SAE layer (15,19,23,27,35,39,43,47,51,59,63). Prediction if faithful: the causal lift concentrates in LATE layers (mirroring #6). A mid-layer or flat profile is itself a finding.
- **Behavioral confirmation:** the patched run must actually GENERATE the CoT answer (greedy), not just raise a logit.
- **Feature naming:** the top causal features → max-activating context → name them (are they "reasoning-result" features? hint-carrying? — connects to Young's thinking-answer divergence).
- **Fidelity gate first:** confirm answer-with-CoT ≠ answer-no-CoT on the filtered set before any causal number.

## 3. Pre-registered criteria (set BEFORE running)
- **FAITHFUL/CAUSAL** iff CoT-feature patch raises logP(answer) above the random-feature baseline by a clear margin at some layer, confirmed by a real generation flip on a meaningful fraction (pre-set ≥30%, mirroring #6's bar).
- **PERFORMATIVE/POST-HOC** iff CoT-feature patch ≈ random-feature baseline (no lift) — the thinking features do not causally carry the result.
- **LEVER-IS-LATE** (secondary): report the per-layer lift profile; faithful-AND-late is the strongest version; faithful-but-early would *contradict* the arc and is a finding.
- **SCALE caveat:** if PERFORMATIVE, compare against Chen's null@70M / causal@2.8B — a 27B-reasoning null would mean reasoning models break the base-model scale trend (a real scientific result, not a failed experiment).

## 4. Differentiator (anti-trap — explicit, the discipline that avoids retrabalho)
- **vs Chen 2507.22928:** 27B *native reasoning* (not Pythia-2.8B base) + per-layer lever-is-late profile.
- **vs Goodfire "Reasoning Theater" (2603.05488):** SAE features **named + causally patched** (not correlational attention probes), and **agent ACTIONS** in Tier-B (not MMLU answers). Do NOT claim "predict answer from internals" as novel — claim the SAE-causal, named, reasoning-scale instantiation.
- **vs Mirtaheri & Belkin (2603.17199):** do NOT claim "internals beat CoT monitoring" as novel — claim the SAE-feature, 27B-reasoning, causal version.
- **vs Young 2603.26410:** causal/mechanistic (not keyword matching), same model family.

## 5. Staging
- **Tier-A (this run):** reasoning tasks, causal feature-patch per layer + random baseline + naming + behavioral flip. Decides causal vs performative + lever-is-late.
- **Tier-B (gated on a clean Tier-A):** real SWE-bench agent traces — do late-layer SAE features predict the committed ACTION before emission, and is the stated-CoT-vs-late-action-channel mismatch a monitoring signal? (the differentiator vs Goodfire/MMLU.)

## 6. Artifacts
`scripts/cot_faithfulness_causal.py` (loads Qwen3.6-27B + the full-stack SAE; reuses the agentguard SAE-autopsy load + the patch/cap ops). Ledger to stdout `OILAB_JSON`. Results → `RESULTS_cot_faithfulness.md`; pre-mint eval → `EVAL_cot_faithfulness.md`. **No DOI mint until the eval gate passes.**

## 7. Riskiest assumption
Causality is scale-dependent (Chen). At 27B it may (a) hold and concentrate late (best case), (b) be null (reasoning-model performative — still a finding), or (c) require reasoning-specific feature selection. The per-layer + random-baseline design distinguishes these; the honest failure mode to avoid is calling a flat/near-random profile "causal."
