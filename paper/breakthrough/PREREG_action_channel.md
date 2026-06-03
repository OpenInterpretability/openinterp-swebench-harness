# PRE-REGISTRATION — Is the agent's control surface the action channel, not the representation?

**Author:** Caio Vicentino · OpenInterpretability · pre-registered 2026-06-03 (before running the experiment).
**Model/data:** Qwen3.6-27B; the 99 SWE-bench Pro trajectories of the WANDERING arc, reconstructed faithfully at the final-turn tool-choice point (harness-assembly reuse; behavioral fidelity already validated: SUCCESS P(finish) 0.58 ≫ WANDERING 0.075). Compute: one GPU (A100). Notebook `nb_breakthrough_action_channel.ipynb`.

---

## 0. The thesis (a candidate LAW)
Across the WANDERING arc, **detection lives in the representation** (probes/features predict the `finish` decision at AUROC 0.81–0.91) while **every representational intervention fails to cause it** (three residual nulls in paper #2; the verdict-feature #22358 clamp ΔP=−0.001 in paper #5), and the **only lever that works is behavioral** (paper #4: a token-level interruption rescues 30→70%). Hypothesis: for long-horizon agents the *control surface of a decision is the action/token channel, not the internal representation of that decision*. The knowledge-action gap (Basu 2603.18353) is demonstrated only on single-turn QA; nobody has the agentic boundary or the closing condition.

**Framing rule (so both outcomes are positive, not another null):** we *characterize the boundary*. (a) If an internal intervention at the right locus DOES cause `finish`, we have found the first internal lever and *closed the gap on agents* (positive). (b) If NO internal intervention at ANY layer/feature works while the behavioral interruption does, we have shown the gap is *irreducible internally and the action channel is the unique control surface* — a law/impossibility, not a null.

## 1. Experiments

**Obs — where is the decision readable (logit-lens across layers).** Capture the residual at a layer sweep (L3,7,…,63) at the decision point; logit-lens each to the `finish`-vs-{bash,editor} direction; per class. Prediction of the thesis: the finish decision becomes readable LATE and/or is carried by the token context, not consolidated at the mid-layer verdict (L23). (Descriptive, no gate.)

**H1 (causal core) — layer-sweep activation patching.** For WANDERING decision points, REPLACE the last-position residual at each swept layer with the SUCCESS-mean donor residual at that layer, and measure ΔP(finish). Controls: LOCKED-mean donor (null), and a single random-SUCCESS donor (cross-prompt). This is the paper-#2 residual injection, now **swept across all layers at the decision point** — paper #2 tried only L11/L55 and failed.
- **GATE (per the framing rule):** GO-to-positive iff SUCCESS-donor patching at some layer raises P(finish) clearly above the LOCKED-donor null (paired) → an internal lever exists at that layer → gap closable. NO internal lever at any layer (all ≈ LOCKED null) → the law (a): control is not in the residual at the decision position.

**H2 — output-targeted feature steering (was the clamp the wrong method?).** The #22358 clamp used naive value-clamping — the field's weakest method (SAE-TS 2505.20063). Rank SAE features by direct-logit-attribution of their decoder direction to the `finish` logit; steer the top-k UP at the decision point; measure ΔP(finish). Compare to the naive #22358 clamp null.
- **GATE:** if output-targeted steering of *some* feature raises P(finish) > random-active-feature steering → a feature lever exists (the clamp null was a method artifact). If not → the no-internal-lever law strengthens.

## 2. Controls / honesty (the program's discipline)
- Behavioral-fidelity gate first (SUCCESS P(finish) ≫ WANDERING) before any causal number.
- Paired per trajectory; LOCKED-donor + random-feature nulls; report effect sizes, not just p.
- A positive (gap-closing) result is EXTRAORDINARY (it contradicts the arc's nulls) → it must replicate on held-out trajectories AND be confirmed by a real GENERATION that actually emits `finish` (not just a one-token probability bump), never the definitional baseline.
- Truncation to last ~4000 tokens for memory (noted); behavioral fidelity confirms the decision state survives.
- n is modest (≤15/class); a clean null across the full layer sweep is the load-bearing law; a positive is held to the higher bar above.

## 3. Why this is the breakthrough (not another null)
Either outcome is a *positive, field-level claim* about agents that nobody has: the first agentic characterization of the knowledge-action gap's boundary. It converts the arc's five nulls from "weakness" into "the evidence that proves the law." Ship as a tool (a `finish-decision locator` / layer-sweep patcher runnable on any open model) + a MATS/Fellows artifact, not only a paper.

## 4. NOT in scope here (separate tracks)
- **H3 generalization** (does decision-locus predict convertibility across OTHER agent decisions) — follow-up once H1 lands.
- **H4 crypto-agent port** — needs new open-weight crypto-agent trajectory collection; separate notebook. Knife: closed models (Grok/GPT) can't be probed; internals may be redundant with cheap behavioral (run probe-vs-behavioral head-to-head first).

## 5. Artifacts
`scripts/build_nb_breakthrough_action_channel.py` → `notebooks/nb_breakthrough_action_channel.ipynb`. Uses the HF data bundle `caiovicentino1/swebench-phase6-verdict-circuit` (account-independent; sidesteps the Drive). Reuses the verified H4 reconstruction infra.
