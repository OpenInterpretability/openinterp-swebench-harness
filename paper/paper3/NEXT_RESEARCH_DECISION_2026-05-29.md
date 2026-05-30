I have verified everything I need. The code confirms both the feasibility (LayerPatch transient detach is a designed-for use case) and the red-team's sharpest objection (v4 is cache-only; v5 fires late). Now I'll write the decision.

---

# DECISION: Run **C→A sequenced**, with a hard pre-registration gate between them

**The most asymmetric next step is the sequenced pair C-then-A: first build the rescuability predictor from owned data (C, ~$0), then run the closed-loop detect→*select*→intervene experiment (A, ~$15–30) — but only if C clears a pre-registered go/no-go bar.** This is not a hedge. It is the only ordering that survives the red-team, and the code I just read proves the sequencing is *necessary*, not merely prudent.

---

## 1. The choice, named

**Phase 0 (C): Rescuability predictor.** On the existing N=20 L11 α-sweep runs + the N=99 multi-channel feature set, fit a leave-one-out model predicting `rescued ∈ {0,1}` (and, where data allows, the rescuing α-window) from pre-intervention mechanistic features (L11_drift_first_last, L11_cosine_consec_late, tool-entropy slope, L43 freeze). Pure analysis on owned data.

**Phase 1 (A): Closed-loop, conditioned on the trigger that actually streams.** Wrap the agent loop so that an *online* detector fires mid-trajectory and attaches L11 SUCCESS-donor injection transiently (then detaches), at the per-instance α the Phase-0 model selects — versus an always-on arm and a no-hook baseline, all freshly run on the same RTX 6000.

The judges picked A-alone (six 28–30 scorecards). The red-team's contrarian case — that **C→A dominates A-alone on cost, compounding, and null-robustness** — is correct, and I can now defend *why* from the source code, not just from the abstract argument.

---

## 2. Why this is the most asymmetric — and the honest red-team reckoning

**Unscoopability: 5/5, unchanged.** Both phases require the LayerTap+LayerPatch+agent-loop trifecta plus precomputed `exp_l11_donors.safetensors` (I confirmed the file exists) and the N=99 residual dataset. No external group can run either phase. The novelty map's cleanest "novel" verdict (closed-loop detection→activation-injection on code agents, heterogeneity-aware) stands; every near-prior (Intervention Paradox 2602.03338, StepShield 2601.22136, AgentForesight 2605.08715) is a *text*-based critic.

**Compounding: 5/5.** This single move operationalizes all three papers — Paper #1's detectors, Paper #2's L11 donors + α-windows, Paper #3's heterogeneity signature — into one deployable primitive. C *is* the missing connective tissue: it turns Paper #3's "we have a signature" into "the signature selects who to rescue," which is exactly what A needs as its trigger gate.

**Now the red-team objection, addressed honestly — because the code makes it concrete, not hand-wavy:**

The hostile reviewer's strongest point is **"the trigger is load-bearing and the only streamable detector fires late."** I verified this in `detectors.py`. `v4_cross_layer` returns `"v4 cache miss"` whenever the precomputed entry is absent (line 115–117) — it is **offline-cache-only**. `v5_tool_entropy` *is* online-computable (line 91–92, a rolling function over the turns list), but its fire-turn is `len(turns) - window` (line 94): it fires at the *start of the low-entropy window near the trajectory end*. So if you ship A naively with v5 as the live trigger, **you fire late — exactly when always-on already works — and you have thrown away the 15-turn lead that is the entire thesis.** The red-team is right, and the judges who waved this away ("only v4-online needs modest wiring") underweighted it.

This is precisely why C goes first and why A's trigger must be built, not assumed:
- **C de-risks the GPU spend.** If C shows the Paper #3 signature does *not* predict rescuability above chance on the N=20, then A's "select" step is impossible and you would be triggering blind on resistant agents at the wrong α — regressing to the always-on null, producing a *muddy* result indistinguishable from "timing doesn't help." C tells you this for $0 before you rent a GPU.
- **C forces you to build a real online trigger.** The streamable signal must be **an incremental v4** (cross-layer probe-score range computed per-turn from the residuals the loop already captures via `tap` — the feature builders `extract_multi_channel_features.py` and `early_warning_v4_cross_layer.py` exist to copy the math from) **gated by the C predictor**, *not* raw v5. If incremental-v4 cannot reproduce the offline lead online, that is itself a finding (and a Phase-0 stopping point), not a silent failure buried in A.

**On the red-team's N-is-fatal point (5/20, 5 events):** real and unfixable by cleverness. The mitigation is design discipline, not more stats theater: (a) pre-register **rescue-rate AND crash-rate jointly**, so "same rescue, fewer crashes" counts as a clean positive (this is the most *likely* true effect of surgical-vs-always-on and it does not need 5→10 events to detect — destabilization at high α was 60%, a large, well-powered contrast); (b) treat the rescuability-prediction LOO-AUC in C as the *primary* powered endpoint, with A's closed-loop rate as confirmatory. The headline migrates from "surgical beats always-on" (underpowered) to "rescuability is mechanistically predictable, and prediction-gated surgical injection reduces destabilization at equal rescue" (powered on the crash contrast).

**On WANDERING not being hardware-stable (your own RTX-6000 0/20):** this is a *confound the design must absorb*, addressed below in the baseline.

---

## 3. Concrete experiment + pre-registered decision rule

**PHASE 0 — C (do first, ~$0, 1–2 days analysis):**
- **Data:** the N=20 L11 α-sweep outcomes (rescued/not, rescuing-α where known) + the N=99 multi-channel features already extracted.
- **Model:** LOO logistic regression / shallow tree predicting `rescued`, features = {L11_drift_first_last, L11_cosine_consec_late, tool_entropy slope, L43_cosine_consec_late}. Report LOO-AUC + permutation-null p.
- **GO/NO-GO (pre-registered):** **GO to Phase 1 iff LOO-AUC ≥ 0.70 with permutation p < 0.10.** If AUC < 0.70 → **STOP and publish C as the honest result**: "WANDERING rescuability is *not* predictable from the Paper #3 signature; heterogeneity is structural, not signature-legible." That is a clean, citable Paper #2/#3-unifying null that resolves the open A-vs-B question *without spending GPU*.

**PHASE 1 — A (only if Phase 0 GO; ~$15–30 on RTX 6000):**
- **Subjects:** the 20 WANDERING instances, re-run fresh on the *same RTX 6000* (the H100/35% baseline cannot be reused — your own hardware-stability finding).
- **Arms (within-instance where determinism allows):**
  1. **No-hook baseline** (establishes the RTX-6000-native WANDERING rate; absorbs the hardware confound — *do not* compare against the H100 number).
  2. **Always-on L11 @ per-instance α** (Paper #2 replication).
  3. **Closed-loop:** incremental-v4 trigger (gated by the C predictor) → transient L11 attach at fire-turn → detach after a pre-set decode budget (the LayerPatch docstring's pattern (b)), at the C-selected α.
- **Pre-registered primary readout (joint):** crash-rate (invalid_tools/max_turns destabilization) AND rescue-rate (finish_tool). Decision rule, committed before running:
  - **Closed-loop crash-rate < always-on crash-rate (one-sided, the powered contrast) at equal-or-better rescue → POSITIVE: surgical timing escapes high-α destabilization.** Flagship loop-closure result.
  - **Closed-loop rescue > both baselines → STRONG POSITIVE** (timing-artifact hypothesis B confirmed).
  - **Closed-loop ≈ always-on on both → NULL, but a *clean* one** (because C already established the trigger fires correctly and selects rescuable instances): "heterogeneity is structural; activation steering hits the Intervention Paradox too." Citable against 2602.03338.
- **Controls:** the existing random-direction donor (direction-specificity), already in the harness.

---

## 4. How it threads the openinterp.org narrative

The blog arc becomes the **canonical four-beat monitorability story** — the cleanest legible version of the whole thread:

1. **DETECT** — "We can see an agent start to wander 15 turns before it fails" (Paper #1, v4/v5).
2. **UNDERSTAND** — "The wandering lives in a specific edge layer (L11), and we can read off *which* wanderers are salvageable" (Paper #3 + C).
3. **INTERVENE** — "We inject a SUCCESS direction into L11 *only when the alarm fires, only into the agents our signature says are rescuable* — and the agent recovers without destabilizing" (Paper #2 + A).
4. **CLOSE THE LOOP** — "Detection → selection → surgical intervention → recovery: a deployable agent-safety primitive on a custom harness no one else has."

Title candidate: *"Closing the Monitorability Loop: Detection-Triggered Activation Steering Rescues Wandering Code Agents."* For the **alignment/AISI audience** this is the textbook mechanistic-monitorability deliverable (probe detects → intervene → recover, with the Steering-Awareness 2511.21399 threat model as a built-in robustness note) — a direct LTFF/OpenPhil/Inspect-Evals hook. For the **crypto/AgentGuard audience** the framing is "we don't just *flag* a rogue trading/tool-calling agent, we *catch and correct* it mid-run" — exactly the $245M-exploit pain point. Distribution per your constraints: openinterp.org/blog → X @0xCVYH → HF Community (LessWrong banned, arXiv→Zenodo only). The C-first ordering even *strengthens* the post: "we predicted who was rescuable before we touched the GPU" is a more rigorous, more honest narrative than "we steered everything and hoped."

---

## 5. What NOT to do now, and why

- **Do NOT run A blind/A-alone (the judges' favorite as stated).** The code proves the live trigger does not exist yet (v4 cache-only; v5 fires late). A-without-C-and-without-incremental-v4 risks a muddy null that cannot separate "timing doesn't help" from "wrong trigger / wrong instances / wrong α." Sequencing converts that risk into a $0 pre-check.
- **Defer B (cross-model L11 universality).** Cost is HIGH and you lack residuals for Llama-70B/GPT-5 — pure re-capture cost, and universality is unmotivated until the single-model loop closes. Natural follow-on paper, not now.
- **Defer D (SAE on L11).** Most crowded niche (Qwen-Scope 2605.11887, Beyond-the-Black-Box already in-space), most scoopable, only indirectly advances the loop. It is a *mechanism-explanation* nice-to-have, not the asymmetric play.
- **Defer E (streaming Inspect eval).** Highest AISI distribution value but strictly downstream: you cannot benchmark a closed loop before you have proven one works. E is A's productized follow-on — build it once A returns a positive (or a clean null worth benchmarking against).
- **Defer F (scale-up).** Incremental; adds N but no new asymmetry. The N=20 limitation is real but is mitigated in A by the *joint crash+rescue* readout and by making C's powered AUC the primary endpoint — not by grinding more WANDERING traces first.

**Bottom line:** Run **C now (free, ~2 days)**. If LOO-AUC ≥ 0.70 / p < 0.10, run **A (~$15–30)** with the incremental-v4 trigger, joint crash+rescue pre-registration, and a fresh RTX-6000 baseline. Either way you publish: a powered rescuability-predictor result, plus *either* the flagship loop-closure *or* a clean, Intervention-Paradox-corroborating null. That sequence is more asymmetric than A-alone on every axis the red-team named — cost, null-robustness, and compounding — and the harness code confirms it is a wrapper, not a refactor.

**Relevant verified files:** `/Volumes/SSD Major/fish/openinterp-swebench-harness/instrumentation/layerpatch.py` (transient attach/detach designed-for, lines 35–39, 79–91), `/Volumes/SSD Major/fish/inspect-tool-entropy-collapse/src/tool_entropy_collapse/detectors.py` (v4 cache-only line 115; v5 online but late line 91–94), `/Volumes/SSD Major/fish/openinterp-swebench-harness/paper_mega_v4_pairs/exp_l11_donors.safetensors` (donors present), `/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/extract_multi_channel_features.py` + `early_warning_v4_cross_layer.py` (the math to copy for incremental-v4 and the C feature set), `/Volumes/SSD Major/fish/openinterp-swebench-harness/agent/loop.py` (per-turn loop to wrap).