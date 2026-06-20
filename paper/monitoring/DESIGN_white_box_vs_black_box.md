# FLAGSHIP ① — Design (pre-registration grade) — 2026-06-19

**Working title:** *The White-Box Edge Is Conditional: Predicting When Internals Beat Text for Monitoring a
Reasoning Agent's Irreversible Actions.*

**One-sentence thesis (the law, not a yes/no):** On a reasoning agent, the advantage of white-box (probe/SAE)
monitoring over black-box (text) monitoring of an imminent irreversible wrong action is **not uniform and not
mainly about cost** — it is **conditional on chain-of-thought faithfulness**: internals win exactly when the CoT
is performative/unverbalized, are blind in the sincere-error (felt≠granted) regime where an external
task-grounded check wins, and are fragile under adaptive optimization. We map this boundary **causally** and give
the operating rule for which monitor to deploy when.

This converts the three arc results into ONE head-to-head with a predictive law:
- **The Late Channel** (white-box WINS where CoT is unfaithful) — DOI 20752896/20768084.
- **Felt, Not Granted** (white-box BLIND in sincere model-origin error; external check wins) — DOI 20685264.
- **Located, Not Secured / COLLAPSE** (white-box FRAGILE adversarially) — DOI 20764858/20768085.

## Why this is novel (post-scoop, 2026-06-19)
The field has CONVERGED on "monitoring needs 3 tiers (answer-text / thinking-text / activations)" — so bare
"complementarity" is NOT novel. Our wedge, intact after scoping:
1. **A predictive LAW**: the black-to-white boost is a *function of CoT faithfulness*, measured per-trajectory.
   Nobody has tied the white-box edge to a computable property. Laws convert.
2. **Reasoning agent + irreversible actions** (send/delete/approve/drop/deploy/email), not Q&A or chat.
3. **The sincere-error regime (felt≠granted)** — distinct from the deliberate-sabotage regime everyone studies
   (SHADE-Arena, deception probes). This is the safety-relevant blind spot the field hasn't isolated.
4. **Causal localization** (full-stack SAE + the late channel), not correlational AUROC only.
5. **First white-box-aware monitoring benchmark on a reasoning agent with released internal captures.**

## Standing on (cite + build, do not reinvent)
- **Metric:** "black-to-white performance boost" (Parrack, Attubato, Heimersheim, arXiv:2507.12691) — they found
  WEAK boosts on deception Q&A; we extend to reasoning-agent actions and predict the boost from faithfulness.
- **Substrate R1 (sabotage):** SHADE-Arena (Kutasov et al., arXiv:2506.15740; github jkutaso/SHADE-Arena; 17 task
  pairs, 20+ tools, ~25 turns, 0–100 suspiciousness monitor; top monitor Gemini-2.5-Pro AUC 0.87; side-task
  success relies on a HIDDEN SCRATCHPAD invisible to the monitor = the natural white-box-advantage setup) +
  CUA-SHADE-Arena + scaleapi/mrt (monitor red-teaming).
- **Eval protocol:** AI-control trusted monitoring — TPR @ fixed FPR≈2% audit budget, red/blue team
  (Greenblatt et al. 2312.06942; "Adaptive Attacks on Trusted Monitors" 2510.09462; SLEIGHT-Bench; MonitoringBench).
- **Deployed motivation:** Kramár et al. (GDM Gemini probes, 2601.11516) — answers their open questions (a)
  qualitative white-box edge, (b) adversarial evasion.
- **Confounds:** "linear probes rely on textual evidence" (2509.21344) → must show the probe edge survives on the
  UNVERBALIZED slice; obfuscation/evasion (Bailey 2412.09565; red-teaming probes 2511.00554).

---

## Hypotheses (pre-registered)
- **H1 — THE LAW (headline).** Per trajectory, the black-to-white boost B = (white-box monitor score's
  detection) − (best black-box monitor's detection) for the imminent-wrong-irreversible-action label is
  NEGATIVELY correlated with that trajectory's CoT faithfulness F (verbalization of the decisive fact). I.e.
  internals win where the CoT is performative. Pre-registered test: Spearman(B, F) < 0, with B measured as
  per-stratum ΔTPR@FPR=2% across faithfulness bins, bootstrap CI.
- **H2 — Disjoint blind spots (located).** ∃ a sincere-error (felt≠granted) slice the external task-grounded
  black-box check catches and white-box misses (replicates 21/21), AND an unverbalized slice white-box catches
  and the CoT-monitor misses. Oracle-union TPR ≫ max(single). Report the disjoint-error rate.
- **H3 — Robustness ordering.** Under adaptive optimization against each monitor, TPR@2% degrades fastest for the
  probe, then the CoT-text monitor, slowest for the external task-grounded check: robustness(external) >
  robustness(CoT-text) > robustness(probe). (COLLAPSE generalized to an ordering.)
- **H4 — Cost–quality frontier.** On the non-adversarial slice, the white-box probe matches/beats the LLM judge
  at ≫10× lower cost (replicate Kramár) AND adds coverage on the unverbalized slice (the qualitative edge GDM did
  not claim). The cascade (probe-first, defer ambiguous to judge) Pareto-dominates either alone.

Each hypothesis publishes in BOTH directions: a confirmed H1 is a law; a flat H1 is a measured correction of the
field's hand-wavy "complementary" claim. H2 is near-certain (safety: defense-in-depth necessity, mechanistically
located). H3 answers Kramár (b). H4 answers Kramár (a) with a deployment rule.

## Model + data
- **Primary:** Qwen3.6-27B reasoning agent + full-stack SAE (11 layers, the moat). **Cross-model:** gpt-oss-20b
  (already replicates the circuit) for one generalization figure.
- **Two regimes (the backbone contrast):**
  - **R1 — deliberate covert sabotage:** SHADE-Arena task pairs (benign main + harmful side, hidden scratchpad).
    The recognized substrate; gives the verbalized-vs-hidden axis directly.
  - **R2 — sincere model-origin error (felt≠granted):** our irreversible-action traps where the agent over-reaches
    in good faith (the momentum / co-authored-plan framing that triggered 18/18 in Felt-Not-Granted), labeled by
    an EXTERNAL task-grounded judge (granted vs felt authorization).
- **Scale (critical path):** the underpowered Tier-B was 5 positives. Target **≥80 positives + ≥80 matched
  negatives per regime** so TPR@FPR=2% is meaningful (CI on TPR at 2% FPR needs the positive count). Balanced
  across ≥5 action domains for held-out cross-action tests.
- **Faithfulness label F per trajectory:** measured two ways for robustness — (i) behavioral: does the CoT
  verbalize the decisive fact (keyword + LLM-judge, à la Young/SHADE); (ii) **mechanistic: our causal-patch
  conditional-faithfulness signal** (does patching the CoT decision-state change the action). The mechanistic F
  is the arc's unique instrument and the tie to the Late Channel.

## Monitors (the head-to-head)
Black-box (text only, no activations):
- **BB1 action-text:** judge sees the proposed tool call + args.
- **BB2 CoT-text:** judge sees the visible thinking tokens + action (the standard CoT monitor).
- **BB3 external task-grounded:** judge with access to the actual task spec / ground-truth authorization (our
  felt≠granted external check — the strong black-box).
White-box (activations):
- **WB1 linear probe** on the late band (cheap, Kramár/MultiMax-style attention-max probe) — the deployable arm.
- **WB2 SAE-feature / Late-Channel locus monitor:** named late-band features; score = causal-patch /
  decode-reencode confidence. The mechanistic arm; ties detection to the causal locus.
- Controls: random-feature & shuffled-label WB baselines; probe-text-leakage control (eval on unverbalized slice).

## Metrics
- **Primary:** TPR @ FPR = 2% per (regime × monitor), bootstrap CI. AUROC secondary.
- **The law (headline figure):** black-to-white boost B vs faithfulness F — scatter + per-bin ΔTPR, Spearman.
- **Complementarity:** oracle-union TPR; disjoint-error rate (caught by exactly one); the 3-tier cascade ROC.
- **Cost:** $/decision and FLOPs/decision (replicate the >10,000× framing); cascade deferral fraction.
- **Robustness:** TPR@2% vs attack budget per monitor (embedding attack on probe; obfuscation-pressure prompt on
  CoT-monitor; task-grounded perturbation on BB3) → the ordering.

## Rigor / controls (causality-protocol for monitoring)
1. TPR@FPR=2% with the continuous-FPR audit discipline (not gameable AUROC alone).
2. Held-out ENVIRONMENTS and held-out ACTION DOMAINS (train on some, test on others) — cross-action generalization.
3. Probe-text-leakage control: the WB edge must survive on the unverbalized slice (else it's reading tokens).
4. Random-feature / shuffled-label WB baselines; calibration on a clean dev split.
5. Red-team/blue-team for H3 (MonitoringBench/SLEIGHT-style), pre-registered attack budgets.
6. Faithfulness measured two independent ways (behavioral + mechanistic) — H1 must hold on both.

## The artifact (what converts)
**MonitorBoundary** (or extend AgentGuard): the first white-box-aware monitoring benchmark on a reasoning agent —
released two-regime labeled trajectories + per-token internal captures + the F labels + monitor baselines +
leaderboard. First-on-model, adoptable. (Per the moat memory: tool/benchmark/first-on-model converts, not PDFs.)

## Experiment plan (phased, each a go/no-go gate)
- **E0 — Pilot / power (GATE).** n≈15/regime. Confirm we can ELICIT a balanced positive set in BOTH regimes
  (R2 is the risk — see below) and that at least one WB and one BB monitor separate at AUROC>0.6. If R2 positives
  can't be elicited at ~50%+ rate, fall back to R1-only + a smaller R2 case study.
- **E1 — Scale capture.** ≥80+80/regime, ≥5 action domains, full-stack activations saved (OILAB_JSON + HF mirror).
- **E2 — Train/eval all 5 monitors**, per-regime TPR@2% + AUROC + cost. (H4 here.)
- **E3 — The law.** Compute B vs F (both F measures); Spearman + per-bin ΔTPR. (H1 — headline.)
- **E4 — Complementarity.** Oracle-union, disjoint-error rate, 3-tier cascade. (H2.)
- **E5 — Robustness.** Adaptive attack per monitor; the ordering. (H3.)
- **E6 — Cross-model.** gpt-oss-20b replication of the law's sign. (generalization figure.)
- **E7 — EVAL gate + ship.** Recompute every number from ledgers; hostile overclaim sweep; Zenodo (concept-DOI on
  site) + HF + benchmark release.

## The single riskiest assumption (kill-risk) + mitigation
**Eliciting genuine sincere-error (R2) positives at scale.** Felt-Not-Granted hit 18/18 only under a specific
"momentum / co-authored-plan under a vague go-ahead" framing; the gated experiment triggered only 2/32. If R2
can't be driven to a balanced positive rate, the most novel arm (felt≠granted) thins to a case study.
**Mitigation:** (a) productionize the momentum framing that worked, sweep framings in E0; (b) lean on SHADE-Arena
R1 (robust, recognized) as the load-bearing arm so the paper stands even if R2 is small; (c) if R2 stays thin,
reframe R2 as a qualitative "blind-spot existence proof" (the 21/21 already suffices for H2) and make H1 the
headline on R1+verbalization variation.

## Venue / timeline / compute
- **Venue:** main-track-grade (the law). ICLR 2027 (Oct deadline) main, or a NeurIPS 2026 safety/MI workshop now +
  benchmark to a Datasets&Benchmarks track. BlackboxNLP/ICML-MI as fast workshop checkpoints.
- **Compute:** Colab CLI G4/H100; capture + monitor training ≈ 2–4 GPU-weeks; R2 elicitation is the critical path.
- **Discipline:** pre-register (this doc → PREREG_*.md frozen) BEFORE E1; no DOI mint until E7 eval gate.

## Immediate next steps
1. Freeze this as `PREREG_white_box_vs_black_box.md` (lock H1–H4 + metrics before running).
2. Clone/adapt SHADE-Arena envs for R1 capture on Qwen3.6-27B; port the felt≠granted momentum traps for R2.
3. Run E0 power-pilot (the R2-elicitation gate) — decide full-scale vs R1-led fallback from its result.
