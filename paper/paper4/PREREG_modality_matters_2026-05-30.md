# Pre-registration — "Modality Matters": does the WANDERING knowledge-action gap close behaviorally, not residually?

**Status:** PRE-REGISTERED 2026-05-30, before collecting any behavioral-arm data. Commit this file before running. (Discipline: [[feedback_walk_back_and_rescue_paper_structure]], [[feedback_paired_baseline_for_run_unstable_interventions]].)

## 1. Motivation & the open conjunction

The "knowledge-action gap" (internal signals DETECT a failure but are not causal levers) is now consensus for **single-forward-pass** settings: Basu et al. (arXiv:2603.18353), Bhalla et al. ("predict/control discrepancy", arXiv:2411.04430). It is **untested on long-horizon agents**, and nobody has shown what the *working* lever is. Our prior work establishes the residual half on agents: at L11 — WANDERING's strongest discriminator (paper #3) — SUCCESS-direction injection does **not** rescue (paper #2, paired McNemar p=0.73); it only destabilizes (dose-dependent crash). 

This experiment tests the other half: **does a transient BEHAVIORAL intervention, gated by the same detector, rescue WANDERING where residual steering cannot — on the same trajectories?** If yes, the gap is real but its lever is behavioral, not representational. That conjunction (agent + dynamic + cross-arch + a positive behavioral lever, contrasted against a residual null at the same locus/trigger) is unclaimed.

**What is NOT our claim** (cite to scope): behavioral course-correction of looping SWE agents already exists (SWE-PRM / "When Agents go Astray", Gandhi et al. arXiv:2509.02360); transient > persistent steering exists (SafeThink, arXiv:2602.11096); activation steering *does* move agent exploration-entropy (EAST, arXiv:2406.00244 — so our residual null is specifically about **finalizing**, not exploration). Our contribution is the **controlled modality contrast**: same trajectories, same trigger turn, same budget — only the intervention *modality* differs.

## 2. Population

The 20 WANDERING trajectories from Phase 6 (Qwen3.6-27B, SWE-bench Pro), identical to paper #2. Same `Runner.seed_for(iid)`. RTX 6000 Pro Blackwell 96 GB, temperature 1.0. (Cross-architecture replication on Llama-70b is a pre-registered stretch goal, §8, not required for the primary claim.)

## 3. Trigger (the detector firing — live, online)

The v5 tool-entropy collapse signal from paper #1: at each turn, compute Shannon entropy of the **last-10 tool-call names** emitted so far. Fire the intervention **once**, at the first turn where entropy < τ (τ = the v5 threshold from the Tool-Entropy paper). If no collapse occurs before turn K_max, record `never_triggered` and apply no intervention (these instances are excluded from the paired primary, reported separately). This mimics real deployment (online detection), not an oracle.

## 4. Arms (all run in the SAME session on the SAME 20 instances — paired)

| Arm | Intervention at trigger turn | Role |
|---|---|---|
| **0 — baseline** | none (no-hook re-run) | **The paired baseline for McNemar.** Expect ~7/20 flip from run-instability — this is why it must be contemporaneous, not the definitional 0/20. |
| **A — residual** | L11 SUCCESS-donor injection (α=0.70) for the remaining turns | The residual lever (paper #2 null), re-run in-session for a clean paired contrast. |
| **B1 — behavioral: re-plan** | inject one `user` message forcing re-plan + decision | **Primary behavioral lever.** |
| **B0 — behavioral control: neutral** | inject one `user` message of matched token length, content-neutral | Isolates "a message appeared / context refreshed / budget" from the re-plan **content** (behavioral analog of the random-direction control). |
| **B2 — tool-restriction** *(stretch)* | disable the looped tool for remaining turns | Second behavioral modality. |

**B1 re-plan prompt (frozen here):** *"You appear to be repeating similar tool calls without converging. Stop and re-plan: in 2–3 sentences state what you have already established, then either (a) call finish_tool with your best patch now, or (b) name the single concrete blocker and the one next action to resolve it."*

**B0 neutral prompt (frozen, matched length):** *"Here is a brief checkpoint note for your records. You may continue working on the task as you see fit; no change of approach is required. Proceed with whatever next step you judge appropriate."*

## 5. Outcome measures (per run)

- **Primary:** `finish_reason ∈ {finish_tool, max_turns, invalid_tools}`. Rescue = `finish_tool`. (WANDERING is non-finalization; emitting finish is the direct rescue.)
- **Secondary (quality):** does the emitted patch PASS the SWE-bench Pro tests (Docker eval)? Guards against the paper-11 "containment vs correction" trap — a finish with a broken patch is finalization but not a fix. Reported, not gated.
- **Safety:** `invalid_tools` rate (does the behavioral nudge destabilize, like the residual high-α did?).

## 6. Pre-registered hypotheses & GATE

- **H1 (behavioral rescues):** paired McNemar (B1 vs Arm 0) on `finish_tool`. **GATE: p < 0.05 AND direction = B1 helps** (more B1-only flips than baseline-only).
- **H2 (modality matters — the headline):** B1 `finish_tool` rate > Arm A rate on the same instances (McNemar B1 vs A). This is the core contrast.
- **H3 (content, not container):** B1 > B0 (McNemar). If B1 ≈ B0, the lever is context-refresh/budget, not re-plan content — report honestly.

**Decision tree (committed):**
- H1 ✓ & H2 ✓ & H3 ✓ → "Modality Matters" confirmed → strong standalone paper #4.
- H1 ✓ but H3 ✗ (B1 ≈ B0) → the lever is *transient context refresh*, not re-plan content → honest, reframed (still a positive behavioral lever, weaker mechanism claim).
- H1 ✗ (B1 null) → behavioral *also* fails → deeper null: simple interventions don't close the agent knowledge-action gap from either modality → honest null, still publishable as the stronger negative.

No definitional-0/N baseline anywhere. All primary tests paired (McNemar) against the same-session Arm 0. Power note: N=20 → only large effects detectable; we report effect sizes + exact p, not just significance.

## 7. Confound controls (lessons banked)

1. **Run-instability** → Arm 0 is a same-session paired re-run (not the 0/20 label). Mandatory.
2. **Information/length confound** → Arm B0 neutral message matched in length isolates content.
3. **Trigger leakage** → trigger computed live from the running trajectory's own tool history (no oracle, no use of the original-run collapse turn).
4. **Selection on outcome** → B0/B2 run on ALL triggered instances, not just the B1-rescued subset (the paper #2 random-control mistake — not repeated).
5. **Quality vs finalization** → Docker patch-pass reported alongside finish_tool.

## 8. Implementation sketch (no GPU needed to write; ready to run)

Injection point confirmed in `agent/loop.py` — `AgentLoop.run()` (line ~158): turn loop `for turn_idx in range(cfg.max_turns)` over a `messages` list. Implement `BehavioralAgentLoop(AgentLoop)` (or a turn-start callback) that, at the start of each turn:
1. computes live v5 entropy of last-10 tool names from `result.turns`;
2. if collapsed & not yet fired & arm∈{B0,B1,B2}: `messages.append({"role":"user","content": PROMPT})` (B0/B1) or restrict the tool registry (B2); set `fired=True`, log `trigger_turn`.
Arm A reuses the paper #2 `LayerPatch` (L11, α=0.70) attached at the trigger turn. Notebook mirrors `nb_paper2_v2_l11_phase1` structure (per-repo donor for Arm A; per-instance JSON in Drive; resume-aware). Analysis reuses `scripts/paper2_l11_honest_paired_analysis.py` (paired McNemar + contamination check), extended to the B-arms.

Estimated compute: 4 arms × 20 instances ≈ 80 runs ≈ paper #2 scale (~13–16 h on RTX 6000 Pro Blackwell, ~$25–30 vast.ai).

## 9. Citations to anchor (paper #4)
Basu 2603.18353 (knowledge-action gap, primary prior) · Bhalla 2411.04430 (predict/control) · SWE-PRM 2509.02360 (behavioral rescue exists — out-specify via the contrast) · EAST 2406.00244 (scope the residual null to finalizing) · SafeThink 2602.11096 (transient > persistent) · StepShield 2601.22136 (Intervention Gap metric) · Tatsat & Shater 2605.06890 (nearest agent-activation competitor, detection-only) · our paper #1/#2/#3.

## 10. Doubles as the MATS Autumn 2026 proposal (deadline 2026-06-07)
This pre-reg is the research proposal for the MATS application (mentors McAleer / Emmons — control & monitoring for coding agents). The application can be submitted with the existing 3 papers + deployed detector + this pre-registered design, before the experiment runs.
