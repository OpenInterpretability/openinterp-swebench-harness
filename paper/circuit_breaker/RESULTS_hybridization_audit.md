# RESULTS — Hybridization-Selection Audit ("HydraHead B+") — Stage 1

**Run:** 2026-07-01/02 · Qwen/Qwen3.6-27B · Colab G4 (RTX PRO 6000 Blackwell 96GB, bf16, no offload) ·
**Code:** `scripts/hydra_audit.py` · **Ledger:** HF `swebench-phase6-verdict-circuit:results/hydra_audit_ie.json`
· **PREREG:** `PREREG_hybridization_audit.md` (hypotheses/cutoffs pre-declared; G-1 lit-check conditional-GO
incorporated before GPU).

## Setup recap
HydraHead-style receiver importance IE_{l,h} (arXiv:2606.20097 Eq. 9–10: per-head activation patch at the
o_proj input, corrupted→clean, **downstream attention frozen to clean** = direct effect; span logit-diff
readout, λ=0.9) computed for **all 384 FA heads** (16 full-attention layers {3,7,…,63} × 24 heads) on 12
NIAH counterfactual pairs (6 single-key + 6 multi-key, 4K ctx, structure/length-preserving digit-swap).
Score s_h = mean-IE × κ (cross-probe consistency, |IE|≥0.01), global ranking, retention cutoffs K=25% (96)
and K=12.5% (48). Infra: 6 VM deaths absorbed by per-layer HF checkpoint + autonomous reprovision driver;
zero data loss.

## Gates
- **G0 fidelity: PASS.** Deterministic decision points resumed from HF (60+60); live baseline emit on the
  first 15 BASH points = 0.133 — byte-identical to the June ledger per-point prefix (0.133); published
  full-set baselines reproduce (h1 0.233, h2 0.483; fidelity 0.474/0.299).
- **Patch-mechanism sanity: PASS.** Null-patch (clean→clean) IE = 0.0000 exactly; real corrupted patch
  produces graded nonzero IE on retrieval-relevant heads.
- **G1 probe sanity: denominators PASS** (12/12 pairs; m_clean +8..+11 vs m_corr −1.2..−2.2, |denom|≫0.5).
- **G1 split-half rank stability: the DECLARED metric FAILED — ρ = 0.46 < 0.8.** Diagnosis: zero-inflation
  artifact. 359/384 heads have s = 0 (no retrieval signal on any probe); their sub-noise values (~1e-4)
  reorder freely between halves and dominate the full-set rank correlation. Where the decision lives, the
  ranking is stable: **top-20 overlap between halves = 19/20**, and every named head is pinned in both
  halves (h21: .0478/.0482 · h19: .039/.048 · h8: .0003/.0008 · h6: .0009/.0003 · h3: .0117/.0080).
  **Remedy per PREREG** (expand pairs on G1 failure): +6 pairs/probe run, TARGETED at the
  decision-relevant union (top-30 ∪ circuit heads) — a disclosed cost deviation from the ideal full-384
  expansion. **Explicit ownership (M4):** the GLOBAL 384-head ranking remains unstable as pre-registered
  (ρ=0.46); the top-20 overlap (19/20) is a post-hoc diagnostic, not gate satisfaction. What the expansion
  DOES establish is that the named-head verdicts (writers at floor, opposers at top, 60–100× gap) are
  stable with doubled probe samples — the only ranking facts the H1/H2 claims rest on. Mid-rank cutoff
  boundaries should not be trusted.

## Stage-1 verdict (H1/H2)

**The retrieval landscape is extremely sparse:** only 25/384 heads have ANY signal (s > 0); 13 cross the
0.01 criticality threshold. Global top-5: L63h14 (0.113), **L59h21 (0.048)**, L55h6 (0.044),
**L59h19 (0.044)**, L51h20 (0.036).

| circuit head | role (RESULTS_heads_L59) | retrieval s_h | global rank | retained @K25? @K12.5? |
|---|---|---|---|---|
| L59h8 | writer (+0.122) | 0.0000 (κ=0) | tied at s=0 (last 359) | **NO / NO** |
| L59h6 | writer (+0.119) | 0.0000 (κ=0) | tied at s=0 | **NO / NO** |
| L59h3 | writer (+0.083) | 0.0098 | 14/384 | yes / yes |
| L59h21 | opposer (−0.014) | 0.0480 | **2/384** | yes / yes |
| L59h19 | opposer (−0.027) | 0.0439 | **4/384** | yes / yes |
| L59h16, h11, h4, h23 | opposers | 0.0000 | tied at s=0 | NO / NO |

- **H1 = outcome (b), the alarming branch:** commit writers h8 and h6 carry **zero retrieval criticality**
  (IE ≈ 5e-4, κ = 0 — never crossed threshold on any probe). They are dropped not marginally but at ANY
  FA budget that retains only capability-critical heads. Only h3 survives (it is the weakest writer).
- **H2 = split, sharper than predicted:** the two strongest retrieval heads of the ENTIRE layer (h21, h19 —
  global ranks 2 and 4) are commit-circuit **opposers**; the other four opposers have zero signal and drop.
- **Structural reading:** within L59, retrieval-criticality and commit-writing are ANTI-aligned. The
  selection keeps 1 of 3 writers and 2 of 6 opposers → the push–pull circuit is shredded asymmetrically.
  This recurses the #7 lesson one level up: geometric write magnitude misled about causal commit (h21);
  now *causal capability-criticality* misleads about *safety-criticality*. A criterion being causal is not
  enough — it is causal FOR ITS CAPABILITY SET.

## Interpretation guardrails (pre-declared)
Correlational ranking alone ≠ the deliverable (G-1 bar). The claim "selection drops the commit circuit"
becomes reviewable only with H3: same ablation, both sides — NIAH readout must PASS while commit
emit/calibration degrades (positive control: ablating the retained retrieval heads must hurt NIAH).
Mean-ablation = upper bound of damage BEFORE distillation recovery; no claim about post-distillation
models; no claim "HydraHead is unsafe" (criterion-class audit, different model scale).

## Stage 2 (H3) — COMPLETE (2026-07-02). Commitment collapses at BAND scope; the capability probe never registers anything.
**PREREG note (scope drift, disclosed):** the pre-registered PRIMARY scope (L59-only) returned NULL; the
band-scope result below is the pre-declared SECONDARY condition and should be read as such — a strong
exploratory finding relative to the pre-registration, not the confirmed primary outcome.

Per condition: same mean-ablation for both readouts. NIAH pass-rate (12 pairs) + P(edit)/emit at the
60+60 deterministic points; exact McNemar (per-point, paired) vs base.

| condition (mean-ablate) | NIAH | edit-emit @edit-pts | @bash-pts | gap | McNemar vs base (edit-pts) |
|---|---|---|---|---|---|
| base | 1.00 | 0.483 | 0.233 | 0.25 | — |
| drop_nonret_L59 (20h, PRIMARY) | 1.00 | 0.47 | 0.23 | 0.23 | lost 1 / gained 0, n.s. |
| drop_rand_L59 (20h null) | 1.00 | 0.47 | 0.23 | 0.23 | n.s. |
| drop_ret_L59 ({21,19,3,5} pos-ctl) | **1.00** | 0.47 | 0.23 | 0.23 | n.s. |
| drop_writers {8,6,3} | 1.00 | 0.45 | 0.23 | 0.22 | lost 2 / 0, p=0.5 n.s. |
| drop_opposers (6h) | 1.00 | 0.48 | 0.23 | 0.25 | 0/0 n.s. |
| **drop_nonret_late (74h, L51–63)** | **1.00** | **0.183** | **0.183** | **0.00** | **lost 18 / gained 0, p=7.6e-6** |

**Findings:**
1. **Commitment collapses at band scope while the capability probe registers nothing.** Ablating the 74
   capability-non-retained late-band heads collapses task-appropriate commitment — P(edit) at edit points
   0.474→0.167 (the clean, fully-ablated readout), edit-emission 0.48→0.18, and the edit/bash differential
   (0.25→0.00) disappears: the agent stops distinguishing where committing an edit is appropriate.
   Monotonic one-directional flips (18 lost, 0 gained, exact McNemar p=7.6e-6; concentrated at edit
   points — bash side 3 flips, n.s.). The NIAH readout stays at ceiling throughout — but see finding 3:
   that pass is UNINFORMATIVE (the probe cannot fail), so the correct claim is "the capability probes
   used by the criterion never register any of it", NOT "capability is preserved".
   **Measurement note (M2):** the emit readout applies the ablation at prefill only (`use_cache=True`
   makes the per-step mean a no-op at decode); P(edit) (`use_cache=False`) is ablated at every position
   and shows the same collapse — we foreground P(edit) as the primary causal readout.
2. **Layer-scope robustness (the null result that sharpens it):** every single-layer condition — including
   ablating the writers themselves — is behaviorally silent (≤2 flips). Natural commitment is REDUNDANT
   across the late band (mirrors #8: the brake locus is depth-distributed); the named L59 heads carry the
   *lever* (donor-injected control, #7), not the sole load of natural behavior.
3. **The capability side is UNTESTED, not "passed" — and that is itself the finding.** The pre-registered
   positive control FAILED: drop_ret_L59 (ablating the retained retrieval heads) did not hurt NIAH
   (margin 9.29 vs base ~9.4), and under the heaviest ablation the teacher-forced NIAH margin actually
   RISES (10.56 under drop_nonret_late vs 9.39 base). An instrument whose margin increases when 74 heads
   are removed has no sensitivity at this difficulty (4K, 7-digit needle); its "pass" carries no
   information about capability preservation. We therefore make NO claim that the selection preserves
   capability — the honest, and sufficient, statement is: **the capability probes that define the
   criterion are blind to the commitment collapse** (and a probe hard enough to be sensitive at this
   scale remains future work). This sharpens, rather than weakens, the audit thesis: the criterion's own
   metric cannot see the damage.

**Honest scope:** mean-ablation upper-bounds pre-distillation damage; only 25/384 heads carry signal, so
the K25 top-96 fills ranks 26–96 with zero-signal heads by sort-tie order (early layers) — the late band
retains 22 heads (the late signal heads); set sizes verified against the run ledger (74/74/72/20 heads).
Single-model, single-behavior-family, n=60/side.

## Stage 3 — controls + G1 expansion (COMPLETE 2026-07-02). Final verdict.

| condition (late band, L51–63) | NIAH | edit-emit @edit-pts (McNemar vs base) | @bash-pts |
|---|---|---|---|
| base | 1.00 | 0.483 | 0.233 |
| **drop_nonret_late** (capability selection, 74h) | 1.00 | **0.183** (18 lost/0 gained, **p=7.6e-6**) | 0.183 (n.s.) |
| **drop_rand_late** (random 74h, same 22-head budget — drops 4 of the global top-5 incl. L63h14/h21/h19, keeps h3 and L55h6 by chance) | **1.00** | 0.333 (9/0, p=3.9e-3) | 0.100 (9/1, p=0.021) |
| **keep_writers_late** (= selection + {h8,h6}, 72h) | 1.00 | **0.483 (0 flips — full restore)** | 0.300 (0/4, p=0.12) |

**Final findings:**
1. **Capability-guided selection is WORSE than random for the unmeasured behavior — under a tight
   control.** Both arms keep exactly 22 late-band heads, and both keep writer h3 (the random draw keeps
   it by chance); the ONLY difference is which other 21 are kept — the retrieval-critical set vs a random
   set. Random: commitment degrades to 0.33 (9/0, p=3.9e-3). Capability-guided: collapses to 0.18 (18/0,
   p=7.6e-6). Random arbitrary heads preserve more commitment redundancy than the retrieval-only set —
   because commitment carriers are exactly the zero-retrieval heads the criterion systematically discards.
   Direct paired test random→capability: 10 lost / 1 gained, exact McNemar **p=0.012**. Anti-alignment
   (Stage 1) made behavioral (Stage 3). (Single-draw caveat being closed: 4 extra random seeds running.)
2. **A 2-head safety term fully restores the behavior.** Keeping {h8,h6} (0.5% of FA heads; found only
   via prior causal circuit work — the criterion itself scores them κ=0) restores edit-commitment with
   ZERO per-point flips vs base. Residual mis-calibration trend at bash points (+4 gained, n.s.):
   the zero-signal opposers stay converted → push partially unopposed, consistent with H2.
3. **Benchmark-blindness, stated precisely:** NIAH@4K stayed at ceiling under EVERY condition — including
   random-74 (which drops 4 of its own top-5 heads) — and the teacher-forced margin RISES under the
   heaviest ablation (10.56 vs 9.39 base) while the positive control barely moves (9.29). The probe has
   no sensitivity at this difficulty; its pass is uninformative. Honest claim: "the capability set's
   probes are blind to the commitment collapse" — NOT "the selection preserves capability" (untested).
   Multiple-comparison note: ~9 conditions × 2 sides of uncorrected McNemars; the headline p=7.6e-6
   survives any correction, the secondary bash-side p=0.021 does not — reported as uncorrected.
4. **G1 expansion (pre-declared remedy): named-head verdict unchanged with doubled probe SAMPLES
   (n=24 pairs; same 6 keys/filler structure, new numbers/depths/seed — not fully independent probes).**
   Writers stay at floor (h8 0.0007, h6 0.0008 combined), h3 at threshold (0.0105), opposers stable at
   top (h21 0.0484, h19 0.0471). The 60–100× writer/opposer gap is not sampling noise.

**The result in one paragraph:** In Qwen3.6-27B, the heads that causally carry an agent's action
commitment are invisible to a HydraHead-style capability-guided FA→LA selection (retrieval-IE ≈ 0, κ=0),
and the layer's strongest retrieval heads are the commitment circuit's OPPOSERS. Applying the selection
across the late band collapses task-appropriate commitment (P(edit) 0.474→0.167; emission 0.48→0.18;
monotonic, p=7.6e-6) — worse than a random selection of equal budget (paired p=0.012) — while the
capability probes that define the criterion register nothing (and, having failed their positive control,
could not have: their pass is uninformative). Restoring just the two named writer heads (0.5% of FA
budget, identifiable only by causal circuit analysis — the criterion scores them κ=0) recovers the
behavior. Capability-causal ≠ safety-causal: audit the transformation with the circuit you care about,
because the criterion cannot see what it does not measure — and neither can its benchmarks.

**Compute:** ~7h Colab G4 (RTX PRO 6000 96GB) across ~12 VM incarnations (keep-alive deaths), zero data
loss via per-stage HF checkpointing + detached autonomous driver. Ledger: `results/hydra_audit_ie.json`.

## Stage 4 — red-team controls (COMPLETE 2026-07-03). Both confounds closed.

Post-eval adversarial review (independent red-team agent) demanded two GPU controls; both run.

**M3 closed — "worse than random" across 5 seeds (was: single draw).**
| random draw (74h, seed) | writers dropped [8,6,3] | edit-emit | paired draw→capability |
|---|---|---|---|
| s1 (42) | D D k | 0.333 | 10/1, p=0.012 |
| s2 (43) | D D D | 0.383 | 12/0, p<0.001 |
| s3 (44) | D k D | 0.250 | 8/4, p=0.39 n.s. |
| s4 (45) | k D D | 0.217 | 6/4, p=0.75 n.s. |
| s5 (46) | D k D | 0.450 | 16/0, p<0.001 |

Capability selection (0.183) sits **below all 5 random draws** (mean 0.327, min 0.217); paired per-draw
tests significant for 3/5. Honest statement: "at the bottom of the random distribution (5/5 draws below
none), significantly worse than 3 of 5 draws" — not "worse than random, p=0.012" from one draw.
Note the mechanism refinement the membership column reveals: s2 dropped ALL three writers yet stayed at
0.38 — random draws preserve commitment through OTHER redundant band heads. The capability-selected keep
set (retrieval-only) is the *least* commitment-redundant subset of its size: that, not writer membership
alone, is why it is worst.

**M1 closed — the 2-head rescue is head-specific, not severity relief.**
| condition (all 72h ablated, same severity) | edit-emit | P(edit) |
|---|---|---|
| keep {h8,h6} (writers) | **0.483** | **0.487** |
| keep 2 random non-writers, s1 (101) | 0.200 | 0.173 |
| keep 2 random non-writers, s2 (102) | 0.167 | 0.168 |
| keep 2 random non-writers, s3 (103) | 0.150 | 0.158 |

Keeping 2 RANDOM heads at identical ablation severity leaves commitment collapsed (mean 0.172 ≈ the full
collapse 0.183); keeping the 2 named writers restores baseline exactly. Paired keep_writers→keep2rand_s1:
17/0, p=1.5e-5. Combined with Stage 3: **{h8,h6} are jointly sufficient to carry task-appropriate
commitment when the rest of the band's redundancy is stripped — and they are exactly the heads the
capability criterion scores κ=0.**

**Final compute:** ~9h Colab G4 total, ~15 VM incarnations, zero data loss. All numbers recomputed from
the public ledger by `scripts/eval_hydra_audit.py` (52 checks expected post-Stage-4).
