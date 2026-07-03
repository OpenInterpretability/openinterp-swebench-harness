# PREREG — Hybridization-Selection Audit ("HydraHead B+")

**Question:** Does capability-guided FA→LA head selection (HydraHead-style, arXiv:2606.20097) RETAIN
the causally-verified agent-commitment circuit of Qwen3.6-27B — or does it silently drop it?

**Date:** 2026-07-01 · **Status:** PRE-REGISTERED, not run. GPU spend gated on Caio OK.
**G-1 lit-check: PASSED CONDITIONALLY (2026-07-01, agent litcheck-bplus).** Two binding corrections:
1. **KILLED CLAUSE:** "the capability-set blind-spot framing for head selection is new" — FALSE.
   arXiv:2510.08525 ("Which Heads Matter for Reasoning?") owns that framing for REASONING (retrieval
   criteria drop reasoning-critical heads). The framing is our SETUP, not our contribution. Headline =
   the OBJECT: named agent-action/commitment circuit × a specific SOTA method (HydraHead) × audit-not-
   protect posture.
2. **CAUSAL BAR:** a purely correlational "our heads aren't in the retained set" is worth ~nothing over
   2510.08525. The deliverable must be causal on the circuit: ablate/convert the non-retained heads and
   measure BOTH the commit circuit AND the C-benchmarks under the same intervention. The reviewable
   positive = "C-benchmarks pass while the commitment circuit degrades/mis-calibrates", on named heads.
   → H3 is PROMOTED to primary deliverable (see below).
**Must-cite:** HydraHead 2606.20097 · 2510.08525 (novelty threat, differentiate by object) · Wei
2402.05162 · safety-heads 2410.13708 · AAPP 2511.07482 · compressed-agents 2505.19433 · Q-resafe
2506.20251 · distillation-breaks-safety 2512.09403 · DuoAttention 2410.10819 · RazorAttention 2407.15891.
Differentiation-by-object paragraph archived in the litcheck report (this session).

## Motivation

HydraHead (Alibaba, jun-2026) selects which attention heads keep Full Attention (FA) vs get converted
to Linear Attention (GDN) via a *causal* head-importance score IE_{l,h} — activation patching with
downstream-freeze (receivers) + path patching (senders) — computed on a capability set
**C = {long-context retrieval (RULER NIAH sub-probes), general ability}**, retaining the top-K
(default 25% FA; matches layer-wise 3:1 quality at 7:1 → 12.5%).

Their method is sound (it is our own toolbox). The auditable blind spot is the **capability set**: a
head that is safety/agency-critical but not retrieval/general-critical scores low → converted → the
circuit dies silently while every benchmark in C passes.

We hold a named, causally-verified target: the **commitment circuit** of Qwen3.6-27B
(`RESULTS_heads_L59.md`): writers **h8/h6/h3 @L59** (ΔP +0.122/+0.119/+0.083; top3 = 117% of the full
attention effect; emit 0.23→0.42), opposed by **h16/19/11/4/21/23** (push–pull). Architecture:
`full_attention_interval=4` → FA layers {3,7,…,59,63} = **16 layers × 24 heads = 384 FA heads**.
Known confound-killer from that work: geometric write magnitude misleads (h21 = largest writer,
causally an opponent) — only patching counts.

**Scope note (honest):** the **brake has no head locus** (all |ΔP| ≤ 0.012, distributed/upstream), so a
head-retention audit *cannot* directly test brake preservation. The named head-level object under audit
is the COMMIT (elicit) circuit. Brake behavior is measured only observationally in H3.

## Hypotheses (pre-declared)

- **H1 (writer retention).** Are h8, h6, h3 @L59 in the retained-FA set under retrieval+general IE
  ranking, at K=25% (96/384) and K=12.5% (48/384)?
  - Prior (stated honestly): writers are induction/copy-style heads reading tool-name history
    (`RESULTS_attn_L59.md`) — same family as retrieval heads → *probably retained*.
  - Outcomes: (a) all 3 retained at both K → **coincidental preservation** (reassuring branch; claim:
    preservation is incidental to C, not guaranteed — the audit method is the contribution);
    (b) ≥1 writer outside top-25% → **selection drops the commit circuit** (alarming branch);
    (c) retained at 25% but dropped at 12.5% → **graded**: aggressive ratios break it.
- **H2 (opposer retention).** Are the opposing heads (16,19,11,4,21,23 @L59) retained? Weak prior:
  opposers are *less* likely retrieval-critical than writers. If writers retained but opposers dropped →
  the push–pull is broken asymmetrically → predicts commitment *mis-calibration* (over-commit), tested
  in H3.
- **H3 (PRIMARY DELIVERABLE — the causal two-sided test).** Under the SAME intervention — mean-ablate
  the c_h of all NON-retained heads (primary @L59; secondary across the late FA band 51–63) — measure
  BOTH sides:
  - **Circuit side:** P(edit) + emit-rate at the #7 deterministic decision points vs (i) baseline
    (0.23), (ii) random equal-size ablation (null), (iii) retained-set ablation (positive control —
    must destroy). Mis-calibration readout for the H2 branch: emit ABOVE baseline / at wrong points
    (over-commit) counts as degradation, not only emit-loss.
  - **Benchmark side (C):** NIAH sub-probe accuracy under the identical ablation. The reviewable
    positive = **C-benchmarks pass while the commitment circuit degrades or mis-calibrates** —
    causal, on named heads, first-on-object.
  Also record brake-side emit (observational only). **Honest scope:** mean-ablation ≈ upper bound of
  damage BEFORE distillation recovery; trained GDN replacement could recover function — untestable
  without training, out of scope, stated in any writeup.

## Method (Stage 1 — the decisive, cheap test)

Re-implement HydraHead's **receiver IE** (their Eq. 9–10) on Qwen3.6-27B:
1. **Probes:** 2 RULER-style NIAH sub-probes (single-key + multi-key MK1), 4K context, ≥6
   counterfactual pairs each (structure/length-preserving digit swap), span logit-diff readout with
   exponential decay λ=0.9. (They use 5 sub-probes × 8 pairs; C.1 shows ranking stable from ~6 —
   we pre-declare 2×6 as primary, expand only if G1 stability fails.)
2. **Patching:** per-head activation patch (corrupted→clean) at the o_proj input slice with
   downstream-attention freeze (direct-effect), normalized IE ∈ [0,1]; criticality threshold |IE|≥0.01
   (theirs). Receivers-only for v1 (their senders converge to the same shallow top-K set; senders =
   robustness appendix if Stage 1 is ambiguous).
3. **Ranking + cutoffs:** global ranking over all 384 FA heads; retained sets at K=25% and K=12.5%
   (both reported; no post-hoc K). "General ability" term: v1 approximates C by retrieval-only (their
   FA-retention signal is dominated by retrieval-criticality; equal-weight fusion noted as limitation);
   if H1 lands in outcome (b), we add a general-ability probe before claiming anything (kill-the-result
   check: writers must not be rescued by the general term).

## Gates

- **G-1 (overclaim gate, BEFORE GPU):** adversarial lit-check (agent `litcheck-bplus`) returns no
  DIRECT-HIT on: (i) head-level hybridization/compression selection dropping non-retrieval safety
  circuits; (ii) circuit-level (named-heads) safety-preservation audit of a selection criterion.
  Known-adjacent to cite regardless: Wei et al. brittleness-via-pruning; safety-heads line;
  quantization-vs-safety line; Wu et al. retrieval heads.
- **G0 (fidelity, on GPU):** head-split reconstructs attention output (relerr ≤ 1e-2, published:
  0.0015); baseline emit reproduces 0.23±; top3 re-injection reproduces ≈+0.26 ΔP.
- **G1 (probe sanity):** IE finds a sparse nonempty critical set; split-half rank correlation ≥ 0.8;
  L59 must not be trivially empty of retrieval signal (if the whole layer is non-critical, report that
  as a finding, not a bug — but verify with the single-key probe at a needle placed late).
- **Nulls:** label-shuffled IE ≈ 0; random head sets for H3.

## Compute plan

Colab A100/G4 (same as #7), bf16 + accelerate offload, reuse `commit_lever_stages/decomp/heads`
module pattern; new module `hydra_audit.py` with per-layer-block checkpoint/resume to HF ledger
`results/hydra_audit_ie.json`. Cost estimate: ~4.6k patched forwards @4K ctx (384 heads × 6 pairs ×
2 probes) + bases ≈ single-digit hours with resume discipline. H3 adds ~6 conditions × 60 rows.

## Verdict → allowed claims (pre-declared)

| Stage-1 outcome | Allowed claim |
|---|---|
| Writers dropped (b) | "Capability-guided head selection silently drops a causally-verified agent-commitment circuit; benchmarks in C cannot see it. Audit the transformation, not just the model." |
| Writers retained, opposers dropped | "Selection preserves the push, breaks the pull: commitment mis-calibration risk under aggressive hybridization" (requires H3 support). |
| All retained (a) + H3 two-sided flat | "Preservation is coincidental (induction-family overlap), not guaranteed by design; the named-circuit audit protocol is the contribution; bound holds only for retrieval-like C." (Weakest branch — publishable as a note, not a paper; per G-1, the framing alone carries no novelty.) |

Non-claims in all branches: nothing about post-distillation models; nothing about the brake's
head-locus (it has none); no "HydraHead is unsafe" (we audit the *criterion class*, their model is
1.7B and not our test target).

**Deliverable:** `RESULTS_hybridization_audit.md` + HF ledger; extends the Located-Not-Secured thesis
from "audit the fixed model" to "audit the transformation". Framing per
[[feedback_stand_on_shoulders_not_compete]]: we extend HydraHead's own causal-selection idea with a
safety term — grateful, not adversarial.
