# Pre-mint EVAL — late_channel.tex (2026-06-18, v2 after overclaim audit + control)

Discipline: recompute every number, verify citations, adversarial overclaim audit — BEFORE any DOI.
This v2 follows Caio's challenge ("tem overclaim?") which surfaced a real methodological gap; the gap was
closed with a dedicated control, and the paper rewritten.

## 1. Numeric recompute — `eval_late_channel.py`
**37/37 claims PASS.** Recomputed from `data/cot_faithfulness_n60.json` (Tier-A), `cot_faithfulness_agent_n32.json`
(Tier-B), and `logit_lens_control.json` (the control), bootstrap seed=0:
- Tier-A: n=60, 16 flips, 44 already-correct; L59 +2.72 [+2.18,+3.28]; L51/L63/L47; LATE +2.13 vs EARLY −0.02; 16/16; min/max.
- Tier-B: n=32, 10 flips, 5 irreversible / 27 cautious; L51/L59/L63; LATE +1.65 vs EARLY +0.08.
- **Control (logit-lens):** L35 −0.93, L47 +0.02, L51 +1.57, L59 +5.18, L63 +6.64, onset L51. All pass.

## 2. Citations
deep-research scan (25/25, adversarial 3-vote) + spot-check of the load-bearing one: **2507.22928** confirmed
("How does Chain of Thought Think? ... Sparse Autoencoding", Pythia-70M/2.8B, GSM8K, scale threshold). DeepMind
quote verified by direct fetch. Zenodo 20534219/20634838 = our arc.

## 3. Overclaim audit — the four issues from the challenge, and how each was fixed
1. **🔴 "late = consolidation" was unproven (could be patch-dilution).** Caio's catch. FIXED by a method-independent
   control (§\,sec:control): logit-lens margin is **negative early** (−0.93 @ L35, model leans System-1),
   crosses 0 at L47, rises to **+6.64 @ L63**, onset L51 — the deciding info is absent early and consolidates
   late, ruling out dilution. The two signals (patch + logit-lens) corroborate. Now the paper's strongest part.
2. **🟡 Title/abstract packed the cited "action commitment" as if new.** FIXED: title is now scoped to
   "reasoned answers and agent actions" (both measured); §6 says action commitment is "connecting to prior
   work, not re-measured here".
3. **🟡 "monitoring locus" over-extended from causal to predictive (the AUROC failed).** FIXED: the paper now
   separates *decodability* (strong: logit-lens +6.6) from *class prediction* (underpowered, 5 positives, future
   work), and says action-level monitoring at scale "remains to be demonstrated".
4. **⚪ "faithfulness" / small n.** The conditional framing (causal only on flips; performative is the common
   case, 44/60) and the n caveats are in Limitations.

## Verdict
**0 blockers.** 37/37 numbers reproduce; the one real overclaim (the "late" confound) is now closed by a
dedicated control and is the paper's strongest evidence; framing/monitoring claims rescoped honestly; figure
added (dual-curve). Mint held for the author's explicit GO — mint is the irreversible step.
**Lesson logged:** my first "audit clean" was leniency (author-blindness); the second pass, prompted, found the
real issue. Reinforces [[feedback_close_confounds_and_name_before_publishing]] — run the adversarial audit as a
hostile reviewer, not the author.
