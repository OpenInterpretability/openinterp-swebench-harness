# Pre-mint EVAL — late_channel.tex (2026-06-18)

Discipline: recompute every number from `data/`, verify citations, adversarial overclaim audit, BEFORE any DOI.

## 1. Numeric recompute — `eval_late_channel.py`
**31/31 claims PASS.** Every number in the paper recomputed from `data/cot_faithfulness_n60.json` (Tier-A) and
`data/cot_faithfulness_agent_n32.json` (Tier-B), bootstrap seed=0 (matches the paper's generation):
- Tier-A: n=60, 16 flips, 44 already-correct-noCoT; L59 +2.72 [+2.18,+3.28]; L51/L63/L47; LATE +2.13 vs EARLY −0.02; 16/16 consistency; min/max.
- Tier-B: n=32, 10 flips, 5 irreversible, 27 cautious; L51/L59/L63; LATE +1.65 vs EARLY +0.08.
All within tolerance. Re-run: `python eval_late_channel.py` (exit 0).

## 2. Citations
- Inherited from the deep-research scan (25/25 claims confirmed, adversarial 3-vote): Turpin 2305.04388,
  Anthropic 2505.05410, monitorability 2507.11473, Baker 2503.11926, Chen 2507.22928, Mirtaheri 2603.17199,
  Goodfire 2603.05488.
- Spot-checked the load-bearing one (the result we extend): **2507.22928** confirmed = "How does Chain of
  Thought Think? Mechanistic Interpretability of CoT Reasoning with Sparse Autoencoding" — Pythia-70M/2.8B,
  GSM8K, SAE+activation patching, scale threshold (null@70M, causal@2.8B). Our framing (extend to 27B reasoning) holds.
- DeepMind quote ("inspect the model's inner workings", "reading verbalized reasoning will not be enough")
  verified by direct fetch of the blog (2026).
- Zenodo DOIs 20534219 (#6), 20634838 (#7): our own published arc papers.

## 3. Adversarial overclaim audit
- **"three-faced law"** — scoped to "on this model" in abstract + §6; face (i) action is CITED (#6/#7), not
  re-measured here, and the text says "a prior result localizes". OK.
- **Tier-B n=10 / Tier-A n=16** — stated; effects are large and (Tier-A) 16/16 consistent; flagged in Limitations.
- **Prediction/monitoring AUROC** — explicitly called underpowered (5 positives) and future work; the causal
  result does not depend on it. OK.
- **COLLAPSE** — cited qualitatively ("0→1.0 at a small budget"); matches the measured ASR 1.0 @ ε=4 (random 0.0).
- **"performative reasoning ... now measured causally"** — defensible: we causally show CoT features do not move
  the answer in the 44/60 already-correct items.
- No "first-on-model" / "universal" wording. Faithfulness claimed as CONDITIONAL, performative as the common case.

## Verdict
**0 blockers.** Numbers reproduce exactly, citations verified, claims scoped with caveats up-front. The paper is
mint-ready pending the author's GO (mint is the irreversible step — held for explicit approval, per
`feedback_close_confounds_and_name_before_publishing`). Suggested before mint (optional, not blockers): a
per-layer figure; the companion synthesis paper ("Located, Not Secured") for cross-reference.
