# Announcement copy — WANDERING arc (papers #2/#3/#4 published)

Draft for Caio to post. Channels (per memory): X @0xCVYH, Farcaster, HF Community (146 followers), HackerNews, cold email to Anthropic interp / ICPs. **LessWrong/AF banned — do not post there.**

Permanent links:
- #2 https://doi.org/10.5281/zenodo.20490278
- #3 https://doi.org/10.5281/zenodo.20490284
- #4 https://doi.org/10.5281/zenodo.20490286
- arc mirror https://huggingface.co/datasets/caiovicentino1/wandering-arc-papers
- code https://github.com/OpenInterpretability/openinterp-swebench-harness

---

## X / Twitter thread (primary)

**1/**
New: 3 papers on **WANDERING** — when an LLM coding agent stays internally sure it solved the task but never hits "finish" and burns its whole turn budget. We can detect it. We tried to steer it back. Here's what actually worked. 🧵
(Qwen3.6-27B · SWE-bench Pro · all CC-BY)

**2/** The arc in one line:
detect → localize → **residual steering fails (3 nulls)** → **a one-line behavioral nudge works.**
The predictive signal lives in the residual stream; the causal lever lives in behavior.

**3/** Paper #2 — "The Right Locus Is Still Not a Rescue Lever."
We injected the SUCCESS direction at L11, the exact layer where WANDERING is most *detectable*. It does **not** rescue (paired McNemar p=0.73). Push harder and it just destabilizes the model into malformed tool calls (0→60%). A monitor is not a lever.
doi.org/10.5281/zenodo.20490278

**4/** The trap we almost shipped: a naive test vs a 0/20 baseline read p=0.02 "positive." But WANDERING isn't run-stable at temp 1.0 — the same 20 runs flip "finish" 7/20 on their own. Pair your tests, or you manufacture a result. We show the walk-back in full.

**5/** Paper #3 — multi-channel signatures.
60 features classify SUCCESS / LOCKED / WANDERING at macro-F1 0.636 (p=0.001), after an honest walk-back from a leaky 0.987. Stability selection independently rediscovers a mid-to-edge mechanism (LOCKED→L43, WANDERING→L11).
doi.org/10.5281/zenodo.20490284

**6/** The kicker in #3: the residual signature does NOT predict which agents respond to intervention (AUC 0.62). A cheap behavioral statistic — tool-entropy collapse depth — does (AUC 0.77). The actionable signal is behavioral, not residual.

**7/** Paper #4 — "Modality Matters" (the first positive).
Same 20 WANDERING runs. One fresh user turn at the collapse point ~doubles finalization (30%→70%, p=0.021). Residual injection stays inert. And it's the *interruption itself*, not the message content (a neutral nudge works as well as a re-plan, p=1.0).
doi.org/10.5281/zenodo.20490286

**8/** Why it matters: the "detection ≠ control" asymmetry that interpretability keeps hitting in single forward passes has a constructive way out for *agents* — detection doesn't give you a steerable direction, but it points to a steerable **action**.

**9/** All four papers, code, and data are open:
📄 arc mirror: huggingface.co/datasets/caiovicentino1/wandering-arc-papers
💻 code: github.com/OpenInterpretability/openinterp-swebench-harness
Honest nulls included on purpose. Feedback welcome.

---

## Short version (HF Community / Farcaster / HN comment / email)

**Three new CC-BY papers on agent WANDERING** (an LLM coding agent stays confident it solved the task but never finalizes, burning its turn budget), on Qwen3.6-27B / SWE-bench Pro:

- **#2** ([DOI](https://doi.org/10.5281/zenodo.20490278)) — injecting the SUCCESS direction at L11, the *strongest detector locus*, does **not** rescue WANDERING (paired McNemar p=0.73); at higher magnitude it just destabilizes the model. A monitor is not a lever. Includes the walk-back where a wrong (unpaired) baseline made the null look like a p=0.02 positive.
- **#3** ([DOI](https://doi.org/10.5281/zenodo.20490284)) — 60-feature classifier (macro-F1 0.636, honest walk-back from a leaky 0.987); the residual signature doesn't predict who responds to intervention, but tool-entropy collapse depth does.
- **#4** ([DOI](https://doi.org/10.5281/zenodo.20490286)) — the first positive: a transient behavioral interruption (one fresh user turn) ~doubles finalization (30%→70%, p=0.021) where residual steering fails — and it's the interruption, not its content.

The thesis: for long-horizon agents the predictive signal is residual but the causal lever is behavioral. Code + data: github.com/OpenInterpretability/openinterp-swebench-harness · mirror: huggingface.co/datasets/caiovicentino1/wandering-arc-papers
