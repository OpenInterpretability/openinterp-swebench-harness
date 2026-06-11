# Announcement copy — WANDERING arc (papers #2–#6 + companion note published — #6 is the FIRST POSITIVE)

Draft for Caio to post. Channels (per memory): X @0xCVYH, Farcaster, HF Community (146 followers), HackerNews, cold email to Anthropic interp / ICPs. **LessWrong/AF banned — do not post there.**

Permanent links:
- #2 https://doi.org/10.5281/zenodo.20490278
- #3 https://doi.org/10.5281/zenodo.20490284
- #4 https://doi.org/10.5281/zenodo.20490286
- companion note ("No Better Than Behavioral") https://doi.org/10.5281/zenodo.20500053
- #5 capstone ("The Verdict Is Not the Lever") https://doi.org/10.5281/zenodo.20532769
- **#6 first positive ("The Lever Is Late") https://doi.org/10.5281/zenodo.20534219**
- **#7 circuit-breaker capstone ("The Lever Generalizes — and It Brakes") https://doi.org/10.5281/zenodo.20634838**
- 🛠 tool `decision-locator` (pip-installable) https://github.com/OpenInterpretability/decision-locator
- arc mirror https://huggingface.co/datasets/caiovicentino1/wandering-arc-papers
- code https://github.com/OpenInterpretability/openinterp-swebench-harness

---

## X / Twitter thread (primary)

**1/**
New: 3 papers + a companion note on **WANDERING** — when an LLM coding agent stays internally sure it solved the task but never hits "finish" and burns its whole turn budget. We can detect it. We tried to steer it back. Here's what actually worked. 🧵
(Qwen3.6-27B · SWE-bench Pro · all CC-BY)

**2/** The arc in one line:
detect → localize → **residual steering fails (3 nulls)** → a behavioral nudge works → the named "done" feature still isn't the lever (#5) → **and we finally find where the internal lever IS: late, ~30 layers downstream of the verdict (#6, the first positive).**
The decision is *known* mid-stream but only *writable* late.

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

**9/** Companion note — "No Better Than Behavioral" (a pre-registered negative).
We asked the skeptic's question: does the *residual geometry* rot before the behavior, giving an earlier detector than the cheap tool-entropy signal? There is a real signal (failing runs' state "freezes" early, 5/5 layers) — but it ties the cheap signal (AUROC 0.70 vs 0.69) and is a worse alarm. The geometry is real but redundant. "Just watch the activations" loses to "watch the tool calls."
doi.org/10.5281/zenodo.20500053

**10/** Paper #5 — "The Verdict Is Not the Lever."
With a sparse autoencoder we find an interpretable "task-done" feature that IS present in WANDERING and predicts the finish action at AUROC 0.91 — yet clamping it does nothing (ΔP=−0.001 = random null). The agent represents "I'm done" cleanly, and that feature still isn't what makes it stop.
doi.org/10.5281/zenodo.20532769

**11/** Paper #6 — "The Lever Is Late" (the FIRST POSITIVE). So where IS the lever?
Logit-lens + activation patching across all 64 layers: the finish decision is invisible until the last ~12 layers (L51–63), ~30 layers downstream of the verdict. Transplant the SUCCESS late-block state into a WANDERING agent at its decision point and it **emits a real `finish` call 42% of the time** (p=0.031) — but only with a task-matched donor. The verdict is *known* mid-stream; it's only *writable* late.
doi.org/10.5281/zenodo.20534219

**12/** The law: the knowledge–action gap on agents is a **layer gap**. Control over a decision is null at the verdict (0%), real at the late task-matched residual (42%), and a behavioral interruption gives a comparable lift (#4). We ship the method as a tool — `decision-locator` finds & steers the commitment layer for any tool-calling decision on any open model.

**13/** Everything — five papers, the companion note, the tool, code, and data — is open:
📄 arc mirror: huggingface.co/datasets/caiovicentino1/wandering-arc-papers
🛠 tool: github.com/OpenInterpretability/decision-locator  (`pip install` + `decision-locator demo`)
💻 code: github.com/OpenInterpretability/openinterp-swebench-harness
Honest nulls + the positive that resolves them, all pre-registered. Feedback welcome.

**14/** (addendum, 2026-06-10) Paper #7 — "The Lever Generalizes — and It Brakes" (the circuit-breaker capstone).
Is the late lever specific to `finish`, or the action channel? We test a 2nd decision — commit an edit vs keep exploring. The same task-matched late patch **elicits** a real edit (0.23→0.77) AND **brakes** one (0.48→**0.02**, 96% off). Monotonic + bidirectional (elicit c=0, brake b=0; 7/7 Holm-sig). Geometry replicates on Mistral-7B + 24B.
The mechanism a safety circuit-breaker needs: one late patch blocks an action at its commit point. (Proxy = undoable edit; irreversible send_tx is next.)
doi.org/10.5281/zenodo.20634838

---

## Short version (HF Community / Farcaster / HN comment / email)

**Three new CC-BY papers on agent WANDERING** (an LLM coding agent stays confident it solved the task but never finalizes, burning its turn budget), on Qwen3.6-27B / SWE-bench Pro:

- **#2** ([DOI](https://doi.org/10.5281/zenodo.20490278)) — injecting the SUCCESS direction at L11, the *strongest detector locus*, does **not** rescue WANDERING (paired McNemar p=0.73); at higher magnitude it just destabilizes the model. A monitor is not a lever. Includes the walk-back where a wrong (unpaired) baseline made the null look like a p=0.02 positive.
- **#3** ([DOI](https://doi.org/10.5281/zenodo.20490284)) — 60-feature classifier (macro-F1 0.636, honest walk-back from a leaky 0.987); the residual signature doesn't predict who responds to intervention, but tool-entropy collapse depth does.
- **#4** ([DOI](https://doi.org/10.5281/zenodo.20490286)) — the first positive: a transient behavioral interruption (one fresh user turn) ~doubles finalization (30%→70%, p=0.021) where residual steering fails — and it's the interruption, not its content.
- **Companion note** ([DOI](https://doi.org/10.5281/zenodo.20500053)) — "No Better Than Behavioral": a pre-registered negative. The residual geometry *does* carry a context-rot fingerprint (failing runs "freeze" early, 5/5 layers), but it merely ties the cheap tool-entropy detector (AUROC 0.70 vs 0.69) and is a worse alarm — real but redundant. "Just watch the activations" loses to "watch the tool calls."
- **#5 capstone** ([DOI](https://doi.org/10.5281/zenodo.20532769)) — "The Verdict Is Not the Lever": with a sparse autoencoder we find an interpretable "task-done" feature (#22358) that is present in WANDERING (AUROC 0.81 vs LOCKED) and predicts the finish action (AUROC 0.91) — yet clamping it does NOT cause finishing (ΔP=−0.001 = random null). The agent represents "I'm done" as a clean, named, predictive feature, and that feature is still not the causal lever. Detection ≠ control, one level deeper.
- **#6 first positive** ([DOI](https://doi.org/10.5281/zenodo.20534219)) — "The Lever Is Late": so where IS the lever? Logit-lens + activation patching across all 64 layers shows the finish decision is invisible until the last ~12 layers (L51–63), ~30 layers downstream of the verdict. Transplanting the SUCCESS late-block state into a WANDERING agent at its decision point makes it emit a real `finish` call in 42% of cases (5/12, exact McNemar p=0.031) — but only with a task-matched donor (a coarse class mean is n.s.). The knowledge–action gap on agents is a **layer gap**: the decision is known mid-stream, writable only late. Shipped with `decision-locator`, a model-agnostic tool that finds & steers the commitment layer on any open model.

The thesis: detection ≠ control for long-horizon agents — but the gap closes. The decision is *known* mid-stream and *writable* late; the cheapest lever is behavioral, the internal lever is late and task-matched, and the verdict representation is not a lever at all. Code + tool + data: github.com/OpenInterpretability/openinterp-swebench-harness · mirror: huggingface.co/datasets/caiovicentino1/wandering-arc-papers
