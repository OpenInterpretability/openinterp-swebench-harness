# PRE-REGISTRATION — Verdict Circuit Stage-1 (model-in-the-loop): interpret the feature, test the readout

**Author:** Caio Vicentino · OpenInterpretability · pre-registered 2026-06-02 (before running the model).
**Gated by:** Stage-0 PRELIMINARY GO (`STAGE0_RESULTS.md`): a SUCCESS-selected SAE feature (L23 #22358, L43 #908) fires at WANDERING-final but not LOCKED-final. **Stage-1's FIRST job is to try to FALSIFY caveat #3** — that the feature is a generic "active-at-the-end" feature, not a "task-done" verdict.
**Compute:** Qwen3.6-27B loaded (GPU). Notebook `nb_verdict_circuit_stage1.ipynb`.

---

## H1 — Interpretation (PRIMARY; gates everything)
The feature (L23 #22358) is semantically a **task-completion / conclusion** feature: its top-activating token contexts — on a general corpus AND inside the SWE-bench trajectories — are completion/finishing language (e.g. "done", "the fix is complete", patch-submission, the `finish` tool call), **not** generic code-editing, generic late-context, or arbitrary tokens.
- **Method:** run model + L23 SAE on (a) a general code+text corpus and (b) a sample of trajectory transcripts; record per-token activation of #22358; collect the top-K activating windows.
- **GATE H1:** a blind judge (Caio, and/or an LLM judge) labels the top-20 contexts. **PASS iff a clear majority (≥12/20) are completion/conclusion-related.** If they are generic "active/long-context/editing", **caveat #3 is NOT falsified → honest negative: "WANDERING looks active-at-the-end, not done" → STOP** (still publishable, it bounds the §13 claim). Only on PASS do we proceed to H2–H4.

## H2 — Readout sanity (the behavioral fact at the logit level)
At the `pre_tool` action-selection point (right after `<tool_call>{"name": "`, where the model chooses among `bash` / `str_replace_editor` / `finish`), the model's **P(finish)** is high at SUCCESS-final and low at WANDERING-final. (Confirms the captures sit at the real decision point; a sanity check, expected true by construction.)

## H3 — Direct readout of the feature (DLA)
The done-feature's decoder direction `W_dec[#22358]` (at L23), pushed through the final norm + unembedding, contributes **positively to the `finish` logit** (more than to `bash`/`str_replace_editor`). I.e. activating the feature directly votes "finish". (Direct path only — L23→logits skips L24–63; supplementary to H4.)

## H4 — The §13 second-half mechanism test (CAUSAL; the decisive question)
WANDERING holds the verdict (Stage-0) but doesn't emit finish. Is the verdict **sufficient** for the action, or is the **readout broken downstream**? At the WANDERING-final `pre_tool` position, **clamp feature #22358 to the SUCCESS-final level** (via `+ (target − current)·W_dec[f]` at L23) and run the full forward pass; measure **ΔP(finish)**.
- **Necessity check:** ablate #22358 (zero it) at SUCCESS-final → pre-registered expectation ΔP(finish) **< 0**.
- **Pre-registered readouts of the clamp (both publishable):**
  - **(a) ΔP(finish) ≈ 0** (clamp does not raise finish): the verdict is present but **not sufficient** — the verdict→action readout is broken downstream → **§13 confirmed; detection ≠ control holds even at the right feature** (consistent with the arc's residual nulls, now sharpened to "even the verdict feature isn't a lever").
  - **(b) ΔP(finish) ≫ 0** (clamp raises finish substantially): the verdict **was** the missing piece → a **surgical, feature-level lever where dense residual steering failed**. This would contradict the arc's residual nulls, so it is held to a HIGH bar: it must (i) beat a random-active-feature clamp of equal magnitude, (ii) replicate at L43 #908, and (iii) be confirmed behaviorally (does the agent actually emit finish in a real continuation, not just a one-token probability bump). We do **not** assume (b); we report whichever fires.

## Controls (mandatory)
- **C-rand:** clamp a *random active* feature of equal magnitude → null; #22358's ΔP(finish) must exceed it.
- **C-pair:** per-trajectory, paired (same instances baseline vs clamped); report effect size, not just p.
- **C-L43:** any positive H4 must replicate at L43 #908 or it is single-layer-fragile.
- **C-runstab:** WANDERING is not run-stable at temp 1.0 — any *behavioral* (generation) confirmation must be paired and compared to a no-clamp re-run, never to the definitional baseline (the arc's hard-won rule).
- **Stage-0 strengthening (CPU, in-notebook):** redo the W>L specificity with a **firing-frequency-matched** feature null and a **label-shuffle** selection null (Stage-0's random-feature null was degenerate under TopK sparsity).

## Gates summary
H1 FAIL → STOP (honest negative, bounds §13). H1 PASS → run H2–H4. H4 outcome (a) → §13 mechanistically confirmed (verdict present, readout broken). H4 outcome (b) → extraordinary claim, hold to C-rand + C-L43 + behavioral confirmation before any external claim. Reported whichever way it falls; no salvage.

## Artifacts
`scripts/build_nb_verdict_circuit_stage1.py` → `notebooks/nb_verdict_circuit_stage1.ipynb`. Outputs saved to Drive `openinterp_runs/verdict_circuit_stage1/`. The model run is the only GPU step in the whole verdict-circuit program.
