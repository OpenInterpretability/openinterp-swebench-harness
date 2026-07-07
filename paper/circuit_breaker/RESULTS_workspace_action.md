# RESULTS — Is the Agent's Action in the Workspace? (beat 11)

PREREG: `PREREG_workspace_action.md`. Model: Qwen3.6-27B. Ledger (public):
HF `caiovicentino1/swebench-phase6-verdict-circuit:results/jspace_action.json`
(+ `jspace_vectors.pt`, `jspace_resid.pt`). Run: 2026-07-07, Colab CLI G4, detached-driver +
HF-ledger discipline (many VM incarnations, zero data loss).

## Question
Anthropic's "A Global Workspace in Language Models" (transformer-circuits.pub/2026/workspace,
2026-07-06) shows a verbalizable **J-space** whose coordinate swaps causally flip multi-hop
**answers** at mid-depth. Their paper has no agent/action content. Does an agent's **action
commitment** pass through the same verbalizable workspace, or bypass it?

## Instrument (and an honest debugging chain)
**Scalar-row J-lens estimator.** For token *v* and layer ℓ, the J-lens row is the residual-space
direction g_{v,ℓ} = ∂ logit_v(final position) / ∂h_{ℓ,final}, one backward per (v, prompt), params
frozen, embedding made the sole grad leaf. Estimated over agent-distribution windows (actions) and
generic text (answer-pool tokens).

Four plumbing bugs and one specification bug were caught by gates-first discipline **before any
scientific number was produced** — recorded here for reproducibility:
1. **OOM.** HF gradient checkpointing only fires in `train()` mode (`self.training` guard); the 27B
   backward stored all 64 layers and killed the kernel. Fix: `train()` during estimation
   (decoder LLM has no active dropout → behaviorally identical).
2. **Malformed positive control.** Zero-shot 2-hop QA gave 0/20 baseline accuracy on a reasoning
   model. Fix: few-shot 2-hop (2 solved demos) → 20/20 baseline.
3. **Driver false-positive.** `colab exec` echoes submitted code, so a grep on the output matched a
   success string even when the step crashed. Fix: verify the HF-ledger key, not the output text.
4. **Device mismatch** (CPU vectors vs CUDA hidden) in two hooks. Fixed.
5. **Estimator specification.** The first estimator summed the gradient over ALL source positions
   (`Σ_t logit_v(t)`, normalized by future-count) — this diluted the token-specific direction to
   near-zero norm (0.007) and it failed the pre-registered coordinate-swap positive control (0/20
   flips). Corrected to the **final-position diagonal** (above): norm rose to 0.28.

**Why the pre-registered swap failed, and the disclosed deviation.** With the corrected estimator,
the pre-registered coordinate-**swap** (h ← h + V(σ(c)−c)) still gave 0/20 answer flips — because
the raw-gradient projections are near-equal (c_correct ≈ c_alt), making the exchange a near-no-op.
A separate validity worry (readout correlation ρ ≈ 0 at L63) turned out to be a **confound**: RMSNorm
is degree-0 homogeneous, so by Euler's theorem ∂norm/∂h·h = 0 and g_v·h ≈ 0 *by construction*,
independent of estimator quality. The decisive instrument check is therefore a **specificity control**,
not ρ:

**Specificity control (steer_control).** A steer along the normalized (g_alt − g_correct) direction,
scaled to α·‖h‖, at the workspace band:

| α | contrast→alt | unrelated-pair→alt | random→alt | random moved off correct |
|---|---|---|---|---|
| 0.5 | **17/20** | 0/20 | 0/20 | 0/20 |
| 1.0 | **20/20** | 0/20 | 0/20 | 19/20 |

The estimated directions are **specific and causal**: at α=0.5 a random half-residual-norm push does
not even move the answer, while the contrast direction flips it to the alternative 17/20; unrelated
and random directions flip 0. We therefore proceed with the **steer** (also an Anthropic-used
intervention) in place of the swap, **disclosed as a deviation** from the pre-registered coordinate
exchange, on the same validated directions.

## Results

**H1 (readout by depth).** AUROC of (g_edit − g_bash)·h separating 60 edit-decision points from 60
bash-decision points, per layer. Weak/inverted early, crosses 0.5 at ~L35, **peaks 0.743 at L47
(73% depth)**, sustains 0.59–0.69 through L55–L63. The action becomes linearly readable in the
verbalizable direction only in the **workspace tail / late band**, not mid-workspace.
*Independently recomputed from public vectors+residuals on CPU: 16/16 layers match the ledger to 1e-6.*

**H2 (causal steer, α=0.5, band-wise, with random control).** Steer the commitment along
(g_other − g_taken) at the decision points; count clean flips to the alternative tool (full-vocab
argmax) vs a magnitude-matched random-direction control:

| band | edit→bash | bash→edit |
|---|---|---|
| midW (L27–43) | 0/60 (random 21) | 0/60 (random 15) |
| tail (L47–55) | 60/60 (random 23) | 44/60 (random 15) |
| motor (L59–63) | 60/60 (random 21) | 60/60 (random 15) |

At mid-workspace the targeted steer is **worse than random** (0 < 21) — no specific
workspace-mediated action rerouting. Specific control emerges at the **tail** and completes at
**motor**. (Among-tools P(edit) does shift at midW — 0.000/0.987 — but the global argmax is not a
tool token, i.e. the mid-workspace intervention perturbs the tool balance without producing a clean
committed emission.)

**H3 (workspace-subspace ablation).** Projecting out the top-10 verbalizable directions (by lens
activation on the agent distribution) at the decision points leaves the commitment differential
essentially intact: P(edit) edit-pts 0.474→0.473, bash-pts 0.299→0.306; differential 0.174→0.167.
The commitment does not live in the verbalizable subspace at the decision position.

**Decisive dissociation control (answer_bands).** The specificity control above used the full
workspace band (L27–55, which includes the tail); to make the answer/action comparison
apples-to-apples we steer the **answer** per-band, exactly parallel to H2:

| band | ANSWER flip→alt (random) | ACTION flip→other (random) |
|---|---|---|
| **midW (L27–43)** | **11/20 (0)** — specific | **0/60 (21)** — not specific |
| tail (L47–55) | 20/20 (0) | 60/60 (23) |
| motor (L59–63) | 20/20 (0) | 60/60 (21) |

At mid-workspace the verbalizable direction **specifically reroutes answers (11/20 vs random 0) but
not actions (0/60, worse than random)**. Both converge at the tail/motor.

## Headline
**The agent's action commitment bypasses the mid-workspace where the answer is already steerable.**
Verbalizable content (a 2-hop answer) is causally reroutable from L27 (42% depth); the action only
becomes readable (H1) and specifically steerable (H2) at the workspace→motor boundary (L47+, 73%+),
and the commitment survives ablation of the verbalizable subspace (H3). The knowledge–action gap is a
**workspace→motor gap**: a J-lens-style verbalizable monitor reads reasoning/answers but not action
commitments. This translates the arc (lever-is-late, the late channel) into the global-workspace
vocabulary, on the week of that paper's release.

## Honest scope
Single model (Qwen3.6-27B), single behavior family (edit vs bash), n=60 action / 20 answer points,
prefill-only, α=0.5 (magnitude controlled within each band by the random baseline). Disclosed
deviation: steer in place of the pre-registered coordinate swap (the swap is a near-no-op in the
raw-gradient basis; both are Anthropic-used; the steer is specificity-validated). The estimator is
the row-restricted variant, not the full J_ℓ matrix. No claim about Claude-scale models. The
"monitoring misses commitments" implication is scoped to J-lens-style verbalizable monitors.

## Reproducibility
Public ledger + vectors + residuals on HF; harness `scripts/jspace_action.py` + PREREG public.
H1 AUROC recomputes offline (CPU) from the vectors/residuals to 1e-6 (16/16 layers). Steer/ablation
numbers are model-dependent; each pipeline step was verified present in the HF ledger (not
string-matched) by the driver.
