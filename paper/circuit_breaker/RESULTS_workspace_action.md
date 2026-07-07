# RESULTS — The Action Lags the Answer (beat 11)

PREREG: `PREREG_workspace_action.md`. Models: **Qwen3.6-27B (dense, primary)** and **gpt-oss-20b
(MoE, cross-model)**. Ledgers (public): HF `caiovicentino1/swebench-phase6-verdict-circuit:`
`results/jspace_action.json` (+ `jspace_vectors.pt`, `jspace_resid.pt`) and
`results/jspace_xmodel_gpt-oss-20b.json`. Run: 2026-07-07, Colab CLI G4, detached-driver +
HF-ledger discipline (many VM incarnations, zero data loss).

**Reframed headline (depth-relative invariant):** the verbalizable direction reaches the agent's
**action commitment strictly deeper than it reaches the answer**. In both models there is a depth band
where the held answer is causally steerable and the committed tool is not; the *absolute* onset depths
are model-dependent (dense: answer@mid, action@tail; MoE: answer@tail, action@motor), so the honest
claim is the **answer→action depth lag**, not a fixed "mid-workspace" band.

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
not actions (0/60, worse than random)**. Both converge at the tail/motor. This is the depth lag on the
dense model: the answer is steerable a full band before the action.

**#3 — What does the mid-workspace action steer actually produce?** Decoding the global argmax under
the mid-workspace action steer confirms it is not a committed tool: the argmax is dominated by
structural tokens (`>` 47/60, newline 8/60), only **1/60** is a tool token, across just 6 distinct
tokens. So the midW intervention perturbs the among-tools balance (P(edit) 0.000/0.987) without
producing a clean committed emission — consistent with "action not yet steerable here."

## Cross-model replication (gpt-oss-20b, MoE, NL=24)
Arch-agnostic port (`scripts/jspace_xmodel.py`) on OpenAI's gpt-oss-20b, bands by depth%: midW
L11–16 (46–67%), tail L18–20 (75–83%), motor L22–23 (92–96%). Decision points built by the shared
`cross_model_realpts` construction; baseline commit fidelity is **weaker than the dense model** (edit
0.14 < bash 0.23), so this is a **corroborating** probe, not primary.

| band | ANSWER flip→alt (random) | ACTION flip→other (random) |
|---|---|---|
| midW (L11–16) | 1/20 (3) | 0/40 (9) |
| tail (L18–20) | **14/20 (0)** — specific | 0/40 (0) — not specific |
| motor (L22–23) | 20/20 (1) | 40/40 (4) |

**Same qualitative dissociation, shifted deeper:** on the MoE model the answer becomes specifically
steerable at the **tail** (14/20 vs random 0) while the action is still not steerable there (0/40);
both converge only at the motor band. The dense model's midW-answer/tail-action lag maps onto the
MoE's tail-answer/motor-action lag — the **depth lag is the invariant**, the absolute band is not.

## Headline (scoped per lit-scan, reframed)
**The action lags the answer in depth.** In both models the verbalizable direction reroutes the held
answer at a shallower depth than it can reroute the committed tool: there is a depth band where the
answer flips specifically (vs random 0) and the action does not (≤ random). The action does become
causally steerable via the *same* verbalizable direction deeper in the network (dense: tail L47+; MoE:
motor L22+) — so the action is **not absent from the workspace**, it is reached later. Ablating the
verbalizable subspace at the decision point leaves the commitment intact (H3). We **localize** the
(previously documented) knowledge–action gap as an **answer→action depth lag**. A J-lens-style
verbalizable monitor thus reads reasoning/answers a band *before* the action commitment is decided
(shown on two open-weights coding agents; not claimed architecture-general). The novelty is the
**depth-relative dissociation + null ablation, replicated across two architectures**, not the depth
ordering (tool signals are decodable early; outputs stabilize late — both known). This translates the
arc (lever-is-late, the late channel) into the global-workspace vocabulary, on the week of that
paper's release.

## Why the action lags (mechanism, #2 — from arc beat-7)
Our prior circuit analysis of this agent (`vicentino2026brake`, DOI 10.5281/zenodo.20679287 and the
L59 head decomposition) found the commit is written by ~3 late attention heads at L59 acting as
induction/copy heads: they promote the tool-name token by **copying it from the tool-call history**,
not composing it from mid-network concepts. A late copy is inherently a *motor* computation,
downstream of and orthogonal to the mid-network concept broadcast the verbalizable workspace steers —
consistent with the observed lag and with the commitment surviving ablation of the verbalizable
subspace. A clean head-level test is future work (L59 attention is output-gated → per-head write
reconstruction is nontrivial).

## Novelty and related work (post-result lit-scan, verdict SOFTEN — no direct hit, nearest neighbor 2605.07990)
No published work tests whether an agent's action/tool-choice commitment routes through the EMERGENT
global workspace; the review of the Anthropic paper explicitly names this as unexamined (the Nanda
replication covers completions/reasoning only). The load-bearing contribution is the **answer→action
depth-lag dissociation via the shared verbalizable (J-space) direction, replicated dense-27B + MoE-20B**;
the null ablation is an INSTANCE of the detect≠control pattern, not a standalone novelty; the raw
depth ordering and the knowledge–action gap are NOT claimed. Components softened accordingly:
- **vs "Tool Calling is Linearly Readable and Steerable" (arXiv:2605.07990, Wu et al., NEAREST NEIGHBOR).**
  They causally steer *which tool* via a bespoke tool direction and report early-readable/late-steerable
  ordering, in DENSE transformers (Gemma3/Qwen3/Qwen2.5/Llama3.1). We do NOT claim that ordering — it is
  theirs. Our contribution is orthogonal: (a) the answer-vs-action dissociation (they never contrast the
  two), (b) the shared EMERGENT verbalizable/J-space direction (not a bespoke tool direction), (c)
  extension to an MoE architecture (gpt-oss-20b) not among their dense families, (d) null ablation of the
  verbalizable subspace. Cite prominently; attribute the depth ordering to them + 2605.27935 (MoE).
- **vs Basu (arXiv:2603.18353, knowledge–action gap).** Prior work shows internal probes separate
  cases the model fails to act on (98% vs 45%) but treats it as an actionability limit; we do not
  claim to discover the gap — we **localize** it as an answer→action depth lag and show the
  verbalizable direction that steers the *answer* at a shallower depth (L27–43) is not yet reachable
  by the *action* commitment there, which becomes specifically steerable only deeper (L47+).
- **vs "LLM Agents Already Know When to Call Tools" (arXiv:2605.09252).** They read *whether* a tool
  is needed (linear probe); we study *which* tool is committed and its causal routing — dissociated
  from the mid-network workspace direction, with ablation leaving the commitment intact.
- **vs "Tool-Call Dependency ... Linearly Decodable" (arXiv:2605.25310, decodable ~L14) and
  construction-refinement depth dynamics (arXiv:2605.27935).** Tool-related signals are linearly
  *readable* early and outputs stabilize late in general; we therefore scope our claim to **specific
  causal steerability of the which-tool commitment via the verbalizable direction**, not readability
  in general. The load-bearing evidence is the ablation-leaves-action-intact result (H3) and the
  answer-vs-action steering contrast (H2 + answer_bands), foregrounded to rebut "you are re-reading
  an early-decodable tool signal."

## Honest scope
Two open-weights models (Qwen3.6-27B dense primary; gpt-oss-20b MoE corroborating), single behavior
family (edit vs bash), n=60 action / 20 answer points on the primary, prefill-only, α=0.5 (magnitude
controlled within each band by the random baseline). **The depth lag replicates in both models but the
absolute onset depths are model-dependent** — hence the depth-relative framing, not a fixed
mid-workspace band. On the MoE the synthetic decision point is a weaker probe (edit-commit fidelity
0.14 < bash 0.23), so its dissociation corroborates rather than leads. Disclosed deviation: steer in
place of the pre-registered coordinate swap (the swap is a near-no-op in the raw-gradient basis; both
are Anthropic-used; the steer is specificity-validated). The estimator is the row-restricted variant,
not the full J_ℓ matrix. No claim about Claude-scale models. The "monitoring reads the answer before
the commitment" implication is scoped to J-lens-style verbalizable monitors.

## Statistics
All positive dissociations significant vs their magnitude-matched random controls (Fisher exact):
answer midW 11/20 vs 0/20 → p=1.5e-4; action tail 60/60 vs 23/60 → p=3.8e-15; action motor 60/60 vs
21/60 → p=2.8e-16; gpt-oss answer tail 14/20 vs 0/20 → p=3.3e-6; gpt-oss action motor 40/40 vs 4/40 →
p=2.5e-18. (The n=20/60 objection is moot given the effect sizes.)

## Independent reimplementation (verify field)
A second implementation of the steer/ablation pipeline, with FRESH random seeds, reproduced all six
targeted counts EXACTLY (answer midW/tail/motor 11/20/20, action 0/60/60) and the ablated differential
(0.167). The targeted counts are seed-independent (identical across implementations/seeds); only the
random controls vary (fresh-seed 22/21/22 vs original 21/23/21). Logged to ledger `verify` field →
this closes the "the causal numbers aren't CPU-recomputable" gap: they ARE GPU-reproducible bit-for-bit.

## Reproducibility
Public ledgers + vectors + residuals on HF (dense: `results/jspace_action.json`; MoE:
`results/jspace_xmodel_gpt-oss-20b.json`); harnesses `scripts/jspace_action.py`,
`scripts/jspace_xmodel.py` + PREREG public. H1 AUROC recomputes offline (CPU) from the
vectors/residuals to 1e-6 (16/16 layers). `scripts/eval_jspace.py` = 41/41 PASS (H1 recompute + both-
model dissociation logic + verify-field cross-check + Fisher significance). Steer/ablation numbers are
model-dependent; each pipeline step was verified present in the HF ledger (not string-matched) + the
independent `verify` reimplementation reproduced the targeted counts.
