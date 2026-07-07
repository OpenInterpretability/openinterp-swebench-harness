# PREREG — Is the Agent's Action in the Workspace? (J-lens × commitment)

**Question:** Does an agent's action commitment pass through the global workspace (the verbalizable
J-space of Anthropic's "Global Workspace in Language Models", transformer-circuits.pub/2026/workspace,
published 2026-07-06) — or does it bypass it, entering verbalizable coordinates only in the motor regime?

**Date:** 2026-07-06 · **Status:** PRE-REGISTERED, not run. GPU gated on Caio OK. **Candidate arc beat 11.**
**G-1 lit-check: PASSED (2026-07-06, agent litcheck-jspace) with binding reframes:**
1. **The Qwen port is NOT a contribution:** Neuronpedia already serves J-lens readouts for our exact
   model (neuronpedia.org/qwen3.6-27b/jlens), and Nanda + MATS scholars (Blank, Bhatia) already
   replicated J-lens on Qwen3.6-27B. Cite as infrastructure; our G0′ gains a cross-check against
   Neuronpedia's readouts. The contribution is the ACTION axis only.
2. **Must-differentiate (the dangerous neighbor):** arXiv:2605.09252 "LLM Agents Already Know When to
   Call Tools" — linear probes on hidden states predict WHETHER-to-call (AUROC 0.89–0.96, incl. Qwen3).
   Our differences, stated explicitly in any writeup: action SELECTION (which tool) not binary need;
   verbalizable-workspace structure (J-lens) not plain probing; CAUSAL coordinate-swaps not readout.
   Light cites: 2606.02871, 2605.28214 (latent agentic reasoning).
3. **GWT×agents prior art = IMPOSED architectures** (VanRullen & Kanai 2012.10390; Goyal 2103.01197;
   2503.01906; Unified Mind Model 2503.03459; 2410.11407; Butlin/Long 2023 indicators): all DESIGN a
   workspace in. Nobody tests whether a trained LLM's action commitment ROUTES through the EMERGENT
   workspace. That is the gap we occupy.
4. **Depth arithmetic resolved (was a real tension) — SOURCE-VERIFIED 2026-07-06:** the paper states
   layers are "reindexed to the range [0–100] so that layer numbers can be interpreted as percentages"
   (verified by direct fetch; Claude's true layer count undisclosed). Their bands on Qwen3.6-27B's 64 layers:
   sensory <L24, workspace ≈ L25–L58 (38–92%), motor ≈ L59–L63 (>92%). Our action band L51–63
   STRADDLES the boundary: L51–55 = workspace tail; L59–63 = motor. The commit writers (L59 = 92.2%)
   sit exactly at the motor edge. Predictions and bands below are restated two-sidedly: the decisive
   contrast ("answers flip @mid-workspace, actions do not") works regardless of where the action
   turns out to live, and H1 maps the actual profile.

## Motivation

Anthropic's paper establishes, on Claude models, a three-regime depth structure — sensory (~first
third), **workspace** (~38–92% depth: abstract, persistent, verbalizable content), **motor**
(~92–100%: imminent output) — where J-lens coordinate swaps causally flip multi-hop **answers**
(54–70% top-1) and top-k=10 J-space ablation destroys multi-hop reasoning while sparing parsing. The
paper contains **no agent/action/tool-use content**. Our arc supplies the exact complementary prior:
the verdict predicts-but-does-not-cause (mid-depth, beat 3), the action lever is late (L51–63 = 80–98%
depth, beats 4–6), and the commit writers (L59 = 92%) sit at the workspace→motor boundary and are
induction/copy heads that *promote tool-name tokens*. In the workspace vocabulary: **our published arc
predicts the knowledge–action gap is a workspace→motor gap.** If true, J-lens monitoring reads
thoughts but not commitments — directly relevant to their monitoring agenda.

## Hypotheses (pre-declared, with priors)

- **H1 (readout — where does the action become verbalizable?).** J-lens score of the taken-action
  token (vs the alternative) at the decision position, across ~16 depth points: prior = discriminative
  signal (AUROC edit-pts vs bash-pts) at chance in mid-workspace, rising across the workspace tail
  into motor (their-def boundary L58.9 = 92%); the profile itself is a deliverable. Alternative outcome: action readable mid-workspace → actions ARE
  workspace content (revises our reading; publishable).
- **H2 (causal — the head-to-head).** J-lens coordinate swap (their h ← h + V(σ(c)−c), swapping
  bash↔str_replace_editor coordinates, orthogonal complement preserved) at the decision position,
  in THREE fixed bands (depth arithmetic above):
  - @ mid-workspace (L27–L43, 42–67%): prior = does NOT flip the action (verdict-not-lever).
  - @ workspace-tail (L47–L55, 73–86%): no strong prior — this is where H1 maps the transition.
  - @ motor (L59–L63, 92–98%): prior = flips the action (lever-is-late).
  - The interpretive fork requires G0′ (below): on the SAME model, workspace swaps must flip multi-hop
    ANSWERS — so the contrast "answers flip @workspace, actions don't" is same-model, same-method.
- **H3 (workspace ablation).** Project out the top-k=10 J-space directions (their ablation) at the
  decision position across the workspace band: prior = the task-appropriate commitment differential
  (emit 0.48 @edit-pts vs 0.23 @bash-pts) survives → the commitment computation does not require the
  verbalizable workspace at the decision position. If it dies → commitment depends on workspace
  content upstream (consistent with the distributed brake; also publishable).

**Pre-declared outcome → claim table:**

| G0′ answers @W | actions @W (H2) | actions @motor (H2) | Allowed claim |
|---|---|---|---|
| flip | no flip @mid-W | flip @motor | "Actions bypass the (mid-)workspace: the knowledge–action gap is a workspace→motor gap; J-lens monitoring reads thoughts, not commitments." (headline) |
| flip | flip | flip | "Action commitment is workspace-mediated — revises lever-is-late: the late lever is workspace broadcast reaching motor." |
| flip | no flip | no flip | "Action commitment lives outside verbalizable (J-space) coordinates at every depth — reachable only by full-residual patching (known 0.77 @L59). The verbalizable subspace misses commitments entirely." (strongest monitoring implication) |
| no flip | — | — | Implementation failure or model lacks Claude-like workspace causality → report as replication-scoped null; NO action-claims licensed. |

## Method

**J-lens vectors (scalar-row estimator, pre-declared).** For token v and layer ℓ, the J-lens vector is
the residual-space direction g_{v,ℓ} = ∂[mean over future positions t′ of logit_v(t′)]/∂h_{ℓ,t},
estimated by ONE backward pass per (v, prompt) — gradients at all layers/positions in one pass; the
causal mask automatically restricts to t′ ≥ t; per-t normalization by count of future positions;
averaged over an estimation set. Tokens: the 3 action tokens (single-token ids verified: finish=28,
bash=21402, str_replace_editor=15462) + ≥6 control tokens (random mid-frequency). Estimation set:
150 prompts from the AGENT distribution (trajectory prefixes, truncated ≤2K) — deviation from their
pretraining-like set, disclosed (we need the lens calibrated on-distribution; robustness check: 50
generic-text prompts).

**Readout (H1).** Score captured decision-point residuals h_{ℓ} (60+60 points, ~16 layers) against
g_{v,ℓ}: rank of action tokens within {actions ∪ controls}; AUROC(upcoming action) per depth.

**Swaps (H2).** At the decision position (prefill-only, matching the #7 protocol): V = [g_edit, g_bash]
at each layer of the band, c = V⁺h, h ← h + V(σ(c)−c) with σ = coordinate exchange; applied at every
layer of the band simultaneously (their band-wise application). Readout: P(edit) + greedy emit, n=60
per side. Controls: norm-matched random-direction swap (null); σ=identity (must be exact no-op, gate);
full-residual donor patch @L59 (positive control, known 0.77/0.02).

**Ablation (H3).** h ← h − V_k V_k⁺ h with V_k = top-k=10 J-space directions (by lens activation on the
agent distribution) at the decision position, workspace band.

**Stats.** Exact McNemar per-point vs base for every condition (house standard); AUROC with split-half
stability for H1; all conditions and cutbands fixed here — no post-hoc bands.

## Gates

- **G-1 (overclaim/lit):** agent litcheck-jspace must return no DIRECT-HIT on "workspace × agent
  action" (incl. Nanda commentary + Neuronpedia open-model J-lens coverage + GWT-LLM prior art:
  VanRullen/Kanai, Goyal, Butlin-Long indicators). Must-cites incorporated before any writeup.
- **G0 (harness fidelity):** resumed decision points reproduce baselines (emit 0.483/0.233; live
  prefix check as in beat 10).
- **G0′ (J-lens implementation + same-model workspace causality — THE key gate):** (i) sanity: at
  late layers on generic text, lens readout ranks the actually-upcoming token highly (vs logit-lens
  baseline), AND our estimated vectors' readouts cross-check against Neuronpedia's published Qwen3.6-27B
  J-lens readouts on shared prompts/tokens (rank correlation, pre-registered ≥0.5); (ii) null swap σ=identity is an exact no-op; (iii) POSITIVE CONTROL: on a 20-item 2-hop
  QA set (constructed, disclosed), workspace-band J-lens answer-swaps flip ≥25% of answers top-1
  (generous threshold vs their 54–70% on Claude; if <25%, STOP — report implementation-scoped null,
  no action-claims).
- **G1 (readout sanity):** some depth shows action-token signal above the control-token null
  (split-half stable); if nowhere, H1 is itself the "not verbalizable anywhere" branch — proceed to
  H2 which is decision-relevant regardless.

## Compute plan

Colab CLI G4 (RTX6000 96GB), detached-driver + HF-ledger discipline (`results/jspace_action.json`).
Estimated: J-lens vector estimation ~600–1000 backwards @≤2K ctx (~1–2h; OOM fallback: ctx 1K /
gradient checkpointing) · capture extra layers ~30min · G0′ QA control ~20min · H1 readout ≈ CPU ·
H2 swaps 3 bands × (60+60) × 2 readouts + nulls ~2h · H3 ~30min. **Total ≈ 4–6 GPU-h** (beat-10
class). Script: `scripts/jspace_action.py` (to be written after G-1), reusing `commit_lever_stages`.

## Honest scope (pre-declared)

Single model (Qwen3.6-27B), single behavior family (edit/bash commitment), n=60/side. The J-lens is
their construct: single-token concepts only (our action tokens qualify); our estimator is the
row-restricted variant (documented above), not the full J_ℓ matrix — differences disclosed. Estimation
on agent distribution is a deliberate deviation. No claim about Claude-scale models; no claim that
the workspace does not exist in Qwen (G0′ tests it); the "monitoring misses commitments" implication
is conditional on the headline branch and scoped to J-lens-style verbalizable monitors.

## Relation to the arc

Beat 3 (verdict-not-lever) + beats 4–6 (lever-is-late) + beat 8 (late channel) supply the priors;
beat 10 supplies the audit discipline. This experiment translates the arc into the workspace
vocabulary on the week of its publication: if the headline branch lands, the knowledge–action gap
becomes a statement about global workspace theory in LLM agents — and a scoped caveat for
workspace-based monitoring agendas.
