# PREREG — Decomposing the late commitment-lever into Attention vs MLP

**Date:** 2026-06-11 · **Author:** Caio Vicentino · **Status:** pre-registered before run.
**Builds on:** Paper #7 "The Lever Generalizes — and It Brakes" (Zenodo 10.5281/zenodo.20634838).
Reuses the exact harness `scripts/commit_lever_stages.py` (model Qwen/Qwen3.6-27B, the same EDIT/BASH
decision points, donor selection, and P(action) metric), so every number here is directly comparable
to the published H1/H2/dense panels.

## 1. Question
Paper #7's lever is a **residual-stream** patch at a late layer L∈{55,59}: overwrite the last-token residual
output `y_L` of a WANDERING/bash point with a task-matched donor's `y_L`. This **elicits** the action
(bash-point edit-rate 0.23→0.77@L59 / 0.62@L55) and, in reverse (bash-donor into an edit-point), **brakes**
it (0.48→0.02@L55). But `y_L` is the *sum* of everything written so far. **Through which sublayer does the
late lever write the action decision — attention or MLP?**

- **H_MLP** — the commit is a feed-forward write (Geva et al. key-value memory writing an "emit-action"
  feature). Prediction: re-injecting only the donor's **MLP**-component recovers most of the lift; the brake
  works by cancelling that MLP write.
- **H_ATTN** — the commit is attention-routed (the late block copies/promotes the consolidated verdict /
  action token from context). Prediction: only the **attention**-component recovers the lift.
- **H_SPLIT** — distributed: both sublayers carry it (redundant), or elicit goes through one and the brake
  through the other (the bidirectional asymmetry could localize differently).

Honest prior: the arc's logit-lens showed the S−W gap *explodes* L51→L63 ~30 layers after the verdict is
already formed — a late, abrupt write. That is *consistent with either* a feed-forward commit (MLP) or a
late attention pull of the consolidated verdict. We do not know; the causal patch decides.

## 2. Exact, additive decomposition (the method)
For a pre-norm block, `y_L = x_L + a_L + m_L` where `a_L = self_attn(ln1(x))`, `m_L = mlp(ln2(x+a))` are the
**additive contributions to the residual** (validated empirically: we capture `x_L` via a pre-hook and assert
‖(x+a+m) − y‖/‖y‖ < 1e-2 before trusting any result). The published full patch is the residual delta
`Δ_full = y_donor − y_point = Δx + Δa + Δm`, which is **exactly additive**. We therefore re-inject each piece
alone by an **additive** hook on the layer-L output (`y += δ`, last token, prefill only):

| condition | δ added to `y_L(point)` |
|---|---|
| `full`     | `y_donor − y_point` (== paper #7 patch, sanity-gated vs published) |
| `inp`      | `x_donor − x_point` (accumulated upstream, ≤ layer L−1) |
| `attn`     | `a_donor − a_point` (the layer-L attention write only) |
| `mlp`      | `m_donor − m_point` (the layer-L MLP write only) |
| `attnmlp`  | `Δa + Δm` (both layer-L sublayers, excludes upstream) |
| `rand_attn`| `a_donor[cross-repo] − a_point` (direction control for the attn channel) |

The components **add to `Δ_full` exactly in residual space**; their downstream effects on P(action) need NOT
add (the network is nonlinear). The residual gap `Δ_full − (Δ_inp+Δ_attn+Δ_mlp)` in *P-space* is reported as
the **interaction/nonlinearity** term — itself informative.

## 3. Conditions (both directions, both layers L∈{55,59})
- **Elicit** (rows = BASH points, donor = same-repo EDIT, action = `str_replace_editor`): expect `full` ΔP > 0.
- **Brake** (rows = EDIT points, donor = same-repo BASH, action = `str_replace_editor`): expect `full` ΔP < 0.
- Primary metric: per-point **ΔP(edit) = P_patched − baseP_edit**, n=60/condition (cheap single-forward,
  deterministic, identical to the dense panel of #7).
- Behavioral confirmation: **emit-rate** under generation for `{full, attn, mlp}` at the headline layer
  (elicit L59, brake L55) + a no-patch baseline, to tie to the published 0.23/0.48 rates.

## 4. Decision rule (pre-committed)
Let `r_attn = mean(ΔP_attn)/mean(ΔP_full)` and `r_mlp = mean(ΔP_mlp)/mean(ΔP_full)` (recovery fractions; for
the brake, signs make both positive when a component suppresses).
- **Attention-localized** if `r_attn ≥ 0.50` AND `r_mlp < 0.25`.
- **MLP-localized** if `r_mlp ≥ 0.50` AND `r_attn < 0.25`.
- **Distributed/redundant** if both ≥ 0.50.
- **Split/additive** if neither ≥ 0.50 but `r_inp+r_attn+r_mlp ≈ 1` (the lever lives partly in the upstream
  residual carried *into* L).
- **Direction-specificity required:** `rand_attn` ΔP must be ≪ task-matched `attn` ΔP (else the channel
  effect is not direction-specific and that channel's claim is void).
- Paired **Wilcoxon signed-rank** on per-point ΔP for `attn` vs `mlp` reports whether the two channels differ.
- The elicit and brake loci are reported **separately** — the headline of #7 is that the lever is bidirectional;
  this prereg explicitly allows them to localize to different sublayers.

## 5. Controls & sanity gates (must pass before interpreting)
1. **Architecture validity:** residual reconstruction relerr < 1e-2 (else the additive split is wrong — abort).
2. **Reproduce #7:** `full` ΔP@L55/L59 must match the published dense `edit_donor` ΔP within noise; `full`
   emit-rate must reproduce ~0.77 (elicit L59) and ~0.02 (brake L55). If `full` does not reproduce, the
   decomposition is not measuring the published lever — stop.
3. **Direction control** (`rand_attn`) and the existing #7 bash-null / cross-task nulls remain the negatives.

## 6. Caveats acknowledged up front
- Single model (Qwen3.6-27B), single position (the `<function` decision token), n=60, MAXLEN 4000 — same
  scope caveats as #7.
- Component re-injection is at the **layer-L output**; it does not re-run L's sublayers on a modified input,
  so it measures the *causal effect of that sublayer's published write*, not a full circuit trace.
- A null (both channels needed, no clean localization) is a publishable result: "the late commit is an
  irreducible residual-sum write, not a single-sublayer feature." We commit to reporting it as such.

Artifacts: `scripts/commit_lever_decomp.py` (run, resumable) · `scripts/commit_lever_decomp_stats.py`
(local CPU analysis) · results → HF `caiovicentino1/swebench-phase6-verdict-circuit:results/commit_lever_decomp.json`.
