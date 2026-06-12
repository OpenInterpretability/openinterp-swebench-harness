# RESULTS — Source-token knockout: is reading the tool-name tokens causally necessary?

**Run:** 2026-06-12 · Qwen/Qwen3.6-27B · EDIT points (n=60) · Colab G4. **Code:**
`scripts/commit_lever_knockout.py`. **Ledger:** HF `…:results/commit_lever_knockout.json`. Tests the copy
hypothesis from `RESULTS_attn_L59.md` (commit heads attend to the trajectory's tool-call tokens).

## Why a source-content ablation (not a head-edge attention knockout)
The clean test would zero heads 8/6/3's attention edge to the `str_replace_editor` keys and recompute o_h.
That requires reconstructing `o_h = a@V` — which **fails on this model** (gate: `o_h=a@V` vs captured o_proj
input relerr **1.7 / 2.4 / 0.97**, not <1e-2). **Qwen3.6's full-attention is output-GATED** (consistent with
its hybrid linear/full design), so the head output is `gate ⊙ (a@V)`, not `a@V`. We therefore pivot to a
**gate-agnostic source-content ablation**: mean-patch the L59 `k_proj`/`v_proj` outputs at the tool-name key
positions (per-channel mean over positions), removing what the attention can read there — surgical to L59's
attention, no gate/residual surgery. Caveat: this ablates the source for **all** L59 heads, not only 8/6/3 (so
it is not head-specific; head specificity is the separately-established head-decomposition result).

## Result (mean over rows that contain the target tokens)
| condition | P(edit) (Δ vs base) | P(bash) (Δ vs base) | n |
|---|---|---|---|
| baseline (n=60) | 0.474 | 0.398 | 60 |
| **ko_edit** (ablate `str_replace_editor`) | 0.874 (**−0.031**) | 0.119 (**+0.025**) | 30 |
| **ko_bash** (ablate `bash`) | 0.476 (+0.008) | 0.376 (**−0.071**) | 50 |
| **ko_rand** (random, count-matched) | 0.904 (−0.001) | 0.095 (+0.000) | 30 |

## Findings (honest)
1. **Tool choice is causally specific to each tool's name tokens — directionally confirmed.** Ablating the
   `str_replace_editor` source lowers P(edit) and raises P(bash); ablating the `bash` source lowers P(bash);
   ablating random positions does nothing. The model's next-tool decision *reads the prior tool-name tokens* —
   exactly the induction/copy signature the attention analysis suggested.
2. **The edit side is ceiling-confounded.** The rows that *contain* `str_replace_editor` tokens are the rows
   where the agent is **already edit-saturated** (per-row baseline P(edit) ≈ 0.87–0.99). So the edit drop can
   only be small (−0.031); emit on these rows is at ceiling (0.97 with and without ablation) and uninformative.
   This is a genuine limitation, not a clean "removing the tokens kills edit."
3. **The bash side is the clean demonstration.** On non-saturated rows (P(bash)≈0.40), ablating the `bash`
   tokens specifically drops P(bash) by **−0.071** (−18% relative) vs −0.000 for random — a clean, specific,
   non-ceiling causal effect that the agent's tool choice copies the tool-name tokens from its history.

## Verdict
**Partial / directional support for the copy hypothesis, not a decisive knockout.** The agent's tool decision
is causally sensitive, *tool-specifically*, to the presence of that tool's name tokens in the trajectory
(cleanest on bash, −0.071 vs random ~0). The edit side is ceiling-limited, and the ablation is source-content
(all heads) rather than the ideal head-specific attention-edge knockout — which this model's gated attention
blocked. So the induction/copy reading of heads 8/6/3 is **supported but not nailed**.

## What would nail it (not run)
A head-specific edge knockout that accounts for the attention gate (`z_h = gate_h ⊙ a_h@V_h`): capture gate_h,
recompute `z_h'' = gate_h ⊙ (a_h''@V_h)`, inject Δz. Or test on a non-saturated edit-token subset. Single
model, single position, n=60 (n=30 with edit tokens).
