# RESULTS — Head-level decomposition of the L59 attention commitment-write

**Run:** 2026-06-12 · Qwen/Qwen3.6-27B · n=60/condition · Colab G4. **Code:** `scripts/commit_lever_heads.py`
+ `make_fig_heads.py`. **Ledger:** HF `…:results/commit_lever_heads.json`. Builds directly on
`RESULTS_attn_vs_mlp_decomp.md` (which localized the elicit lever to the **L59 attention** sublayer).

## Architecture note
Qwen3.6-27B is a **hybrid linear/full-attention** stack (`full_attention_interval=4`): only layers
3,7,…,55,**59**,63 are full softmax-attention. The lever lives in a full-attention layer (L59), which has
**24 query heads, head_dim 256, o_proj with no bias** (GQA 24:4). The attention output is an exact sum over
heads: `attn_out = Σ_h c_h`, `c_h = z[h_slice] @ W_O[:, h_slice]^T` (z = o_proj input). **GATE:** the head
split reconstructs the captured attention output to relerr **0.0015** (≪1e-2).

## Per-head causal ΔP(edit), elicit, n=60 (re-inject one head's donor→point Δc_h at L59)
| head | 8 | 6 | 3 | 1 | 5 | … | 4 | 16 | 19 | 11 | 21 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ΔP | **+0.122** | **+0.119** | **+0.083** | +0.020 | +0.010 | ~0 | −0.024 | −0.024 | −0.027 | −0.028 | −0.014 | −0.013 |

## Cumulative & behavioral
| set | heads | ΔP | % of full | emit-rate |
|---|---|---|---|---|
| top1 | {8} | +0.121 | 54% | — |
| top3 | {8,6,3} | **+0.262** | **117%** | **0.42** |
| top5 | {8,6,3,1,5} | +0.271 | 121% | 0.40 |
| top8 | +5 more | +0.274 | 122% | — |
| **all 24** | — | **+0.224** | 100% | 0.47 |
| ctl (head 8, cross-repo donor) | {8} | +0.077 | — | — |
| baseline (no patch) | — | — | — | 0.23 |

## Findings
1. **The commitment-write is SPARSE: ~3 attention heads carry it.** Heads **8, 6, 3** alone reproduce the
   entire L59 attention effect — top3 ΔP +0.262 ≥ all-24 +0.224 — and behaviorally raise emission from the
   0.23 baseline to **0.42** (vs 0.45 for the full attention channel; ~89% of the effect from 3 of 24 heads).
2. **Opposing heads net the effect down.** A cluster of heads (16, 19, 11, 4, 21, 23) push P(edit) *down*
   (ΔP −0.01…−0.03 each). That is why top3 (+0.262) **overshoots** the all-heads net (+0.224): the 3 commit
   heads are partially cancelled by an opposing set. The lever is a **small push–pull head circuit**, not a
   diffuse sum.
3. **Geometric write-magnitude is MISLEADING.** The largest writer along the attention direction is head 21
   (projection 14.0, ‖Δc‖ 26.9) — but its causal ΔP is **−0.014** (an *opponent*). The causal commit heads
   (8, 6, 3) are only mid-sized geometrically (proj 8.2 / 4.9 / 8.6). Magnitude of residual write ≠ causal
   effect on the decision; only the patch tells the truth.
4. **Direction-specific.** Head 8 with a task-matched donor gives +0.121; with a cross-repo donor only +0.077.
5. **Sanity:** all-24 ΔP +0.224 == the attn-only ΔP +0.223 from the sublayer decomposition → the head split
   is exact and consistent with the level above.

## Verdict
The late agentic commitment lever — which the prior level localized to L59 attention and away from the MLP —
is, one level deeper, written by a **sparse ~3-head push circuit (heads 8/6/3)** partially opposed by a
counter-set, in a single full-attention layer of a hybrid linear/full stack. This is a concrete, sparse,
causally-verified circuit for an agent's decision to commit an action.

## Caveats
Single model, single decision position, n=60. Heads ranked by mean ΔP (per-row variance not modeled into the
ranking). We did not decompose Q/K/V or trace what heads 8/6/3 attend to (the natural next step: do they read
the consolidated verdict tokens?). The brake direction has no attention locus (prior level), so no head circuit
there. Figure: `figures/fig_heads_L59.png`.
