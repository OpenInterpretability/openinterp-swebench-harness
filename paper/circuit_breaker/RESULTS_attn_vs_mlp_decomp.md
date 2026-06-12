# RESULTS — Attn-vs-MLP decomposition of the late commitment-lever

**Run:** 2026-06-11/12 · Qwen/Qwen3.6-27B · n=60/condition · Colab G4 (RTX PRO 6000 Blackwell).
**Pre-reg:** `PREREG_attn_vs_mlp_decomp.md`. **Code:** `scripts/commit_lever_decomp.py` + `_stats.py` + `make_fig_decomp.py`.
**Ledger:** HF `caiovicentino1/swebench-phase6-verdict-circuit:results/commit_lever_decomp.json`.

## Gates (both passed)
1. **Residual reconstruction** ‖(x+a+m)−y‖/‖y‖ = **0.0025 @L55, 0.0024 @L59** (≪1e-2) → the additive split
   `y_L = x_L + attn_out_L + mlp_out_L` holds on Qwen3.6-27B; the decomposition is exact up to bf16 rounding.
2. **Reproduces paper #7**: `full` re-injection recovers the published lever — elicit emit 0.23→**0.77@L59**,
   brake emit 0.48→**0.03@L55** (published 0.77 / 0.02). The decomposition measures the real lever.

## Core table — recovery fraction r = ΔP_comp / ΔP_full  (per-point, n=60)
| direction | layer | ΔP_full | inp (≤L−1) | **attn@L** | **mlp@L** | attn+mlp | attn_rand (ctl) | interaction gap |
|---|---|---|---|---|---|---|---|---|
| elicit | L55 | +0.319 | +0.52 | +0.05 | −0.09 | −0.07 | +0.23 | +0.166 |
| **elicit** | **L59** | **+0.518** | +0.22 | **+0.43** | −0.04 | +0.46 | +0.20 | +0.203 |
| brake | L55 | −0.404 | +0.40 | +0.02 | +0.03 | +0.07 | −0.04 | −0.227 |
| brake | L59 | −0.382 | +0.35 | +0.00 | −0.02 | −0.03 | −0.18 | −0.255 |

(attn_rand column = recovery of a cross-repo random-donor attn swap, the direction control.)

## Behavioral confirmation — emit-rate under generation
| | baseline | full | attn-only | mlp-only |
|---|---|---|---|---|
| **elicit @L59** | 0.23 | **0.77** | **0.45** | 0.25 |
| **brake @L55** | 0.48 | **0.03** | **0.48** | 0.48* |

(*brake@L55 mlp-only: confirmed ≈ baseline.)

## Findings (honest)
1. **Elicit and brake are mechanistically ASYMMETRIC.** The bidirectional lever of paper #7 is not one
   mechanism run forwards and backwards — the two directions decompose differently.
2. **Elicit is ATTENTION-dominated and late-specific.** At L59 the attention sublayer is by far the largest
   single writer of the elicit (r_attn = +0.43, 95% CI [0.33,0.53]) while the **MLP is null** (r_mlp = −0.04;
   Wilcoxon attn≫mlp p=1.8e-8), and it is **direction-specific** (attn ΔP +0.223 vs random-donor +0.100, 2.2×).
   `attn+mlp` ≈ `attn` (MLP adds nothing). Behaviorally, attn-only roughly doubles emission (0.23→0.45) while
   mlp-only is inert (0.25). At L55 neither sublayer writes it (r_attn 0.05, r_mlp −0.09) — the signal is still
   *arriving via the residual* from below (inp r=0.52); the attention write fires specifically at **L59**.
   → This **rules out the Geva-style MLP key-value "emit-action feature"** for the commit. The late action
   decision is **moved/promoted by attention**, consistent with the logit-lens "explosion" L51→L63.
3. **Brake localizes to NEITHER sublayer.** At both L55 and L59, attn and mlp recover ≈0 (all CIs near zero);
   suppression is carried ~35–40% by the upstream residual (inp) and the rest by a large **negative nonlinear
   interaction** (gap −0.23/−0.26). attn-only fails to brake behaviorally (emit stays 0.48). → Killing the
   action is **not a single-sublayer write**; it needs the whole late residual state (a distributed,
   super-additive effect).
4. **The lever is substantially non-additive.** Across all four cells the interaction gap is large and
   consistent (elicit +0.17/+0.20, brake −0.23/−0.26): re-injecting the additive components recovers only part
   of the full-swap effect. The lever is partly a distributed residual phenomenon, not a clean sum of sublayer
   writes — which honestly tempers any "it's just attention" headline.

## Verdict vs the pre-registered rule
The strict rule (attention-localized iff r_attn≥0.50 ∧ r_mlp<0.25) returns **SPLIT/additive** for every cell,
because elicit-L59 attn r=0.43 sits just under the 0.50 bar. The honest statement is therefore:
**"Elicit is attention-DOMINATED but not attention-SUFFICIENT"** (attn is the single largest, direction-specific,
MLP-null channel, but upstream + nonlinear interaction carry the remainder), and **"the brake is an irreducible
distributed residual effect with no single-sublayer locus."** The MLP is null in all four cells — the cleanest
single takeaway: **the late commit is an attention/residual phenomenon, never a feed-forward write.**

## Caveats
Single model (Qwen3.6-27B), single decision position (`<function`), n=60, MAXLEN 4000 — same scope as #7.
Component re-injection is at the layer-L output (causal effect of the published sublayer write), not a full
circuit re-run. Attention is not decomposed into heads (next step). Figure: `figures/fig_decomp_attn_mlp.png`.
