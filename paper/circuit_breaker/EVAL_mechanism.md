# EVAL — mechanistic decomposition section (adversarial pre-submission)

**Date:** 2026-06-12 · **Scope:** `section_mechanism.tex` (the new §"Opening the lever") of the extended
paper. **Method:** `scripts/eval_mechanism_section.py` re-fetches the four raw HF ledgers (the released source
of truth) and recomputes every number in the section, flagging any mismatch. Run on CPU.

## Result: 53/53 claims verified against the released ledgers.
Ledgers checked (HF `caiovicentino1/swebench-phase6-verdict-circuit`): `commit_lever_decomp.json`,
`commit_lever_heads.json`, `commit_lever_attn.json`, `commit_lever_knockout.json`.

- **Gates** (5/5): recon relerr 0.0025/0.0024 (decomp), 0.0015 (head split); `full` reproduces the published
  lever (elicit emit 0.767→reported 0.77; brake 0.033→0.03).
- **Attn-vs-MLP table** (16/16): all four `full` ΔP and all twelve recovery fractions match to ≤0.02.
- **Decomp details** (11/11): attn meanΔP 0.223, attn_rand 0.100, ratio 2.23×, **Wilcoxon attn≫mlp p=1.79e-8**
  (claim 1.8e-8), emit 0.45/0.25/0.23, interaction gaps +0.20/−0.23/−0.26.
- **Heads** (17/17): n=24, head_dim 256; heads 8/6/3 = +0.122/+0.119/+0.083; cumulative top1/top3/all/ctl
  = 0.121/0.262/0.224/0.077; emit top3 0.42 / all 0.47; head-21 geom proj 14.0 with causal ΔP −0.014
  (opponent); brake heads range [−0.005,+0.012].
- **Attention** (1/1): commit-head rec128 0.10/0.12/0.16 (in [0.10,0.16]).
- **Knockout** (5/5): gate recon 1.68 (fails → justifies the source-ablation pivot); ko_bash ΔP(bash) −0.071
  (−17.8% relative), ko_rand ≈ 0 (−0.0005), ko_edit −0.031 (ceiling).

## One correction applied (the eval earned its keep)
The prose paired ``$\sim$89\% of the full attention channel's $0.45$'' with the top-3 emit rate $0.42$, but
$0.42/0.45 = 0.93$; the $89\%$ figure is $0.42/0.47$ (the **all-heads** rate). Fixed to ``$89\%$ of the
all-24-heads rate ($0.47$) and near the attention channel's $0.45$.'' Also softened ``this is an induction/copy
signature'' → ``the attention signature of an induction/copy mechanism'' (the causal test is separately, and
only partially, confirmatory).

## Overclaim audit (prose vs. evidence) — clean
- ``MLP is null'' — supported (r≈0 in all four cells; Wilcoxon attn≫mlp p=1.8e-8). ✓
- ``attention-dominated but not attention-sufficient'' — honest (r_attn=0.43 < the pre-registered 0.5). ✓
- ``three heads reproduce (and overshoot)'' — top3 0.262 ≥ all 0.224, by construction of the opposing set. ✓
- ``geometry misleads'' — head 21 largest writer (proj 14.0) is causal opponent (−0.014). ✓
- ``brake has no head locus'' — all 24 in [−0.005,+0.012]. ✓
- ``copy reading supported, not nailed'' — explicitly hedged; ceiling + non-head-specific caveats stated. ✓
- Citations: Geva (arXiv:2012.14913), Elhage (Anthropic 2021, no arXiv), Olsson (arXiv:2209.11895),
  Wang/IOI (arXiv:2211.00593) — canonical IDs, manually checked.

## Compile
`lever_generalizes_extended.tex` (the extended paper, `\input`ing the section): pdflatex, **10 pages, 0 errors,
0 undefined references/citations**.

**Verdict: PASS — no fabrications, no surviving overclaims, all numbers reproduce from released data. Ready to
submit.**
