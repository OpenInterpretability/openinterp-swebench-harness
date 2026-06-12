#!/usr/bin/env python3
"""Figure + stats: per-head causal ΔP of the L59 attention write, and cumulative top-k recovery."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

p = hf_hub_download("caiovicentino1/swebench-phase6-verdict-circuit",
                    "results/commit_lever_heads.json", repo_type="dataset", force_download=True)
H = json.load(open(p))
nh = H["n_heads"]
dph = H["dp_head"]; cum = H.get("dp_cum", {}); emit = H.get("emit", {})
means = np.array([dph[str(h)]["mean"] for h in range(nh)])
order = np.argsort(-means)

print("=== GATE head-split relerr:", H.get("headsplit_relerr"), "| nh", nh, "hd", H.get("head_dim"))
full = cum.get("all", {}).get("mean", float("nan"))
print(f"=== full attn (all {nh} heads) ΔP = {full:+.3f}  (should ~ +0.223 from the sublayer decomp)")
print("=== top heads by causal ΔP:")
for h in order[:8]:
    print(f"  head {h:2d}: ΔP {means[h]:+.3f}  ({means[h]/full*100:.0f}% of full)" if full else f"  head {h}: {means[h]:+.3f}")
print("=== cumulative:")
for k in ("top1", "top2", "top3", "top5", "top8", "all"):
    if k in cum: print(f"  {k:5s} ({len(cum[k]['heads'])}h): ΔP {cum[k]['mean']:+.3f}  ({cum[k]['mean']/full*100:.0f}%)")
if "topctl" in cum: print(f"  topctl (head {cum['topctl']['head']}, cross-repo donor): ΔP {cum['topctl']['mean']:+.3f}")
if emit:
    print("=== emit-rate (cumulative head sets):")
    for k in sorted(emit): print(f"  {k:5s}: {emit[k]['rate']:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
# panel A: per-head ΔP sorted
ax[0].bar(range(nh), means[order], color=["#1f77b4" if means[order][i] > 0 else "#aaa" for i in range(nh)])
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_xticks(range(nh)); ax[0].set_xticklabels([str(h) for h in order], fontsize=6, rotation=90)
ax[0].set_xlabel("attention head (L59), sorted by ΔP"); ax[0].set_ylabel("causal ΔP(edit)")
ax[0].set_title("Per-head write of the elicit lever (L59)")
# panel B: cumulative
ks = [k for k in ("top1", "top2", "top3", "top5", "top8", "all") if k in cum]
xs = [len(cum[k]["heads"]) for k in ks]; ys = [cum[k]["mean"] for k in ks]
ax[1].plot(xs, ys, "o-", color="#1f77b4")
if not np.isnan(full): ax[1].axhline(full, color="k", ls=":", lw=0.8, label=f"all {nh} heads ({full:+.2f})")
if "topctl" in cum: ax[1].axhline(cum["topctl"]["mean"], color="#d62728", ls="--", lw=0.8, label="cross-repo ctl")
ax[1].set_xlabel("# heads (cumulative, by ΔP rank)"); ax[1].set_ylabel("ΔP(edit)")
ax[1].set_title("How many heads carry the lever?"); ax[1].legend(fontsize=8)
fig.suptitle("Head-level decomposition of the L59 attention commitment-write", fontsize=12)
fig.tight_layout()
out = "paper/circuit_breaker/figures/fig_heads_L59.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
