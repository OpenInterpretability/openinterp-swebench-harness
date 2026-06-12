#!/usr/bin/env python3
"""Figure: attn-vs-MLP decomposition of the late commitment-lever. Recovery fraction per component."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

p = hf_hub_download("caiovicentino1/swebench-phase6-verdict-circuit",
                    "results/commit_lever_decomp.json", repo_type="dataset", force_download=True)
D = json.load(open(p)); dp = D["dp"]
comps = ["inp", "attn", "mlp", "attnmlp"]
labels = {"inp": "upstream\n(≤L−1)", "attn": "attn@L", "mlp": "MLP@L", "attnmlp": "attn+MLP@L"}
colors = {"inp": "#888", "attn": "#1f77b4", "mlp": "#d62728", "attnmlp": "#9467bd"}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, direction, title in zip(axes, ("elicit", "brake"),
                                ("Elicit (bash→edit donor)", "Brake (edit→bash donor)")):
    x = np.arange(2); w = 0.2
    for ci, comp in enumerate(comps):
        rs = []
        for L in (55, 59):
            full = dp[f"dp_{direction}_L{L}_full"]["mean"]
            v = dp[f"dp_{direction}_L{L}_{comp}"]["mean"]
            rs.append(v / full)
        ax.bar(x + (ci - 1.5) * w, rs, w, label=labels[comp], color=colors[comp])
    # direction control marker (attn_rand recovery)
    for xi, L in zip(x, (55, 59)):
        full = dp[f"dp_{direction}_L{L}_full"]["mean"]
        rr = dp[f"dp_{direction}_L{L}_attn_rand"]["mean"] / full
        ax.plot(xi - 0.5 * w, rr, "kx", ms=8, mew=2)
    ax.axhline(0, color="k", lw=0.6); ax.axhline(1, color="k", lw=0.5, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(["L55", "L59"]); ax.set_title(title, fontsize=11)
    ax.set_ylim(-0.3, 1.15)
axes[0].set_ylabel("recovery fraction  r = ΔP_comp / ΔP_full")
axes[0].legend(fontsize=8, loc="upper left", ncol=2)
axes[1].plot([], [], "kx", ms=8, mew=2, label="attn (random-donor ctl)")
axes[1].legend(fontsize=8, loc="upper left")
fig.suptitle("Decomposing the late action-commitment lever: where is it written?", fontsize=12)
fig.tight_layout()
out = "paper/circuit_breaker/figures/fig_decomp_attn_mlp.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
