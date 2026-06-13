#!/usr/bin/env python3
"""Figure: the irreversible-action brake LAW across 3 architectures x 6 actions."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
REPO = "caiovicentino1/swebench-phase6-verdict-circuit"
def L(f): return json.load(open(hf_hub_download(REPO, f"results/{f}", repo_type="dataset", force_download=True)))
runs = {"Qwen3.6-27B": L("multi_action_brake.json"),
        "Llama-3.1-8B": L("multi_action_brake_Meta-Llama-3.1-8B-Instruct.json"),
        "Mistral-24B": L("multi_action_brake_Mistral-Small-24B-Instruct-2501.json")}
acts = ["crypto_send", "approve", "fs_delete", "db_drop", "deploy", "email_send"]
alab = ["send_tx", "approve", "delete", "drop", "deploy", "email"]
models = list(runs)
grid = np.full((len(models), len(acts)), np.nan); txt = [["" for _ in acts] for _ in models]
for mi, m in enumerate(models):
    D = runs[m]["domains"]
    for ai, a in enumerate(acts):
        d = D.get(a)
        if not d: continue
        bl = d["brake_layer"]; em = d["brake"][f"L{bl}"]["act_emit"]; fid = d["fidelity"]["commit"]; rs = d["redirect_safe_frac"]
        if fid < 0.3:
            grid[mi, ai] = 0.5; txt[mi][ai] = "fid✗"  # no commit state
        else:
            grid[mi, ai] = rs; txt[mi][ai] = f"L{bl}\n{rs:.0%}"
fig, ax = plt.subplots(figsize=(9, 3.4))
cmap = plt.cm.RdYlGn
im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(acts))); ax.set_xticklabels(alab, fontsize=9)
ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=9)
for mi in range(len(models)):
    for ai in range(len(acts)):
        ax.text(ai, mi, txt[mi][ai], ha="center", va="center", fontsize=7.5,
                color="black" if (grid[mi, ai] > 0.4) else "white")
ax.set_title("Brake redirects an irreversible commit to a SAFE action — across 3 architectures × 6 actions\n"
             "(green = brake works & 100% safe-redirect; cell = brake layer; fid✗ = model won't commit the action)", fontsize=9)
fig.tight_layout()
out = "paper/circuit_breaker/figures/fig_xmodel_brake.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
