#!/usr/bin/env python3
"""Figure: the late brake generalizes across irreversible agent actions (emit + redirect-to-safe)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

R = json.load(open(hf_hub_download("caiovicentino1/swebench-phase6-verdict-circuit",
              "results/multi_action_brake.json", repo_type="dataset", force_download=True)))
D = R["domains"]
order = ["send_transaction", "approve_allowance", "delete_file", "drop_table", "deploy_production", "send_email"]
key = {"send_transaction": "crypto_send", "approve_allowance": "approve", "delete_file": "fs_delete",
       "drop_table": "db_drop", "deploy_production": "deploy", "send_email": "email_send"}
labels, base, brake, rand, redir, blay = [], [], [], [], [], []
for a in order:
    d = D[key[a]]; bl = d["brake_layer"]
    labels.append(f"{a}\n(L{bl})"); base.append(1.0)
    brake.append(d["brake"][f"L{bl}"]["act_emit"]); rand.append(d["ctrl_random"]["act_emit"])
    redir.append(d["redirect_safe_frac"])

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
x = np.arange(len(order)); w = 0.26
ax[0].bar(x - w, base, w, label="baseline (no brake)", color="#d62728")
ax[0].bar(x, brake, w, label="safe-donor BRAKE", color="#1f77b4")
ax[0].bar(x + w, rand, w, label="random ctrl", color="#bbb")
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
ax[0].set_ylabel("irreversible-action emit rate"); ax[0].set_ylim(0, 1.05)
ax[0].set_title("Brake collapses the irreversible commit (6/6)"); ax[0].legend(fontsize=8)
ax[1].bar(x, redir, 0.6, color="#2ca02c")
ax[1].set_xticks(x); ax[1].set_xticklabels([a for a in order], fontsize=7, rotation=20, ha="right")
ax[1].set_ylabel("fraction redirected to a SAFE action"); ax[1].set_ylim(0, 1.05)
ax[1].set_title("...and redirects to a safe action (100%)")
fig.suptitle("The late action-commitment brake generalizes across irreversible agent actions (Qwen3.6-27B, simulated)", fontsize=11)
fig.tight_layout()
out = "paper/circuit_breaker/figures/fig_multi_action_brake.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
