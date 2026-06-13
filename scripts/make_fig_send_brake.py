#!/usr/bin/env python3
"""Figure: send_transaction brake — the lever is late (locate) and the brake is at the final layer."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
R = json.load(open(hf_hub_download("caiovicentino1/swebench-phase6-verdict-circuit",
              "results/send_brake.json", repo_type="dataset", force_download=True)))
loc = {int(k): v for k, v in R["locate_gap"].items()}
Ls = sorted(loc); gap = [loc[L] for L in Ls]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(Ls, gap, "o-", color="#1f77b4")
ax[0].axhline(0, color="k", lw=0.6); ax[0].axvspan(54, 64, color="orange", alpha=0.12)
ax[0].set_xlabel("layer"); ax[0].set_ylabel("logit-lens send-vs-safe gap")
ax[0].set_title("The send-commit decision is computed LATE")
# brake by layer
lays = [51, 55, 59, 63]; emit = [R["blocks"][f"brake_safedonor_L{L}"]["send_emit"] for L in lays]
x = np.arange(len(lays))
ax[1].bar(x, emit, 0.5, color=["#d62728" if e > 0.5 else "#1f77b4" for e in emit])
ax[1].axhline(1.0, color="#888", ls=":", lw=0.8)
for xi, e in zip(x, emit): ax[1].text(xi, e + 0.03, f"{e:.2f}", ha="center", fontsize=9)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"L{L}" for L in lays])
ax[1].set_ylabel("send_transaction emit rate"); ax[1].set_ylim(0, 1.12)
ax[1].set_title("Safe-donor brake: works only at the final layer L63\n(0.998 fidelity; ctrl send-donor@L63=1.00, random@L63=garbage; 24/24 -> get_balance)", fontsize=8.5)
fig.suptitle("Braking an irreversible send_transaction (Qwen3.6-27B, simulated wallet)", fontsize=11)
fig.tight_layout()
out = "paper/circuit_breaker/figures/fig_send_brake.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
