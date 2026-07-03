"""Regenerate the 3 figures of hybridization_audit.tex from the PUBLIC HF ledger. CPU-only.
Usage: python3 make_figs_hydra.py   (writes paper/circuit_breaker/figures/fig{1,2,3}_*.pdf)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

REPO = "caiovicentino1/swebench-phase6-verdict-circuit"
FIG = os.path.join(os.path.dirname(__file__), "..", "paper", "circuit_breaker", "figures")
os.makedirs(FIG, exist_ok=True)
A = json.load(open(hf_hub_download(REPO, "results/hydra_audit_ie.json", repo_type="dataset")))

# FIG 1 — anti-alignment at L59
res = A["ie_L59"]; ies = [float(np.mean([r["ie"] for r in res[str(h)]])) for h in range(24)]
writers, opposers = {8, 6, 3}, {16, 19, 11, 4, 21, 23}
colors = ["#c0392b" if h in writers else ("#2980b9" if h in opposers else "#bdc3c7") for h in range(24)]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar(range(24), ies, color=colors)
ax.axhline(0.01, ls="--", c="k", lw=0.8); ax.text(23.4, 0.011, "criticality threshold", ha="right", fontsize=8)
for h in (8, 6, 3, 21, 19):
    ax.annotate(f"h{h}\n({'writer' if h in writers else 'opposer'})", (h, ies[h]),
                textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
ax.set_xlabel("L59 attention head"); ax.set_ylabel("retrieval IE (mean)")
ax.set_title("Anti-alignment at L59: commit writers (red) carry no retrieval criticality;\n"
             "the layer's top retrieval heads are the circuit's opposers (blue)", fontsize=10)
ax.set_xticks(range(0, 24, 2)); fig.tight_layout(); fig.savefig(f"{FIG}/fig1_antialign_L59.pdf")

# FIG 2 — two-sided audit
conds = [("base", "base"), ("drop_nonret_L59", "drop non-ret.\nL59 (20h)"), ("drop_writers", "drop writers\n(3h)"),
         ("drop_nonret_late", "drop non-ret.\nband (74h)"), ("drop_rand_late", "random band\n(74h, s42)"),
         ("keep_writers_late", "selection\n+{h8,h6} (72h)")]
edit = [A[f"h3_{c}_mean"]["edit_emit"] for c, _ in conds]
bash = [A[f"h3_{c}_mean"]["bash_emit"] for c, _ in conds]
niah = [float(np.mean([b["m"] for b in A[f"h3_{c}_mean"]["niah"]])) for c, _ in conds]
x = np.arange(len(conds)); w = 0.28
fig, ax1 = plt.subplots(figsize=(8, 3.4))
ax1.bar(x - w / 2, edit, w, label="commit emit @ edit points", color="#c0392b")
ax1.bar(x + w / 2, bash, w, label="commit emit @ bash points", color="#e8a598")
ax1.set_ylabel("emission rate"); ax1.set_ylim(0, 0.6); ax1.legend(loc="upper left", fontsize=8)
ax2 = ax1.twinx(); ax2.plot(x, niah, "o--", c="#27ae60", label="NIAH margin (teacher-forced)")
ax2.set_ylabel("NIAH margin", color="#27ae60"); ax2.set_ylim(0, 12); ax2.legend(loc="upper right", fontsize=8)
ax1.set_xticks(x); ax1.set_xticklabels([n for _, n in conds], fontsize=8)
ax1.set_title("The two-sided audit: commitment collapses under the criterion's band ablation;\n"
              "the capability probe never registers anything (and rises)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_twosided.pdf")

# FIG 3 — rescue specificity
groups = [("collapse\n(selection, 74h)", [A["h3_drop_nonret_late_mean"]["edit_emit"]], "#c0392b"),
          ("random draws\n(74h, 5 seeds)", [A[f"h3_{c}_mean"]["edit_emit"] for c in
           ("drop_rand_late", "drop_rand_late_s2", "drop_rand_late_s3", "drop_rand_late_s4", "drop_rand_late_s5")], "#7f8c8d"),
          ("keep 2 random\n(72h, 3 seeds)", [A[f"h3_keep2rand_late_s{i}_mean"]["edit_emit"] for i in (1, 2, 3)], "#95a5a6"),
          ("keep {h8,h6}\n(72h)", [A["h3_keep_writers_late_mean"]["edit_emit"]], "#27ae60")]
fig, ax = plt.subplots(figsize=(7, 3.2))
for i, (name, vals, c) in enumerate(groups):
    ax.scatter([i] * len(vals), vals, s=60, c=c, zorder=3)
    ax.hlines(np.mean(vals), i - 0.18, i + 0.18, colors=c, lw=2)
ax.axhline(0.483, ls=":", c="k", lw=1); ax.text(3.35, 0.49, "baseline 0.483", fontsize=8, ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels([g[0] for g in groups], fontsize=9)
ax.set_ylabel("commit emit @ edit points"); ax.set_ylim(0, 0.55)
ax.set_title("The rescue is head-specific: two named writers restore baseline exactly;\n"
             "two random heads at identical severity do not", fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_rescue.pdf")
print("3 figures written to", os.path.abspath(FIG))
