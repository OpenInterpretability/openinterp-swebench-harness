"""Regenerate beat-11 figures from the PUBLIC HF ledger + vectors + residuals. CPU-only.
Outputs to paper/circuit_breaker/figures/figJ{1,2,3}_*.pdf
"""
import os, json
os.environ.setdefault("HF_TOKEN", open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
                      if os.path.exists(os.path.expanduser("~/.cache/huggingface/token")) else "")
from huggingface_hub import hf_hub_download
import torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "caiovicentino1/swebench-phase6-verdict-circuit"; TOK = os.environ.get("HF_TOKEN") or None
def dl(f): return hf_hub_download(REPO, f, repo_type="dataset", token=TOK, force_download=True)
A = json.load(open(dl("results/jspace_action.json")))
OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "circuit_breaker", "figures")
os.makedirs(OUT, exist_ok=True)
NL = 64

# ---- Fig J1: H1 AUROC by depth, with regime shading ----
L = sorted(A["h1"], key=int); xs = [int(k) for k in L]
depth = [A["h1"][k]["depth_pct"] for k in L]; au = [A["h1"][k]["auroc"] for k in L]
fig, ax = plt.subplots(figsize=(5.6, 3.1))
ax.axvspan(0, 38, color="#dfeaf6", alpha=.6); ax.axvspan(38, 92, color="#e7f5e5", alpha=.6)
ax.axvspan(92, 100, color="#f7e2e2", alpha=.6)
ax.text(19, 0.79, "sensory", ha="center", fontsize=8, color="#33506e")
ax.text(65, 0.79, "workspace", ha="center", fontsize=8, color="#2f6b2a")
ax.text(96, 0.79, "motor", ha="center", fontsize=8, color="#8a3030", rotation=90, va="top")
ax.axhline(0.5, color="gray", lw=.8, ls="--")
ax.plot(depth, au, "-o", color="#1b3a6b", ms=4, lw=1.6)
pk = np.argmax(au); ax.annotate(f"L47\n{au[pk]:.2f}", (depth[pk], au[pk]), textcoords="offset points",
    xytext=(6, -22), fontsize=8, arrowprops=dict(arrowstyle="->", lw=.7))
ax.set_xlabel("depth (%)"); ax.set_ylabel("AUROC  (g$_{edit}$-g$_{bash}$)$\\cdot$h")
ax.set_title("H1: the action becomes readable only in the workspace tail", fontsize=9.5)
ax.set_ylim(0.25, 0.82); ax.set_xlim(0, 100)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figJ1_readout_depth.pdf")); plt.close(fig)

# ---- Fig J2: the dissociation — answer vs action flip rate per band (+ random) ----
bands = ["midW", "tail", "motor"]; blab = ["mid-workspace\nL27-43", "tail\nL47-55", "motor\nL59-63"]
ans = [A["answer_bands"][b]["contrast_to_alt"]/A["answer_bands"][b]["n"] for b in bands]
ans_r = [A["answer_bands"][b]["random_to_alt"]/A["answer_bands"][b]["n"] for b in bands]
act = [A[f"h2steer_{b}"]["edit_pts"]["flip_to_other"]/A[f"h2steer_{b}"]["edit_pts"]["n"] for b in bands]
act_r = [A[f"h2steer_{b}"]["edit_pts"]["random_flip_to_other"]/A[f"h2steer_{b}"]["edit_pts"]["n"] for b in bands]
x = np.arange(3); w = 0.36
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.bar(x-w/2, ans, w, color="#2f6b2a", label="answer (steer→alt)")
ax.bar(x+w/2, act, w, color="#8a3030", label="action (steer→other tool)")
ax.plot(x-w/2, ans_r, "_", color="k", ms=14, mew=1.6, label="random-direction control")
ax.plot(x+w/2, act_r, "_", color="k", ms=14, mew=1.6)
ax.set_xticks(x); ax.set_xticklabels(blab, fontsize=8.5)
ax.set_ylabel("specific flip rate"); ax.set_ylim(0, 1.05)
ax.set_title("H2: answers reroute in mid-workspace; the action does not", fontsize=9.5)
ax.annotate("answer flips,\naction doesn't", (0, 0.55), fontsize=8, color="#333",
            ha="center", xytext=(0, 0.78), textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=.7))
ax.legend(fontsize=7.5, loc="upper left", framealpha=.9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figJ2_dissociation.pdf")); plt.close(fig)

# ---- Fig J3: H3 ablation leaves commitment intact ----
h3 = A["h3"]
base = [h3["P_edit_base"]["edit_pts"], h3["P_edit_base"]["bash_pts"]]
abl = [h3["P_edit_ablated"]["edit_pts"], h3["P_edit_ablated"]["bash_pts"]]
fig, ax = plt.subplots(figsize=(3.7, 3.0))
x = np.arange(2); w = 0.36
ax.bar(x-w/2, base, w, color="#4a6fa5", label="baseline")
ax.bar(x+w/2, abl, w, color="#a5764a", label="top-10 workspace ablated")
ax.set_xticks(x); ax.set_xticklabels(["edit\npoints", "bash\npoints"], fontsize=8.5)
ax.set_ylabel("P(edit)"); ax.set_ylim(0, 0.6)
ax.set_title("H3: commitment survives\nworkspace-subspace ablation", fontsize=9)
ax.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figJ3_ablation.pdf")); plt.close(fig)
# ---- Fig J4: the depth lag replicates across two architectures (reframe headline) ----
X = json.load(open(dl("results/jspace_xmodel_gpt-oss-20b.json")))
# dense bands (of 64 layers), MoE bands (of 24) -> depth% midpoints
dense_mid = {"midW": 100*(27+43)/2/64, "tail": 100*(47+55)/2/64, "motor": 100*(59+63)/2/64}
moe_mid   = {"midW": 100*(11+16)/2/24, "tail": 100*(18+20)/2/24, "motor": 100*(22+23)/2/24}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
for ax, (name, mid, ans_src, act_src, an, actn) in zip(axes, [
    ("Qwen3.6-27B (dense)", dense_mid,
     {b: A["answer_bands"][b]["contrast_to_alt"]/A["answer_bands"][b]["n"] for b in bands},
     {b: A[f"h2steer_{b}"]["edit_pts"]["flip_to_other"]/A[f"h2steer_{b}"]["edit_pts"]["n"] for b in bands}, 20, 60),
    ("gpt-oss-20b (MoE)", moe_mid,
     {b: X["dissoc"]["answer"][b]["contrast_to_alt"]/X["dissoc"]["answer"][b]["n"] for b in bands},
     {b: X["dissoc"]["action"][b]["edit_to_bash"]/X["dissoc"]["action"][b]["n"] for b in bands}, 20, 40)]):
    dx = [mid[b] for b in bands]
    ax.axvspan(38, 92, color="#e7f5e5", alpha=.5); ax.axvspan(92, 100, color="#f7e2e2", alpha=.5)
    ax.plot(dx, [ans_src[b] for b in bands], "-o", color="#2f6b2a", ms=6, lw=1.8, label="answer")
    ax.plot(dx, [act_src[b] for b in bands], "-s", color="#8a3030", ms=6, lw=1.8, label="action")
    # annotate the lag: first band where answer is specific (>=0.4) vs where action is
    a_on = next((mid[b] for b in bands if ans_src[b] >= 0.4), None)
    c_on = next((mid[b] for b in bands if act_src[b] >= 0.4), None)
    if a_on and c_on and c_on > a_on:
        ax.annotate("", xy=(c_on, 0.5), xytext=(a_on, 0.5),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.4))
        ax.text((a_on+c_on)/2, 0.56, "lag", ha="center", fontsize=8, color="#555")
    ax.set_title(name, fontsize=9.5); ax.set_xlabel("depth (%)"); ax.set_xlim(30, 100); ax.set_ylim(0, 1.05)
axes[0].set_ylabel("specific flip rate"); axes[0].legend(fontsize=8, loc="upper left", framealpha=.9)
fig.suptitle("J4: the action lags the answer in depth, in both architectures", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figJ4_depthlag_xmodel.pdf")); plt.close(fig)
print("wrote figJ1_readout_depth.pdf, figJ2_dissociation.pdf, figJ3_ablation.pdf, figJ4_depthlag_xmodel.pdf to", os.path.abspath(OUT))
