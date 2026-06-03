#!/usr/bin/env python3
"""Figures for Paper #6 — "The Lever Is Late". All numbers are the verified run outputs from
paper/breakthrough/RESULTS_action_channel.md (logit-lens + H1 patching, A100 n=12/class) and
RESULTS_confirm.md (generation confirmation, A100 n=12). No fabricated values; early flat-region
layers (where the run reported 'flat ~0' without per-layer logit-lens numbers) are drawn at 0.0.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({"font.family":"serif","font.size":10,"axes.titlesize":10,
                     "axes.labelsize":10,"legend.fontsize":8.5,"figure.dpi":150})
OUT = Path(__file__).resolve().parent
C_SUCC, C_LOCK, C_PAIR, C_MEAN, C_BASE = "#1b7837","#b2182b","#2166ac","#92c5de","#999999"

# ---------------------------------------------------------------- Fig 1: layer sweep
# logit-lens SUCCESS-WANDERING gap in the finish-direction (reported anchors; early layers flat ~0)
LL_X = [3,7,11,15,19,23,27,31,35,43,47,51,55,59,63]
LL_Y = [0,0,0,0,0,-0.02,0,0,0.38,0.20,0.36,1.01,2.10,2.73,3.57]
# H1 activation patching: ΔP(finish) injecting class-donor late-block state into WANDERING
H1_X = [3,7,11,15,19,23,27,31,35,43,47,51,55,59,63]
H1_S = [0,0,0,0,0,0,0,0,0,0,0,0.059,0.134,0.153,0.747]   # SUCCESS-donor
H1_L = [0,0,0,0,0,0,0,0,0,0,0,-0.039,-0.063,-0.067,-0.067] # LOCKED-donor (null)

fig, (a1,a2) = plt.subplots(1,2, figsize=(8.4,3.2))
a1.axhline(0, color="#cccccc", lw=0.8, zorder=0)
a1.plot(LL_X, LL_Y, "-o", color="#404040", ms=4, lw=1.4)
a1.axvline(23, color=C_LOCK, ls=":", lw=1.2)
a1.annotate("verdict layer\nL23  (gap $\\approx$ 0)", xy=(23,-0.02), xytext=(8,1.6),
            fontsize=8, color=C_LOCK, arrowprops=dict(arrowstyle="->",color=C_LOCK,lw=0.9))
a1.axvspan(51,63, color="#fde0c8", alpha=0.5, zorder=0)
a1.annotate("decision emerges\nL51–L63", xy=(57,2.4), fontsize=8, color="#a0530a", ha="center")
a1.set_xlabel("layer"); a1.set_ylabel("logit-lens gap S$-$W\n(finish-direction)")
a1.set_title("(a) The finish decision is computed late")

a2.axhline(0, color="#cccccc", lw=0.8, zorder=0)
a2.plot(H1_X, H1_S, "-o", color=C_SUCC, ms=4, lw=1.4, label="SUCCESS-donor")
a2.plot(H1_X, H1_L, "-s", color=C_LOCK, ms=4, lw=1.4, label="LOCKED-donor (null)")
a2.axvspan(51,63, color="#fde0c8", alpha=0.5, zorder=0)
a2.set_xlabel("layer of injection"); a2.set_ylabel("$\\Delta P$(finish)")
a2.set_title("(b) The lever is late and donor-specific")
a2.legend(loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(OUT/"fig1_layer_sweep.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: generation confirmation
fig, (b1,b2) = plt.subplots(1,2, figsize=(8.4,3.2), gridspec_kw={"width_ratios":[1.4,1]})
layers = ["L55","L59","L63"]; x = np.arange(3); w=0.26
mean_r = [0.08,0.08,0.17]; lock_r=[0.00,0.00,0.00]; pair_r=[0.25,0.33,0.17]
b1.bar(x-w, mean_r, w, color=C_MEAN, label="SUCCESS-mean donor")
b1.bar(x,   lock_r, w, color=C_LOCK,  label="LOCKED donor (null)")
b1.bar(x+w, pair_r, w, color=C_PAIR,  label="per-pair (task-matched)")
b1.axhline(0.0, color=C_BASE, lw=1.2, ls="--")
b1.text(2.35, 0.012, "baseline 0.00", color="#666", fontsize=7.5, ha="right")
b1.set_xticks(x); b1.set_xticklabels(layers)
b1.set_ylabel("finish emission rate"); b1.set_ylim(0,0.5)
b1.set_title("(a) Per-layer generation finish-rate"); b1.legend(loc="upper left", frameon=False)

donors = ["baseline","LOCKED\nnull","SUCCESS\n-mean","per-pair\n(matched)"]
rates  = [0.00, 0.00, 0.25, 0.42]
cols   = [C_BASE, C_LOCK, C_MEAN, C_PAIR]
bars = b2.bar(donors, rates, color=cols)
b2.set_ylabel("any-late-layer finish rate  (n=12)"); b2.set_ylim(0,0.55)
b2.set_title("(b) Headline: union over L55/59/63")
for bar,r,lab in zip(bars, rates, ["","p=1.0","p=0.125","p=0.031 *"]):
    if lab: b2.text(bar.get_x()+bar.get_width()/2, r+0.015, lab, ha="center", fontsize=7.8,
                    color=("#a00000" if "0.031" in lab else "#555"),
                    fontweight=("bold" if "0.031" in lab else "normal"))
fig.tight_layout(); fig.savefig(OUT/"fig2_generation.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: where control lives
# Honest framing: the CONTROLLED claim is verdict-null -> late-lever (same experiment, solid bars).
# The behavioral bar is a SEPARATE experiment (paper #4); by lift it is comparable to the late lever.
fig, c = plt.subplots(figsize=(5.4,3.7))
fig.subplots_adjust(bottom=0.32)
sites = ["verdict feature\nL23 · paper #5","late residual\nL55–63 · here","behavioral interrupt.\ntoken · paper #4"]
vals  = [0.00, 0.42, 0.70]
base  = [0.00, 0.00, 0.30]                     # baselines differ across experiments
cols  = ["#b2182b","#2166ac","#1b7837"]
bars = c.bar(sites, vals, color=cols, width=0.60, edgecolor="white")
# the behavioral bar is a different experiment -> hatch + lighter, mark its baseline
bars[2].set_hatch("//"); bars[2].set_alpha(0.75)
c.bar(2, base[2], width=0.60, color="white", edgecolor="#1b7837", lw=0)   # mask baseline portion
c.bar(2, base[2], width=0.60, color="#cfe8d6")
labels = ["0%\n($\\Delta P\\!\\approx\\!0$)", "42%\nfrom 0%, $p{=}.031$", "70%\nfrom 30% (sep. exp.)"]
for bar,v,lab in zip(bars,vals,labels):
    c.text(bar.get_x()+bar.get_width()/2, v+0.015, lab, ha="center", va="bottom",
           fontsize=8.5, fontweight="bold")
c.set_ylabel("finish / rescue rate achieved")
c.set_ylim(0,0.88)
c.set_title("Termination is known mid-stream, writable only late")
# bracket the controlled contrast
c.annotate("", xy=(1.0,-0.34), xytext=(0.0,-0.34), annotation_clip=False,
           arrowprops=dict(arrowstyle="-", color="#222", lw=1.2))
c.text(0.5,-0.45,"controlled (same experiment):\nnull $\\rightarrow$ real lever",
       ha="center", fontsize=7.8, color="#222", clip_on=False)
c.text(2.0,-0.40,"separate experiment;\ncomparable by lift (+42 vs +40 pp)",
       ha="center", fontsize=7.5, color="#666", clip_on=False)
fig.savefig(OUT/"fig3_graded_law.pdf", bbox_inches="tight"); plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob("fig*_*.pdf"))])
