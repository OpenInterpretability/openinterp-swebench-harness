#!/usr/bin/env python3
"""Regenerate the Exp B paired confusion matrix figure for paper #2.

Source of truth = the per-instance paired table in two_honest_nulls.tex
(\\label{tab:exp_b_per_instance}), N=20 WANDERING instances:

                       Primary (hook L55 SUCCESS donor, alpha=0.3)
                       max_turns   finish_tool
  Baseline max_turns      11            2     <- primary-only flips (hook "rescued")
  Baseline finish_tool     3            4     <- 3 = baseline-only flips (hook "blocked")

Concordant (same outcome both): 11 + 4 = 15/20. Discordant: 3 + 2 = 5.
McNemar exact two-tailed p = 1.00 (binomial CDF saturated at n_discordant=5).

Writes figures/exp_b_confusion_matrix.{pdf,png}.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# rows = Baseline outcome, cols = Primary outcome; order [max_turns, finish_tool]
M = np.array([[11, 2],
              [3, 4]])
assert M.sum() == 20, "must total 20 instances"
labels = ["max\\_turns", "finish\\_tool"]

fig, ax = plt.subplots(figsize=(5.0, 4.2))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max())

ax.set_xticks([0, 1], labels=["max_turns", "finish_tool"])
ax.set_yticks([0, 1], labels=["max_turns", "finish_tool"])
ax.set_xlabel("Primary (hook L55 SUCCESS donor, $\\alpha=0.3$)", fontsize=11)
ax.set_ylabel("Baseline (no hook)", fontsize=11)
ax.set_title("Exp B paired outcomes ($N=20$ WANDERING)", fontsize=12, pad=10)

annot = [["11\n(max both)", "2\n(primary only)"],
         ["3\n(baseline only)", "4\n(finish both)"]]
thresh = M.max() / 2.0
for i in range(2):
    for j in range(2):
        ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=11,
                color="white" if M[i, j] > thresh else "black")

# mark the discordant (off-diagonal) cells that drive McNemar
for (i, j) in [(0, 1), (1, 0)]:
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="#c0392b", lw=2.2))

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="instances")
fig.tight_layout()

out_dir = Path(__file__).resolve().parent.parent / "paper" / "paper2" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    p = out_dir / f"exp_b_confusion_matrix.{ext}"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"wrote {p}")
