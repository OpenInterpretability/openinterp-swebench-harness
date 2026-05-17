"""
Paper-MEGA "Conditionally-Causal Probes" — generate 3 figures.

Inputs: hardcoded measurements from papers 5/6/8 and Phase 6c/10/11/2A/2B/2C.
Outputs: 3 PNGs in public/images/papers/mega-conditionally-causal/

Run from repo root:
    python3 scripts/paper_mega_figures.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "public" / "images" / "papers" / "mega-conditionally-causal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Brand palette (matches openinterp.org Tailwind brand-*)
# ============================================================
BRAND = {
    "ink_900": "#0f172a",
    "ink_700": "#334155",
    "ink_500": "#64748b",
    "ink_300": "#cbd5e1",
    "ink_100": "#f1f5f9",
    "brand_600": "#4f46e5",
    "brand_400": "#818cf8",
    "accent_500": "#06b6d4",
    "red_500": "#ef4444",
    "amber_500": "#f59e0b",
    "green_500": "#10b981",
    "purple_500": "#8b5cf6",
    "white": "#ffffff",
}

# Regime → color
REGIME_COLOR = {
    "R1 causal trajectory-shaping": BRAND["green_500"],
    "R2 pushup-asymmetric": BRAND["brand_600"],
    "R3 pushdown-asymmetric": BRAND["accent_500"],
    "R4 structurally-locked": BRAND["ink_500"],
    "R5 epiphenomenal-softmax-temp": BRAND["red_500"],
}

# ============================================================
# Probe table (single source of truth for all 3 figures)
# ============================================================
PROBES = [
    # name, layer, position, regime
    ("ST_L31_gen", 31, "free-gen", "R1 causal trajectory-shaping"),
    ("ST_L11_gen", 11, "free-gen", "R4 structurally-locked"),
    ("ST_L55_gen", 55, "free-gen", "R4 structurally-locked"),
    ("SWE_L43_pre_tool", 43, "pre_tool", "R5 epiphenomenal-softmax-temp"),
    ("CoT_L55_mid_think", 55, "mid_think", "R4 structurally-locked"),
    ("FG_L31_pre_tool", 31, "pre_tool", "R5 epiphenomenal-softmax-temp"),
    ("RG_L55_mid_think", 55, "mid_think", "R2 pushup-asymmetric"),
    ("Cap_L23_pre_tool", 23, "pre_tool", "R3 pushdown-asymmetric"),
    ("Cap_L31_pre_tool", 31, "pre_tool", "R3 pushdown-asymmetric"),
    ("Cap_L43_turn_end", 43, "turn_end", "R3 pushdown-asymmetric"),
    ("Cap_L55_pre_tool", 55, "pre_tool", "R3 pushdown-asymmetric"),
]

# α-sweep gap-from-random (percentage points). Negative = no measurable effect.
# Source: paper-8 Phase 2A/2C, paper-5 Phase 10/11/11e, paper-6 Phase 7/8.
# Zeros indicate "tested, no measurable lever above random control".
# NaN indicates "not measured at that α".
# Convention: positive value = lever-from-random gap in pp at that α.
ALPHA_GRID = [-200, -100, -50, -25, -5, 5, 25, 50, 100, 200]

# Hand-curated from explicit reported values in papers 5/6/8 across
# Phase 2A/2C/7/8/10/11/11b/11e. HE+MBPP baseline distribution.
# Cells with explicit measurements are filled; others left at 0.
# Each row matches PROBES order.
ALPHA_GAP = np.array([
    # ST_L31_gen — Phase 2A: 9/14 (~64%) at α=+50 vs random 2/14 (~14%) = +50pp gap
    [0,    0,    0,    0,    0,    0,    0,   50,    0,    0],
    # ST_L11_gen — Phase 2C: 1/10 ragged at α=+25 (~10pp gap); inert elsewhere up to ±500
    [0,    0,    0,    0,    0,    0,   10,    0,    0,    0],
    # ST_L55_gen — Phase 2C: inert at α up to ±500
    [0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    # SWE_L43_pre_tool — Phase 7: Δrel ≈ 0 once control-token normalized
    [0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    # CoT_L55_mid_think — Phase 8: zero behavioral change ±5 to ±500
    [0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    # FG_L31_pre_tool — Phase 10: flips match random at all tested α
    [0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    # RG_L55_mid_think — Phase 10: 0% until +100, 32% at +200 (vs 2% random = +30pp gap)
    [0,    0,    0,    0,    0,    0,    0,    0,    0,   30],
    # Cap_L23_pre_tool — Phase 11/11e: +43pp at α=-100 (HE+MBPP)
    [0,   43,    0,    0,    0,    0,    0,    0,    0,    0],
    # Cap_L31_pre_tool — Phase 11: +40pp at α=-100 (HE+MBPP)
    [0,   40,    0,    0,    0,    0,    0,    0,    0,    0],
    # Cap_L43_turn_end — Phase 11b/11e: +60pp at α=-200, +40pp at α=-100 (HE+MBPP)
    [60,  40,    0,    0,    0,    0,    0,    0,    0,    0],
    # Cap_L55_pre_tool — Phase 11/11e: +34pp at α=-100, +7pp pushup at α=+200 (HE+MBPP)
    [0,   34,    0,    0,    0,    0,    0,    0,    0,    7],
], dtype=float)

# ============================================================
# FIGURE 1 — Probe regime map (layer × position scatter)
# ============================================================
def figure_1_probe_regime_map():
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    fig.patch.set_facecolor(BRAND["white"])
    ax.set_facecolor(BRAND["ink_100"])

    position_y = {"pre_tool": 0, "mid_think": 1, "turn_end": 2, "free-gen": 3}

    # Group probes by (layer, position) to handle overlap
    from collections import defaultdict
    bucket = defaultdict(list)
    for name, layer, pos, regime in PROBES:
        bucket[(layer, pos)].append((name, regime))

    for (layer, pos), items in bucket.items():
        n = len(items)
        for i, (name, regime) in enumerate(items):
            jitter = (i - (n - 1) / 2) * 0.18
            x = layer + jitter
            y = position_y[pos]
            color = REGIME_COLOR[regime]
            ax.scatter([x], [y], s=280, color=color, edgecolors=BRAND["ink_900"],
                       linewidths=1.5, zorder=3)

    # Layer grid
    for L in [11, 23, 31, 43, 55]:
        ax.axvline(L, color=BRAND["ink_300"], linestyle=":", linewidth=0.8, zorder=1)
        ax.text(L, -0.5, f"L{L}", ha="center", va="top",
                color=BRAND["ink_700"], fontsize=10, fontweight="bold")

    ax.set_yticks(list(position_y.values()))
    ax.set_yticklabels(list(position_y.keys()), fontsize=10, color=BRAND["ink_700"])
    ax.set_xticks([])
    ax.set_xlim(8, 58)
    ax.set_ylim(-0.8, 3.5)
    ax.set_xlabel("Layer depth →", color=BRAND["ink_700"], fontsize=10, labelpad=18)
    ax.set_ylabel("Generation position", color=BRAND["ink_700"], fontsize=10)
    ax.set_title("Figure 1 — The eleven causality-tested probes in Qwen3.6-27B",
                 color=BRAND["ink_900"], fontsize=13, fontweight="bold", pad=15)

    # Legend
    legend_elements = [
        plt.scatter([], [], color=c, s=120, edgecolors=BRAND["ink_900"],
                    linewidths=1.2, label=r)
        for r, c in REGIME_COLOR.items()
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=9, title="Empirical regime",
              title_fontsize=10, alignment="left")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(BRAND["ink_300"])

    plt.tight_layout()
    out = OUT_DIR / "fig1_probe_regime_map.png"
    plt.savefig(out, bbox_inches="tight", facecolor=BRAND["white"])
    plt.close()
    print(f"wrote {out}")

# ============================================================
# FIGURE 2 — α-sweep gap-from-random heatmap (11 probes × 10 α)
# ============================================================
def figure_2_alpha_sweep_heatmap():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    fig.patch.set_facecolor(BRAND["white"])

    cmap = LinearSegmentedColormap.from_list(
        "brand_gap",
        [BRAND["ink_100"], BRAND["brand_400"], BRAND["brand_600"], BRAND["ink_900"]],
    )

    im = ax.imshow(ALPHA_GAP, aspect="auto", cmap=cmap, vmin=0, vmax=60)

    ax.set_xticks(range(len(ALPHA_GRID)))
    ax.set_xticklabels([f"{a:+d}" for a in ALPHA_GRID], fontsize=9,
                       color=BRAND["ink_700"])
    ax.set_yticks(range(len(PROBES)))
    ax.set_yticklabels([p[0] for p in PROBES], fontsize=9,
                       color=BRAND["ink_700"])
    ax.set_xlabel("Steering α (multiples of unit probe direction)",
                  color=BRAND["ink_700"], fontsize=10, labelpad=8)
    ax.set_title(
        "Figure 2 — α-sweep behavioral gap from random-direction control (percentage points)",
        color=BRAND["ink_900"], fontsize=12, fontweight="bold", pad=15
    )

    # Annotate non-zero cells
    for i in range(ALPHA_GAP.shape[0]):
        for j in range(ALPHA_GAP.shape[1]):
            v = ALPHA_GAP[i, j]
            if v >= 5:
                color = BRAND["white"] if v > 30 else BRAND["ink_900"]
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        color=color, fontsize=8, fontweight="bold")

    # Color marker bar for regime per row
    for i, (_, _, _, regime) in enumerate(PROBES):
        ax.add_patch(plt.Rectangle((-1.0, i - 0.45), 0.4, 0.9,
                                   color=REGIME_COLOR[regime], clip_on=False))

    ax.set_xlim(-1.1, len(ALPHA_GRID) - 0.5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Gap from random (pp)", color=BRAND["ink_700"], fontsize=9)
    cbar.ax.tick_params(colors=BRAND["ink_700"], labelsize=8)

    # Legend for regime color strip
    legend_elements = [
        plt.scatter([], [], color=c, marker="s", s=80, label=r)
        for r, c in REGIME_COLOR.items()
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.18, 1.0),
              frameon=False, fontsize=8, title="Regime", title_fontsize=9,
              alignment="left")

    for spine in ax.spines.values():
        spine.set_color(BRAND["ink_300"])

    plt.tight_layout()
    out = OUT_DIR / "fig2_alpha_sweep_heatmap.png"
    plt.savefig(out, bbox_inches="tight", facecolor=BRAND["white"])
    plt.close()
    print(f"wrote {out}")

# ============================================================
# FIGURE 3 — Constraint-violation decision flowchart
# ============================================================
def figure_3_constraint_flowchart():
    fig, ax = plt.subplots(figsize=(11, 7), dpi=160)
    fig.patch.set_facecolor(BRAND["white"])
    ax.set_facecolor(BRAND["white"])

    def box(x, y, w, h, text, fill=BRAND["ink_100"], fontcolor=BRAND["ink_900"],
            bold=False):
        b = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.02",
                           linewidth=1.2, edgecolor=BRAND["ink_700"],
                           facecolor=fill)
        ax.add_patch(b)
        ax.text(x, y, text, ha="center", va="center", color=fontcolor,
                fontsize=9, fontweight="bold" if bold else "normal",
                wrap=True)

    def arrow(x1, y1, x2, y2, label=None, dy=0):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>", mutation_scale=12,
                            color=BRAND["ink_700"], linewidth=1.0)
        ax.add_patch(a)
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label,
                    ha="center", va="center", fontsize=8,
                    color=BRAND["ink_700"],
                    bbox=dict(facecolor=BRAND["white"], edgecolor="none",
                              pad=2))

    # Title
    ax.text(5.5, 7.6, "Figure 3 — Constraint-violation decision flowchart",
            ha="center", fontsize=13, fontweight="bold", color=BRAND["ink_900"])
    ax.text(5.5, 7.2, "Diagnostic order for assigning a candidate probe to one of R1-R5",
            ha="center", fontsize=10, color=BRAND["ink_500"])

    # Decision nodes (centered x-coords)
    box(5.5, 6.4, 4.5, 0.6, "Probe predicts target with AUROC > 0.65?", fill=BRAND["brand_400"])
    arrow(5.5, 6.1, 5.5, 5.7, "yes")
    arrow(5.5, 6.1, 2.0, 5.7, "no", dy=0.05)

    # No path → not a useful probe
    box(2.0, 5.4, 2.8, 0.5, "Not a useful probe (no signal)", fill=BRAND["ink_300"])

    # Yes path → D1 + D2
    box(5.5, 5.4, 4.5, 0.6, "D1 + D2 pass? (above random + above shuffled-source)",
        fill=BRAND["brand_400"])
    arrow(5.5, 5.1, 5.5, 4.7, "yes")
    arrow(5.5, 5.1, 9.5, 4.7, "no", dy=0.05)

    box(9.5, 4.4, 2.8, 0.55, "Marginal-fit or\nover-parameterization", fill=BRAND["amber_500"],
        fontcolor=BRAND["white"])

    # D3 control-token
    box(5.5, 4.4, 4.5, 0.6, "D3 control-token Δrel ≠ 0?",
        fill=BRAND["brand_400"])
    arrow(5.5, 4.1, 5.5, 3.7, "yes")
    arrow(5.5, 4.1, 9.5, 3.7, "no", dy=0.05)

    box(9.5, 3.4, 2.8, 0.55, "R5\nEpiphenomenal\n(softmax-temp)",
        fill=REGIME_COLOR["R5 epiphenomenal-softmax-temp"], fontcolor=BRAND["white"])

    # D4 structural rigidity
    box(5.5, 3.4, 4.5, 0.6, "D4 α-sweep to ±200 produces behavioral change?",
        fill=BRAND["brand_400"])
    arrow(5.5, 3.1, 5.5, 2.7, "yes")
    arrow(5.5, 3.1, 9.5, 2.7, "no", dy=0.05)

    box(9.5, 2.4, 2.8, 0.55, "R4\nStructurally-locked\n(template / inert)",
        fill=REGIME_COLOR["R4 structurally-locked"], fontcolor=BRAND["white"])

    # D5 + D6 + decision asymmetry classify R1-R3
    box(5.5, 2.4, 4.5, 0.6,
        "D5 strip-flip passes? D6 onset-timing decay?",
        fill=BRAND["brand_400"])
    arrow(5.5, 2.1, 2.5, 1.4, "yes,\nyes")
    arrow(5.5, 2.1, 5.5, 1.4, "yes,\nflat")
    arrow(5.5, 2.1, 8.5, 1.4, "no", dy=0.05)

    box(2.5, 1.1, 2.6, 0.65, "R1\nCausal\ntrajectory-shaping",
        fill=REGIME_COLOR["R1 causal trajectory-shaping"], fontcolor=BRAND["white"])
    box(5.5, 1.1, 2.6, 0.65,
        "R2 (pushup) or R3\n(pushdown) by C4\nsaturation direction",
        fill=REGIME_COLOR["R2 pushup-asymmetric"], fontcolor=BRAND["white"])
    box(8.5, 1.1, 2.6, 0.65, "False flip\n(tokenization /\nrandom OOD)", fill=BRAND["amber_500"],
        fontcolor=BRAND["white"])

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")

    plt.tight_layout()
    out = OUT_DIR / "fig3_constraint_flowchart.png"
    plt.savefig(out, bbox_inches="tight", facecolor=BRAND["white"])
    plt.close()
    print(f"wrote {out}")

# ============================================================
if __name__ == "__main__":
    figure_1_probe_regime_map()
    figure_2_alpha_sweep_heatmap()
    figure_3_constraint_flowchart()
    print(f"\nDone. 3 figures in {OUT_DIR.relative_to(REPO)}/")
