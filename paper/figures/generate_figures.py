"""
Generate publication-ready figures for the Tool-Entropy Collapse paper.

Outputs to paper/figures/:
  fig1_cross_arch_entropy.png       — KEY: tool-entropy distributions per
                                       sub-class for Qwen + Llama + GPT-5
  fig2_disagreement_trajectory.png  — (copy from existing) cross-layer
                                       disagreement evolution
  fig3_detector_comparison.png      — 6 detectors recall × FP scatter +
                                       3-tier deployment regions
  fig4_lab_summary.png              — 4-lab W/S ratio bar chart with
                                       significance markers
  fig5_venn_orthogonality.png       — v1 ∩ v4 ∩ v5 captures on WANDERING
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

ROOT = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness")
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "scripts" / "inflection_turn_out"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "success": "#2ca02c",
    "locked": "#ff7f0e",
    "wandering": "#d62728",
}


# ============================================================
# Fig 1: Cross-architecture tool-entropy distribution (HEADLINE)
# ============================================================
def fig1_cross_arch_entropy():
    # Load Qwen v5 results
    qwen_v5 = json.load(open(DATA / "v5_tool_entropy.json"))
    qwen_by = {"success": [], "locked": [], "wandering": []}
    for t in qwen_v5:
        if t["sub_class"] in qwen_by:
            qwen_by[t["sub_class"]].append(t["tool_entropy_last10"])

    # Load Llama
    llama_df = pd.read_csv(DATA / "v5_cross_model_llama_full.csv")
    llama_by = {cls: llama_df[llama_df["sub_class"] == cls]["tool_entropy_last10"].tolist()
                for cls in ["success", "locked", "wandering"]}

    # Load GPT-5
    gpt_df = pd.read_csv(DATA / "v5_gpt_metrics.csv")
    gpt_by = {cls: gpt_df[gpt_df["sub_class"] == cls]["tool_entropy_last10"].tolist()
              for cls in ["success", "locked", "wandering"]}

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=False)
    titles = [
        f"Qwen3.6-27B (Alibaba)\nW/S ratio = 0.41, p = 1.0×10⁻⁶",
        f"Llama-70b (Meta)\nW/S ratio = 0.41, p < 10⁻¹⁵",
        f"GPT-5 router (OpenAI)\nW/S ratio = 0.71, p = 8.9×10⁻³⁵",
    ]
    data_list = [qwen_by, llama_by, gpt_by]
    xmax = [2.5, 4.0, 4.5]
    for ax, by, title, xm in zip(axes, data_list, titles, xmax):
        for cls, color in COLORS.items():
            vals = [v for v in by[cls] if v is not None and not np.isnan(v)]
            if vals:
                ax.hist(vals, bins=20, alpha=0.55, label=f"{cls} (n={len(vals)})",
                        color=color, edgecolor="white", linewidth=0.5)
                med = np.median(vals)
                ax.axvline(med, color=color, linestyle="--", linewidth=1.2)
        ax.set_xlabel("tool_entropy_last10 (Shannon, last 10 turns)")
        ax.set_xlim(0, xm)
        ax.set_title(title)
        ax.legend(loc="upper right", framealpha=0.9)
    axes[0].set_ylabel("Trajectory count")
    fig.suptitle("Tool-entropy collapse across architectures: WANDERING agents (red) collapse to ~40-71% of SUCCESS entropy",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    p = OUT / "fig1_cross_arch_entropy.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")


# ============================================================
# Fig 2: Cross-layer disagreement trajectory (already exists, recreate cleaner)
# ============================================================
def fig2_disagreement_trajectory():
    v4 = json.load(open(DATA / "early_warning_v4_cross_layer.json"))
    by_class = {"success": [], "locked": [], "wandering": []}
    for t in v4["per_trajectory"]:
        if t["sub_class"] in by_class:
            by_class[t["sub_class"]].append(t)

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    n_bins = 20
    for cls, color in COLORS.items():
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        for t in by_class[cls]:
            for turn_idx, r in enumerate(t["ranges_per_turn"]):
                bi = min(int(n_bins * turn_idx / t["n_turns"]), n_bins - 1)
                bin_sums[bi] += r
                bin_counts[bi] += 1
        mean_per_bin = bin_sums / np.maximum(bin_counts, 1)
        x = (np.arange(n_bins) + 0.5) / n_bins
        ax.plot(x, mean_per_bin, marker="o", markersize=4, color=color,
                label=f"{cls} (n={len(by_class[cls])})", linewidth=1.8)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8, label="late-half boundary")
    ax.set_xlabel("Fraction of trajectory length")
    ax.set_ylabel("Cross-layer range(t) = max$_L$ probe − min$_L$ probe")
    ax.set_title("Cross-layer probe disagreement evolution (5-fold CV, Qwen3.6-27B Phase 6)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = OUT / "fig2_disagreement_trajectory.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")


# ============================================================
# Fig 3: Detector comparison — 6 detectors with 3-tier regions
# ============================================================
def fig3_detector_comparison():
    # (name, recall, FP, tier, color, marker, label_dx, label_dy)
    detectors = [
        ("v1 post-hoc text",        0.35, 0.00, 1,    "#1f77b4", "o",   8, -12),
        ("v2 naive early-warning",  0.30, 0.15, None, "#7f7f7f", "X",   8,   8),
        ("v3 persistence (refuted)",0.10, 0.05, None, "#7f7f7f", "X",   8,  12),
        ("v4 cross-layer",          0.65, 0.30, 2,    "#9467bd", "s",   8,   8),
        ("v5 tool-entropy",         0.55, 0.05, 3,    "#d62728", "D",   8,  -4),
        ("v6 residual stability",   0.15, 0.05, None, "#7f7f7f", "X",   8,  -16),
        ("v1 ∪ v4 (Tier 2)",        0.80, 0.30, 2,    "#9467bd", "*",  10,  -12),
        ("v1 ∪ v5 (Tier 3)",        0.70, 0.05, 3,    "#d62728", "*",  10,  12),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    # Shade tier regions
    ax.axhspan(0, 0.05, xmin=0, xmax=1, color="#d62728", alpha=0.07)
    ax.axhspan(0.05, 0.30, xmin=0, xmax=1, color="#9467bd", alpha=0.07)
    ax.text(0.02, 0.027, "Tier 3 region (FP ≤ 5%)", fontsize=8.5, color="#a01010", style="italic")
    ax.text(0.02, 0.175, "Tier 2 region (FP ≤ 30%)", fontsize=8.5, color="#5a3a7a", style="italic")
    ax.text(0.02, 0.305, "Outside admissible tiers", fontsize=8.5, color="#7f7f7f", style="italic")

    for name, recall, fp, tier, color, marker, dx, dy in detectors:
        ax.scatter(recall, fp, c=color, marker=marker, s=180, edgecolor="black",
                   linewidth=0.6, zorder=5)
        ax.annotate(name, (recall, fp), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    ax.set_xlabel("WANDERING recall")
    ax.set_ylabel("SUCCESS false-positive rate")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.02, 0.35)
    ax.set_title("Detector comparison: 6 designs across 3 signal channels (text / residual / action)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = OUT / "fig3_detector_comparison.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")


# ============================================================
# Fig 4: 4-lab cross-validation W/S ratio
# ============================================================
def fig4_lab_summary():
    data = [
        # (model, lab, ratio, p_text, n_w, n_s, valid)
        ("Qwen3.6-27B", "Alibaba", 0.41, "p = 1.0×10⁻⁶", 20, 40, True),
        ("Llama-70b", "Meta", 0.41, "p < 10⁻¹⁵", 1358, 488, True),
        ("GPT-5 router", "OpenAI", 0.71, "p = 8.9×10⁻³⁵", 137, 664, True),
        ("Claude 3.7 Sonnet", "Anthropic", 1.00, "p = 0.16 (UNSUITABLE)", 1116, 1813, False),
    ]
    models = [d[0] + f"\n({d[1]})" for d in data]
    ratios = [d[2] for d in data]
    p_texts = [d[3] for d in data]
    ns = [f"n_W={d[4]:,}\nn_S={d[5]:,}" for d in data]
    colors = ["#2ca02c" if d[6] else "#a0a0a0" for d in data]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.barh(models, ratios, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="no discrimination")
    ax.axvline(0.41, color="#d62728", linestyle=":", linewidth=1.5,
               label="ratio invariance (Qwen=Llama)")
    for bar, ratio, p_text, n_text in zip(bars, ratios, p_texts, ns):
        ax.text(ratio + 0.02, bar.get_y() + bar.get_height() / 2,
                f"  {ratio:.2f}  ({p_text})  {n_text}",
                va="center", fontsize=8)
    ax.set_xlabel("WANDERING / SUCCESS tool-entropy ratio (medians)")
    ax.set_xlim(0, 1.4)
    ax.set_title("Cross-architecture validation: 3 of 4 labs confirm tool-entropy collapse")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    p = OUT / "fig4_lab_summary.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")


# ============================================================
# Fig 5: Venn diagram of v1 ∩ v4 ∩ v5 on WANDERING captures
# ============================================================
def fig5_venn_orthogonality():
    # Per inflection_results + complementary_monitor + v4 + v5 datasets,
    # compute which WANDERING trajectories each detector catches at its best op.
    infl = json.load(open(DATA / "inflection_results.json"))
    v1_sig = json.load(open(DATA / "complementary_monitor.json"))
    v4 = json.load(open(DATA / "early_warning_v4_cross_layer.json"))
    v5 = json.load(open(DATA / "v5_tool_entropy.json"))

    # Sub-class
    sub_class = {}
    final_probe = {}
    for t in infl["per_trajectory"]:
        if t["label"] == 1:
            sub_class[t["iid"]] = "success"
        elif t.get("lock_fail_0.40") is not None:
            sub_class[t["iid"]] = "locked"
        else:
            sub_class[t["iid"]] = "wandering"
        final_probe[t["iid"]] = t.get("final_score", 0.5)

    v1_lookup = {s["iid"]: s for s in v1_sig["per_trajectory_signals"]}
    v4_lookup = {m["iid"]: m for m in v4["per_trajectory"]}
    v5_lookup = {m["iid"]: m for m in v5}

    wander_iids = [iid for iid, cls in sub_class.items() if cls == "wandering"]

    v1_caught, v4_caught, v5_caught = set(), set(), set()
    for iid in wander_iids:
        v1 = v1_lookup.get(iid, {})
        v4i = v4_lookup.get(iid, {})
        v5i = v5_lookup.get(iid, {})
        if v1:
            probe_neg = final_probe.get(iid, 0.5) < 0.5
            text_pos = (v1.get("m_complete_last5", 0) >= 0.4 and
                        not v1.get("emit_finish", False) and
                        v1.get("patch_n_bytes", 0) > 0)
            if probe_neg or text_pos:
                v1_caught.add(iid)
        if v4i:
            ranges = v4i.get("ranges_per_turn", [])
            n = v4i.get("n_turns", 50)
            T = min(int(0.7 * n), len(ranges) - 1)
            if T >= 5:
                half = T // 2
                if float(np.mean(ranges[half:T + 1])) > 0.52:
                    v4_caught.add(iid)
        if v5i:
            if v5i.get("tool_entropy_last10", 999) < 0.50:
                v5_caught.add(iid)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    v = venn3([v1_caught, v4_caught, v5_caught], ("v1 post-hoc text", "v4 cross-layer", "v5 tool-entropy"),
              set_colors=("#1f77b4", "#9467bd", "#d62728"), ax=ax)
    n_total = len(wander_iids)
    n_missed = n_total - len(v1_caught | v4_caught | v5_caught)
    ax.set_title(f"Detector orthogonality on WANDERING captures (n = {n_total}); "
                 f"{n_missed} missed by all three")
    plt.tight_layout()
    p = OUT / "fig5_venn_orthogonality.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p}")
    print(f"  v1_only: {len(v1_caught - v4_caught - v5_caught)}, "
          f"v4_only: {len(v4_caught - v1_caught - v5_caught)}, "
          f"v5_only: {len(v5_caught - v1_caught - v4_caught)}")
    print(f"  v1∩v5: {len(v1_caught & v5_caught - v4_caught)}, "
          f"v1∩v4: {len(v1_caught & v4_caught - v5_caught)}, "
          f"v4∩v5: {len(v4_caught & v5_caught - v1_caught)}")
    print(f"  all three: {len(v1_caught & v4_caught & v5_caught)}, "
          f"union: {len(v1_caught | v4_caught | v5_caught)}/{n_total}")


if __name__ == "__main__":
    fig1_cross_arch_entropy()
    fig2_disagreement_trajectory()
    fig3_detector_comparison()
    fig4_lab_summary()
    fig5_venn_orthogonality()
    print("\nAll figures saved to", OUT)
