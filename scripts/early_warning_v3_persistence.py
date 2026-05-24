"""
Early-warning v3: persistence-based signal.

Hypothesis from v2 negative finding: completion-language APPEARS in both
SUCCESS and WANDERING, but in SUCCESS it appears briefly (1-2 turns, then
finish_tool). In WANDERING-A, it PERSISTS across many turns (agent stuck
in verification loop, can't break out to emit finish).

Test signals:
  - streak_max: longest run of consecutive completion-language turns
  - n_complete_total: total turns with completion-language
  - complete_density_late: fraction of late-half turns with completion-language

If WANDERING-A has streak_max ≥ K with K ≥ 3 while SUCCESS streak_max < 3,
the signal is TEMPORALLY EXCLUSIVE and viable for early-warning.

Operational signal v3:
  alert(T) = streak of K consecutive completion-language turns ends at some
             t ≤ T  AND  no finish_tool by T  AND  patch made
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")
INFLECTION = json.load(open(OUT_DIR / "inflection_results.json"))

COMPLETE_PATTERNS = [
    r"\ball tests pass(ed|ing)?\b",
    r"\b(test|tests) (pass|passes|passed|passing|are passing|all passed)\b",
    r"\bsuccessful(ly)? (made|verified|implemented|fixed|completed|tested)\b",
    r"\bcode (is|are) (correct|working|fine|good)\b",
    r"\bchanges? (have been|are) (made|verified|tested|complete)\b",
    r"\b(verify|verified|verification) (the )?final\b",
    r"\bsummary of (the )?(changes|modifications|what was done)\b",
    r"\bdone\b.{0,30}\b(fixing|implementing|making|adding|modifying)\b",
    r"\binfrastructure (issue|limitation|problem)\b",
    r"\bnot (a |an )?code (issue|problem|bug)\b",
    r"\b(env|environment) (issue|limitation|problem)\b",
    r"\bdisplay (server|issue|missing)\b",
    r"\b(works|worked) correctly\b",
    r"\bimplementation is (correct|complete|done)\b",
    r"\bnothing (more|else) (to do|left)\b",
]
COMPLETE_RE = re.compile("|".join(COMPLETE_PATTERNS), re.IGNORECASE)


def text_of(turn): return " ".join((turn.get(k) or "") for k in ("thinking", "content", "raw_response") if isinstance(turn.get(k), str))
def has_complete(turn): return bool(COMPLETE_RE.search(text_of(turn)))
def emit_finish_at(turns):
    for i, t in enumerate(turns):
        for c in t.get("tool_calls", []) or []:
            if isinstance(c, dict) and c.get("name", "").lower().startswith("finish"):
                return i
    return None


def streak_max(bools):
    """Longest run of consecutive True."""
    best = current = 0
    for b in bools:
        current = current + 1 if b else 0
        best = max(best, current)
    return best


def streak_end_turn(bools, K):
    """Earliest turn T at which a streak of length >= K ends."""
    current = 0
    for i, b in enumerate(bools):
        current = current + 1 if b else 0
        if current >= K:
            return i
    return None


def main():
    sub_class = {}
    for t in INFLECTION["per_trajectory"]:
        if t["label"] == 1:
            sub_class[t["iid"]] = "success"
        elif t.get("lock_fail_0.40") is not None:
            sub_class[t["iid"]] = "locked"
        else:
            sub_class[t["iid"]] = "wandering"

    results = json.load(open(PHASE6 / "phase6_results.json"))
    traj = []
    for iid, entry in results.items():
        if iid not in sub_class:
            continue
        trace_path = PHASE6 / "traces" / (iid + ".json")
        if not trace_path.exists():
            continue
        try:
            trace = json.load(open(trace_path))
            turns = trace.get("turns", [])
        except Exception:
            continue
        if not turns:
            continue
        cbools = [has_complete(turn) for turn in turns]
        traj.append({
            "iid": iid,
            "sub_class": sub_class[iid],
            "n_turns": len(turns),
            "complete_bools": cbools,
            "streak_max": streak_max(cbools),
            "n_complete_total": sum(cbools),
            "complete_density_late": sum(cbools[len(cbools)//2:]) / (len(cbools) - len(cbools)//2),
            "finish_emit_turn": emit_finish_at(turns),
            "patch_n_bytes": entry.get("patch_n_bytes", 0),
        })

    by_class = defaultdict(list)
    for t in traj:
        by_class[t["sub_class"]].append(t)

    print(f"Loaded {len(traj)} (success={len(by_class['success'])}, locked={len(by_class['locked'])}, wandering={len(by_class['wandering'])})\n")

    # Compare distributions
    print("=== streak_max distribution per sub-class ===")
    for cls in ["success", "locked", "wandering"]:
        s = [t["streak_max"] for t in by_class[cls]]
        print(f"  {cls}: mean={np.mean(s):.2f}, median={np.median(s):.0f}, q75={np.percentile(s, 75):.0f}, q90={np.percentile(s, 90):.0f}, max={max(s)}")
        # Distribution histogram
        dist = np.bincount(s)
        for k in range(max(s) + 1):
            print(f"    streak_max={k}: {dist[k] if k < len(dist) else 0}/{len(s)}")

    w_streaks = [t["streak_max"] for t in by_class["wandering"]]
    s_streaks = [t["streak_max"] for t in by_class["success"]]
    l_streaks = [t["streak_max"] for t in by_class["locked"]]

    u_ws, p_ws = mannwhitneyu(w_streaks, s_streaks, alternative="greater")
    u_wl, p_wl = mannwhitneyu(w_streaks, l_streaks, alternative="greater")
    print(f"\n  WANDERING > SUCCESS (one-tail): U={u_ws}, p={p_ws:.4f}")
    print(f"  WANDERING > LOCKED (one-tail):  U={u_wl}, p={p_wl:.4f}")

    # n_complete_total
    print("\n=== n_complete_total ===")
    for cls in ["success", "locked", "wandering"]:
        s = [t["n_complete_total"] for t in by_class[cls]]
        print(f"  {cls}: mean={np.mean(s):.2f}, median={np.median(s):.0f}, q75={np.percentile(s, 75):.0f}")

    # complete_density_late
    print("\n=== complete_density_late (fraction of 2nd-half turns with completion-language) ===")
    for cls in ["success", "locked", "wandering"]:
        s = [t["complete_density_late"] for t in by_class[cls]]
        print(f"  {cls}: mean={np.mean(s):.3f}, median={np.median(s):.3f}, q75={np.percentile(s, 75):.3f}")

    # Operational sweep — streak-based alarm
    print("\n=== Operational sweep: alarm = streak ≥ K ends by turn T AND no finish AND patch ===")
    print(f"{'K':>3} {'T_frac':>7} | {'W_rec':>7} {'W_n':>7} {'lead':>5} | {'S_FP':>6} {'S_FP_rate':>11} | {'L_caught':>8}")
    print("-" * 80)
    sweep = []
    for K in [2, 3, 4, 5]:
        for T_frac in [0.50, 0.70, 0.80, 0.90, 1.00]:
            w_caught = 0
            w_leads = []
            s_fp = 0
            l_caught = 0
            for t in traj:
                T = int(T_frac * t["n_turns"])
                end_turn = streak_end_turn(t["complete_bools"][:T+1], K)
                if end_turn is None:
                    continue
                # Skip if no patch
                if t["patch_n_bytes"] == 0:
                    continue
                # Alarm fires if no finish by end_turn
                if t["finish_emit_turn"] is not None and t["finish_emit_turn"] <= end_turn:
                    continue
                if t["sub_class"] == "wandering":
                    w_caught += 1
                    w_leads.append(t["n_turns"] - end_turn)
                elif t["sub_class"] == "success":
                    # FP if success eventually emits finish AFTER alarm
                    if t["finish_emit_turn"] is not None and t["finish_emit_turn"] > end_turn:
                        s_fp += 1
                elif t["sub_class"] == "locked":
                    l_caught += 1
            wcr = w_caught / len(by_class["wandering"])
            sfpr = s_fp / len(by_class["success"])
            lm = float(np.median(w_leads)) if w_leads else None
            row = {"K": K, "T_frac": T_frac, "wander_caught": w_caught, "wander_recall": wcr,
                   "lead_median": lm, "success_fp": s_fp, "success_fp_rate": sfpr,
                   "locked_caught": l_caught}
            sweep.append(row)
            lm_str = f"{lm:.0f}" if lm is not None else "  -"
            print(f"{K:>3} {T_frac:>7.2f} | {100*wcr:>5.0f}% {w_caught:>2}/{len(by_class['wandering']):<2} {lm_str:>5} | {s_fp:>2}/{len(by_class['success']):<2} {100*sfpr:>9.1f}% | {l_caught:>3}/{len(by_class['locked'])}")

    print("\n=== Comparison summary ===")
    print(f"  v1 post-hoc:              35% recall, 0%  FP, lead=0")
    print(f"  v2 W=20 B=0.8:            30% recall, 15% FP, lead=10")
    # Best v3 op point: K=2 or K=3 at T=1.0
    best_v3 = sorted(sweep, key=lambda r: r["wander_recall"] - 2 * r["success_fp_rate"], reverse=True)[0]
    print(f"  v3 best (K={best_v3['K']}, T={best_v3['T_frac']}): {100*best_v3['wander_recall']:.0f}% recall, {100*best_v3['success_fp_rate']:.1f}% FP, lead={best_v3['lead_median']}")
    # Best v3 at T<1 (true early-warning)
    early_only = [r for r in sweep if r["T_frac"] < 1.0]
    if early_only:
        best_v3_early = sorted(early_only, key=lambda r: r["wander_recall"] - 2 * r["success_fp_rate"], reverse=True)[0]
        print(f"  v3 best early-only (T<1, K={best_v3_early['K']}, T={best_v3_early['T_frac']}): {100*best_v3_early['wander_recall']:.0f}% recall, {100*best_v3_early['success_fp_rate']:.1f}% FP, lead={best_v3_early['lead_median']}")

    # Save
    (OUT_DIR / "early_warning_v3_persistence.json").write_text(json.dumps({
        "streak_max_per_class": {cls: {"mean": float(np.mean([t["streak_max"] for t in by_class[cls]])),
                                       "median": float(np.median([t["streak_max"] for t in by_class[cls]])),
                                       "max": int(max([t["streak_max"] for t in by_class[cls]])),
                                       "values": [int(t["streak_max"]) for t in by_class[cls]]}
                                 for cls in ["success", "locked", "wandering"]},
        "mannwhitney_W_vs_S_greater": {"U": float(u_ws), "p": float(p_ws)},
        "mannwhitney_W_vs_L_greater": {"U": float(u_wl), "p": float(p_wl)},
        "sweep": sweep,
    }, indent=2))
    print(f"\nSaved {OUT_DIR}/early_warning_v3_persistence.json")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, max(max(w_streaks), max(s_streaks), max(l_streaks)) + 2)
    for cls, color in [("success", "tab:green"), ("locked", "tab:orange"), ("wandering", "tab:red")]:
        s = [t["streak_max"] for t in by_class[cls]]
        ax.hist(s, bins=bins, alpha=0.5, label=f"{cls} (n={len(s)}, median={np.median(s):.0f})",
                color=color, edgecolor="black")
    ax.set_xlabel("streak_max (longest consecutive completion-language turns)")
    ax.set_ylabel("Trajectory count")
    ax.set_title("Persistence of completion-language verbalization, by sub-class")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT_DIR / "early_warning_v3_streak_dist.png", dpi=120, bbox_inches="tight")
    print(f"Saved plot to {OUT_DIR}/early_warning_v3_streak_dist.png")


if __name__ == "__main__":
    main()
