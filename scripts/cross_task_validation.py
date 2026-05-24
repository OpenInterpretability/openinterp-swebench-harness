"""
Cross-task validation of v1 + v4 detectors.

The Phase 6 dataset spans multiple repos. v4's signal could be:
  (a) genuine — disagreement pattern holds across repos
  (b) qutebrowser-specific — driven by one repo's idiosyncrasies
  (c) mixed — holds in some repos, breaks in others

Per-repo analysis:
  1. Distribution of trajectories by repo and by sub-class
  2. WANDERING-vs-SUCCESS disagreement per repo (Mann-Whitney + medians)
  3. v1∪v4 operational performance per repo

Output: per-repo summary table. Pattern interpretation guides paper §7 caveat
language.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")
PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")

INFLECTION = json.load(open(OUT_DIR / "inflection_results.json"))
V4 = json.load(open(OUT_DIR / "early_warning_v4_cross_layer.json"))
V1 = json.load(open(OUT_DIR / "complementary_monitor.json"))


def parse_repo(iid):
    """e.g. instance_qutebrowser__qutebrowser-2dd896... → qutebrowser"""
    if iid.startswith("instance_"):
        s = iid[len("instance_"):]
    else:
        s = iid
    # Split on __ — first part is org, second is repo
    parts = s.split("__")
    if len(parts) >= 2:
        return parts[1].split("-")[0]
    return s.split("-")[0]


def main():
    # Build per-iid metadata
    sub_class = {}
    final_probe = {}
    for t in INFLECTION["per_trajectory"]:
        if t["label"] == 1:
            sub_class[t["iid"]] = "success"
        elif t.get("lock_fail_0.40") is not None:
            sub_class[t["iid"]] = "locked"
        else:
            sub_class[t["iid"]] = "wandering"
        final_probe[t["iid"]] = t.get("final_score", 0.5)

    v4_signals = {m["iid"]: m for m in V4["per_trajectory"]}
    v1_signals = {s["iid"]: s for s in V1["per_trajectory_signals"]}

    # Group by repo
    repos = defaultdict(list)
    for iid, sc in sub_class.items():
        repo = parse_repo(iid)
        repos[repo].append({
            "iid": iid,
            "sub_class": sc,
            "final_probe": final_probe.get(iid),
            "v4": v4_signals.get(iid, {}),
            "v1": v1_signals.get(iid, {}),
        })

    print("=== Repo distribution ===")
    for repo in sorted(repos.keys()):
        ts = repos[repo]
        n_s = sum(1 for t in ts if t["sub_class"] == "success")
        n_l = sum(1 for t in ts if t["sub_class"] == "locked")
        n_w = sum(1 for t in ts if t["sub_class"] == "wandering")
        print(f"  {repo}: n={len(ts)}, success={n_s}, locked={n_l}, wandering={n_w}")

    # Per-repo Mann-Whitney on range_late (v4 primary metric)
    print("\n=== Per-repo: WANDERING vs SUCCESS on range_late (v4) ===")
    print(f"{'repo':>18} {'n_W':>4} {'n_S':>4} {'W_med':>7} {'S_med':>7} {'effect':>7} {'p':>8} {'direction':>9}")
    for repo in sorted(repos.keys()):
        ts = repos[repo]
        w_vals = [t["v4"].get("range_late", np.nan) for t in ts if t["sub_class"] == "wandering" and t["v4"]]
        s_vals = [t["v4"].get("range_late", np.nan) for t in ts if t["sub_class"] == "success" and t["v4"]]
        w_vals = [v for v in w_vals if not np.isnan(v)]
        s_vals = [v for v in s_vals if not np.isnan(v)]
        if not w_vals or not s_vals:
            print(f"{repo:>18} {len(w_vals):>4} {len(s_vals):>4} insufficient data")
            continue
        w_med = np.median(w_vals)
        s_med = np.median(s_vals)
        effect = w_med - s_med
        try:
            u, p = mannwhitneyu(w_vals, s_vals, alternative="two-sided")
        except Exception:
            u, p = -1, 1.0
        direction = "✓W>S" if w_med > s_med else "✗W<S"
        print(f"{repo:>18} {len(w_vals):>4} {len(s_vals):>4} {w_med:>7.3f} {s_med:>7.3f} {effect:>+7.3f} {p:>8.4f} {direction:>9}")

    # Per-repo v1∪v4 operational results
    print("\n=== Per-repo: v1∪v4 detector performance (best op point thresh=0.52, T_frac=0.7) ===")
    print(f"{'repo':>18} {'W_recall':>10} {'S_FP_rate':>11} {'L_caught':>10}")
    THRESH = 0.52
    T_FRAC = 0.7
    for repo in sorted(repos.keys()):
        ts = repos[repo]
        w_total = sum(1 for t in ts if t["sub_class"] == "wandering")
        s_total = sum(1 for t in ts if t["sub_class"] == "success")
        l_total = sum(1 for t in ts if t["sub_class"] == "locked")
        if w_total == 0 or s_total == 0:
            print(f"{repo:>18}  (insufficient)")
            continue
        w_caught = 0
        s_fp = 0
        l_caught = 0
        for t in ts:
            v1 = t["v1"]
            v4 = t["v4"]
            if not v1 or not v4:
                continue
            # v1 detection: probe<0.5 OR (complete>=0.4 AND no_finish AND patch>0)
            probe_neg = t["final_probe"] is not None and t["final_probe"] < 0.5
            text_pos = (v1.get("m_complete_last5", 0) >= 0.4 and not v1.get("emit_finish", False)
                        and v1.get("patch_n_bytes", 0) > 0)
            v1_fire = probe_neg or text_pos
            # v4 detection: late_mean(range[0:T]) > thresh
            ranges = v4.get("ranges_per_turn", [])
            n = v4.get("n_turns", 50)
            T = min(int(T_FRAC * n), len(ranges) - 1)
            v4_fire = False
            if T >= 5:
                half = T // 2
                late = float(np.mean(ranges[half:T+1]))
                v4_fire = (late > THRESH)
            fire = v1_fire or v4_fire
            if t["sub_class"] == "wandering" and fire:
                w_caught += 1
            elif t["sub_class"] == "success" and fire:
                s_fp += 1
            elif t["sub_class"] == "locked" and fire:
                l_caught += 1
        print(f"{repo:>18} {w_caught}/{w_total} ({100*w_caught/w_total:>3.0f}%) {s_fp}/{s_total} ({100*s_fp/s_total:>3.0f}%) {l_caught}/{l_total} ({100*l_caught/l_total:>3.0f}%)")

    # LORO: train probe on N-1 repos, test on heldout
    print("\n=== Pseudo-LORO (using already-CV-fit probe scores from v4) ===")
    print("    Note: this is NOT a strict LORO refit; we use the existing 5-fold CV scores.")
    print("    Per-repo: WANDERING vs SUCCESS range_late on CV-heldout scores.")
    print("    The CV folds were repo-balanced; refitting per-repo would lose power but is")
    print("    more rigorous. We report both per-repo statistics as the practical proxy.")

    # Save
    out = {
        "repo_distribution": {repo: {"n": len(ts),
                                     "success": sum(1 for t in ts if t["sub_class"] == "success"),
                                     "locked": sum(1 for t in ts if t["sub_class"] == "locked"),
                                     "wandering": sum(1 for t in ts if t["sub_class"] == "wandering")}
                              for repo, ts in repos.items()},
        "per_repo_range_late_W_vs_S": {},
        "per_repo_v1_v4_op": {},
    }
    for repo in sorted(repos.keys()):
        ts = repos[repo]
        w_vals = [t["v4"].get("range_late", np.nan) for t in ts if t["sub_class"] == "wandering" and t["v4"]]
        s_vals = [t["v4"].get("range_late", np.nan) for t in ts if t["sub_class"] == "success" and t["v4"]]
        w_vals = [v for v in w_vals if not np.isnan(v)]
        s_vals = [v for v in s_vals if not np.isnan(v)]
        if w_vals and s_vals:
            u, p = mannwhitneyu(w_vals, s_vals, alternative="two-sided")
            out["per_repo_range_late_W_vs_S"][repo] = {
                "n_w": len(w_vals), "n_s": len(s_vals),
                "w_median": float(np.median(w_vals)), "s_median": float(np.median(s_vals)),
                "p": float(p), "direction": "W>S" if np.median(w_vals) > np.median(s_vals) else "W<S",
            }
    (OUT_DIR / "cross_task_validation.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_DIR}/cross_task_validation.json")


if __name__ == "__main__":
    main()
