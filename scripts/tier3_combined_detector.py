"""
Tier 3 combined detector — final test.

Combines all viable signals (v1 text post-hoc, v4 cross-layer disagreement,
v5 tool-entropy) with each tuned to its own threshold, then finds the
union operating point that achieves FP <= 5%.

Operational signal:
  alarm = v1_fire OR v4_fire OR v5_fire

Where each component is calibrated for high precision (low FP individually).
Union increases recall but may exceed 5% FP — need to find combination that
respects FP budget.

Strategy:
  1. Compute each signal at multiple thresholds
  2. Enumerate combinations (v1_thresh, v4_thresh, v5_thresh)
  3. Find combination maximizing WANDERING recall at FP <= 5%
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")

INFLECTION = json.load(open(OUT_DIR / "inflection_results.json"))
V1 = json.load(open(OUT_DIR / "complementary_monitor.json"))
V4 = json.load(open(OUT_DIR / "early_warning_v4_cross_layer.json"))
V5 = json.load(open(OUT_DIR / "v5_tool_entropy.json"))


def main():
    # Build sub-class map
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

    v1_sig = {s["iid"]: s for s in V1["per_trajectory_signals"]}
    v4_sig = {m["iid"]: m for m in V4["per_trajectory"]}
    v5_sig = {m["iid"]: m for m in V5}

    # Build per-trajectory unified signal table
    rows = []
    for iid, sc in sub_class.items():
        v1 = v1_sig.get(iid, {})
        v4 = v4_sig.get(iid, {})
        v5 = v5_sig.get(iid, {})
        if not v1 or not v4 or not v5:
            continue
        # v4 range_late at T_frac=0.7 (best op point from §7)
        ranges = v4.get("ranges_per_turn", [])
        n = v4.get("n_turns", 50)
        T = min(int(0.7 * n), len(ranges) - 1)
        v4_late_mean = float(np.mean(ranges[max(1, T // 2):T+1])) if T >= 5 else 0.0
        rows.append({
            "iid": iid,
            "sub_class": sc,
            "n_turns": n,
            "final_probe": final_probe[iid],
            "v1_m_complete": v1.get("m_complete_last5", 0),
            "v1_emit_finish": v1.get("emit_finish", False),
            "v1_patch": v1.get("patch_n_bytes", 0),
            "v4_late_mean": v4_late_mean,
            "v5_entropy": v5.get("tool_entropy_last10", 0),
            "v5_repetition": v5.get("tool_repetition_last10", 0),
        })

    success_rows = [r for r in rows if r["sub_class"] == "success"]
    wander_rows = [r for r in rows if r["sub_class"] == "wandering"]
    locked_rows = [r for r in rows if r["sub_class"] == "locked"]
    print(f"Loaded {len(rows)} trajectories: success={len(success_rows)}, locked={len(locked_rows)}, wandering={len(wander_rows)}\n")

    # Component detectors (parameterized)
    def v1_fires(r):
        # v1: probe<0.5 OR (complete>=0.4 AND no_finish AND patch>0)
        return (r["final_probe"] < 0.5) or (r["v1_m_complete"] >= 0.4 and not r["v1_emit_finish"] and r["v1_patch"] > 0)

    def v4_fires(r, thresh):
        return r["v4_late_mean"] > thresh

    def v5_fires(r, thresh):
        return r["v5_entropy"] < thresh

    def evaluate(detect_fn, label):
        w_caught = sum(1 for r in wander_rows if detect_fn(r))
        s_fp = sum(1 for r in success_rows if detect_fn(r))
        l_caught = sum(1 for r in locked_rows if detect_fn(r))
        return {
            "label": label,
            "w_recall": w_caught / len(wander_rows),
            "s_fp_rate": s_fp / len(success_rows),
            "w_caught": w_caught, "s_fp": s_fp, "l_caught": l_caught,
        }

    # Baseline detectors at default ops
    print("=== Individual detectors (default operating points) ===")
    res_v1 = evaluate(v1_fires, "v1 only")
    print(f"  v1 only:        W={res_v1['w_caught']}/20 ({100*res_v1['w_recall']:.0f}%), S_FP={res_v1['s_fp']}/40 ({100*res_v1['s_fp_rate']:.0f}%), L={res_v1['l_caught']}")
    res_v5 = evaluate(lambda r: v5_fires(r, 0.503), "v5 entropy<0.503")
    print(f"  v5 entropy<0.50: W={res_v5['w_caught']}/20 ({100*res_v5['w_recall']:.0f}%), S_FP={res_v5['s_fp']}/40 ({100*res_v5['s_fp_rate']:.0f}%), L={res_v5['l_caught']}")
    res_v1_v5 = evaluate(lambda r: v1_fires(r) or v5_fires(r, 0.503), "v1 ∪ v5")
    print(f"  v1 ∪ v5:        W={res_v1_v5['w_caught']}/20 ({100*res_v1_v5['w_recall']:.0f}%), S_FP={res_v1_v5['s_fp']}/40 ({100*res_v1_v5['s_fp_rate']:.0f}%), L={res_v1_v5['l_caught']}")

    # Sweep v4 and v5 thresholds in conjunction with v1
    print("\n=== Tier 3 sweep: v1 OR v4(t4) OR v5(t5), find combo with FP <= 5% ===")
    best = None
    for t5 in [0.4, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for t4 in [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
            def fn(r, t4=t4, t5=t5):
                return v1_fires(r) or v4_fires(r, t4) or v5_fires(r, t5)
            res = evaluate(fn, f"v1∪v4(>{t4:.2f})∪v5(<{t5:.2f})")
            if res["s_fp_rate"] <= 0.05:
                if best is None or res["w_recall"] > best["w_recall"]:
                    best = res
                    best["t4"] = t4
                    best["t5"] = t5

    if best:
        print(f"\n  BEST Tier 3 (v1∪v4∪v5): t4={best['t4']}, t5={best['t5']}")
        print(f"    WANDERING: {best['w_caught']}/20 ({100*best['w_recall']:.0f}%)")
        print(f"    SUCCESS FP: {best['s_fp']}/40 ({100*best['s_fp_rate']:.1f}%)")
        print(f"    LOCKED caught: {best['l_caught']}/39")

    # v1 ∪ v5 alone — simpler combination (v5 already has 5% FP, adding v1 may or may not exceed)
    print("\n=== Simpler v1 ∪ v5 thresholds ===")
    for t5 in [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]:
        res = evaluate(lambda r, t=t5: v1_fires(r) or v5_fires(r, t), f"v1∪v5(<{t5:.2f})")
        marker = " ✓ TIER 3" if res["s_fp_rate"] <= 0.05 else ""
        print(f"  v5<{t5:.2f}: W={res['w_caught']}/20 ({100*res['w_recall']:.0f}%), S_FP={res['s_fp']}/40 ({100*res['s_fp_rate']:.1f}%), L={res['l_caught']}{marker}")

    # Single-signal v5 sweep
    print("\n=== v5 entropy alone sweep (Tier 3 verification) ===")
    for t5 in [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]:
        res = evaluate(lambda r, t=t5: v5_fires(r, t), f"v5(<{t5:.2f})")
        marker = " ✓ TIER 3" if res["s_fp_rate"] <= 0.05 else ""
        print(f"  v5<{t5:.2f}: W={res['w_caught']}/20 ({100*res['w_recall']:.0f}%), S_FP={res['s_fp']}/40 ({100*res['s_fp_rate']:.1f}%), L={res['l_caught']}{marker}")

    # Orthogonality check: which trajectories does each detector catch?
    print("\n=== Orthogonality (Tier 3 best v5<0.503) ===")
    caught_v1 = set(r["iid"] for r in wander_rows if v1_fires(r))
    caught_v5 = set(r["iid"] for r in wander_rows if v5_fires(r, 0.503))
    caught_both = caught_v1 & caught_v5
    only_v1 = caught_v1 - caught_v5
    only_v5 = caught_v5 - caught_v1
    print(f"  WANDERING caught by v1 only: {len(only_v1)}")
    print(f"  WANDERING caught by v5 only: {len(only_v5)}")
    print(f"  WANDERING caught by BOTH:    {len(caught_both)}")
    print(f"  WANDERING caught by EITHER:  {len(caught_v1 | caught_v5)}/20")

    # Save
    (OUT_DIR / "tier3_combined_detector.json").write_text(json.dumps({
        "rows": rows,
        "best_tier3": best,
        "v1_only": res_v1,
        "v5_only": res_v5,
        "v1_union_v5": res_v1_v5,
    }, indent=2, default=float))
    print(f"\nSaved {OUT_DIR}/tier3_combined_detector.json")


if __name__ == "__main__":
    main()
