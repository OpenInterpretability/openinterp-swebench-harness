"""
v4 operational sweep — fixed.

The v4 stats show real signal: range_late, std_late, sign_dis_late all
p<0.005 WANDERING vs SUCCESS. But the original sweep used "earliest turn
above threshold" which fires on every trajectory (early-turn probes are
noisy for all classes).

Correct operational design: at decision turn T, compute the LATE-HALF MEAN
of disagreement up to T. Alarm if that exceeds threshold AND no_finish.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")
V4 = json.load(open(OUT_DIR / "early_warning_v4_cross_layer.json"))


def emit_finish_at_from_trace(trace_path):
    try:
        trace = json.load(open(trace_path))
    except Exception:
        return None
    for i, t in enumerate(trace.get("turns", [])):
        for c in t.get("tool_calls", []) or []:
            if isinstance(c, dict) and c.get("name", "").lower().startswith("finish"):
                return i
    return None


def main():
    # Enrich with finish_emit_turn
    finish_map = {}
    for m in V4["per_trajectory"]:
        finish_map[m["iid"]] = emit_finish_at_from_trace(PHASE6 / "traces" / (m["iid"] + ".json"))

    traj = V4["per_trajectory"]
    by_class = defaultdict(list)
    for m in traj:
        by_class[m["sub_class"]].append(m)

    print(f"Loaded {len(traj)} (success={len(by_class['success'])}, locked={len(by_class['locked'])}, wandering={len(by_class['wandering'])})\n")

    # Operational sweep — running late-half mean disagreement
    print("=== Operational sweep: late-half mean(range[0:T]) > threshold AND no_finish ===")
    print(f"{'thresh':>7} {'T_frac':>8} | {'W_rec':>9} {'lead':>5} | {'S_FP':>10} | {'L_caught':>8}")
    print("-" * 70)

    sweep = []
    best_op = None
    best_score = -1
    for thresh in [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]:
        for T_frac in [0.50, 0.60, 0.70, 0.80, 0.90]:
            w_caught = 0
            w_leads = []
            s_fp = 0
            l_caught = 0
            for m in traj:
                T = int(T_frac * m["n_turns"])
                ranges = m["ranges_per_turn"]
                if T >= len(ranges):
                    T = len(ranges) - 1
                if T < 5:
                    continue
                # Late-half mean up to T
                half = T // 2
                late_mean = float(np.mean(ranges[half:T+1]))
                if late_mean <= thresh:
                    continue
                fire_turn = T
                # Skip if finish_tool was emitted before alarm
                ft = finish_map.get(m["iid"])
                if ft is not None and ft <= fire_turn:
                    continue
                if m["sub_class"] == "wandering":
                    w_caught += 1
                    w_leads.append(m["n_turns"] - fire_turn)
                elif m["sub_class"] == "success":
                    if ft is not None and ft > fire_turn:
                        s_fp += 1
                elif m["sub_class"] == "locked":
                    l_caught += 1
            wcr = w_caught / len(by_class["wandering"])
            sfpr = s_fp / len(by_class["success"])
            score = wcr - 2 * sfpr
            lm = f"{np.median(w_leads):.0f}" if w_leads else "  -"
            row = {"thresh": thresh, "T_frac": T_frac, "w_caught": w_caught, "w_recall": wcr,
                   "lead_median": float(np.median(w_leads)) if w_leads else None,
                   "s_fp": s_fp, "s_fp_rate": sfpr, "l_caught": l_caught,
                   "score": score}
            sweep.append(row)
            if score > best_score:
                best_score = score
                best_op = row
            print(f"{thresh:>7.2f} {T_frac:>8.2f} | {100*wcr:>5.0f}% {w_caught:>2}/{len(by_class['wandering']):<2} {lm:>5} | {s_fp:>2}/{len(by_class['success']):<2}={100*sfpr:>5.1f}% | {l_caught:>3}/{len(by_class['locked'])}")

    print(f"\n=== Best operating point (score = recall - 2×FP): thresh={best_op['thresh']}, T_frac={best_op['T_frac']} ===")
    print(f"    WANDERING caught: {best_op['w_caught']}/20 ({100*best_op['w_recall']:.0f}%)")
    print(f"    SUCCESS FP:        {best_op['s_fp']}/40 ({100*best_op['s_fp_rate']:.1f}%)")
    print(f"    Lead time median:  {best_op['lead_median']}")
    print(f"    LOCKED caught:     {best_op['l_caught']}/39 (bonus)")

    # COMBINED with text-based (§5 v1) — do they catch DIFFERENT trajectories?
    # Pull v1 results
    v1 = json.load(open(OUT_DIR / "complementary_monitor.json"))
    v1_signals = {s["iid"]: s for s in v1["per_trajectory_signals"]}

    print("\n=== Combined v1-text + v4-cross-layer detector ===")
    # v1 detector at trajectory END: probe<0.5 OR (complete>=0.4 AND no_finish AND patch>0)
    # v4 detector at decision turn T: late-half mean range > thresh AND no_finish
    # Combined: v1 OR v4
    iid_to_probe = {}
    for t in V4["per_trajectory"]:
        # Use L43 score as the "primary probe" — last turn's value
        # We computed per-turn-per-layer scores in v4 but didn't save them all
        # Just use sub_class for binary success/fail prediction here
        pass

    iid_to_v1 = {}
    for iid, sig in v1_signals.items():
        # v1 catches FAIL if probe negative (final<0.5) OR (complete>=0.4 AND no_finish AND patch>0)
        # We have probe_score in inflection_results.json — load it
        pass
    infl = json.load(open(OUT_DIR / "inflection_results.json"))
    iid_to_probe_final = {t["iid"]: t.get("final_score", 0.5) for t in infl["per_trajectory"]}

    # Compute at best v4 op point + v1 union
    best_t = best_op["T_frac"]
    best_th = best_op["thresh"]
    n_w_caught_v1_only = 0
    n_w_caught_v4_only = 0
    n_w_caught_both = 0
    n_w_caught_either = 0
    n_w_total = 0
    s_fp_v1 = 0
    s_fp_v4 = 0
    s_fp_either = 0
    for m in traj:
        if m["sub_class"] == "wandering":
            n_w_total += 1
            # v1 catches?
            sig = v1_signals.get(m["iid"])
            v1_caught = False
            if sig:
                probe_neg = iid_to_probe_final.get(m["iid"], 0.5) < 0.5
                txt_signal = (sig["m_complete_last5"] >= 0.4 and not sig["emit_finish"] and sig["patch_n_bytes"] > 0)
                v1_caught = probe_neg or txt_signal
            # v4 catches?
            T = int(best_t * m["n_turns"])
            T = min(T, len(m["ranges_per_turn"]) - 1)
            half = T // 2
            v4_caught = (np.mean(m["ranges_per_turn"][half:T+1]) > best_th)
            ft = finish_map.get(m["iid"])
            if ft is not None and ft <= T:
                v4_caught = False
            if v1_caught and v4_caught: n_w_caught_both += 1
            elif v1_caught: n_w_caught_v1_only += 1
            elif v4_caught: n_w_caught_v4_only += 1
            if v1_caught or v4_caught: n_w_caught_either += 1
        elif m["sub_class"] == "success":
            sig = v1_signals.get(m["iid"])
            v1_fp = False
            if sig:
                probe_neg = iid_to_probe_final.get(m["iid"], 0.5) < 0.5
                txt_signal = (sig["m_complete_last5"] >= 0.4 and not sig["emit_finish"] and sig["patch_n_bytes"] > 0)
                v1_fp = probe_neg or txt_signal
            T = int(best_t * m["n_turns"])
            T = min(T, len(m["ranges_per_turn"]) - 1)
            half = T // 2
            v4_fp = (np.mean(m["ranges_per_turn"][half:T+1]) > best_th)
            ft = finish_map.get(m["iid"])
            if ft is not None and ft <= T:
                v4_fp = False
            if v1_fp: s_fp_v1 += 1
            if v4_fp: s_fp_v4 += 1
            if v1_fp or v4_fp: s_fp_either += 1
    print(f"  WANDERING: caught_both={n_w_caught_both}, v1_only={n_w_caught_v1_only}, v4_only={n_w_caught_v4_only}, either={n_w_caught_either}/{n_w_total}")
    print(f"  SUCCESS FP: v1={s_fp_v1}/40, v4={s_fp_v4}/40, either={s_fp_either}/40")
    if n_w_caught_v4_only > 0:
        print(f"  v4 ADDS {n_w_caught_v4_only} new WANDERING captures beyond v1 — signals are COMPLEMENTARY")
    else:
        print(f"  v4 only catches subset of v1 captures — signals are REDUNDANT")

    out_path = OUT_DIR / "early_warning_v4_op_sweep.json"
    out_path.write_text(json.dumps({
        "sweep": sweep,
        "best_op": best_op,
        "combined_v1_v4": {
            "wander_caught_both": n_w_caught_both,
            "wander_v1_only": n_w_caught_v1_only,
            "wander_v4_only": n_w_caught_v4_only,
            "wander_either": n_w_caught_either,
            "wander_total": n_w_total,
            "success_fp_v1": s_fp_v1,
            "success_fp_v4": s_fp_v4,
            "success_fp_either": s_fp_either,
        },
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
