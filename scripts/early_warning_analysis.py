"""
Early-warning analysis: can the completion-language monitor fire IN TIME
to prevent budget exhaustion?

Three questions:

1. WHEN does completion-language first appear in WANDERING vs SUCCESS?
2. What's the lead time (turns between first-alert and budget-end or finish_tool)?
3. What's the precision/recall curve as we vary decision-turn T?

Operational signal under test:
    alert(T, K) = ∃ turn t in [T-K+1, T] with completion-language phrase
                  AND finish_tool not yet emitted by T

Fires at turn T, alarm = "this trajectory is likely WANDERING — terminate or escalate".

Outputs:
    - per-trajectory per-turn completion-language indicator (boolean)
    - first_complete_turn per trajectory
    - finish_emit_turn per success trajectory
    - lead_time distributions
    - precision/recall sweep over (T, K) grid
    - trajectory plot of completion signal per sub-class
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


def text_of(turn: dict) -> str:
    parts = []
    for k in ("thinking", "content", "raw_response"):
        v = turn.get(k) or ""
        if isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


def has_complete(turn: dict) -> bool:
    return bool(COMPLETE_RE.search(text_of(turn)))


def emit_finish_at(turns: list) -> int | None:
    for i, t in enumerate(turns):
        for c in t.get("tool_calls", []) or []:
            if isinstance(c, dict) and c.get("name", "").lower().startswith("finish"):
                return i
    return None


def first_complete_turn(complete_bools: list) -> int | None:
    for i, b in enumerate(complete_bools):
        if b:
            return i
    return None


def sliding_alert(complete_bools: list, K: int) -> list:
    """For each turn t, did any of [max(0, t-K+1)..t] have completion-language?"""
    out = []
    for t in range(len(complete_bools)):
        lo = max(0, t - K + 1)
        out.append(any(complete_bools[lo:t+1]))
    return out


def run():
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
        cbools = [has_complete(t) for t in turns]
        finish_t = emit_finish_at(turns)
        fct = first_complete_turn(cbools)
        traj.append({
            "iid": iid,
            "sub_class": sub_class[iid],
            "n_turns": len(turns),
            "complete_bools": cbools,
            "first_complete_turn": fct,
            "first_complete_frac": fct / len(turns) if fct is not None else None,
            "finish_emit_turn": finish_t,
            "n_complete_total": sum(cbools),
            "complete_rate_overall": sum(cbools) / len(turns),
        })

    print(f"Loaded {len(traj)} trajectories\n")
    by_class = defaultdict(list)
    for t in traj:
        by_class[t["sub_class"]].append(t)
    for cls, ts in by_class.items():
        print(f"  {cls}: n={len(ts)}")

    # ----------------------------------------------------------------
    # Q1: WHEN does completion-language first appear?
    # ----------------------------------------------------------------
    print("\n=== Q1: First-completion-turn distribution per sub-class ===")
    for cls in ["success", "locked", "wandering"]:
        ts = by_class[cls]
        firsts = [t["first_complete_turn"] for t in ts if t["first_complete_turn"] is not None]
        n_never = sum(1 for t in ts if t["first_complete_turn"] is None)
        if firsts:
            q25, q50, q75 = np.percentile(firsts, [25, 50, 75])
            print(f"  {cls}: n_with_completion={len(firsts)}/{len(ts)}, n_never={n_never}")
            print(f"    first_complete_turn: q25={q25:.0f}, q50={q50:.0f}, q75={q75:.0f}, min={min(firsts)}, max={max(firsts)}")
            # As fraction of length
            fracs = [t["first_complete_frac"] for t in ts if t["first_complete_frac"] is not None]
            fq25, fq50, fq75 = np.percentile(fracs, [25, 50, 75])
            print(f"    first_complete_frac: q25={fq25:.2f}, q50={fq50:.2f}, q75={fq75:.2f}")

    # ----------------------------------------------------------------
    # Q2: Lead time
    #   For WANDERING-A (first_complete present): lead = n_turns - first_complete_turn
    #   For SUCCESS that emit finish: gap = finish_emit_turn - first_complete_turn
    # ----------------------------------------------------------------
    print("\n=== Q2: Lead-time distribution ===")
    # WANDERING-A lead time (turns between first completion-language and budget end)
    w_leads = [t["n_turns"] - t["first_complete_turn"] for t in by_class["wandering"]
               if t["first_complete_turn"] is not None]
    print(f"\n  WANDERING-A (n={len(w_leads)}): turns from first-complete to budget end")
    if w_leads:
        print(f"    median={np.median(w_leads):.0f}, q25={np.percentile(w_leads, 25):.0f}, q75={np.percentile(w_leads, 75):.0f}")
        print(f"    min={min(w_leads)}, max={max(w_leads)}, mean={np.mean(w_leads):.1f}")

    # SUCCESS gap between first completion-language and finish_tool emission
    s_gaps = []
    for t in by_class["success"]:
        if t["first_complete_turn"] is not None and t["finish_emit_turn"] is not None:
            gap = t["finish_emit_turn"] - t["first_complete_turn"]
            if gap >= 0:
                s_gaps.append(gap)
    print(f"\n  SUCCESS (n={len(s_gaps)}): turns between first-complete and finish_tool emission")
    if s_gaps:
        print(f"    median={np.median(s_gaps):.0f}, q25={np.percentile(s_gaps, 25):.0f}, q75={np.percentile(s_gaps, 75):.0f}")
        print(f"    min={min(s_gaps)}, max={max(s_gaps)}, mean={np.mean(s_gaps):.1f}")
        print(f"    fraction of success with gap >= 5: {sum(1 for g in s_gaps if g >= 5) / len(s_gaps):.2%}")
        print(f"    fraction of success with gap >= 10: {sum(1 for g in s_gaps if g >= 10) / len(s_gaps):.2%}")

    # ----------------------------------------------------------------
    # Q3: Operational sweep — alert(T, K) signal
    #   alert fires at turn T if any [T-K+1..T] has completion AND no finish by T
    #   Evaluate as a PREDICTOR of WANDERING vs SUCCESS-that-finishes-later
    # ----------------------------------------------------------------
    print("\n=== Q3: Operational alert(T, K) sweep ===")
    print("    Alert fires at turn T iff completion-language in [T-K+1..T] AND no finish_tool by turn T")
    print("    Eval per (T, K):")
    print("      WANDERING_caught = fraction of WANDERING trajectories with alert firing by turn T")
    print("      SUCCESS_FP       = fraction of SUCCESS trajectories with alert firing by turn T")
    print("                        (these would emit finish_tool LATER — operationally a false alarm)")
    print("      lead_time_med    = median turns between alarm and budget-end (for caught WANDERING)")

    sweep_results = []
    for K in [3, 5, 7, 10]:
        for T_frac in [0.50, 0.60, 0.70, 0.80, 0.90]:
            # Compute alert-fire-by-T per trajectory
            wander_alarms = []
            success_alarms = []
            wander_lead = []
            success_alarm_lead = []
            for t in traj:
                T = int(T_frac * t["n_turns"])
                if T < K:
                    continue
                alerts_sliding = sliding_alert(t["complete_bools"][:T+1], K)
                fired_idx = None
                for tt in range(T + 1):
                    if alerts_sliding[tt]:
                        if t["finish_emit_turn"] is None or t["finish_emit_turn"] > tt:
                            fired_idx = tt
                            break
                if t["sub_class"] == "wandering":
                    wander_alarms.append(fired_idx is not None)
                    if fired_idx is not None:
                        wander_lead.append(t["n_turns"] - fired_idx)
                elif t["sub_class"] == "success":
                    # Alarm = false positive ONLY if finish was emitted after the alarm
                    is_fp = (fired_idx is not None and (t["finish_emit_turn"] is None or t["finish_emit_turn"] > fired_idx))
                    success_alarms.append(is_fp)
                    if is_fp and t["finish_emit_turn"] is not None:
                        # Op cost: turns "wasted" between FP alarm and actual successful finish
                        success_alarm_lead.append(t["finish_emit_turn"] - fired_idx)
            wcr = sum(wander_alarms) / len(wander_alarms) if wander_alarms else 0
            sfpr = sum(success_alarms) / len(success_alarms) if success_alarms else 0
            wlm = np.median(wander_lead) if wander_lead else None
            sweep_results.append({
                "K": K, "T_frac": T_frac,
                "wander_recall": wcr, "success_fp": sfpr,
                "wander_caught": sum(wander_alarms), "wander_total": len(wander_alarms),
                "success_fp_n": sum(success_alarms), "success_total": len(success_alarms),
                "wander_lead_median": wlm,
            })
            print(f"\n  K={K}, T_frac={T_frac:.2f}: WANDERING caught {sum(wander_alarms)}/{len(wander_alarms)} ({100*wcr:.0f}%), "
                  f"SUCCESS FP {sum(success_alarms)}/{len(success_alarms)} ({100*sfpr:.1f}%), "
                  f"WANDERING lead median = {wlm}")

    # ----------------------------------------------------------------
    # Visualize: per-turn completion rate by sub-class (normalized turn fraction)
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"success": "tab:green", "locked": "tab:orange", "wandering": "tab:red"}
    for cls in ["success", "locked", "wandering"]:
        ts = by_class[cls]
        n_bins = 20
        bin_hits = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        for t in ts:
            for turn_idx, b in enumerate(t["complete_bools"]):
                bin_idx = min(int(n_bins * turn_idx / t["n_turns"]), n_bins - 1)
                bin_hits[bin_idx] += int(b)
                bin_counts[bin_idx] += 1
        rates = bin_hits / np.maximum(bin_counts, 1)
        x = (np.arange(n_bins) + 0.5) / n_bins
        ax.plot(x, rates, marker="o", label=f"{cls} (n={len(ts)})", color=colors[cls], lw=2)
    ax.set_xlabel("Fraction of trajectory length")
    ax.set_ylabel("Per-turn rate of completion-language phrase")
    ax.set_title("Completion-language verbalization across trajectory length, by sub-class")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_path = OUT_DIR / "early_warning_per_turn_rate.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"\n=== Saved figure to {fig_path} ===")

    # Save raw
    out_path = OUT_DIR / "early_warning_results.json"
    out_path.write_text(json.dumps({
        "n_trajectories": len(traj),
        "n_by_class": {k: len(v) for k, v in by_class.items()},
        "first_complete_turn_per_class": {
            cls: {
                "n_with_completion": sum(1 for t in by_class[cls] if t["first_complete_turn"] is not None),
                "n_total": len(by_class[cls]),
                "median_turn": float(np.median([t["first_complete_turn"] for t in by_class[cls] if t["first_complete_turn"] is not None])) if any(t["first_complete_turn"] is not None for t in by_class[cls]) else None,
                "median_frac": float(np.median([t["first_complete_frac"] for t in by_class[cls] if t["first_complete_frac"] is not None])) if any(t["first_complete_frac"] is not None for t in by_class[cls]) else None,
            }
            for cls in ["success", "locked", "wandering"]
        },
        "wandering_lead_time_median": float(np.median(w_leads)) if w_leads else None,
        "wandering_lead_time_q25_q75": [float(np.percentile(w_leads, 25)), float(np.percentile(w_leads, 75))] if w_leads else None,
        "success_finish_gap_median": float(np.median(s_gaps)) if s_gaps else None,
        "operational_sweep": sweep_results,
    }, indent=2))
    print(f"=== Saved {out_path} ===")


if __name__ == "__main__":
    run()
