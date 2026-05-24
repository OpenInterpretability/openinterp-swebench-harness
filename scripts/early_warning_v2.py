"""
Early-warning v2: with the WAIT clause + budget-aware gate.

v1 design was naive: alarm fires on ANY completion-language + no finish-yet.
This false-alarms on ~25% of SUCCESS trajectories because successes verbalize
"all tests pass" ~11 turns (median) before emitting finish_tool.

v2 fixes:
  alert(T, W, B) =
      ∃ t* ≤ T - W with completion-language at t*  (verbalization aged ≥ W turns)
      AND no finish_tool by turn T
      AND budget-fraction T/n_turns ≥ B          (only worry when budget near)
      AND patch made

W is the WAIT window: how long after first completion-language we wait before
treating "still no finish" as a sign of WANDERING.

B is the budget-fraction gate: only fire when we're far enough into the budget
that termination is justified. Below B, we let the agent keep working.

Operational interpretation: at turn T, fire alarm only if agent said "done"
at least W turns ago AND still hasn't acted on it AND we're at risk of budget
exhaustion (>= B of budget used).
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
INFLECTION = json.load(open(OUT_DIR / "inflection_results.json"))

# Same regex as v1
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


def first_complete_turn(cbools):
    for i, b in enumerate(cbools):
        if b:
            return i
    return None


def alert_v2(t, W, B):
    """At what turn (if any) does v2 alert first fire on trajectory t?
       Returns (fire_turn, None) or (None, reason)."""
    cbools = t["complete_bools"]
    n = t["n_turns"]
    finish_t = t["finish_emit_turn"]
    # Earliest completion-language turn
    first_comp = first_complete_turn(cbools)
    if first_comp is None:
        return None, "no_completion_ever"
    # If finish was emitted before W elapses after completion, no alarm
    # Alarm earliest turn = first_comp + W
    alarm_turn = first_comp + W
    if alarm_turn >= n:
        return None, "alarm_turn_past_budget"
    # Budget gate: alarm_turn must be ≥ B * n
    if alarm_turn / n < B:
        # Wait until budget gate
        alarm_turn = max(alarm_turn, int(B * n))
        if alarm_turn >= n:
            return None, "budget_gate_past_budget"
    # Check that finish hasn't been emitted by alarm_turn
    if finish_t is not None and finish_t <= alarm_turn:
        return None, "finish_before_alarm"
    # Need patch made by alarm_turn — approximate via patch_n_bytes at end
    # (Could be more sophisticated but Phase 6 trace doesn't store per-turn patch size)
    return alarm_turn, None


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
        finish_t = emit_finish_at(turns)
        traj.append({
            "iid": iid,
            "sub_class": sub_class[iid],
            "n_turns": len(turns),
            "complete_bools": cbools,
            "first_complete_turn": first_complete_turn(cbools),
            "finish_emit_turn": finish_t,
            "patch_n_bytes": entry.get("patch_n_bytes", 0),
        })

    by_class = defaultdict(list)
    for t in traj:
        by_class[t["sub_class"]].append(t)

    print(f"Loaded {len(traj)} (success={len(by_class['success'])}, locked={len(by_class['locked'])}, wandering={len(by_class['wandering'])})\n")

    # Sweep (W, B) — patch_made gate
    print("=== alert_v2 sweep: (W=wait turns, B=budget-fraction gate) ===")
    print(f"{'W':>3} {'B':>5} | {'W_caught':>10} {'W_recall':>9} {'lead_med':>9} | {'S_FP':>5} {'S_FP_rate':>10} | {'L_caught':>8}")
    print("-" * 90)
    sweep = []
    best_op_point = None
    best_op_score = -1
    for W in [5, 10, 15, 20, 25]:
        for B in [0.0, 0.5, 0.7, 0.8, 0.9]:
            w_caught = 0
            w_leads = []
            s_fp = 0
            l_caught = 0
            for t in traj:
                fire_turn, _ = alert_v2(t, W, B)
                if fire_turn is None:
                    continue
                # Skip if no patch (low-confidence WANDERING signal)
                if t["patch_n_bytes"] == 0 and t["sub_class"] != "locked":
                    continue
                if t["sub_class"] == "wandering":
                    w_caught += 1
                    w_leads.append(t["n_turns"] - fire_turn)
                elif t["sub_class"] == "success":
                    # FP only if alarm fired AND finish was actually emitted after the alarm
                    if t["finish_emit_turn"] is not None and t["finish_emit_turn"] > fire_turn:
                        s_fp += 1
                elif t["sub_class"] == "locked":
                    l_caught += 1
            wcr = w_caught / len(by_class["wandering"])
            sfpr = s_fp / len(by_class["success"])
            score = wcr - 2 * sfpr  # weight FP twice (FP=force-terminate a success = BAD)
            row = {
                "W": W, "B": B,
                "wander_caught": w_caught, "wander_recall": wcr,
                "lead_median": float(np.median(w_leads)) if w_leads else None,
                "success_fp": s_fp, "success_fp_rate": sfpr,
                "locked_caught": l_caught,
                "score": score,
            }
            sweep.append(row)
            if score > best_op_score:
                best_op_score = score
                best_op_point = row
            lm = f"{float(np.median(w_leads)):.0f}" if w_leads else "  -"
            print(f"{W:>3} {B:>5.2f} | {w_caught:>4}/{len(by_class['wandering'])} {100*wcr:>7.0f}% {lm:>9} | {s_fp:>2}/{len(by_class['success']):>2} {100*sfpr:>8.1f}% | {l_caught:>3}/{len(by_class['locked'])}")

    print(f"\n=== Best operating point (score = recall - 2×FP): W={best_op_point['W']}, B={best_op_point['B']} ===")
    print(f"    WANDERING caught: {best_op_point['wander_caught']}/{len(by_class['wandering'])} ({100*best_op_point['wander_recall']:.0f}%)")
    print(f"    SUCCESS FP:        {best_op_point['success_fp']}/{len(by_class['success'])} ({100*best_op_point['success_fp_rate']:.1f}%)")
    print(f"    Lead-time median:  {best_op_point['lead_median']} turns before budget end")
    print(f"    LOCKED caught:     {best_op_point['locked_caught']}/{len(by_class['locked'])} (bonus, since probe catches these)")

    # Compare to v1 post-hoc detector
    print("\n=== Comparison vs §5 post-hoc detector ===")
    print(f"  v1 post-hoc (END-of-trajectory):  WANDERING 7/20 (35%), SUCCESS FP 0/40, lead=0")
    print(f"  v2 best ({best_op_point['W']},{best_op_point['B']}): WANDERING {best_op_point['wander_caught']}/20 ({100*best_op_point['wander_recall']:.0f}%), SUCCESS FP {best_op_point['success_fp']}/40 ({100*best_op_point['success_fp_rate']:.1f}%), lead={best_op_point['lead_median']}")

    # Save
    out_path = OUT_DIR / "early_warning_v2_results.json"
    out_path.write_text(json.dumps({
        "sweep": sweep,
        "best_op_point": best_op_point,
    }, indent=2))
    print(f"\n=== Saved {out_path} ===")

    # Plot precision-recall-style curve: lead-time vs recall vs FP rate
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Left: ROC-ish at B=0.8 across W
    points = [r for r in sweep if r["B"] == 0.8]
    fps = [r["success_fp_rate"] for r in points]
    rcs = [r["wander_recall"] for r in points]
    Ws = [r["W"] for r in points]
    axes[0].plot(fps, rcs, "o-", color="tab:blue")
    for fp, rc, W in zip(fps, rcs, Ws):
        axes[0].annotate(f"W={W}", (fp, rc), xytext=(5, 5), textcoords="offset points")
    axes[0].set_xlabel("SUCCESS false-positive rate")
    axes[0].set_ylabel("WANDERING recall")
    axes[0].set_title("Recall vs FP curve (B=0.8 budget gate, varying wait W)")
    axes[0].grid(True, alpha=0.3)

    # Right: lead-time vs recall
    leads = [r["lead_median"] for r in points if r["lead_median"] is not None]
    rcs2 = [r["wander_recall"] for r in points if r["lead_median"] is not None]
    Ws2 = [r["W"] for r in points if r["lead_median"] is not None]
    axes[1].plot(leads, rcs2, "o-", color="tab:red")
    for ld, rc, W in zip(leads, rcs2, Ws2):
        axes[1].annotate(f"W={W}", (ld, rc), xytext=(5, 5), textcoords="offset points")
    axes[1].set_xlabel("Median lead-time (turns before budget end)")
    axes[1].set_ylabel("WANDERING recall")
    axes[1].set_title("Lead-time vs recall (B=0.8)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "early_warning_v2_tradeoff.png", dpi=120, bbox_inches="tight")
    print(f"=== Saved tradeoff plot to {OUT_DIR}/early_warning_v2_tradeoff.png ===")


if __name__ == "__main__":
    main()
