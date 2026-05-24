"""
Empirical test of complementary monitors for the WANDERING blind spot.

Probe alone misses 34% of failures (WANDERING sub-class — agent over-confident,
probe says success, behavior says fail). Goal: identify a complementary signal
that catches WANDERING while preserving probe's catch of LOCKED failures.

Candidate monitors (all derived from existing trace data, no new compute):

  M_complete: rate of completion-language phrases in last 5 turn outputs
    ("all tests pass", "verified", "code is correct", "summary of changes",
     "infrastructure limitation", "not a code issue", etc.)
  M_static: number of CONSECUTIVE turns at end without a file_edit/str_replace tool call
  M_repeat: rate of repeat tool calls (same tool, similar args) in last 5 turns
  M_combined: any-of (M_complete > 0.4 OR M_static >= 3)

Test: per failure trajectory, compute each monitor signal. Compare WANDERING
vs LOCKED via Mann-Whitney + effect size. If a monitor separates them, it's
the complementary signal.

Then: combined recall = probe_negative ∨ monitor_positive. Compare to probe-only.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")

INFLECTION = json.load(open(OUT_DIR / "inflection_results.json"))

# ----------------------------------------------------------------------------
# Completion-language patterns
# ----------------------------------------------------------------------------
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


def get_text_for_turn(turn_entry: dict) -> str:
    """Pull combined text from a turn (thinking + content + raw_response best-effort)."""
    parts = []
    for k in ("thinking", "content", "raw_response"):
        v = turn_entry.get(k) or ""
        if isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


def m_complete(turns: list, last_n: int = 5) -> float:
    """Rate of completion-language matches per turn in last N."""
    last = turns[-last_n:] if len(turns) >= last_n else turns
    if not last:
        return 0.0
    hits = 0
    for t in last:
        text = get_text_for_turn(t)
        if COMPLETE_RE.search(text):
            hits += 1
    return hits / len(last)


def m_static(turns: list) -> int:
    """Consecutive turns from end without an edit-class tool call."""
    EDIT_TOOLS = {"str_replace_editor", "create_file", "edit_file", "write_file"}
    count = 0
    for t in reversed(turns):
        calls = t.get("tool_calls", []) or []
        edited = False
        for c in calls:
            if isinstance(c, dict):
                # tool_calls structure
                name = c.get("name", "")
                # str_replace_editor is OK; check command field
                if name in EDIT_TOOLS:
                    args = c.get("arguments", {}) or {}
                    cmd = args.get("command", "") if isinstance(args, dict) else ""
                    if cmd not in ("view",):  # 'view' is read-only
                        edited = True
                        break
        if edited:
            break
        count += 1
    return count


def m_repeat(turns: list, last_n: int = 5) -> int:
    """Number of repeat (same-tool, similar-arg) calls in last N turns."""
    last = turns[-last_n:] if len(turns) >= last_n else turns
    signatures = []
    for t in last:
        calls = t.get("tool_calls", []) or []
        for c in calls:
            if isinstance(c, dict):
                name = c.get("name", "")
                args = c.get("arguments", {})
                # signature: name + first 100 chars of arg representation
                sig = f"{name}::{str(args)[:100]}"
                signatures.append(sig)
    if not signatures:
        return 0
    cnt = Counter(signatures)
    return max(cnt.values()) - 1  # number of "extra" occurrences of most-repeated


def emit_finish(turns: list) -> bool:
    """Did the agent emit finish_tool in any turn?"""
    for t in turns:
        for c in t.get("tool_calls", []) or []:
            if isinstance(c, dict) and c.get("name", "").lower().startswith("finish"):
                return True
    return False


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
def run():
    # Map iid -> sub-class from inflection results
    sub_class = {}
    for t in INFLECTION["per_trajectory"]:
        if t["label"] == 1:
            sub_class[t["iid"]] = "success"
        elif t.get("lock_fail_0.40") is not None:
            sub_class[t["iid"]] = "locked"
        else:
            sub_class[t["iid"]] = "wandering"

    # Load full traces + compute monitor signals
    results = json.load(open(PHASE6 / "phase6_results.json"))
    traj_signals = []
    for iid, entry in results.items():
        if iid not in sub_class:
            continue
        trace_path = PHASE6 / "traces" / (iid + ".json")
        if not trace_path.exists():
            continue
        try:
            trace = json.load(open(trace_path))
            turns = trace.get("turns", []) if isinstance(trace, dict) else []
        except Exception:
            continue
        if not turns:
            continue
        sig = {
            "iid": iid,
            "sub_class": sub_class[iid],
            "label": 1 if sub_class[iid] == "success" else 0,
            "patch_n_bytes": entry.get("patch_n_bytes", 0),
            "n_turns_actual": len(turns),
            "m_complete_last5": m_complete(turns, 5),
            "m_complete_last10": m_complete(turns, 10),
            "m_static_end": m_static(turns),
            "m_repeat_last5": m_repeat(turns, 5),
            "emit_finish": emit_finish(turns),
        }
        traj_signals.append(sig)

    print(f"Loaded {len(traj_signals)} trajectories with full traces")
    by_class = defaultdict(list)
    for s in traj_signals:
        by_class[s["sub_class"]].append(s)
    print(f"  success: {len(by_class['success'])}, locked: {len(by_class['locked'])}, wandering: {len(by_class['wandering'])}")

    # Compare WANDERING vs LOCKED on each monitor signal
    print("\n=== Monitor signal comparison (WANDERING vs LOCKED failures) ===")
    wander = by_class["wandering"]
    locked = by_class["locked"]
    success = by_class["success"]

    for metric in ["m_complete_last5", "m_complete_last10", "m_static_end", "m_repeat_last5"]:
        w_vals = [s[metric] for s in wander]
        l_vals = [s[metric] for s in locked]
        s_vals = [s[metric] for s in success]
        try:
            u, p = mannwhitneyu(w_vals, l_vals, alternative="two-sided")
        except Exception:
            u, p = -1, 1.0
        # Cohen's h-ish (median difference / pooled SD)
        med_w = np.median(w_vals); med_l = np.median(l_vals); med_s = np.median(s_vals)
        mean_w = np.mean(w_vals); mean_l = np.mean(l_vals); mean_s = np.mean(s_vals)
        print(f"\n  {metric}")
        print(f"    wandering (n={len(wander)}): mean={mean_w:.3f}, median={med_w:.3f}")
        print(f"    locked    (n={len(locked)}): mean={mean_l:.3f}, median={med_l:.3f}")
        print(f"    success   (n={len(success)}): mean={mean_s:.3f}, median={med_s:.3f}")
        print(f"    Mann-Whitney W vs L: U={u}, p={p:.4f}")

    # Combined detector evaluation
    print("\n=== Detector comparison ===")
    # Probe-alone classifier: predict FAIL if final probe < 0.5
    # Complementary: predict FAIL if (final probe < 0.5) OR (m_complete_last5 >= 0.4 AND not emit_finish)

    # Get final probe scores from inflection results
    iid_to_probe_final = {}
    for t in INFLECTION["per_trajectory"]:
        iid_to_probe_final[t["iid"]] = t.get("final_score", 0.5)

    n_success = sum(1 for s in traj_signals if s["sub_class"] == "success")
    n_fail = sum(1 for s in traj_signals if s["sub_class"] != "success")

    def evaluate(detect_fn, label):
        TP = FP = TN = FN = 0
        for s in traj_signals:
            is_fail_pred = detect_fn(s)
            is_fail_true = s["sub_class"] != "success"
            if is_fail_true:
                if is_fail_pred: TP += 1
                else: FN += 1
            else:
                if is_fail_pred: FP += 1
                else: TN += 1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        # Recall on WANDERING specifically
        wander_caught = sum(1 for s in traj_signals if s["sub_class"] == "wandering" and detect_fn(s))
        locked_caught = sum(1 for s in traj_signals if s["sub_class"] == "locked" and detect_fn(s))
        false_pos_on_success = sum(1 for s in traj_signals if s["sub_class"] == "success" and detect_fn(s))
        print(f"\n  {label}:")
        print(f"    Overall: TP={TP}, FP={FP}, TN={TN}, FN={FN}")
        print(f"    Precision={precision:.3f}, Recall={recall:.3f}")
        print(f"    On WANDERING: caught {wander_caught}/{len(by_class['wandering'])} ({100*wander_caught/len(by_class['wandering']):.0f}%)")
        print(f"    On LOCKED:    caught {locked_caught}/{len(by_class['locked'])} ({100*locked_caught/len(by_class['locked']):.0f}%)")
        print(f"    False-positive on SUCCESS: {false_pos_on_success}/{len(by_class['success'])}")
        return {"TP": TP, "FP": FP, "TN": TN, "FN": FN, "precision": precision, "recall": recall,
                "wander_caught": wander_caught, "locked_caught": locked_caught, "fp_on_success": false_pos_on_success}

    def probe_only(s):
        return iid_to_probe_final.get(s["iid"], 0.5) < 0.5

    def complementary(s):
        # Add: completion language + no finish + patch made
        probe_fail = iid_to_probe_final.get(s["iid"], 0.5) < 0.5
        wander_signal = (s["m_complete_last5"] >= 0.4 and not s["emit_finish"] and s["patch_n_bytes"] > 0)
        return probe_fail or wander_signal

    def comp_strict(s):
        # Just: completion-language signal alone (no probe)
        return (s["m_complete_last5"] >= 0.4 and not s["emit_finish"] and s["patch_n_bytes"] > 0)

    def comp_any_complete(s):
        # Any completion-language hit in last 5
        return (s["m_complete_last5"] >= 0.2 and not s["emit_finish"])

    results_dict = {}
    results_dict["probe_only"] = evaluate(probe_only, "Probe-only (final_score < 0.5)")
    results_dict["complementary_AND"] = evaluate(complementary, "Probe OR (complete>=0.4 AND no_finish AND patch>0)")
    results_dict["just_completion"] = evaluate(comp_strict, "Just-completion (no probe)")
    results_dict["any_complete"] = evaluate(comp_any_complete, "Any complete>=0.2 AND no_finish")

    # Save
    out_path = OUT_DIR / "complementary_monitor.json"
    out_path.write_text(json.dumps({
        "n_trajectories": len(traj_signals),
        "n_by_class": {k: len(v) for k, v in by_class.items()},
        "detector_results": results_dict,
        "per_trajectory_signals": traj_signals,
    }, indent=2))
    print(f"\n=== Saved {out_path} ===")


if __name__ == "__main__":
    run()
