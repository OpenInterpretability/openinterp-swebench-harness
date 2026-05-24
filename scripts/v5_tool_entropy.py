"""
v5 — Tool-use entropy / diversity / repetition as Tier 3 signal candidate.

Hypothesis: WANDERING agents have characteristic tool-use patterns distinct
from SUCCESS. Could be either direction:
  (a) Low diversity (collapsed onto verify-loop: test, view, test, view, ...)
  (b) High diversity (scattered exploration, never converges)

We test 4 metrics per trajectory:
  - tool_entropy_last10: Shannon entropy of tool distribution in last 10 turns
  - tool_diversity_last10: unique tools / 10
  - tool_repetition_last10: most-common-tool fraction in last 10
  - bigram_repeat_rate: fraction of 2-gram tool pairs that repeat in last 20 turns

Compare WANDERING vs SUCCESS vs LOCKED. Test Mann-Whitney + report which
metric (if any) provides Tier 3 precision (FP <= 5%).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")
INFLECTION = json.load(open(OUT_DIR / "inflection_results.json"))


def get_tool_calls_per_turn(turns):
    """Returns list of (turn_idx, list of tool_names)."""
    out = []
    for i, t in enumerate(turns):
        names = []
        for c in t.get("tool_calls", []) or []:
            if isinstance(c, dict):
                names.append(c.get("name", "unknown"))
        out.append(names)
    return out


def shannon_entropy(items):
    if not items:
        return 0.0
    cnt = Counter(items)
    total = sum(cnt.values())
    H = 0.0
    for c in cnt.values():
        p = c / total
        if p > 0:
            H -= p * np.log2(p)
    return H


def emit_finish_at(turns):
    for i, t in enumerate(turns):
        for c in t.get("tool_calls", []) or []:
            if isinstance(c, dict) and c.get("name", "").lower().startswith("finish"):
                return i
    return None


def compute_metrics(turns, last_n=10):
    tc = get_tool_calls_per_turn(turns)
    # Last N turn tools
    last = tc[-last_n:] if len(tc) >= last_n else tc
    # Flatten
    all_tools = [n for sub in last for n in sub]
    entropy = shannon_entropy(all_tools)
    diversity = len(set(all_tools)) / max(1, len(last))
    if all_tools:
        cnt = Counter(all_tools)
        repetition = max(cnt.values()) / len(all_tools)
    else:
        repetition = 0.0
    # Bigram repeat rate over last 2*N
    last2 = tc[-2*last_n:] if len(tc) >= 2*last_n else tc
    bigrams = []
    prev = None
    for sub in last2:
        for n in sub:
            if prev is not None:
                bigrams.append((prev, n))
            prev = n
    if bigrams:
        bg_cnt = Counter(bigrams)
        bg_rep = sum(c for c in bg_cnt.values() if c > 1) / len(bigrams)
    else:
        bg_rep = 0.0
    return {
        "tool_entropy_last10": entropy,
        "tool_diversity_last10": diversity,
        "tool_repetition_last10": repetition,
        "bigram_repeat_rate": bg_rep,
        "n_tools_last10": len(all_tools),
    }


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
    for iid in sub_class:
        trace_path = PHASE6 / "traces" / (iid + ".json")
        if not trace_path.exists(): continue
        try:
            trace = json.load(open(trace_path))
            turns = trace.get("turns", [])
        except Exception:
            continue
        if not turns: continue
        m = compute_metrics(turns, last_n=10)
        m["iid"] = iid
        m["sub_class"] = sub_class[iid]
        m["n_turns"] = len(turns)
        m["emit_finish"] = emit_finish_at(turns) is not None
        m["patch_n_bytes"] = results.get(iid, {}).get("patch_n_bytes", 0)
        traj.append(m)

    by_class = defaultdict(list)
    for m in traj:
        by_class[m["sub_class"]].append(m)
    print(f"Loaded {len(traj)} trajectories: success={len(by_class['success'])}, locked={len(by_class['locked'])}, wandering={len(by_class['wandering'])}\n")

    print("=== v5 tool-use metrics: WANDERING vs SUCCESS vs LOCKED ===")
    for metric in ["tool_entropy_last10", "tool_diversity_last10", "tool_repetition_last10", "bigram_repeat_rate"]:
        s_vals = [m[metric] for m in by_class["success"]]
        l_vals = [m[metric] for m in by_class["locked"]]
        w_vals = [m[metric] for m in by_class["wandering"]]
        try:
            u_ws, p_ws = mannwhitneyu(w_vals, s_vals, alternative="two-sided")
        except Exception:
            u_ws, p_ws = -1, 1.0
        try:
            u_wl, p_wl = mannwhitneyu(w_vals, l_vals, alternative="two-sided")
        except Exception:
            u_wl, p_wl = -1, 1.0
        print(f"\n  {metric}")
        print(f"    SUCCESS:   mean={np.mean(s_vals):.3f}, median={np.median(s_vals):.3f}")
        print(f"    LOCKED:    mean={np.mean(l_vals):.3f}, median={np.median(l_vals):.3f}")
        print(f"    WANDERING: mean={np.mean(w_vals):.3f}, median={np.median(w_vals):.3f}")
        print(f"    W vs S: p={p_ws:.4f},  W vs L: p={p_wl:.4f}")

    # ROC-style: per metric, sweep threshold to find best precision/recall
    print("\n=== Tier 3 search: per-metric threshold sweep (FP <= 5%) ===")
    print(f"{'metric':>30} {'direction':>10} {'best_thresh':>13} {'W_rec':>7} {'S_FP':>7} {'L_caught':>10}")
    for metric in ["tool_entropy_last10", "tool_diversity_last10", "tool_repetition_last10", "bigram_repeat_rate"]:
        for direction in [">", "<"]:
            vals = sorted(set([m[metric] for m in traj]))
            best_w = 0
            best_t = None
            best_sfp = 0
            best_lc = 0
            for t in vals:
                w_pos = sum(1 for m in by_class["wandering"] if (m[metric] > t if direction == ">" else m[metric] < t))
                s_pos = sum(1 for m in by_class["success"] if (m[metric] > t if direction == ">" else m[metric] < t))
                l_pos = sum(1 for m in by_class["locked"] if (m[metric] > t if direction == ">" else m[metric] < t))
                sfp_rate = s_pos / len(by_class["success"])
                if sfp_rate <= 0.05 and w_pos > best_w:
                    best_w = w_pos
                    best_t = t
                    best_sfp = s_pos
                    best_lc = l_pos
            if best_t is not None and best_w > 0:
                print(f"{metric:>30} {direction:>10} {best_t:>13.3f} {best_w}/{len(by_class['wandering'])} ({100*best_w/len(by_class['wandering']):>3.0f}%) {best_sfp}/{len(by_class['success'])} ({100*best_sfp/len(by_class['success']):>3.0f}%) {best_lc}/{len(by_class['locked'])}")
            else:
                print(f"{metric:>30} {direction:>10} {'no signal at FP<=5%':>13}")

    # Save
    (OUT_DIR / "v5_tool_entropy.json").write_text(json.dumps(traj, indent=2, default=float))
    print(f"\nSaved {OUT_DIR}/v5_tool_entropy.json")


if __name__ == "__main__":
    main()
