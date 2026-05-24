"""
v4 mid-layer-only sweep — filter noise from L11 (AUROC 0.873) and L55
(0.888) by restricting to L23/L31/L43 (all >= 0.928 AUROC).

If signal stays clean, mid-layer disagreement is the actual operational
signal. If signal weakens, then noisy-layer disagreement was carrying
the discrimination.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import safetensors.torch as st
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")
INFLECTION = json.load(open(OUT_DIR / "inflection_results.json"))

LAYERS = [23, 31, 43]  # mid-layers only
POSITION = "pre_tool"
TOP_K = 10
TENSOR_NAME_RE = re.compile(r"t(\d+)_(\w+?)_p(\d+)_L(\d+)$")


def load_turn_residuals_for_layer(path, layer):
    tensors = st.load_file(str(path))
    by_turn = defaultdict(list)
    for name, t in tensors.items():
        m = TENSOR_NAME_RE.match(name)
        if m and m.group(2) == POSITION and int(m.group(4)) == layer:
            by_turn[int(m.group(1))].append(t.float().numpy())
    return {turn: np.mean(np.stack(vs), axis=0) for turn, vs in by_turn.items()}


def select_top_k(X, y, k=10):
    diff = np.abs(X[y == 1].mean(0) - X[y == 0].mean(0))
    return np.argsort(-diff)[:k].tolist()


def emit_finish_at(trace_path):
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
    sub_class = {}
    for t in INFLECTION["per_trajectory"]:
        if t["label"] == 1:
            sub_class[t["iid"]] = "success"
        elif t.get("lock_fail_0.40") is not None:
            sub_class[t["iid"]] = "locked"
        else:
            sub_class[t["iid"]] = "wandering"

    results = json.load(open(PHASE6 / "phase6_results.json"))
    captures_dir = PHASE6 / "captures"

    base_data = []
    for iid, entry in results.items():
        if iid not in sub_class or not entry.get("captures_safetensors"):
            continue
        local = captures_dir / Path(entry["captures_safetensors"]).name
        if not local.exists():
            continue
        base_data.append({
            "iid": iid,
            "local": local,
            "label": 1 if entry["finish_reason"] == "finish_tool" else 0,
            "sub_class": sub_class[iid],
            "n_turns": entry["n_turns"],
            "finish_emit_turn": emit_finish_at(PHASE6 / "traces" / (iid + ".json")),
        })

    n = len(base_data)
    rep_y = np.array([t["label"] for t in base_data])
    print(f"Loaded {n} trajectories\n")

    per_layer_per_turn = {L: [None] * n for L in LAYERS}
    for L in LAYERS:
        print(f"=== L{L} ===")
        traj_residuals = []
        for t in base_data:
            try:
                tr = load_turn_residuals_for_layer(t["local"], L)
            except Exception:
                tr = None
            traj_residuals.append(tr if tr else {})
        rep_X = np.stack([
            np.mean(np.stack(list(r.values())), axis=0) if r else np.zeros(5120, dtype=np.float32)
            for r in traj_residuals
        ])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(rep_X, rep_y):
            dims = select_top_k(rep_X[train_idx], rep_y[train_idx], k=TOP_K)
            scaler = StandardScaler()
            s = scaler.fit_transform(rep_X[train_idx][:, dims])
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(s, rep_y[train_idx])
            for i in test_idx:
                scores = {}
                for turn, res in traj_residuals[i].items():
                    feats = res[dims].reshape(1, -1)
                    scaled = scaler.transform(feats)
                    scores[turn] = float(clf.predict_proba(scaled)[0, 1])
                per_layer_per_turn[L][i] = scores

    # Compute disagreement metrics
    traj_metrics = []
    for i, t in enumerate(base_data):
        all_turns = set()
        for L in LAYERS:
            sc = per_layer_per_turn[L][i]
            if sc:
                all_turns |= set(sc.keys())
        shared = sorted([turn for turn in all_turns
                         if all(per_layer_per_turn[L][i] and turn in per_layer_per_turn[L][i] for L in LAYERS)])
        if not shared:
            continue
        ranges = []
        for turn in shared:
            scores_at_t = np.array([per_layer_per_turn[L][i][turn] for L in LAYERS])
            ranges.append(float(scores_at_t.max() - scores_at_t.min()))
        traj_metrics.append({
            "iid": t["iid"],
            "sub_class": t["sub_class"],
            "n_turns": t["n_turns"],
            "finish_emit_turn": t["finish_emit_turn"],
            "ranges": ranges,
            "range_late": float(np.mean(ranges[len(ranges)//2:])),
            "range_mean": float(np.mean(ranges)),
        })

    by_class = defaultdict(list)
    for m in traj_metrics:
        by_class[m["sub_class"]].append(m)

    print(f"\nsuccess={len(by_class['success'])}, locked={len(by_class['locked'])}, wandering={len(by_class['wandering'])}")

    print("\n=== Mid-layer (L23/31/43) range stats ===")
    for metric in ["range_mean", "range_late"]:
        s = [m[metric] for m in by_class["success"]]
        l = [m[metric] for m in by_class["locked"]]
        w = [m[metric] for m in by_class["wandering"]]
        u_ws, p_ws = mannwhitneyu(w, s, alternative="two-sided")
        u_wl, p_wl = mannwhitneyu(w, l, alternative="two-sided")
        print(f"\n  {metric}: SUCCESS {np.mean(s):.3f}, LOCKED {np.mean(l):.3f}, WANDERING {np.mean(w):.3f}")
        print(f"    W vs S p={p_ws:.4f}, W vs L p={p_wl:.4f}")

    # Operational sweep
    print("\n=== Mid-layer operational sweep ===")
    print(f"{'thresh':>7} {'T_frac':>8} | {'W_rec':>9} {'lead':>5} | {'S_FP':>10} | {'L_caught':>8}")
    print("-" * 75)
    best_op = None
    best_score = -1
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        for T_frac in [0.5, 0.6, 0.7, 0.8, 0.9]:
            w_caught, w_leads, s_fp, l_caught = 0, [], 0, 0
            for m in traj_metrics:
                T = min(int(T_frac * m["n_turns"]), len(m["ranges"]) - 1)
                if T < 5: continue
                half = T // 2
                late_mean = float(np.mean(m["ranges"][half:T+1]))
                if late_mean <= thresh: continue
                ft = m["finish_emit_turn"]
                if ft is not None and ft <= T: continue
                if m["sub_class"] == "wandering":
                    w_caught += 1
                    w_leads.append(m["n_turns"] - T)
                elif m["sub_class"] == "success":
                    if ft is not None and ft > T:
                        s_fp += 1
                elif m["sub_class"] == "locked":
                    l_caught += 1
            wcr = w_caught / len(by_class["wandering"])
            sfpr = s_fp / len(by_class["success"])
            score = wcr - 2 * sfpr
            if score > best_score:
                best_score = score
                best_op = {"thresh": thresh, "T_frac": T_frac, "w_caught": w_caught, "w_recall": wcr,
                           "s_fp": s_fp, "s_fp_rate": sfpr, "lead_median": float(np.median(w_leads)) if w_leads else None,
                           "l_caught": l_caught}
            lm = f"{np.median(w_leads):.0f}" if w_leads else "  -"
            print(f"{thresh:>7.2f} {T_frac:>8.2f} | {100*wcr:>5.0f}% {w_caught:>2}/{len(by_class['wandering']):<2} {lm:>5} | {s_fp:>2}/{len(by_class['success']):<2}={100*sfpr:>5.1f}% | {l_caught:>3}/{len(by_class['locked'])}")

    print(f"\n=== Best mid-layer op (score=recall-2×FP): thresh={best_op['thresh']}, T_frac={best_op['T_frac']} ===")
    print(f"    WANDERING: {best_op['w_caught']}/20 ({100*best_op['w_recall']:.0f}%)")
    print(f"    SUCCESS FP: {best_op['s_fp']}/40 ({100*best_op['s_fp_rate']:.1f}%)")
    print(f"    Lead: {best_op['lead_median']}")

    # Save
    (OUT_DIR / "early_warning_v4_midlayer.json").write_text(json.dumps({
        "layers": LAYERS,
        "best_op": best_op,
        "per_trajectory": [{k: v for k, v in m.items() if k != "ranges"} for m in traj_metrics],
    }, indent=2))
    print(f"\nSaved {OUT_DIR}/early_warning_v4_midlayer.json")


if __name__ == "__main__":
    main()
