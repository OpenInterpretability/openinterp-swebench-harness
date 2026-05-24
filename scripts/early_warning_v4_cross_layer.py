"""
Early-warning v4: cross-layer probe disagreement.

Mechanistic hypothesis: WANDERING = convergence failure. If the model has
NOT consolidated a decision, different layers of the residual stream should
encode CONFLICTING verdicts. SUCCESS should show layer-convergence over
time; WANDERING should stay layer-dispersed.

This is structurally different from v1/v2/v3 (text-based) — uses ONLY
residual-derived probe scores at L11/23/31/43/55 per turn.

Signals tested:
  - range(t)       = max_L probe_L(t) - min_L probe_L(t)
  - std(t)         = std across L probe_L(t)
  - sign_dis(t)    = fraction of layer pairs (i,j) with (s_i>0.5) != (s_j>0.5)
  - late_convergence_rate = slope of range(t) over last 30% of turns
                            (negative slope = convergence; ~0 = stuck dispersion)

Operational signal: alarm if late-turn disagreement high AND no_finish.

If WANDERING > SUCCESS on any metric with effect size + early-fire possibility,
the signal is viable. If not, the convergence hypothesis is refuted as a
cross-layer phenomenon.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import safetensors.torch as st
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")

LAYERS = [11, 23, 31, 43, 55]
POSITION = "pre_tool"
TOP_K = 10
TENSOR_NAME_RE = re.compile(r"t(\d+)_(\w+?)_p(\d+)_L(\d+)$")

INFLECTION = json.load(open(OUT_DIR / "inflection_results.json"))


def load_turn_residuals_for_layer(path: Path, layer: int):
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
            "finish_emit_turn": None,  # filled later if needed
        })

    n = len(base_data)
    rep_y = np.array([t["label"] for t in base_data])
    print(f"Loaded {n} trajectories\n")

    # For each layer: load + CV-fit + per-turn-score all trajectories
    per_layer_per_turn_scores = {L: [None] * n for L in LAYERS}
    per_layer_auroc = {}

    for L in LAYERS:
        print(f"=== Layer L{L} pre_tool: loading + CV-fitting + per-turn scoring ===")
        traj_residuals = []
        for t in base_data:
            try:
                tr = load_turn_residuals_for_layer(t["local"], L)
            except Exception:
                tr = None
            traj_residuals.append(tr if tr else {})
        # Build rep_X (mean across turns)
        rep_X = np.stack([
            np.mean(np.stack(list(r.values())), axis=0) if r else np.zeros(5120, dtype=np.float32)
            for r in traj_residuals
        ])
        fold_aurocs = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(rep_X, rep_y):
            dims = select_top_k(rep_X[train_idx], rep_y[train_idx], k=TOP_K)
            scaler = StandardScaler()
            s = scaler.fit_transform(rep_X[train_idx][:, dims])
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(s, rep_y[train_idx])
            ts = clf.predict_proba(scaler.transform(rep_X[test_idx][:, dims]))[:, 1]
            fold_aurocs.append(roc_auc_score(rep_y[test_idx], ts))
            for i in test_idx:
                scores = {}
                for turn, res in traj_residuals[i].items():
                    feats = res[dims].reshape(1, -1)
                    scaled = scaler.transform(feats)
                    scores[turn] = float(clf.predict_proba(scaled)[0, 1])
                per_layer_per_turn_scores[L][i] = scores
        per_layer_auroc[L] = float(np.mean(fold_aurocs))
        print(f"  CV AUROC: {per_layer_auroc[L]:.3f}")

    # Now compute disagreement metrics per trajectory per turn
    print("\n=== Computing cross-layer disagreement metrics ===")
    traj_metrics = []
    for i, t in enumerate(base_data):
        per_turn_layer_scores = {}
        # Find turns shared by ALL layers
        all_turns = set()
        for L in LAYERS:
            sc = per_layer_per_turn_scores[L][i]
            if sc:
                all_turns |= set(sc.keys())
        shared = sorted([turn for turn in all_turns
                         if all(per_layer_per_turn_scores[L][i] and turn in per_layer_per_turn_scores[L][i] for L in LAYERS)])
        if not shared:
            continue
        ranges = []
        stds = []
        sign_dis = []
        for turn in shared:
            scores_at_t = np.array([per_layer_per_turn_scores[L][i][turn] for L in LAYERS])
            ranges.append(float(scores_at_t.max() - scores_at_t.min()))
            stds.append(float(scores_at_t.std()))
            # sign disagreement
            signs = (scores_at_t > 0.5).astype(int)
            n_pairs = len(LAYERS) * (len(LAYERS) - 1) / 2
            n_disagree_pairs = 0
            for i_l in range(len(LAYERS)):
                for j_l in range(i_l + 1, len(LAYERS)):
                    if signs[i_l] != signs[j_l]:
                        n_disagree_pairs += 1
            sign_dis.append(n_disagree_pairs / n_pairs)
        # Aggregate
        range_mean = float(np.mean(ranges))
        std_mean = float(np.mean(stds))
        sign_dis_mean = float(np.mean(sign_dis))
        range_final = ranges[-1]
        std_final = stds[-1]
        sign_dis_final = sign_dis[-1]
        # Late-half mean
        half = len(shared) // 2
        range_late = float(np.mean(ranges[half:]))
        std_late = float(np.mean(stds[half:]))
        sign_dis_late = float(np.mean(sign_dis[half:]))
        # Convergence trend: slope of range over last 30%
        last_30 = max(1, int(0.3 * len(shared)))
        if last_30 >= 2:
            x = np.arange(last_30)
            late_ranges = ranges[-last_30:]
            slope, _ = np.polyfit(x, late_ranges, 1)
            late_convergence_slope = float(slope)
        else:
            late_convergence_slope = 0.0
        # Final-turn range (most "decision time")
        traj_metrics.append({
            "iid": t["iid"],
            "sub_class": t["sub_class"],
            "n_turns": t["n_turns"],
            "range_mean": range_mean,
            "std_mean": std_mean,
            "sign_dis_mean": sign_dis_mean,
            "range_final": range_final,
            "std_final": std_final,
            "sign_dis_final": sign_dis_final,
            "range_late": range_late,
            "std_late": std_late,
            "sign_dis_late": sign_dis_late,
            "late_convergence_slope": late_convergence_slope,
            "ranges_per_turn": ranges,
        })

    by_class = defaultdict(list)
    for m in traj_metrics:
        by_class[m["sub_class"]].append(m)
    print(f"\n  success={len(by_class['success'])}, locked={len(by_class['locked'])}, wandering={len(by_class['wandering'])}")

    # Test each metric WANDERING vs SUCCESS
    print("\n=== Cross-layer disagreement: WANDERING vs SUCCESS (Mann-Whitney) ===")
    for metric in ["range_mean", "std_mean", "sign_dis_mean", "range_final", "std_final", "sign_dis_final",
                   "range_late", "std_late", "sign_dis_late", "late_convergence_slope"]:
        w = [m[metric] for m in by_class["wandering"]]
        s = [m[metric] for m in by_class["success"]]
        l = [m[metric] for m in by_class["locked"]]
        try:
            u_ws, p_ws = mannwhitneyu(w, s, alternative="two-sided")
        except Exception:
            u_ws, p_ws = -1, 1.0
        try:
            u_wl, p_wl = mannwhitneyu(w, l, alternative="two-sided")
        except Exception:
            u_wl, p_wl = -1, 1.0
        print(f"\n  {metric}")
        print(f"    SUCCESS:   mean={np.mean(s):.3f}, median={np.median(s):.3f}")
        print(f"    LOCKED:    mean={np.mean(l):.3f}, median={np.median(l):.3f}")
        print(f"    WANDERING: mean={np.mean(w):.3f}, median={np.median(w):.3f}")
        print(f"    W vs S p={p_ws:.4f}, W vs L p={p_wl:.4f}")

    # Test if disagreement at EARLY turns predicts WANDERING vs SUCCESS
    print("\n=== Early-turn disagreement as predictor (turns 1, 3, 5, 10) ===")
    for early_t in [1, 3, 5, 10]:
        s_vals, w_vals, l_vals = [], [], []
        for m in by_class["success"]:
            if early_t < len(m["ranges_per_turn"]):
                s_vals.append(m["ranges_per_turn"][early_t])
        for m in by_class["wandering"]:
            if early_t < len(m["ranges_per_turn"]):
                w_vals.append(m["ranges_per_turn"][early_t])
        for m in by_class["locked"]:
            if early_t < len(m["ranges_per_turn"]):
                l_vals.append(m["ranges_per_turn"][early_t])
        if s_vals and w_vals:
            u, p = mannwhitneyu(w_vals, s_vals, alternative="two-sided")
            print(f"  Turn {early_t}: range — SUCCESS median={np.median(s_vals):.3f} (n={len(s_vals)}), "
                  f"WANDERING median={np.median(w_vals):.3f} (n={len(w_vals)}), p={p:.4f}")

    # Build best operational signal — late-half range threshold + no_finish
    print("\n=== Operational sweep: late-half range threshold ===")
    print(f"{'thresh':>7} {'T_frac':>8} | {'W_rec':>7} {'lead':>5} | {'S_FP':>6}")
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for T_frac in [0.5, 0.7, 0.8, 0.9]:
            w_caught = 0
            w_leads = []
            s_fp = 0
            for m in traj_metrics:
                # For each turn t <= T_frac * n_turns, find if range > thresh
                T = int(T_frac * m["n_turns"])
                ranges = m["ranges_per_turn"]
                fire_turn = None
                for tt in range(min(T + 1, len(ranges))):
                    if ranges[tt] > thresh:
                        fire_turn = tt
                        break
                if fire_turn is None:
                    continue
                if m["sub_class"] == "wandering":
                    w_caught += 1
                    w_leads.append(m["n_turns"] - fire_turn)
                elif m["sub_class"] == "success":
                    s_fp += 1
            wcr = w_caught / len(by_class["wandering"])
            sfpr = s_fp / len(by_class["success"])
            lm = f"{np.median(w_leads):.0f}" if w_leads else "  -"
            print(f"{thresh:>7.2f} {T_frac:>8.2f} | {100*wcr:>5.0f}% {lm:>5} | {s_fp:>2}/{len(by_class['success']):<2}={100*sfpr:>5.1f}%")

    # Save
    out_path = OUT_DIR / "early_warning_v4_cross_layer.json"
    serial = []
    for m in traj_metrics:
        sd = {k: v for k, v in m.items() if k != "ranges_per_turn"}
        sd["ranges_per_turn"] = m["ranges_per_turn"]
        serial.append(sd)
    out_path.write_text(json.dumps({
        "per_layer_auroc": per_layer_auroc,
        "per_trajectory": serial,
    }, indent=2))
    print(f"\nSaved {out_path}")

    # Plot per-class disagreement trajectory
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"success": "tab:green", "locked": "tab:orange", "wandering": "tab:red"}
    for cls in ["success", "locked", "wandering"]:
        # Average range over normalized fraction-of-trajectory bins
        n_bins = 20
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        for m in by_class[cls]:
            for turn_idx, r in enumerate(m["ranges_per_turn"]):
                bi = min(int(n_bins * turn_idx / m["n_turns"]), n_bins - 1)
                bin_sums[bi] += r
                bin_counts[bi] += 1
        mean_per_bin = bin_sums / np.maximum(bin_counts, 1)
        x = (np.arange(n_bins) + 0.5) / n_bins
        ax.plot(x, mean_per_bin, marker="o", label=f"{cls} (n={len(by_class[cls])})",
                color=colors[cls], lw=2)
    ax.set_xlabel("Fraction of trajectory length")
    ax.set_ylabel("Cross-layer range(t) = max_L probe_L - min_L probe_L")
    ax.set_title("Cross-layer probe disagreement evolution, by sub-class")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT_DIR / "early_warning_v4_disagreement_trajectory.png", dpi=120, bbox_inches="tight")
    print(f"Saved plot to {OUT_DIR}/early_warning_v4_disagreement_trajectory.png")


if __name__ == "__main__":
    main()
