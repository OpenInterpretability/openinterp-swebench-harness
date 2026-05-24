"""
Cross-validated per-turn AUROC analysis.

Re-trains probe (top-10 diff-means + StandardScaler + LR) within each fold.
Per-trajectory representative residual = mean across all turns at L43 pre_tool.
Then per fold, holdout trajectories get per-turn AUROC computed using fold-local probe.

This eliminates Phase 6 leakage concern for the per-turn AUROC pattern claim.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import safetensors.torch as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/trajectory_probe_horizon_out")
OUT_DIR.mkdir(exist_ok=True)

LAYER = 43
POSITION = "pre_tool"
TOP_K_DIMS = 10

TENSOR_NAME_RE = re.compile(r"t(\d+)_(\w+?)_p(\d+)_L(\d+)$")


def load_turn_residuals(safetensors_path: Path, target_position: str, target_layer: int) -> dict[int, np.ndarray]:
    """Return {turn_idx: mean residual across tokens at (target_position, target_layer)}."""
    tensors = st.load_file(str(safetensors_path))
    by_turn: dict[int, list[np.ndarray]] = defaultdict(list)
    for name, t in tensors.items():
        m = TENSOR_NAME_RE.match(name)
        if not m:
            continue
        turn = int(m.group(1))
        position = m.group(2)
        layer = int(m.group(4))
        if position == target_position and layer == target_layer:
            by_turn[turn].append(t.float().numpy())
    return {turn: np.mean(np.stack(vs), axis=0) for turn, vs in by_turn.items()}


def select_top_k_dims(X_train: np.ndarray, y_train: np.ndarray, k: int = 10) -> list[int]:
    """Top-k features by absolute mean difference between positives and negatives."""
    mu_pos = X_train[y_train == 1].mean(axis=0)
    mu_neg = X_train[y_train == 0].mean(axis=0)
    diff = np.abs(mu_pos - mu_neg)
    top_k = np.argsort(-diff)[:k].tolist()
    return top_k


def fit_probe(X_train: np.ndarray, y_train: np.ndarray, dims: list[int]) -> tuple:
    """Fit StandardScaler + LR on selected dims."""
    feats = X_train[:, dims]
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(feats_scaled, y_train)
    return scaler, clf


def score_residual(residual: np.ndarray, scaler, clf, dims: list[int]) -> float:
    feats = residual[dims].reshape(1, -1)
    scaled = scaler.transform(feats)
    return float(clf.predict_proba(scaled)[0, 1])


def run():
    results = json.load(open(PHASE6 / "phase6_results.json"))
    captures_dir = PHASE6 / "captures"

    print("Loading trajectories + per-turn residuals (this takes a couple of min)...")
    items = []
    for iid, entry in results.items():
        if entry.get("captures_safetensors"):
            local_st = captures_dir / Path(entry["captures_safetensors"]).name
            if local_st.exists():
                items.append({
                    "iid": iid,
                    "label": 1 if entry["finish_reason"] == "finish_tool" else 0,
                    "n_turns": entry["n_turns"],
                    "patch_n_bytes": entry.get("patch_n_bytes", 0),
                    "capture_path": local_st,
                })

    # For each trajectory, load all turn residuals at L43 pre_tool
    traj_residuals: list[dict[int, np.ndarray]] = []
    rep_residuals: list[np.ndarray] = []  # mean across all turns = "trajectory representative"
    labels: list[int] = []
    iids: list[str] = []
    for ix, it in enumerate(items):
        try:
            tr = load_turn_residuals(it["capture_path"], POSITION, LAYER)
            if not tr:
                continue
            traj_residuals.append(tr)
            rep = np.mean(np.stack(list(tr.values())), axis=0)
            rep_residuals.append(rep)
            labels.append(it["label"])
            iids.append(it["iid"])
            if ix % 20 == 0:
                print(f"  loaded {ix+1}/{len(items)}")
        except Exception as e:
            print(f"  error {it['iid'][:50]}: {e}")
    rep_X = np.stack(rep_residuals)
    rep_y = np.array(labels)
    n_traj = len(rep_y)
    n_success = int(rep_y.sum())
    print(f"\nLoaded {n_traj} trajectories: {n_success} success / {n_traj-n_success} fail")
    print(f"Rep feature matrix shape: {rep_X.shape}")

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    per_turn_aurocs: dict[int, list[float]] = defaultdict(list)  # turn -> list of fold AUROCs
    per_turn_n: dict[int, list[int]] = defaultdict(list)  # turn -> list of fold N

    fold_baseline_auroc = []  # baseline = standard probe AUROC on representative residual

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(rep_X, rep_y)):
        # Train probe on representative residuals (mean across trajectory turns)
        X_train, X_test = rep_X[train_idx], rep_X[test_idx]
        y_train, y_test = rep_y[train_idx], rep_y[test_idx]
        dims = select_top_k_dims(X_train, y_train, k=TOP_K_DIMS)
        scaler, clf = fit_probe(X_train, y_train, dims)

        # Baseline: representative AUROC on test
        test_feats = X_test[:, dims]
        test_scaled = scaler.transform(test_feats)
        test_scores = clf.predict_proba(test_scaled)[:, 1]
        b_auroc = roc_auc_score(y_test, test_scores)
        fold_baseline_auroc.append(b_auroc)

        # Per-turn AUROC on test holdout
        max_turn = max(max(traj_residuals[i].keys()) for i in test_idx)
        for turn in range(max_turn + 1):
            ys, ss = [], []
            for i in test_idx:
                if turn in traj_residuals[i]:
                    s = score_residual(traj_residuals[i][turn], scaler, clf, dims)
                    ys.append(rep_y[i])
                    ss.append(s)
            if len(ys) >= 5 and len(set(ys)) == 2:
                try:
                    a = roc_auc_score(ys, ss)
                    per_turn_aurocs[turn].append(a)
                    per_turn_n[turn].append(len(ys))
                except Exception:
                    pass

        print(f"  fold {fold_idx+1}/5: baseline_auroc={b_auroc:.3f} dims={dims[:5]}...")

    # Aggregate
    print(f"\n=== Baseline AUROC (rep residual, 5-fold) ===")
    print(f"  mean={np.mean(fold_baseline_auroc):.3f}  std={np.std(fold_baseline_auroc):.3f}")
    print(f"  per fold: {[round(a,3) for a in fold_baseline_auroc]}")

    print(f"\n=== Cross-validated Per-Turn AUROC (mean ± std across 5 folds) ===")
    print(f"{'turn':>4} {'mean':>6} {'std':>6} {'n_folds':>8} {'avg_n':>6}")
    summary = []
    for turn in sorted(per_turn_aurocs.keys()):
        scores = per_turn_aurocs[turn]
        ns = per_turn_n[turn]
        if len(scores) >= 3:  # require ≥3 folds
            mean_a = np.mean(scores)
            std_a = np.std(scores)
            mark = " ⭐" if mean_a >= 0.62 else ""
            print(f"{turn:>4} {mean_a:>6.3f} {std_a:>6.3f} {len(scores):>8} {int(np.mean(ns)):>6}{mark}")
            summary.append({"turn": turn, "mean_auroc": float(mean_a), "std_auroc": float(std_a),
                            "n_folds": len(scores), "avg_n": int(np.mean(ns))})

    out_path = OUT_DIR / "cv_horizon.json"
    out_path.write_text(json.dumps({
        "n_trajectories": n_traj,
        "n_success": n_success,
        "n_fail": n_traj - n_success,
        "baseline_auroc_mean": float(np.mean(fold_baseline_auroc)),
        "baseline_auroc_std": float(np.std(fold_baseline_auroc)),
        "per_turn": summary,
    }, indent=2))
    print(f"\n=== Saved to {out_path} ===")


if __name__ == "__main__":
    run()
