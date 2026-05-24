"""
Robustness check: does the WANDERING/LOCKED sub-class assignment replicate
across layers? Tests L11, L23, L31, L43, L55 with top-10 diff-means probe
at pre_tool position.

If sub-class membership is consistent across layers → real phenomenon.
If inconsistent → L43-specific artifact.

Also computes bootstrap 95% CI on the 34% WANDERING rate.
"""
from __future__ import annotations

import json
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
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")

LAYERS = [11, 23, 31, 43, 55]
POSITION = "pre_tool"
TOP_K = 10
TENSOR_NAME_RE = re.compile(r"t(\d+)_(\w+?)_p(\d+)_L(\d+)$")


def load_turn_residuals(path: Path, layer: int):
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


def fit_and_score_per_layer(traj_data_per_layer, rep_y):
    """Return per-trajectory per-fold scores trajectory dict for one layer."""
    rep_X = np.stack([np.mean(np.stack(list(turns.values())), axis=0) for turns in traj_data_per_layer])
    n = len(rep_y)
    per_traj_scores = [None] * n
    aurocs = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(rep_X, rep_y)):
        dims = select_top_k(rep_X[train_idx], rep_y[train_idx], k=TOP_K)
        scaler = StandardScaler()
        s = scaler.fit_transform(rep_X[train_idx][:, dims])
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(s, rep_y[train_idx])
        # test rep AUROC
        ts = clf.predict_proba(scaler.transform(rep_X[test_idx][:, dims]))[:, 1]
        aurocs.append(roc_auc_score(rep_y[test_idx], ts))
        for i in test_idx:
            scores = {}
            for turn, res in traj_data_per_layer[i].items():
                f = res[dims].reshape(1, -1)
                scores[turn] = float(clf.predict_proba(scaler.transform(f))[0, 1])
            per_traj_scores[i] = scores
    return per_traj_scores, aurocs


def classify_subclass(scores_per_turn, label, fail_thresh=0.30):
    """Assigns to: success / locked-fail / wandering-fail."""
    if label == 1:
        return "success"
    if not scores_per_turn:
        return "no_data"
    turns = sorted(scores_per_turn.keys())
    # Earliest T s.t. score(t) < fail_thresh for all t >= T
    for T in turns:
        if all(scores_per_turn[t] < fail_thresh for t in turns if t >= T):
            return "locked"
    return "wandering"


def main():
    results = json.load(open(PHASE6 / "phase6_results.json"))
    captures_dir = PHASE6 / "captures"

    print("Loading trajectories...")
    base_data = []
    for iid, entry in results.items():
        if entry.get("captures_safetensors"):
            local = captures_dir / Path(entry["captures_safetensors"]).name
            if local.exists():
                base_data.append({
                    "iid": iid,
                    "local": local,
                    "label": 1 if entry["finish_reason"] == "finish_tool" else 0,
                    "n_turns": entry["n_turns"],
                })
    n = len(base_data)
    rep_y = np.array([t["label"] for t in base_data])
    print(f"Loaded {n} ({int(rep_y.sum())} success / {n-int(rep_y.sum())} fail)")

    # Per layer: load + score + classify
    per_layer_assignments = {}
    per_layer_auroc = {}

    for L in LAYERS:
        print(f"\n=== Layer L{L} pre_tool ===")
        traj_data = []
        valid_idx = []
        for i, t in enumerate(base_data):
            try:
                tr = load_turn_residuals(t["local"], L)
                if tr:
                    traj_data.append(tr)
                    valid_idx.append(i)
            except Exception:
                pass
        if len(traj_data) < n * 0.9:
            print(f"  WARNING: only {len(traj_data)}/{n} trajectories have L{L} data")
        sub_y = rep_y[valid_idx]
        sub_iids = [base_data[i]["iid"] for i in valid_idx]
        per_traj_scores, fold_aurocs = fit_and_score_per_layer(traj_data, sub_y)
        per_layer_auroc[L] = {
            "mean": float(np.mean(fold_aurocs)),
            "std": float(np.std(fold_aurocs)),
            "folds": [float(a) for a in fold_aurocs],
            "n": len(sub_y),
        }
        # Classify each
        assignments = {}
        for i, scores in enumerate(per_traj_scores):
            cls = classify_subclass(scores, sub_y[i], fail_thresh=0.30)
            assignments[sub_iids[i]] = cls
        per_layer_assignments[L] = assignments
        from collections import Counter
        c = Counter(assignments.values())
        print(f"  CV AUROC: {per_layer_auroc[L]['mean']:.3f} ± {per_layer_auroc[L]['std']:.3f}")
        print(f"  Sub-class distribution: {dict(c)}")

    # Cross-layer consistency: for each (iid, layer1, layer2) check agreement on locked/wandering for failures
    print("\n=== Cross-layer agreement on LOCKED vs WANDERING (failures only) ===")
    fail_iids = [t["iid"] for t in base_data if t["label"] == 0]
    print(f"{'L1':>4} {'L2':>4} {'agree':>7} {'locked_only_L1':>15} {'locked_only_L2':>15} {'wander_both':>12} {'wander_only_L1':>16} {'wander_only_L2':>16}")
    pair_agreement = {}
    for i, L1 in enumerate(LAYERS):
        for L2 in LAYERS[i+1:]:
            a1 = per_layer_assignments[L1]
            a2 = per_layer_assignments[L2]
            agree = 0
            locked_L1_only = 0
            locked_L2_only = 0
            wander_both = 0
            wander_L1_only = 0
            wander_L2_only = 0
            total = 0
            for iid in fail_iids:
                if iid in a1 and iid in a2 and a1[iid] in ("locked", "wandering") and a2[iid] in ("locked", "wandering"):
                    total += 1
                    if a1[iid] == a2[iid]:
                        agree += 1
                        if a1[iid] == "wandering":
                            wander_both += 1
                    else:
                        if a1[iid] == "locked":
                            locked_L1_only += 1
                            wander_L2_only += 1
                        else:
                            wander_L1_only += 1
                            locked_L2_only += 1
            agree_pct = 100 * agree / total if total else 0
            print(f"{L1:>4} {L2:>4} {agree:>3}/{total} ({agree_pct:>3.0f}%) {locked_L1_only:>15} {locked_L2_only:>15} {wander_both:>12} {wander_L1_only:>16} {wander_L2_only:>16}")
            pair_agreement[f"L{L1}_vs_L{L2}"] = {"agree": agree, "total": total, "pct": agree_pct}

    # Bootstrap CI on WANDERING rate at L43 (primary layer)
    print("\n=== Bootstrap CI on WANDERING rate (L43) ===")
    a43 = per_layer_assignments[43]
    n_fail = sum(1 for iid in fail_iids if iid in a43)
    n_wander = sum(1 for iid in fail_iids if a43.get(iid) == "wandering")
    rate = n_wander / n_fail if n_fail else 0
    print(f"  Point estimate: {n_wander}/{n_fail} = {100*rate:.1f}%")
    # Bootstrap
    np.random.seed(42)
    boot_rates = []
    for _ in range(10000):
        sample_iids = np.random.choice(fail_iids, size=n_fail, replace=True)
        n_w = sum(1 for iid in sample_iids if a43.get(iid) == "wandering")
        boot_rates.append(n_w / n_fail)
    ci_low, ci_high = np.percentile(boot_rates, [2.5, 97.5])
    print(f"  Bootstrap 95% CI: [{100*ci_low:.1f}%, {100*ci_high:.1f}%]")
    print(f"  Bootstrap mean: {100*np.mean(boot_rates):.1f}%")

    out = {
        "per_layer_auroc": per_layer_auroc,
        "pair_agreement": pair_agreement,
        "L43_wandering_rate": rate,
        "L43_wandering_ci_95": [float(ci_low), float(ci_high)],
        "L43_wandering_bootstrap_n": 10000,
        "per_layer_subclass_counts": {
            f"L{L}": dict({k: sum(1 for v in per_layer_assignments[L].values() if v == k)
                            for k in ["success", "locked", "wandering"]})
            for L in LAYERS
        },
    }
    out_path = OUT_DIR / "per_layer_robustness.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n=== Saved {out_path} ===")


if __name__ == "__main__":
    main()
