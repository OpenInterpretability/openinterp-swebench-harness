"""
Inflection Turn Analysis — Phase 6 N=99 Qwen3.6-27B SWE-bench Pro trajectories.

Question: at what turn does agent failure become "irrecoverable" — i.e., the
probe-score commits to the failure regime and stays there?

Three operational definitions, all probe-based (no rollouts needed):

  - LOCK_FAIL: earliest turn T such that probe_score(t) < 0.3 for all t in [T, end]
  - LOCK_SUCC: earliest turn T such that probe_score(t) > 0.7 for all t in [T, end]
  - DIVERGENCE_TURN: turn at which residual diverges from successful-trajectory mean

Cross-validated to avoid leakage. Uses L43 pre_tool, top-10 diff-means probe.
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
OUT_DIR.mkdir(exist_ok=True)

LAYER = 43
POSITION = "pre_tool"
TOP_K = 10

TENSOR_NAME_RE = re.compile(r"t(\d+)_(\w+?)_p(\d+)_L(\d+)$")


def load_turn_residuals(path: Path):
    tensors = st.load_file(str(path))
    by_turn = defaultdict(list)
    for name, t in tensors.items():
        m = TENSOR_NAME_RE.match(name)
        if m and m.group(2) == POSITION and int(m.group(4)) == LAYER:
            by_turn[int(m.group(1))].append(t.float().numpy())
    return {turn: np.mean(np.stack(vs), axis=0) for turn, vs in by_turn.items()}


def select_top_k(X, y, k=10):
    diff = np.abs(X[y == 1].mean(0) - X[y == 0].mean(0))
    return np.argsort(-diff)[:k].tolist()


def fit_probe(X, y, dims):
    scaler = StandardScaler()
    s = scaler.fit_transform(X[:, dims])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(s, y)
    return scaler, clf


def score(residual, scaler, clf, dims):
    return float(clf.predict_proba(scaler.transform(residual[dims].reshape(1, -1)))[0, 1])


def lock_in_turn(scores_per_turn, threshold, direction):
    """earliest T s.t. score(t) {< or >} threshold for all t in [T, end]."""
    turns = sorted(scores_per_turn.keys())
    if not turns:
        return None
    for T in turns:
        rest = [scores_per_turn[t] for t in turns if t >= T]
        if direction == "below" and all(s < threshold for s in rest):
            return T
        if direction == "above" and all(s > threshold for s in rest):
            return T
    return None


def main():
    results = json.load(open(PHASE6 / "phase6_results.json"))
    captures_dir = PHASE6 / "captures"

    print("Loading trajectories...")
    traj_data = []
    for iid, entry in results.items():
        if entry.get("captures_safetensors"):
            local = captures_dir / Path(entry["captures_safetensors"]).name
            if local.exists():
                try:
                    tr = load_turn_residuals(local)
                    if tr:
                        traj_data.append({
                            "iid": iid,
                            "label": 1 if entry["finish_reason"] == "finish_tool" else 0,
                            "n_turns": entry["n_turns"],
                            "patch_n_bytes": entry.get("patch_n_bytes", 0),
                            "turns": tr,
                            "rep": np.mean(np.stack(list(tr.values())), axis=0),
                        })
                except Exception:
                    pass
    n = len(traj_data)
    rep_X = np.stack([t["rep"] for t in traj_data])
    rep_y = np.array([t["label"] for t in traj_data])
    print(f"Loaded {n} (success={int(rep_y.sum())}, fail={n - int(rep_y.sum())})")

    # 5-fold CV: per-fold, score every test trajectory's per-turn residual
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    per_traj_scores = [None] * n
    fold_aurocs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(rep_X, rep_y)):
        dims = select_top_k(rep_X[train_idx], rep_y[train_idx], k=TOP_K)
        scaler, clf = fit_probe(rep_X[train_idx], rep_y[train_idx], dims)
        # baseline rep AUROC on test
        test_scores = clf.predict_proba(scaler.transform(rep_X[test_idx][:, dims]))[:, 1]
        fold_aurocs.append(roc_auc_score(rep_y[test_idx], test_scores))
        # per-turn scores for test trajectories
        for i in test_idx:
            scores_dict = {turn: score(res, scaler, clf, dims) for turn, res in traj_data[i]["turns"].items()}
            per_traj_scores[i] = scores_dict

    print(f"CV rep AUROC: {np.mean(fold_aurocs):.3f} ± {np.std(fold_aurocs):.3f}")

    # Compute inflection metrics per trajectory
    for i, t in enumerate(traj_data):
        s = per_traj_scores[i]
        if not s:
            continue
        t["lock_fail_0.30"] = lock_in_turn(s, 0.30, "below")
        t["lock_fail_0.40"] = lock_in_turn(s, 0.40, "below")
        t["lock_succ_0.70"] = lock_in_turn(s, 0.70, "above")
        t["lock_succ_0.60"] = lock_in_turn(s, 0.60, "above")
        t["final_score"] = s[max(s.keys())]
        t["first_score"] = s[min(s.keys())]
        t["score_trajectory"] = s

    # Print results
    print("\n=== Distribution of LOCK_FAIL (earliest T s.t. probe<thresh until end) ===")
    for thresh in [0.30, 0.40]:
        succ_locks = [t[f"lock_fail_{thresh:.2f}"] for t in traj_data if t["label"] == 1 and t.get(f"lock_fail_{thresh:.2f}") is not None]
        fail_locks = [t[f"lock_fail_{thresh:.2f}"] for t in traj_data if t["label"] == 0 and t.get(f"lock_fail_{thresh:.2f}") is not None]
        n_succ_no = sum(1 for t in traj_data if t["label"] == 1 and t.get(f"lock_fail_{thresh:.2f}") is None)
        n_fail_no = sum(1 for t in traj_data if t["label"] == 0 and t.get(f"lock_fail_{thresh:.2f}") is None)
        print(f"\n  thresh={thresh}:")
        print(f"    Success → lock_fail: n={len(succ_locks)} ({len(succ_locks)}/40 reached fail-lock), {n_succ_no}/40 never. {'median=' + str(int(np.median(succ_locks))) if succ_locks else ''}")
        print(f"    Failure → lock_fail: n={len(fail_locks)} ({len(fail_locks)}/59 reached fail-lock), {n_fail_no}/59 never. {'median=' + str(int(np.median(fail_locks))) if fail_locks else ''}")
        if fail_locks:
            q25, q50, q75 = np.percentile(fail_locks, [25, 50, 75])
            print(f"      Failure lock_fail quartiles: q25={q25:.0f}, q50={q50:.0f}, q75={q75:.0f}")
            # FRACTIONAL inflection (locked-in turn / total turns) for failures
            fail_frac = [t[f"lock_fail_{thresh:.2f}"] / t["n_turns"] for t in traj_data if t["label"] == 0 and t.get(f"lock_fail_{thresh:.2f}") is not None]
            if fail_frac:
                fq25, fq50, fq75 = np.percentile(fail_frac, [25, 50, 75])
                print(f"      Failure lock_fail FRACTION of total turns: q25={fq25:.2f}, q50={fq50:.2f}, q75={fq75:.2f}")

    print("\n=== Distribution of LOCK_SUCC (earliest T s.t. probe>thresh until end) ===")
    for thresh in [0.70, 0.60]:
        succ_locks = [t[f"lock_succ_{thresh:.2f}"] for t in traj_data if t["label"] == 1 and t.get(f"lock_succ_{thresh:.2f}") is not None]
        fail_locks = [t[f"lock_succ_{thresh:.2f}"] for t in traj_data if t["label"] == 0 and t.get(f"lock_succ_{thresh:.2f}") is not None]
        n_succ_no = sum(1 for t in traj_data if t["label"] == 1 and t.get(f"lock_succ_{thresh:.2f}") is None)
        n_fail_no = sum(1 for t in traj_data if t["label"] == 0 and t.get(f"lock_succ_{thresh:.2f}") is None)
        print(f"\n  thresh={thresh}:")
        print(f"    Success → lock_succ: n={len(succ_locks)} ({len(succ_locks)}/40 reached succ-lock), {n_succ_no}/40 never. {'median=' + str(int(np.median(succ_locks))) if succ_locks else ''}")
        print(f"    Failure → lock_succ: n={len(fail_locks)} ({len(fail_locks)}/59 reached succ-lock), {n_fail_no}/59 never. {'median=' + str(int(np.median(fail_locks))) if fail_locks else ''}")
        if succ_locks:
            q25, q50, q75 = np.percentile(succ_locks, [25, 50, 75])
            print(f"      Success lock_succ quartiles: q25={q25:.0f}, q50={q50:.0f}, q75={q75:.0f}")
            # FRACTIONAL
            succ_frac = [t[f"lock_succ_{thresh:.2f}"] / t["n_turns"] for t in traj_data if t["label"] == 1 and t.get(f"lock_succ_{thresh:.2f}") is not None]
            if succ_frac:
                fq25, fq50, fq75 = np.percentile(succ_frac, [25, 50, 75])
                print(f"      Success lock_succ FRACTION of total turns: q25={fq25:.2f}, q50={fq50:.2f}, q75={fq75:.2f}")

    # Predictability from early turns
    print("\n=== Can early-turn score predict eventual lock_fail or lock_succ? ===")
    # For failures: among those that lock_fail, is early-turn score correlated with lock turn?
    # Among ALL trajectories, can turn-T score predict whether trajectory locks-fail?
    from scipy.stats import spearmanr, mannwhitneyu

    # Within failures, do those that lock_fail EARLY vs LATE differ in early-turn score?
    fail_locked = [t for t in traj_data if t["label"] == 0 and t.get("lock_fail_0.40") is not None]
    if fail_locked:
        for early_turn in [1, 3, 5, 10]:
            early_locks_t = [t["lock_fail_0.40"] for t in fail_locked if early_turn in t["score_trajectory"]]
            early_scores = [t["score_trajectory"][early_turn] for t in fail_locked if early_turn in t["score_trajectory"]]
            if len(early_locks_t) >= 10:
                rho, p = spearmanr(early_scores, early_locks_t)
                print(f"  Within failures: turn-{early_turn} score vs lock_fail_0.40 turn — rho={rho:.3f}, p={p:.4f}, n={len(early_scores)}")

    # Save raw + summary
    summary = {
        "n_trajectories": n,
        "n_success": int(rep_y.sum()),
        "n_failure": n - int(rep_y.sum()),
        "cv_baseline_auroc": float(np.mean(fold_aurocs)),
        "cv_baseline_auroc_std": float(np.std(fold_aurocs)),
        "per_trajectory": [
            {k: (v if not isinstance(v, dict) else {str(kk): float(vv) for kk, vv in v.items()})
             for k, v in t.items()
             if k not in ["turns", "rep"]}
            for t in traj_data
        ],
    }
    out = OUT_DIR / "inflection_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Saved {out} ===")


if __name__ == "__main__":
    main()
