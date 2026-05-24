"""
Probe Saturation Time analysis.

For each trajectory, find the earliest turn T such that probe_score(turn) is
within ε of the final-turn score for all subsequent turns. T is the
"saturation time" — when the probe locks in.

Per-trajectory T distribution by outcome class (success vs fail) tells us:
- Universal decision turn? (T concentrates)
- Bimodal trajectory classes? (T splits)
- Outcome-dependent timing? (T differs by class)

This is a NEW analytical concept not in surveyed prior art.
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
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/trajectory_probe_horizon_out")
OUT_DIR.mkdir(exist_ok=True)

LAYER = 43
POSITION = "pre_tool"
TOP_K = 10

TENSOR_NAME_RE = re.compile(r"t(\d+)_(\w+?)_p(\d+)_L(\d+)$")


def load_turn_residuals(safetensors_path: Path):
    tensors = st.load_file(str(safetensors_path))
    by_turn = defaultdict(list)
    for name, t in tensors.items():
        m = TENSOR_NAME_RE.match(name)
        if not m:
            continue
        turn = int(m.group(1))
        position = m.group(2)
        layer = int(m.group(4))
        if position == POSITION and layer == LAYER:
            by_turn[turn].append(t.float().numpy())
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


def score_turn(residual, scaler, clf, dims):
    f = residual[dims].reshape(1, -1)
    return float(clf.predict_proba(scaler.transform(f))[0, 1])


def saturation_time(scores_per_turn, epsilon=0.10):
    """Earliest turn T such that |score(t) - score(final)| < ε for all t >= T.

    Returns: (T, final_score) or (-1, final_score) if never saturates.
    """
    if not scores_per_turn:
        return -1, 0.5
    turns = sorted(scores_per_turn.keys())
    final = scores_per_turn[turns[-1]]
    for T in turns:
        # Check all turns >= T are within epsilon of final
        ok = all(abs(scores_per_turn[t] - final) < epsilon for t in turns if t >= T)
        if ok:
            return T, final
    return -1, final


def main():
    results = json.load(open(PHASE6 / "phase6_results.json"))
    captures_dir = PHASE6 / "captures"

    print("Loading trajectories + per-turn residuals...")
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
                            "turns": tr,
                            "rep": np.mean(np.stack(list(tr.values())), axis=0),
                        })
                except Exception:
                    pass
    n = len(traj_data)
    print(f"Loaded {n} trajectories")

    rep_X = np.stack([t["rep"] for t in traj_data])
    rep_y = np.array([t["label"] for t in traj_data])

    # 5-fold CV: per-fold, train probe + compute saturation time per HOLD-OUT trajectory
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    per_traj_data = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(rep_X, rep_y)):
        dims = select_top_k(rep_X[train_idx], rep_y[train_idx], k=TOP_K)
        scaler, clf = fit_probe(rep_X[train_idx], rep_y[train_idx], dims)

        for i in test_idx:
            tr = traj_data[i]
            scores = {turn: score_turn(res, scaler, clf, dims) for turn, res in tr["turns"].items()}
            T, final = saturation_time(scores, epsilon=0.10)
            T_loose, _ = saturation_time(scores, epsilon=0.20)
            per_traj_data.append({
                "iid": tr["iid"],
                "label": tr["label"],
                "n_turns": tr["n_turns"],
                "final_score": final,
                "saturation_turn_e0.10": T,
                "saturation_turn_e0.20": T_loose,
                "fold": fold_idx,
                "n_captures": len(scores),
            })

    # Aggregate
    print(f"\n=== Probe Saturation Time Distribution ===")
    print(f"epsilon=0.10 (tight)")
    succ_sats = [d["saturation_turn_e0.10"] for d in per_traj_data if d["label"] == 1 and d["saturation_turn_e0.10"] >= 0]
    fail_sats = [d["saturation_turn_e0.10"] for d in per_traj_data if d["label"] == 0 and d["saturation_turn_e0.10"] >= 0]
    print(f"  Success (n={len(succ_sats)}): median={np.median(succ_sats):.0f}, mean={np.mean(succ_sats):.1f}, q25={np.percentile(succ_sats,25):.0f}, q75={np.percentile(succ_sats,75):.0f}")
    print(f"  Failure (n={len(fail_sats)}): median={np.median(fail_sats):.0f}, mean={np.mean(fail_sats):.1f}, q25={np.percentile(fail_sats,25):.0f}, q75={np.percentile(fail_sats,75):.0f}")

    # Did probe ever lock in? Count non-saturating
    n_unsat_succ = sum(1 for d in per_traj_data if d["label"] == 1 and d["saturation_turn_e0.10"] < 0)
    n_unsat_fail = sum(1 for d in per_traj_data if d["label"] == 0 and d["saturation_turn_e0.10"] < 0)
    print(f"  Non-saturating: success {n_unsat_succ}/{40}, failure {n_unsat_fail}/{59}")

    print(f"\nepsilon=0.20 (loose)")
    succ_sats20 = [d["saturation_turn_e0.20"] for d in per_traj_data if d["label"] == 1 and d["saturation_turn_e0.20"] >= 0]
    fail_sats20 = [d["saturation_turn_e0.20"] for d in per_traj_data if d["label"] == 0 and d["saturation_turn_e0.20"] >= 0]
    print(f"  Success (n={len(succ_sats20)}): median={np.median(succ_sats20):.0f}, mean={np.mean(succ_sats20):.1f}, q25={np.percentile(succ_sats20,25):.0f}, q75={np.percentile(succ_sats20,75):.0f}")
    print(f"  Failure (n={len(fail_sats20)}): median={np.median(fail_sats20):.0f}, mean={np.mean(fail_sats20):.1f}, q25={np.percentile(fail_sats20,25):.0f}, q75={np.percentile(fail_sats20,75):.0f}")

    # Mann-Whitney on saturation times
    from scipy.stats import mannwhitneyu
    if succ_sats and fail_sats:
        u, p = mannwhitneyu(succ_sats, fail_sats, alternative="two-sided")
        print(f"\nMann-Whitney U success vs fail saturation_turn (ε=0.10): U={u}, p={p:.4f}")
    if succ_sats20 and fail_sats20:
        u20, p20 = mannwhitneyu(succ_sats20, fail_sats20, alternative="two-sided")
        print(f"Mann-Whitney U success vs fail saturation_turn (ε=0.20): U={u20}, p={p20:.4f}")

    # Does saturation_turn correlate with n_turns?
    from scipy.stats import spearmanr
    sats_for_corr = [(d["saturation_turn_e0.20"], d["n_turns"]) for d in per_traj_data if d["saturation_turn_e0.20"] >= 0]
    if sats_for_corr:
        s_t, n_t = zip(*sats_for_corr)
        rho, p = spearmanr(s_t, n_t)
        print(f"\nSpearman saturation_turn vs n_turns: rho={rho:.3f}, p={p:.4f}")

    # Save raw + summary
    out_path = OUT_DIR / "saturation_time.json"
    out_path.write_text(json.dumps({
        "per_trajectory": per_traj_data,
        "summary": {
            "epsilon": [0.10, 0.20],
            "success": {
                "n": len(succ_sats),
                "median_sat_turn": float(np.median(succ_sats)) if succ_sats else None,
                "n_non_saturating": n_unsat_succ,
            },
            "failure": {
                "n": len(fail_sats),
                "median_sat_turn": float(np.median(fail_sats)) if fail_sats else None,
                "n_non_saturating": n_unsat_fail,
            },
        },
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
