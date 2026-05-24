"""
Exp C prereq — Extract v5 tool-entropy collapse direction in residual space.

Computes the residual direction that DISTINGUISHES WANDERING from SUCCESS
trajectories at L43 pre_tool position, late-half turns. This is the direction
used in Exp C steering sweep:
  - Push in +direction → toward WANDERING (collapsed entropy)
  - Push in -direction → toward SUCCESS (high entropy)

Two methods (we save BOTH for comparison):

Method 1 — Diff-of-means (Caio's standard, used in paper-MEGA):
  v_collapse = (mean residual of WANDERING) - (mean residual of SUCCESS)
  at L43 pre_tool, averaged over each trajectory's late-half turns

Method 2 — Linear probe direction:
  Train LogReg classifier on L43 pre_tool residuals to predict tool_entropy
  bin (high vs low). The probe's coefficient direction is the projection.

The DIRECTION matters for Exp C; magnitude is variable (α sweep).
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import safetensors.torch as st
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/paper_mega_v4_pairs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER = 43
POSITION = "pre_tool"
TENSOR_NAME_RE = re.compile(r"t(\d+)_(\w+?)_p(\d+)_L(\d+)$")


def load_per_turn_residuals(capture_path: Path, layer: int, position: str):
    tensors = st.load_file(str(capture_path))
    by_turn = defaultdict(list)
    for name, t in tensors.items():
        m = TENSOR_NAME_RE.match(name)
        if m and m.group(2) == position and int(m.group(4)) == layer:
            by_turn[int(m.group(1))].append(t.float().numpy())
    return {turn: np.mean(np.stack(vs), axis=0) for turn, vs in by_turn.items()}


def main():
    infl = json.load(open("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out/inflection_results.json"))
    p6 = json.load(open(PHASE6 / "phase6_results.json"))

    sub_class = {}
    n_turns = {}
    for t in infl['per_trajectory']:
        if t['label']==1: sub_class[t['iid']] = 'success'
        elif t.get('lock_fail_0.40') is not None: sub_class[t['iid']] = 'locked'
        else: sub_class[t['iid']] = 'wandering'
        n_turns[t['iid']] = t['n_turns']

    # Extract late-half-mean L43 pre_tool residual per trajectory
    print(f"=== Extracting L{LAYER} {POSITION} late-half residuals per trajectory ===")
    per_traj_residual = {}
    for iid, cls in sub_class.items():
        if cls not in ('success', 'wandering'):  # skip locked for the contrast
            continue
        entry = p6.get(iid, {})
        cap_path = entry.get('captures_safetensors', '')
        if not cap_path:
            continue
        local = PHASE6 / "captures" / Path(cap_path).name
        if not local.exists():
            continue
        try:
            per_turn = load_per_turn_residuals(local, LAYER, POSITION)
        except Exception as e:
            print(f"  load err {iid[:40]}: {e}")
            continue
        if not per_turn:
            continue
        # Late half: turns from n_turns/2 to end
        nt = n_turns[iid]
        half = nt // 2
        late_residuals = [r for turn, r in per_turn.items() if turn >= half]
        if not late_residuals:
            continue
        per_traj_residual[iid] = {
            'class': cls,
            'late_mean': np.mean(np.stack(late_residuals), axis=0),
            'n_late_turns': len(late_residuals),
        }

    print(f"\n  Extracted: {sum(1 for v in per_traj_residual.values() if v['class']=='success')} SUCCESS, "
          f"{sum(1 for v in per_traj_residual.values() if v['class']=='wandering')} WANDERING")

    # Method 1: diff-of-means
    print(f"\n=== Method 1: diff-of-means direction ===")
    succ_residuals = np.stack([v['late_mean'] for v in per_traj_residual.values() if v['class']=='success'])
    wand_residuals = np.stack([v['late_mean'] for v in per_traj_residual.values() if v['class']=='wandering'])
    print(f"  SUCCESS mean residual shape: {succ_residuals.shape}, mean L2: {np.linalg.norm(succ_residuals.mean(0)):.2f}")
    print(f"  WANDERING mean residual shape: {wand_residuals.shape}, mean L2: {np.linalg.norm(wand_residuals.mean(0)):.2f}")

    v_diff = wand_residuals.mean(0) - succ_residuals.mean(0)
    v_diff_normed = v_diff / np.linalg.norm(v_diff)
    print(f"  Diff direction (W - S) L2 norm: {np.linalg.norm(v_diff):.2f}")
    print(f"  Normalized direction shape: {v_diff_normed.shape}")

    # Method 2: linear probe direction (top-10 features for parsimony)
    print(f"\n=== Method 2: linear probe direction (LogReg on all dims) ===")
    X = np.concatenate([succ_residuals, wand_residuals], axis=0)
    y = np.concatenate([np.zeros(len(succ_residuals)), np.ones(len(wand_residuals))])
    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xs, y)
    accuracy = clf.score(Xs, y)
    print(f"  Train accuracy: {accuracy:.3f}")
    # Coefficient direction in the standardized space; convert back
    coef_std = clf.coef_[0]
    # The direction in original space that corresponds to coef in standardized space:
    # x_std = (x - mu) / sigma; dot(coef_std, x_std) = dot(coef_std/sigma, x - mu)
    # So the direction in original space is coef_std / sigma_diag
    sigma_diag = scaler.scale_
    v_probe = coef_std / sigma_diag
    v_probe_normed = v_probe / np.linalg.norm(v_probe)
    print(f"  Probe direction L2 norm: {np.linalg.norm(v_probe):.2f}")

    # Cosine similarity between diff-of-means and probe direction
    cos = float(np.dot(v_diff_normed, v_probe_normed))
    print(f"\n  Cosine(diff-of-means, probe direction): {cos:.3f}")

    # Save both directions
    out_tensors = {
        'v5_collapse_diff_of_means_L43_pre_tool': torch.from_numpy(v_diff_normed.astype(np.float32)),
        'v5_collapse_probe_direction_L43_pre_tool': torch.from_numpy(v_probe_normed.astype(np.float32)),
        # Also save un-normalized versions for reference
        'v5_collapse_diff_unnormalized_L43_pre_tool': torch.from_numpy(v_diff.astype(np.float32)),
    }
    out_path = OUT_DIR / "exp_c_v5_direction.safetensors"
    st.save_file(out_tensors, str(out_path))

    meta = {
        'layer': LAYER,
        'position': POSITION,
        'method_1_diff_of_means': {
            'description': 'Mean WANDERING residual - mean SUCCESS residual (late-half per trajectory)',
            'n_success': int(len(succ_residuals)),
            'n_wandering': int(len(wand_residuals)),
            'unnormalized_l2': float(np.linalg.norm(v_diff)),
        },
        'method_2_probe_direction': {
            'description': 'LogReg coefficient (standardized → original space) on late-half residuals',
            'classifier': 'sklearn.LogisticRegression(C=1.0)',
            'train_accuracy': float(accuracy),
            'unnormalized_l2': float(np.linalg.norm(v_probe)),
        },
        'cosine_similarity_between_methods': cos,
        'd_model': int(v_diff.shape[0]),
        'reference_residual_l2_for_alpha_scaling': float(np.linalg.norm(succ_residuals.mean(0))),
    }
    (OUT_DIR / "exp_c_v5_direction_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\n=== Saved ===")
    print(f"  {out_path}")
    print(f"  {OUT_DIR}/exp_c_v5_direction_metadata.json")
    print(f"\n=== Suggested α-sweep for Exp C ===")
    print(f"  Residual L2 ≈ {np.linalg.norm(succ_residuals.mean(0)):.0f} (success baseline)")
    print(f"  Suggested α values: {{-200, -100, -50, -10, 0, +10, +50, +100, +200}}")
    print(f"  At α=±200 we exceed residual norm — captures structural-rigidity regime")
    print(f"  Random direction control: norm-matched to {np.linalg.norm(succ_residuals.mean(0)):.0f}, seed=42")


if __name__ == "__main__":
    main()
