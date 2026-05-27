"""
Paper #2 v2 — Pre-compute L11/L23/L31 donor residuals for L11 intervention experiment.

Companion to exp_b_donor_residuals.py (which produced L43+L55 donors). This script
extends donor coverage to early layers (L11), mid-early (L23), mid (L31) so the
L11 SUCCESS-donor injection can be tested per paper #3 Phase 2 stability finding.

Strategy mirrors exp_b: per-repo mean SUCCESS L{layer} pre_tool residual at lock_succ turn.

Output: paper_mega_v4_pairs/exp_l11_donors.safetensors with keys:
  L11_donor_success_{repo}, L23_donor_success_{repo}, L31_donor_success_{repo}
  L11_donor_locked_{repo} (control)
  L11_random_direction_norm_matched (random direction control)
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import safetensors.torch as st

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/paper_mega_v4_pairs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TENSOR_NAME_RE = re.compile(r"t(\d+)_(\w+?)_p(\d+)_L(\d+)$")
TARGET_LAYERS = [11, 23, 31]


def parse_repo(iid: str) -> str:
    if iid.startswith("instance_"):
        s = iid[len("instance_"):]
    else:
        s = iid
    parts = s.split("__")
    return parts[1].split("-")[0] if len(parts) >= 2 else s


def load_residual_at_turn(capture_path: Path, layer: int, position: str, turn: int):
    try:
        tensors = st.load_file(str(capture_path))
    except Exception as e:
        return None, f"load_error: {e}"
    matching = []
    for name, t in tensors.items():
        m = TENSOR_NAME_RE.match(name)
        if m and m.group(2) == position and int(m.group(4)) == layer and int(m.group(1)) == turn:
            matching.append(t.float().numpy())
    if not matching:
        return None, f"no_match: layer={layer} pos={position} turn={turn}"
    return np.mean(np.stack(matching), axis=0), None


def main():
    infl = json.load(open("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out/inflection_results.json"))
    p6_results = json.load(open(PHASE6 / "phase6_results.json"))

    sub_class = {}
    lock_turn = {}
    for t in infl['per_trajectory']:
        if t['label'] == 1:
            sub_class[t['iid']] = 'success'
        elif t.get('lock_fail_0.40') is not None:
            sub_class[t['iid']] = 'locked'
        else:
            sub_class[t['iid']] = 'wandering'
        if sub_class[t['iid']] == 'success':
            lock_turn[t['iid']] = t.get('lock_succ_0.70') or int(0.85 * t['n_turns'])
        elif sub_class[t['iid']] == 'locked':
            lock_turn[t['iid']] = t.get('lock_fail_0.40') or int(0.90 * t['n_turns'])
        else:
            lock_turn[t['iid']] = int(0.80 * t['n_turns'])

    print(f"Sub-class counts: {dict({c: sum(1 for v in sub_class.values() if v == c) for c in ('success', 'locked', 'wandering')})}")

    # Extract residuals at lock turn for L11/L23/L31 from SUCCESS + LOCKED
    donors = {}
    skipped = []
    for iid, cls in sub_class.items():
        if cls == 'wandering':
            continue
        repo = parse_repo(iid)
        entry = p6_results.get(iid, {})
        cap_field = entry.get('captures_safetensors', '')
        if not cap_field:
            skipped.append((iid, 'no_capture_path'))
            continue
        local = PHASE6 / "captures" / Path(cap_field).name
        if not local.exists():
            skipped.append((iid, f'capture_missing: {local.name}'))
            continue
        lt = lock_turn[iid]
        for layer in TARGET_LAYERS:
            resid, err = load_residual_at_turn(local, layer, 'pre_tool', lt)
            if resid is None:
                for delta in [-1, 1, -2, 2, -3, 3]:
                    resid, err = load_residual_at_turn(local, layer, 'pre_tool', lt + delta)
                    if resid is not None:
                        break
            if resid is not None:
                donors.setdefault((repo, cls, layer), []).append(resid)

    print(f"\n=== Donor counts per (repo, class, layer) ===")
    for (repo, cls, layer), residuals in sorted(donors.items()):
        print(f"  ({repo}, {cls}, L{layer:02d}): {len(residuals)} donors, d_model={residuals[0].shape[0]}, mean_norm={np.mean([np.linalg.norm(r) for r in residuals]):.1f}")

    # Compute means + save
    out_tensors = {}
    metadata = {}
    for (repo, cls, layer), residuals in donors.items():
        mean_res = np.mean(np.stack(residuals), axis=0).astype(np.float32)
        key = f"L{layer:02d}_donor_{cls}_{repo}"
        out_tensors[key] = torch.from_numpy(mean_res)
        metadata[key] = {
            'repo': repo,
            'class': cls,
            'layer': layer,
            'position': 'pre_tool',
            'n_donors': len(residuals),
            'norm_l2': float(np.linalg.norm(mean_res)),
            'mean_donor_norm': float(np.mean([np.linalg.norm(r) for r in residuals])),
        }

    # Random direction (norm-matched to L11 mean)
    all_L11_donors = [r for (_, cls, layer), residuals in donors.items() if layer == 11 and cls == 'success' for r in residuals]
    if all_L11_donors:
        avg_norm = np.mean([np.linalg.norm(r) for r in all_L11_donors])
        d_model = all_L11_donors[0].shape[0]
        rng = np.random.default_rng(seed=42)
        random_vec = rng.standard_normal(d_model).astype(np.float32)
        random_vec = (random_vec / np.linalg.norm(random_vec)) * avg_norm
        out_tensors['L11_random_direction_norm_matched'] = torch.from_numpy(random_vec)
        metadata['L11_random_direction_norm_matched'] = {
            'class': 'control_random',
            'layer': 11,
            'position': 'pre_tool',
            'norm_l2': float(avg_norm),
            'seed': 42,
            'd_model': d_model,
        }
        print(f"\n  Added L11 random control: norm_matched={avg_norm:.2f}, seed=42")

    out_path = OUT_DIR / "exp_l11_donors.safetensors"
    st.save_file(out_tensors, str(out_path))
    print(f"\nSaved {out_path} ({out_path.stat().st_size:,} bytes, {len(out_tensors)} tensors)")

    meta_path = OUT_DIR / "exp_l11_donors_metadata.json"
    meta_path.write_text(json.dumps({
        'tensors': metadata,
        'lock_turn_per_iid': lock_turn,
        'sub_class_per_iid': sub_class,
        'skipped': skipped,
        'target_layers': TARGET_LAYERS,
    }, indent=2))
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
