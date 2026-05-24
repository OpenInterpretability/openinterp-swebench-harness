"""
Exp B prereq — Pre-compute donor L55 residuals from SUCCESS trajectories.

For each WANDERING trajectory, we need a "donor" L55 pre_tool residual representing
'what a successful agent in consolidated-decision state looks like'. The donor is
computed as the per-repo mean L55 pre_tool residual across SUCCESS trajectories,
sampled at the donor's lock_succ turn (when probe stabilizes high — the analog
of WANDERING's lock_in turn).

Strategy:
  1. For each SUCCESS trajectory: extract L55 pre_tool residual at lock_succ turn
  2. Group by repo (qutebrowser, ansible, openlibrary)
  3. Compute per-repo mean → 3 donor vectors (d_model = 5120)
  4. For each WANDERING trajectory, save matched donor (per its repo)
  5. Also save: random direction (matched norm), LOCKED donor mean, L43 donor mean
     for the 4 controls (C1-C4)

Output: paper_mega_v4_pairs/exp_b_donors.safetensors with keys:
  L55_donor_qutebrowser, L55_donor_ansible, L55_donor_openlibrary
  L55_donor_locked_qutebrowser, ... (LOCKED-derived control donors)
  L43_donor_qutebrowser, ... (mid-layer C3 control donors)
  random_direction_norm_matched (C1 control, one vector)
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


def parse_repo(iid: str) -> str:
    if iid.startswith("instance_"):
        s = iid[len("instance_"):]
    else:
        s = iid
    parts = s.split("__")
    return parts[1].split("-")[0] if len(parts) >= 2 else s


def load_residual_at_turn(capture_path: Path, layer: int, position: str, turn: int):
    """Load L{layer} {position} residual at turn={turn} from a capture safetensors file."""
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
    # Average across positions within same turn (if multiple captures)
    return np.mean(np.stack(matching), axis=0), None


def main():
    infl = json.load(open("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out/inflection_results.json"))
    p6_results = json.load(open(PHASE6 / "phase6_results.json"))

    sub_class = {}
    lock_turn = {}
    n_turns = {}
    for t in infl['per_trajectory']:
        if t['label']==1:
            sub_class[t['iid']] = 'success'
        elif t.get('lock_fail_0.40') is not None:
            sub_class[t['iid']] = 'locked'
        else:
            sub_class[t['iid']] = 'wandering'
        n_turns[t['iid']] = t['n_turns']
        # For success: lock_succ turn; for locked: lock_fail turn; for wandering: midpoint
        if sub_class[t['iid']] == 'success':
            lock_turn[t['iid']] = t.get('lock_succ_0.70') or int(0.85 * t['n_turns'])
        elif sub_class[t['iid']] == 'locked':
            lock_turn[t['iid']] = t.get('lock_fail_0.40') or int(0.90 * t['n_turns'])
        else:
            lock_turn[t['iid']] = int(0.80 * t['n_turns'])

    print(f"Sub-class counts: {dict({c: sum(1 for v in sub_class.values() if v == c) for c in ('success', 'locked', 'wandering')})}")

    # Group by repo
    by_repo_class = defaultdict(lambda: defaultdict(list))
    for iid, cls in sub_class.items():
        repo = parse_repo(iid)
        by_repo_class[repo][cls].append(iid)
    print(f"\nRepo distribution:")
    for repo, by_cls in by_repo_class.items():
        cnt = {k: len(v) for k, v in by_cls.items()}
        print(f"  {repo}: {cnt}")

    # For each (repo, class, layer), extract residuals at lock turn from each trajectory
    print(f"\n=== Extracting residuals (L43 + L55, position=pre_tool) at lock turns ===")
    donors = {}  # (repo, class, layer) -> list of residual arrays
    skipped = []
    for iid, cls in sub_class.items():
        if cls == 'wandering':
            continue  # we donate FROM success/locked, not from wandering
        repo = parse_repo(iid)
        entry = p6_results.get(iid, {})
        cap_path_field = entry.get('captures_safetensors', '')
        if not cap_path_field:
            skipped.append((iid, 'no_capture_path'))
            continue
        local = PHASE6 / "captures" / Path(cap_path_field).name
        if not local.exists():
            skipped.append((iid, f'capture_missing: {local.name}'))
            continue
        lt = lock_turn[iid]
        for layer in [43, 55]:
            resid, err = load_residual_at_turn(local, layer, 'pre_tool', lt)
            if resid is None:
                # try nearest turn fallback
                for delta in [-1, 1, -2, 2, -3, 3]:
                    resid, err = load_residual_at_turn(local, layer, 'pre_tool', lt + delta)
                    if resid is not None:
                        break
            if resid is not None:
                donors.setdefault((repo, cls, layer), []).append(resid)

    print(f"\n=== Donor counts per (repo, class, layer) ===")
    for (repo, cls, layer), residuals in sorted(donors.items()):
        print(f"  ({repo}, {cls}, L{layer}): {len(residuals)} donors, d_model={residuals[0].shape[0]}")

    # Compute means + save
    out_tensors = {}
    metadata = {}
    for (repo, cls, layer), residuals in donors.items():
        mean_res = np.mean(np.stack(residuals), axis=0).astype(np.float32)
        key = f"L{layer}_donor_{cls}_{repo}"
        out_tensors[key] = torch.from_numpy(mean_res)
        metadata[key] = {
            'repo': repo,
            'class': cls,
            'layer': layer,
            'position': 'pre_tool',
            'n_donors': len(residuals),
            'norm_l2': float(np.linalg.norm(mean_res)),
            'mean_norm': float(np.mean([np.linalg.norm(r) for r in residuals])),
        }

    # Add random direction (C1 control) matched to overall L55 norm
    all_L55_donors = [r for (_, _, layer), residuals in donors.items() if layer == 55 for r in residuals]
    if all_L55_donors:
        avg_norm = np.mean([np.linalg.norm(r) for r in all_L55_donors])
        d_model = all_L55_donors[0].shape[0]
        rng = np.random.default_rng(seed=42)
        random_vec = rng.standard_normal(d_model).astype(np.float32)
        random_vec = (random_vec / np.linalg.norm(random_vec)) * avg_norm
        out_tensors['L55_random_direction_norm_matched'] = torch.from_numpy(random_vec)
        metadata['L55_random_direction_norm_matched'] = {
            'class': 'control_C1_random',
            'layer': 55,
            'position': 'pre_tool',
            'norm_l2': float(avg_norm),
            'seed': 42,
            'd_model': d_model,
        }
        print(f"\n  Added C1 random control: norm_matched={avg_norm:.2f}, seed=42")

    # Save safetensors + metadata
    out_path = OUT_DIR / "exp_b_donors.safetensors"
    st.save_file(out_tensors, str(out_path))
    print(f"\n=== Saved {out_path} ===")
    print(f"  Total tensors: {len(out_tensors)}")
    print(f"  File size: {out_path.stat().st_size:,} bytes")

    meta_path = OUT_DIR / "exp_b_donors_metadata.json"
    meta_path.write_text(json.dumps({
        'tensors': metadata,
        'lock_turn_per_iid': lock_turn,
        'sub_class_per_iid': sub_class,
        'skipped': skipped,
    }, indent=2))
    print(f"  Metadata: {meta_path}")

    # Build per-WANDERING donor assignment table (for Exp B execution)
    print(f"\n=== Per-WANDERING donor mapping (for Exp B execution) ===")
    wandering_donors = {}
    for iid, cls in sub_class.items():
        if cls != 'wandering':
            continue
        repo = parse_repo(iid)
        wandering_donors[iid] = {
            'repo': repo,
            'n_turns': n_turns[iid],
            'lock_turn_estimate': lock_turn[iid],
            'donors_available': {
                'L55_success': f"L55_donor_success_{repo}" if (repo, 'success', 55) in donors else 'MISSING',
                'L55_locked':  f"L55_donor_locked_{repo}"  if (repo, 'locked', 55)  in donors else 'MISSING',
                'L43_success': f"L43_donor_success_{repo}" if (repo, 'success', 43) in donors else 'MISSING',
                'random':       'L55_random_direction_norm_matched',
            },
        }
    (OUT_DIR / "exp_b_wandering_assignment.json").write_text(json.dumps(wandering_donors, indent=2))
    print(f"\n  Saved per-WANDERING mapping to {OUT_DIR}/exp_b_wandering_assignment.json")
    missing = sum(1 for w in wandering_donors.values()
                  for v in w['donors_available'].values() if v == 'MISSING')
    print(f"  Missing donors: {missing} / {4*len(wandering_donors)}")


if __name__ == "__main__":
    main()
