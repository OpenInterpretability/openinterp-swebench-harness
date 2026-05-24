"""
Trajectory Probe Horizon — analyze how probe AUROC varies along trajectory turns.

Novel angle (per prior art search 2026-05-23): existing work on activation-trajectory
features (arXiv 2604.28129 'adversarial restlessness') uses aggregate scalars per
trajectory. This script does per-turn AUROC across trajectories using probe weights
already trained on independent data — measures EARLIEST detection horizon.

Question: at what turn does each probe first separate success vs failure
trajectories at AUROC > 0.65? Does the curve plateau, rise, or peak-then-drop?

Data: 99 SWE-bench Pro Qwen3.6-27B trajectories from Phase 6.
Probes: agent-probe-guard v0.1 (capability L43 pre_tool, thinking L55 think_end).
Labels: finish_reason == 'finish_tool' (n=40) vs 'max_turns' (n=59).
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import safetensors.torch as st
from sklearn.metrics import roc_auc_score

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
PROBES_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/artifacts/agent_probe_guard_qwen36_27b")
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/trajectory_probe_horizon_out")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------
PROBES = {
    "capability_L43_pre_tool": {
        "path": PROBES_DIR / "probe_L43_pre_tool.joblib",
        "layer": 43,
        "position": "pre_tool",
    },
    "thinking_L55_think_end": {
        "path": PROBES_DIR / "probe_L55_thinking.joblib",
        "layer": 55,
        "position": "think_end",
    },
}


def load_probe(probe_meta: dict):
    obj = joblib.load(probe_meta["path"])
    return obj


# ---------------------------------------------------------------------------
# Capture parsing
# ---------------------------------------------------------------------------
TENSOR_NAME_RE = re.compile(r"t(\d+)_(\w+?)_p(\d+)_L(\d+)$")


def parse_tensors_by_turn(safetensors_path: Path) -> dict:
    """Return {turn_idx: {(position, layer): tensor}}."""
    tensors = st.load_file(str(safetensors_path))
    by_turn: dict[int, dict] = defaultdict(dict)
    for name, t in tensors.items():
        m = TENSOR_NAME_RE.match(name)
        if not m:
            continue
        turn = int(m.group(1))
        position = m.group(2)
        layer = int(m.group(4))
        # Multiple tokens per turn-position-layer; keep first encountered (last_token, fired during gen).
        # Better: average or pick by token_pos. For first pass, use mean across tokens at same (turn,pos,layer).
        key = (position, layer)
        if key not in by_turn[turn]:
            by_turn[turn][key] = [t.float().numpy()]
        else:
            by_turn[turn][key].append(t.float().numpy())

    # Average within each (turn, pos, layer)
    out: dict[int, dict] = {}
    for turn, posmap in by_turn.items():
        out[turn] = {key: np.mean(np.stack(vs, axis=0), axis=0) for key, vs in posmap.items()}
    return out


def probe_score(probe_obj, residual: np.ndarray) -> float:
    """Apply probe to a single residual vector → probability of positive class.

    agent-probe-guard format: dict with keys ['probe' (LogisticRegression),
    'scaler' (StandardScaler), 'dims' (list of feature indices), 'layer' (int)].
    Pipeline: extract dims → scale → classify.
    """
    if isinstance(probe_obj, dict) and "probe" in probe_obj and "scaler" in probe_obj and "dims" in probe_obj:
        dims = probe_obj["dims"]
        scaler = probe_obj["scaler"]
        clf = probe_obj["probe"]
        feats = residual[list(dims)].reshape(1, -1)
        scaled = scaler.transform(feats)
        prob = clf.predict_proba(scaled)[0]
        # classes_ = [0, 1]; return prob of class 1 (positive)
        return float(prob[1] if len(prob) == 2 else prob[0])
    # Fallback: raw sklearn classifier on whole residual
    if hasattr(probe_obj, "predict_proba"):
        prob = probe_obj.predict_proba(residual.reshape(1, -1))[0]
        return float(prob[1] if len(prob) == 2 else prob[0])
    if isinstance(probe_obj, dict) and ("coef" in probe_obj or "coef_" in probe_obj):
        coef = probe_obj.get("coef", probe_obj.get("coef_")).flatten()
        intercept = float(probe_obj.get("intercept", probe_obj.get("intercept_", 0.0)))
        logit = float(np.dot(coef, residual) + intercept)
        return 1.0 / (1.0 + np.exp(-logit))
    raise ValueError(f"Unknown probe format: {type(probe_obj)}, keys: {list(probe_obj.keys()) if isinstance(probe_obj,dict) else 'n/a'}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run():
    results = json.load(open(PHASE6 / "phase6_results.json"))
    captures_dir = PHASE6 / "captures"

    # Build label vector aligned with trajectory IDs
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
    n_success = sum(it["label"] for it in items)
    n_fail = len(items) - n_success
    print(f"Loaded {len(items)} trajectories: {n_success} success / {n_fail} fail")

    # Load probes
    loaded_probes = {}
    for name, meta in PROBES.items():
        try:
            probe = load_probe(meta)
            loaded_probes[name] = (probe, meta)
            kind = type(probe).__name__ if not isinstance(probe, dict) else f"dict({list(probe.keys())[:5]})"
            print(f"  loaded probe {name}: {kind}")
        except Exception as e:
            print(f"  FAILED to load {name}: {e}")

    # For each probe + each trajectory + each turn → score
    # all_scores[probe_name][iid][turn] = score
    all_scores: dict = defaultdict(lambda: defaultdict(dict))

    for ix, it in enumerate(items):
        if ix % 10 == 0:
            print(f"  trajectory {ix+1}/{len(items)}: {it['iid'][:60]}...")
        try:
            by_turn = parse_tensors_by_turn(it["capture_path"])
        except Exception as e:
            print(f"    parse error {e}")
            continue
        for probe_name, (probe, meta) in loaded_probes.items():
            key = (meta["position"], meta["layer"])
            for turn, posmap in by_turn.items():
                if key in posmap:
                    try:
                        s = probe_score(probe, posmap[key])
                        all_scores[probe_name][it["iid"]][turn] = s
                    except Exception as e:
                        if ix == 0:
                            print(f"    probe scoring err {probe_name} t{turn}: {e}")

    # Compute per-turn AUROC
    horizon = {}
    for probe_name in loaded_probes:
        scores = all_scores[probe_name]
        # find max turn present
        max_turn = 0
        for iid_scores in scores.values():
            if iid_scores:
                max_turn = max(max_turn, max(iid_scores.keys()))

        per_turn_auroc = []
        for turn in range(max_turn + 1):
            ys, ss = [], []
            for it in items:
                iid = it["iid"]
                if iid in scores and turn in scores[iid]:
                    ys.append(it["label"])
                    ss.append(scores[iid][turn])
            if len(ys) >= 10 and len(set(ys)) == 2:
                try:
                    a = roc_auc_score(ys, ss)
                    per_turn_auroc.append({
                        "turn": turn,
                        "n": len(ys),
                        "n_success": int(sum(ys)),
                        "auroc": float(a),
                    })
                except Exception:
                    pass
        horizon[probe_name] = per_turn_auroc
        print(f"\n=== {probe_name} ===")
        print(f"{'turn':>4} {'n':>4} {'n_succ':>7} {'auroc':>7}")
        for row in per_turn_auroc[:35]:
            mark = " ⭐" if row["auroc"] >= 0.65 else ""
            print(f"{row['turn']:>4} {row['n']:>4} {row['n_success']:>7} {row['auroc']:>7.3f}{mark}")

    # Save
    out_path = OUT_DIR / "horizon_results.json"
    out_path.write_text(json.dumps({
        "n_trajectories": len(items),
        "n_success": n_success,
        "n_fail": n_fail,
        "horizon": horizon,
    }, indent=2))
    print(f"\n=== Saved to {out_path} ===")


if __name__ == "__main__":
    run()
