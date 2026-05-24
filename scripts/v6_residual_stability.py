"""
v6 — Residual action-state divergence as Tier 3 signal candidate.

Per layer L, per turn t, compute ||res_L(t) - res_L(t-1)||_2 / ||res_L(t-1)||_2
(relative L2 change). Aggregate per trajectory: late-half mean.

Hypothesis 1 (frozen-state): WANDERING agents have LOW residual change (state
not evolving — agent stuck in a fixed belief about completion).
Hypothesis 2 (oscillation): WANDERING agents have HIGH residual change (flipping
between done/not-done states).

Test on all 5 layers (L11/23/31/43/55). Report which (if any) discriminate
at Tier 3 (FP <= 5%).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import safetensors.torch as st
from scipy.stats import mannwhitneyu

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")
INFLECTION = json.load(open(OUT_DIR / "inflection_results.json"))

LAYERS = [11, 23, 31, 43, 55]
POSITION = "pre_tool"
TENSOR_NAME_RE = re.compile(r"t(\d+)_(\w+?)_p(\d+)_L(\d+)$")


def load_per_turn_residuals(path, layer):
    tensors = st.load_file(str(path))
    by_turn = defaultdict(list)
    for name, t in tensors.items():
        m = TENSOR_NAME_RE.match(name)
        if m and m.group(2) == POSITION and int(m.group(4)) == layer:
            by_turn[int(m.group(1))].append(t.float().numpy())
    return {turn: np.mean(np.stack(vs), axis=0) for turn, vs in by_turn.items()}


def compute_stability_metrics(turn_residuals):
    """Returns dict of stability metrics for one trajectory at one layer."""
    turns = sorted(turn_residuals.keys())
    if len(turns) < 2:
        return None
    rel_changes = []
    cosine_sims = []
    abs_changes = []
    for i in range(1, len(turns)):
        prev = turn_residuals[turns[i-1]]
        curr = turn_residuals[turns[i]]
        diff = curr - prev
        prev_norm = np.linalg.norm(prev)
        curr_norm = np.linalg.norm(curr)
        if prev_norm > 0 and curr_norm > 0:
            rel_changes.append(float(np.linalg.norm(diff) / prev_norm))
            cosine_sims.append(float(np.dot(prev, curr) / (prev_norm * curr_norm)))
            abs_changes.append(float(np.linalg.norm(diff)))
    if not rel_changes:
        return None
    half = len(rel_changes) // 2
    return {
        "rel_change_mean": float(np.mean(rel_changes)),
        "rel_change_late": float(np.mean(rel_changes[half:])),
        "cosine_sim_mean": float(np.mean(cosine_sims)),
        "cosine_sim_late": float(np.mean(cosine_sims[half:])),
        "abs_change_late": float(np.mean(abs_changes[half:])),
        "n_turns": len(turns),
    }


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
        if not local.exists(): continue
        base_data.append({"iid": iid, "local": local, "sub_class": sub_class[iid]})

    print(f"Loaded {len(base_data)} trajectories")

    # Per layer, compute stability per trajectory
    per_layer_traj = {L: [] for L in LAYERS}
    for L in LAYERS:
        print(f"\n=== L{L} ===")
        for t in base_data:
            try:
                tr = load_per_turn_residuals(t["local"], L)
            except Exception:
                tr = None
            metrics = compute_stability_metrics(tr) if tr else None
            if metrics:
                metrics["iid"] = t["iid"]
                metrics["sub_class"] = t["sub_class"]
                per_layer_traj[L].append(metrics)

    by_class_per_layer = {L: defaultdict(list) for L in LAYERS}
    for L in LAYERS:
        for m in per_layer_traj[L]:
            by_class_per_layer[L][m["sub_class"]].append(m)

    # Stats per layer per metric
    print("\n=== Per-layer stability stats: WANDERING vs SUCCESS ===")
    for L in LAYERS:
        print(f"\n  L{L}:")
        for metric in ["rel_change_late", "cosine_sim_late", "abs_change_late"]:
            s = [m[metric] for m in by_class_per_layer[L]["success"]]
            w = [m[metric] for m in by_class_per_layer[L]["wandering"]]
            l = [m[metric] for m in by_class_per_layer[L]["locked"]]
            try:
                u, p = mannwhitneyu(w, s, alternative="two-sided")
            except Exception:
                u, p = -1, 1.0
            print(f"    {metric}: S={np.mean(s):.4f}, L={np.mean(l):.4f}, W={np.mean(w):.4f}, p(W vs S)={p:.4f}")

    # Tier 3 sweep: for each (layer, metric), find best threshold at FP<=5%
    print("\n=== Tier 3 sweep (FP <= 5%) per (layer, metric, direction) ===")
    print(f"{'L':>3} {'metric':>20} {'dir':>4} {'thresh':>10} {'W_rec':>8} {'S_FP':>7} {'L_caught':>10}")
    best_overall = None
    for L in LAYERS:
        for metric in ["rel_change_late", "cosine_sim_late", "abs_change_late"]:
            for direction in [">", "<"]:
                vals = sorted(set([m[metric] for m in per_layer_traj[L]]))
                best = None
                for t in vals:
                    w_pos = sum(1 for m in by_class_per_layer[L]["wandering"]
                                if (m[metric] > t if direction == ">" else m[metric] < t))
                    s_pos = sum(1 for m in by_class_per_layer[L]["success"]
                                if (m[metric] > t if direction == ">" else m[metric] < t))
                    l_pos = sum(1 for m in by_class_per_layer[L]["locked"]
                                if (m[metric] > t if direction == ">" else m[metric] < t))
                    sfp_rate = s_pos / max(1, len(by_class_per_layer[L]["success"]))
                    if sfp_rate <= 0.05:
                        if best is None or w_pos > best["w"]:
                            best = {"w": w_pos, "t": t, "sfp": s_pos, "l": l_pos}
                if best and best["w"] > 0:
                    print(f"{L:>3} {metric:>20} {direction:>4} {best['t']:>10.4f} {best['w']}/{len(by_class_per_layer[L]['wandering'])} ({100*best['w']/len(by_class_per_layer[L]['wandering']):>3.0f}%) {best['sfp']}/{len(by_class_per_layer[L]['success'])} ({100*best['sfp']/len(by_class_per_layer[L]['success']):>3.0f}%) {best['l']}/{len(by_class_per_layer[L]['locked'])}")
                    if best_overall is None or best["w"] > best_overall["w"]:
                        best_overall = {"L": L, "metric": metric, "dir": direction, **best}

    if best_overall:
        print(f"\n=== BEST v6 single-signal: L{best_overall['L']} {best_overall['metric']} {best_overall['dir']} {best_overall['t']:.4f} ===")
        print(f"    WANDERING: {best_overall['w']}/20 ({100*best_overall['w']/20:.0f}%)")
        print(f"    SUCCESS FP: {best_overall['sfp']}/40")
        print(f"    LOCKED caught: {best_overall['l']}/39")

    (OUT_DIR / "v6_residual_stability.json").write_text(json.dumps({
        L: [{k: v for k, v in m.items()} for m in per_layer_traj[L]]
        for L in LAYERS
    }, indent=2, default=float))
    print(f"\nSaved {OUT_DIR}/v6_residual_stability.json")


if __name__ == "__main__":
    main()
