#!/usr/bin/env python3
"""
Context-rot Stage 2 (P3, predictive). Pre-registered in paper/context_rot/PREREG §8.

Decisive question: does early-window VELOCITY-FREEZING (Stage-1 fingerprint) predict
eventual WANDERING with AUROC and/or lead-time >= the cheap probe-free tool-entropy
detector it would replace? If not -> honest negative, STOP (prereg gate).

Reuses the Stage-1 loader (same outcome-blind early window). CPU, existing N=99 captures.
Single-feature AUROCs are fit-free (rank statistic) => no CV/leakage. Differences via
paired bootstrap. Lead-time at matched 5% SUCCESS false-positive rate vs detector_v5_fire_turn.

Output: scripts/context_rot_out/stage2_predictive.json
"""
import sys, json, csv
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import context_rot_stage1 as s1

OUT = Path(__file__).resolve().parent / "context_rot_out" / "stage2_predictive.json"
LAYERS = s1.LAYERS
SUCCESS, FAIL = s1.SUCCESS, s1.FAIL
RNG = np.random.default_rng(20260601)
N_BOOT = 2000
FP_TARGET = 0.05            # matched operating point (Tool-Entropy paper 5% FP)

# ---------------------------------------------------------------- AUROC
def auroc(scores, y):
    """AUROC for label==1 (FAIL); `scores` already oriented so higher => more FAIL."""
    scores = np.asarray(scores, float); y = np.asarray(y, int)
    pos, neg = scores[y == 1], scores[y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(scores)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def boot_ci(scores, y, n=N_BOOT):
    scores = np.asarray(scores, float); y = np.asarray(y, int)
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        b = RNG.choice(idx, len(idx), replace=True)
        if len(set(y[b])) < 2: continue
        vals.append(auroc(scores[b], y[b]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def paired_diff_ci(sA, sB, y, n=N_BOOT):
    """CI of AUROC(A) - AUROC(B), paired bootstrap over the same resampled trajectories."""
    sA = np.asarray(sA, float); sB = np.asarray(sB, float); y = np.asarray(y, int)
    idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = RNG.choice(idx, len(idx), replace=True)
        if len(set(y[b])) < 2: continue
        d.append(auroc(sA[b], y[b]) - auroc(sB[b], y[b]))
    d = np.array(d)
    return float(auroc(sA, y) - auroc(sB, y)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

# ---------------------------------------------------------------- load CSV behavioral features
def load_csv():
    rows = {}
    with open(s1.LABELS_CSV) as f:
        for row in csv.DictReader(f):
            rows[row["iid"]] = row
    return rows

def csv_get(rows, iid, col):
    r = rows.get(iid) or next((rows[k] for k in rows if iid.startswith(k) or k.startswith(iid)), None)
    if r is None: return None
    v = r.get(col, "")
    try: return float(v)
    except Exception: return None

# ---------------------------------------------------------------- main
def main():
    if not s1.PHASE6.exists():
        print(f"Phase-6 not found at {s1.PHASE6}", file=sys.stderr); sys.exit(2)
    csv_rows = load_csv()

    # velocity per layer: early-window mean per trajectory + full per-turn series (for lead-time)
    velo_early = {L: {} for L in LAYERS}     # L -> iid -> early-window mean velocity
    velo_series = {L: {} for L in LAYERS}    # L -> iid -> {turn: velocity}
    lab_of = {}; nturns_of = {}
    for L in LAYERS:
        trajs = s1.build_real(L)
        ew = s1._early_window(trajs, "velocity")
        t_lo, t_hi = ew
        for iid, lab, m, n in trajs:
            if lab not in (SUCCESS, FAIL): continue
            lab_of[iid] = lab; nturns_of[iid] = n
            vser = m.get("velocity", {})
            velo_series[L][iid] = vser
            evals = [v for t, v in vser.items() if t_lo <= t <= t_hi]
            if evals: velo_early[L][iid] = float(np.mean(evals))
        print(f"  L{L}: early window [{t_lo},{t_hi}], {len(velo_early[L])} trajs w/ velocity", file=sys.stderr)

    iids = [i for i in lab_of if lab_of[i] in (SUCCESS, FAIL)]
    y = np.array([1 if lab_of[i] == FAIL else 0 for i in iids])
    print(f"WANDERING vs SUCCESS: n_fail={int(y.sum())} n_succ={int((1-y).sum())}", file=sys.stderr)

    # ---- pre-registered features & directions (score oriented so higher => more FAIL) ----
    def vec_velo(L):   # fail = LOWER velocity -> score = -velocity
        return np.array([-velo_early[L].get(i, np.nan) for i in iids])
    def vec_csv(col, sign):  # sign=-1 if fail=lower (tool-entropy collapse), +1 if fail=higher (length)
        return np.array([sign * (csv_get(csv_rows, i, col) if csv_get(csv_rows, i, col) is not None else np.nan)
                         for i in iids])

    features = {}
    for L in LAYERS:
        features[f"velocity_L{L}"] = vec_velo(L)
    # pooled velocity = mean z-scored early velocity across layers (fail=lower)
    zs = []
    for L in LAYERS:
        v = np.array([velo_early[L].get(i, np.nan) for i in iids])
        mu, sd = np.nanmean(v), np.nanstd(v) + 1e-9
        zs.append((v - mu) / sd)
    features["velocity_pooled"] = -np.nanmean(np.vstack(zs), axis=0)   # fail=lower => negate
    features["tool_entropy_first10"] = vec_csv("tool_entropy_first10", -1)
    features["tool_entropy_last10"]  = vec_csv("tool_entropy_last10", -1)
    features["tool_entropy_full"]    = vec_csv("tool_entropy_full", -1)
    features["probe_score_first"]    = vec_csv("probe_score_first", +1)   # higher probe => more wandering (reported, dir uncertain)
    features["n_turns_lengthfloor"]  = vec_csv("n_turns", +1)            # fail=longer

    # ---- AUROC + CI for each (drop NaNs pairwise) ----
    auc_report = {}
    for name, s in features.items():
        ok = ~np.isnan(s)
        a = auroc(s[ok], y[ok]); lo, hi = boot_ci(s[ok], y[ok])
        auc_report[name] = {"auroc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": int(ok.sum())}

    # ---- paired difference: primary velocity (L31) and pooled vs fair early baseline ----
    diffs = {}
    base = features["tool_entropy_first10"]
    for cand in ["velocity_L31", "velocity_pooled"]:
        s = features[cand]; ok = ~np.isnan(s) & ~np.isnan(base)
        d, lo, hi = paired_diff_ci(s[ok], base[ok], y[ok])
        diffs[f"{cand}_minus_tool_entropy_first10"] = {"delta_auroc": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": int(ok.sum())}

    # ---- lead-time at matched 5% SUCCESS FP (per layer; headline L31) ----
    lead = {}
    for L in LAYERS:
        succ_mins = [min(velo_series[L][i].values()) for i in iids
                     if lab_of[i] == SUCCESS and velo_series[L].get(i)]
        if not succ_mins: continue
        theta = float(np.percentile(succ_mins, FP_TARGET * 100))   # <=5% of success ever below theta
        fp = float(np.mean([m < theta for m in succ_mins]))
        v_alarm, v5_turn = [], []; caught = 0
        fail_iids = [i for i in iids if lab_of[i] == FAIL and velo_series[L].get(i)]
        for i in fail_iids:
            ser = velo_series[L][i]
            fire = sorted(t for t, v in ser.items() if v < theta)
            v5 = csv_get(csv_rows, i, "detector_v5_fire_turn")
            if fire:
                caught += 1
                if v5 is not None and v5 >= 0:
                    v_alarm.append(fire[0]); v5_turn.append(v5)
        w_p = None
        if len(v_alarm) >= 5:
            try: w_p = float(wilcoxon(np.array(v_alarm), np.array(v5_turn)).pvalue)
            except Exception: w_p = None
        lead[f"L{L}"] = {
            "theta": round(theta, 5), "success_FP": round(fp, 3),
            "velocity_catch_rate": f"{caught}/{len(fail_iids)}",
            "median_velocity_alarm_turn": (round(float(np.median(v_alarm)), 1) if v_alarm else None),
            "median_v5_fire_turn": (round(float(np.median(v5_turn)), 1) if v5_turn else None),
            "paired_wilcoxon_p": w_p, "n_both_caught": len(v_alarm),
        }

    # ---- GATE (prereg §8) ----
    a_v31 = auc_report["velocity_L31"]["auroc"]; a_vp = auc_report["velocity_pooled"]["auroc"]
    a_te1 = auc_report["tool_entropy_first10"]["auroc"]; a_len = auc_report["n_turns_lengthfloor"]["auroc"]
    best_v = max(a_v31, a_vp)
    diff31 = diffs["velocity_L31_minus_tool_entropy_first10"]
    auc_tie_or_win = bool(diff31["ci95"][1] >= 0)    # CI of (velo - first10) includes 0 or positive
    # lead-time win at L31: velocity median alarm <= v5 median, with a paired test available
    l31 = lead.get("L31", {})
    lead_win = bool(l31.get("median_velocity_alarm_turn") is not None and l31.get("median_v5_fire_turn") is not None
                    and l31["median_velocity_alarm_turn"] <= l31["median_v5_fire_turn"])
    length_ok = bool(best_v > a_len)
    go = bool(length_ok and (auc_tie_or_win or lead_win))
    gate = {
        "verdict": "GO" if go else "NO-GO (honest negative)",
        "best_velocity_auroc": best_v, "tool_entropy_first10_auroc": a_te1,
        "tool_entropy_last10_auroc": auc_report["tool_entropy_last10"]["auroc"],
        "length_floor_auroc": a_len, "clears_length_floor": length_ok,
        "auc_tie_or_win_vs_first10": auc_tie_or_win, "lead_time_win_L31": lead_win,
        "rule": "GO iff clears length floor AND (early-AUROC ties/beats tool_entropy_first10 OR fires no later than v5 at 5% FP). "
                "NO-GO = freezing real but adds nothing over the cheap behavioral detector -> stop, publish negative.",
    }

    res = {"task": "WANDERING vs SUCCESS", "n_fail": int(y.sum()), "n_succ": int((1 - y).sum()),
           "auroc": auc_report, "paired_diffs": diffs, "lead_time": lead, "GATE": gate}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(gate, indent=2))
    print("AUROC:", {k: v["auroc"] for k, v in auc_report.items()})
    print("full ->", OUT)

if __name__ == "__main__":
    main()
