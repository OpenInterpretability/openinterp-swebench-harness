#!/usr/bin/env python3
"""
Context-Rot Stage 1a — does context-rot have a RAW residual-geometry fingerprint?

Pre-reg: paper/context_rot/PREREG_context_rot_fingerprint.md

Runs on the EXISTING Phase-6 N=99 captures (no SAE, no download, CPU, ~minutes).
For each trajectory it builds one residual vector per turn per layer (mean over
that turn's capture positions) and computes four raw geometry metrics vs turn:

  M1 norm      = ||v_t||
  M2 velocity  = 1 - cos(v_t, v_{t-1})           (-> 0 = state freezing)
  M3 drift     = cos(v_t, v_0)                    (how far from the start region)
  M4 eff_dim   = windowed participation ratio     (collapse to low-D subspace)

Then the two pre-registered tests + the length control + the shuffle-time null,
and prints the GO / NO-GO gate. Honest by construction: a metric that only
"degrades" because failing runs are longer is reported as a NULL.

  python3 scripts/context_rot_stage1.py --smoke      # synthetic self-test FIRST
  python3 scripts/context_rot_stage1.py               # real run on Phase-6
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.stats import mannwhitneyu, linregress, binomtest
except Exception:
    print("need scipy:  pip install scipy", file=sys.stderr); raise

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/"
              "Meu Drive/openinterp_runs/swebench_v6_phase6")
LABELS_CSV = Path(__file__).resolve().parent / "inflection_turn_out" / "features_n99.csv"
OUT = Path(__file__).resolve().parent / "context_rot_out"; OUT.mkdir(exist_ok=True)
LAYERS = [11, 23, 31, 43, 55]
KEY_RE = re.compile(r"t(\d+)_(\w+?)_p(\d+)_L(\d+)$")
WIN = 5          # window for eff_dim
EARLY_MIN_PER_CLASS = 5   # need >=5 of each class live at a turn to contrast
SUCCESS, FAIL = "success", "wandering"   # primary contrast (also report vs locked+wandering)

# ---------------------------------------------------------------- metrics
def _unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-9 else v

def metrics_for_traj(turn_vecs: dict) -> dict:
    """turn_vecs: {turn:int -> vec[d]} for one layer. Returns {metric: {turn: val}}."""
    turns = sorted(turn_vecs)
    if len(turns) < 3: return {}
    V = np.stack([turn_vecs[t] for t in turns])          # [T, d]
    v0 = _unit(V[0])
    out = {"norm": {}, "velocity": {}, "drift": {}, "eff_dim": {}}
    for i, t in enumerate(turns):
        out["norm"][t] = float(np.linalg.norm(V[i]))
        out["drift"][t] = float(np.dot(_unit(V[i]), v0))
        if i > 0:
            out["velocity"][t] = float(1.0 - np.dot(_unit(V[i]), _unit(V[i - 1])))
        lo = max(0, i - WIN + 1)
        if i - lo >= 2:                                   # need >=3 vecs for PR
            W = V[lo:i + 1]
            Wc = W - W.mean(0)
            G = Wc @ Wc.T                                 # Gram (small) — same nonzero eig as cov
            ev = np.clip(np.linalg.eigvalsh(G), 0, None)
            s = ev.sum()
            out["eff_dim"][t] = float((s * s) / (np.square(ev).sum() + 1e-12)) if s > 1e-9 else 0.0
    return out

# ---------------------------------------------------------------- data
def load_labels():
    import csv
    return {r["iid"]: r["sub_class"] for r in csv.DictReader(open(LABELS_CSV))}

def load_turns(path: Path, layer: int) -> dict:
    import safetensors.torch as st
    by_turn = defaultdict(list)
    for name, t in st.load_file(str(path)).items():
        m = KEY_RE.match(name)
        if m and int(m.group(4)) == layer:
            by_turn[int(m.group(1))].append(t.float().numpy())
    return {turn: np.mean(np.stack(vs), 0) for turn, vs in by_turn.items()}

def build_real(layer: int):
    labels = load_labels()
    caps = sorted(PHASE6.glob("captures/*.safetensors"))
    trajs = []  # list of (iid, label, {metric:{turn:val}}, n_turns)
    for c in caps:
        iid = c.stem
        lab = labels.get(iid)
        if lab is None:  # filename may carry extra suffix; match by prefix
            lab = next((labels[k] for k in labels if iid.startswith(k) or k.startswith(iid)), None)
        if lab is None: continue
        tv = load_turns(c, layer)
        m = metrics_for_traj(tv)
        if m: trajs.append((iid, lab, m, max(tv) + 1))
    return trajs

# ---------------------------------------------------------------- synthetic (smoke)
def build_synth(layer, planted=True, seed=0):
    """planted: failures degrade by ABSOLUTE turn (class-tied, length-independent).
       null:    both classes identical process, differing ONLY in length (the trap)."""
    rng = np.random.default_rng(seed); d = 256; trajs = []
    sub = _unit(rng.standard_normal(d))               # shared collapse direction
    def traj(T, degrade):
        base = rng.standard_normal(d); vecs = {}
        for t in range(T):
            v = base + 0.5 * rng.standard_normal(d)
            if degrade:
                a = min(1.0, t / 25.0)                 # absolute-turn degradation (NOT t/T): early, class-tied
                v = (1 - 0.7 * a) * v + 0.7 * a * (3 * sub)
            vecs[t] = v
        return vecs
    for i in range(40):
        T = int(rng.integers(12, 22)); trajs.append((f"s{i}", "success", metrics_for_traj(traj(T, False)), T))
    for i in range(20):
        T = int(rng.integers(30, 50)); trajs.append((f"w{i}", "wandering", metrics_for_traj(traj(T, planted)), T))
    return trajs

# ---------------------------------------------------------------- tests
def per_turn_series(trajs, metric):
    by_turn = {SUCCESS: defaultdict(list), FAIL: defaultdict(list)}
    slopes = {SUCCESS: [], FAIL: []}
    for iid, lab, m, n in trajs:
        if lab not in by_turn or metric not in m or len(m[metric]) < 3: continue
        ts = sorted(m[metric]); ys = [m[metric][t] for t in ts]
        for t, y in zip(ts, ys): by_turn[lab][t].append(y)
        slopes[lab].append(linregress(ts, ys).slope)
    return by_turn, slopes

def _early_window(trajs, metric):
    """First ~10 turns where BOTH classes have >= EARLY_MIN live trajectories."""
    cnt = {SUCCESS: defaultdict(int), FAIL: defaultdict(int)}
    for _, lab, m, _ in trajs:
        if lab in cnt and metric in m:
            for t in m[metric]: cnt[lab][t] += 1
    overlap = sorted(t for t in set(cnt[SUCCESS]) & set(cnt[FAIL])
                     if cnt[SUCCESS][t] >= EARLY_MIN_PER_CLASS and cnt[FAIL][t] >= EARLY_MIN_PER_CLASS)
    if not overlap: return None
    return overlap[0], overlap[min(len(overlap) - 1, 9)]

def analyze(trajs, label=""):
    res = {"label": label, "n_traj": len(trajs),
           "n_by_class": {k: sum(1 for _, l, _, _ in trajs if l == k) for k in (SUCCESS, FAIL, "locked")}}
    metric_report = {}
    for metric in ["norm", "velocity", "drift", "eff_dim"]:
        ew = _early_window(trajs, metric)
        if ew is None: metric_report[metric] = {"skip": "no early overlap region"}; continue
        t_lo, t_hi = ew
        # ONE summary per TRAJECTORY (independent samples — no pseudo-replication)
        rows, s_means, f_means = [], [], []
        for iid, lab, m, n in trajs:
            if lab not in (SUCCESS, FAIL) or metric not in m: continue
            vals = [v for t, v in m[metric].items() if t_lo <= t <= t_hi]
            if not vals: continue
            em = float(np.mean(vals)); rows.append([em, float(n), 1.0 if lab == FAIL else 0.0])
            (f_means if lab == FAIL else s_means).append(em)
        if len(s_means) < 5 or len(f_means) < 5:
            metric_report[metric] = {"skip": "too few per-class in early window"}; continue
        # P2: per-trajectory early-window success-vs-failure divergence
        div_p = float(mannwhitneyu(s_means, f_means, alternative="two-sided").pvalue)
        # length control: partial corr (early_mean vs is_fail | total length)
        X = np.array(rows); L = np.c_[np.ones(len(X)), X[:, 1]]
        def _resid(y): b, *_ = np.linalg.lstsq(L, y, rcond=None); return y - L @ b
        rm, ro = _resid(X[:, 0]), _resid(X[:, 2])
        partial_r = float(np.corrcoef(rm, ro)[0, 1]) if np.std(rm) > 1e-9 and np.std(ro) > 1e-9 else 0.0
        # descriptive trend (failure class within-traj slope sign-consistency) — reported, NOT gated
        _, slopes = per_turn_series(trajs, metric); fs = np.array(slopes[FAIL])
        trend_p = float(binomtest(int(max((fs > 0).sum(), (fs <= 0).sum())), len(fs), 0.5).pvalue) if len(fs) >= 5 else None
        # prereg §4 GO = P1 (monotone trend) AND P2 (early success-vs-fail divergence surviving length control)
        gate = bool(trend_p is not None and trend_p < 0.05 and div_p < 0.05 and abs(partial_r) > 0.15)
        metric_report[metric] = {
            "early_window": [int(t_lo), int(t_hi)], "n_success": len(s_means), "n_fail": len(f_means),
            "success_mean": float(np.mean(s_means)), "fail_mean": float(np.mean(f_means)),
            "early_divergence_MW_p": div_p, "partial_r_given_length": partial_r,
            "trend_signconsistency_p": trend_p, "GATE_pass": gate,
        }
    gos = [m for m, r in metric_report.items() if isinstance(r, dict) and r.get("GATE_pass")]
    res["metrics"] = metric_report
    res["GATE"] = {"verdict": "GO" if gos else "NO-GO (honest negative)", "passing_metrics": gos,
                   "note": "GO (prereg §4) = P1 within-traj monotone trend (sign-consistency p<0.05) AND P2 per-TRAJECTORY "
                           "early-window success-vs-failure divergence (MW p<0.05) SURVIVING the length control "
                           "(|partial r| > 0.15). Trend-without-discrimination and pure length artifacts are NULL. "
                           "NOTE: uncorrected; cross-check vs Bonferroni/FDR over the full metric×layer grid before headlining."}
    return res

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="synthetic self-test (planted signal + planted null)")
    ap.add_argument("--layer", type=int, default=43)
    args = ap.parse_args()

    if args.smoke:
        print("=== SMOKE: planted-signal synthetic (gate MUST fire) ===")
        r1 = analyze(build_synth(args.layer, planted=True), "synthetic-planted")
        print(json.dumps(r1["GATE"], indent=2))
        print("=== SMOKE: planted-null synthetic (gate MUST NOT fire) ===")
        r0 = analyze(build_synth(args.layer, planted=False, seed=1), "synthetic-null")
        print(json.dumps(r0["GATE"], indent=2))
        ok = r1["GATE"]["verdict"].startswith("GO") and r0["GATE"]["verdict"].startswith("NO-GO")
        print(f"\nSMOKE {'PASS' if ok else 'FAIL'} — pipeline {'detects planted signal & rejects null' if ok else 'BROKEN'}")
        (OUT / "smoke.json").write_text(json.dumps({"planted": r1, "null": r0}, indent=2))
        sys.exit(0 if ok else 1)

    if not PHASE6.exists():
        print(f"Phase-6 captures not found at {PHASE6}\nMount the Drive or edit PHASE6.", file=sys.stderr); sys.exit(2)

    all_layers = {}
    for L in LAYERS:
        print(f"--- layer L{L} ---")
        trajs = build_real(L)
        rep = analyze(trajs, f"phase6_L{L}")
        print(json.dumps(rep["GATE"], indent=2))
        all_layers[f"L{L}"] = rep
    (OUT / "stage1_metrics.json").write_text(json.dumps(all_layers, indent=2))
    go_layers = [k for k, v in all_layers.items() if v["GATE"]["verdict"].startswith("GO")]
    print("\n==== STAGE-1 VERDICT ====")
    print("GO layers:", go_layers or "NONE → honest negative (context-rot has no raw-geometry fingerprint at this granularity)")
    print(f"full metrics -> {OUT/'stage1_metrics.json'}")

if __name__ == "__main__":
    main()
