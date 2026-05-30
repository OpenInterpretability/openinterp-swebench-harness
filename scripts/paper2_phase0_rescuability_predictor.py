#!/usr/bin/env python3
"""
Phase 0 (C) — Rescuability predictor. PRE-REGISTERED gate for the closed-loop (A) experiment.

Question: can the paper #3 pre-intervention mechanistic signature predict WHICH WANDERING
instances are causally rescuable by L11 SUCCESS-donor injection?

PRE-REGISTERED (committed before fitting):
- Label: rescued in {0,1} — the 6 instances that emitted finish_tool at EITHER α (0.70 or 1.15)
  in the L11 Phase 1 experiment, vs 14 resistant.
- Features (mechanistically motivated, paper #3; NO search over 60): L11_drift_first_last,
  L11_cosine_consec_late, tool_entropy_last10, L43_cosine_consec_late. All from the ORIGINAL
  no-hook Phase 6 trajectory (anti-leakage: nothing from the intervention runs).
- Primary: LOO-AUC + 1000-permutation null.
- GATE: GO to A iff LOO-AUC >= 0.70 AND permutation p < 0.10. Else STOP, publish C as clean null.
- Report whatever comes out. No tuning to hit 0.70.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score

CSV = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out/features_n99.csv")

# Rescued-instance fragments (from L11 Phase 1 per-instance windows; rescue at EITHER alpha)
RESCUED_FRAGMENTS = [
    "6fdcf11972062c540b7a787e4",   # qutebrowser-2dd896 — rescue BOTH alpha
    "0d2afd58f3d0e34af21cee7d8a",  # qutebrowser-0d2afd58 — rescue @1.15
    "865f5fb549694d969f0a8e49b",   # openlibrary-09865f5f — rescue @0.70
    "e7de19211e71b29b2f2ba3b1d",   # openlibrary-5de7de — rescue @0.70
    "6b722a10f822171501d027cad",   # openlibrary-7f6b722 — rescue @0.70
    "3bc963930dc85e0f4ca359674",   # ansible-3bc963 — rescue @0.70
]
FEATURES = ["L11_drift_first_last", "L11_cosine_consec_late", "tool_entropy_last10", "L43_cosine_consec_late"]

df = pd.read_csv(CSV)
w = df[df.sub_class == "wandering"].copy().reset_index(drop=True)
assert len(w) == 20, f"expected 20 WANDERING, got {len(w)}"

# Build rescued label by substring match
def is_rescued(iid):
    return int(any(frag in iid for frag in RESCUED_FRAGMENTS))
w["rescued"] = w["iid"].apply(is_rescued)
n_resc = int(w["rescued"].sum())
print(f"=== Rescue-label matching ===")
print(f"Matched rescued: {n_resc}/20 (expected 6)")
matched_frags = [frag for frag in RESCUED_FRAGMENTS if any(frag in i for i in w["iid"])]
print(f"Fragments that matched: {len(matched_frags)}/6")
if n_resc != 6:
    print("WARNING: rescue count != 6 — check fragment matching")
    for frag in RESCUED_FRAGMENTS:
        hits = [i[-50:] for i in w["iid"] if frag in i]
        print(f"  {frag}: {hits}")

X = w[FEATURES].values
# impute any NaN with column median (defensive)
col_med = np.nanmedian(X, axis=0)
inds = np.where(np.isnan(X)); X[inds] = np.take(col_med, inds[1])
y = w["rescued"].values

print(f"\n=== Per-class feature medians (rescued vs resistant) ===")
for f in FEATURES:
    r = w[w.rescued==1][f].median(); nr = w[w.rescued==0][f].median()
    print(f"  {f:28s}  rescued={r:.4f}  resistant={nr:.4f}  Δ={r-nr:+.4f}")

# Univariate AUC per feature (robust, low-overfit)
print(f"\n=== Univariate AUC per feature (single-feature discrimination) ===")
for i, f in enumerate(FEATURES):
    xi = X[:, i]
    # AUC of feature vs label (try both directions, report max)
    auc = roc_auc_score(y, xi); auc = max(auc, 1-auc)
    print(f"  {f:28s}  AUC={auc:.3f}")

# Multivariate LOO-AUC
def loo_auc(Xm, ym):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(penalty="l2", C=1.0, max_iter=2000))])
    yp = cross_val_predict(pipe, Xm, ym, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    return roc_auc_score(ym, yp)

obs = loo_auc(X, y)
print(f"\n=== Multivariate LOO-AUC (4 features) ===")
print(f"  Observed LOO-AUC = {obs:.3f}")

# Permutation null
rng = np.random.RandomState(42)
null = []
for _ in range(1000):
    ys = rng.permutation(y)
    try: null.append(loo_auc(X, ys))
    except Exception: pass
null = np.array(null)
p = (1 + np.sum(null >= obs)) / (1 + len(null))
z = (obs - null.mean()) / max(null.std(), 1e-9)
print(f"  Permutation null: mean={null.mean():.3f} std={null.std():.3f}")
print(f"  p-value (one-sided) = {p:.4f}, z = {z:+.2f}")

# PRE-REGISTERED GATE
print(f"\n{'='*55}")
print(f"PRE-REGISTERED GATE: GO iff LOO-AUC >= 0.70 AND p < 0.10")
print(f"  LOO-AUC = {obs:.3f} ({'PASS' if obs>=0.70 else 'FAIL'} >= 0.70)")
print(f"  p       = {p:.4f} ({'PASS' if p<0.10 else 'FAIL'} < 0.10)")
GO = obs >= 0.70 and p < 0.10
print(f"\n  >>> VERDICT: {'GO — proceed to closed-loop experiment A' if GO else 'STOP — publish C as clean null (rescuability NOT signature-legible)'}")
print(f"{'='*55}")

import json
out = {"n_wandering":20, "n_rescued":n_resc, "features":FEATURES,
       "univariate_auc":{f: float(max(roc_auc_score(y,X[:,i]),1-roc_auc_score(y,X[:,i]))) for i,f in enumerate(FEATURES)},
       "loo_auc":float(obs), "perm_p":float(p), "perm_z":float(z),
       "null_mean":float(null.mean()), "null_std":float(null.std()),
       "gate_go": bool(GO)}
Path("scripts/inflection_turn_out/paper2_phase0_rescuability.json").write_text(json.dumps(out, indent=2))
print(f"\nSaved scripts/inflection_turn_out/paper2_phase0_rescuability.json")
