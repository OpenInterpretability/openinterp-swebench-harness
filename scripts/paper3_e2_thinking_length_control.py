#!/usr/bin/env python3
"""
Paper #3 E2 — Confound control for L11_drift ↔ text_thinking_length_final (ρ=0.63).

Question: Is L11_drift_first_last predictive of WANDERING because of mechanism,
or because it's a proxy for thinking_length (which differs across sub-classes)?

Three tests:
  (A) Partial correlation: cor(L11_drift, sub_class | thinking_length).
      Use Spearman partial via residualization.
  (B) Residualize ALL features by thinking_length, refit L1 logistic, check
      if L11_drift retains high selection_freq.
  (C) Match WANDERING/LOCKED on thinking_length (nearest-neighbor pairs),
      run paired Wilcoxon on L11_drift differences.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

CSV = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out/features_n99.csv")
OUT = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out/paper3_e2_thinking_length_control.json")

df = pd.read_csv(CSV)
print(f"Loaded N={len(df)}, sub_class counts: {df.sub_class.value_counts().to_dict()}")
print()

# ============================================================
# (A) PARTIAL CORRELATION via residualization
# ============================================================
print("=" * 70)
print("(A) Partial correlation: L11_drift vs sub_class | thinking_length")
print("=" * 70)

x = df["L11_drift_first_last"].values
z = df["text_thinking_length_final"].values
y_w = (df["sub_class"] == "wandering").astype(int).values
y_l = (df["sub_class"] == "locked").astype(int).values

# Residualize L11_drift on thinking_length (regression residuals)
def residualize(target, covariate):
    # target = a + b*covariate + residual
    valid = ~(np.isnan(target) | np.isnan(covariate))
    coef = np.polyfit(covariate[valid], target[valid], 1)
    predicted = coef[0] * covariate + coef[1]
    return target - predicted

x_res = residualize(x, z)

# Raw correlations
rho_raw, p_raw = stats.spearmanr(x, y_w)
rho_partial, p_partial = stats.spearmanr(x_res, y_w)
print(f"WANDERING vs L11_drift (raw):     ρ={rho_raw:+.3f}, p={p_raw:.4f}")
print(f"WANDERING vs L11_drift (partial): ρ={rho_partial:+.3f}, p={p_partial:.4f}")
print(f"  → {'L11_drift HAS independent signal' if abs(rho_partial) > 0.15 and p_partial < 0.05 else 'L11_drift mostly proxy of thinking_length'}")

rho_raw_l, p_raw_l = stats.spearmanr(x, y_l)
rho_partial_l, p_partial_l = stats.spearmanr(x_res, y_l)
print(f"LOCKED vs L11_drift (raw):        ρ={rho_raw_l:+.3f}, p={p_raw_l:.4f}")
print(f"LOCKED vs L11_drift (partial):    ρ={rho_partial_l:+.3f}, p={p_partial_l:.4f}")

# ============================================================
# (B) Residualize ALL features by thinking_length, refit L1
# ============================================================
print()
print("=" * 70)
print("(B) Residualize ALL features by thinking_length, refit L1 multinomial")
print("=" * 70)

LEAKY = [c for c in df.columns if any(c.startswith(p) for p in [
    "probe_score_", "probe_first_", "detector_v4_", "n_turns", "patch_n_bytes"
])]
ALL_NAN = ["detector_v1_fire_turn", "detector_v5_fire_turn"]
ZERO_VAR = ["probe_first_threshold_cross_turn"]

feat_df = df.drop(columns=["iid", "sub_class"]).select_dtypes(include=[np.number])
feat_df = feat_df.drop(columns=[c for c in feat_df.columns if c in LEAKY or c in ALL_NAN or c in ZERO_VAR])
print(f"Pre-residualize features: {len(feat_df.columns)}")

# Drop the covariate itself so we don't residualize it against itself
covariate_name = "text_thinking_length_final"
feat_no_cov = feat_df.drop(columns=[covariate_name])
covariate = feat_df[covariate_name].values

# Residualize each feature
imp = SimpleImputer(strategy="median")
covariate_imp = imp.fit_transform(covariate.reshape(-1, 1)).flatten()

X_res = np.zeros((len(feat_no_cov), len(feat_no_cov.columns)))
for i, col in enumerate(feat_no_cov.columns):
    target = imp.fit_transform(feat_no_cov[col].values.reshape(-1, 1)).flatten()
    coef = np.polyfit(covariate_imp, target, 1)
    predicted = coef[0] * covariate_imp + coef[1]
    X_res[:, i] = target - predicted

X_res = StandardScaler().fit_transform(X_res)
y = df.sub_class.map({"locked": 0, "success": 1, "wandering": 2}).values

# L1 multinomial logistic
clf = LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=5000, random_state=42)
clf.fit(X_res, y)

# Top L1 coefs per class
fnames = list(feat_no_cov.columns)
print(f"\nL1 coefficients post-residualization (top-8 by |coef| per class):")
for cls_idx, cls_name in [(0, "LOCKED"), (1, "SUCCESS"), (2, "WANDERING")]:
    coefs = clf.coef_[cls_idx]
    ranked = sorted(zip(fnames, coefs), key=lambda x: -abs(x[1]))
    print(f"\n  {cls_name}:")
    for name, c in ranked[:8]:
        if abs(c) > 1e-6:
            print(f"    {name:<40s} {c:+.3f}")

# Check L11_drift specifically
if "L11_drift_first_last" in fnames:
    idx = fnames.index("L11_drift_first_last")
    coefs_l11 = clf.coef_[:, idx]
    print(f"\n  L11_drift_first_last (POST-RESIDUALIZE):")
    print(f"    LOCKED:   {coefs_l11[0]:+.3f}")
    print(f"    SUCCESS:  {coefs_l11[1]:+.3f}")
    print(f"    WANDERING:{coefs_l11[2]:+.3f}")
    print(f"  → {'L11_drift retains separability' if abs(coefs_l11[2]) > 0.1 else 'L11_drift mostly explained by thinking_length'}")

# ============================================================
# (C) Match WANDERING/LOCKED on thinking_length, paired Wilcoxon
# ============================================================
print()
print("=" * 70)
print("(C) Match WANDERING/LOCKED on thinking_length → paired Wilcoxon on L11_drift")
print("=" * 70)

w_subset = df[df.sub_class == "wandering"].copy()
l_subset = df[df.sub_class == "locked"].copy()

# Greedy nearest-neighbor match on thinking_length
matches = []
l_unused = set(l_subset.index)
for w_idx, w_row in w_subset.iterrows():
    w_tlen = w_row["text_thinking_length_final"]
    best_l = None
    best_dist = np.inf
    for l_idx in l_unused:
        l_tlen = l_subset.loc[l_idx, "text_thinking_length_final"]
        dist = abs(w_tlen - l_tlen)
        if dist < best_dist:
            best_dist = dist
            best_l = l_idx
    if best_l is not None and best_dist < 100:  # tolerance: within 100 chars
        matches.append((w_idx, best_l, best_dist))
        l_unused.discard(best_l)

print(f"Matched pairs: {len(matches)} (max distance threshold: 100 chars)")

if len(matches) >= 5:
    w_l11 = [w_subset.loc[m[0], "L11_drift_first_last"] for m in matches]
    l_l11 = [l_subset.loc[m[1], "L11_drift_first_last"] for m in matches]
    diffs = np.array(w_l11) - np.array(l_l11)
    print(f"  Median L11_drift WANDERING (matched): {np.median(w_l11):.4f}")
    print(f"  Median L11_drift LOCKED (matched):    {np.median(l_l11):.4f}")
    print(f"  Median paired difference (W-L):       {np.median(diffs):+.4f}")
    try:
        stat, p_wilcox = stats.wilcoxon(diffs)
        print(f"  Wilcoxon signed-rank: stat={stat:.2f}, p={p_wilcox:.4f}")
        if p_wilcox < 0.05:
            print(f"  → L11_drift DIFFERS at matched thinking_length: real mechanism signal")
        else:
            print(f"  → L11_drift does NOT differ at matched thinking_length: proxy explanation more plausible")
    except Exception as e:
        print(f"  Wilcoxon failed: {e}")
else:
    print(f"  Too few matched pairs ({len(matches)}) for paired test")

# ============================================================
# Save results
# ============================================================
results = {
    "partial_correlation": {
        "wandering_l11drift_raw": {"rho": float(rho_raw), "p": float(p_raw)},
        "wandering_l11drift_partial_thinking": {"rho": float(rho_partial), "p": float(p_partial)},
        "locked_l11drift_raw": {"rho": float(rho_raw_l), "p": float(p_raw_l)},
        "locked_l11drift_partial_thinking": {"rho": float(rho_partial_l), "p": float(p_partial_l)},
    },
    "post_residualize_L11drift_coefs": {
        "LOCKED": float(clf.coef_[0, fnames.index("L11_drift_first_last")]) if "L11_drift_first_last" in fnames else None,
        "SUCCESS": float(clf.coef_[1, fnames.index("L11_drift_first_last")]) if "L11_drift_first_last" in fnames else None,
        "WANDERING": float(clf.coef_[2, fnames.index("L11_drift_first_last")]) if "L11_drift_first_last" in fnames else None,
    },
    "matched_subset": {
        "n_pairs": len(matches),
        "median_W_l11": float(np.median(w_l11)) if len(matches) >= 5 else None,
        "median_L_l11": float(np.median(l_l11)) if len(matches) >= 5 else None,
        "median_paired_diff_W_minus_L": float(np.median(diffs)) if len(matches) >= 5 else None,
        "wilcoxon_p": float(p_wilcox) if len(matches) >= 5 else None,
    },
}

OUT.write_text(json.dumps(results, indent=2))
print(f"\nSaved: {OUT}")
