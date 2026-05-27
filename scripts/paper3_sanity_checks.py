#!/usr/bin/env python3
"""Paper #3 sanity checks before committing to mechanism story.

(1) Per-class median + IQR for top stable features.
(2) L1 logistic coefficient signs (full data fit).
(3) Correlation matrix of top-12 features.
(4) Ultra-conservative drop: remove ALL L43 features (probe substrate) + re-fit
    pairwise W vs L. Does L11 signature survive?
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer

CSV = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out/features_n99.csv")

df = pd.read_csv(CSV)
LEAKY = [c for c in df.columns if any(c.startswith(p) for p in [
    "probe_score_", "probe_first_", "detector_v4_", "n_turns", "patch_n_bytes"
])]
ALL_NAN = ["detector_v1_fire_turn", "detector_v5_fire_turn"]
ZERO_VAR = ["probe_first_threshold_cross_turn"]  # already in LEAKY but explicit
TOP = [
    "L11_drift_first_last",
    "L43_cosine_consec_late",
    "tool_diversity_count",
    "text_doubt_density_late",
    "text_completion_density",
    "L31_norm_std",
    "text_thinking_length_final",
    "text_exploration_density",
    "L43_norm_std",
]

# ============= (1) Per-class median + IQR =============
print("=" * 70)
print("(1) Per-class median (IQR) for top features")
print("=" * 70)
print(f"{'feature':<35} {'SUCCESS':<18} {'LOCKED':<18} {'WANDERING':<18}")
print("-" * 90)
for f in TOP:
    row = f"{f:<35}"
    for cls in ["success", "locked", "wandering"]:
        sub = df[df.sub_class == cls][f].dropna()
        med = sub.median()
        q25, q75 = sub.quantile(0.25), sub.quantile(0.75)
        row += f"{med:.3f} ({q25:.2f}-{q75:.2f})  "
    print(row)

# ============= (2) L1 coefficient signs (full data fit) =============
print("\n" + "=" * 70)
print("(2) L1 coefficients (full-data fit, multinomial, C=1.0)")
print("=" * 70)
feat_df = df.drop(columns=["iid", "sub_class"]).select_dtypes(include=[np.number])
feat_df = feat_df.drop(columns=[c for c in feat_df.columns if c in LEAKY or c in ALL_NAN or c in ZERO_VAR])
X = SimpleImputer(strategy="median").fit_transform(feat_df.values)
X = StandardScaler().fit_transform(X)
y = df.sub_class.map({"locked": 0, "success": 1, "wandering": 2}).values
clf = LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=5000, random_state=42)
clf.fit(X, y)
classes = ["LOCKED", "SUCCESS", "WANDERING"]
fnames = list(feat_df.columns)

print(f"{'feature':<35} {'LOCKED':<10} {'SUCCESS':<10} {'WANDERING':<10}")
print("-" * 70)
for f in TOP:
    if f in fnames:
        idx = fnames.index(f)
        coefs = clf.coef_[:, idx]
        print(f"{f:<35} {coefs[0]:+.3f}     {coefs[1]:+.3f}     {coefs[2]:+.3f}")

# ============= (3) Correlation matrix top-12 =============
print("\n" + "=" * 70)
print("(3) Correlation matrix (Spearman) — top-12 stable features")
print("=" * 70)
top12 = [f for f in TOP if f in feat_df.columns]
corr = feat_df[top12].corr(method="spearman").round(2)
print(corr.to_string())

# ============= (4) Ultra-conservative: drop all L43, re-fit W vs L =============
print("\n" + "=" * 70)
print("(4) ULTRA-CONSERVATIVE: drop ALL L43 features + tool_diversity")
print("    (L43 substrate of probe; tool_diversity dominated SUCCESS)")
print("    Re-fit pairwise WANDERING vs LOCKED")
print("=" * 70)
ultra_drop = [c for c in feat_df.columns if c.startswith("L43_")] + ["tool_diversity_count"]
print(f"Dropped: {ultra_drop}")
feat_ultra = feat_df.drop(columns=ultra_drop)
print(f"Remaining features: {len(feat_ultra.columns)}")

mask = df.sub_class.isin(["wandering", "locked"])
X_wl = SimpleImputer(strategy="median").fit_transform(feat_ultra[mask].values)
X_wl = StandardScaler().fit_transform(X_wl)
y_wl = (df.sub_class[mask] == "wandering").astype(int).values

# 5-fold stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = cross_val_score(
    LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=5000, random_state=42),
    X_wl, y_wl, cv=cv, scoring="roc_auc"
)
print(f"\nW vs L pairwise AUC (5-fold CV, L1 logistic): {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
print(f"  per-fold: {[f'{a:.3f}' for a in auc_scores]}")

# Permutation null (200 perms, quick)
rng = np.random.RandomState(42)
null_aucs = []
for _ in range(200):
    y_shuf = rng.permutation(y_wl)
    null_aucs.append(cross_val_score(
        LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=5000, random_state=42),
        X_wl, y_shuf, cv=cv, scoring="roc_auc"
    ).mean())
null_aucs = np.array(null_aucs)
p = (1 + (null_aucs >= auc_scores.mean()).sum()) / (1 + len(null_aucs))
z = (auc_scores.mean() - null_aucs.mean()) / max(null_aucs.std(), 1e-9)
print(f"Null mean ± std: {null_aucs.mean():.3f} ± {null_aucs.std():.3f}")
print(f"Z-score: {z:+.2f}, p-value: {p:.4f}")

# Top L1 coefficients in this ultra-conservative model
clf_wl = LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=5000, random_state=42).fit(X_wl, y_wl)
coefs_wl = clf_wl.coef_[0]
ranked = sorted(zip(feat_ultra.columns, coefs_wl), key=lambda x: -abs(x[1]))
print(f"\nTop 10 L1 coefficients (positive = WANDERING-direction):")
for name, c in ranked[:10]:
    if abs(c) > 1e-6:
        print(f"  {name:<40s} {c:+.3f}")
