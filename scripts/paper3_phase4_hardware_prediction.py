#!/usr/bin/env python3
"""
Phase 4 — Cross-Hardware Prediction Analysis for paper #3.

Join features_n99.csv with paper3_hardware_validation.csv on iid.
Train binary classifier on WANDERING subset (n=20) features (from RTX 6000)
to predict h100_flipped outcome (7/13 split).

Reports AUROC + 1000-permutation null + top-features.

Honest about N=20 limits: report multiple methods (L1 logistic, RF) +
LOO cross-validation (not k-fold — N=20 too small).
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer

FEATURES_CSV = Path(
    "/Volumes/SSD Major/fish/openinterp-swebench-harness/"
    "scripts/inflection_turn_out/features_n99.csv"
)
HARDWARE_CSV = Path(
    "/Volumes/SSD Major/fish/openinterp-swebench-harness/"
    "scripts/paper3_hardware_validation.csv"
)
OUT_PATH = Path(
    "/Volumes/SSD Major/fish/openinterp-swebench-harness/"
    "scripts/inflection_turn_out/paper3_hardware_prediction.json"
)


def loo_auc(X, y, clf):
    """Leave-one-out AUC using cross_val_predict with predict_proba."""
    loo = LeaveOneOut()
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", clf),
    ])
    try:
        y_prob = cross_val_predict(pipe, X, y, cv=loo, method="predict_proba", n_jobs=-1)[:, 1]
    except Exception as e:
        return {"error": str(e), "auc": None}
    auc = roc_auc_score(y, y_prob)
    # accuracy at 0.5 threshold
    acc = accuracy_score(y, y_prob > 0.5)
    return {"auc": float(auc), "acc": float(acc), "y_prob": y_prob.tolist(), "y_true": y.tolist()}


def permutation_null(X, y, clf, n_perm=1000, seed=42):
    """Shuffle y, retrain, compute LOO AUC. Return null distribution + p-value."""
    rng = np.random.RandomState(seed)
    observed = loo_auc(X, y, clf)
    if observed.get("auc") is None:
        return {"observed": observed, "null": None, "p_value": None}
    null_aucs = []
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        try:
            res = loo_auc(X, y_shuf, type(clf)(**clf.get_params()))
            if res.get("auc") is not None:
                null_aucs.append(res["auc"])
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  perm {i+1}/{n_perm}", flush=True)
    null_aucs = np.array(null_aucs)
    p = (1 + np.sum(null_aucs >= observed["auc"])) / (1 + len(null_aucs))
    return {
        "observed_auc": observed["auc"],
        "observed_acc": observed["acc"],
        "null_mean": float(null_aucs.mean()),
        "null_std": float(null_aucs.std()),
        "null_n": int(len(null_aucs)),
        "z_score": float((observed["auc"] - null_aucs.mean()) / max(null_aucs.std(), 1e-9)),
        "p_value": float(p),
    }


def main():
    print("=== Paper #3 Phase 4 — Cross-Hardware Prediction ===\n")

    # Load both CSVs
    feats = pd.read_csv(FEATURES_CSV)
    hw = pd.read_csv(HARDWARE_CSV)
    print(f"Features: {len(feats)} rows × {len(feats.columns)} cols")
    print(f"Hardware: {len(hw)} rows × {len(hw.columns)} cols")

    # Build join key — hw.iid may have full or fragment; check
    print(f"Sample hw iid: {hw['iid'].iloc[0][:80]}")
    print(f"Sample features iid: {feats['iid'].iloc[0][:80]}")

    # Filter features to WANDERING only (n=20)
    wand_feats = feats[feats["sub_class"] == "wandering"].copy()
    print(f"WANDERING features: {len(wand_feats)}")

    # Join on iid (or iid fragment)
    merged = wand_feats.merge(hw, on="iid", how="inner")
    print(f"Merged rows: {len(merged)}")
    if len(merged) < 20:
        # Try fragment matching
        print("Full-iid match incomplete; trying fragment-prefix match...")
        feat_iids = set(wand_feats["iid"].tolist())
        hw_iids = set(hw["iid"].tolist())
        # find max common prefix
        common = feat_iids & hw_iids
        print(f"Direct overlap: {len(common)}")
        if len(common) < 20:
            # Try to print missing
            missing = feat_iids - hw_iids
            print(f"In features but NOT in hardware: {len(missing)}")
            for m in list(missing)[:3]:
                print(f"  feat: {m}")
            for h in list(hw_iids - feat_iids)[:3]:
                print(f"  hw:   {h}")

    # Assume we have merged. Extract X, y
    if len(merged) >= 15:
        # Drop non-numeric/identifier columns
        drop_cols = [c for c in merged.columns if not pd.api.types.is_numeric_dtype(merged[c])]
        drop_cols += ["h100_flipped"]
        feature_cols = [c for c in merged.columns if c not in drop_cols and c not in ["rtx6000_n_turns"]]
        # Some columns might still be id-like; filter
        feature_cols = [c for c in feature_cols if merged[c].notna().sum() > 5]
        X = merged[feature_cols].values
        y = merged["h100_flipped"].astype(int).values

        print(f"\nFeature matrix: {X.shape}")
        print(f"y distribution: flipped={int(y.sum())}, stayed={int((1-y).sum())}")

        # Try L1 logistic
        print("\n--- L1 Logistic (LOO) ---")
        clf1 = LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=2000, random_state=42)
        res1 = permutation_null(X, y, clf1, n_perm=1000)
        print(f"  AUC: {res1['observed_auc']:.3f} (acc: {res1['observed_acc']:.3f})")
        print(f"  Null mean ± std: {res1['null_mean']:.3f} ± {res1['null_std']:.3f}")
        print(f"  Z-score: {res1['z_score']:+.2f}, p={res1['p_value']:.4f}")

        # Try RF for nonlinear
        print("\n--- Random Forest (LOO) ---")
        clf2 = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
        res2 = permutation_null(X, y, clf2, n_perm=200)  # cheaper null for RF
        print(f"  AUC: {res2['observed_auc']:.3f} (acc: {res2['observed_acc']:.3f})")
        print(f"  Null mean ± std: {res2['null_mean']:.3f} ± {res2['null_std']:.3f}")
        print(f"  Z-score: {res2['z_score']:+.2f}, p={res2['p_value']:.4f}")

        # Get L1 coefficients for feature importance (fit on full data)
        pipe_full = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(penalty="l1", C=1.0, solver="saga", max_iter=2000, random_state=42)),
        ])
        pipe_full.fit(X, y)
        coefs = pipe_full.named_steps["clf"].coef_[0]
        top_idx = np.argsort(np.abs(coefs))[::-1][:10]
        top_features = [(feature_cols[i], float(coefs[i])) for i in top_idx if abs(coefs[i]) > 1e-6]
        print(f"\n--- Top L1 features (sparsity = {(coefs == 0).sum()}/{len(coefs)} zero) ---")
        for name, c in top_features:
            print(f"  {name:40s}  coef={c:+.4f}")

        results = {
            "n_samples": len(merged),
            "n_features": X.shape[1],
            "y_distribution": {"flipped": int(y.sum()), "stayed": int((1-y).sum())},
            "l1_logistic": res1,
            "random_forest": res2,
            "top_l1_features": top_features,
            "feature_names": feature_cols,
        }
        OUT_PATH.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nSaved: {OUT_PATH}")
    else:
        print(f"\n!! Cannot proceed — only {len(merged)} merged rows (need >=15)")


if __name__ == "__main__":
    main()
