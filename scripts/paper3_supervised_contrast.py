"""
Paper #3 — Multi-Channel Mechanistic Signatures of Agent WANDERING.

Supervised contrast pipeline (3-way + pairwise) for N=99 trajectories of
Qwen3.6-27B SWE-bench-Pro Phase 6 (sub_class in {success, locked, wandering}).

Design notes
------------
* L1-penalized multinomial logistic (saga solver, OvR-style L1).
  Rationale: sparsity + interpretable coefficients per class (Meinshausen-
  Bühlmann 2010 stability selection assumes an L1 base learner). L2 would
  smear weight across collinear features (e.g. v4 cross-layer probes), making
  "top-k" report meaningless. ElasticNet was considered but adds a second
  hyperparameter on N=99 — under-powered tuning.
* Nested CV: outer 5-fold StratifiedKFold for *score* estimation, inner
  3-fold StratifiedKFold GridSearchCV for *C* selection. Scaler is fit
  INSIDE each outer training fold via Pipeline (no leakage).
* Permutation null: shuffle labels ONCE per replicate then run the full
  nested CV. This is expensive but the only correct way (shuffling inside
  GridSearchCV would also work but breaks calibration of inner-vs-outer).
* Stability selection: bootstrap (with replacement) the training set
  n_bootstrap times, fit a single L1 logistic at the *modal* C from nested
  CV, record top-k features by |coef| per class then aggregate. A feature
  is "stable" if it appears in top-k of >= pi_thr (default 0.80) bootstraps.
  Per Meinshausen-Bühlmann this gives FWER control for sparse models.
* Pairwise contrasts use the same pipeline on the 2-class subset; reported
  as APPENDIX (the 3-way is the registered primary).

Usage
-----
    python paper3_supervised_contrast.py \\
        --csv features_n99.csv --out_dir out/ \\
        --n_perm 1000 --n_bootstrap 200 --top_k 10 --n_jobs -1

    # With no --csv, runs synthetic-data self-test.
    python paper3_supervised_contrast.py --self_test --out_dir out/

Outputs
-------
    out/paper3_supervised_results.json   full numerical results
    out/paper3_summary.md                human-readable report
    out/paper3_null_distribution.npy     1000 null scores (3-way primary)
    out/paper3_stability_features.csv    ranked stability table

Author: OpenInterpretability (Caio Vicentino)
"""
from __future__ import annotations

import os
# Silence numpy/sklearn runtime warnings in joblib workers (set BEFORE numpy import in children)
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::RuntimeWarning,ignore::FutureWarning,ignore::UserWarning",
)

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Hyperparameter grid for L1 logistic. Wide log-grid; saga handles L1 + multinomial.
# Narrow upper bound (1e2) because N=99 — very weak data shouldn't get C=infinity.
C_GRID: list[float] = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]
MAX_ITER: int = 5000  # saga convergence with L1 on small N

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------
def make_pipeline(C: float = 1.0) -> Pipeline:
    """L1-multinomial logistic with StandardScaler. C set at grid time."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=C,
                    max_iter=MAX_ITER,
                    # sklearn 1.5+ deprecated multi_class; default is multinomial for
                    # multiclass with solver='saga' + penalty='l1'. Omit to avoid FutureWarning.
                    random_state=0,  # for solver reproducibility; CV split RNG is separate
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_and_prepare(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], LabelEncoder]:
    """Returns X, y (int-encoded), feature_names, iids, label_encoder."""
    df = pd.read_csv(csv_path)
    if "sub_class" not in df.columns:
        raise ValueError("CSV must contain a 'sub_class' column")
    if "iid" not in df.columns:
        df = df.assign(iid=[f"row{i:04d}" for i in range(len(df))])

    iids = df["iid"].astype(str).tolist()
    y_str = df["sub_class"].astype(str).str.lower().values
    feature_cols = [c for c in df.columns if c not in ("iid", "sub_class")]
    # numeric-only safety: drop non-numeric, warn
    feat_df = df[feature_cols].select_dtypes(include=[np.number])
    dropped = sorted(set(feature_cols) - set(feat_df.columns))
    if dropped:
        warnings.warn(f"Dropped non-numeric feature columns: {dropped}")
    # Drop columns that are entirely NaN (cannot impute, no signal)
    all_nan = [c for c in feat_df.columns if feat_df[c].isna().all()]
    if all_nan:
        warnings.warn(f"Dropped all-NaN feature columns: {all_nan}")
        feat_df = feat_df.drop(columns=all_nan)
    # Drop zero-variance columns (StandardScaler division by 0 → NaN)
    zero_var = [c for c in feat_df.columns if feat_df[c].std(skipna=True) < 1e-9]
    if zero_var:
        warnings.warn(f"Dropped zero-variance feature columns: {zero_var}")
        feat_df = feat_df.drop(columns=zero_var)
    feature_names = list(feat_df.columns)

    # Impute NaN with column median (per-fold imputation overkill for N=99; document this)
    X = feat_df.values.astype(float)
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    return X, y, feature_names, iids, le


# ---------------------------------------------------------------------------
# Nested CV
# ---------------------------------------------------------------------------
def _fit_outer_fold(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_inner: int,
    inner_seed: int,
) -> dict:
    """Run inner GridSearchCV on the outer-train, score on outer-test.
    Pipeline ensures scaler is fit only on inner-train fold inside GridSearchCV.
    """
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=inner_seed)
    pipe = make_pipeline()
    gs = GridSearchCV(
        pipe,
        param_grid={"clf__C": C_GRID},
        cv=inner_cv,
        scoring="f1_macro",
        n_jobs=1,  # outer loop parallelizes
        refit=True,
    )
    gs.fit(X_tr, y_tr)

    y_pred = gs.predict(X_te)
    classes_present = np.unique(y)
    n_classes = len(classes_present)
    # AUROC: per-class OvR — need probability output
    proba = gs.predict_proba(X_te)
    per_class_auroc: dict[int, float] = {}
    for k, cls in enumerate(classes_present):
        y_te_bin = (y_te == cls).astype(int)
        if y_te_bin.sum() == 0 or y_te_bin.sum() == len(y_te_bin):
            per_class_auroc[int(cls)] = float("nan")
        else:
            per_class_auroc[int(cls)] = float(roc_auc_score(y_te_bin, proba[:, k]))

    return {
        "C_selected": float(gs.best_params_["clf__C"]),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "per_class_auroc": per_class_auroc,
        "y_true": y_te.tolist(),
        "y_pred": y_pred.tolist(),
    }


def nested_cv_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_outer: int = 5,
    n_inner: int = 3,
    random_state: int = 42,
    n_jobs: int = 1,
    return_predictions: bool = True,
) -> dict:
    """Nested CV: outer macro-F1 + per-class AUROC, inner C tuning.
    Pipeline-based, no leakage. Outer folds run in parallel.
    """
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    folds = list(outer_cv.split(X, y))

    fold_results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_fit_outer_fold)(tr, te, X, y, n_inner, random_state + i)
        for i, (tr, te) in enumerate(folds)
    )

    macro_f1s = np.array([f["macro_f1"] for f in fold_results])
    C_selected = [f["C_selected"] for f in fold_results]

    # Aggregate per-class AUROC across folds
    classes_present = np.unique(y).tolist()
    aurocs_by_class: dict[int, list[float]] = {int(c): [] for c in classes_present}
    for f in fold_results:
        for c, v in f["per_class_auroc"].items():
            if not np.isnan(v):
                aurocs_by_class[c].append(v)

    # Aggregated confusion matrix (concat OOF predictions)
    y_true_all = np.concatenate([np.asarray(f["y_true"]) for f in fold_results])
    y_pred_all = np.concatenate([np.asarray(f["y_pred"]) for f in fold_results])
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes_present).tolist()

    out: dict = {
        "macro_f1_mean": float(macro_f1s.mean()),
        "macro_f1_std": float(macro_f1s.std(ddof=1)) if len(macro_f1s) > 1 else 0.0,
        "macro_f1_per_fold": macro_f1s.tolist(),
        "C_selected_per_fold": C_selected,
        "auroc_mean_per_class": {
            str(c): (float(np.mean(v)) if v else float("nan"))
            for c, v in aurocs_by_class.items()
        },
        "auroc_std_per_class": {
            str(c): (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            for c, v in aurocs_by_class.items()
        },
        "confusion_matrix": cm,
        "class_labels": classes_present,
    }
    if return_predictions:
        out["oof_y_true"] = y_true_all.tolist()
        out["oof_y_pred"] = y_pred_all.tolist()
    return out


# ---------------------------------------------------------------------------
# Permutation null
# ---------------------------------------------------------------------------
def _one_permutation(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    n_outer: int,
    n_inner: int,
) -> float:
    rng = np.random.default_rng(seed)
    y_perm = rng.permutation(y)
    res = nested_cv_classifier(
        X, y_perm, n_outer=n_outer, n_inner=n_inner,
        random_state=seed, n_jobs=1, return_predictions=False,
    )
    return res["macro_f1_mean"]


def permutation_null(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
    n_outer: int = 5,
    n_inner: int = 3,
    n_jobs: int = 1,
    observed_score: float | None = None,
) -> dict:
    """1000-permutation null on macro-F1. Returns p-value (one-sided, >=)."""
    if observed_score is None:
        obs = nested_cv_classifier(
            X, y, n_outer=n_outer, n_inner=n_inner,
            random_state=seed, n_jobs=n_jobs, return_predictions=False,
        )
        observed_score = obs["macro_f1_mean"]

    rng_seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_permutations)
    null_scores = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_one_permutation)(int(s), X, y, n_outer, n_inner) for s in rng_seeds
    )
    null_scores = np.asarray(null_scores, dtype=float)

    # one-sided p with +1 numerator/denominator (Phipson-Smyth)
    p_value = (1 + np.sum(null_scores >= observed_score)) / (1 + n_permutations)
    z_score = (observed_score - null_scores.mean()) / (null_scores.std(ddof=1) + 1e-12)

    return {
        "observed_score": float(observed_score),
        "null_mean": float(null_scores.mean()),
        "null_std": float(null_scores.std(ddof=1)),
        "p_value": float(p_value),
        "z_score": float(z_score),
        "n_permutations": int(n_permutations),
        "null_distribution": null_scores.tolist(),
    }


# ---------------------------------------------------------------------------
# Stability selection (Meinshausen-Bühlmann 2010)
# ---------------------------------------------------------------------------
def _bootstrap_topk(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    top_k: int,
) -> dict[int, set[int]]:
    """Bootstrap-resample, fit L1 logistic, return per-class top-k feature indices."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.integers(0, n, size=n)
    Xb, yb = X[idx], y[idx]
    if len(np.unique(yb)) < len(np.unique(y)):
        # rare: a class missing from bootstrap; skip this draw
        return {}
    pipe = make_pipeline(C=C)
    pipe.fit(Xb, yb)
    coef = pipe.named_steps["clf"].coef_  # shape (n_classes, n_features)
    per_class: dict[int, set[int]] = {}
    for k in range(coef.shape[0]):
        abs_w = np.abs(coef[k])
        top = np.argsort(abs_w)[::-1][:top_k]
        # only keep features with non-zero weight (L1)
        top = [int(i) for i in top if abs_w[i] > 1e-12]
        per_class[k] = set(top)
    return per_class


def stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    C: float,
    n_bootstrap: int = 200,
    top_k: int = 10,
    pi_thr: float = 0.80,
    seed: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Bootstrap L1 logistic at fixed C (modal from nested CV), record top-k
    per class per bootstrap, aggregate selection frequencies.

    Returns DataFrame with columns:
        feature, class, selection_freq, stable (bool, freq>=pi_thr)
    """
    rng_seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_bootstrap)
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_bootstrap_topk)(int(s), X, y, C, top_k) for s in rng_seeds
    )
    results = [r for r in results if r]  # drop failed bootstraps
    n_eff = len(results)

    n_features = X.shape[1]
    n_classes = len(class_labels)
    counts = np.zeros((n_classes, n_features), dtype=int)
    for per_class in results:
        for k, idxs in per_class.items():
            for i in idxs:
                counts[k, i] += 1

    rows = []
    for k, cls in enumerate(class_labels):
        for i, fname in enumerate(feature_names):
            freq = counts[k, i] / n_eff if n_eff > 0 else 0.0
            rows.append(
                {
                    "feature": fname,
                    "class": cls,
                    "selection_freq": float(freq),
                    "stable": bool(freq >= pi_thr),
                }
            )
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["class", "selection_freq"], ascending=[True, False]
    ).reset_index(drop=True)
    out.attrs["n_bootstrap_effective"] = n_eff
    out.attrs["C"] = C
    out.attrs["pi_thr"] = pi_thr
    out.attrs["top_k"] = top_k
    return out


# ---------------------------------------------------------------------------
# Pairwise contrasts (appendix)
# ---------------------------------------------------------------------------
def pairwise_contrasts(
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    contrasts: list[tuple[str, str]],
    n_perm: int = 200,
    n_outer: int = 5,
    n_inner: int = 3,
    seed: int = 42,
    n_jobs: int = 1,
) -> dict:
    """For each (cls_a, cls_b): subset, nested CV + permutation null."""
    label_map = {lbl: i for i, lbl in enumerate(le.classes_)}
    out: dict = {}
    for a, b in contrasts:
        if a not in label_map or b not in label_map:
            out[f"{a}_vs_{b}"] = {"error": f"label missing in data: {a} or {b}"}
            continue
        ia, ib = label_map[a], label_map[b]
        mask = (y == ia) | (y == ib)
        X_sub, y_sub = X[mask], (y[mask] == ia).astype(int)  # 1 = a, 0 = b
        primary = nested_cv_classifier(
            X_sub, y_sub, n_outer=n_outer, n_inner=n_inner,
            random_state=seed, n_jobs=n_jobs, return_predictions=False,
        )
        null = permutation_null(
            X_sub, y_sub, n_permutations=n_perm, seed=seed,
            n_outer=n_outer, n_inner=n_inner, n_jobs=n_jobs,
            observed_score=primary["macro_f1_mean"],
        )
        # drop bulky null_distribution from pairwise (kept only for primary)
        null.pop("null_distribution", None)
        out[f"{a}_vs_{b}"] = {
            "n_a": int((y == ia).sum()),
            "n_b": int((y == ib).sum()),
            "primary": primary,
            "null": null,
        }
    return out


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def synthesize_data(
    n_per_class: tuple[int, int, int] = (40, 39, 20),
    n_features: int = 50,
    n_signal: int = 5,
    signal_strength: float = 1.2,
    seed: int = 0,
    null: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Make a (sum(n_per_class), n_features) frame with `n_signal` features
    that shift mean by `signal_strength` across the 3 classes (if null=False).
    """
    rng = np.random.default_rng(seed)
    classes = ["success", "locked", "wandering"]
    n_total = sum(n_per_class)
    X = rng.standard_normal((n_total, n_features))
    y_str: list[str] = []
    for k, (cls, n_k) in enumerate(zip(classes, n_per_class)):
        y_str.extend([cls] * n_k)
    y_str = np.array(y_str)

    if not null:
        # inject signal in first n_signal features: class-dependent mean shift
        for fi in range(n_signal):
            for k, cls in enumerate(classes):
                mask = y_str == cls
                # shifts: (+1, -1, +2) for k=0,1,2, scaled by signal_strength
                shift = ((-1) ** k) * (1.0 + 0.5 * k) * signal_strength
                X[mask, fi] += shift

    feature_names = [f"feat_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df.insert(0, "sub_class", y_str)
    df.insert(0, "iid", [f"row{i:04d}" for i in range(n_total)])
    return df, feature_names


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def render_markdown(results: dict, out_path: Path) -> None:
    p = results["primary_3way"]
    null = results["primary_null"]
    stab_path = results.get("stability_path", "(see CSV)")
    lines = [
        "# Paper #3 — Supervised contrast results",
        "",
        f"Generated by `paper3_supervised_contrast.py` v1.0.",
        "",
        "## Primary 3-way classifier (WANDERING vs SUCCESS vs LOCKED)",
        "",
        f"- Macro-F1 (nested CV, 5x3): **{p['macro_f1_mean']:.4f} ± {p['macro_f1_std']:.4f}**",
        f"- Per-fold C selected: {p['C_selected_per_fold']}",
        f"- Per-class AUROC (mean ± std):",
    ]
    for k in p["auroc_mean_per_class"]:
        m = p["auroc_mean_per_class"][k]
        s = p["auroc_std_per_class"][k]
        lines.append(f"    - class {k}: {m:.4f} ± {s:.4f}")
    lines += [
        "",
        f"- Confusion matrix (rows=true, cols=pred, label order {p['class_labels']}):",
        "  ```",
    ]
    for row in p["confusion_matrix"]:
        lines.append("  " + " ".join(f"{x:4d}" for x in row))
    lines += [
        "  ```",
        "",
        "## Permutation null (3-way primary)",
        "",
        f"- Observed macro-F1: {null['observed_score']:.4f}",
        f"- Null mean ± std: {null['null_mean']:.4f} ± {null['null_std']:.4f}",
        f"- Z-score: {null['z_score']:.3f}",
        f"- p-value (one-sided, n={null['n_permutations']}): **{null['p_value']:.4g}**",
        "",
        "## Stability-selected features",
        f"  (See `{stab_path}` for full table.)",
        "",
        "## Pairwise contrasts (appendix)",
    ]
    for k, v in results.get("pairwise", {}).items():
        if "error" in v:
            lines.append(f"- {k}: ERROR — {v['error']}")
            continue
        lines.append(
            f"- **{k}** (n_a={v['n_a']}, n_b={v['n_b']}): "
            f"macro-F1 {v['primary']['macro_f1_mean']:.4f} "
            f"(null {v['null']['null_mean']:.4f}, p={v['null']['p_value']:.4g}, "
            f"z={v['null']['z_score']:.2f})"
        )

    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(
    df: pd.DataFrame,
    out_dir: Path,
    n_perm: int,
    n_bootstrap: int,
    top_k: int,
    n_outer: int,
    n_inner: int,
    seed: int,
    n_jobs: int,
    pairwise_perm: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # adapt load_and_prepare to in-memory df
    iids = df["iid"].astype(str).tolist()
    y_str = df["sub_class"].astype(str).str.lower().values
    feat_df = df.drop(columns=["iid", "sub_class"]).select_dtypes(include=[np.number])
    # DROP LABEL-LEAKING FEATURES (probe_score_* used to DEFINE sub-class via lock_fail_0.40 threshold)
    # Also drop termination-proxy features (n_turns, patch_n_bytes correlate with finish_reason which defines SUCCESS)
    # Also drop detector_v4_fire_turn (correlates with sub-class assignment process)
    leaky = [c for c in feat_df.columns if any(c.startswith(p) for p in [
        "probe_score_", "probe_first_", "detector_v4_", "n_turns", "patch_n_bytes"
    ])]
    if leaky:
        print(f"[load] Dropped LABEL-LEAKY features: {leaky}")
        feat_df = feat_df.drop(columns=leaky)
    # Drop all-NaN columns (cannot impute, no signal)
    all_nan = [c for c in feat_df.columns if feat_df[c].isna().all()]
    if all_nan:
        print(f"[load] Dropped all-NaN columns: {all_nan}")
        feat_df = feat_df.drop(columns=all_nan)
    # Drop zero-variance columns (StandardScaler div-by-0 → NaN)
    zero_var = [c for c in feat_df.columns if feat_df[c].std(skipna=True) < 1e-9]
    if zero_var:
        print(f"[load] Dropped zero-variance columns: {zero_var}")
        feat_df = feat_df.drop(columns=zero_var)
    feature_names = list(feat_df.columns)
    X = feat_df.values.astype(float)
    col_med = np.nanmedian(X, axis=0)
    nan_inds = np.where(np.isnan(X))
    X[nan_inds] = np.take(col_med, nan_inds[1])
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    print(f"[load] N={X.shape[0]} features={X.shape[1]} classes={list(le.classes_)}")
    print(f"[load] class counts: {np.bincount(y).tolist()}")

    t0 = time.time()
    primary = nested_cv_classifier(
        X, y, n_outer=n_outer, n_inner=n_inner,
        random_state=seed, n_jobs=n_jobs,
    )
    t_primary = time.time() - t0
    print(f"[primary] macro-F1 = {primary['macro_f1_mean']:.4f} "
          f"± {primary['macro_f1_std']:.4f}  ({t_primary:.1f}s)")

    t0 = time.time()
    null = permutation_null(
        X, y, n_permutations=n_perm, seed=seed,
        n_outer=n_outer, n_inner=n_inner, n_jobs=n_jobs,
        observed_score=primary["macro_f1_mean"],
    )
    t_null = time.time() - t0
    print(f"[null] p={null['p_value']:.4g} z={null['z_score']:.2f}  ({t_null:.1f}s for {n_perm} perms)")

    # save null distribution as npy
    np.save(out_dir / "paper3_null_distribution.npy", np.asarray(null["null_distribution"]))
    # drop bulky list from json
    null_for_json = {k: v for k, v in null.items() if k != "null_distribution"}

    # Modal C from nested CV → stability selection
    cs = primary["C_selected_per_fold"]
    modal_C = max(set(cs), key=cs.count)
    t0 = time.time()
    stab = stability_selection(
        X, y, feature_names, list(le.classes_),
        C=modal_C, n_bootstrap=n_bootstrap, top_k=top_k,
        seed=seed, n_jobs=n_jobs,
    )
    t_stab = time.time() - t0
    stab_path = out_dir / "paper3_stability_features.csv"
    stab.to_csv(stab_path, index=False)
    n_stable = int(stab["stable"].sum())
    print(f"[stability] modal_C={modal_C}, {n_stable} stable (>={stab.attrs['pi_thr']}) "
          f"out of {len(stab)} (feature×class)  ({t_stab:.1f}s)")

    t0 = time.time()
    pairs = pairwise_contrasts(
        X, y, le,
        contrasts=[("wandering", "success"), ("wandering", "locked"), ("success", "locked")],
        n_perm=pairwise_perm, n_outer=n_outer, n_inner=n_inner,
        seed=seed, n_jobs=n_jobs,
    )
    t_pair = time.time() - t0
    print(f"[pairwise] done  ({t_pair:.1f}s)")

    results = {
        "config": {
            "n_outer": n_outer,
            "n_inner": n_inner,
            "n_perm": n_perm,
            "n_bootstrap": n_bootstrap,
            "top_k": top_k,
            "pairwise_perm": pairwise_perm,
            "C_grid": C_GRID,
            "max_iter": MAX_ITER,
            "seed": seed,
        },
        "data": {
            "n": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "classes": list(le.classes_),
            "class_counts": np.bincount(y).tolist(),
        },
        "primary_3way": primary,
        "primary_null": null_for_json,
        "stability_modal_C": float(modal_C),
        "stability_n_stable": n_stable,
        "stability_path": str(stab_path),
        "pairwise": pairs,
        "timings_seconds": {
            "primary": t_primary,
            "null_total": t_null,
            "null_per_perm": t_null / max(n_perm, 1),
            "stability": t_stab,
            "pairwise": t_pair,
        },
    }

    json_path = out_dir / "paper3_supervised_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    render_markdown(results, out_dir / "paper3_summary.md")
    print(f"[done] wrote {json_path}, paper3_summary.md, "
          f"paper3_null_distribution.npy, paper3_stability_features.csv")
    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=None, help="features_n99.csv")
    p.add_argument("--out_dir", type=Path, default=Path("out"))
    p.add_argument("--n_perm", type=int, default=1000)
    p.add_argument("--n_bootstrap", type=int, default=200)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--n_outer", type=int, default=5)
    p.add_argument("--n_inner", type=int, default=3)
    p.add_argument("--pairwise_perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--self_test", action="store_true", help="Run on synthetic data (signal + null).")
    p.add_argument("--self_test_n_perm", type=int, default=200, help="Perms used in self-test (kept small).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test or args.csv is None:
        # SIGNAL-INJECTED synthetic
        print("=" * 70)
        print("SELF-TEST 1/2: synthetic with 5/50 informative features")
        print("=" * 70)
        df_sig, _ = synthesize_data(seed=0, null=False)
        out_sig = args.out_dir / "self_test_signal"
        run(
            df_sig, out_sig,
            n_perm=args.self_test_n_perm,
            n_bootstrap=args.n_bootstrap,
            top_k=args.top_k,
            n_outer=args.n_outer, n_inner=args.n_inner,
            seed=args.seed, n_jobs=args.n_jobs,
            pairwise_perm=max(50, args.self_test_n_perm // 4),
        )

        print()
        print("=" * 70)
        print("SELF-TEST 2/2: synthetic NULL (no signal in any feature)")
        print("=" * 70)
        df_null, _ = synthesize_data(seed=1, null=True)
        out_null = args.out_dir / "self_test_null"
        run(
            df_null, out_null,
            n_perm=args.self_test_n_perm,
            n_bootstrap=max(50, args.n_bootstrap // 4),
            top_k=args.top_k,
            n_outer=args.n_outer, n_inner=args.n_inner,
            seed=args.seed, n_jobs=args.n_jobs,
            pairwise_perm=max(50, args.self_test_n_perm // 4),
        )
        return

    df = pd.read_csv(args.csv)
    run(
        df, args.out_dir,
        n_perm=args.n_perm,
        n_bootstrap=args.n_bootstrap,
        top_k=args.top_k,
        n_outer=args.n_outer, n_inner=args.n_inner,
        seed=args.seed, n_jobs=args.n_jobs,
        pairwise_perm=args.pairwise_perm,
    )


if __name__ == "__main__":
    main()
