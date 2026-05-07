"""Phase 6c methodology sweep — test if alternative probe configs reveal signal.

The N=42 preview with top-50 diff-means + LR shows all probes near random (0.50-0.68).
This script tests whether the methodology is the bottleneck:

(A) Random-feature baseline at N=42 — calibrates noise floor at this sample size.
    Without this, we cannot distinguish "signal collapsed" from "noise floor."

(B) Top-K sweep — K = 5, 10, 20, 50, 100, 200 features per probe. If K=10 holds
    while K=50 collapses, over-parameterization is the issue and we have a fix.

(C) L1-regularized LR on all 5120 features — let regularization pick features,
    no manual top-K.

(D) Single best feature (K=1) — extreme regularization, most robust to small N.

(E) PCA top-50 — data-driven feature reduction.

Run on top probes from Phase 5d (L43/L55 think_start, L11 pre_tool, L23/L31 turn_end).
"""
from __future__ import annotations
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DRIVE = Path('/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
P1 = DRIVE / 'swebench_v1_phase1'
P5 = DRIVE / 'swebench_v5_phase5'
P6 = DRIVE / 'swebench_v6_phase6'
P6B = P6 / 'phase6b'
OUT = P6 / 'phase6c_methodology_sweep.json'

TOP_PROBES = [
    (43, 'think_start'),
    (55, 'think_start'),
    (11, 'pre_tool'),
    (23, 'turn_end'),
    (31, 'turn_end'),
    (43, 'pre_tool'),
]


def classify(r: dict) -> str:
    if r.get('pull_failed'):
        return 'env_mismatch'
    c = r.get('conditions', {})
    n = c.get('none', {}).get('n_pass', 0)
    g = c.get('golden', {}).get('n_pass', 0)
    a = c.get('agent', {}).get('n_pass', 0)
    total = c.get('golden', {}).get('total', 0)
    if total == 0 or g <= n:
        return 'env_mismatch'
    if a >= g:
        return 'solves'
    if a > n:
        return 'partial'
    return 'fails'


def load_pos_mean(iid: str, layer: int, position: str) -> np.ndarray | None:
    for root in [P6 / 'captures', P1 / 'captures']:
        if not root.exists():
            continue
        meta_glob = list(root.glob(f'{iid}*.meta.json'))
        if not meta_glob:
            continue
        meta_path = meta_glob[0]
        weights = meta_path.with_suffix('').with_suffix('.safetensors')
        if not weights.exists():
            continue
        m = json.loads(meta_path.read_text())
        t = load_file(str(weights))
        vecs = [t[r['activation_key']].to(torch.float32).numpy()
                for r in m['records']
                if r['layer'] == layer and r['position_label'] == position
                and r['activation_key'] in t]
        if vecs:
            return np.mean(np.stack(vecs, axis=0), axis=0)
    return None


def cv_with_features(X: np.ndarray, y: np.ndarray, feature_select_fn, n_splits: int = 4, seed: int = 42) -> float:
    """Generic CV-AUROC: feature_select_fn(X_train, y_train) -> selected indices."""
    n_pos = int(y.sum()); n_neg = int(len(y) - y.sum())
    if n_pos < 2 or n_neg < 2:
        return float('nan')
    n_splits = min(n_splits, n_pos, n_neg)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs = []
    for tr, te in skf.split(X, y):
        sel = feature_select_fn(X[tr], y[tr])
        Xtr = X[tr][:, sel] if sel is not None else X[tr]
        Xte = X[te][:, sel] if sel is not None else X[te]
        if Xtr.shape[1] == 0:
            continue
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        # Replace NaN/inf from constant features
        Xtr_s = np.nan_to_num(Xtr_s, nan=0.0, posinf=0.0, neginf=0.0)
        Xte_s = np.nan_to_num(Xte_s, nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
        clf.fit(Xtr_s, y[tr])
        if len(np.unique(y[te])) >= 2:
            aurocs.append(roc_auc_score(y[te], clf.predict_proba(Xte_s)[:, 1]))
    return float(np.mean(aurocs)) if aurocs else float('nan')


def select_top_k_diff(X_tr, y_tr, k):
    d = np.abs(X_tr[y_tr == 1].mean(axis=0) - X_tr[y_tr == 0].mean(axis=0))
    return np.argsort(-d)[:k]


def select_random_k(X_tr, y_tr, k, rng):
    return rng.choice(X_tr.shape[1], size=k, replace=False)


def cv_l1(X: np.ndarray, y: np.ndarray, C: float = 0.1, n_splits: int = 4, seed: int = 42) -> float:
    """L1-regularized LR on ALL features, no manual feature selection."""
    n_pos = int(y.sum()); n_neg = int(len(y) - y.sum())
    if n_pos < 2 or n_neg < 2:
        return float('nan')
    n_splits = min(n_splits, n_pos, n_neg)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = np.nan_to_num(sc.fit_transform(X[tr]), nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.nan_to_num(sc.transform(X[te]), nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(C=C, max_iter=5000, penalty='l1', solver='saga',
                                 class_weight='balanced')
        clf.fit(Xtr, y[tr])
        if len(np.unique(y[te])) >= 2:
            aurocs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aurocs)) if aurocs else float('nan')


def cv_pca(X: np.ndarray, y: np.ndarray, n_components: int = 50, n_splits: int = 4, seed: int = 42) -> float:
    """PCA-reduced features (top-N components by variance, no label leakage)."""
    n_pos = int(y.sum()); n_neg = int(len(y) - y.sum())
    if n_pos < 2 or n_neg < 2:
        return float('nan')
    n_splits = min(n_splits, n_pos, n_neg)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = np.nan_to_num(sc.fit_transform(X[tr]), nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.nan_to_num(sc.transform(X[te]), nan=0.0, posinf=0.0, neginf=0.0)
        n_comp = min(n_components, Xtr.shape[0] - 1, Xtr.shape[1])
        pca = PCA(n_components=n_comp)
        Xtr_pc = pca.fit_transform(Xtr)
        Xte_pc = pca.transform(Xte)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
        clf.fit(Xtr_pc, y[tr])
        if len(np.unique(y[te])) >= 2:
            aurocs.append(roc_auc_score(y[te], clf.predict_proba(Xte_pc)[:, 1]))
    return float(np.mean(aurocs)) if aurocs else float('nan')


def main() -> None:
    p6_results_path = P6 / 'phase6_results.json'
    p6b_results_path = P6B / 'phase6b_results.json'
    p5a_path = P5 / 'phase5a_results.json'

    p6 = json.loads(p6_results_path.read_text())
    p6b = json.loads(p6b_results_path.read_text())
    p5a = json.loads(p5a_path.read_text())

    verdicts = {iid: classify(r) for iid, r in p6b.items()}
    for iid, r in p5a.items():
        if iid not in verdicts:
            verdicts[iid] = classify(r)

    valid_iids = [iid for iid, v in verdicts.items() if v != 'env_mismatch']
    y = np.array([1 if verdicts[iid] == 'solves' else 0 for iid in valid_iids], dtype=int)
    print(f'N valid={len(valid_iids)}, strict_pos={int(y.sum())}, neg={int(len(y) - y.sum())}')

    out: dict = {'n_valid': len(valid_iids), 'n_pos': int(y.sum()), 'probes': {}}

    for L, pos in TOP_PROBES:
        Xs = []
        for iid in valid_iids:
            v = load_pos_mean(iid, L, pos)
            Xs.append(v)
        if any(v is None for v in Xs):
            missing = sum(1 for v in Xs if v is None)
            print(f'\nL{L} {pos}: SKIP (missing {missing} captures)')
            continue
        X = np.stack(Xs, axis=0)
        print(f'\n=== L{L} {pos} (X shape {X.shape}) ===')

        probe_results: dict = {}

        # (A) Random-feature baseline (top-50 random)
        n_rand = 50
        rng = np.random.default_rng(2026)
        rand_aurocs = []
        for k in range(n_rand):
            r_local = np.random.default_rng(2026 + k)
            au = cv_with_features(X, y, lambda Xt, yt: select_random_k(Xt, yt, 50, r_local))
            if not np.isnan(au):
                rand_aurocs.append(au)
        rand_mean = float(np.mean(rand_aurocs)) if rand_aurocs else float('nan')
        rand_p95 = float(np.quantile(rand_aurocs, 0.95)) if rand_aurocs else float('nan')
        print(f'  Random top-50:        mean={rand_mean:.3f}  p95={rand_p95:.3f}')
        probe_results['random_top50_mean'] = rand_mean
        probe_results['random_top50_p95'] = rand_p95

        # (B) Top-K sweep (diff-of-means)
        topk_results = {}
        for k in [5, 10, 20, 50, 100, 200]:
            if k > X.shape[1]:
                continue
            au = cv_with_features(X, y, lambda Xt, yt, kk=k: select_top_k_diff(Xt, yt, kk))
            topk_results[k] = round(au, 4)
            print(f'  Top-{k:3d} diffmeans:    AUROC={au:.3f}')
        probe_results['topk_sweep'] = topk_results

        # (D) Single best feature (top-1)
        au_top1 = cv_with_features(X, y, lambda Xt, yt: select_top_k_diff(Xt, yt, 1))
        probe_results['top1_diffmeans'] = round(au_top1, 4)
        print(f'  Top-1 single feat:    AUROC={au_top1:.3f}')

        # (C) L1-regularized LR all features
        l1_results = {}
        for c in [0.01, 0.1, 1.0]:
            au_l1 = cv_l1(X, y, C=c)
            l1_results[c] = round(au_l1, 4)
            print(f'  L1-LR all (C={c:.2f}): AUROC={au_l1:.3f}')
        probe_results['l1_all_features'] = l1_results

        # (E) PCA top-N components
        pca_results = {}
        for n_pc in [5, 10, 20, 50]:
            au_pca = cv_pca(X, y, n_components=n_pc)
            pca_results[n_pc] = round(au_pca, 4)
            print(f'  PCA-{n_pc:3d} features:   AUROC={au_pca:.3f}')
        probe_results['pca'] = pca_results

        # Summary: gap between best methodology and random baseline
        all_methods = [
            ('top-50 diffmeans', topk_results.get(50, float('nan'))),
            ('top-10 diffmeans', topk_results.get(10, float('nan'))),
            ('top-1 single', au_top1),
            ('L1 C=0.1', l1_results.get(0.1, float('nan'))),
            ('PCA-10', pca_results.get(10, float('nan'))),
        ]
        best_name, best_score = max(all_methods, key=lambda x: x[1] if not np.isnan(x[1]) else 0)
        gap_vs_random = best_score - rand_mean if not np.isnan(rand_mean) else float('nan')
        print(f'  BEST: {best_name} = {best_score:.3f}  (random_mean={rand_mean:.3f}, gap={gap_vs_random:+.3f})')
        probe_results['best'] = {'method': best_name, 'auroc': round(best_score, 4),
                                  'gap_vs_random': round(gap_vs_random, 4)}

        out['probes'][f'L{L}_{pos}'] = probe_results

    # Final overall verdict
    print(f'\n=== METHODOLOGY SWEEP VERDICT ===')
    max_gap = -1.0
    max_label = None
    for label, p in out['probes'].items():
        gap = p.get('best', {}).get('gap_vs_random', -1)
        if gap > max_gap:
            max_gap = gap
            max_label = label
    print(f'Max gap vs random across all probes: {max_label} = +{max_gap:.3f}')
    if max_gap < 0.05:
        print('🔴 NO methodology achieves >0.05 gap over random — true null suspected')
    elif max_gap < 0.10:
        print('🟡 BORDERLINE: gap exists but small — may be noise')
    else:
        print(f'🟢 GAP > 0.10: real signal at {max_label} ({max_gap:+.3f} above random)')

    OUT.write_text(json.dumps(out, indent=2))
    print(f'\nWrote {OUT}')


if __name__ == '__main__':
    main()
