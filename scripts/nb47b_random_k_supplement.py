"""Supplement to nb47b_train_cot_integrity_probe.py — random K-matched baselines.

The original sweep computed random K=50 only, but at N=240 with 5120 dims that's
already overparameterized — random achieves ~0.83 AUROC at L43/L55. We need
random-K-matched baselines for K ∈ {5, 10, 20, 50, 100, 200} to identify the
real signal floor at each capacity.

Output: nb47b_capture/cot_integrity_random_k_supplement.json with random
baselines (mean + p95 across 50 samples) at each K, paired with top-K diffmeans
gap per K.
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

DRIVE = Path('/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
NB47B = DRIVE / 'nb47b_capture'
LAYERS = [11, 23, 31, 43, 55]
KS = [5, 10, 20, 50, 100, 200]
N_RAND = 50


def cv_with_features(X, y, sel, n_splits=4, seed=42):
    n_pos = int(y.sum()); n_neg = int(len(y) - y.sum())
    n_splits = min(n_splits, n_pos, n_neg)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs = []
    for tr, te in skf.split(X, y):
        Xtr = X[tr][:, sel]; Xte = X[te][:, sel]
        sc = StandardScaler()
        Xtr_s = np.nan_to_num(sc.fit_transform(Xtr), nan=0.0, posinf=0.0, neginf=0.0)
        Xte_s = np.nan_to_num(sc.transform(Xte), nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
        clf.fit(Xtr_s, y[tr])
        if len(np.unique(y[te])) >= 2:
            aurocs.append(roc_auc_score(y[te], clf.predict_proba(Xte_s)[:, 1]))
    return float(np.mean(aurocs)) if aurocs else float('nan')


def topk_diff(X_tr, y_tr, k):
    d = np.abs(X_tr[y_tr == 1].mean(axis=0) - X_tr[y_tr == 0].mean(axis=0))
    return np.argsort(-d)[:k]


def main():
    meta = json.loads((NB47B / 'metadata.json').read_text())
    records = meta['records']
    y = np.array([1 if r['has_think_v1'] else 0 for r in records], dtype=int)
    print(f'N={len(y)}, pos={int(y.sum())} ({y.mean()*100:.1f}%)')

    results = {}
    for L in LAYERS:
        path = NB47B / f'L{L}_pre_gen_activations.safetensors'
        X = load_file(str(path))['activations'].to(torch.float32).numpy()
        print(f'\n=== L{L} ({X.shape}) ===')

        layer = {}
        for K in KS:
            # Random K baseline (50 samples)
            rand_aurocs = []
            for s in range(N_RAND):
                rng = np.random.default_rng(2026 + s)
                sel = rng.choice(X.shape[1], size=K, replace=False)
                # Compute fold-level random selection (refresh per fold for honesty)
                # Actually use single seed-fixed random selection across all folds — matches original script behavior
                au = cv_with_features(X, y, sel)
                if not np.isnan(au):
                    rand_aurocs.append(au)
            rand_mean = float(np.mean(rand_aurocs)) if rand_aurocs else float('nan')
            rand_p95 = float(np.quantile(rand_aurocs, 0.95)) if rand_aurocs else float('nan')

            # Top-K diff means (re-selected per fold inside CV — proper)
            au_topk = []
            n_pos = int(y.sum()); n_neg = int(len(y) - y.sum())
            n_splits = min(4, n_pos, n_neg)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for tr, te in skf.split(X, y):
                sel = topk_diff(X[tr], y[tr], K)
                Xtr = X[tr][:, sel]; Xte = X[te][:, sel]
                sc = StandardScaler()
                Xtr_s = np.nan_to_num(sc.fit_transform(Xtr), nan=0.0, posinf=0.0, neginf=0.0)
                Xte_s = np.nan_to_num(sc.transform(Xte), nan=0.0, posinf=0.0, neginf=0.0)
                clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
                clf.fit(Xtr_s, y[tr])
                if len(np.unique(y[te])) >= 2:
                    au_topk.append(roc_auc_score(y[te], clf.predict_proba(Xte_s)[:, 1]))
            top_mean = float(np.mean(au_topk)) if au_topk else float('nan')
            gap = top_mean - rand_mean

            layer[K] = {
                'random_mean': round(rand_mean, 4),
                'random_p95': round(rand_p95, 4),
                'topk_diff': round(top_mean, 4),
                'gap_vs_random': round(gap, 4),
            }
            sig = '🟢' if gap >= 0.10 else ('🟡' if gap >= 0.05 else '🔴')
            print(f'  K={K:>3d}: random_mean={rand_mean:.3f} (p95={rand_p95:.3f}) | top-K={top_mean:.3f} | gap={gap:+.3f} {sig}')

        # Best K = highest gap
        best_K = max(layer.keys(), key=lambda k: layer[k]['gap_vs_random'])
        layer['best'] = {
            'K': best_K,
            'topk_auroc': layer[best_K]['topk_diff'],
            'random_mean': layer[best_K]['random_mean'],
            'gap_vs_random': layer[best_K]['gap_vs_random'],
        }
        results[f'L{L}'] = layer

    OUT = NB47B / 'cot_integrity_random_k_supplement.json'
    OUT.write_text(json.dumps(results, indent=2))
    print(f'\nSaved {OUT}')

    # Summary
    print('\n========== VERDICT (gap ≥ +0.10 = paper-grade) ==========')
    for L in LAYERS:
        r = results[f'L{L}']['best']
        sig = '🟢' if r['gap_vs_random'] >= 0.10 else ('🟡' if r['gap_vs_random'] >= 0.05 else '🔴')
        print(f'  L{L} best K={r["K"]:>3d}: top-K={r["topk_auroc"]:.3f} vs random={r["random_mean"]:.3f} gap={r["gap_vs_random"]:+.3f} {sig}')


if __name__ == '__main__':
    main()
