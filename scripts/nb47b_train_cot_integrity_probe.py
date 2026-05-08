"""Train CoT-Integrity probe from nb47b_capture activations + nb47 v1 labels.

Methodology sweep (random-feature baseline + top-K + L1-LR + PCA) at each
of 5 layers (L11/L23/L31/L43/L55), predicting has_think_v1 from the
activation at the last prompt token (right after auto-injected <think>).

Goal: identify (layer, method, K) with AUROC ≥ 0.75 + gap ≥ +0.10 above
random-feature baseline. That's ProbeGated-RAG v0.1 candidate.
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
NB47B = DRIVE / 'nb47b_capture'

LAYERS = [11, 23, 31, 43, 55]


def cv_with_features(X, y, feature_select_fn, n_splits=4, seed=42):
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
        Xtr_s = np.nan_to_num(sc.fit_transform(Xtr), nan=0.0, posinf=0.0, neginf=0.0)
        Xte_s = np.nan_to_num(sc.transform(Xte), nan=0.0, posinf=0.0, neginf=0.0)
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


def cv_l1(X, y, C=0.1, n_splits=4, seed=42):
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


def cv_pca(X, y, n_components=10, n_splits=4, seed=42):
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


def main():
    meta = json.loads((NB47B / 'metadata.json').read_text())
    records = meta['records']
    print(f'N records: {len(records)}')
    print(f'Capture position: {meta["capture_position"]}')

    # Build labels (use v1 = nb47's measurement)
    y_v1 = np.array([1 if r['has_think_v1'] else 0 for r in records], dtype=int)
    conds = [r['condition'] for r in records]
    print(f'has_think_v1 distribution: {int(y_v1.sum())}/{len(y_v1)} positive ({y_v1.mean()*100:.1f}%)')

    from collections import Counter
    print('By condition:')
    for cond in ['none', 'ensemble-gated', 'all-admit', 'random-50']:
        sub_y = [r['has_think_v1'] for r in records if r['condition'] == cond]
        print(f'  {cond:<20s}: {sum(sub_y)}/{len(sub_y)} = {sum(sub_y)/len(sub_y)*100:.1f}%')

    print()

    results = {}
    for L in LAYERS:
        path = NB47B / f'L{L}_pre_gen_activations.safetensors'
        X = load_file(str(path))['activations'].to(torch.float32).numpy()
        print(f'=== L{L} ({X.shape}) ===')

        # Random-feature baseline
        n_rand = 50
        rand_aurocs = []
        for k in range(n_rand):
            r_local = np.random.default_rng(2026 + k)
            au = cv_with_features(X, y_v1, lambda Xt, yt: select_random_k(Xt, yt, 50, r_local))
            if not np.isnan(au):
                rand_aurocs.append(au)
        rand_mean = float(np.mean(rand_aurocs)) if rand_aurocs else float('nan')
        rand_p95 = float(np.quantile(rand_aurocs, 0.95)) if rand_aurocs else float('nan')

        layer_results = {'random_top50_mean': rand_mean, 'random_top50_p95': rand_p95}

        # Top-K diff-means sweep
        topk_results = {}
        for k in [5, 10, 20, 50, 100, 200]:
            au = cv_with_features(X, y_v1, lambda Xt, yt, kk=k: select_top_k_diff(Xt, yt, kk))
            topk_results[k] = round(au, 4)
        layer_results['topk_diffmeans'] = topk_results
        best_topk = max(topk_results.items(), key=lambda x: x[1])
        print(f'  random top50: mean={rand_mean:.3f} p95={rand_p95:.3f}')
        for k, au in topk_results.items():
            print(f'  top-{k:3d} diffmeans: {au:.3f}')

        # Top-1
        au_top1 = cv_with_features(X, y_v1, lambda Xt, yt: select_top_k_diff(Xt, yt, 1))
        layer_results['top1_diffmeans'] = round(au_top1, 4)
        print(f'  top-1 single: {au_top1:.3f}')

        # L1-LR
        l1_results = {}
        for c in [0.01, 0.1, 1.0]:
            au = cv_l1(X, y_v1, C=c)
            l1_results[c] = round(au, 4)
            print(f'  L1 C={c}: {au:.3f}')
        layer_results['l1'] = l1_results

        # PCA
        pca_results = {}
        for n_pc in [5, 10, 20, 50]:
            au = cv_pca(X, y_v1, n_components=n_pc)
            pca_results[n_pc] = round(au, 4)
            print(f'  PCA-{n_pc}: {au:.3f}')
        layer_results['pca'] = pca_results

        # Best
        all_methods = [
            ('random_baseline', rand_mean),
            ('top-50 diffmeans', topk_results[50]),
            ('top-10 diffmeans', topk_results[10]),
            ('top-1 single', au_top1),
            ('L1 C=0.1', l1_results[0.1]),
            ('PCA-10', pca_results[10]),
        ]
        best_name, best_score = max(all_methods, key=lambda x: x[1] if not np.isnan(x[1]) else 0)
        gap = best_score - rand_mean
        layer_results['best'] = {'method': best_name, 'auroc': round(best_score, 4),
                                  'gap_vs_random': round(gap, 4)}
        print(f'  BEST: {best_name} = {best_score:.3f} (gap={gap:+.3f} vs random {rand_mean:.3f})')
        print()

        results[f'L{L}'] = layer_results

    # Cross-condition test: train on "none" + half "ensemble-gated", test on rest
    print('========== CROSS-CONDITION TEST ==========')
    print('Train on subset, test on held-out (per-condition split)')
    for L in LAYERS:
        path = NB47B / f'L{L}_pre_gen_activations.safetensors'
        X = load_file(str(path))['activations'].to(torch.float32).numpy()

        # Stratified by condition
        idx_train = []
        idx_test = []
        rng = np.random.default_rng(42)
        for cond in ['none', 'ensemble-gated', 'all-admit', 'random-50']:
            cond_idx = [i for i, c in enumerate(conds) if c == cond]
            rng.shuffle(cond_idx)
            split = len(cond_idx) // 2
            idx_train.extend(cond_idx[:split])
            idx_test.extend(cond_idx[split:])

        # Train top-10 diff means on train half, eval on test half
        sel = select_top_k_diff(X[idx_train], y_v1[idx_train], 10)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[idx_train][:, sel])
        Xte = sc.transform(X[idx_test][:, sel])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
        clf.fit(Xtr, y_v1[idx_train])
        au_tr = roc_auc_score(y_v1[idx_train], clf.predict_proba(Xtr)[:, 1])
        au_te = roc_auc_score(y_v1[idx_test], clf.predict_proba(Xte)[:, 1])
        print(f'  L{L}: train AUROC {au_tr:.3f} → test AUROC {au_te:.3f}')

    # Save results
    OUT = NB47B / 'cot_integrity_probe_sweep.json'
    OUT.write_text(json.dumps(results, indent=2))
    print(f'\nSaved {OUT}')

    # Verdict
    print('\n========== VERDICT ==========')
    best_overall = None
    best_gap = -1
    for layer, r in results.items():
        gap = r['best']['gap_vs_random']
        if gap > best_gap:
            best_gap = gap
            best_overall = (layer, r['best'])
    print(f'Best probe: {best_overall[0]} {best_overall[1]["method"]}: AUROC {best_overall[1]["auroc"]:.3f} (gap {best_overall[1]["gap_vs_random"]:+.3f})')
    if best_gap >= 0.15:
        print('🟢 STRONG signal — paper-grade')
    elif best_gap >= 0.10:
        print('🟡 MODERATE signal — usable but workshop-bound')
    elif best_gap >= 0.05:
        print('🟠 WEAK signal — needs more data')
    else:
        print('🔴 NULL — probe does not detect CoT collapse from prompt activation')


if __name__ == '__main__':
    main()
