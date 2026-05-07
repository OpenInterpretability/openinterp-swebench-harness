"""Phase 6c PREVIEW — preliminary AUROC on whatever Phase 6b has labeled so far.

Designed to run repeatedly as Phase 6b labels more instances. Computes:
  - Verdict distribution (solves/partial/fails/env_mismatch)
  - Top-probe AUROC (L43/L55 think_start, L11 pre_tool, L31 turn_end)
  - 4-fold CV with top-50 features chosen INSIDE each fold (no leakage)
  - 1000-iter bootstrap CI
  - Optional: permutation null (only if N >= 30 to be meaningful)

Outputs to stdout + saves to phase6c_preview.json on Drive.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DRIVE = Path('/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
P6 = DRIVE / 'swebench_v6_phase6'
P6B = P6 / 'phase6b'
OUT = P6 / 'phase6c_preview.json'

TOP_PROBES = [
    (43, 'think_start'),
    (55, 'think_start'),
    (11, 'pre_tool'),
    (31, 'turn_end'),
    (23, 'turn_end'),
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


def true_y(verdict: str, loose: bool = False) -> int | None:
    if verdict == 'env_mismatch':
        return None
    if verdict == 'solves':
        return 1
    if verdict == 'partial':
        return 1 if loose else 0
    return 0


def load_pos_mean(iid: str, layer: int, position: str) -> np.ndarray | None:
    captures = P6 / 'captures'
    meta_glob = list(captures.glob(f'{iid}*.meta.json'))
    if not meta_glob:
        return None
    meta_path = meta_glob[0]
    weights = meta_path.with_suffix('').with_suffix('.safetensors')
    if not weights.exists():
        return None
    m = json.loads(meta_path.read_text())
    t = load_file(str(weights))
    vecs = [t[r['activation_key']].to(torch.float32).numpy()
            for r in m['records']
            if r['layer'] == layer and r['position_label'] == position
            and r['activation_key'] in t]
    return np.mean(np.stack(vecs, axis=0), axis=0) if vecs else None


def cv_auroc(X: np.ndarray, y: np.ndarray, top_k: int = 50, n_splits: int = 4, seed: int = 42) -> float:
    n_pos = int(y.sum()); n_neg = int(len(y) - y.sum())
    if n_pos < 2 or n_neg < 2:
        return float('nan')
    n_splits = min(n_splits, n_pos, n_neg)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs = []
    for tr, te in skf.split(X, y):
        d = np.abs(X[tr][y[tr] == 1].mean(axis=0) - X[tr][y[tr] == 0].mean(axis=0))
        sel = np.argsort(-d)[:top_k]
        Xtr, Xte = X[tr][:, sel], X[te][:, sel]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
        clf.fit(Xtr, y[tr])
        if len(np.unique(y[te])) >= 2:
            aurocs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aurocs)) if aurocs else float('nan')


def bootstrap_auroc(X: np.ndarray, y: np.ndarray, n_iter: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return (float('nan'), float('nan'), float('nan'))
    aurocs = []
    for _ in range(n_iter):
        bp = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn])
        a = cv_auroc(X[idx], y[idx])
        if not np.isnan(a):
            aurocs.append(a)
    if not aurocs:
        return (float('nan'), float('nan'), float('nan'))
    aurocs = np.array(aurocs)
    return float(aurocs.mean()), float(np.quantile(aurocs, 0.025)), float(np.quantile(aurocs, 0.975))


def main() -> None:
    p6_results_path = P6 / 'phase6_results.json'
    p6b_results_path = P6B / 'phase6b_results.json'

    if not p6_results_path.exists():
        print(f'No Phase 6 results yet at {p6_results_path}'); return
    if not p6b_results_path.exists():
        print(f'No Phase 6b verdicts yet at {p6b_results_path}'); return

    p6 = json.loads(p6_results_path.read_text())
    p6b = json.loads(p6b_results_path.read_text())

    verdicts = {iid: classify(r) for iid, r in p6b.items()}
    from collections import Counter
    dist = Counter(verdicts.values())
    print(f'\n=== Phase 6c PREVIEW (N={len(p6b)} labeled, Phase 6 progress: {len(p6)}) ===')
    print(f'Verdict distribution: {dict(dist)}')

    # Strict labels (partial = 0)
    valid_iids = [iid for iid, v in verdicts.items() if v != 'env_mismatch']
    y_strict = np.array([1 if verdicts[iid] == 'solves' else 0 for iid in valid_iids], dtype=int)
    y_loose = np.array([1 if verdicts[iid] in ('solves', 'partial') else 0 for iid in valid_iids], dtype=int)
    print(f'Valid (non-env_mismatch): N={len(valid_iids)}, strict_pos={int(y_strict.sum())}, loose_pos={int(y_loose.sum())}')

    if len(valid_iids) < 4:
        print('N<4 — skipping AUROC (not enough data)'); return

    out: dict = {'n_total': len(p6b), 'n_valid': len(valid_iids), 'verdict_dist': dict(dist), 'probes': {}}

    print(f'\n{"layer-pos":18s} {"strict_AUROC":>13s} {"strict_CI":>22s} {"loose_AUROC":>13s} {"loose_CI":>22s}')
    print('-' * 95)

    for L, pos in TOP_PROBES:
        Xs = []
        for iid in valid_iids:
            v = load_pos_mean(iid, L, pos)
            Xs.append(v)
        if any(v is None for v in Xs):
            missing = sum(1 for v in Xs if v is None)
            print(f'L{L:2d} {pos:13s}  -- skipped (missing {missing} captures)')
            continue
        X_all = np.stack(Xs, axis=0)

        results: dict = {}
        for label_name, y in [('strict', y_strict), ('loose', y_loose)]:
            if y.sum() < 2 or len(y) - y.sum() < 2:
                results[label_name] = {'auroc_cv': None, 'reason': 'insufficient class balance'}
                continue
            au_cv = cv_auroc(X_all, y)
            au_mean, au_lo, au_hi = bootstrap_auroc(X_all, y, n_iter=200)  # 200 fast preview
            results[label_name] = {
                'auroc_cv': round(au_cv, 4),
                'bootstrap_mean': round(au_mean, 4),
                'bootstrap_ci_95': [round(au_lo, 4), round(au_hi, 4)],
            }

        s = results.get('strict', {})
        l = results.get('loose', {})
        s_str = f'{s.get("auroc_cv", "NA"):>13}' if s.get('auroc_cv') is None else f'{s["auroc_cv"]:>13.4f}'
        s_ci = f'[{s["bootstrap_ci_95"][0]:.3f},{s["bootstrap_ci_95"][1]:.3f}]' if s.get('bootstrap_ci_95') else '       --'
        l_str = f'{l.get("auroc_cv", "NA"):>13}' if l.get('auroc_cv') is None else f'{l["auroc_cv"]:>13.4f}'
        l_ci = f'[{l["bootstrap_ci_95"][0]:.3f},{l["bootstrap_ci_95"][1]:.3f}]' if l.get('bootstrap_ci_95') else '       --'
        print(f'L{L:2d} {pos:13s}  {s_str} {s_ci:>22s} {l_str} {l_ci:>22s}')
        out['probes'][f'L{L}_{pos}'] = results

    OUT.write_text(json.dumps(out, indent=2))
    print(f'\nWrote {OUT}')


if __name__ == '__main__':
    main()
