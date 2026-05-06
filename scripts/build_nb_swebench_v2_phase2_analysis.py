"""Generate notebooks/nb_swebench_v2_phase2_analysis.ipynb — differential analysis on Phase 1 data.

Loads 20 captures from Drive, labels by patch_bytes>0, runs:
- per (layer, position_label) diff-of-means + bootstrap CI
- 25 logistic-regression probes with 4-fold stratified CV → AUROC
- multi-probe ensemble lift
- leave-one-repo-out cross-repo holdout
- pairwise probe orthogonality (cosine sim)
- AUROC heatmap layer × position

Run from repo root:
    python3 scripts/build_nb_swebench_v2_phase2_analysis.py
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "nb_swebench_v2_phase2_analysis.ipynb"


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": src.lstrip("\n").rstrip() + "\n",
        "outputs": [],
        "execution_count": None,
    }


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip("\n").rstrip() + "\n"}


cells: list[dict] = [
    md("""
# SWE-bench Pro Failure Anatomy — Phase 2 Differential Analysis

Phase 1 produced ~12k capture vectors across 20 stratified Python problems on Qwen3.6-27B.
Soft-pass label: `patch_bytes > 0` (11 success / 9 fail).

This notebook tests whether SAE residual-stream features at decision points
distinguish patch-success from patch-fail traces. Methodology mirrors
`paper-2 multi-probe + nb45/nb46 ensemble methodology`:

- **Single probe per (layer, position)**: 5 layers × 5 position labels = 25 probes
- **Diff-of-means** with bootstrap CI per (layer, position)
- **Multi-probe ensemble** lift vs single best (nb45 analog)
- **Cross-repo holdout** (leave-one-repo-out, 3 repos) — generalization test (nb46 analog)
- **Pairwise probe orthogonality** via cosine sim between weight vectors
- **AUROC heatmap** layer × position landscape

**Decision criteria**:
- 🟢 max AUROC ≥ 0.75 with cross-repo holdout ≥ 0.65 → strong signal, paper-worthy
- 🟡 max AUROC 0.65-0.75 → moderate, refine in Phase 1.5
- 🔴 max AUROC < 0.65 across all (layer, position) → no signal, pivot
"""),
    code("""
# 1) Install — analysis-only deps (no GPU needed for this notebook)
!pip install -q scikit-learn matplotlib seaborn safetensors huggingface-hub
"""),
    code("""
# 2) Mount Drive (where Phase 1 captures + traces live)
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE_ROOT = '/content/drive/MyDrive/openinterp_runs/swebench_v1_phase1'
assert os.path.exists(DRIVE_ROOT), f'Phase 1 dir not found: {DRIVE_ROOT}'
print('Drive root:', DRIVE_ROOT)
"""),
    code("""
# 3) Pull harness repo (latest)
import subprocess, sys
HARNESS_PATH = '/content/openinterp-swebench-harness'
GITHUB_URL = 'https://github.com/OpenInterpretability/openinterp-swebench-harness'
if not os.path.exists(HARNESS_PATH):
    subprocess.run(['git', 'clone', GITHUB_URL, HARNESS_PATH], check=True)
else:
    subprocess.run(['git', '-C', HARNESS_PATH, 'pull', '--quiet'], check=False)
sys.path.insert(0, HARNESS_PATH)
print('Harness HEAD:', subprocess.run(['git', '-C', HARNESS_PATH, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip())
"""),
    code("""
# 4) Load phase1_report.json + walk captures dir + assemble labels
import json
from pathlib import Path
import re

with open(f'{DRIVE_ROOT}/phase1_report.json') as f:
    report = json.load(f)

instance_results = report['instance_results']
print(f'Phase 1 instances: {len(instance_results)}')

# Build label dict + repo dict
labels: dict[str, int] = {}
repos: dict[str, str] = {}
patch_bytes: dict[str, int] = {}
for iid, r in instance_results.items():
    pb = r.get('patch_bytes', 0)
    labels[iid] = 1 if pb > 0 else 0
    patch_bytes[iid] = pb
    # Extract repo from instance_id (e.g. 'instance_internetarchive__openlibrary-...' -> 'openlibrary')
    m = re.match(r'instance_([^_]+)__([^-]+)', iid)
    if m:
        owner, repo = m.group(1), m.group(2)
        repos[iid] = repo
    else:
        repos[iid] = 'unknown'

n_pos = sum(labels.values())
n_neg = len(labels) - n_pos
print(f'Labels: {n_pos} success / {n_neg} fail')
from collections import Counter
print(f'Repos: {Counter(repos.values())}')
"""),
    code("""
# 5) Load all captures: build (problem × layer × position_label) -> mean activation vector
import torch
from safetensors.torch import load_file
import numpy as np

CAPTURE_DIR = Path(DRIVE_ROOT) / 'captures'
LAYERS = [11, 23, 31, 43, 55]
POSITIONS = ['think_start', 'think_mid', 'think_end', 'pre_tool', 'turn_end']

# Indexed: data[layer][position] -> dict {instance_id: mean_activation_vector (numpy)}
data: dict[int, dict[str, dict[str, np.ndarray]]] = {L: {p: {} for p in POSITIONS} for L in LAYERS}

for iid in instance_results:
    weights_path = CAPTURE_DIR / f'{iid}.safetensors'
    meta_path = CAPTURE_DIR / f'{iid}.meta.json'
    if not weights_path.exists() or not meta_path.exists():
        print(f'SKIP {iid}: missing capture files')
        continue
    meta = json.loads(meta_path.read_text())
    tensors = load_file(str(weights_path))

    # Group keys by (layer, position_label)
    grouped: dict[tuple[int, str], list[np.ndarray]] = {}
    for rec in meta['records']:
        L = rec['layer']
        pos = rec['position_label']
        if L not in LAYERS or pos not in POSITIONS:
            continue
        key = rec['activation_key']
        if key not in tensors:
            continue
        vec = tensors[key].to(torch.float32).numpy()
        grouped.setdefault((L, pos), []).append(vec)

    # Mean per (layer, position)
    for (L, pos), vecs in grouped.items():
        data[L][pos][iid] = np.mean(np.stack(vecs, axis=0), axis=0)

# Coverage check
for L in LAYERS:
    for pos in POSITIONS:
        n_with = len(data[L][pos])
        if n_with < 15:
            print(f'WARN: L{L} {pos:13s} only {n_with}/{len(instance_results)} problems have data')

print('\\nLoaded captures.')
print(f'Per (layer, position): {len(data[LAYERS[0]][POSITIONS[0]])} problems × {data[LAYERS[0]][POSITIONS[0]][next(iter(data[LAYERS[0]][POSITIONS[0]]))].shape[-1]}-dim')
"""),
    code("""
# 6) Diff-of-means per (layer, position) + bootstrap CI on top-K
import numpy as np

def build_xy(data_lp: dict, labels: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    iids = sorted(data_lp.keys())
    X = np.stack([data_lp[i] for i in iids], axis=0)
    y = np.array([labels[i] for i in iids], dtype=int)
    return X, y, iids

def diff_of_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pos = X[y == 1].mean(axis=0)
    neg = X[y == 0].mean(axis=0)
    return pos - neg

def bootstrap_top_features(X: np.ndarray, y: np.ndarray, top_k: int = 20, n_boot: int = 200, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    diffs = diff_of_means(X, y)
    abs_diffs = np.abs(diffs)
    top_idx = np.argsort(-abs_diffs)[:top_k]

    boot_diffs = np.zeros((n_boot, X.shape[1]))
    n = len(y)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_diffs[b] = diff_of_means(X[idx], y[idx])
    ci_lo = np.quantile(boot_diffs, 0.025, axis=0)
    ci_hi = np.quantile(boot_diffs, 0.975, axis=0)
    sign_consistent = (np.sign(ci_lo) == np.sign(ci_hi)) & (ci_lo != 0)

    return {
        'top_idx': top_idx.tolist(),
        'top_diff': diffs[top_idx].tolist(),
        'top_abs_diff': abs_diffs[top_idx].tolist(),
        'top_ci_lo': ci_lo[top_idx].tolist(),
        'top_ci_hi': ci_hi[top_idx].tolist(),
        'top_sign_consistent': sign_consistent[top_idx].tolist(),
        'n_features_total': int(X.shape[1]),
        'n_significant': int(sign_consistent.sum()),
    }

dm_results: dict[tuple[int, str], dict] = {}
for L in LAYERS:
    for pos in POSITIONS:
        if len(data[L][pos]) < 10:
            continue
        X, y, _ = build_xy(data[L][pos], labels)
        if y.sum() < 3 or (len(y) - y.sum()) < 3:
            continue
        dm_results[(L, pos)] = bootstrap_top_features(X, y)

print(f'Diff-of-means computed for {len(dm_results)} (layer, position) groups')
print('\\nTop-3 (layer, position) by # significantly-different features (sign consistent in 95% CI):')
sorted_groups = sorted(dm_results.items(), key=lambda kv: -kv[1]['n_significant'])
for (L, pos), r in sorted_groups[:5]:
    print(f'  L{L:2d} {pos:13s}: {r[\"n_significant\"]:5d}/{r[\"n_features_total\"]} features signif. | top |diff|={r[\"top_abs_diff\"][0]:.3f}')
"""),
    code("""
# 7) Single-probe LR: 25 probes (5 layers × 5 positions), 4-fold stratified CV → AUROC
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def probe_auroc_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 4, top_k_features: int | None = None, C: float = 1.0, seed: int = 42) -> dict:
    if top_k_features is not None and top_k_features < X.shape[1]:
        diffs = np.abs(diff_of_means(X, y))
        sel = np.argsort(-diffs)[:top_k_features]
        X = X[:, sel]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs: list[float] = []
    coefs: list[np.ndarray] = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=C, max_iter=2000, class_weight='balanced')
        clf.fit(Xtr, y[train_idx])
        ytrue = y[test_idx]
        if len(np.unique(ytrue)) < 2:
            continue
        yhat = clf.predict_proba(Xte)[:, 1]
        aurocs.append(roc_auc_score(ytrue, yhat))
        coefs.append(clf.coef_[0])
    if not aurocs:
        return {'auroc_mean': float('nan'), 'auroc_std': float('nan'), 'coef_mean': None, 'n_folds': 0}
    return {
        'auroc_mean': float(np.mean(aurocs)),
        'auroc_std': float(np.std(aurocs)),
        'auroc_per_fold': [float(a) for a in aurocs],
        'coef_mean': np.mean(np.stack(coefs, axis=0), axis=0),
        'n_folds': len(aurocs),
    }

# Use top-50 features per (layer, position) — N=20 with d=5120 would massively overfit
TOP_K_FEATURES = 50

probe_results: dict[tuple[int, str], dict] = {}
for L in LAYERS:
    for pos in POSITIONS:
        if len(data[L][pos]) < 10:
            continue
        X, y, _ = build_xy(data[L][pos], labels)
        if y.sum() < 3 or (len(y) - y.sum()) < 3:
            continue
        probe_results[(L, pos)] = probe_auroc_cv(X, y, n_splits=4, top_k_features=TOP_K_FEATURES)

print(f'Probes trained: {len(probe_results)}')
print('\\nAUROC per (layer, position), sorted desc:')
sorted_probes = sorted(probe_results.items(), key=lambda kv: -kv[1]['auroc_mean'])
for (L, pos), r in sorted_probes:
    print(f'  L{L:2d} {pos:13s}: AUROC={r[\"auroc_mean\"]:.3f} ± {r[\"auroc_std\"]:.3f}  (n_folds={r[\"n_folds\"]})')

best_lp = sorted_probes[0][0]
best_auroc = sorted_probes[0][1]['auroc_mean']
print(f'\\n=== Single-probe best: L{best_lp[0]} {best_lp[1]} → AUROC {best_auroc:.3f} ===')
"""),
    code("""
# 8) Multi-probe ensemble — averaged probability across all (layer, position) probes
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def ensemble_cv(data_dict: dict, labels: dict, layers: list, positions: list, n_splits: int = 4, top_k: int = 50, seed: int = 42) -> dict:
    # Common iids (intersection)
    common = None
    for L in layers:
        for pos in positions:
            if len(data_dict[L][pos]) < 10:
                continue
            ids = set(data_dict[L][pos].keys())
            common = ids if common is None else (common & ids)
    if not common:
        return {'auroc_mean': float('nan')}
    iids_sorted = sorted(common)
    y = np.array([labels[i] for i in iids_sorted], dtype=int)

    # Build per-(L, pos) feature matrix
    Xs: list[tuple[tuple[int, str], np.ndarray]] = []
    for L in layers:
        for pos in positions:
            if len(data_dict[L][pos]) < 10:
                continue
            X = np.stack([data_dict[L][pos][i] for i in iids_sorted], axis=0)
            Xs.append(((L, pos), X))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aurocs = []
    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        ytrue = y[test_idx]
        if len(np.unique(ytrue)) < 2:
            continue
        # Soft-vote average across probes
        probs = np.zeros(len(test_idx))
        n_used = 0
        for (L, pos), X in Xs:
            diffs = np.abs(diff_of_means(X[train_idx], y[train_idx]))
            sel = np.argsort(-diffs)[:top_k]
            Xs_train = X[train_idx][:, sel]
            Xs_test = X[test_idx][:, sel]
            scaler = StandardScaler()
            Xs_train = scaler.fit_transform(Xs_train)
            Xs_test = scaler.transform(Xs_test)
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
            clf.fit(Xs_train, y[train_idx])
            probs += clf.predict_proba(Xs_test)[:, 1]
            n_used += 1
        if n_used == 0:
            continue
        probs /= n_used
        fold_aurocs.append(roc_auc_score(ytrue, probs))
    return {
        'auroc_mean': float(np.mean(fold_aurocs)) if fold_aurocs else float('nan'),
        'auroc_std': float(np.std(fold_aurocs)) if fold_aurocs else float('nan'),
        'n_folds': len(fold_aurocs),
        'n_probes': len(Xs),
    }

ens = ensemble_cv(data, labels, LAYERS, POSITIONS, n_splits=4, top_k=TOP_K_FEATURES)
print(f'Ensemble (all {ens[\"n_probes\"]} probes): AUROC = {ens[\"auroc_mean\"]:.3f} ± {ens[\"auroc_std\"]:.3f}')
print(f'Lift vs single-probe best ({best_auroc:.3f}): {ens[\"auroc_mean\"] - best_auroc:+.3f}')
"""),
    code("""
# 9) Cross-repo holdout — leave-one-repo-out CV (generalization check, nb46 analog)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def loro_auroc(X_dict: dict, y_dict: dict, repo_dict: dict, top_k: int = 50) -> dict:
    iids = sorted(X_dict.keys())
    X_full = np.stack([X_dict[i] for i in iids], axis=0)
    y_full = np.array([y_dict[i] for i in iids], dtype=int)
    repos_arr = np.array([repo_dict[i] for i in iids])
    unique_repos = sorted(set(repos_arr))

    per_repo: dict[str, float | None] = {}
    aurocs = []
    for held in unique_repos:
        train_mask = repos_arr != held
        test_mask = repos_arr == held
        if y_full[train_mask].sum() < 2 or (len(y_full[train_mask]) - y_full[train_mask].sum()) < 2:
            per_repo[held] = None
            continue
        ytest = y_full[test_mask]
        if len(np.unique(ytest)) < 2:
            per_repo[held] = None
            continue
        diffs = np.abs(diff_of_means(X_full[train_mask], y_full[train_mask]))
        sel = np.argsort(-diffs)[:top_k]
        Xtr = X_full[train_mask][:, sel]
        Xte = X_full[test_mask][:, sel]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')
        clf.fit(Xtr, y_full[train_mask])
        yhat = clf.predict_proba(Xte)[:, 1]
        a = roc_auc_score(ytest, yhat)
        per_repo[held] = float(a)
        aurocs.append(a)
    return {
        'mean_auroc': float(np.mean(aurocs)) if aurocs else float('nan'),
        'per_repo': per_repo,
    }

print('=== Leave-One-Repo-Out (per layer, position) ===')
loro_grid: dict[tuple[int, str], dict] = {}
for L in LAYERS:
    for pos in POSITIONS:
        if len(data[L][pos]) < 10:
            continue
        loro_grid[(L, pos)] = loro_auroc(data[L][pos], labels, repos, top_k=TOP_K_FEATURES)

sorted_loro = sorted(loro_grid.items(), key=lambda kv: -(kv[1]['mean_auroc'] if not np.isnan(kv[1]['mean_auroc']) else -1))
for (L, pos), r in sorted_loro[:10]:
    per = r['per_repo']
    formatted = ', '.join(f'{k}={v:.3f}' if v is not None else f'{k}=N/A' for k, v in per.items())
    print(f'  L{L:2d} {pos:13s}: mean={r[\"mean_auroc\"]:.3f}  ({formatted})')

best_loro = sorted_loro[0][0]
print(f'\\n=== Best cross-repo: L{best_loro[0]} {best_loro[1]} → mean LORO AUROC {sorted_loro[0][1][\"mean_auroc\"]:.3f} ===')
"""),
    code("""
# 10) Probe orthogonality — pairwise cosine similarity between coef vectors
import numpy as np

probe_keys = sorted(probe_results.keys())
coefs = np.stack([probe_results[k]['coef_mean'] for k in probe_keys if probe_results[k]['coef_mean'] is not None])
labels_str = [f'L{k[0]}/{k[1][:9]}' for k in probe_keys if probe_results[k]['coef_mean'] is not None]

norms = np.linalg.norm(coefs, axis=1, keepdims=True)
unit = coefs / (norms + 1e-10)
cos_mat = unit @ unit.T

n = len(probe_keys)
# Stats: off-diagonal mean, max
mask = ~np.eye(n, dtype=bool)
off_mean = float(cos_mat[mask].mean())
off_max = float(cos_mat[mask].max())
off_min = float(cos_mat[mask].min())
print(f'Probe orthogonality stats (off-diagonal cosine):')
print(f'  mean = {off_mean:+.3f}')
print(f'  max  = {off_max:+.3f}  (most redundant pair)')
print(f'  min  = {off_min:+.3f}  (most opposite pair)')
print(f'  N pairs above 0.5 (redundant): {int((cos_mat[mask] > 0.5).sum() / 2)}')
print(f'  N pairs below 0.1 (orthogonal): {int((np.abs(cos_mat[mask]) < 0.1).sum() / 2)}')

# Find top-1 most redundant pair
ut = np.triu_indices(n, k=1)
upper = cos_mat[ut]
top_redund_idx = np.argmax(upper)
i_red, j_red = ut[0][top_redund_idx], ut[1][top_redund_idx]
print(f'\\nMost redundant: {labels_str[i_red]} <-> {labels_str[j_red]}: cos={cos_mat[i_red, j_red]:.3f}')
"""),
    code("""
# 11) Plot AUROC heatmap (layer × position)
import numpy as np
import matplotlib.pyplot as plt

heat = np.full((len(LAYERS), len(POSITIONS)), np.nan)
heat_loro = np.full((len(LAYERS), len(POSITIONS)), np.nan)
for i, L in enumerate(LAYERS):
    for j, pos in enumerate(POSITIONS):
        if (L, pos) in probe_results:
            heat[i, j] = probe_results[(L, pos)]['auroc_mean']
        if (L, pos) in loro_grid:
            heat_loro[i, j] = loro_grid[(L, pos)]['mean_auroc']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, mat, title in zip(axes, [heat, heat_loro], ['Within-set 4-fold CV', 'Leave-one-repo-out']):
    im = ax.imshow(mat, vmin=0.4, vmax=1.0, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(POSITIONS)))
    ax.set_xticklabels(POSITIONS, rotation=30, ha='right')
    ax.set_yticks(range(len(LAYERS)))
    ax.set_yticklabels([f'L{L}' for L in LAYERS])
    ax.set_title(f'AUROC — {title}')
    for i in range(len(LAYERS)):
        for j in range(len(POSITIONS)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9, color='black' if v > 0.5 else 'white')
    plt.colorbar(im, ax=ax, label='AUROC')
plt.tight_layout()
plt.savefig(f'{DRIVE_ROOT}/phase2_auroc_heatmap.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved: {DRIVE_ROOT}/phase2_auroc_heatmap.png')
"""),
    code("""
# 12) Verdict + save report
import json
import numpy as np

best_single_auroc = max(r['auroc_mean'] for r in probe_results.values())
best_single_lp = max(probe_results.items(), key=lambda kv: kv[1]['auroc_mean'])[0]

best_loro_auroc_finite = max(
    (r['mean_auroc'] for r in loro_grid.values() if not np.isnan(r['mean_auroc'])),
    default=float('nan'),
)
best_loro_lp = None
for k, r in loro_grid.items():
    if r['mean_auroc'] == best_loro_auroc_finite:
        best_loro_lp = k
        break

ensemble_auroc = ens['auroc_mean']

if best_single_auroc >= 0.75 and (np.isnan(best_loro_auroc_finite) or best_loro_auroc_finite >= 0.65):
    verdict = 'STRONG_SIGNAL'
    color = '🟢'
elif best_single_auroc >= 0.65:
    verdict = 'MODERATE_SIGNAL'
    color = '🟡'
else:
    verdict = 'NO_SIGNAL'
    color = '🔴'

print(f'\\n=========== Phase 2 Verdict ===========')
print(f'Single-probe best: L{best_single_lp[0]} {best_single_lp[1]} → AUROC {best_single_auroc:.3f}')
print(f'Cross-repo (LORO) best: {best_loro_lp} → AUROC {best_loro_auroc_finite:.3f}')
print(f'Multi-probe ensemble: AUROC {ensemble_auroc:.3f} (lift {ensemble_auroc - best_single_auroc:+.3f})')
print(f'\\n{color} Verdict: {verdict}')

if verdict == 'STRONG_SIGNAL':
    print('  → Phase 3 attribution patching on top features at best (layer, position)')
elif verdict == 'MODERATE_SIGNAL':
    print('  → Phase 1.5: redo with max_turns=50 + redefined soft_pass; analyze on bigger N')
else:
    print('  → Pivot: features do not separate patch/no-patch. Likely substrate gap or label too noisy.')

# Save report
out = {
    'phase': 2,
    'n_problems': len(labels),
    'n_pos': int(sum(labels.values())),
    'n_neg': int(len(labels) - sum(labels.values())),
    'top_k_features_per_probe': TOP_K_FEATURES,
    'single_probe_best': {
        'layer': best_single_lp[0],
        'position': best_single_lp[1],
        'auroc': best_single_auroc,
    },
    'cross_repo_loro_best': {
        'layer': best_loro_lp[0] if best_loro_lp else None,
        'position': best_loro_lp[1] if best_loro_lp else None,
        'auroc': best_loro_auroc_finite,
    },
    'ensemble_auroc': ensemble_auroc,
    'orthogonality': {
        'mean_off_diag_cosine': off_mean,
        'max_off_diag_cosine': off_max,
        'most_redundant_pair': [labels_str[i_red], labels_str[j_red]],
    },
    'verdict': verdict,
    'probe_grid_within': {f'L{k[0]}_{k[1]}': v['auroc_mean'] for k, v in probe_results.items()},
    'probe_grid_loro': {f'L{k[0]}_{k[1]}': v['mean_auroc'] for k, v in loro_grid.items()},
}
with open(f'{DRIVE_ROOT}/phase2_report.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f'\\nWrote {DRIVE_ROOT}/phase2_report.json')
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "None",
        "colab": {"name": "nb_swebench_v2_phase2_analysis.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
