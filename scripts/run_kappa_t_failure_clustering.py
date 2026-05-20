"""Path E core analysis: cluster the 60 failed κ_t trajectories by shape to discover
failure-mode taxonomy.

Approach:
  1. Load per-trace κ_t time series (output of run_kappa_t_per_trace_export.py).
  2. For each failure, interpolate κ_t shape to fixed grid (50 points) and normalize.
  3. K-means cluster shapes; sweep K=2..6; pick best by silhouette.
  4. For each cluster, compute behavioral correlates from traces (tool usage, repo, etc).
  5. Optional: shuffled-label baseline to test cluster meaningfulness.
  6. Save cluster assignments + centroids + summary table.

Output: clustering_results.json + diagnostic plots.
"""
import os, json, warnings
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.interpolate import interp1d
from collections import Counter

DRIVE = '/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6'
KAPPA_PATH = os.path.join(DRIVE, 'kappa_t_per_trace.json')
TRACES_DIR = os.path.join(DRIVE, 'traces')
OUT_JSON = os.path.join(DRIVE, 'kappa_t_failure_clusters.json')

GRID_N = 50  # resample each trajectory to 50 points
N_K_SWEEP = range(2, 7)
RNG = np.random.default_rng(42)

# === 1. Load ===
with open(KAPPA_PATH) as f:
    kappa = json.load(f)

fails = {iid: r for iid, r in kappa.items() if not r['success']}
succs = {iid: r for iid, r in kappa.items() if r['success']}
print(f'Loaded {len(kappa)} traces: {len(succs)} succ, {len(fails)} fail')

# === 2. Interpolate to fixed grid ===
def interpolate_shape(kappa_t, grid_n=GRID_N):
    arr = np.asarray(kappa_t, dtype=float)
    if len(arr) < 3:
        return None
    # Drop leading NaN (κ_t paper convention: first W-1 turns are NaN)
    mask = ~np.isnan(arr)
    if mask.sum() < 3:
        return None
    valid = arr[mask]
    if valid.std() < 1e-9:
        return None
    x_old = np.linspace(0, 1, len(valid))
    x_new = np.linspace(0, 1, grid_n)
    f = interp1d(x_old, valid, kind='linear')
    return f(x_new)

shapes_fail = []
iids_fail = []
for iid, rec in fails.items():
    s = interpolate_shape(rec['kappa_t'])
    if s is None:
        continue
    shapes_fail.append(s)
    iids_fail.append(iid)
shapes_fail = np.array(shapes_fail)
print(f'Failure shape matrix: {shapes_fail.shape}')

# === 3. K-means sweep ===
print('\nK-means sweep (K=2..6):')
best_K = None
best_sil = -np.inf
sil_by_k = {}
for K in N_K_SWEEP:
    km = KMeans(n_clusters=K, n_init=20, random_state=42).fit(shapes_fail)
    if K > 1:
        sil = silhouette_score(shapes_fail, km.labels_)
        sil_by_k[K] = sil
        marker = ' ← best' if sil > best_sil else ''
        if sil > best_sil:
            best_sil = sil
            best_K = K
        print(f'  K={K}: silhouette = {sil:+.3f}{marker}')

print(f'\nSelected K={best_K} (silhouette={best_sil:.3f})')

# === 4. Final clustering at best K + shuffled-label baseline ===
km_final = KMeans(n_clusters=best_K, n_init=50, random_state=42).fit(shapes_fail)
labels = km_final.labels_
centroids = km_final.cluster_centers_

# Shuffled-label baseline: shuffle shape rows, re-fit, compare silhouette
shuf_sils = []
for seed in range(20):
    rng = np.random.default_rng(1000+seed)
    shapes_shuf = shapes_fail.copy()
    # Permute each shape's time axis independently — destroys temporal structure
    for i in range(len(shapes_shuf)):
        rng.shuffle(shapes_shuf[i])
    km_shuf = KMeans(n_clusters=best_K, n_init=10, random_state=seed).fit(shapes_shuf)
    shuf_sils.append(silhouette_score(shapes_shuf, km_shuf.labels_))
shuf_mean = np.mean(shuf_sils)
shuf_p95 = np.percentile(shuf_sils, 95)
gap = best_sil - shuf_mean

print(f'\nShuffled-time baseline silhouette: mean={shuf_mean:+.3f}, p95={shuf_p95:+.3f}')
print(f'Real-vs-shuffled gap: {gap:+.3f}')
verdict = 'STRONG (gap > 0.10)' if gap > 0.10 else 'MODERATE (gap > 0.05)' if gap > 0.05 else 'WEAK'
print(f'Verdict: {verdict}')

# === 5. Per-cluster behavioral correlates ===
print(f'\n=== Per-cluster behavioral correlates (K={best_K}) ===')

cluster_summary = {}
for c in range(best_K):
    members = [iids_fail[i] for i in range(len(iids_fail)) if labels[i] == c]
    n = len(members)
    # Aggregate stats from kappa records + traces
    n_turns_list = [fails[iid]['phase6_n_turns'] for iid in members]
    wall_list = [fails[iid].get('wall_seconds', 0) for iid in members]
    repos = Counter()
    finish_reasons = Counter()
    primary_tools_all = Counter()
    avg_kappa = []
    avg_kappa_early = []
    avg_kappa_late = []
    for iid in members:
        rec = fails[iid]
        repos[rec['repo']] += 1
        finish_reasons[rec['finish_reason']] += 1
        # Average κ_t (skip leading NaN)
        kt = [x for x in rec['kappa_t'] if not np.isnan(x)]
        if kt:
            avg_kappa.append(float(np.mean(kt)))
            half = len(kt) // 2
            if half >= 1:
                avg_kappa_early.append(float(np.mean(kt[:half])))
                avg_kappa_late.append(float(np.mean(kt[half:])))
        # Tool usage breakdown from trace JSON
        trace_path = os.path.join(TRACES_DIR, f'{iid}.json')
        if os.path.exists(trace_path):
            with open(trace_path) as f:
                trace = json.load(f)
            for turn in trace.get('turns', []):
                tc = turn.get('tool_calls', [])
                if tc:
                    primary_tools_all[tc[0]['name']] += 1
                else:
                    primary_tools_all['none'] += 1

    cluster_summary[c] = {
        'n': n,
        'centroid': centroids[c].tolist(),
        'n_turns_mean': float(np.mean(n_turns_list)),
        'n_turns_std': float(np.std(n_turns_list)),
        'wall_seconds_mean': float(np.mean(wall_list)),
        'avg_kappa_mean': float(np.mean(avg_kappa)) if avg_kappa else None,
        'avg_kappa_early': float(np.mean(avg_kappa_early)) if avg_kappa_early else None,
        'avg_kappa_late': float(np.mean(avg_kappa_late)) if avg_kappa_late else None,
        'top_repos': dict(repos.most_common(5)),
        'finish_reasons': dict(finish_reasons),
        'top_tools': dict(primary_tools_all.most_common(8)),
        'members': members,
    }
    print(f'\nCluster {c} (n={n}):')
    print(f'  Mean turns: {cluster_summary[c]["n_turns_mean"]:.1f} ± {cluster_summary[c]["n_turns_std"]:.1f}')
    print(f'  Mean κ̄ (whole / early-half / late-half): {cluster_summary[c]["avg_kappa_mean"]:.3f} / {cluster_summary[c]["avg_kappa_early"]:.3f} / {cluster_summary[c]["avg_kappa_late"]:.3f}')
    print(f'  Top repos: {cluster_summary[c]["top_repos"]}')
    print(f'  Top tools: {cluster_summary[c]["top_tools"]}')
    print(f'  Centroid shape (5 sample points): start={centroids[c][0]:.3f}, q25={centroids[c][12]:.3f}, mid={centroids[c][25]:.3f}, q75={centroids[c][37]:.3f}, end={centroids[c][49]:.3f}')

# === 6. Cross-cluster discrimination test ===
# If clusters truly differ, the κ̄ distributions should differ
from scipy.stats import kruskal
kappa_by_cluster = []
for c in range(best_K):
    members = [iids_fail[i] for i in range(len(iids_fail)) if labels[i] == c]
    kappa_by_cluster.append([cluster_summary[c]['avg_kappa_mean']])
# Use Kruskal-Wallis across per-trace κ̄ within each cluster
per_trace_kappa = {c: [] for c in range(best_K)}
for i, iid in enumerate(iids_fail):
    c = labels[i]
    kt = [x for x in fails[iid]['kappa_t'] if not np.isnan(x)]
    if kt:
        per_trace_kappa[c].append(float(np.mean(kt)))

if all(len(per_trace_kappa[c]) > 1 for c in range(best_K)):
    stat, p_kw = kruskal(*[per_trace_kappa[c] for c in range(best_K)])
    print(f'\nKruskal-Wallis on cluster κ̄ distributions: H={stat:.2f}, p={p_kw:.4f}')
else:
    p_kw = None
    print('\nKruskal-Wallis: insufficient samples per cluster')

# === 7. Save ===
out = {
    'best_K': int(best_K),
    'silhouette_real': float(best_sil),
    'silhouette_shuffled_mean': float(shuf_mean),
    'silhouette_shuffled_p95': float(shuf_p95),
    'silhouette_gap': float(gap),
    'verdict': verdict,
    'p_kruskal_wallis': float(p_kw) if p_kw else None,
    'silhouette_by_k': sil_by_k,
    'cluster_summary': cluster_summary,
    'failure_iids_ordered': iids_fail,
    'failure_labels': labels.tolist(),
    'grid_n': GRID_N,
    'n_failures_clustered': len(iids_fail),
    'n_successes_excluded': len(succs),
}
with open(OUT_JSON, 'w') as f:
    json.dump(out, f, indent=1)
print(f'\nSaved clustering results to {OUT_JSON}')
print('Done.')
