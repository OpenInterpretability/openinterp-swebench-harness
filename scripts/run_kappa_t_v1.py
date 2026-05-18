"""κ_t Cross-Probe Correlation Phase 1 — cardiac uncoupling test.

Hypothesis: multiple probes (different axes) become DECORRELATED during agent
failure trajectories, analogous to ICU cross-vital uncoupling preceding decompensation.

Setup:
1. Train 5 probes on Phase 6 residuals with DIFFERENT label axes:
   A. trace_failure (any failure)
   B. max_turns_failure (specific failure mode)
   C. long_trace (>= 20 turns)
   D. wall_time_slow (top 50%)
   E. repo_ansible (binary domain)

2. Use GroupKFold for out-of-fold predictions on ALL turns.

3. For each trace, compute κ_t per turn t = mean(|corr_ij|) over window [t-W, t+W]
   across the 5 probe score streams.

4. Test: κ_t at turn t predicts trace failure by turn t+K?
   - Build (κ_t value, future-failure-within-K-turns) pairs
   - Compute AUROC

Gate: κ_t-alone AUROC > 0.65 + gap vs shuffled κ_t > 0.10.
"""
import os, json, sys, warnings, time
import numpy as np
warnings.filterwarnings('ignore')

from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

OUT_PATH = '/tmp/kappa_t_phase1_out.txt'
out_f = open(OUT_PATH, 'w', buffering=1)
def p(*a):
    out_f.write(' '.join(str(x) for x in a) + '\n')
    out_f.flush()

DRIVE_ROOT = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
PHASE6_DIR = os.path.join(DRIVE_ROOT, 'swebench_v6_phase6')
CAPTURES_DIR = os.path.join(PHASE6_DIR, 'captures')
RESULTS_JSON = os.path.join(PHASE6_DIR, 'phase6_results.json')
OUT_DIR = os.path.join(DRIVE_ROOT, 'kappa_t_phase1')
os.makedirs(OUT_DIR, exist_ok=True)

PROBE_LAYER = 43  # best from Inner-Outer = L43 paired_concat
POS_INNER = 'think_end'
POS_OUTER = 'turn_end'

t_start = time.time()

# === Load Phase 6 metadata ===
with open(RESULTS_JSON) as f:
    phase6_results = json.load(f)
p(f'[{time.time()-t_start:.0f}s] Phase 6 instances: {len(phase6_results)}')

# === Build label dictionary per (iid, turn) ===
# Per-instance metadata: failure, max_turns, length, wall_seconds, repo

def repo_from_iid(iid):
    # iid format: instance_REPO__REPO-COMMIT-VERSION
    parts = iid.replace('instance_', '').split('__')
    return parts[0] if parts else 'unknown'

per_iid_meta = {}
for iid, info in phase6_results.items():
    finished = info.get('finished', False)
    fr = info.get('finish_reason', '')
    per_iid_meta[iid] = {
        'failure': 0 if (finished and fr == 'finish_tool') else 1,
        'max_turns': 1 if fr == 'max_turns' else 0,
        'n_turns': info.get('n_turns', 0),
        'wall_seconds': info.get('wall_seconds', 0.0),
        'repo': repo_from_iid(iid),
    }

n_turns_list = [m['n_turns'] for m in per_iid_meta.values()]
wall_list = [m['wall_seconds'] for m in per_iid_meta.values()]
n_turns_median = float(np.median(n_turns_list))
wall_median = float(np.median(wall_list))
p(f'  Median n_turns: {n_turns_median}, median wall: {wall_median:.0f}s')

# === Load paired residuals at L43 (only layer needed) ===
def load_residuals(iid):
    sp = os.path.join(CAPTURES_DIR, f'{iid}.safetensors')
    mp = os.path.join(CAPTURES_DIR, f'{iid}.meta.json')
    if not (os.path.exists(sp) and os.path.exists(mp)):
        return None
    with open(mp) as f:
        meta = json.load(f)
    by_key = {(r['turn_idx'], r['position_label'], r['layer']): r['activation_key'] for r in meta['records']}
    turns = sorted({r['turn_idx'] for r in meta['records']
                    if (r['turn_idx'], POS_INNER, PROBE_LAYER) in by_key
                    and (r['turn_idx'], POS_OUTER, PROBE_LAYER) in by_key})
    if not turns:
        return None
    out = []
    with safe_open(sp, framework='pt') as f:
        for t in turns:
            v_i = f.get_tensor(by_key[(t, POS_INNER, PROBE_LAYER)]).float().numpy()
            v_o = f.get_tensor(by_key[(t, POS_OUTER, PROBE_LAYER)]).float().numpy()
            out.append((t, np.concatenate([v_i, v_o])))  # paired_concat features (10240)
    return out

p(f'[{time.time()-t_start:.0f}s] Loading L{PROBE_LAYER} paired residuals (this took ~5min last run)...')
X_list, iid_list, turn_list = [], [], []
n_skipped = 0
for i, (iid, info) in enumerate(phase6_results.items()):
    res = load_residuals(iid)
    if res is None:
        n_skipped += 1
        continue
    for (t, vec) in res:
        X_list.append(vec)
        iid_list.append(iid)
        turn_list.append(t)
    if (i + 1) % 20 == 0:
        p(f'  [{i+1}/{len(phase6_results)}] {(time.time()-t_start)/60:.1f}min')

X = np.stack(X_list).astype(np.float32)
iid_arr = np.array(iid_list)
turn_arr = np.array(turn_list)
groups = iid_arr.copy()
p(f'[{time.time()-t_start:.0f}s] Loaded {len(X)} samples from {len(set(iid_arr))} instances ({n_skipped} skipped)')
p(f'  Feature shape: {X.shape}')

# === Build label arrays for 5 probes ===
y_axes = {
    'A_failure':     np.array([per_iid_meta[iid]['failure'] for iid in iid_arr], dtype=int),
    'B_max_turns':   np.array([per_iid_meta[iid]['max_turns'] for iid in iid_arr], dtype=int),
    'C_long_trace':  np.array([1 if per_iid_meta[iid]['n_turns'] >= n_turns_median else 0 for iid in iid_arr], dtype=int),
    'D_wall_slow':   np.array([1 if per_iid_meta[iid]['wall_seconds'] >= wall_median else 0 for iid in iid_arr], dtype=int),
    'E_repo_ansible': np.array([1 if per_iid_meta[iid]['repo'] == 'ansible' else 0 for iid in iid_arr], dtype=int),
}
for name, y in y_axes.items():
    p(f'  {name}: positive rate {100*y.mean():.0f}% ({y.sum()}/{len(y)})')

# === Train 5 probes with GroupKFold; collect out-of-fold predictions ===
SEED = 42
n_splits = 5
oof_scores = {name: np.zeros(len(X), dtype=np.float32) for name in y_axes}
probe_aurocs = {}

p(f'\n[{time.time()-t_start:.0f}s] Training 5 probes with GroupKFold...')
for name, y in y_axes.items():
    t_p = time.time()
    np.random.seed(SEED)
    gkf = GroupKFold(n_splits=n_splits)
    aurocs = []
    for tr, te in gkf.split(X, y, groups):
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X[tr]).astype(np.float32)
        Xte_s = scaler.transform(X[te]).astype(np.float32)
        clf = LogisticRegression(class_weight='balanced', max_iter=500, C=0.1, solver='liblinear')
        clf.fit(Xtr_s, y[tr])
        scores_te = clf.decision_function(Xte_s)
        oof_scores[name][te] = scores_te
        try: aurocs.append(roc_auc_score(y[te], scores_te))
        except ValueError: pass
    probe_aurocs[name] = float(np.mean(aurocs)) if aurocs else float('nan')
    p(f'  {name}: AUROC {probe_aurocs[name]:.3f}  ({time.time()-t_p:.0f}s)')

# === Build (trace, turn) → 5-dim score vector ===
# For each trace, organize scores in turn order
p(f'\n[{time.time()-t_start:.0f}s] Computing κ_t per (trace, turn)...')

# Group by iid: for each iid, sort turns, get scores
per_trace_scores = {}  # iid → array (n_turns_iid, 5)
for iid in set(iid_arr):
    mask = iid_arr == iid
    idxs = np.where(mask)[0]
    turns_iid = turn_arr[idxs]
    sort_perm = np.argsort(turns_iid)
    scores_iid = np.stack([oof_scores[name][idxs[sort_perm]] for name in y_axes], axis=1)  # (n_turns, 5)
    per_trace_scores[iid] = {
        'turns': turns_iid[sort_perm],
        'scores': scores_iid,
        'failure': per_iid_meta[iid]['failure'],
    }

# === Compute κ_t with moving window ===
WINDOW = 3  # turns radius (window size = 2W+1 = 7)
N_PROBES = len(y_axes)

def compute_kappa(scores_window):
    """scores_window: (window_len, n_probes). Return mean |pairwise corr|."""
    if scores_window.shape[0] < 3:
        return np.nan
    corr_mat = np.corrcoef(scores_window.T)  # (n_probes, n_probes)
    if np.isnan(corr_mat).any():
        return np.nan
    # off-diagonal |corr|
    mask = ~np.eye(N_PROBES, dtype=bool)
    return float(np.mean(np.abs(corr_mat[mask])))

# Per-trace, per-turn κ_t
per_trace_kappa = {}
for iid, d in per_trace_scores.items():
    n_t = len(d['turns'])
    kappa = np.full(n_t, np.nan)
    for t_idx in range(n_t):
        lo = max(0, t_idx - WINDOW)
        hi = min(n_t, t_idx + WINDOW + 1)
        kappa[t_idx] = compute_kappa(d['scores'][lo:hi])
    per_trace_kappa[iid] = kappa

# === Test 1: per-trace mean κ_t — does it differ between success and fail traces? ===
mean_kappa_success = []
mean_kappa_fail = []
for iid, kappa in per_trace_kappa.items():
    valid = kappa[~np.isnan(kappa)]
    if len(valid) == 0:
        continue
    if per_iid_meta[iid]['failure'] == 0:
        mean_kappa_success.append(np.mean(valid))
    else:
        mean_kappa_fail.append(np.mean(valid))

p(f'\n=== Test 1: per-trace mean κ_t ===')
p(f'  Success traces: N={len(mean_kappa_success)}, mean κ_t = {np.mean(mean_kappa_success):.3f} ± {np.std(mean_kappa_success):.3f}')
p(f'  Failed  traces: N={len(mean_kappa_fail)},  mean κ_t = {np.mean(mean_kappa_fail):.3f} ± {np.std(mean_kappa_fail):.3f}')

# Two-sample test
from scipy.stats import mannwhitneyu
if mean_kappa_success and mean_kappa_fail:
    stat, pval = mannwhitneyu(mean_kappa_success, mean_kappa_fail, alternative='two-sided')
    p(f'  Mann-Whitney U: stat={stat:.0f}, p={pval:.4f}')

# === Test 2: AUROC of mean-κ_t as trace failure predictor ===
trace_labels, trace_kappas = [], []
for iid, kappa in per_trace_kappa.items():
    valid = kappa[~np.isnan(kappa)]
    if len(valid) == 0:
        continue
    trace_labels.append(per_iid_meta[iid]['failure'])
    trace_kappas.append(np.mean(valid))

trace_labels = np.array(trace_labels)
trace_kappas = np.array(trace_kappas)
# κ_t HIGH = aligned = success expected → predict failure as (1 - κ_t) or use negative
try:
    auroc_kappa = roc_auc_score(trace_labels, -trace_kappas)  # high failure when low κ_t
    p(f'\n=== Test 2: AUROC of (−mean κ_t) → trace failure ===')
    p(f'  AUROC = {auroc_kappa:.3f}  (N={len(trace_labels)})')
except ValueError as e:
    p(f'\n  AUROC FAILED: {e}')
    auroc_kappa = float('nan')

# Random baseline (shuffle labels)
rng = np.random.default_rng(42)
shuffled_aurocs = []
for _ in range(100):
    sh = rng.permutation(trace_kappas)
    try:
        shuffled_aurocs.append(roc_auc_score(trace_labels, -sh))
    except ValueError:
        pass
shuf_mean = float(np.mean(shuffled_aurocs))
shuf_p95 = float(np.percentile(shuffled_aurocs, 95))
p(f'  Shuffled-κ_t baseline: mean {shuf_mean:.3f}, 95th-pct {shuf_p95:.3f}')
p(f'  Gap (real − shuffled): {auroc_kappa - shuf_mean:+.3f}')

# === Test 3: early κ_t predicting eventual failure (anticipation test) ===
# For each trace, take κ_t at first 5 turns; predict trace-level failure
early_kappa_per_trace = []
labels_for_early = []
for iid, kappa in per_trace_kappa.items():
    early = kappa[:5]
    valid = early[~np.isnan(early)]
    if len(valid) < 2:
        continue
    early_kappa_per_trace.append(np.mean(valid))
    labels_for_early.append(per_iid_meta[iid]['failure'])

if early_kappa_per_trace:
    try:
        auroc_early = roc_auc_score(labels_for_early, -np.array(early_kappa_per_trace))
        p(f'\n=== Test 3: EARLY κ_t (first 5 turns) → eventual failure ===')
        p(f'  AUROC = {auroc_early:.3f}  (N={len(labels_for_early)})')
        p(f'  Anticipation strength: early κ_t predicts trace outcome at AUROC {auroc_early:.3f}')
    except ValueError:
        pass

# === Gate decision ===
p('\n' + '=' * 78)
p('PRE-REGISTERED GATE CHECK')
p('=' * 78)
G1 = auroc_kappa > 0.65
G2 = (auroc_kappa - shuf_mean) > 0.10
G3 = (pval < 0.05) if 'pval' in dir() else False
G4 = (auroc_early > 0.60) if 'auroc_early' in dir() else False

gates = [
    (f'G1 κ_t AUROC > 0.65', auroc_kappa, '>0.65', G1),
    (f'G2 gap vs shuffled > 0.10', auroc_kappa - shuf_mean, '>0.10', G2),
    (f'G3 Mann-Whitney p < 0.05', pval if 'pval' in dir() else float('nan'), '<0.05', G3),
    (f'G4 early κ_t (first 5) AUROC > 0.60', auroc_early if 'auroc_early' in dir() else float('nan'), '>0.60', G4),
]
for name, val, thr, g in gates:
    p(f'  {"PASS" if g else "FAIL"} {name}: actual={val:+.3f} (need {thr})')

n_pass = sum(1 for _,_,_,g in gates if g)
p(f'\n=== {n_pass}/4 PASS ===')
if n_pass >= 3:
    p('GREEN — κ_t is a meaningful cross-probe signal. Build out for SDK + paper.')
elif n_pass == 2:
    p('YELLOW — partial signal. Investigate before scaling.')
else:
    p('RED — κ_t does not detect cross-probe decorrelation here. Walk-back.')

# === Save ===
with open(os.path.join(OUT_DIR, 'kappa_t_phase1_results.json'), 'w') as f:
    json.dump({
        'probe_aurocs': probe_aurocs,
        'n_total_samples': int(len(X)),
        'n_instances': int(len(set(iid_arr))),
        'n_skipped': int(n_skipped),
        'mean_kappa_success': float(np.mean(mean_kappa_success)) if mean_kappa_success else None,
        'mean_kappa_fail': float(np.mean(mean_kappa_fail)) if mean_kappa_fail else None,
        'mw_pval': float(pval) if 'pval' in dir() else None,
        'auroc_mean_kappa': float(auroc_kappa) if not np.isnan(auroc_kappa) else None,
        'auroc_shuffled_mean': float(shuf_mean),
        'auroc_shuffled_p95': float(shuf_p95),
        'gap_vs_shuffled': float(auroc_kappa - shuf_mean) if not np.isnan(auroc_kappa) else None,
        'auroc_early_kappa': float(auroc_early) if 'auroc_early' in dir() else None,
        'n_pass': n_pass,
        'window': WINDOW,
        'probe_layer': PROBE_LAYER,
        'n_probes': N_PROBES,
    }, f, indent=2)

p(f'\nTotal time: {(time.time()-t_start)/60:.1f}min')
p(f'Saved: {os.path.join(OUT_DIR, "kappa_t_phase1_results.json")}')
out_f.close()
