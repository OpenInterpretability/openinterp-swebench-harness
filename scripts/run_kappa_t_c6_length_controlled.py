"""κ_t Option-B test — length-controlled slope.

Three variants of "first-N-turns slope" + early-vs-late-half test.

If slope-on-first-10-turns shows succ > fail and survives, headline is rescued
(temporal buildup independent of total trace length). If all variants fail,
slope is fundamentally length-confounded → use option A reframe.

Run: python3 /tmp/run_kappa_t_c6_length_controlled.py
"""
import os, json, time, warnings
import numpy as np
warnings.filterwarnings('ignore')

from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import mannwhitneyu

OUT_PATH = '/tmp/kappa_t_c6_out.txt'
out_f = open(OUT_PATH, 'w', buffering=1)
def p(*a):
    out_f.write(' '.join(str(x) for x in a) + '\n')
    out_f.flush()

DRIVE_ROOT = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
PHASE6_DIR = os.path.join(DRIVE_ROOT, 'swebench_v6_phase6')
CAPTURES_DIR = os.path.join(PHASE6_DIR, 'captures')
TRACES_DIR = os.path.join(PHASE6_DIR, 'traces')
RESULTS_JSON = os.path.join(PHASE6_DIR, 'phase6_results.json')
OUT_DIR = os.path.join(DRIVE_ROOT, 'kappa_t_c6')
os.makedirs(OUT_DIR, exist_ok=True)

PROBE_LAYER = 43
POS_INNER = 'think_end'
POS_OUTER = 'turn_end'
WINDOW = 3

t_start = time.time()

# Load (same as controls)
with open(RESULTS_JSON) as f:
    phase6_results = json.load(f)

def repo_from_iid(iid):
    return iid.replace('instance_', '').split('__')[0] if iid else 'unknown'

per_trace_turns = {}
trace_n_turns = {}
for iid in phase6_results:
    trace_path = os.path.join(TRACES_DIR, f'{iid}.json')
    if not os.path.exists(trace_path): continue
    with open(trace_path) as f:
        trace = json.load(f)
    turns = trace.get('turns', [])
    n_turns_trace = len(turns)
    trace_n_turns[iid] = n_turns_trace
    turn_info = {}
    for turn in turns:
        t = turn['turn_idx']
        tool_calls = turn.get('tool_calls', [])
        primary_tool = tool_calls[0]['name'] if tool_calls else 'none'
        thinking = turn.get('thinking') or ''
        tool_results = turn.get('tool_results', [])
        tool_ok = tool_results[0].get('result', {}).get('ok', False) if tool_results else False
        turn_info[t] = {
            'tool_name': primary_tool,
            'thinking_chars': len(thinking),
            'tool_ok': int(bool(tool_ok)),
            'n_new_tokens': turn.get('new_tokens', 0),
            'wall_seconds': turn.get('wall_seconds', 0.0),
        }
    tcs = [v['thinking_chars'] for v in turn_info.values()]
    med_tc = float(np.median(tcs)) if tcs else 0
    for v in turn_info.values():
        v['long_thinking'] = int(v['thinking_chars'] > med_tc)
    per_trace_turns[iid] = turn_info

per_iid_meta = {iid: {
    'failure': 0 if (info.get('finished') and info.get('finish_reason') == 'finish_tool') else 1,
    'repo': repo_from_iid(iid),
} for iid, info in phase6_results.items()}

p(f'[{time.time()-t_start:.0f}s] Loaded {len(per_trace_turns)} trace JSONs')

def load_residuals(iid):
    sp = os.path.join(CAPTURES_DIR, f'{iid}.safetensors')
    mp = os.path.join(CAPTURES_DIR, f'{iid}.meta.json')
    if not (os.path.exists(sp) and os.path.exists(mp)): return None
    with open(mp) as f: meta = json.load(f)
    by_key = {(r['turn_idx'], r['position_label'], r['layer']): r['activation_key'] for r in meta['records']}
    turns = sorted({r['turn_idx'] for r in meta['records']
                    if (r['turn_idx'], POS_INNER, PROBE_LAYER) in by_key
                    and (r['turn_idx'], POS_OUTER, PROBE_LAYER) in by_key})
    if not turns: return None
    out = []
    with safe_open(sp, framework='pt') as f:
        for t in turns:
            v_i = f.get_tensor(by_key[(t, POS_INNER, PROBE_LAYER)]).float().numpy()
            v_o = f.get_tensor(by_key[(t, POS_OUTER, PROBE_LAYER)]).float().numpy()
            out.append((t, np.concatenate([v_i, v_o])))
    return out

p(f'[{time.time()-t_start:.0f}s] Loading residuals (Drive should be warm)...')
X_list, iid_list, turn_list = [], [], []
for i, iid in enumerate(phase6_results):
    res = load_residuals(iid)
    if res is None: continue
    for (t, vec) in res:
        X_list.append(vec); iid_list.append(iid); turn_list.append(t)
    if (i + 1) % 25 == 0:
        p(f'  [{i+1}/{len(phase6_results)}] {(time.time()-t_start)/60:.1f}min')

X = np.stack(X_list).astype(np.float32)
iid_arr = np.array(iid_list)
turn_arr = np.array(turn_list)
groups = iid_arr.copy()
p(f'[{time.time()-t_start:.0f}s] Loaded {len(X)} samples from {len(set(iid_arr))} instances')

# 5 axes (v2 set)
def get_turn(iid, t, key, default=0):
    return per_trace_turns.get(iid, {}).get(t, {}).get(key, default)

y_axes = {
    'A_tool_finish':   np.array([1 if get_turn(iid_arr[i], int(turn_arr[i]), 'tool_name') == 'finish' else 0 for i in range(len(X))], dtype=int),
    'B_tool_bash':     np.array([1 if get_turn(iid_arr[i], int(turn_arr[i]), 'tool_name') == 'bash' else 0 for i in range(len(X))], dtype=int),
    'C_long_thinking': np.array([get_turn(iid_arr[i], int(turn_arr[i]), 'long_thinking', 0) for i in range(len(X))], dtype=int),
    'D_tool_ok':       np.array([get_turn(iid_arr[i], int(turn_arr[i]), 'tool_ok', 0) for i in range(len(X))], dtype=int),
    'E_repo_ansible':  np.array([1 if per_iid_meta[iid_arr[i]]['repo'] == 'ansible' else 0 for i in range(len(X))], dtype=int),
}

oof_scores = {name: np.zeros(len(X), dtype=np.float32) for name in y_axes}
p(f'\n[{time.time()-t_start:.0f}s] Training 5 probes...')
for name, y in y_axes.items():
    np.random.seed(42)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[tr]).astype(np.float32)
        Xte_s = sc.transform(X[te]).astype(np.float32)
        clf = LogisticRegression(class_weight='balanced', max_iter=500, C=0.1, solver='liblinear')
        clf.fit(Xtr_s, y[tr])
        oof_scores[name][te] = clf.decision_function(Xte_s)
    p(f'  {name}: done')

names = list(y_axes.keys())
N_PROBES = len(names)

def compute_kappa_window(window, n_probes):
    if window.shape[0] < 3: return np.nan
    cm = np.corrcoef(window.T)
    if np.isnan(cm).any(): return np.nan
    mask = ~np.eye(n_probes, dtype=bool)
    return float(np.mean(np.abs(cm[mask])))

# Build per-trace κ_t full series
per_trace_kappa = {}
for iid in set(iid_arr):
    mask = iid_arr == iid
    idxs = np.where(mask)[0]
    sort_perm = np.argsort(turn_arr[idxs])
    sorted_idxs = idxs[sort_perm]
    scores_iid = np.stack([oof_scores[n][sorted_idxs] for n in names], axis=1)
    n_t = scores_iid.shape[0]
    kappa = np.full(n_t, np.nan)
    for t_idx in range(n_t):
        lo = max(0, t_idx - WINDOW)
        hi = min(n_t, t_idx + WINDOW + 1)
        kappa[t_idx] = compute_kappa_window(scores_iid[lo:hi], N_PROBES)
    per_trace_kappa[iid] = kappa

def slope_over_window(kappa, lo, hi):
    """Compute slope over kappa[lo:hi], NaN-safe. Returns NaN if <5 valid points."""
    seg = kappa[lo:hi]
    v_mask = ~np.isnan(seg)
    if v_mask.sum() < 5: return np.nan
    turns = np.arange(len(seg))[v_mask]
    vals = seg[v_mask]
    return float(np.polyfit(turns, vals, 1)[0])

p('\n' + '='*78)
p('C6 — LENGTH-CONTROLLED SLOPE TESTS')
p('='*78)

results = {}

# Variant 1-4: slope over first N turns, all traces with n_turns >= N
for N in [10, 15, 20, 25]:
    succ_slopes = []
    fail_slopes = []
    succ_n = 0
    fail_n = 0
    for iid, kappa in per_trace_kappa.items():
        if len(kappa) < N: continue
        slope = slope_over_window(kappa, 0, N)
        if np.isnan(slope): continue
        if per_iid_meta[iid]['failure'] == 0:
            succ_slopes.append(slope); succ_n += 1
        else:
            fail_slopes.append(slope); fail_n += 1
    if len(succ_slopes) >= 5 and len(fail_slopes) >= 5:
        stat, pval = mannwhitneyu(succ_slopes, fail_slopes, alternative='two-sided')
    else:
        pval = float('nan')
    ratio = (np.mean(succ_slopes) / max(abs(np.mean(fail_slopes)), 1e-9)) if fail_slopes else float('nan')
    p(f'\nFirst {N} turns:')
    p(f'  Success N={succ_n}: slope = {np.mean(succ_slopes):+.5f} ± {np.std(succ_slopes):.5f}')
    p(f'  Failure N={fail_n}: slope = {np.mean(fail_slopes):+.5f} ± {np.std(fail_slopes):.5f}')
    p(f'  Mann-Whitney p = {pval:.4f}  |  effect ratio = {ratio:.1f}×  |  {"✓ SIGNIFICANT" if pval < 0.05 else "✗ ns"}')
    results[f'first_{N}_turns'] = {
        'succ_n': succ_n, 'fail_n': fail_n,
        'succ_slope': float(np.mean(succ_slopes)) if succ_slopes else None,
        'fail_slope': float(np.mean(fail_slopes)) if fail_slopes else None,
        'pval': float(pval) if not np.isnan(pval) else None,
        'effect_ratio': float(ratio) if not np.isnan(ratio) else None,
    }

# Variant 5: early-half vs late-half slope per trace (cleaner length normalization)
p('\n--- Early-half vs late-half slope (per trace, no fixed length) ---')
early_succ, early_fail, late_succ, late_fail = [], [], [], []
for iid, kappa in per_trace_kappa.items():
    n_t = len(kappa)
    if n_t < 12: continue
    half = n_t // 2
    early_s = slope_over_window(kappa, 0, half)
    late_s = slope_over_window(kappa, half, n_t)
    if np.isnan(early_s) or np.isnan(late_s): continue
    if per_iid_meta[iid]['failure'] == 0:
        early_succ.append(early_s); late_succ.append(late_s)
    else:
        early_fail.append(early_s); late_fail.append(late_s)
p(f'  EARLY-half slope succ N={len(early_succ)}: {np.mean(early_succ):+.5f}')
p(f'  EARLY-half slope fail N={len(early_fail)}: {np.mean(early_fail):+.5f}')
early_stat, early_pval = mannwhitneyu(early_succ, early_fail, alternative='two-sided') if early_succ and early_fail else (None, float('nan'))
p(f'  Early-half MW p={early_pval:.4f}  {"✓" if early_pval < 0.05 else "✗"}')
p(f'  LATE-half  slope succ N={len(late_succ)}: {np.mean(late_succ):+.5f}')
p(f'  LATE-half  slope fail N={len(late_fail)}: {np.mean(late_fail):+.5f}')
late_stat, late_pval = mannwhitneyu(late_succ, late_fail, alternative='two-sided') if late_succ and late_fail else (None, float('nan'))
p(f'  Late-half MW p={late_pval:.4f}  {"✓" if late_pval < 0.05 else "✗"}')
results['early_late_half'] = {
    'early_succ_slope': float(np.mean(early_succ)) if early_succ else None,
    'early_fail_slope': float(np.mean(early_fail)) if early_fail else None,
    'early_pval': float(early_pval) if not np.isnan(early_pval) else None,
    'late_succ_slope': float(np.mean(late_succ)) if late_succ else None,
    'late_fail_slope': float(np.mean(late_fail)) if late_fail else None,
    'late_pval': float(late_pval) if not np.isnan(late_pval) else None,
}

# Variant 6: within-length-bucket stratification
p('\n--- Length-bucket stratified slope (3 buckets: short/med/long) ---')
all_iids = list(per_trace_kappa.keys())
all_lens = np.array([trace_n_turns[iid] for iid in all_iids])
all_slopes = []
for iid in all_iids:
    kappa = per_trace_kappa[iid]
    v_mask = ~np.isnan(kappa)
    if v_mask.sum() < 5:
        all_slopes.append(np.nan); continue
    turns = np.arange(len(kappa))[v_mask]
    vals = kappa[v_mask]
    all_slopes.append(np.polyfit(turns, vals, 1)[0])
all_slopes = np.array(all_slopes)
labels = np.array([per_iid_meta[iid]['failure'] for iid in all_iids])

q1, q2 = np.percentile(all_lens, [33, 67])
results['length_buckets'] = {}
for bucket_name, mask in [
    ('short (n_turns <= {})'.format(int(q1)), all_lens <= q1),
    ('medium ({}-{})'.format(int(q1), int(q2)), (all_lens > q1) & (all_lens <= q2)),
    ('long (n_turns > {})'.format(int(q2)), all_lens > q2),
]:
    s_succ = all_slopes[mask & (labels == 0)]
    s_fail = all_slopes[mask & (labels == 1)]
    s_succ = s_succ[~np.isnan(s_succ)]
    s_fail = s_fail[~np.isnan(s_fail)]
    if len(s_succ) >= 3 and len(s_fail) >= 3:
        stat, pval = mannwhitneyu(s_succ, s_fail, alternative='two-sided')
    else:
        pval = float('nan')
    p(f'  {bucket_name}: succ N={len(s_succ)} slope={np.mean(s_succ) if len(s_succ) else float("nan"):+.5f}, fail N={len(s_fail)} slope={np.mean(s_fail) if len(s_fail) else float("nan"):+.5f}, MW p={pval:.4f}')
    results['length_buckets'][bucket_name] = {
        'succ_n': len(s_succ), 'fail_n': len(s_fail),
        'succ_slope': float(np.mean(s_succ)) if len(s_succ) else None,
        'fail_slope': float(np.mean(s_fail)) if len(s_fail) else None,
        'pval': float(pval) if not np.isnan(pval) else None,
    }

# Decision
p('\n' + '='*78)
p('VERDICT')
p('='*78)
n_significant = sum(1 for k in ['first_10_turns', 'first_15_turns', 'first_20_turns'] if results[k]['pval'] is not None and results[k]['pval'] < 0.05)
p(f'First-N-turns variants significant: {n_significant}/3')
p(f'Early-half p={results["early_late_half"]["early_pval"]}  Late-half p={results["early_late_half"]["late_pval"]}')
p(f'Length buckets significant: {sum(1 for b in results["length_buckets"].values() if b["pval"] is not None and b["pval"] < 0.05)}/3')

p(f'\nIf first-N significant in ≥2/3: HEADLINE RESCUED (buildup is real, not length-confounded).')
p(f'If early-half significant but late-half not: buildup is genuine early-trace phenomenon.')
p(f'If within-length-bucket significant in ≥2/3: buildup holds within fixed-length strata.')

with open(os.path.join(OUT_DIR, 'kappa_t_c6_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
p(f'\nSaved: {os.path.join(OUT_DIR, "kappa_t_c6_results.json")}')
p(f'Total: {(time.time()-t_start)/60:.1f}min')
out_f.close()
