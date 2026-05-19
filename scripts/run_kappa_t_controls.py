"""κ_t v4 — controls to defend paper against 5 reviewer critiques.

C1: trace-length confound (does longer trace = bigger slope mechanically?)
C2: within-trace shuffle (does the temporal order matter, or just class mean?)
C3: orthogonal-pair κ_t (is signal artifact of shared encoder?)
C4: class trace-length distribution (do success/fail have different lengths?)
C5: max-turns-only stratification (does signal survive among uniformly-length traces?)

Run: python3 /tmp/run_kappa_t_controls.py
Output: /tmp/kappa_t_controls_out.txt + JSON to Drive
"""
import os, json, time, warnings
import numpy as np
warnings.filterwarnings('ignore')

from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

OUT_PATH = '/tmp/kappa_t_controls_out.txt'
out_f = open(OUT_PATH, 'w', buffering=1)
def p(*a):
    out_f.write(' '.join(str(x) for x in a) + '\n')
    out_f.flush()

DRIVE_ROOT = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
PHASE6_DIR = os.path.join(DRIVE_ROOT, 'swebench_v6_phase6')
CAPTURES_DIR = os.path.join(PHASE6_DIR, 'captures')
TRACES_DIR = os.path.join(PHASE6_DIR, 'traces')
RESULTS_JSON = os.path.join(PHASE6_DIR, 'phase6_results.json')
OUT_DIR = os.path.join(DRIVE_ROOT, 'kappa_t_controls')
os.makedirs(OUT_DIR, exist_ok=True)

PROBE_LAYER = 43
POS_INNER = 'think_end'
POS_OUTER = 'turn_end'
WINDOW = 3

t_start = time.time()

# === Load trace metadata ===
with open(RESULTS_JSON) as f:
    phase6_results = json.load(f)

def repo_from_iid(iid):
    return iid.replace('instance_', '').split('__')[0] if iid else 'unknown'

per_trace_turns = {}
trace_n_turns = {}
for iid in phase6_results:
    trace_path = os.path.join(TRACES_DIR, f'{iid}.json')
    if not os.path.exists(trace_path):
        continue
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
            'turn_idx': t, 'n_turns_trace': n_turns_trace,
        }
    tcs = [v['thinking_chars'] for v in turn_info.values()]
    nts = [v['n_new_tokens'] for v in turn_info.values()]
    wss = [v['wall_seconds'] for v in turn_info.values()]
    med_tc = float(np.median(tcs)) if tcs else 0
    med_nt = float(np.median(nts)) if nts else 0
    med_ws = float(np.median(wss)) if wss else 0
    for v in turn_info.values():
        v['long_thinking'] = int(v['thinking_chars'] > med_tc)
        v['response_long'] = int(v['n_new_tokens'] > med_nt)
        v['slow_turn'] = int(v['wall_seconds'] > med_ws)
        v['early_trace'] = int(v['turn_idx'] <= 5)
        v['late_trace'] = int(v['turn_idx'] >= max(0, n_turns_trace - 5))
    per_trace_turns[iid] = turn_info

per_iid_meta = {iid: {
    'failure': 0 if (info.get('finished') and info.get('finish_reason') == 'finish_tool') else 1,
    'repo': repo_from_iid(iid),
    'finish_reason': info.get('finish_reason', 'unknown'),
} for iid, info in phase6_results.items()}

p(f'[{time.time()-t_start:.0f}s] Loaded {len(per_trace_turns)} trace JSONs')

# === Load residuals ===
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
    if not turns: return None
    out = []
    with safe_open(sp, framework='pt') as f:
        for t in turns:
            v_i = f.get_tensor(by_key[(t, POS_INNER, PROBE_LAYER)]).float().numpy()
            v_o = f.get_tensor(by_key[(t, POS_OUTER, PROBE_LAYER)]).float().numpy()
            out.append((t, np.concatenate([v_i, v_o])))
    return out

p(f'[{time.time()-t_start:.0f}s] Loading L{PROBE_LAYER} paired residuals...')
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

# === Build 9 axes (v3 set) ===
def get_turn(iid, t, key, default=0):
    return per_trace_turns.get(iid, {}).get(t, {}).get(key, default)

y_axes_all = {
    'A_tool_finish':   np.array([1 if get_turn(iid_arr[i], int(turn_arr[i]), 'tool_name') == 'finish' else 0 for i in range(len(X))], dtype=int),
    'B_tool_bash':     np.array([1 if get_turn(iid_arr[i], int(turn_arr[i]), 'tool_name') == 'bash' else 0 for i in range(len(X))], dtype=int),
    'C_long_thinking': np.array([get_turn(iid_arr[i], int(turn_arr[i]), 'long_thinking', 0) for i in range(len(X))], dtype=int),
    'D_tool_ok':       np.array([get_turn(iid_arr[i], int(turn_arr[i]), 'tool_ok', 0) for i in range(len(X))], dtype=int),
    'E_repo_ansible':  np.array([1 if per_iid_meta[iid_arr[i]]['repo'] == 'ansible' else 0 for i in range(len(X))], dtype=int),
    'F_response_long': np.array([get_turn(iid_arr[i], int(turn_arr[i]), 'response_long', 0) for i in range(len(X))], dtype=int),
    'G_early_trace':   np.array([get_turn(iid_arr[i], int(turn_arr[i]), 'early_trace', 0) for i in range(len(X))], dtype=int),
    'H_late_trace':    np.array([get_turn(iid_arr[i], int(turn_arr[i]), 'late_trace', 0) for i in range(len(X))], dtype=int),
    'I_slow_turn':     np.array([get_turn(iid_arr[i], int(turn_arr[i]), 'slow_turn', 0) for i in range(len(X))], dtype=int),
}
y_axes = {k: v for k, v in y_axes_all.items() if v.sum() >= 30 and (len(v) - v.sum()) >= 30}
p(f'  {len(y_axes)} viable probes')

# === Train probes ===
oof_scores = {name: np.zeros(len(X), dtype=np.float32) for name in y_axes}
probe_aurocs = {}
p(f'\n[{time.time()-t_start:.0f}s] Training {len(y_axes)} probes...')
for name, y in y_axes.items():
    t_p = time.time()
    np.random.seed(42)
    gkf = GroupKFold(n_splits=5)
    aurocs = []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[tr]).astype(np.float32)
        Xte_s = sc.transform(X[te]).astype(np.float32)
        clf = LogisticRegression(class_weight='balanced', max_iter=500, C=0.1, solver='liblinear')
        clf.fit(Xtr_s, y[tr])
        s_te = clf.decision_function(Xte_s)
        oof_scores[name][te] = s_te
        try: aurocs.append(roc_auc_score(y[te], s_te))
        except ValueError: pass
    probe_aurocs[name] = float(np.mean(aurocs)) if aurocs else float('nan')
    p(f'  {name}: AUROC {probe_aurocs[name]:.3f}  ({time.time()-t_p:.0f}s)')

# === κ_t function ===
def compute_kappa_window(window, n_probes):
    if window.shape[0] < 3: return np.nan
    cm = np.corrcoef(window.T)
    if np.isnan(cm).any(): return np.nan
    mask = ~np.eye(n_probes, dtype=bool)
    return float(np.mean(np.abs(cm[mask])))

def per_trace_kappa_slopes(score_matrix_dict, names_subset, iid_arr, turn_arr, per_iid_meta, window=WINDOW):
    """Compute per-trace κ_t time series and slopes for a given probe subset."""
    n_probes = len(names_subset)
    per_trace_kappa = {}
    slopes = {}
    mean_kappa = {}
    for iid in set(iid_arr):
        mask = iid_arr == iid
        idxs = np.where(mask)[0]
        sort_perm = np.argsort(turn_arr[idxs])
        sorted_idxs = idxs[sort_perm]
        scores_iid = np.stack([score_matrix_dict[n][sorted_idxs] for n in names_subset], axis=1)
        n_t = scores_iid.shape[0]
        kappa = np.full(n_t, np.nan)
        for t_idx in range(n_t):
            lo = max(0, t_idx - window)
            hi = min(n_t, t_idx + window + 1)
            kappa[t_idx] = compute_kappa_window(scores_iid[lo:hi], n_probes)
        per_trace_kappa[iid] = kappa
        v_mask = ~np.isnan(kappa)
        if v_mask.sum() >= 5:
            turns = np.arange(len(kappa))[v_mask]
            vals = kappa[v_mask]
            slopes[iid] = float(np.polyfit(turns, vals, 1)[0])
        mean_kappa[iid] = float(np.mean(kappa[v_mask])) if v_mask.sum() > 0 else np.nan
    return per_trace_kappa, slopes, mean_kappa

names = list(y_axes.keys())
N_PROBES = len(names)

# ===================== ORIGINAL (reproduce v3) =====================
p(f'\n[{time.time()-t_start:.0f}s] === ORIGINAL (v3 reproduction, all {N_PROBES} probes) ===')
per_trace_kappa, slopes_per_trace, mean_kappa_per_trace = per_trace_kappa_slopes(
    oof_scores, names, iid_arr, turn_arr, per_iid_meta)

slopes_succ = [slopes_per_trace[iid] for iid in slopes_per_trace if per_iid_meta[iid]['failure']==0]
slopes_fail = [slopes_per_trace[iid] for iid in slopes_per_trace if per_iid_meta[iid]['failure']==1]
mean_succ = [mean_kappa_per_trace[iid] for iid in mean_kappa_per_trace if per_iid_meta[iid]['failure']==0 and not np.isnan(mean_kappa_per_trace[iid])]
mean_fail = [mean_kappa_per_trace[iid] for iid in mean_kappa_per_trace if per_iid_meta[iid]['failure']==1 and not np.isnan(mean_kappa_per_trace[iid])]

p(f'  Mean κ_t: succ={np.mean(mean_succ):.3f} N={len(mean_succ)}, fail={np.mean(mean_fail):.3f} N={len(mean_fail)}')
mw_stat, mw_pval = mannwhitneyu(mean_succ, mean_fail, alternative='two-sided')
p(f'  Mann-Whitney mean: p={mw_pval:.4f}')
p(f'  Slope succ: {np.mean(slopes_succ):+.5f} ± {np.std(slopes_succ):.5f}, N={len(slopes_succ)}')
p(f'  Slope fail: {np.mean(slopes_fail):+.5f} ± {np.std(slopes_fail):.5f}, N={len(slopes_fail)}')
s_stat, slope_pval = mannwhitneyu(slopes_succ, slopes_fail, alternative='two-sided')
p(f'  Slope Mann-Whitney p={slope_pval:.4f}')

# ===================== C4: CLASS TRACE-LENGTH DISTRIBUTION =====================
p(f'\n[{time.time()-t_start:.0f}s] === C4: Trace-length distribution by class ===')
len_succ = [trace_n_turns[iid] for iid in slopes_per_trace if per_iid_meta[iid]['failure']==0 and iid in trace_n_turns]
len_fail = [trace_n_turns[iid] for iid in slopes_per_trace if per_iid_meta[iid]['failure']==1 and iid in trace_n_turns]
p(f'  Success n_turns: median={np.median(len_succ):.0f}, mean={np.mean(len_succ):.1f}, min={min(len_succ)}, max={max(len_succ)}, N={len(len_succ)}')
p(f'  Failed  n_turns: median={np.median(len_fail):.0f}, mean={np.mean(len_fail):.1f}, min={min(len_fail)}, max={max(len_fail)}, N={len(len_fail)}')
len_stat, len_pval = mannwhitneyu(len_succ, len_fail, alternative='two-sided')
p(f'  Length MW p={len_pval:.6f}  →  {"DIFFER" if len_pval < 0.05 else "SAME"}')
# Distribution of finish_reason among failures
reasons = {}
for iid in slopes_per_trace:
    if per_iid_meta[iid]['failure'] == 1:
        r = per_iid_meta[iid]['finish_reason']
        reasons[r] = reasons.get(r, 0) + 1
p(f'  Failure finish_reason breakdown: {reasons}')

# ===================== C1: TRACE-LENGTH CONFOUND =====================
p(f'\n[{time.time()-t_start:.0f}s] === C1: Trace-length confound ===')
trace_iids = list(slopes_per_trace.keys())
trace_lens = np.array([trace_n_turns[iid] for iid in trace_iids])
trace_slopes = np.array([slopes_per_trace[iid] for iid in trace_iids])
trace_labels = np.array([per_iid_meta[iid]['failure'] for iid in trace_iids])

pear_r, pear_p = pearsonr(trace_lens, trace_slopes)
spear_r, spear_p = spearmanr(trace_lens, trace_slopes)
p(f'  Pearson(n_turns, slope)  = {pear_r:+.3f} (p={pear_p:.4f})')
p(f'  Spearman(n_turns, slope) = {spear_r:+.3f} (p={spear_p:.4f})')

# Partial: does class predict slope CONTROLLING for length?
# Method: regress slope on length, take residuals, test class effect on residuals
from numpy.polynomial import polynomial as P
A_fit = np.polyfit(trace_lens, trace_slopes, 1)
predicted = np.polyval(A_fit, trace_lens)
slope_residuals = trace_slopes - predicted
resid_succ = slope_residuals[trace_labels == 0]
resid_fail = slope_residuals[trace_labels == 1]
ps_stat, partial_pval = mannwhitneyu(resid_succ, resid_fail, alternative='two-sided')
p(f'  After regressing OUT length: slope-residual succ={np.mean(resid_succ):+.5f}, fail={np.mean(resid_fail):+.5f}')
p(f'  Partial MW p={partial_pval:.4f}  →  {"SURVIVES length control" if partial_pval < 0.05 else "DIES under length control"}')

# Same-length stratum check: take only traces with n_turns >= 20 (mostly fails) or split by tertiles
# Stratified subsample: traces with n_turns in [10, 25] — has both classes
stratum_mask = (trace_lens >= 10) & (trace_lens <= 25)
n_in_stratum = stratum_mask.sum()
if n_in_stratum >= 10:
    s_succ = trace_slopes[stratum_mask & (trace_labels == 0)]
    s_fail = trace_slopes[stratum_mask & (trace_labels == 1)]
    p(f'  Stratum n_turns∈[10,25]: succ N={len(s_succ)} slope={np.mean(s_succ) if len(s_succ) else float("nan"):+.5f}, fail N={len(s_fail)} slope={np.mean(s_fail) if len(s_fail) else float("nan"):+.5f}')
    if len(s_succ) >= 3 and len(s_fail) >= 3:
        sm_stat, sm_p = mannwhitneyu(s_succ, s_fail, alternative='two-sided')
        p(f'  Stratum MW p={sm_p:.4f}')

# ===================== C2: WITHIN-TRACE SHUFFLE =====================
p(f'\n[{time.time()-t_start:.0f}s] === C2: Within-trace turn-order shuffle ===')
N_PERMS = 200
rng = np.random.default_rng(42)
shuffled_class_diffs = []
for perm in range(N_PERMS):
    shuf_slopes = {}
    for iid in set(iid_arr):
        mask = iid_arr == iid
        idxs = np.where(mask)[0]
        sort_perm = np.argsort(turn_arr[idxs])
        sorted_idxs = idxs[sort_perm]
        n_t = len(sorted_idxs)
        # PERMUTE the order of turns within trace
        perm_order = rng.permutation(n_t)
        permuted_idxs = sorted_idxs[perm_order]
        scores_iid = np.stack([oof_scores[n][permuted_idxs] for n in names], axis=1)
        kappa = np.full(n_t, np.nan)
        for t_idx in range(n_t):
            lo = max(0, t_idx - WINDOW)
            hi = min(n_t, t_idx + WINDOW + 1)
            kappa[t_idx] = compute_kappa_window(scores_iid[lo:hi], N_PROBES)
        v_mask = ~np.isnan(kappa)
        if v_mask.sum() >= 5:
            turns = np.arange(len(kappa))[v_mask]
            vals = kappa[v_mask]
            shuf_slopes[iid] = float(np.polyfit(turns, vals, 1)[0])
    s_succ = [shuf_slopes[iid] for iid in shuf_slopes if per_iid_meta[iid]['failure']==0]
    s_fail = [shuf_slopes[iid] for iid in shuf_slopes if per_iid_meta[iid]['failure']==1]
    shuffled_class_diffs.append(np.mean(s_succ) - np.mean(s_fail))
    if (perm + 1) % 50 == 0:
        p(f'  perm {perm+1}/{N_PERMS}: class diff = {shuffled_class_diffs[-1]:+.5f}')

real_diff = np.mean(slopes_succ) - np.mean(slopes_fail)
null_mean = np.mean(shuffled_class_diffs)
null_std = np.std(shuffled_class_diffs)
# One-sided p-value: how often does shuffled diff exceed real?
n_extreme = sum(1 for d in shuffled_class_diffs if d >= real_diff)
shuf_pval = n_extreme / N_PERMS
p(f'  Real class-diff (succ-fail) slope: {real_diff:+.5f}')
p(f'  Shuffled null: mean={null_mean:+.5f}, std={null_std:.5f}')
p(f'  Shuffled-permutation p={shuf_pval:.4f}  →  {"SURVIVES shuffle" if shuf_pval < 0.05 else "DIES under shuffle"}')

# ===================== C3: ORTHOGONAL-PAIR-ONLY κ_t =====================
p(f'\n[{time.time()-t_start:.0f}s] === C3: Orthogonal-pair κ_t ===')
score_matrix = np.stack([oof_scores[n] for n in names], axis=1)
corr_overall = np.corrcoef(score_matrix.T)
abs_corr = np.abs(corr_overall.copy())
np.fill_diagonal(abs_corr, 1)
# Greedy: pick probes with max independence
# Start with probe 0, then add probe with min max-corr to picked set
selected_idx = [0]
remaining = set(range(N_PROBES)) - set(selected_idx)
while remaining:
    candidates = sorted(remaining, key=lambda j: max(abs_corr[j, i] for i in selected_idx))
    next_p = candidates[0]
    max_corr_to_set = max(abs_corr[next_p, i] for i in selected_idx)
    if max_corr_to_set > 0.20: break  # stop when all remaining are correlated
    selected_idx.append(next_p)
    remaining.remove(next_p)
orth_names = [names[i] for i in selected_idx]
p(f'  Orthogonal subset (|corr|<0.20 to all): {orth_names}')
if len(orth_names) >= 3:
    _, slopes_orth, mean_orth = per_trace_kappa_slopes(
        oof_scores, orth_names, iid_arr, turn_arr, per_iid_meta)
    s_succ_o = [slopes_orth[iid] for iid in slopes_orth if per_iid_meta[iid]['failure']==0]
    s_fail_o = [slopes_orth[iid] for iid in slopes_orth if per_iid_meta[iid]['failure']==1]
    p(f'  Orth-only slope succ: {np.mean(s_succ_o):+.5f} N={len(s_succ_o)}')
    p(f'  Orth-only slope fail: {np.mean(s_fail_o):+.5f} N={len(s_fail_o)}')
    o_stat, o_pval = mannwhitneyu(s_succ_o, s_fail_o, alternative='two-sided')
    p(f'  Orth-only Mann-Whitney p={o_pval:.4f}  →  {"SURVIVES orthogonal restriction" if o_pval < 0.05 else "DIES under orth restriction"}')
else:
    p(f'  Insufficient orthogonal probes for κ_t (need ≥3). Skipping C3.')
    o_pval = float('nan')

# ===================== C5: MAX-TURNS-ONLY STRATIFICATION =====================
p(f'\n[{time.time()-t_start:.0f}s] === C5: Max-turns-only (length-controlled) ===')
# Compare slopes among traces that ALL hit max_turns (= same length 30)
max_turns_iids = [iid for iid in slopes_per_trace if trace_n_turns.get(iid, 0) == 30]
mt_succ = [slopes_per_trace[iid] for iid in max_turns_iids if per_iid_meta[iid]['failure']==0]
mt_fail = [slopes_per_trace[iid] for iid in max_turns_iids if per_iid_meta[iid]['failure']==1]
p(f'  N max_turns: succ={len(mt_succ)}, fail={len(mt_fail)}')
if len(mt_succ) >= 3 and len(mt_fail) >= 3:
    p(f'  Max-turns slope succ: {np.mean(mt_succ):+.5f} ± {np.std(mt_succ):.5f}')
    p(f'  Max-turns slope fail: {np.mean(mt_fail):+.5f} ± {np.std(mt_fail):.5f}')
    mt_stat, mt_pval = mannwhitneyu(mt_succ, mt_fail, alternative='two-sided')
    p(f'  Max-turns Mann-Whitney p={mt_pval:.4f}  →  {"SURVIVES iso-length" if mt_pval < 0.05 else "DIES under iso-length"}')
else:
    p(f'  Insufficient max-turns examples in both classes (rare success at max_turns).')
    mt_pval = float('nan')

# ===================== SUMMARY =====================
p(f'\n[{time.time()-t_start:.0f}s] ' + '='*78)
p('CONTROLS SUMMARY')
p('='*78)
p(f'Original slope p (v3 baseline):              p={slope_pval:.4f}  {"PASS" if slope_pval<0.05 else "FAIL"}')
p(f'C1 Partial slope after length-regression:    p={partial_pval:.4f}  {"PASS" if partial_pval<0.05 else "FAIL"}')
p(f'C2 Within-trace shuffle permutation:         p={shuf_pval:.4f}  {"PASS" if shuf_pval<0.05 else "FAIL"}')
p(f'C3 Orthogonal-probe-pair only:               p={o_pval:.4f}  {"PASS" if not np.isnan(o_pval) and o_pval<0.05 else "FAIL/NA"}')
p(f'C4 Length distribution differs by class:     p={len_pval:.6f}  (info — succ tends shorter due to early-finish)')
p(f'C5 Iso-length (max_turns only) slope diff:   p={mt_pval}  {"PASS" if not np.isnan(mt_pval) and mt_pval<0.05 else "FAIL/NA"}')

n_critical_pass = sum([slope_pval<0.05, partial_pval<0.05, shuf_pval<0.05])
p(f'\nCritical defenses (orig + C1 + C2): {n_critical_pass}/3 PASS')
p(f'C3 + C5 are nice-to-have (stronger paper if pass, but not blocking).')

# === Save ===
results = {
    'orig': {
        'mean_succ': float(np.mean(mean_succ)), 'mean_fail': float(np.mean(mean_fail)),
        'mean_mw_pval': float(mw_pval),
        'slope_succ': float(np.mean(slopes_succ)), 'slope_fail': float(np.mean(slopes_fail)),
        'slope_pval': float(slope_pval),
        'n_succ': len(slopes_succ), 'n_fail': len(slopes_fail),
    },
    'C1_length_confound': {
        'pearson_r': float(pear_r), 'pearson_p': float(pear_p),
        'spearman_r': float(spear_r), 'spearman_p': float(spear_p),
        'partial_pval': float(partial_pval),
        'resid_slope_succ': float(np.mean(resid_succ)), 'resid_slope_fail': float(np.mean(resid_fail)),
    },
    'C2_within_trace_shuffle': {
        'real_class_diff': float(real_diff),
        'null_mean': float(null_mean), 'null_std': float(null_std),
        'shuf_pval': float(shuf_pval), 'n_perms': N_PERMS,
    },
    'C3_orthogonal_pair': {
        'orth_names': orth_names,
        'n_orth_probes': len(orth_names),
        'slope_succ': float(np.mean(s_succ_o)) if len(orth_names) >= 3 else None,
        'slope_fail': float(np.mean(s_fail_o)) if len(orth_names) >= 3 else None,
        'pval': float(o_pval) if not np.isnan(o_pval) else None,
    },
    'C4_length_distribution': {
        'len_succ_median': float(np.median(len_succ)), 'len_fail_median': float(np.median(len_fail)),
        'len_succ_mean': float(np.mean(len_succ)), 'len_fail_mean': float(np.mean(len_fail)),
        'len_mw_pval': float(len_pval),
        'finish_reasons_failure': reasons,
    },
    'C5_iso_length_max_turns': {
        'n_succ': len(mt_succ), 'n_fail': len(mt_fail),
        'slope_succ': float(np.mean(mt_succ)) if len(mt_succ) >= 3 else None,
        'slope_fail': float(np.mean(mt_fail)) if len(mt_fail) >= 3 else None,
        'pval': float(mt_pval) if not np.isnan(mt_pval) else None,
    },
    'n_critical_pass': int(n_critical_pass),
}
out_json = os.path.join(OUT_DIR, 'kappa_t_controls_results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
p(f'\nSaved: {out_json}')
p(f'Total: {(time.time()-t_start)/60:.1f}min')
out_f.close()
