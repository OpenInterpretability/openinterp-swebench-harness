"""κ_t v3 — 9 axes (5 original + 4 new per-turn) to test robustness of slope finding."""
import os, json, sys, warnings, time
import numpy as np
warnings.filterwarnings('ignore')

from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import mannwhitneyu

OUT_PATH = '/tmp/kappa_t_v3_out.txt'
out_f = open(OUT_PATH, 'w', buffering=1)
def p(*a):
    out_f.write(' '.join(str(x) for x in a) + '\n')
    out_f.flush()

DRIVE_ROOT = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
PHASE6_DIR = os.path.join(DRIVE_ROOT, 'swebench_v6_phase6')
CAPTURES_DIR = os.path.join(PHASE6_DIR, 'captures')
TRACES_DIR = os.path.join(PHASE6_DIR, 'traces')
RESULTS_JSON = os.path.join(PHASE6_DIR, 'phase6_results.json')
OUT_DIR = os.path.join(DRIVE_ROOT, 'kappa_t_v3')
os.makedirs(OUT_DIR, exist_ok=True)

PROBE_LAYER = 43
POS_INNER = 'think_end'
POS_OUTER = 'turn_end'

t_start = time.time()

# === Load Phase 6 metadata + per-turn data ===
with open(RESULTS_JSON) as f:
    phase6_results = json.load(f)
p(f'[{time.time()-t_start:.0f}s] Loading per-turn data from trace JSONs...')

def repo_from_iid(iid):
    return iid.replace('instance_', '').split('__')[0] if iid else 'unknown'

per_trace_turns = {}
for iid in phase6_results:
    trace_path = os.path.join(TRACES_DIR, f'{iid}.json')
    if not os.path.exists(trace_path):
        continue
    with open(trace_path) as f:
        trace = json.load(f)
    turns = trace.get('turns', [])
    n_turns_trace = len(turns)
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
            'turn_idx': t,
            'n_turns_trace': n_turns_trace,
        }
    # Compute medians for binarization
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
p(f'  Loaded turn-level data for {len(per_trace_turns)} traces')

per_iid_meta = {iid: {
    'failure': 0 if (info.get('finished') and info.get('finish_reason') == 'finish_tool') else 1,
    'repo': repo_from_iid(iid),
} for iid, info in phase6_results.items()}

# === Load L43 paired residuals ===
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

p(f'\n[{time.time()-t_start:.0f}s] Loading L{PROBE_LAYER} paired residuals (5min on Drive)...')
X_list, iid_list, turn_list = [], [], []
n_skipped = 0
for i, iid in enumerate(phase6_results):
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
p(f'[{time.time()-t_start:.0f}s] Loaded {len(X)} samples from {len(set(iid_arr))} instances')

# === Build 9 per-turn label arrays ===
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
for name, y in y_axes_all.items():
    p(f'  {name}: positive rate {100*y.mean():.1f}% ({y.sum()}/{len(y)})')

# Skip degenerate
y_axes = {k: v for k, v in y_axes_all.items() if v.sum() >= 30 and (len(v) - v.sum()) >= 30}
p(f'  → {len(y_axes)} viable probes after rate filter')

# === Train probes with GroupKFold ===
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

# === Probe-score independence check ===
names = list(y_axes.keys())
N_PROBES = len(names)
score_matrix = np.stack([oof_scores[n] for n in names], axis=1)
corr_overall = np.corrcoef(score_matrix.T)
p(f'\n=== Probe pairwise correlations (top off-diagonal abs) ===')
abs_corr = np.abs(corr_overall)
np.fill_diagonal(abs_corr, 0)
flat = [(i, j, abs_corr[i,j]) for i in range(N_PROBES) for j in range(i+1, N_PROBES)]
flat.sort(key=lambda x: -x[2])
for (i, j, c) in flat[:10]:
    p(f'  {names[i]:18s} ↔ {names[j]:18s}: {corr_overall[i,j]:+.3f}')

# === κ_t computation ===
p(f'\n[{time.time()-t_start:.0f}s] Computing κ_t per (trace, turn) with N={N_PROBES} probes...')
WINDOW = 3
per_trace_scores = {}
for iid in set(iid_arr):
    mask = iid_arr == iid
    idxs = np.where(mask)[0]
    sort_perm = np.argsort(turn_arr[idxs])
    scores_iid = np.stack([oof_scores[n][idxs[sort_perm]] for n in names], axis=1)
    per_trace_scores[iid] = {
        'turns': turn_arr[idxs][sort_perm],
        'scores': scores_iid,
        'failure': per_iid_meta[iid]['failure'],
    }

def compute_kappa(window):
    if window.shape[0] < 3: return np.nan
    cm = np.corrcoef(window.T)
    if np.isnan(cm).any(): return np.nan
    mask = ~np.eye(N_PROBES, dtype=bool)
    return float(np.mean(np.abs(cm[mask])))

per_trace_kappa = {}
for iid, d in per_trace_scores.items():
    n_t = len(d['turns'])
    kappa = np.full(n_t, np.nan)
    for t_idx in range(n_t):
        lo = max(0, t_idx - WINDOW)
        hi = min(n_t, t_idx + WINDOW + 1)
        kappa[t_idx] = compute_kappa(d['scores'][lo:hi])
    per_trace_kappa[iid] = kappa

# === Tests ===
mean_succ, mean_fail = [], []
for iid, kappa in per_trace_kappa.items():
    v = kappa[~np.isnan(kappa)]
    if len(v) == 0: continue
    (mean_succ if per_iid_meta[iid]['failure']==0 else mean_fail).append(np.mean(v))

p(f'\n=== Test 1: per-trace mean κ_t ===')
p(f'  Success N={len(mean_succ)}: {np.mean(mean_succ):.3f} ± {np.std(mean_succ):.3f}')
p(f'  Failed  N={len(mean_fail)}: {np.mean(mean_fail):.3f} ± {np.std(mean_fail):.3f}')
pval = float('nan')
if mean_succ and mean_fail:
    stat, pval = mannwhitneyu(mean_succ, mean_fail, alternative='two-sided')
    p(f'  Mann-Whitney p={pval:.4f}')

# Test 2: AUROC
trace_labels, trace_kappas = [], []
for iid, kappa in per_trace_kappa.items():
    v = kappa[~np.isnan(kappa)]
    if len(v) == 0: continue
    trace_labels.append(per_iid_meta[iid]['failure'])
    trace_kappas.append(np.mean(v))
trace_labels = np.array(trace_labels)
trace_kappas = np.array(trace_kappas)
auroc_kappa = float('nan')
try: auroc_kappa = roc_auc_score(trace_labels, -trace_kappas)
except ValueError: pass
p(f'\n=== Test 2: AUROC (-mean κ_t) → trace failure ===')
p(f'  AUROC = {auroc_kappa:.3f} (N={len(trace_labels)})')

rng = np.random.default_rng(42)
shuf_aurocs = []
for _ in range(200):
    sh = rng.permutation(trace_kappas)
    try: shuf_aurocs.append(roc_auc_score(trace_labels, -sh))
    except ValueError: pass
shuf_mean = float(np.mean(shuf_aurocs))
p(f'  Shuffled baseline: mean={shuf_mean:.3f}')
p(f'  Gap = {auroc_kappa - shuf_mean:+.3f}')

# Test 3: EARLY κ_t
early_labels, early_kappas = [], []
for iid, kappa in per_trace_kappa.items():
    early = kappa[:5]
    v = early[~np.isnan(early)]
    if len(v) < 2: continue
    early_labels.append(per_iid_meta[iid]['failure'])
    early_kappas.append(np.mean(v))
auroc_early = float('nan')
if early_kappas:
    try: auroc_early = roc_auc_score(early_labels, -np.array(early_kappas))
    except ValueError: pass
p(f'\n=== Test 3: EARLY κ_t (first 5) → outcome ===')
p(f'  AUROC = {auroc_early:.3f} (N={len(early_labels)})')

# Test 4: SLOPE
slopes_succ, slopes_fail = [], []
for iid, kappa in per_trace_kappa.items():
    v_mask = ~np.isnan(kappa)
    if v_mask.sum() < 5: continue
    turns = np.arange(len(kappa))[v_mask]
    vals = kappa[v_mask]
    slope = np.polyfit(turns, vals, 1)[0]
    (slopes_succ if per_iid_meta[iid]['failure']==0 else slopes_fail).append(slope)
p(f'\n=== Test 4: κ_t SLOPE (HEADLINE) ===')
p(f'  Success N={len(slopes_succ)}: slope = {np.mean(slopes_succ):+.5f} ± {np.std(slopes_succ):.5f}')
p(f'  Failed  N={len(slopes_fail)}: slope = {np.mean(slopes_fail):+.5f} ± {np.std(slopes_fail):.5f}')
slope_pval = float('nan')
if slopes_succ and slopes_fail:
    s_stat, slope_pval = mannwhitneyu(slopes_succ, slopes_fail, alternative='two-sided')
    p(f'  Mann-Whitney slope: p={slope_pval:.4f}')

# === Gate check ===
p('\n' + '='*78)
p(f'GATE CHECK (v3, N_probes={N_PROBES}, vs v2 N_probes=5)')
p('='*78)
G1 = auroc_kappa > 0.65
G2 = (auroc_kappa - shuf_mean) > 0.10
G3 = (pval < 0.05)
G4 = (auroc_early > 0.60) if not np.isnan(auroc_early) else False
G5 = (slope_pval < 0.05) if not np.isnan(slope_pval) else False  # bonus

for name, val, thr, g in [
    ('G1 κ_t AUROC > 0.65', auroc_kappa, '>0.65', G1),
    ('G2 gap vs shuffled > 0.10', auroc_kappa - shuf_mean, '>0.10', G2),
    ('G3 Mann-Whitney p < 0.05', pval, '<0.05', G3),
    ('G4 EARLY κ_t > 0.60', auroc_early, '>0.60', G4),
    ('G5 slope p < 0.05 (HEADLINE)', slope_pval, '<0.05', G5),
]:
    p(f'  {"PASS" if g else "FAIL"} {name}: actual={val:+.4f} (need {thr})')

n_pass = sum([G1, G2, G3, G4, G5])
p(f'\n=== {n_pass}/5 PASS (vs v2 = 3/4 + slope) ===')

# Comparison to v2
p('\n=== v2 vs v3 comparison ===')
p(f'                    v2 (5 probes)   v3 ({N_PROBES} probes)')
p(f'  κ_t AUROC:       0.677           {auroc_kappa:.3f}')
p(f'  Mann-Whitney p:  0.003           {pval:.4f}')
p(f'  EARLY κ_t:       0.572           {auroc_early:.3f}')
p(f'  Slope succ:      +0.00368        {np.mean(slopes_succ):+.5f}')
p(f'  Slope fail:      +0.00029        {np.mean(slopes_fail):+.5f}')
p(f'  Slope p:         0.0003          {slope_pval:.4f}')

# Save
def _to_serializable(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v

with open(os.path.join(OUT_DIR, 'kappa_t_v3_results.json'), 'w') as f:
    json.dump({
        'probe_aurocs': probe_aurocs,
        'n_probes': N_PROBES,
        'mean_kappa_success': float(np.mean(mean_succ)) if mean_succ else None,
        'mean_kappa_fail': float(np.mean(mean_fail)) if mean_fail else None,
        'mw_pval': float(pval),
        'auroc_mean_kappa': float(auroc_kappa) if not np.isnan(auroc_kappa) else None,
        'gap_vs_shuffled': float(auroc_kappa - shuf_mean) if not np.isnan(auroc_kappa) else None,
        'auroc_early_kappa': float(auroc_early) if not np.isnan(auroc_early) else None,
        'slope_success': float(np.mean(slopes_succ)) if slopes_succ else None,
        'slope_failed': float(np.mean(slopes_fail)) if slopes_fail else None,
        'slope_pval': float(slope_pval),
        'n_pass': int(n_pass),
        'comparison_v2': {
            'v2_auroc': 0.677, 'v2_pval': 0.003, 'v2_early': 0.572,
            'v2_slope_succ': 0.00368, 'v2_slope_fail': 0.00029, 'v2_slope_pval': 0.0003,
        },
    }, f, indent=2)
p(f'\nTotal: {(time.time()-t_start)/60:.1f}min')
p(f'Saved: {os.path.join(OUT_DIR, "kappa_t_v3_results.json")}')
out_f.close()
