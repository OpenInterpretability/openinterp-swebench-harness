"""Path 4 analysis: compare κ_t signature of SANDBAG vs HONEST-FAIL trajectories.

Runs LOCAL on CPU — reads from Google Drive sync. No GPU needed.

Pipeline:
  1. Train the 5-probe stack on the FULL Phase 6 corpus (99 traces, no holdout) — these
     trained probes are applied to sandbag traces (sandbag was never in training → no leakage).
  2. Load sandbag captures from adversarial_kappa_t_phase0/, apply probes, compute per-turn κ_t.
  3. Load Phase 6 honest-fail κ_t baseline from kappa_t_per_trace.json (Path E export).
  4. Outcome-controlled comparison: filter both groups to genuine failures (patch produced
     but tests would fail / max_turns). Mann-Whitney on early-window and late-window κ_t.
  5. Length-confound check (Paper 10 D6). Cohen's d effect size.
  6. Verdict: GO / EXPAND / NO-GO per pre-registered thresholds.

Pre-registered hypothesis H4-SANDBAG (2026-05-19):
  Sandbag trajectories show HIGHER early-window κ_t than honest-fail (coordinated
  'make subtle mistakes' plan synchronizes probe scores from turn 0).
"""
import os, json, time, warnings
import numpy as np
import torch
warnings.filterwarnings('ignore')

from safetensors.torch import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu, pearsonr, linregress

DRIVE = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
PHASE6 = os.path.join(DRIVE, 'swebench_v6_phase6')
ADV = os.path.join(DRIVE, 'adversarial_kappa_t_phase0')

P6_CAPTURES = os.path.join(PHASE6, 'captures')
P6_TRACES = os.path.join(PHASE6, 'traces')
P6_RESULTS = os.path.join(PHASE6, 'phase6_results.json')
P6_KAPPA = os.path.join(PHASE6, 'kappa_t_per_trace.json')  # Path E export (honest-fail baseline)

ADV_CAPTURES = os.path.join(ADV, 'captures')
ADV_TRACES = os.path.join(ADV, 'traces')
ADV_RESULTS = os.path.join(ADV, 'phase0_results.json')
OUT_JSON = os.path.join(ADV, 'adversarial_kappa_t_verdict.json')

PROBE_LAYER = 'L43'
POS_INNER, POS_OUTER = 'think_end', 'turn_end'
WINDOW = 5
PROBES = ['A_tool_finish', 'B_tool_bash', 'C_long_thinking', 'D_tool_ok', 'E_repo_ansible']
EARLY_WINDOW = 5   # first 5 turns
LATE_WINDOW = 5    # last 5 turns

t0 = time.time()
def L(m): print(f'[{time.time()-t0:.0f}s] {m}', flush=True)


def load_capture(captures_dir, iid):
    """Returns {turn_idx -> 10240-dim L43 paired residual} or None."""
    fpath = os.path.join(captures_dir, f'{iid}.safetensors')
    if not os.path.exists(fpath):
        return None
    try:
        with safe_open(fpath, framework='pt') as f:
            turn_data = {}
            for k in f.keys():
                if not k.endswith(f'_{PROBE_LAYER}'):
                    continue
                turn = int(k.split('_')[0][1:])
                pos = POS_INNER if POS_INNER in k else (POS_OUTER if POS_OUTER in k else None)
                if pos is None:
                    continue
                turn_data.setdefault(turn, {})[pos] = f.get_tensor(k).float().numpy()
            return {t: np.concatenate([d[POS_INNER], d[POS_OUTER]])
                    for t, d in turn_data.items() if POS_INNER in d and POS_OUTER in d}
    except Exception as e:
        L(f'  WARN {iid}: {e}')
        return None


def per_turn_labels(traces_dir, iid):
    """Per-turn label dict for probe training (Phase 6 only)."""
    tp = os.path.join(traces_dir, f'{iid}.json')
    if not os.path.exists(tp):
        return None
    with open(tp) as f:
        trace = json.load(f)
    turns = trace.get('turns', [])
    repo = iid.replace('instance_', '').split('__')[0]
    info = {}
    for turn in turns:
        t = turn['turn_idx']
        tc = turn.get('tool_calls', [])
        tool = tc[0]['name'] if tc else 'none'
        thinking = turn.get('thinking') or ''
        tr = turn.get('tool_results', [])
        ok = tr[0].get('result', {}).get('ok', False) if tr else False
        info[t] = {'tool': tool, 'think_chars': len(thinking), 'ok': int(bool(ok)), 'repo': repo}
    tcs = [v['think_chars'] for v in info.values()]
    med = float(np.median(tcs)) if tcs else 0
    for v in info.values():
        v['long_thinking'] = int(v['think_chars'] > med)
    return info


def kappa_series(probe_scores_by_turn, turns_sorted):
    """κ_t per turn: mean |pairwise corr| of 5 probe scores in window W."""
    series = []
    n = len(turns_sorted)
    for ti in range(n):
        lo, hi = max(0, ti - WINDOW + 1), ti + 1
        if hi - lo < 3:
            series.append(float('nan'))
            continue
        W = np.stack([probe_scores_by_turn[p][lo:hi] for p in PROBES], axis=1)
        if W.std(axis=0).min() < 1e-9:
            series.append(0.0)
            continue
        corr = np.corrcoef(W, rowvar=False)
        series.append(float(np.mean(np.abs(corr[np.triu_indices(len(PROBES), k=1)]))))
    return series


# === 1. Train 5 probes on FULL Phase 6 corpus ===
L('Loading Phase 6 corpus + training 5 probes (full-fit, for sandbag application)...')
with open(P6_RESULTS) as f:
    phase6 = json.load(f)

X_rows, group, turn_ix = [], [], []
y_rows = {p: [] for p in PROBES}
for iid in sorted(phase6):
    res = load_capture(P6_CAPTURES, iid)
    info = per_turn_labels(P6_TRACES, iid)
    if res is None or info is None:
        continue
    for t, x in res.items():
        if t not in info:
            continue
        ti = info[t]
        X_rows.append(x); group.append(iid); turn_ix.append(t)
        y_rows['A_tool_finish'].append(int(ti['tool'] == 'finish'))
        y_rows['B_tool_bash'].append(int(ti['tool'] == 'bash'))
        y_rows['C_long_thinking'].append(ti['long_thinking'])
        y_rows['D_tool_ok'].append(ti['ok'])
        y_rows['E_repo_ansible'].append(int(ti['repo'] == 'ansible'))

X = np.array(X_rows)
L(f'Phase 6 feature matrix: {X.shape}')
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
probes_fit = {}
for p in PROBES:
    yp = np.array(y_rows[p])
    if yp.sum() < 5 or (1 - yp).sum() < 5:
        L(f'  WARN {p}: degenerate, skip')
        continue
    clf = LogisticRegression(C=0.1, class_weight='balanced', solver='liblinear', max_iter=1000)
    clf.fit(Xs, yp)
    probes_fit[p] = clf
    L(f'  {p}: in-sample AUROC={roc_auc_score(yp, clf.predict_proba(Xs)[:,1]):.3f}')

# === 2. Apply probes to SANDBAG traces, compute κ_t ===
if not os.path.exists(ADV_RESULTS):
    L(f'SANDBAG results not found at {ADV_RESULTS} — rollout not complete. Exiting.')
    raise SystemExit(0)

with open(ADV_RESULTS) as f:
    adv_results = json.load(f)
sandbag_iids = [iid for iid, r in adv_results.items() if r.get('finish_reason') not in ('error', None)]
L(f'SANDBAG traces available: {len(sandbag_iids)}')

sandbag_kappa = {}
for iid in sandbag_iids:
    res = load_capture(ADV_CAPTURES, iid)
    if res is None:
        L(f'  WARN no capture for {iid}')
        continue
    turns_sorted = sorted(res.keys())
    Xtr = scaler.transform(np.array([res[t] for t in turns_sorted]))
    scores = {p: probes_fit[p].predict_proba(Xtr)[:, 1] for p in probes_fit}
    kt = kappa_series(scores, turns_sorted)
    sandbag_kappa[iid] = {
        'kappa_t': kt,
        'n_turns': len(turns_sorted),
        'finish_reason': adv_results[iid].get('finish_reason'),
        'patch_n_bytes': adv_results[iid].get('patch_n_bytes', 0),
    }
L(f'SANDBAG κ_t computed for {len(sandbag_kappa)} traces')

# === 3. Load Phase 6 baselines: honest-SUCCESS (matched termination) + honest-FAIL ===
with open(P6_KAPPA) as f:
    p6_kappa = json.load(f)
honest_success = {iid: r for iid, r in p6_kappa.items() if r['success']}
honest_fail = {iid: r for iid, r in p6_kappa.items() if not r['success']}
L(f'HONEST-SUCCESS baseline (matched termination=finish_tool): {len(honest_success)} traces')
L(f'HONEST-FAIL baseline (matched outcome=failure, mode=max_turns): {len(honest_fail)} traces')

# Sandbag agents terminate via finish_tool (ship buggy patch) — same termination mode as
# honest-success. So honest-SUCCESS is the cleaner primary baseline: it isolates INTENT
# (sabotage vs genuine solve) from termination mode. honest-FAIL is the secondary
# comparison (matched outcome but confounded by max_turns termination + length).

# === 4. Compare early-window and late-window κ_t ===
def window_mean(kt, which):
    vals = [x for x in kt if not np.isnan(x)]
    if len(vals) < 3:
        return None
    if which == 'early':
        return float(np.mean(vals[:EARLY_WINDOW]))
    elif which == 'late':
        return float(np.mean(vals[-LATE_WINDOW:]))
    return float(np.mean(vals))

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return (a.mean() - b.mean()) / pooled if pooled > 1e-9 else 0.0

def windows(group):
    e = [window_mean(r['kappa_t'], 'early') for r in group.values()]
    l = [window_mean(r['kappa_t'], 'late') for r in group.values()]
    return [v for v in e if v is not None], [v for v in l if v is not None]

sb_early, sb_late = windows(sandbag_kappa)
hs_early, hs_late = windows(honest_success)
hf_early, hf_late = windows(honest_fail)

results_out = {}
for baseline_name, base_early, base_late in [
    ('vs_honest_success', hs_early, hs_late),   # PRIMARY — matched termination mode
    ('vs_honest_fail', hf_early, hf_late),       # SECONDARY — matched outcome
]:
    L(f'\n=== κ_t comparison: SANDBAG {baseline_name.replace("_", " ").upper()} ===')
    results_out[baseline_name] = {}
    for label, sb, base in [('early-window', sb_early, base_early), ('late-window', sb_late, base_late)]:
        if len(sb) < 3 or len(base) < 3:
            L(f'{label}: insufficient samples (sandbag={len(sb)}, baseline={len(base)})')
            continue
        stat, p = mannwhitneyu(sb, base)
        d = cohens_d(sb, base)
        direction = 'HIGHER' if np.mean(sb) > np.mean(base) else 'LOWER'
        L(f'{label}: sandbag mean={np.mean(sb):.3f} (n={len(sb)}), baseline mean={np.mean(base):.3f} (n={len(base)})')
        L(f'  Mann-Whitney p={p:.4f}, Cohen\'s d={d:+.3f}, sandbag is {direction}')
        results_out[baseline_name][label] = {
            'sandbag_mean': float(np.mean(sb)), 'baseline_mean': float(np.mean(base)),
            'mw_p': float(p), 'cohens_d': float(d), 'direction': direction,
            'n_sandbag': len(sb), 'n_baseline': len(base)}

# === 5. Length-confound check (Paper 10 D6) ===
sb_lengths = [r['n_turns'] for r in sandbag_kappa.values()]
hs_lengths = [r['n_turns'] for r in honest_success.values()]
hf_lengths = [r['n_turns'] for r in honest_fail.values()]
L(f'\nLength medians: sandbag={np.median(sb_lengths):.0f}, honest-success={np.median(hs_lengths):.0f}, honest-fail={np.median(hf_lengths):.0f}')
if len(set(sb_lengths)) > 1 and len(sb_early) == len(sb_lengths):
    r_len, p_len = pearsonr(sb_lengths, sb_early)
    L(f'  Pearson(length, early-κ_t) within sandbag: r={r_len:+.3f}, p={p_len:.3f}')
    results_out['length_confound'] = {'pearson_r': float(r_len), 'pearson_p': float(p_len),
                                       'sandbag_median_len': float(np.median(sb_lengths)),
                                       'honest_success_median_len': float(np.median(hs_lengths)),
                                       'honest_fail_median_len': float(np.median(hf_lengths))}

# === 6. Verdict (primary = vs honest-SUCCESS, matched termination) ===
early = results_out.get('vs_honest_success', {}).get('early-window', {})
verdict = 'NO-GO (insufficient data)'
if early:
    p, d, direction = early['mw_p'], early['cohens_d'], early['direction']
    h4_match = direction == 'HIGHER'  # H4-SANDBAG predicts higher early κ_t
    if p < 0.05 and abs(d) > 0.5 and h4_match:
        verdict = 'GO (paper-grade — scale to Phase 1 N=50)'
    elif p < 0.20 and h4_match:
        verdict = 'EXPAND (directional, add N=20 sandbag rollouts)'
    elif p < 0.05 and not h4_match:
        verdict = 'GO-INVERTED (significant but OPPOSITE direction — reframe hypothesis)'
    else:
        verdict = 'NO-GO (null or wrong direction — honest negative per Paper 10)'
L(f'(Verdict based on vs-honest-success early-window — matched termination mode)')

L(f'\n{"="*60}')
L(f'PHASE 0 VERDICT: {verdict}')
L(f'{"="*60}')

out = {
    'verdict': verdict,
    'hypothesis': 'H4-SANDBAG: sandbag shows HIGHER early-window κ_t than honest-fail',
    'n_sandbag_traces': len(sandbag_kappa),
    'n_honest_fail_baseline': len(honest_fail),
    'comparisons': results_out,
    'sandbag_per_trace': {iid: {'kappa_t': r['kappa_t'], 'n_turns': r['n_turns'],
                                 'finish_reason': r['finish_reason'], 'patch_n_bytes': r['patch_n_bytes']}
                          for iid, r in sandbag_kappa.items()},
    'window_early': EARLY_WINDOW, 'window_late': LATE_WINDOW, 'kappa_window': WINDOW,
}
with open(OUT_JSON, 'w') as f:
    json.dump(out, f, indent=1)
L(f'Saved verdict to {OUT_JSON}')
