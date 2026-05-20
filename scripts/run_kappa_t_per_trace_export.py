"""Path E: extract per-trajectory κ_t time series from Phase 6 captures for failure-mode clustering.

Differs from run_kappa_t_v3.py: saves per-trace κ_t shape (turn x κ_t value) instead of just
aggregate statistics. Output is JSON with per-trace records for downstream clustering.

Uses κ_t paper's 5-probe stack at L43 paired (think_end + turn_end concat, 10240 dim).
CPU-only — loading + analysis takes ~3-5 minutes on local Drive sync.

Output: {iid -> {success: bool, n_turns: int, kappa_t: [float, ...], probe_scores: {probe: [...]}}}
"""
import os, json, time, warnings
import numpy as np
import torch
warnings.filterwarnings('ignore')

from safetensors.torch import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

DRIVE = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6')
CAPTURES = os.path.join(DRIVE, 'captures')
TRACES = os.path.join(DRIVE, 'traces')
RESULTS_JSON = os.path.join(DRIVE, 'phase6_results.json')
OUT_JSON = os.path.join(DRIVE, 'kappa_t_per_trace.json')

PROBE_LAYER = 'L43'
POS_INNER = 'think_end'   # paired with turn_end at same layer
POS_OUTER = 'turn_end'
WINDOW = 5                 # κ_t paper baseline window
PROBES = ['A_tool_finish', 'B_tool_bash', 'C_long_thinking', 'D_tool_ok', 'E_repo_ansible']

t0 = time.time()
def L(msg): print(f'[{time.time()-t0:.0f}s] {msg}', flush=True)

# === 1. Load Phase 6 outcomes + trace metadata ===
L('Loading phase6 results + trace metadata...')
with open(RESULTS_JSON) as f:
    phase6 = json.load(f)

per_trace_turn_info = {}
for iid, meta in phase6.items():
    trace_path = os.path.join(TRACES, f'{iid}.json')
    if not os.path.exists(trace_path):
        continue
    with open(trace_path) as f:
        trace = json.load(f)
    turns = trace.get('turns', [])
    n_turns_trace = len(turns)
    repo = iid.replace('instance_', '').split('__')[0] if iid else 'unknown'
    info = {}
    for turn in turns:
        t = turn['turn_idx']
        tool_calls = turn.get('tool_calls', [])
        primary_tool = tool_calls[0]['name'] if tool_calls else 'none'
        thinking = turn.get('thinking') or ''
        tool_results = turn.get('tool_results', [])
        tool_ok = tool_results[0].get('result', {}).get('ok', False) if tool_results else False
        info[t] = {
            'tool_name': primary_tool,
            'thinking_chars': len(thinking),
            'tool_ok': int(bool(tool_ok)),
            'turn_idx': t,
            'n_turns_trace': n_turns_trace,
            'repo': repo,
        }
    # Compute median for binarization
    tcs = [v['thinking_chars'] for v in info.values()]
    med_tc = float(np.median(tcs)) if tcs else 0
    for v in info.values():
        v['long_thinking'] = int(v['thinking_chars'] > med_tc)
    per_trace_turn_info[iid] = info

L(f'Loaded {len(per_trace_turn_info)} traces')

# === 2. Load L43 paired residuals + build feature matrix ===
L('Loading L43 paired residuals (this may take 2-3 min on Drive sync)...')

# For each turn in each trace: residual = concat(think_end, turn_end) at L43
# We need to match the position tokens within each capture
def load_capture(iid):
    """Returns dict {turn_idx -> 10240-dim float32 array} or None if file missing."""
    fpath = os.path.join(CAPTURES, f'{iid}.safetensors')
    if not os.path.exists(fpath):
        return None
    try:
        with safe_open(fpath, framework='pt') as f:
            keys = list(f.keys())
            # Find all (turn, position) combos for L43
            # Key format: t<NNN>_<position>_p<token_pos>_L<layer>
            turn_data = {}
            for k in keys:
                if not k.endswith(f'_{PROBE_LAYER}'):
                    continue
                parts = k.split('_')
                turn_str = parts[0]  # t000
                turn = int(turn_str[1:])
                if POS_INNER in k:
                    pos_key = POS_INNER
                elif POS_OUTER in k:
                    pos_key = POS_OUTER
                else:
                    continue
                if turn not in turn_data:
                    turn_data[turn] = {}
                turn_data[turn][pos_key] = f.get_tensor(k).float().numpy()
            # Pair turns that have BOTH positions
            paired = {}
            for turn, d in turn_data.items():
                if POS_INNER in d and POS_OUTER in d:
                    paired[turn] = np.concatenate([d[POS_INNER], d[POS_OUTER]])
            return paired
    except Exception as e:
        L(f'  WARN {iid}: {e}')
        return None

# Build full per-turn feature matrix
X_rows = []          # 10240-dim residuals
y_rows = {p: [] for p in PROBES}   # per-probe label
group_iid = []
turn_idx = []

iids = sorted(per_trace_turn_info.keys())
for i, iid in enumerate(iids):
    if i % 20 == 0:
        L(f'  Loading capture {i+1}/{len(iids)}...')
    residuals_by_turn = load_capture(iid)
    if residuals_by_turn is None:
        continue
    info = per_trace_turn_info[iid]
    for t, x in residuals_by_turn.items():
        if t not in info:
            continue
        tinfo = info[t]
        X_rows.append(x)
        group_iid.append(iid)
        turn_idx.append(t)
        # Labels (binary, per κ_t paper)
        y_rows['A_tool_finish'].append(int(tinfo['tool_name'] == 'finish'))
        y_rows['B_tool_bash'].append(int(tinfo['tool_name'] == 'bash'))
        y_rows['C_long_thinking'].append(tinfo['long_thinking'])
        y_rows['D_tool_ok'].append(tinfo['tool_ok'])
        y_rows['E_repo_ansible'].append(int(tinfo['repo'] == 'ansible'))

X = np.array(X_rows)
group_iid = np.array(group_iid)
turn_idx = np.array(turn_idx)
y = {p: np.array(v) for p, v in y_rows.items()}
L(f'Feature matrix: X shape={X.shape}, N traces={len(set(group_iid))}')

# === 3. Train probes via GroupKFold + collect out-of-fold scores ===
L('Training 5 probes via GroupKFold (3-fold by instance_id)...')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

oof_scores = {p: np.zeros(len(X)) for p in PROBES}
for probe in PROBES:
    y_probe = y[probe]
    if y_probe.sum() < 5 or (1 - y_probe).sum() < 5:
        L(f'  WARN {probe}: degenerate label, skipping')
        oof_scores[probe] = y_probe.astype(float)  # fallback
        continue
    gkf = GroupKFold(n_splits=3)
    for fold, (tr, te) in enumerate(gkf.split(X_scaled, y_probe, groups=group_iid)):
        clf = LogisticRegression(C=0.1, class_weight='balanced', solver='liblinear', max_iter=1000)
        clf.fit(X_scaled[tr], y_probe[tr])
        oof_scores[probe][te] = clf.predict_proba(X_scaled[te])[:, 1]
    auroc = roc_auc_score(y_probe, oof_scores[probe])
    L(f'  {probe}: AUROC={auroc:.3f}')

# === 4. Compute κ_t per turn per trace ===
L('Computing per-turn κ_t (window=5) for each trace...')

per_trace = {}
for iid in sorted(set(group_iid)):
    mask = group_iid == iid
    turns = turn_idx[mask]
    sort_ord = np.argsort(turns)
    turns_sorted = turns[sort_ord]
    # Per-probe score vectors for this trace, sorted by turn
    probe_scores_this = {p: oof_scores[p][mask][sort_ord] for p in PROBES}
    n_turns = len(turns_sorted)
    kappa_series = []
    for t_idx in range(n_turns):
        lo = max(0, t_idx - WINDOW + 1)
        hi = t_idx + 1
        if hi - lo < 3:   # need at least 3 turns in window
            kappa_series.append(float('nan'))
            continue
        # Build [n_in_window x n_probes] matrix
        W_mat = np.stack([probe_scores_this[p][lo:hi] for p in PROBES], axis=1)
        # Pairwise |Pearson|
        if W_mat.std(axis=0).min() < 1e-9:
            kappa_series.append(0.0)
            continue
        corr = np.corrcoef(W_mat, rowvar=False)
        # Mean of |off-diagonal|
        n = corr.shape[0]
        off_diag = corr[np.triu_indices(n, k=1)]
        kappa_series.append(float(np.mean(np.abs(off_diag))))

    # Outcome label
    success = bool(phase6[iid].get('patch_generated', False))
    per_trace[iid] = {
        'success': success,
        'n_turns': n_turns,
        'turns': turns_sorted.tolist(),
        'kappa_t': kappa_series,
        'probe_scores': {p: probe_scores_this[p].tolist() for p in PROBES},
        'repo': per_trace_turn_info[iid][turns_sorted[0]]['repo'] if n_turns > 0 else 'unknown',
    }

# === 5. Save ===
L(f'Saving per-trace data to {OUT_JSON}...')
with open(OUT_JSON, 'w') as f:
    json.dump(per_trace, f, indent=1)
L(f'Done. {len(per_trace)} traces, {sum(t["success"] for t in per_trace.values())} successful, {sum(not t["success"] for t in per_trace.values())} failed.')
