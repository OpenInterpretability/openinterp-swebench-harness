"""Generate paper figures for κ_t Cross-Probe Coherence Buildup paper.

Outputs 6 figures to /Volumes/SSD Major/fish/openinterp-swebench-harness/paper/figures/:
  fig1_kappa_t_trajectories.png — example success vs fail κ_t over turns
  fig2_slope_distributions.png — histogram of κ_t slopes per outcome
  fig3_probe_aurocs.png — bar chart of 9 probe AUROCs
  fig4_probe_correlations.png — 9×9 heatmap of probe-score correlations
  fig5_mean_kappa_violin.png — per-trace mean κ_t success vs fail
  fig6_v2_vs_v3_replication.png — v2 vs v3 gate comparison
"""
import os, json, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# Setup matplotlib defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLOR_SUCC = '#2E7D32'  # green
COLOR_FAIL = '#C62828'  # red
COLOR_SHUF = '#9E9E9E'  # grey

DRIVE_ROOT = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
PHASE6_DIR = os.path.join(DRIVE_ROOT, 'swebench_v6_phase6')
CAPTURES_DIR = os.path.join(PHASE6_DIR, 'captures')
TRACES_DIR = os.path.join(PHASE6_DIR, 'traces')
RESULTS_JSON = os.path.join(PHASE6_DIR, 'phase6_results.json')
OUT_DIR = '/Volumes/SSD Major/fish/openinterp-swebench-harness/paper/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# === Load saved κ_t v2 results ===
V2_RESULTS = os.path.join(DRIVE_ROOT, 'kappa_t_v2', 'kappa_t_v2_results.json')
with open(V2_RESULTS) as f:
    v2 = json.load(f)
print(f'v2 results loaded: {list(v2.keys())[:5]}...')

# v3 results in /tmp output but JSON failed — reconstruct from log
v3_summary = {
    'auroc': 0.697, 'mw_pval': 0.0009, 'shuf_mean': 0.503,
    'slope_succ_mean': 0.00251, 'slope_succ_std': 0.00564,
    'slope_fail_mean': 0.00016, 'slope_fail_std': 0.00169,
    'slope_pval': 0.0033,
    'early_auroc': 0.536,
    'probe_aurocs': {
        'A_tool_finish': 1.000, 'B_tool_bash': 1.000, 'C_long_thinking': 0.990,
        'D_tool_ok': 0.998, 'E_repo_ansible': 0.970, 'F_response_long': 0.922,
        'G_early_trace': 0.991, 'H_late_trace': 0.863, 'I_slow_turn': 0.924,
    },
}

# === Re-load captures + recompute κ_t v2 (5 axes) per trace for plotting ===
PROBE_LAYER = 43
POS_INNER = 'think_end'
POS_OUTER = 'turn_end'

print('Loading Phase 6 results + per-turn data...')
with open(RESULTS_JSON) as f:
    phase6_results = json.load(f)

def repo_from_iid(iid):
    return iid.replace('instance_', '').split('__')[0] if iid else 'unknown'

per_trace_turns = {}
for iid in phase6_results:
    trace_path = os.path.join(TRACES_DIR, f'{iid}.json')
    if not os.path.exists(trace_path): continue
    with open(trace_path) as f:
        trace = json.load(f)
    turn_info = {}
    for turn in trace.get('turns', []):
        t = turn['turn_idx']
        tool_calls = turn.get('tool_calls', [])
        primary_tool = tool_calls[0]['name'] if tool_calls else 'none'
        thinking = turn.get('thinking') or ''
        tool_results = turn.get('tool_results', [])
        tool_ok = tool_results[0].get('result', {}).get('ok', False) if tool_results else False
        turn_info[t] = {
            'tool_name': primary_tool, 'thinking_chars': len(thinking),
            'tool_ok': int(bool(tool_ok)),
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

# Load residuals
def load_residuals(iid):
    sp = os.path.join(CAPTURES_DIR, f'{iid}.safetensors')
    mp = os.path.join(CAPTURES_DIR, f'{iid}.meta.json')
    if not (os.path.exists(sp) and os.path.exists(mp)): return None
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

print('Loading L43 paired residuals (5min on Drive)...')
X_list, iid_list, turn_list = [], [], []
for i, iid in enumerate(phase6_results):
    res = load_residuals(iid)
    if res is None: continue
    for (t, vec) in res:
        X_list.append(vec)
        iid_list.append(iid)
        turn_list.append(t)
    if (i + 1) % 20 == 0:
        print(f'  [{i+1}/99]')

X = np.stack(X_list).astype(np.float32)
iid_arr = np.array(iid_list)
turn_arr = np.array(turn_list)
groups = iid_arr.copy()
print(f'Loaded {len(X)} samples')

def get_turn(iid, t, key, default=0):
    return per_trace_turns.get(iid, {}).get(t, {}).get(key, default)

# Train 5 probes v2
y_axes_v2 = {
    'A_tool_finish':   np.array([1 if get_turn(iid_arr[i], int(turn_arr[i]), 'tool_name') == 'finish' else 0 for i in range(len(X))], dtype=int),
    'B_tool_bash':     np.array([1 if get_turn(iid_arr[i], int(turn_arr[i]), 'tool_name') == 'bash' else 0 for i in range(len(X))], dtype=int),
    'C_long_thinking': np.array([get_turn(iid_arr[i], int(turn_arr[i]), 'long_thinking', 0) for i in range(len(X))], dtype=int),
    'D_tool_ok':       np.array([get_turn(iid_arr[i], int(turn_arr[i]), 'tool_ok', 0) for i in range(len(X))], dtype=int),
    'E_repo_ansible':  np.array([1 if per_iid_meta[iid_arr[i]]['repo'] == 'ansible' else 0 for i in range(len(X))], dtype=int),
}

print('Training 5 v2 probes...')
oof_v2 = {}
for name, y in y_axes_v2.items():
    np.random.seed(42)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(X), dtype=np.float32)
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[tr]).astype(np.float32)
        Xte_s = sc.transform(X[te]).astype(np.float32)
        clf = LogisticRegression(class_weight='balanced', max_iter=500, C=0.1, solver='liblinear')
        clf.fit(Xtr_s, y[tr])
        oof[te] = clf.decision_function(Xte_s)
    oof_v2[name] = oof
    print(f'  {name}: done')

# Compute κ_t per trace
names_v2 = list(y_axes_v2.keys())
N_V2 = len(names_v2)
WINDOW = 3

per_trace_scores = {}
for iid in set(iid_arr):
    mask = iid_arr == iid
    idxs = np.where(mask)[0]
    sort_perm = np.argsort(turn_arr[idxs])
    scores_iid = np.stack([oof_v2[n][idxs[sort_perm]] for n in names_v2], axis=1)
    per_trace_scores[iid] = {
        'turns': turn_arr[idxs][sort_perm],
        'scores': scores_iid,
        'failure': per_iid_meta[iid]['failure'],
    }

def compute_kappa(window):
    if window.shape[0] < 3: return np.nan
    cm = np.corrcoef(window.T)
    if np.isnan(cm).any(): return np.nan
    mask = ~np.eye(N_V2, dtype=bool)
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

# === FIGURE 1: κ_t trajectories (example traces) ===
print('Generating Figure 1...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Pick 5 success + 5 fail traces with good length
succ_iids = [iid for iid in per_trace_kappa if per_iid_meta[iid]['failure'] == 0]
fail_iids = [iid for iid in per_trace_kappa if per_iid_meta[iid]['failure'] == 1]
# Pick mid-length ones for clarity
succ_iids_sorted = sorted(succ_iids, key=lambda i: -len(per_trace_kappa[i]))[:8]
fail_iids_sorted = sorted(fail_iids, key=lambda i: -len(per_trace_kappa[i]))[:8]

ax = axes[0]
for iid in succ_iids_sorted:
    k = per_trace_kappa[iid]
    valid = ~np.isnan(k)
    if valid.sum() < 5: continue
    ax.plot(np.arange(len(k))[valid], k[valid], alpha=0.4, color=COLOR_SUCC, lw=1)
# Mean across success traces (resampled to common time)
ax.set_title(f'Success traces (N={len(succ_iids)})', fontweight='bold')
ax.set_xlabel('Turn index')
ax.set_ylabel('κ_t (window=±3)')
ax.set_ylim(0.0, 0.8)
ax.grid(True, alpha=0.3)

ax = axes[1]
for iid in fail_iids_sorted:
    k = per_trace_kappa[iid]
    valid = ~np.isnan(k)
    if valid.sum() < 5: continue
    ax.plot(np.arange(len(k))[valid], k[valid], alpha=0.4, color=COLOR_FAIL, lw=1)
ax.set_title(f'Failed traces (N={len(fail_iids)})', fontweight='bold')
ax.set_xlabel('Turn index')
ax.set_ylim(0.0, 0.8)
ax.grid(True, alpha=0.3)

fig.suptitle(r'$\kappa_t$ Trajectory — Success Builds Coherence, Failure Stays Flat',
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig1_kappa_t_trajectories.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved fig1')

# === FIGURE 2: Slope distributions ===
print('Generating Figure 2...')
slopes_succ, slopes_fail = [], []
for iid, kappa in per_trace_kappa.items():
    v_mask = ~np.isnan(kappa)
    if v_mask.sum() < 5: continue
    turns = np.arange(len(kappa))[v_mask]
    vals = kappa[v_mask]
    slope = np.polyfit(turns, vals, 1)[0]
    (slopes_succ if per_iid_meta[iid]['failure']==0 else slopes_fail).append(slope)

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(-0.015, 0.025, 40)
ax.hist(slopes_succ, bins=bins, alpha=0.6, color=COLOR_SUCC, label=f'Success (N={len(slopes_succ)}, mean={np.mean(slopes_succ):+.5f})', edgecolor='black', linewidth=0.5)
ax.hist(slopes_fail, bins=bins, alpha=0.6, color=COLOR_FAIL, label=f'Failed (N={len(slopes_fail)}, mean={np.mean(slopes_fail):+.5f})', edgecolor='black', linewidth=0.5)
ax.axvline(0, color='black', linestyle='--', alpha=0.5, lw=1)
ax.axvline(np.mean(slopes_succ), color=COLOR_SUCC, linestyle='-', alpha=0.8, lw=2)
ax.axvline(np.mean(slopes_fail), color=COLOR_FAIL, linestyle='-', alpha=0.8, lw=2)
ax.set_xlabel(r'$\kappa_t$ slope over trace turns')
ax.set_ylabel('Trace count')
ax.set_title(r'Per-Trace $\kappa_t$ Slope Distributions (Mann-Whitney $p = 0.0003$)',
             fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig2_slope_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved fig2')

# === FIGURE 3: Probe AUROCs (9 probes) ===
print('Generating Figure 3...')
v3_probes = v3_summary['probe_aurocs']
names_sorted = sorted(v3_probes.keys())
aurocs = [v3_probes[n] for n in names_sorted]
labels = [n.replace('_', ' ').replace('A tool', 'A: tool=').replace('B tool', 'B: tool=').replace('C long', 'C: long').replace('D tool', 'D: tool ').replace('E repo', 'E: repo=').replace('F response', 'F: response').replace('G early', 'G: early').replace('H late', 'H: late').replace('I slow', 'I: slow') for n in names_sorted]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(labels, aurocs, color='#1976D2', edgecolor='black', linewidth=0.5)
ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Chance')
ax.set_xlim(0.4, 1.05)
ax.set_xlabel('AUROC (GroupKFold CV)')
ax.set_title('Per-Turn Probe AUROCs (9 axes on L43 paired residual)', fontweight='bold')
for i, (b, v) in enumerate(zip(bars, aurocs)):
    ax.text(v + 0.005, b.get_y() + b.get_height()/2, f'{v:.3f}', va='center', fontsize=9)
ax.legend(loc='lower right')
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig3_probe_aurocs.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved fig3')

# === FIGURE 4: Probe correlation heatmap (5 axes from v2) ===
print('Generating Figure 4...')
score_mat = np.stack([oof_v2[n] for n in names_v2], axis=1)
corr = np.corrcoef(score_mat.T)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(N_V2))
ax.set_yticks(range(N_V2))
ax.set_xticklabels([n.replace('_', '\n') for n in names_v2], rotation=45, ha='right', fontsize=9)
ax.set_yticklabels([n.replace('_', '\n') for n in names_v2], fontsize=9)
for i in range(N_V2):
    for j in range(N_V2):
        c = 'white' if abs(corr[i, j]) > 0.5 else 'black'
        ax.text(j, i, f'{corr[i, j]:+.2f}', ha='center', va='center', color=c, fontsize=9, fontweight='bold')
ax.set_title('Probe-Score Correlation Matrix (5 axes, v2)', fontweight='bold')
plt.colorbar(im, ax=ax, label='Pearson correlation')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig4_probe_correlations.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved fig4')

# === FIGURE 5: Mean κ_t violin ===
print('Generating Figure 5...')
mean_succ_list, mean_fail_list = [], []
for iid, kappa in per_trace_kappa.items():
    v = kappa[~np.isnan(kappa)]
    if len(v) == 0: continue
    (mean_succ_list if per_iid_meta[iid]['failure'] == 0 else mean_fail_list).append(np.mean(v))

fig, ax = plt.subplots(figsize=(7, 5))
parts = ax.violinplot([mean_succ_list, mean_fail_list], positions=[1, 2], widths=0.7, showmeans=True, showmedians=True)
parts['bodies'][0].set_facecolor(COLOR_SUCC); parts['bodies'][0].set_alpha(0.7)
parts['bodies'][1].set_facecolor(COLOR_FAIL); parts['bodies'][1].set_alpha(0.7)
for partname in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
    if partname in parts:
        parts[partname].set_color('black')
ax.scatter([1]*len(mean_succ_list), mean_succ_list, alpha=0.3, color=COLOR_SUCC, s=20)
ax.scatter([2]*len(mean_fail_list), mean_fail_list, alpha=0.3, color=COLOR_FAIL, s=20)
ax.set_xticks([1, 2])
ax.set_xticklabels([f'Success (N={len(mean_succ_list)})', f'Failed (N={len(mean_fail_list)})'])
ax.set_ylabel(r'Per-trace mean $\kappa_t$')
ax.set_title(r'Per-Trace Mean $\kappa_t$ — Mann-Whitney $p = 0.003$, AUROC $= 0.677$',
             fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig5_mean_kappa_violin.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved fig5')

# === FIGURE 6: v2 vs v3 replication ===
print('Generating Figure 6...')
gates = ['κ_t AUROC', 'Gap vs\nshuffled', 'MW p\n(−log10)', 'Early κ_t', 'Slope p\n(−log10)']
v2_vals = [0.677, 0.176, -np.log10(0.003), 0.572, -np.log10(0.0003)]
v3_vals = [0.697, 0.194, -np.log10(0.0009), 0.536, -np.log10(0.0033)]
thresholds = [0.65, 0.10, -np.log10(0.05), 0.60, -np.log10(0.05)]

x = np.arange(len(gates))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5.5))
b1 = ax.bar(x - width/2, v2_vals, width, label='v2 (5 axes)', color='#1976D2', edgecolor='black', linewidth=0.5)
b2 = ax.bar(x + width/2, v3_vals, width, label='v3 (9 axes)', color='#FFA000', edgecolor='black', linewidth=0.5)
for i, (xi, t) in enumerate(zip(x, thresholds)):
    ax.hlines(t, xi - 0.4, xi + 0.4, colors='red', linestyles='--', lw=1.5, label='Threshold' if i == 0 else '')

# Pass/fail markers
for i in range(len(gates)):
    v2_pass = v2_vals[i] > thresholds[i]
    v3_pass = v3_vals[i] > thresholds[i]
    ax.text(x[i] - width/2, v2_vals[i] + 0.05, '✓' if v2_pass else '✗', ha='center', fontsize=14, fontweight='bold', color='green' if v2_pass else 'red')
    ax.text(x[i] + width/2, v3_vals[i] + 0.05, '✓' if v3_pass else '✗', ha='center', fontsize=14, fontweight='bold', color='green' if v3_pass else 'red')

ax.set_xticks(x)
ax.set_xticklabels(gates)
ax.set_ylabel('Metric value (higher = better)')
ax.set_title('Pre-Registered Gates: v2 vs v3 Replication', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig6_v2_vs_v3_replication.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved fig6')

# === Summary index ===
print('\n=== ALL FIGURES GENERATED ===')
for f in sorted(os.listdir(OUT_DIR)):
    full = os.path.join(OUT_DIR, f)
    sz = os.path.getsize(full) / 1024
    print(f'  {f}: {sz:.0f} KB')
