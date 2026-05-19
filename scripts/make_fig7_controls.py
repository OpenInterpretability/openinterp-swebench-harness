"""Generate fig7_controls_summary.png from kappa_t_controls_results.json.

Layout: 2×3 grid
  [0,0] C1 — slope vs trace-length scatter, colored by class
  [0,1] C1 — slope-residuals distribution by class (length-adjusted)
  [0,2] C2 — within-trace shuffle null histogram vs real diff
  [1,0] C3 — orth-pair κ_t slope by class
  [1,1] C4 — trace-length distribution by class
  [1,2] Summary table panel (p-values + PASS/FAIL)
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DRIVE_ROOT = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
RESULTS_JSON = os.path.join(DRIVE_ROOT, 'kappa_t_controls', 'kappa_t_controls_results.json')
OUT_PATH = '/Volumes/SSD Major/fish/openinterp-swebench-harness/paper/figures/fig7_controls_summary.png'

if not os.path.exists(RESULTS_JSON):
    raise SystemExit(f'Controls JSON not found: {RESULTS_JSON}')

with open(RESULTS_JSON) as f:
    R = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('κ_t Robustness Controls — N=99 SWE-bench Pro / Qwen3.6-27B', fontsize=14, y=1.00)

# Pull C1, C2 data (we only have summary stats; for scatter we need to re-run, so we approximate)
# C1: slope-residuals by class
ax = axes[0, 0]
c1 = R['C1_length_confound']
ax.bar(['Success\nN=40', 'Failure\nN=59'],
       [c1['resid_slope_succ'], c1['resid_slope_fail']],
       color=['#2ca02c', '#d62728'])
ax.set_title(f'C1: slope after length-regression\np = {c1["partial_pval"]:.4f}')
ax.set_ylabel('Mean slope-residual (length-adjusted)')
ax.axhline(0, color='k', lw=0.5)

# C1 inset: Pearson correlation as text
ax = axes[0, 1]
ax.axis('off')
c1text = (
    f'C1 detail — does trace length explain slope?\n\n'
    f'Pearson(n_turns, slope)  = {c1["pearson_r"]:+.3f}  (p={c1["pearson_p"]:.4f})\n'
    f'Spearman(n_turns, slope) = {c1["spearman_r"]:+.3f}  (p={c1["spearman_p"]:.4f})\n\n'
    f'After regressing OUT length:\n'
    f'  Success residual = {c1["resid_slope_succ"]:+.5f}\n'
    f'  Failure residual = {c1["resid_slope_fail"]:+.5f}\n'
    f'  Mann-Whitney p   = {c1["partial_pval"]:.4f}\n\n'
    f'VERDICT: {"SLOPE survives length control" if c1["partial_pval"] < 0.05 else "Slope may be length artifact"}'
)
ax.text(0.05, 0.95, c1text, transform=ax.transAxes, fontsize=10, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray'))

# C2: shuffle null histogram
ax = axes[0, 2]
c2 = R['C2_within_trace_shuffle']
# We don't have the raw null distribution saved, just mean+std. Approximate as Gaussian.
null_mean = c2['null_mean']
null_std = c2['null_std']
real_diff = c2['real_class_diff']
x = np.linspace(null_mean - 4*null_std, max(null_mean + 4*null_std, real_diff + 0.0005), 200)
y = (1/(null_std * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - null_mean) / null_std)**2)
ax.plot(x, y, color='#888', label=f'Null (shuffled)\nμ={null_mean:+.4f}, σ={null_std:.4f}')
ax.fill_between(x, y, color='#888', alpha=0.3)
ax.axvline(real_diff, color='red', lw=2, label=f'Real diff = {real_diff:+.4f}')
ax.set_title(f'C2: within-trace turn shuffle\np = {c2["shuf_pval"]:.4f}')
ax.set_xlabel('Class-mean slope diff (succ − fail)')
ax.set_ylabel('Null density')
ax.legend(fontsize=8, loc='upper left')

# C3: orth-pair slope
ax = axes[1, 0]
c3 = R['C3_orthogonal_pair']
if c3['pval'] is not None:
    ax.bar(['Success', 'Failure'],
           [c3['slope_succ'], c3['slope_fail']],
           color=['#2ca02c', '#d62728'])
    ax.set_title(f'C3: orthogonal-pair κ_t (N={c3["n_orth_probes"]} probes)\np = {c3["pval"]:.4f}')
    ax.set_ylabel('Mean κ_t slope')
    ax.axhline(0, color='k', lw=0.5)
    probe_text = '\n'.join(c3['orth_names'])
    ax.text(0.98, 0.97, f'Probes:\n{probe_text}', transform=ax.transAxes, fontsize=8,
            va='top', ha='right', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
else:
    ax.axis('off')
    ax.text(0.5, 0.5, f'C3: insufficient orth probes\n({c3["n_orth_probes"]} found, need ≥3)',
            ha='center', va='center', transform=ax.transAxes, fontsize=11)

# C4: trace length distribution
ax = axes[1, 1]
c4 = R['C4_length_distribution']
# Approximation: show medians + a simple bar
ax.bar(['Success', 'Failure'],
       [c4['len_succ_median'], c4['len_fail_median']],
       yerr=[abs(c4['len_succ_mean'] - c4['len_succ_median']),
             abs(c4['len_fail_mean'] - c4['len_fail_median'])],
       color=['#2ca02c', '#d62728'], capsize=5)
ax.set_title(f'C4: trace length by class (info)\nMW p = {c4["len_mw_pval"]:.3g}')
ax.set_ylabel('Median n_turns')
reasons_text = '\n'.join(f'{k}: {v}' for k, v in c4['finish_reasons_failure'].items())
ax.text(0.98, 0.97, f'Failure reasons:\n{reasons_text}', transform=ax.transAxes, fontsize=8,
        va='top', ha='right', family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Summary table panel
ax = axes[1, 2]
ax.axis('off')
orig = R['orig']
c5 = R['C5_iso_length_max_turns']
def pp(p): return f'{p:.4f}' if p is not None and not np.isnan(p) else 'N/A'
def status(p, thr=0.05): return '✓' if p is not None and not np.isnan(p) and p < thr else '✗' if p is not None else '–'

table_text = f"""SUMMARY — slope-test robustness

Original (5 probes, v2):     p={pp(orig['slope_pval'])}   {status(orig['slope_pval'])}
C1 length-regressed slope:   p={pp(c1['partial_pval'])}   {status(c1['partial_pval'])}
C2 within-trace shuffle:     p={pp(c2['shuf_pval'])}   {status(c2['shuf_pval'])}
C3 orthogonal-pair only:     p={pp(c3['pval'])}   {status(c3['pval'])}
C5 iso-length (max_turns):   p={pp(c5['pval'])}   {status(c5['pval'])}

Critical (orig + C1 + C2): {R['n_critical_pass']}/3 pass
C3 + C5: stronger if pass, not blocking.

Effect sizes (orig):
  Success slope: {orig['slope_succ']:+.5f}
  Failure slope: {orig['slope_fail']:+.5f}
  Ratio: {orig['slope_succ']/max(abs(orig['slope_fail']), 1e-9):.1f}×"""
ax.text(0.02, 0.98, table_text, transform=ax.transAxes, fontsize=10,
        va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#fffbea', edgecolor='gray'))

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=130, bbox_inches='tight')
print(f'Saved: {OUT_PATH}')
