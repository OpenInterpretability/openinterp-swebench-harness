"""Generate fig8_early_late_half.png from kappa_t_c6_results.json.

PRIMARY new figure: U-shape decomposition showing the early-half drop
and late-half rise, by class.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

DRIVE_ROOT = os.path.expanduser('~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs')
C6_JSON = os.path.join(DRIVE_ROOT, 'kappa_t_c6', 'kappa_t_c6_results.json')
OUT_PATH = '/Volumes/SSD Major/fish/openinterp-swebench-harness/paper/figures/fig8_early_late_half.png'

with open(C6_JSON) as f:
    R = json.load(f)

el = R['early_late_half']
first_n = R  # Has first_10_turns, etc.

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle('U-Shape Decomposition of κ_t (N=99 SWE-bench Pro / Qwen3.6-27B)', fontsize=13, y=1.02)

# Left: bar chart of early/late slopes by class
ax = axes[0]
labels = ['Early-half\n(exploration)', 'Late-half\n(consolidation)']
succ_vals = [el['early_succ_slope'], el['late_succ_slope']]
fail_vals = [el['early_fail_slope'], el['late_fail_slope']]
x = np.arange(len(labels))
w = 0.35
b1 = ax.bar(x - w/2, succ_vals, w, label='Success (N=39)', color='#2ca02c')
b2 = ax.bar(x + w/2, fail_vals, w, label='Failure (N=59)', color='#d62728')
for bars, vals in [(b1, succ_vals), (b2, fail_vals)]:
    for b, v in zip(bars, vals):
        y = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, y + (0.0005 if y >= 0 else -0.0015),
                f'{v:+.4f}', ha='center', va='bottom' if y >= 0 else 'top', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Mean per-trace slope of κ_t')
ax.axhline(0, color='k', lw=0.5)
ax.legend(loc='upper left')
ax.set_title('Both halves separate outcomes — in opposite directions')
# Annotate p-values
ax.text(0 - w/2, succ_vals[0] - 0.003, f'p={el["early_pval"]:.4f}', ha='center', fontsize=9,
        color='black', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fffbea', edgecolor='black'))
ax.text(1 - w/2, succ_vals[1] + 0.002, f'p={el["late_pval"]:.5f}', ha='center', fontsize=9,
        color='black', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fffbea', edgecolor='black'))

# Right: interpretation text
ax = axes[1]
ax.axis('off')
text = """U-Shape Mechanism

Successful trajectories (N=39):
  early-half slope = −0.0078 (drops fast)
  late-half  slope = +0.0149 (rises fast)
  → high-amplitude EXPLORE→CONSOLIDATE arc

Failed trajectories (N=59):
  early-half slope = −0.0007 (≈ flat)
  late-half  slope = +0.0025 (≈ flat)
  → no engagement of either phase

Interpretation
  Successful agents have a dynamic κ_t
  trajectory: probes de-correlate during
  exploration, then re-couple during
  consolidation. Failed agents lack this
  oscillation — κ_t stays flat throughout.

  This is the INVERSE of cardiac uncoupling.
  Cardiology: failure = collapse from coupled
  baseline.
  Here: failure = absence of explore-consolidate
  oscillation from a baseline that should be
  oscillatory.

Why early/late decomposition matters
  The monolithic per-trace slope averaged the
  early drop and late rise into a single number.
  Pre-registered control C1 found that
  monolithic slope was substantially length-
  confounded (Mann-Whitney p=0.56 after
  length regression).
  The early/late decomposition is length-
  normalized by construction and the class
  effects are STRONGER (p=2e-4 + p=4e-5) than
  the original monolithic statistic (p=3e-3)."""
ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9.5,
        va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#fafafa', edgecolor='gray'))

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=130, bbox_inches='tight')
print(f'Saved: {OUT_PATH}')
