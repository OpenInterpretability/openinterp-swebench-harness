"""
Exp D — Forced-finish counterfactual: are WANDERING patches actually correct?

Tests the causal hypothesis from Tool-Entropy paper §13:
  "WANDERING agents have consolidated a mid-layer verdict (model 'knows' it's done),
   but edge circuits that translate verdict → finish_tool haven't aligned."

If true, the patches that WANDERING agents PRODUCED but never SUBMITTED should pass
SWE-bench Pro tests at a rate comparable to SUCCESS agent patches. If WANDERING
patches FAIL at much higher rate than SUCCESS, the mechanism is more complex than
"pure action-emission failure" — WANDERING agents would be self-deceived about
patch quality, not just about completion timing.

Uses existing Phase 6b Docker eval data (phase6b_results.json) which has 3 conditions:
  - none:   baseline (no patch) → expected fail
  - golden: ground-truth reference patch → expected pass
  - agent:  Qwen3.6-27B's own generated patch → THE question

Decision rule:
  agent.n_pass > 0 AND agent.n_fail == 0 → PASS
  agent.n_pass == 0 OR agent.n_fail > 0  → FAIL (or anything else)
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact

PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")
OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")

# Load sub-class assignments from inflection_results
infl = json.load(open(OUT_DIR / "inflection_results.json"))
sub_class = {}
for t in infl['per_trajectory']:
    if t['label'] == 1:
        sub_class[t['iid']] = 'success'
    elif t.get('lock_fail_0.40') is not None:
        sub_class[t['iid']] = 'locked'
    else:
        sub_class[t['iid']] = 'wandering'

# Load eval data
p6b = json.load(open(PHASE6 / "phase6b" / "phase6b_results.json"))
print(f"phase6b eval entries: {len(p6b)}")

# Match eval to sub-classes
def classify_verdict(condition_result):
    """Return PASS / FAIL / ERROR / NO_PATCH based on pytest results."""
    if not condition_result:
        return 'NO_DATA'
    n_pass = condition_result.get('n_pass', 0)
    n_fail = condition_result.get('n_fail', 0)
    n_err = condition_result.get('n_err', 0)
    n_skip = condition_result.get('n_skip', 0)
    rc = condition_result.get('rc', -1)
    if n_pass > 0 and n_fail == 0 and n_err == 0:
        return 'PASS'
    if n_fail > 0 or rc != 0:
        return 'FAIL'
    if n_err > 0:
        return 'ERROR'
    return 'OTHER'

# Build per-class statistics
class_results = {'success': {'pass':0,'fail':0,'error':0,'other':0,'total':0,'instances':[]},
                 'wandering': {'pass':0,'fail':0,'error':0,'other':0,'total':0,'instances':[]},
                 'locked': {'pass':0,'fail':0,'error':0,'other':0,'total':0,'instances':[]}}

for iid, entry in p6b.items():
    if iid not in sub_class:
        continue
    cls = sub_class[iid]
    conditions = entry.get('conditions', {})
    agent_result = conditions.get('agent')
    verdict = classify_verdict(agent_result)
    class_results[cls]['total'] += 1
    class_results[cls]['instances'].append((iid, verdict, agent_result))
    if verdict == 'PASS':
        class_results[cls]['pass'] += 1
    elif verdict == 'FAIL':
        class_results[cls]['fail'] += 1
    elif verdict == 'ERROR':
        class_results[cls]['error'] += 1
    else:
        class_results[cls]['other'] += 1

print(f"\n=== Coverage of Phase 6b eval across sub-classes ===")
for cls in ['success', 'locked', 'wandering']:
    r = class_results[cls]
    print(f"  {cls.upper():>10}: {r['total']} evaluated (PASS={r['pass']}, FAIL={r['fail']}, ERROR={r['error']}, OTHER={r['other']})")

# Pass rates
print(f"\n=== Pass rates (agent condition) ===")
for cls in ['success', 'locked', 'wandering']:
    r = class_results[cls]
    if r['total'] > 0:
        rate = 100 * r['pass'] / r['total']
        print(f"  {cls.upper():>10}: {r['pass']}/{r['total']} = {rate:.1f}% pass rate")

# Sanity check: golden + none baselines (across all classes)
print(f"\n=== Sanity check: baseline conditions across all 62 instances ===")
counts = {'none':Counter(), 'golden':Counter(), 'agent':Counter()}
for iid, entry in p6b.items():
    for cond_name, cond_result in entry.get('conditions', {}).items():
        v = classify_verdict(cond_result)
        counts[cond_name][v] += 1
for cond in ['none', 'golden', 'agent']:
    total = sum(counts[cond].values())
    pass_pct = 100 * counts[cond]['PASS'] / max(1, total)
    print(f"  {cond:>7}: PASS={counts[cond]['PASS']}/{total} ({pass_pct:.0f}%), FAIL={counts[cond]['FAIL']}, ERROR={counts[cond]['ERROR']}, OTHER={counts[cond]['OTHER']}")

# Statistical comparison: WANDERING vs SUCCESS pass rate
print(f"\n=== STATISTICAL TEST: WANDERING vs SUCCESS pass rate (Fisher's exact) ===")
w = class_results['wandering']
s = class_results['success']
if w['total'] >= 2 and s['total'] >= 2:
    # Contingency table:
    #              PASS    FAIL+ERROR+OTHER
    # WANDERING    w_pass  w_total - w_pass
    # SUCCESS      s_pass  s_total - s_pass
    table = [[w['pass'], w['total'] - w['pass']],
             [s['pass'], s['total'] - s['pass']]]
    print(f"  Contingency table:")
    print(f"               PASS  NOT_PASS")
    print(f"    WANDERING   {table[0][0]:3d}    {table[0][1]:3d}")
    print(f"    SUCCESS     {table[1][0]:3d}    {table[1][1]:3d}")
    odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
    print(f"\n  Odds ratio: {odds_ratio:.3f}")
    print(f"  Fisher's exact p-value (two-sided): {p_value:.4f}")
    # Same-direction one-sided (WANDERING pass < SUCCESS pass)
    _, p_less = fisher_exact(table, alternative='less')
    print(f"  One-sided p (WANDERING pass < SUCCESS pass): {p_less:.4f}")

# Interpretation
print(f"\n=== INTERPRETATION ===")
w_rate = 100 * w['pass'] / w['total'] if w['total'] > 0 else 0
s_rate = 100 * s['pass'] / s['total'] if s['total'] > 0 else 0
ratio = (w['pass']/w['total']) / (s['pass']/s['total']) if (s['total']>0 and s['pass']>0 and w['total']>0) else float('nan')
print(f"  WANDERING agent-patch pass rate: {w_rate:.1f}% ({w['pass']}/{w['total']})")
print(f"  SUCCESS   agent-patch pass rate: {s_rate:.1f}% ({s['pass']}/{s['total']})")
print(f"  WANDERING/SUCCESS ratio: {ratio:.3f}")
print()
if w_rate >= 0.7 * s_rate:
    print(f"  >>> Pass rates COMPARABLE — WANDERING patches mostly correct.")
    print(f"      Causal interpretation: WANDERING is primarily ACTION-EMISSION failure.")
    print(f"      Mid-layer-to-edge alignment hypothesis SUPPORTED.")
elif w_rate < 0.3 * s_rate:
    print(f"  >>> WANDERING pass rate MUCH LOWER — WANDERING agents had bad patches AND failed to submit.")
    print(f"      Causal interpretation: WANDERING is dual failure (patch quality + emission).")
    print(f"      Mid-layer-to-edge alignment hypothesis NEEDS REVISION.")
else:
    print(f"  >>> INTERMEDIATE — mixed mechanism.")
    print(f"      Sub-class analysis (e.g., WANDERING-A verbal vs WANDERING-B silent) may help disambiguate.")

# Per-instance dump for paper
print(f"\n=== Per-instance WANDERING agent-patch verdicts ===")
for iid, verdict, result in sorted(class_results['wandering']['instances'], key=lambda x: x[1]):
    print(f"  {verdict:>5}: {iid[:50]:<50}  n_pass={result.get('n_pass','?')} n_fail={result.get('n_fail','?')} n_err={result.get('n_err','?')}")

# Save
out = {
    'class_results': {cls: {k: v if k != 'instances' else [(i,vd) for i,vd,_ in v]
                            for k,v in r.items()}
                       for cls, r in class_results.items()},
    'fisher_p_two_sided': float(p_value) if 'p_value' in dir() else None,
    'wandering_pass_rate': w_rate,
    'success_pass_rate': s_rate,
    'ratio': ratio if not np.isnan(ratio) else None,
    'n_eval_data': len(p6b),
}
(OUT_DIR / 'exp_d_forced_finish_counterfactual.json').write_text(json.dumps(out, indent=2, default=str))
print(f"\nSaved {OUT_DIR}/exp_d_forced_finish_counterfactual.json")
