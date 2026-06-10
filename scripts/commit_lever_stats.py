#!/usr/bin/env python3
"""Per-contrast statistics for the commitment-lever Tier-1, from the HF results ledger.

The ledger stores per-block RATES (k/60), not per-point outcomes, so the exact paired McNemar
discordance table (b,c) is not directly available. Two valid analyses are reported:

  * Wilson 95% CI per block rate (exact from k/n).
  * Fisher exact (two-sided), treating the two runs as INDEPENDENT samples — a different,
    also-valid framing (not the paired design, but a legitimate conservative comparison).
  * **Worst-case exact McNemar (paired):** with b-c = k2-k1 fixed by the marginals, McNemar's
    statistic is minimized (least significant) at MAXIMUM discordance (b+c as large as the
    marginals allow). Reporting that worst case gives a rigorous UPPER BOUND on the paired p-value
    using only the marginals — no per-point data needed. The true paired p (with realistic overlap)
    is smaller. Run scripts/commit_lever_stages.py with per-point logging for the exact value.

Usage: python commit_lever_stats.py     (reads results/commit_lever_results.json from HF)
"""
import json, math
from math import comb, sqrt
from huggingface_hub import hf_hub_download

DATA_REPO = "caiovicentino1/swebench-phase6-verdict-circuit"

def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))

def fisher_2x2(a, b, c, d):
    n = a + b + c + d
    def lp(a, b, c, d):
        return (math.lgamma(a + b + 1) + math.lgamma(c + d + 1) + math.lgamma(a + c + 1)
                + math.lgamma(b + d + 1) - math.lgamma(a + 1) - math.lgamma(b + 1)
                - math.lgamma(c + 1) - math.lgamma(d + 1) - math.lgamma(n + 1))
    p0 = lp(a, b, c, d); tot = 0.0
    r1, c1, r2 = a + b, a + c, c + d
    for x in range(max(0, c1 - r2), min(r1, c1) + 1):
        pr = lp(x, r1 - x, c1 - x, r2 - (c1 - x))
        if pr <= p0 + 1e-9: tot += math.exp(pr)
    return min(1.0, tot)

def mcnemar_exact(base_pp, treat_pp):
    """Exact two-sided paired McNemar from per-point 0/1 vectors (same points, same order)."""
    b = sum(1 for x, y in zip(base_pp, treat_pp) if x == 0 and y == 1)
    c = sum(1 for x, y in zip(base_pp, treat_pp) if x == 1 and y == 0)
    bc = b + c
    if bc == 0: return 1.0, (b, c)
    lo = min(b, c)
    p = 2 * sum(comb(bc, i) for i in range(0, lo + 1)) / (2 ** bc)
    return min(1.0, p), (b, c)

def mcnemar_worstcase(k1, k2, n):
    """Upper bound on the paired two-sided exact McNemar p, from marginals only."""
    delta = k2 - k1
    if delta >= 0:
        b = min(k2, n - k1); c = b - delta
    else:
        c = min(k1, n - k2); b = c + delta
    bc = b + c; lo = min(b, c)
    if bc == 0: return 1.0, (b, c)
    p = 2 * sum(comb(bc, i) for i in range(0, lo + 1)) / (2 ** bc)
    return min(1.0, p), (b, c)

CONTRASTS = [
    ("H1 edit-donor@L59 vs baseline", "h1_editdonor_L59", "h1_baseline"),
    ("H1 edit-donor@L55 vs baseline", "h1_editdonor_L55", "h1_baseline"),
    ("H1 edit-donor@L59 vs bash-null@L59 (position)", "h1_editdonor_L59", "h1_bashnull_L59"),
    ("H1 edit-donor@L59 vs cross-task@L59 (specificity)", "h1_editdonor_L59", "h1_crosstask_L59"),
    ("H2 brake suppress@L55 vs baseline", "h2_suppress_L55", "h2_baseline"),
    ("H2 suppress@L59 vs baseline", "h2_suppress_L59", "h2_baseline"),
    ("H2 suppress@L55 vs edit-donor ctl@L55", "h2_suppress_L55", "h2_ctl_L55"),
]

def main():
    d = json.load(open(hf_hub_download(DATA_REPO, "results/commit_lever_results.json",
                                       repo_type="dataset", force_download=True)))
    B = d["blocks"]
    def kn(name): return round(B[name]["rate"] * B[name]["n"]), B[name]["n"]
    print("Wilson 95% CI per block:")
    for nm in B:
        k, n = kn(nm); lo, hi = wilson(k, n)
        print(f"  {k:>2}/{n} ={k/n:.3f}  [{lo:.3f},{hi:.3f}]  {nm}")
    have_pp = all(B[nm].get("per_point") for nm in B)
    print(f"\nContrasts ({'EXACT paired McNemar' if have_pp else 'worst-case paired McNemar'}):")
    rows = []
    for label, t, b in CONTRASTS:
        k2, n = kn(t); k1, _ = kn(b)
        fp = fisher_2x2(k2, n - k2, k1, n - k1)
        if have_pp:
            mp, (bb, cc) = mcnemar_exact(B[b]["per_point"], B[t]["per_point"]); kind = "exact"
        else:
            mp, (bb, cc) = mcnemar_worstcase(k1, k2, n); kind = "<="
        rows.append((label, (k2 - k1) / n, fp, mp))
        print(f"  {label:50} RD={(k2-k1)/n:+.2f}  Fisher={fp:.1e}  McNemar{kind} {mp:.1e}  (b={bb},c={cc})")
    print(f"\nHolm-Bonferroni on paired McNemar (m={len(CONTRASTS)}):")
    ps = sorted((mp, l) for l, _, _, mp in rows)
    for i, (p, l) in enumerate(ps):
        thr = 0.05 / (len(ps) - i)
        print(f"  {'PASS' if p < thr else 'FAIL'}  p<={p:.1e}  thr={thr:.1e}  {l}")

if __name__ == "__main__":
    main()
