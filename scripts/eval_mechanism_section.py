#!/usr/bin/env python3
"""Adversarial eval of section_mechanism.tex: recompute EVERY number from the HF ledgers and flag mismatches.
Reads the four raw ledgers (source of truth) and cross-checks each claim in the LaTeX section.
"""
import json
import numpy as np
from huggingface_hub import hf_hub_download
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

REPO = "caiovicentino1/swebench-phase6-verdict-circuit"
def L(name): return json.load(open(hf_hub_download(REPO, f"results/{name}", repo_type="dataset", force_download=True)))

dec = L("commit_lever_decomp.json"); hds = L("commit_lever_heads.json")
att = L("commit_lever_attn.json"); ko = L("commit_lever_knockout.json")

PASS, FAIL = [], []
def chk(label, claim, actual, tol=0.01):
    ok = abs(claim - actual) <= tol
    (PASS if ok else FAIL).append((label, claim, round(actual, 4), ok))
    print(f"  [{'OK ' if ok else 'XXX'}] {label:42s} claim={claim:>8} ledger={actual:+.4f}")

dp = dec["dp"]; em = dec["emit"]
print("== GATES ==")
chk("recon relerr L55 <1e-2", 0.0025, dec["recon_relerr"]["55"], 0.0003)
chk("recon relerr L59 <1e-2", 0.0024, dec["recon_relerr"]["59"], 0.0003)
chk("headsplit relerr <1e-2", 0.0015, hds["headsplit_relerr"], 0.0003)
chk("full reproduces elicit emit L59", 0.77, em["emit_elicit_L59_full"]["rate"])
chk("full reproduces brake emit L55", 0.03, em["emit_brake_L55_full"]["rate"])

print("== DECOMP TABLE (recovery fractions) ==")
def rf(direction, Lr, comp):
    full = dp[f"dp_{direction}_L{Lr}_full"]["mean"]; v = dp[f"dp_{direction}_L{Lr}_{comp}"]["mean"]
    return v / full
chk("elicit L55 full", 0.319, dp["dp_elicit_L55_full"]["mean"])
chk("elicit L55 r_inp", 0.52, rf("elicit", 55, "inp"), 0.02)
chk("elicit L55 r_attn", 0.05, rf("elicit", 55, "attn"), 0.02)
chk("elicit L55 r_mlp", -0.09, rf("elicit", 55, "mlp"), 0.02)
chk("elicit L59 full", 0.518, dp["dp_elicit_L59_full"]["mean"])
chk("elicit L59 r_inp", 0.22, rf("elicit", 59, "inp"), 0.02)
chk("elicit L59 r_attn", 0.43, rf("elicit", 59, "attn"), 0.02)
chk("elicit L59 r_mlp", -0.04, rf("elicit", 59, "mlp"), 0.02)
chk("brake L55 full", -0.404, dp["dp_brake_L55_full"]["mean"])
chk("brake L55 r_inp", 0.40, rf("brake", 55, "inp"), 0.02)
chk("brake L55 r_attn", 0.02, rf("brake", 55, "attn"), 0.02)
chk("brake L55 r_mlp", 0.03, rf("brake", 55, "mlp"), 0.02)
chk("brake L59 full", -0.382, dp["dp_brake_L59_full"]["mean"])
chk("brake L59 r_inp", 0.35, rf("brake", 59, "inp"), 0.02)
chk("brake L59 r_attn", 0.00, rf("brake", 59, "attn"), 0.02)
chk("brake L59 r_mlp", -0.02, rf("brake", 59, "mlp"), 0.02)

print("== DECOMP details ==")
chk("elicit L59 attn meanDP", 0.223, dp["dp_elicit_L59_attn"]["mean"])
chk("elicit L59 attn_rand meanDP", 0.100, dp["dp_elicit_L59_attn_rand"]["mean"])
chk("direction-specific ratio 2.2x", 2.2, dp["dp_elicit_L59_attn"]["mean"]/dp["dp_elicit_L59_attn_rand"]["mean"], 0.2)
if wilcoxon:
    a = np.array(dp["dp_elicit_L59_attn"]["per"]); m = np.array(dp["dp_elicit_L59_mlp"]["per"])
    pv = wilcoxon(a, m).pvalue; print(f"  [{'OK ' if pv<1e-6 else 'XXX'}] Wilcoxon attn vs mlp L59          claim~1.8e-8 ledger={pv:.2e}")
    (PASS if pv < 1e-6 else FAIL).append(("wilcoxon", "1.8e-8", pv, pv < 1e-6))
chk("emit elicit attn-only", 0.45, em["emit_elicit_L59_attn"]["rate"])
chk("emit elicit mlp-only", 0.25, em["emit_elicit_L59_mlp"]["rate"])
chk("emit elicit baseline", 0.23, em["emit_elicit_baseline"]["rate"])
# interaction gaps
for d, Lr, want in (("elicit", 59, 0.20), ("brake", 55, -0.23), ("brake", 59, -0.26)):
    full = dp[f"dp_{d}_L{Lr}_full"]["mean"]; gap = full - sum(dp[f"dp_{d}_L{Lr}_{c}"]["mean"] for c in ("inp", "attn", "mlp"))
    chk(f"interaction gap {d} L{Lr}", want, gap, 0.02)

print("== HEADS ==")
chk("n_heads 24", 24, hds["n_heads"], 0); chk("head_dim 256", 256, hds["head_dim"], 0)
chk("head 8 dP", 0.122, hds["dp_head"]["8"]["mean"])
chk("head 6 dP", 0.119, hds["dp_head"]["6"]["mean"])
chk("head 3 dP", 0.083, hds["dp_head"]["3"]["mean"])
chk("cum top1 (head8)", 0.121, hds["dp_cum"]["top1"]["mean"])
chk("cum top3", 0.262, hds["dp_cum"]["top3"]["mean"])
chk("cum all-24", 0.224, hds["dp_cum"]["all"]["mean"])
chk("cum cross-repo ctl", 0.077, hds["dp_cum"]["topctl"]["mean"])
chk("emit heads top3", 0.42, hds["emit"]["top3"]["rate"])
chk("emit heads all", 0.47, hds["emit"]["all"]["rate"])
g21 = [r for r in hds["geom"]["per_head"] if r["head"] == 21][0]
chk("head21 geom proj 14.0", 14.0, g21["proj"], 0.2)
chk("head21 causal dP (opponent)", -0.014, hds["dp_head"]["21"]["mean"])
bk = [hds["dp_head_brake"][str(h)]["mean"] for h in range(24)]
chk("brake heads min", -0.005, min(bk), 0.003); chk("brake heads max", 0.012, max(bk), 0.003)
# the 89% claim
chk("top3/all-heads emit ratio (=89%)", 0.89, hds["emit"]["top3"]["rate"]/hds["emit"]["all"]["rate"], 0.02)
print(f"  NOTE: top3 emit 0.42 / attn-channel emit 0.45 = {0.42/0.45:.2f}; / all-heads 0.47 = {0.42/0.47:.2f}")

print("== ATTENTION ==")
for h in (8, 6, 3):
    rc = att["summary"]["edit"][str(h)]["rec128"]; print(f"  head {h} rec128 (edit) = {rc:.3f}")
chk("commit-head rec128 in [0.10,0.16]", 0.13, np.mean([att["summary"]["edit"][str(h)]["rec128"] for h in (8, 6, 3)]), 0.05)

print("== KNOCKOUT ==")
R = ko["rows"]; idx = list(R)
def kpair(name, key, basekey):
    sub = [i for i in idx if R[i].get(name)]
    return np.mean([R[i][name][key] for i in sub]) - np.mean([R[i][basekey] for i in sub])
chk("knockout recon FAILED (gate>1, justifies pivot)", 1.68, ko["recon"][0], 0.2)
chk("ko_bash dP(bash)", -0.071, kpair("ko_bash", "pb", "base_b"))
chk("ko_rand dP(edit)~0", -0.001, kpair("ko_rand", "pe", "base_e"), 0.01)
chk("ko_edit dP(edit) ceiling", -0.031, kpair("ko_edit", "pe", "base_e"), 0.015)
chk("ko_bash -18% relative", -0.18, kpair("ko_bash", "pb", "base_b")/0.398, 0.03)

print("\n" + "=" * 60)
print(f"PASS {len(PASS)} / {len(PASS)+len(FAIL)}")
if FAIL:
    print("MISMATCHES:")
    for lab, c, a, _ in FAIL: print(f"  - {lab}: claim {c} vs ledger {a}")
else:
    print("ALL CLAIMS VERIFIED AGAINST RELEASED LEDGERS.")
