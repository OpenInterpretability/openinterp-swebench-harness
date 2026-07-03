"""EVAL-before-mint — recompute EVERY number claimed in RESULTS_hybridization_audit.md
from the HF ledgers (never from the doc). CPU-only. House discipline (cf. eval_safety_brake.py).

Usage: python3 eval_hydra_audit.py   -> prints PASS/FAIL per claim + summary.
"""
import json, random
import numpy as np
from math import comb
from huggingface_hub import hf_hub_download

REPO = "caiovicentino1/swebench-phase6-verdict-circuit"
CHECKS = []

def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def close(a, b, tol=5e-4): return abs(a - b) <= tol

def mcnemar(a, b):
    l = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    g = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = l + g
    p = 1.0 if n == 0 else min(1.0, sum(comb(n, k) for k in range(0, min(l, g) + 1)) / 2 ** n * 2)
    return l, g, p

A = json.load(open(hf_hub_download(REPO, "results/hydra_audit_ie.json", repo_type="dataset", force_download=True)))
P7 = json.load(open(hf_hub_download(REPO, "results/commit_lever_results.json", repo_type="dataset", force_download=True)))

# ---------------------------------------------------------------- Stage 1: ranking (recomputed from raw IE)
FA = list(range(3, 64, 4)); IE_MIN = 0.01
rows = []
for L in FA:
    res = A[f"ie_L{L}"]
    for h in range(24):
        recs = res[str(h)]
        by = {}
        for r in recs: by.setdefault(r["probe"], []).append(r["ie"])
        ie = float(np.mean([r["ie"] for r in recs]))
        kap = float(np.mean([1.0 if max(v) >= IE_MIN else 0.0 for v in by.values()]))
        rows.append({"L": L, "h": h, "ie": ie, "kappa": kap, "s": ie * kap,
                     "ha": float(np.mean([r["ie"] for r in recs[0::2]])),
                     "hb": float(np.mean([r["ie"] for r in recs[1::2]]))})
rows.sort(key=lambda r: -r["s"])
for i, r in enumerate(rows): r["rank"] = i + 1
def head(L, h): return next(r for r in rows if r["L"] == L and r["h"] == h)

check("claim: 25/384 heads with s>0", len([r for r in rows if r["s"] > 0]) == 25)
check("claim: 13 heads with s>=0.01", len([r for r in rows if r["s"] >= 0.01]) == 13)
top5 = [(f"L{r['L']}h{r['h']}", round(r["s"], 3)) for r in rows[:5]]
check("claim: top-5 = L63h14 .113 / L59h21 .048 / L55h6 .044 / L59h19 .044 / L51h20 .036",
      top5 == [("L63h14", 0.113), ("L59h21", 0.048), ("L55h6", 0.044), ("L59h19", 0.044), ("L51h20", 0.036)], str(top5))
w8, w6, w3 = head(59, 8), head(59, 6), head(59, 3)
check("claim: h8 kappa=0, rank 358, ie~0.0005", w8["kappa"] == 0 and w8["rank"] == 358 and close(w8["ie"], 0.0005, 1e-4))
check("claim: h6 kappa=0, rank 356, ie~0.0006", w6["kappa"] == 0 and w6["rank"] == 356 and close(w6["ie"], 0.0006, 1e-4))
check("claim: h3 s=0.0098, rank 14", close(w3["s"], 0.0098, 1e-4) and w3["rank"] == 14)
o = {h: head(59, h) for h in (16, 19, 11, 4, 21, 23)}
check("claim: h21 s=0.0480 rank 2", close(o[21]["s"], 0.0480, 1e-4) and o[21]["rank"] == 2)
check("claim: h19 s=0.0439 rank 4", close(o[19]["s"], 0.0439, 1e-4) and o[19]["rank"] == 4)
check("claim: zero-signal opposers ranks 355-371",
      all(355 <= o[h]["rank"] <= 371 and o[h]["s"] == 0 for h in (16, 11, 4, 23)),
      str({h: o[h]["rank"] for h in (16, 11, 4, 23)}))
for name, K in (("K25", 96), ("K125", 48)):
    keep = {(r["L"], r["h"]) for r in rows[:K]}
    check(f"claim: {name} writers kept == [3]", [h for h in (8, 6, 3) if (59, h) in keep] == [3])
    check(f"claim: {name} opposers kept == [19,21]", sorted(h for h in (16, 19, 11, 4, 21, 23) if (59, h) in keep) == [19, 21])

# G1
a = np.argsort(np.argsort([r["ha"] for r in rows])); b = np.argsort(np.argsort([r["hb"] for r in rows]))
rho = float(np.corrcoef(a, b)[0, 1])
check("claim: split-half rho ~0.457 (<0.8, G1 declared FAIL)", close(rho, 0.457, 2e-3) and rho < 0.8, f"{rho:.3f}")
ta = {(r["L"], r["h"]) for r in sorted(rows, key=lambda r: -r["ha"])[:20]}
tb = {(r["L"], r["h"]) for r in sorted(rows, key=lambda r: -r["hb"])[:20]}
check("claim: top-20 half overlap = 19/20", len(ta & tb) == 19)
check("claim: h21 halves .0478/.0482", close(o[21]["ha"], 0.0478, 1e-4) and close(o[21]["hb"], 0.0482, 1e-4))
check("claim: h8 halves .0003/.0008", close(w8["ha"], 0.0003, 1e-4) and close(w8["hb"], 0.0008, 1e-4))

# bases
bs = A["bases"]
check("claim: 12/12 usable pairs", len(A["pairs_ok"]) == 12)
mc_ = [bs[k]["m_clean"] for k in bs]; mx = [bs[k]["m_corr"] for k in bs]
check("claim: m_clean in +6.6..+11.5, m_corr in -2.2..-1.2",
      min(mc_) >= 6.6 and max(mc_) <= 11.5 and min(mx) >= -2.2 and max(mx) <= -1.2,
      f"clean [{min(mc_):.2f},{max(mc_):.2f}] corr [{min(mx):.2f},{max(mx):.2f}]")

# G0 (from the #7 ledger)
fid = P7["fidelity"]
check("claim: fidelity 0.474/0.299", close(fid["edit_pts_baseP_edit"], 0.4736, 1e-3) and close(fid["bash_pts_baseP_edit"], 0.2994, 1e-3))
check("claim: published baselines 0.233/0.483",
      close(P7["blocks"]["h1_baseline"]["rate"], 0.2333, 1e-3) and close(P7["blocks"]["h2_baseline"]["rate"], 0.4833, 1e-3))
check("claim: G0 live first-15 == ledger prefix 0.133",
      close(sum(P7["blocks"]["h1_baseline"]["per_point"][:15]) / 15, 0.1333, 1e-3))

# ---------------------------------------------------------------- H3 conditions
base = A["h3_base_mean"]
check("claim: base NIAH 1.00, edit 0.483, bash 0.233",
      base["niah_pass_rate"] == 1.0 and close(base["edit_emit"], 0.4833, 1e-3) and close(base["bash_emit"], 0.2333, 1e-3))
exp = {  # cond: (edit_emit, bash_emit, mcn_edit(l,g,p), mcn_bash(l,g,p) or None)
    "drop_nonret_L59": (0.4667, 0.2333, (1, 0, 1.0), None),
    "drop_rand_L59": (0.4667, 0.2333, (1, 0, 1.0), None),
    "drop_ret_L59": (0.4667, 0.2333, (1, 0, 1.0), None),
    "drop_writers": (0.45, 0.2333, (2, 0, 0.5), None),
    "drop_opposers": (0.4833, 0.2333, (0, 0, 1.0), None),
    "drop_nonret_late": (0.1833, 0.1833, (18, 0, 7.63e-6), (3, 0, 0.25)),
    "drop_rand_late": (0.3333, 0.10, (9, 0, 3.91e-3), (9, 1, 0.0215)),
    "keep_writers_late": (0.4833, 0.30, (0, 0, 1.0), (0, 4, 0.125)),
}
for c, (ee, be, me, mb) in exp.items():
    d = A[f"h3_{c}_mean"]
    ok = d["niah_pass_rate"] == 1.0 and close(d["edit_emit"], ee, 1e-3) and close(d["bash_emit"], be, 1e-3)
    l, g, p = mcnemar(base["edit_emit_per"], d["edit_emit_per"])
    ok = ok and (l, g) == me[:2] and close(p, me[2], me[2] * 0.02 + 1e-9)
    det = f"edit {d['edit_emit']:.3f} mcn {l}/{g} p={p:.2e}"
    if mb:
        lb, gb, pb = mcnemar(base["bash_emit_per"], d["bash_emit_per"])
        ok = ok and (lb, gb) == mb[:2] and close(pb, mb[2], mb[2] * 0.02 + 1e-9)
        det += f" | bash {lb}/{gb} p={pb:.2e}"
    check(f"claim: {c} numbers", ok, det)

# gap collapse
check("claim: gap 0.25 -> 0.00 under drop_nonret_late",
      close(base["edit_emit"] - base["bash_emit"], 0.25, 1e-3) and
      close(A["h3_drop_nonret_late_mean"]["edit_emit"] - A["h3_drop_nonret_late_mean"]["bash_emit"], 0.0, 1e-3))
check("claim: NIAH 1.00 in ALL 9 conditions",
      all(A[f"h3_{c}_mean"]["niah_pass_rate"] == 1.0 for c in list(exp) + ["base"]))

# headset semantics (reproduced from ranking + seeds)
keep25 = {(r["L"], r["h"]) for r in rows[:96]}
late = [(L, h) for L in (51, 55, 59, 63) for h in range(24)]
nonret_late = [x for x in late if x not in keep25]
check("claim: drop_nonret_late = 74 heads (96 late - 22 retained)", len(nonret_late) == 74, str(A["h3_drop_nonret_late_mean"]["n_heads"]))
check("claim: ledger n_heads match (74/74/72/20)",
      A["h3_drop_nonret_late_mean"]["n_heads"] == 74 and A["h3_drop_rand_late_mean"]["n_heads"] == 74
      and A["h3_keep_writers_late_mean"]["n_heads"] == 72 and A["h3_drop_nonret_L59_mean"]["n_heads"] == 20)
kw = [x for x in nonret_late if x not in {(59, 8), (59, 6), (59, 3)}]
check("claim: keep_writers set = drop_nonret_late minus {h8,h6} (h3 was never dropped)",
      len(kw) == 72 and set(nonret_late) - set(kw) == {(59, 8), (59, 6)})
rng = random.Random(42)
rs = set(rng.sample(late, len(nonret_late)))  # reproduces the actual run draw (74)
rand_ok = all(x in rs for x in [(59, 8), (59, 6), (59, 21), (59, 19), (63, 14)]) and (59, 3) not in rs
check("claim: random-74 (seed 42) drops h8,h6,h21,h19,L63h14 and KEEPS h3 (same as selection)", rand_ok)
check("claim: both arms keep 22 late heads (same budget), both incl. h3; only the other 21 differ",
      len([x for x in late if x not in rs]) == 22 and len([x for x in late if x in keep25]) == 22)

# expansion
st = A["stability_v2"]
for k, v in (("L59h8", 0.0007), ("L59h6", 0.0008), ("L59h3", 0.0105), ("L59h21", 0.0484), ("L59h19", 0.0471)):
    check(f"claim: expansion combined {k}={v}", close(st[k]["combined"], v, 1e-4) and st[k]["n"] == 24,
          f"{st[k]['combined']:.4f} n={st[k]['n']}")
check("claim: expansion 12/12 usable v2 pairs", len(A["pairs_ok_v2"]) == 12)

# red-team additions: direct paired test + NIAH margin insensitivity (B2)
l, g, p = mcnemar(A["h3_drop_rand_late_mean"]["edit_emit_per"], A["h3_drop_nonret_late_mean"]["edit_emit_per"])
check("claim: direct paired random->capability = 10/1, p=0.012", (l, g) == (10, 1) and close(p, 0.012, 5e-4), f"{l}/{g} p={p:.3f}")
mg = lambda c: float(np.mean([b["m"] for b in A[f"h3_{c}_mean"]["niah"]]))
base_m = float(np.mean([bs[k]["m_clean"] for k in A["pairs_ok"]]))
check("claim: NIAH margin RISES under drop_nonret_late (10.56 vs 9.39 base)", close(mg("drop_nonret_late"), 10.56, 5e-3) and close(base_m, 9.39, 5e-3))
check("claim: positive control barely moves (drop_ret_L59 margin 9.29)", close(mg("drop_ret_L59"), 9.29, 5e-3))

# Stage-4 controls (red-team M1+M3)
rand_conds = [("drop_rand_late", 0.3333), ("drop_rand_late_s2", 0.3833), ("drop_rand_late_s3", 0.25),
              ("drop_rand_late_s4", 0.2167), ("drop_rand_late_s5", 0.45)]
rr = []
for c, expct in rand_conds:
    d = A[f"h3_{c}_mean"]
    check(f"claim: {c} edit-emit {expct}", close(d["edit_emit"], expct, 1e-3) and d["n_heads"] == 74)
    rr.append(d["edit_emit"])
check("claim: capability (0.183) below ALL 5 random draws (mean 0.327, min 0.217)",
      all(r > 0.1834 for r in rr) and close(float(np.mean(rr)), 0.3267, 1e-3) and close(min(rr), 0.2167, 1e-3))
pairsig = []
for c, _ in rand_conds:
    l, g, p = mcnemar(A[f"h3_{c}_mean"]["edit_emit_per"], A["h3_drop_nonret_late_mean"]["edit_emit_per"])
    pairsig.append(p < 0.05)
check("claim: paired draw->capability significant for 3/5 draws", sum(pairsig) == 3, str(pairsig))
k2 = [(1, 0.20), (2, 0.1667), (3, 0.15)]
for i, expct in k2:
    d = A[f"h3_keep2rand_late_s{i}_mean"]
    check(f"claim: keep2rand_s{i} stays collapsed ({expct})", close(d["edit_emit"], expct, 1e-3) and d["n_heads"] == 72)
check("claim: keep2rand mean 0.172 vs keep_writers 0.483",
      close(float(np.mean([A[f"h3_keep2rand_late_s{i}_mean"]["edit_emit"] for i in (1, 2, 3)])), 0.1722, 1e-3)
      and close(A["h3_keep_writers_late_mean"]["edit_emit"], 0.4833, 1e-3))
l, g, p = mcnemar(A["h3_keep_writers_late_mean"]["edit_emit_per"], A["h3_keep2rand_late_s1_mean"]["edit_emit_per"])
check("claim: paired keep_writers->keep2rand_s1 = 17/0, p=1.5e-5", (l, g) == (17, 0) and close(p, 1.5e-5, 2e-6), f"{l}/{g} p={p:.1e}")

n_pass = sum(1 for _, ok in CHECKS if ok)
print(f"\n=== EVAL SUMMARY: {n_pass}/{len(CHECKS)} PASS ===")
if n_pass < len(CHECKS):
    print("FAILURES:", [n for n, ok in CHECKS if not ok])
