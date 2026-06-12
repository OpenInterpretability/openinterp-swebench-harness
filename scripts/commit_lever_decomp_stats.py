#!/usr/bin/env python3
"""Local CPU analysis of the attn-vs-MLP decomposition. Applies the pre-registered decision rule.

Reads results/commit_lever_decomp.json from HF and reports, per direction (elicit/brake) and layer:
  * recovery fraction r_comp = mean(ΔP_comp)/mean(ΔP_full), with paired bootstrap 95% CI
  * Wilcoxon signed-rank (paired) attn vs mlp per-point ΔP
  * additive interaction gap = mean(ΔP_full) - [mean(ΔP_inp)+mean(ΔP_attn)+mean(ΔP_mlp)]
  * direction-specificity: attn vs attn_rand
  * verdict per the prereg rule (attn-localized / mlp-localized / distributed / split)

Usage: python commit_lever_decomp_stats.py
"""
import json
import numpy as np
from huggingface_hub import hf_hub_download
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

DREPO = "caiovicentino1/swebench-phase6-verdict-circuit"

def boot_ratio(num, den, n=5000):
    num, den = np.asarray(num), np.asarray(den); k = len(num)
    rng = np.random.default_rng(0); rs = []
    md = den.mean()
    if abs(md) < 1e-9: return (float("nan"), float("nan"))
    for _ in range(n):
        idx = rng.integers(0, k, k)
        d = den[idx].mean()
        rs.append(num[idx].mean() / d if abs(d) > 1e-9 else np.nan)
    rs = np.array([x for x in rs if np.isfinite(x)])
    return (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))

def main():
    p = hf_hub_download(DREPO, "results/commit_lever_decomp.json", repo_type="dataset", force_download=True)
    D = json.load(open(p))
    dp = D.get("dp", {}); emit = D.get("emit", {})
    print("=== GATE: residual reconstruction relerr (want <1e-2) ===")
    print(" ", D.get("recon_relerr"))
    for direction in ("elicit", "brake"):
        for L in (55, 59):
            base = f"dp_{direction}_L{L}_"
            if base + "full" not in dp: continue
            full = np.array(dp[base + "full"]["per"])
            print(f"\n=== {direction.upper()} L{L} ===  mean ΔP_full = {full.mean():+.3f}")
            comps = {}
            for comp in ("inp", "attn", "mlp", "attnmlp"):
                if base + comp not in dp: continue
                v = np.array(dp[base + comp]["per"]); comps[comp] = v
                lo, hi = boot_ratio(v, full)
                print(f"  {comp:8s} meanΔP {v.mean():+.3f}   recovery r={v.mean()/full.mean():+.2f}  [{lo:+.2f},{hi:+.2f}]")
            if base + "attn_rand" in dp:
                vr = np.array(dp[base + "attn_rand"]["per"])
                print(f"  attn_rand meanΔP {vr.mean():+.3f}  (direction control — want |≪ attn|)")
            if "attn" in comps and "mlp" in comps and wilcoxon is not None:
                try:
                    st, pv = wilcoxon(comps["attn"], comps["mlp"])
                    print(f"  Wilcoxon attn vs mlp (paired): p={pv:.2e}")
                except Exception as e:
                    print("  Wilcoxon n/a:", e)
            if all(k in comps for k in ("inp", "attn", "mlp")):
                gap = full.mean() - (comps["inp"].mean() + comps["attn"].mean() + comps["mlp"].mean())
                print(f"  interaction gap (nonlinearity) = {gap:+.3f}")
            # verdict
            if "attn" in comps and "mlp" in comps:
                ra, rm = comps["attn"].mean() / full.mean(), comps["mlp"].mean() / full.mean()
                if ra >= 0.5 and rm < 0.25:   v = "ATTENTION-localized"
                elif rm >= 0.5 and ra < 0.25: v = "MLP-localized"
                elif ra >= 0.5 and rm >= 0.5: v = "DISTRIBUTED / redundant"
                else:                          v = "SPLIT / additive (check inp + interaction)"
                print(f"  >>> VERDICT: {v}  (r_attn={ra:+.2f}, r_mlp={rm:+.2f})")
    if emit:
        print("\n=== emit-rate (behavioral confirm) ===")
        for k in sorted(emit): print(f"  {k:28s} {emit[k]['rate']:.2f}")

if __name__ == "__main__":
    main()
