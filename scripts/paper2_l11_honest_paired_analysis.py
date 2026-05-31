#!/usr/bin/env python3
"""
Paper #2 v2 — HONEST paired analysis of the L11 SUCCESS-donor experiment.

The notebook (nb_paper2_v2_l11_phase1) computed Fisher exact against the DEFINITIONAL
0/20 baseline (table = [[n_rescued, n_total-n_rescued],[0,20]]). That is the wrong
baseline: this paper itself documents that the SAME 20 WANDERING instances flip 7/20
(35%) under a no-hook re-run (run-instability under temperature=1.0). The correct test
is PAIRED (McNemar) over the same 20 instances: no-hook baseline vs hook.

This script recomputes the honest numbers from the raw per-run JSON on Drive:
  - paired McNemar (no-hook baseline vs hook alpha=0.70)
  - contamination check: of the "rescued" instances, how many ALREADY flip no-hook?
  - dose-dependent crash structure (the one robust L11-specific causal effect)
  - random-direction control (acknowledged selection-biased)

Run on the machine where the Drive mount is available:
  DRIVE="~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs"
"""
import json, os, re
from pathlib import Path
from scipy.stats import binomtest, fisher_exact

DRIVE = Path(os.path.expanduser(
    os.environ.get("DRIVE",
    "~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs")))
BASE = DRIVE / "swebench_exp_b_c/exp_b_baseline_nohook/results.json"
RUNS = DRIVE / "swebench_paper2_v2_l11_pilot/runs"

def fr(o):
    return o.get("finish_reason") if isinstance(o, dict) else None

def core(iid):
    m = re.search(r"(qutebrowser|ansible|openlibrary)-([0-9a-f]+)", iid)
    return m.group(0) if m else iid

# 1) Exp B no-hook baseline (the run-stable comparison condition)
base = json.load(open(BASE))
base_fr = {iid: fr(o) for iid, o in base.items()} if isinstance(base, dict) \
          else {r["iid"]: fr(r) for r in base}
base_core = {core(i): f for i, f in base_fr.items()}
base_flip = {c for c, f in base_core.items() if f == "finish_tool"}
print(f"Exp B no-hook baseline: {len(base_core)} iids, {len(base_flip)} flip finish_tool "
      f"({len(base_flip)/len(base_core):.0%})")

# 2) L11 hook runs
def load(glob):
    d = {}
    for p in RUNS.glob(glob):
        r = json.load(open(p)); d[core(r.get("iid", p.stem))] = r.get("finish_reason")
    return d
a070 = load("*__alpha_0.70.json")
a115 = load("*__alpha_1.15.json")

# 3) Contamination: of rescued-at-either, how many already flip no-hook?
all_l11 = set(a070) | set(a115)
clean, contaminated = [], []
for c in sorted(all_l11):
    resc = (a070.get(c) == "finish_tool") or (a115.get(c) == "finish_tool")
    if not resc:
        continue
    (contaminated if base_core.get(c) == "finish_tool" else clean).append(c)
print(f"\nRescued at either alpha: {len(clean)+len(contaminated)}/20  "
      f"-> CLEAN (no-hook=max_turns): {len(clean)}  |  "
      f"CONTAMINATED (no-hook already finish_tool): {len(contaminated)}")

# 4) PAIRED McNemar, no-hook baseline vs hook alpha=0.70 (the honest primary test)
both = sum(1 for c in a070 if base_core.get(c) == "finish_tool" and a070[c] == "finish_tool")
base_only = sum(1 for c in a070 if base_core.get(c) == "finish_tool" and a070[c] != "finish_tool")
hook_only = sum(1 for c in a070 if base_core.get(c) not in (None, "finish_tool") and a070[c] == "finish_tool")
disc = base_only + hook_only
p_mc = binomtest(min(base_only, hook_only), disc, 0.5).pvalue if disc else 1.0
print(f"\n=== PRIMARY (honest): paired McNemar, no-hook baseline vs hook alpha=0.70 ===")
print(f"  both flip={both}  baseline-only={base_only}  hook-only={hook_only}")
print(f"  McNemar exact p={p_mc:.3f}  "
      f"direction={'HOOK HELPS' if hook_only > base_only else 'HOOK HURTS/NULL'}")
print(f"  hook flip total = {both+hook_only}/20  vs  baseline flip total = {both+base_only}/20")

# 5) The (wrong) Fisher-vs-0/20 the notebook reported, for transparency
n_resc = len(clean) + len(contaminated)
_, p_wrong = fisher_exact([[n_resc, 20 - n_resc], [0, 20]])
print(f"\n  [for transparency] notebook's Fisher vs definitional 0/20: "
      f"{n_resc}/20 vs 0/20, p={p_wrong:.4f} (WRONG baseline — ignores run-instability)")

# 6) Dose-dependent crash — the one robust L11-specific causal effect
crash = lambda d: sum(1 for v in d.values() if v == "invalid_tools")
resc  = lambda d: sum(1 for v in d.values() if v == "finish_tool")
print(f"\n=== Dose-response (robust L11-specific effect) ===")
print(f"  alpha=0.70: rescue {resc(a070)}/20, crash {crash(a070)}/20")
print(f"  alpha=1.15: rescue {resc(a115)}/20, crash {crash(a115)}/20")
print(f"  -> 0/20 -> {crash(a115)}/20 invalid_tools: L11 injection reaches behavior "
      f"(L55 was inert), but produces INCOHERENCE, not completion.")

# 7) Random-direction control (selection-biased; report honestly)
rand = {}
for p in RUNS.glob("*__RANDOM_alpha_*.json"):
    r = json.load(open(p)); rand[(core(r.get("iid")), r.get("alpha"))] = r.get("finish_reason")
print(f"\n=== Random-direction control (SELECTION-BIASED: only on rescued subset) ===")
for (c, a), v in sorted(rand.items()):
    print(f"  {c:28s} alpha={a} RANDOM -> {v}")
print("  Note: at alpha=1.15 random ALSO crashes (invalid_tools) -> crash is "
      "direction-AGNOSTIC. At alpha=0.70 n is too small + biased to claim direction-specificity.")

print("\nVERDICT: NULL on the rescue axis (McNemar p=%.2f). L11 is causally LIVE "
      "(dose-dependent crash) but the SUCCESS direction is not a rescue lever, "
      "even at the locus paper #3 identifies as WANDERING's best discriminator." % p_mc)
