"""
Which WANDERING trajectories does the completion-language monitor MISS?

The monitor (m_complete_last5 >= 0.4 AND not emit_finish AND patch > 0) catches
7/20 WANDERING (35%). We need to know what the OTHER 13 look like — that
characterizes the residual blind spot after adding the monitor.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

OUT_DIR = Path("/Volumes/SSD Major/fish/openinterp-swebench-harness/scripts/inflection_turn_out")
PHASE6 = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/swebench_v6_phase6")

data = json.load(open(OUT_DIR / "complementary_monitor.json"))
signals = data["per_trajectory_signals"]

# Get probe scores
infl = json.load(open(OUT_DIR / "inflection_results.json"))
iid_to_probe_final = {t["iid"]: t.get("final_score", 0.5) for t in infl["per_trajectory"]}


def strict_monitor(s):
    return (s["m_complete_last5"] >= 0.4 and not s["emit_finish"] and s["patch_n_bytes"] > 0)


def lax_monitor(s):
    return (s["m_complete_last5"] >= 0.2 and not s["emit_finish"])


wander = [s for s in signals if s["sub_class"] == "wandering"]
print(f"WANDERING n={len(wander)}\n")

print("=== Caught by STRICT monitor (m_complete_last5>=0.4 AND no_finish AND patch>0) ===")
caught_strict = [s for s in wander if strict_monitor(s)]
for s in caught_strict:
    print(f"  {s['iid'][:40]:<40} m_complete={s['m_complete_last5']:.2f}, patch={s['patch_n_bytes']}B, turns={s['n_turns_actual']}")

print(f"\nN caught strict: {len(caught_strict)}/{len(wander)} ({100*len(caught_strict)/len(wander):.0f}%)\n")

print("=== Caught by LAX (m_complete>=0.2 AND no_finish) ===")
caught_lax = [s for s in wander if lax_monitor(s)]
print(f"N caught lax: {len(caught_lax)}/{len(wander)}\n")

print("=== MISSED by both monitors ===")
missed = [s for s in wander if not lax_monitor(s)]
for s in missed:
    print(f"  {s['iid'][:40]:<40} m_complete={s['m_complete_last5']:.2f}, m_repeat={s['m_repeat_last5']}, m_static={s['m_static_end']}, patch={s['patch_n_bytes']}B, turns={s['n_turns_actual']}, final_probe={iid_to_probe_final.get(s['iid'], 0.5):.2f}")

print(f"\nN missed: {len(missed)}/{len(wander)} ({100*len(missed)/len(wander):.0f}%)")

# Pattern in missed
print("\n=== Patterns in MISSED WANDERING ===")
print(f"  Mean m_complete_last5: {sum(s['m_complete_last5'] for s in missed)/len(missed):.3f}")
print(f"  Mean m_static_end: {sum(s['m_static_end'] for s in missed)/len(missed):.1f}")
print(f"  Mean m_repeat: {sum(s['m_repeat_last5'] for s in missed)/len(missed):.1f}")
print(f"  Median patch bytes: {sorted([s['patch_n_bytes'] for s in missed])[len(missed)//2]}")
print(f"  Mean final probe: {sum(iid_to_probe_final.get(s['iid'], 0.5) for s in missed)/len(missed):.3f}")

# Save list of MISSED for trace inspection
miss_iids = [s["iid"] for s in missed]
(OUT_DIR / "monitor_missed_wandering_iids.json").write_text(json.dumps(miss_iids, indent=2))
print(f"\nSaved missed IIDs to {OUT_DIR}/monitor_missed_wandering_iids.json")
