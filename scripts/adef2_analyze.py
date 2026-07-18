#!/usr/bin/env python3
"""Analyze the adaptive_defense_v2 ledger: per-defense verdicts + CIs + dose-response
+ stress-test (gradient-masking) flags + cross-domain, and an honest headline read.
Runnable anytime (shows partial progress) or at completion. Usage: python scripts/adef2_analyze.py
"""
import json, os, sys
from huggingface_hub import hf_hub_download
tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
REPO = "caiovicentino1/swebench-phase6-verdict-circuit"
DEFS = ["D0", "D1", "D2", "D3", "D4", "D5", "D6"]
NAME = {"D0": "single-dir", "D1": "ensemble(real)", "D2": "randomized(EOT)", "D3": "adaptive-steer",
        "D4": "multi-layer", "D5": "noise(EOT)", "D6": "detect-then-brake(adaptive-atk)"}
d = json.load(open(hf_hub_download(REPO, "results/adaptive_defense_v2.json", repo_type="dataset", token=tok, force_download=True)))
meta = d.get("meta", {})
print(f"=== adaptive_defense v2 — {meta.get('model')} ===")
print(f"config: N_TEST={meta.get('N_TEST')} N_TRAIN={meta.get('N_TRAIN')} STEPS={meta.get('STEPS')} "
      f"EOT_K={meta.get('EOT_K')} EPS={meta.get('EPS')}")
print(f"fixes: {meta.get('fixes')}\n")

def ci(x): return f"[{x[0]:.2f},{x[1]:.2f}]" if x else "?"
survived, brittle, errored, masking = [], [], [], []
for dom, dd in d.get("domains", {}).items():
    fid = dd.get("fidelity", {}).get("emit_no_brake")
    print(f"### {dom}   (no-brake commit fidelity: {fid})")
    defs = dd.get("defenses", {})
    for k in DEFS:
        s = defs.get(k)
        if not s: print(f"  {k} {NAME[k]:26s}  (pending)"); continue
        if not s.get("complete"):
            att = s.get("attack", {})
            frac = sum(len(v.get("done_idx", [])) for v in att.values())
            print(f"  {k} {NAME[k]:26s}  (running, {len(att)} eps started)"); continue
        v = s.get("verdict", "?")
        att = s.get("attack", {})
        # dose-response
        dr = " ".join(f"e{e.replace('eps','')}:{att[e]['asr_adaptive']:.2f}" for e in
                      sorted(att, key=lambda x: float(x.replace('eps', ''))) if 'asr_adaptive' in att[e])
        hi = f"eps{max(meta.get('EPS', [16]))}"
        asr = att.get(hi, {}).get("asr_adaptive"); rci = att.get(hi, {}).get("asr_ci")
        rnd = att.get(hi, {}).get("asr_random")
        st = s.get("stress", {}); stf = st.get("flag", "")
        sts = f" | stress@{st.get('asr'):.2f}->{stf}" if st else ""
        print(f"  {k} {NAME[k]:26s}  {v:14s} ASR@max={asr} CI{ci(rci)} rand={rnd}  [{dr}]{sts}")
        if v == "ERROR": errored.append((dom, k))
        elif v == "BRITTLE": brittle.append((dom, k))
        else: survived.append((dom, k, v))
        if stf == "GRADIENT_MASKING_SUSPECTED": masking.append((dom, k))
    print()

print("=== HEADLINE ===")
print(f"BRITTLE (collapsed under adaptive attack): {len(brittle)}  {brittle}")
print(f"SURVIVED (ROBUST/LIKELY/PARTIAL):          {len(survived)}  {survived}")
print(f"GRADIENT-MASKING SUSPECTED (distrust!):    {len(masking)}  {masking}")
print(f"ERRORED:                                   {len(errored)}  {errored}")
if survived and not masking:
    print("\n>> A defense appears to resist the adaptive attack AND passes the stress-test.")
    print("   This is the interesting case — inspect it before any claim; it would COMPLICATE the")
    print("   'interp defenses all collapse -> audit not defend' thesis. Needs a manual stronger attack.")
elif survived and masking:
    print("\n>> Apparent survivors are flagged as gradient masking (stress-test broke them).")
    print("   Consistent with the thesis: the robustness was an artifact, not real.")
elif not survived:
    print("\n>> Every defense collapses under the adaptive attack. Clean, thorough negative that")
    print("   CEMENTS the 'audit, don't defend' pivot — now with real ensemble, EOT, adaptive-D6,")
    print("   held-out, CIs, and a stress-test guard (the exact holes v1 had).")
sys.exit(0)
