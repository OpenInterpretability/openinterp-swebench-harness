#!/usr/bin/env python3
"""Pre-mint eval for late_channel.tex: recompute EVERY number from data/, compare to the paper's claims.
A claim passes if recompute is within tol. Run: python eval_late_channel.py"""
import json, numpy as np, sys
rng=np.random.default_rng(0)
A=json.load(open("data/cot_faithfulness_n60.json")); B=json.load(open("data/cot_faithfulness_agent_n32.json"))
LAYERS=[15,19,23,27,35,39,43,47,51,59,63]
def boot(v,n=10000):
    v=np.array(v); return float(v.mean()),*[float(x) for x in np.percentile([rng.choice(v,len(v),replace=True).mean() for _ in range(n)],[2.5,97.5])]
PASS=[]; FAIL=[]
def chk(name, got, claim, tol=0.06):
    ok=abs(got-claim)<=tol; (PASS if ok else FAIL).append((name,round(got,3),claim,ok))
def chk_int(name, got, claim):
    ok=(got==claim); (PASS if ok else FAIL).append((name,got,claim,ok))

# ---- Tier-A (n60) ----
itsA=A["items"]; flipA=[it for it in itsA if it["flipped"]]
chk_int("A: n_items", len(itsA), 60)
chk_int("A: n_flipped", len(flipA), 16)
chk_int("A: already_correct_noCoT", sum(1 for it in itsA if not it["flipped"] and it["cot_correct"]), 44)
def causal(its,L): return [it["layers"][f"L{L}"]["d_cot"]-it["layers"][f"L{L}"]["d_self"] for it in its]
for L,claim,lo,hi in [(51,1.55,1.28,1.85),(59,2.72,2.18,3.28),(63,2.12,1.62,2.61),(47,0.17,0.03,0.29)]:
    m,clo,chi=boot(causal(flipA,L)); chk(f"A: L{L} causal", m, claim); chk(f"A: L{L} CIlo", clo, lo,0.12); chk(f"A: L{L} CIhi", chi, hi,0.18)
lateA=[np.mean([it['layers'][f'L{L}']['d_cot']-it['layers'][f'L{L}']['d_self'] for L in LAYERS if L>=51]) for it in flipA]
earlyA=[np.mean([it['layers'][f'L{L}']['d_cot']-it['layers'][f'L{L}']['d_self'] for L in LAYERS if L<=27]) for it in flipA]
lm,llo,lhi=boot(lateA); em,elo,ehi=boot(earlyA)
chk("A: LATE mean", lm,2.13); chk("A: LATE CIlo",llo,1.83,0.12); chk("A: EARLY mean",em,-0.02); chk("A: EARLY CIhi",ehi,0.01,0.08)
l59=[it["layers"]["L59"]["d_cot"]-it["layers"]["L59"]["d_self"] for it in flipA]
chk_int("A: L59 consistency (>0)", sum(1 for v in l59 if v>0), 16)
chk("A: L59 min", min(l59), 0.60,0.06); chk("A: L59 max", max(l59), 4.81,0.06)

# ---- Tier-B (n32) ----
itsB=B["items"]; flipB=[it for it in itsB if it["flipped"]]
chk_int("B: n_items", len(itsB), 32)
chk_int("B: n_flipped", len(flipB), 10)
chk_int("B: irreversible_cot", sum(1 for it in itsB if it["irreversible_cot"]), 5)
chk_int("B: cautious (32-irrev)", 32-sum(1 for it in itsB if it["irreversible_cot"]), 27)
for L,claim in [(51,0.84),(59,1.89),(63,2.21)]:
    chk(f"B: L{L} causal", float(np.mean(causal(flipB,L))), claim)
lateB=[np.mean([it['layers'][f'L{L}']['d_cot']-it['layers'][f'L{L}']['d_self'] for L in LAYERS if L>=51]) for it in flipB]
earlyB=[np.mean([it['layers'][f'L{L}']['d_cot']-it['layers'][f'L{L}']['d_self'] for L in LAYERS if L<=27]) for it in flipB]
bm,blo,bhi=boot(lateB); ebm,eblo,ebhi=boot(earlyB)
chk("B: LATE mean", bm,1.65,0.10); chk("B: EARLY mean", ebm,0.08,0.10)

# ---- Control: logit-lens consolidation (rules out patch-dilution) ----
import os
if os.path.exists("data/logit_lens_control.json"):
    LL=json.load(open("data/logit_lens_control.json")); pm=LL["per_layer_margin"]
    chk("LL: L35 margin (neg early)", pm["35"], -0.93, 0.05)
    chk("LL: L47 margin (~0)", pm["47"], 0.02, 0.05)
    chk("LL: L51 margin", pm["51"], 1.57, 0.05)
    chk("LL: L59 margin", pm["59"], 5.18, 0.06)
    chk("LL: L63 margin", pm["63"], 6.64, 0.06)
    chk_int("LL: onset layer", LL["onset"], 51)
else:
    FAIL.append(("LL: control json missing","-","-",False))

# ---- report ----
print(f"{'CLAIM':<32}{'RECOMPUTED':>12}{'PAPER':>10}  OK")
for n,g,c,ok in PASS+FAIL: print(f"{n:<32}{str(g):>12}{str(c):>10}  {'PASS' if ok else 'XXXX FAIL'}")
print(f"\n{len(PASS)}/{len(PASS)+len(FAIL)} claims pass. " + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAIL -> fix paper before mint"))
sys.exit(0 if not FAIL else 1)
