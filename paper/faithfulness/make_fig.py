import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng=np.random.default_rng(0)
A=json.load(open("data/cot_faithfulness_n60.json")); B=json.load(open("data/cot_faithfulness_agent_n32.json"))
L=[15,19,23,27,35,39,43,47,51,59,63]
def causal(d):
    f=[it for it in d["items"] if it["flipped"]]
    out={}
    for l in L:
        v=[it["layers"][f"L{l}"]["d_cot"]-it["layers"][f"L{l}"]["d_self"] for it in f]
        bs=[rng.choice(v,len(v),replace=True).mean() for _ in range(5000)]
        out[l]=(np.mean(v), np.percentile(bs,2.5), np.percentile(bs,97.5))
    return out
ca=causal(A); cb=causal(B)
# logit-lens onset (added once the control returns) -- placeholder read if present
ll=None
import os
if os.path.exists("data/logit_lens_control.json"):
    ll=json.load(open("data/logit_lens_control.json"))["per_layer_margin"]
fig,ax=plt.subplots(figsize=(6.2,4.0))
for c,lab,col,mk in [(ca,"Reasoning answer (Tier-A, n=16)","#1f77b4","o"),(cb,"Agent action (Tier-B, n=10)","#d62728","s")]:
    m=[c[l][0] for l in L]; lo=[c[l][0]-c[l][1] for l in L]; hi=[c[l][2]-c[l][0] for l in L]
    ax.errorbar(L,m,yerr=[lo,hi],marker=mk,color=col,label=lab,capsize=2,lw=1.6,ms=4)
if ll:
    lm=[ll[str(l)] for l in L]
    ax2=ax.twinx(); ax2.plot(L,lm,"--",color="#2ca02c",lw=1.4,label="logit-lens margin (consolidation)")
    ax2.set_ylabel("logit-lens margin (ans$_{CoT}$ - ans$_{noCoT}$)",color="#2ca02c",fontsize=9)
    ax2.tick_params(axis="y",labelcolor="#2ca02c")
ax.axhline(0,color="gray",lw=0.7,ls=":")
ax.axvspan(51,63,alpha=0.08,color="green")
ax.set_xlabel("SAE layer"); ax.set_ylabel("causal signal  $d_{cot}-d_{self}$  ($\\Delta\\log p$)")
ax.set_title("The late channel: causal CoT content concentrates in L51--63")
ax.legend(loc="upper left",fontsize=8.5,frameon=False)
plt.tight_layout(); plt.savefig("fig_late_channel.pdf"); print("saved fig_late_channel.pdf", "with logit-lens" if ll else "(causal only; add logit-lens when control returns)")
