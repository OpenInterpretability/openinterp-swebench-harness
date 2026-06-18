import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
A=json.load(open("data/cot_faithfulness_n60.json"))
def sig(it): return it["layers"]["L59"]["d_cot"]-it["layers"]["L59"]["d_self"]
flip=[sig(it) for it in A["items"] if it["flipped"]]
noflip=[sig(it) for it in A["items"] if not it["flipped"]]
rng=np.random.default_rng(1)
fig,ax=plt.subplots(figsize=(5.0,3.6))
for x,vals,col,lab in [(0,noflip,"#888888","Model already correct\nwithout CoT (n=%d)"%len(noflip)),
                       (1,flip,"#1f77b4","CoT flips the answer\n(n=%d)"%len(flip))]:
    jx=x+rng.uniform(-0.12,0.12,len(vals))
    ax.scatter(jx,vals,s=26,color=col,alpha=0.75,edgecolor="white",lw=0.5,zorder=3)
    ax.plot([x-0.22,x+0.22],[np.mean(vals)]*2,color=col,lw=2.4,zorder=4)
ax.axhline(0,color="gray",lw=0.7,ls=":")
ax.set_xticks([0,1]); ax.set_xticklabels(["performative","causal (faithful)"],fontsize=10)
ax.set_ylabel("causal signal $d_{cot}-d_{self}$ @ L59")
ax.set_title("Faithfulness is conditional",fontsize=11)
ax.text(0,np.mean(noflip)+0.25,"mean %.2f"%np.mean(noflip),ha="center",fontsize=8,color="#555")
ax.text(1,np.mean(flip)+0.25,"mean %.2f"%np.mean(flip),ha="center",fontsize=8,color="#1f77b4")
ax.set_xlim(-0.5,1.5)
plt.tight_layout(); plt.savefig("fig_conditional.pdf"); print("saved fig_conditional.pdf | flip mean %.2f (n=%d), noflip mean %.2f (n=%d)"%(np.mean(flip),len(flip),np.mean(noflip),len(noflip)))
