import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
limits=["Detect $\\to$ control\n(AUROC / $\\Delta P$)",
        "Form\n(raw / struct-matched AUROC)","Brake $\\to$ robust\n(benign / adversarial)"]
fav=[0.91,0.838,1.00]; surv=[0.00,0.08,0.00]
x=np.arange(len(limits)); w=0.38
fig,ax=plt.subplots(figsize=(6.4,3.7))
b1=ax.bar(x-w/2,fav,w,color="#1f77b4",label="favorable view")
b2=ax.bar(x+w/2,surv,w,color="#d62728",label="survives scrutiny")
for b in b1: ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f"{b.get_height():.2f}",ha="center",fontsize=8)
for b in b2: ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f"{b.get_height():.2f}",ha="center",fontsize=8,color="#d62728")
ax.set_xticks(x); ax.set_xticklabels(limits,fontsize=8.5)
ax.set_ylim(0,1.30); ax.set_ylabel("metric (0–1)")
ax.set_title("Located, not secured: every control promise collapses under scrutiny",fontsize=10.5)
ax.legend(loc="upper center",ncol=2,fontsize=8.5,frameon=False,bbox_to_anchor=(0.5,1.0))
plt.tight_layout(); plt.savefig("fig_located.pdf"); print("saved fig_located.pdf")
