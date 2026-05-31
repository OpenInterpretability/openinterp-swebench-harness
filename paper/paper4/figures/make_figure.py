import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'serif'})
fig,(a1,a2)=plt.subplots(1,2,figsize=(8.4,3.3))
GRY='#9aa0a6'; BLU='#3a6ea5'
# (a) finalization (verified: 30/40/70/70)
arms=['No\nintervention','Residual\nL11','Behavioral\nneutral','Behavioral\nre-plan']
fin=[30,40,70,70]; cols=[GRY,GRY,BLU,BLU]
b=a1.bar(range(4),fin,color=cols,width=0.66,edgecolor='white')
a1.set_ylim(0,90); a1.set_ylabel('% emit finish (rescue)'); a1.set_title('(a) Finalization rate (n=20, paired)')
a1.set_xticks(range(4)); a1.set_xticklabels(arms,fontsize=8.5)
for i,v in enumerate(fin): a1.text(i,v+2,f'{v}%',ha='center',fontsize=9)
# significance brackets B1 vs 0 and B1 vs A
def brk(ax,x1,x2,y,txt):
    ax.plot([x1,x1,x2,x2],[y,y+2,y+2,y],lw=1,c='k'); ax.text((x1+x2)/2,y+2.3,txt,ha='center',fontsize=8)
brk(a1,0,3,79,'McNemar p=0.021'); brk(a1,1,3,86,'p=0.031')
a1.text(0.5,46,'residual\ninert\n(p=0.63)',ha='center',fontsize=7.5,color=GRY)
# (b) solve-rate (verified: baselines 25/21, B1 50)
lab=['No-hook\nbaseline','Phase-6\nbaseline','B1\ninterruption']; sv=[25,21,50]; c2=[GRY,GRY,BLU]
a2.bar(range(3),sv,color=c2,width=0.6,edgecolor='white')
a2.set_ylim(0,72); a2.set_ylabel('% SWE-bench Pro resolved'); a2.set_title('(b) Solve-rate (suggestive)')
a2.set_xticks(range(3)); a2.set_xticklabels(lab,fontsize=8.5)
for i,v in enumerate(sv): a2.text(i,v+1.5,f'{v}%',ha='center',fontsize=9)
brk(a2,0,2,58,'paired 5-0, p=0.062')
a2.text(1.0,66,'cross-session caveat',ha='center',fontsize=7.5,style='italic',color='#777')
plt.tight_layout(); plt.savefig('figures/main_result.pdf',bbox_inches='tight'); plt.savefig('figures/main_result.png',dpi=150,bbox_inches='tight')
print('wrote figures/main_result.pdf + .png')
