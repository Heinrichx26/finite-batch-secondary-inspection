"""Draw vector publication figures from saved, verified experiment results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE/'results/capacity_capped_review_20261018/figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.labelsize':9,'axes.titlesize':10,
                     'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
                     'svg.fonttype':'none','axes.spines.top':False})
stopped = pd.read_csv(BASE/'results/capacity_capped_review_20261018/expanded_capacity_frontier.csv')
fixed = pd.read_csv(BASE/'results/manuscript_20261017/source_policy_frontier.csv')
dstar = .9999**286*.99**4
best = fixed.groupby('capacity').lower.max()/dstar
full = fixed[fixed.reviews.eq(244)].set_index('capacity').lower/dstar
original = stopped[stopped.zeta.eq(0)].set_index('capacity')
omissions = stopped[stopped.zeta.eq(.05)].set_index('capacity')
fig, axes = plt.subplots(1,2,figsize=(7.15,3.65),layout='constrained')
ax=axes[0]
ax.plot(original.index,100*original.lower_S,color='#164f86',lw=1.8,label='Referral stopping, at most 244')
ax.plot(best.index,100*best,color='#a45424',ls='--',lw=1.5,label='Best fixed count')
ax.step(full.index,100*full,where='post',color='#666666',ls=':',lw=1.4,label='Fixed full review')
for y in (80,90):ax.axhline(y,color='#bbbbbb',lw=.7,zorder=0)
point_stop=float(original.loc[895,'lower_S']*100)
point_fixed=float(best.loc[930]*100)
ax.scatter([895,930],[point_stop,point_fixed],color=['#164f86','#a45424'],s=17,zorder=4)
ax.annotate('895',(895,point_stop),xytext=(891,68),fontsize=8,arrowprops={'arrowstyle':'-','lw':.6})
ax.annotate('930',(930,80),xytext=(931,66),fontsize=8,arrowprops={'arrowstyle':'-','lw':.6})
ax.set(xlim=(880,970),ylim=(45,102),xlabel='Total search allowance, C',
       ylabel='Lower referral-and-capacity probability (%)',title='(a) Review rule and required capacity')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),borderaxespad=0,
          frameon=False)
ax.set_xticks([880,900,920,940,960])
ax.grid(axis='x',color='#dddddd',lw=.5)

ax=axes[1]
line1,=ax.plot(omissions.index,100*omissions.lower_J_specified,color='#164f86',lw=1.8,label='Lower final success')
ax.axhline(80,color='#164f86',lw=.8,ls=':')
ax.axvline(915,color='#777777',lw=.8,ls='--')
ax.text(917,65,'Joint minimum: 915',fontsize=8,color='#333333')
ax2=ax.twinx()
ax2.spines['top'].set_visible(False)
line2,=ax2.plot(omissions.index,100*omissions.risk_upper_specified,color='#a45424',lw=1.5,ls='--',label='Upper mean missed-target fraction')
ax2.axhline(.1,color='#a45424',lw=.8,ls=':')
ax.set(xlim=(890,970),ylim=(60,94),xlabel='Total search allowance, C',
       ylabel='Final success (%)',title='(b) 5% target-omission allowance')
ax2.set(ylim=(0,.18),ylabel='Mean missed-target fraction (%)')
ax.set_xticks([900,920,940,960])
ax.grid(axis='x',color='#dddddd',lw=.5)
ax.legend([line1,line2],[line1.get_label(),line2.get_label()],loc='upper center',
          bbox_to_anchor=(.5,-.24),borderaxespad=0,frameon=False)
for extension in ('pdf','svg','png'):
    fig.savefig(OUT/f'stopping_resource_requirements.{extension}',dpi=220)
plt.close(fig)
print(OUT/'stopping_resource_requirements.pdf')
