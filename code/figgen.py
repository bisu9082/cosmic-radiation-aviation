import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=np.load('fit_params.npz'); FT2KM=0.0003048
c,c7,a,n,Rcref,Dref=d['c'],d['c7'],float(d['a']),float(d['n']),float(d['Rcref']),float(d['Dref'])
latRc,latD,hRc,hD=d['latRc'],d['latD'],d['hRc'],d['hD']
alt_ft,Dmax,Nboot=d['alt_ft'],d['Dmax'],d['Nboot']
cross=float(d['cross'])
plt.rcParams.update({'font.size':10,'font.family':'serif','axes.linewidth':0.8,
  'xtick.direction':'in','ytick.direction':'in','legend.frameon':False})

# ========== FIG 1 : altitude ==========
fig,ax=plt.subplots(figsize=(6.5,4.6))
h=alt_ft*FT2KM
ax.semilogy(h,Dmax,'o',mfc='none',mec='#c0392b',mew=1.6,ms=9,label='CARI-7 reference (HCP=1141 MV)',zorder=5)
hh=np.linspace(3.0,15.3,300)
# fit curve: FL200-400 solid, extension dashed
fit=np.exp(np.polyval(c,hh))
in_band=(hh>=20000*FT2KM)&(hh<=40000*FT2KM)
ax.semilogy(hh[in_band],fit[in_band],'-',color='#c0392b',lw=2,label='Log-quadratic screen (Eq. 1), FL200--FL400')
ax.semilogy(hh[~in_band],fit[~in_band],':',color='#c0392b',lw=1.3,alpha=0.8)
for a_ft in [10000,20000,30000,40000]:
    hv=a_ft*FT2KM; Dv=Dmax[list(alt_ft).index(a_ft)]
    ax.annotate(f"{a_ft//1000}k ft",(hv,Dv),textcoords="offset points",xytext=(6,-12),fontsize=8,color='#555')
ax.set_xlabel('Altitude (km)'); ax.set_ylabel(r'Effective dose rate ($\mu$Sv h$^{-1}$)')
ax.set_xlim(2.5,15.5); ax.set_ylim(0.08,3.5)
secx=ax.secondary_xaxis('top',functions=(lambda x:x/FT2KM/1000,lambda x:x*1000*FT2KM))
secx.set_xlabel('Altitude (ft, thousands)')
txt=(r"$\ln D = -0.0232\,h^2 + 0.701\,h - 4.172$"+"\n"
     r"MARE $=0.10\%$ (FL200--FL400); $1.05\%$ (FL100--FL400)")
ax.text(0.97,0.06,txt,transform=ax.transAxes,ha='right',va='bottom',fontsize=8.5,
        bbox=dict(boxstyle='round',fc='white',ec='#aaa',alpha=0.9))
ax.legend(loc='upper left',fontsize=8.5)
ax.grid(True,which='both',ls=':',lw=0.4,alpha=0.5)
fig.tight_layout(); fig.savefig('figure1.eps',format='eps',bbox_inches='tight'); fig.savefig('figure1.png',dpi=150,bbox_inches='tight'); plt.close()

# ========== FIG 3 : annual dose bars ==========
ann=d['annual']  # rows 40,30,20,10 k ; cols nominal,smin,conserv
labels=['40,000 ft\n(672 h/yr)','30,000 ft\n(576 h/yr)','20,000 ft\n(360 h/yr)','10,000 ft\n(600 h/yr)']
x=np.arange(4); w=0.26
fig,ax=plt.subplots(figsize=(6.8,4.4))
cols=['#34495e','#e08a1e','#e8c39e']; names=['Nominal (solar max)','Solar minimum','Conservative (+15%)']
for i in range(3):
    bars=ax.bar(x+(i-1)*w,ann[:,i],w,color=cols[i],edgecolor='k',lw=0.5,label=names[i])
    for b,v in zip(bars,ann[:,i]):
        ax.text(b.get_x()+b.get_width()/2,v+0.02,f"{v:.2f}",ha='center',va='bottom',fontsize=7)
ax.axhline(1.0,ls='--',color='#c0392b',lw=1.2)
ax.text(3.4,1.03,r'ICRP 1 mSv yr$^{-1}$ (public reference level)',color='#c0392b',fontsize=8,ha='right')
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8.5)
ax.set_ylabel(r'Annual effective dose (mSv yr$^{-1}$)'); ax.set_ylim(0,2.5)
ax.legend(loc='upper right',fontsize=8.5)
fig.tight_layout(); fig.savefig('figure3.eps',format='eps',bbox_inches='tight'); fig.savefig('figure3.png',dpi=150,bbox_inches='tight'); plt.close()

# ========== FIG 6 : latitude (3 panels) ==========
def sat(rc,a,n): return (1+a*Rcref**n)/(1+a*rc**n)
flat=latD/Dref; hf=hD/Dref
fig,axs=plt.subplots(1,3,figsize=(11,3.5))
# (a)
ax=axs[0]; rr=np.linspace(0,18,300)
ax.plot(rr,sat(rr,a,n),'-',color='#2c3e50',lw=2,label=f'Saturation fit\n(a={a:.3f}, n={n:.2f})')
ax.plot(latRc,flat,'o',mfc='#c0392b',mec='k',ms=7,label='CARI-7 fit nodes (11)')
ax.plot(hRc,hf,'^',mfc='#f1c40f',mec='k',ms=8,label='Held-out (3)')
ax.axhline(1,ls=':',color='#999',lw=0.8); ax.axvline(Rcref,ls=':',color='#999',lw=0.8)
ax.set_xlabel(r'Cutoff rigidity $R_c$ (GV)'); ax.set_ylabel(r'$f_{\rm lat}=D(R_c)/D(R_{c,\rm ref})$')
ax.set_title('(a) Latitude correction factor',fontsize=10); ax.legend(fontsize=7.5,loc='upper right')
ax.grid(True,ls=':',lw=0.4,alpha=0.5)
# (b)
ax=axs[1]; ax.hist(Nboot,bins=40,color='#5499c7',edgecolor='k',lw=0.3)
ax.axvline(np.median(Nboot),color='#c0392b',lw=1.5)
ax.axvline(np.percentile(Nboot,2.5),color='#c0392b',ls='--',lw=1)
ax.axvline(np.percentile(Nboot,97.5),color='#c0392b',ls='--',lw=1)
ax.set_xlabel(r'Exponent $\hat n$'); ax.set_ylabel('Bootstrap count')
ax.set_title(r'(b) Bootstrap of $\hat n$ (B=10,000)',fontsize=10)
ax.text(0.95,0.95,f"$\\hat n$={np.median(Nboot):.2f}\n95% CI [{np.percentile(Nboot,2.5):.2f}, {np.percentile(Nboot,97.5):.2f}]",
        transform=ax.transAxes,ha='right',va='top',fontsize=8)
# (c)
ax=axs[2]
allobs=np.concatenate([flat,hf]); allpred=np.concatenate([sat(latRc,a,n),sat(hRc,a,n)])
ax.plot([0.6,2],[0.6,2],'-',color='#999',lw=0.8)
ax.plot(sat(latRc,a,n),flat,'o',mfc='#c0392b',mec='k',ms=7,label='In-sample (3.4%)')
ax.plot(sat(hRc,a,n),hf,'^',mfc='#f1c40f',mec='k',ms=8,label='Held-out (1.9%)')
ax.set_xlabel('Predicted $f_{\\rm lat}$'); ax.set_ylabel('CARI-7 $f_{\\rm lat}$')
ax.set_title('(c) Screen vs CARI-7',fontsize=10); ax.legend(fontsize=8,loc='upper left')
ax.grid(True,ls=':',lw=0.4,alpha=0.5); ax.set_xlim(0.6,2); ax.set_ylim(0.6,2)
fig.tight_layout(); fig.savefig('figure6.eps',format='eps',bbox_inches='tight'); fig.savefig('figure6.png',dpi=150,bbox_inches='tight'); plt.close()
print("figures regenerated:", __import__('os').popen('ls -la figure1.eps figure3.eps figure6.eps').read())
