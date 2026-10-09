import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams.update({'font.size':10,'axes.linewidth':0.8,'font.family':'serif'})
FT2KM=0.0003048
# ---- REAL CARI-7 DATA ----
alt_ft=np.array([10000,15000,20000,25000,30000,35000,40000]); h=alt_ft*FT2KM
Dmax=np.array([0.1117,0.2321,0.4654,0.8353,1.3395,1.9308,2.5055])  # Nov2025 HCP1141
smin_ft=np.array([10000,20000,30000,40000])
Dmin=np.array([0.12882,0.53981,1.5436,2.8870])                     # 2020 HCP275
Rcref=10.52; Dref=1.3395
latRc=np.array([0.03,0.00,0.48,1.85,3.74,4.86,8.34,10.52,15.91,16.79,17.21])
latD =np.array([2.4250,2.4250,2.4250,2.3975,2.1753,2.0099,1.5734,1.3395,1.0239,0.98808,0.97125])
hRc=np.array([4.04,6.34,11.27]); hD=np.array([2.1305,1.8061,1.2835])
hours={40000:672,30000:576,20000:360,10000:600}

print("="*60)
# ---- ALTITUDE log-quadratic FL200-400 ----
m=(alt_ft>=20000)
c=np.polyfit(h[m],np.log(Dmax[m]),2)
pred=np.exp(np.polyval(c,h[m])); mare=np.mean(np.abs(pred-Dmax[m])/Dmax[m])*100
print(f"ALT log-quad FL200-400: lnD={c[0]:.5f}h^2+{c[1]:.5f}h+{c[2]:.5f}  MARE={mare:.3f}%")
c7=np.polyfit(h,np.log(Dmax),2); pr7=np.exp(np.polyval(c7,h))
mare7=np.mean(np.abs(pr7-Dmax)/Dmax)*100
print(f"ALT log-quad FL100-400: lnD={c7[0]:.5f}h^2+{c7[1]:.5f}h+{c7[2]:.5f}  MARE={mare7:.3f}%")

# ---- LATITUDE saturation ----
flat=latD/Dref
def sat(rc,a,n): return (1+a*Rcref**n)/(1+a*rc**n)
popt,_=curve_fit(sat,latRc,flat,p0=[0.05,1.4],maxfev=30000); a,n=popt
mlat=np.mean(np.abs(sat(latRc,a,n)-flat)/flat)*100
print(f"\nLAT saturation: a={a:.4f} n={n:.4f}  in-sample MARE={mlat:.3f}%")
rng=np.random.default_rng(7); A=[];N=[]
for _ in range(10000):
    idx=rng.integers(0,len(latRc),len(latRc))
    try: pp,_=curve_fit(sat,latRc[idx],flat[idx],p0=popt,maxfev=30000); A.append(pp[0]);N.append(pp[1])
    except: pass
A=np.array(A);N=np.array(N)
aCI=(np.percentile(A,2.5),np.percentile(A,97.5)); nCI=(np.percentile(N,2.5),np.percentile(N,97.5))
print(f"  bootstrap a={np.median(A):.4f} CI[{aCI[0]:.4f},{aCI[1]:.4f}]  n={np.median(N):.4f} CI[{nCI[0]:.3f},{nCI[1]:.3f}]")
hf=hD/Dref; hpred=sat(hRc,a,n); hmare=np.mean(np.abs(hpred-hf)/hf)*100
print(f"  HELD-OUT MARE={hmare:.2f}%  (Rc,obs,pred): "+", ".join(f"{r:.1f}:{o:.3f}/{p:.3f}" for r,o,p in zip(hRc,hf,hpred)))
print(f"  polar/equator f_lat: {flat.max():.3f}/{flat.min():.3f}={flat.max()/flat.min():.2f}")

# ---- SOLAR scaling ----
mi=np.isin(alt_ft,smin_ft)
ratio=Dmin/Dmax[mi]
print(f"\nSOLAR min/max ratio per alt: "+", ".join(f"{a}:{r:.3f}" for a,r in zip(smin_ft,ratio))+f"  mean=+{(ratio.mean()-1)*100:.1f}%")

# ---- ANNUAL DOSES ----
print("\nANNUAL DOSE (mSv/yr): alt  nominal  solarmin  conservative")
Dmax_d={10000:0.1117,20000:0.4654,30000:1.3395,40000:2.5055}
Dmin_d={10000:0.12882,20000:0.53981,30000:1.5436,40000:2.8870}
annual={}
for a_ft in [40000,30000,20000,10000]:
    nom=Dmax_d[a_ft]*hours[a_ft]/1000; smn=Dmin_d[a_ft]*hours[a_ft]/1000; con=smn*1.15
    annual[a_ft]=(nom,smn,con)
    flag=lambda v:'†' if v>1 else ' '
    print(f"  {a_ft}ft h={hours[a_ft]}: {nom:.3f}{flag(nom)} {smn:.3f}{flag(smn)} {con:.3f}{flag(con)}")

# ---- CROSSOVER Rc at FL300 600h/yr (nominal/solar max) ----
from scipy.optimize import brentq
Dof=lambda rc: Dref*sat(rc,a,n)
annual_lat=lambda rc: Dof(rc)*600/1000
cross=brentq(lambda rc: annual_lat(rc)-1.0, 0.1, 17)
print(f"\nCROSSOVER Rc (FL300,600h,nominal)= {cross:.2f} GV")
print(f"  polar(Rc=0.03) annual={annual_lat(0.03):.3f}  equator(Rc=17.2) annual={annual_lat(17.2):.3f}")
print(f"  Korea(Rc=10.52) annual={annual_lat(10.52):.3f}")

# save raw CSV
import csv
with open('raw_results_cari7.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['block','lat_or_Rc','altitude_ft','HCP_MV','Rc_GV','dose_uSv_h'])
    for a_ft,D in zip(alt_ft,Dmax): w.writerow(['alt_smax',37.47,a_ft,1141,10.52,D])
    for a_ft,D in zip(smin_ft,Dmin): w.writerow(['alt_smin',37.47,a_ft,275,10.52,D])
    for rc,D in zip(latRc,latD): w.writerow(['lat_fit','',30000,1141,rc,D])
    for rc,D in zip(hRc,hD): w.writerow(['lat_heldout','',30000,1141,rc,D])
print("\nsaved raw_results_cari7.csv")

# store params for figure script
np.savez('fit_params.npz',c=c,c7=c7,a=a,n=n,Rcref=Rcref,Dref=Dref,
         latRc=latRc,latD=latD,hRc=hRc,hD=hD,alt_ft=alt_ft,Dmax=Dmax,
         Nboot=N,aCI=aCI,nCI=nCI,cross=cross,annual=np.array([annual[k] for k in [40000,30000,20000,10000]]),
         mare_fl200=mare,mare_fl100=mare7,mlat=mlat,hmare=hmare)
