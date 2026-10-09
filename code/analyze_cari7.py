import numpy as np, re
from scipy.optimize import curve_fit
rows=[]
for ln in open('KUNODES.ANS'):
    if re.match(r'\s*[-\d]+\.\d+,', ln):
        p=[x.strip() for x in ln.split(',')]
        # lat,lon,alt,unit(F),date,hr,vcr,particle,dose,unit,qty
        try:
            lat=float(p[0]); lon=float(p[1]); alt=float(p[2]); vcr=float(p[6]); dose=float(p[8])
            rows.append((lat,lon,alt,vcr,dose))
        except: pass
print("parsed rows:",len(rows))
# altitude profile = Korea (lat 37.47, lon 126.45), all altitudes
alt_rows=[r for r in rows if abs(r[0]-37.47)<0.01 and abs(r[1]-126.45)<0.01]
alt_rows=sorted(alt_rows,key=lambda r:r[2])
h_km=np.array([r[2]*0.0003048 for r in alt_rows]); Dalt=np.array([r[4] for r in alt_rows])
print("\n=== ALTITUDE PROFILE (Korea, Rc=%.2f) ==="%alt_rows[0][3])
for h,D in zip(h_km,Dalt): print(f"  {h*1000/0.3048:6.0f}ft  {h:6.3f}km  D={D:.4f}")

def expfit(hh,DD,label):
    k,b=np.polyfit(hh,np.log(DD),1); D0=np.exp(b)
    pred=D0*np.exp(k*hh); mare=np.mean(np.abs(pred-DD)/DD)*100
    print(f"  [{label}] D={D0:.4f}*exp({k:.4f}*h)  MARE={mare:.2f}%  (n={len(hh)})")
    for h,D,pr in zip(hh,DD,pred): print(f"      h={h:.3f} D={D:.4f} fit={pr:.4f} RE={100*(pr-D)/D:+.1f}%")
    return D0,k,mare
print("\n-- exponential over FL200-400 (real) --")
m=(h_km>=6.0)&(h_km<=12.2); expfit(h_km[m],Dalt[m],"FL200-400")
print("-- exponential over FL250-400 --")
m2=(h_km>=7.5)&(h_km<=12.2); expfit(h_km[m2],Dalt[m2],"FL250-400")

# latitude nodes at 30000ft
fl300=[r for r in rows if abs(r[2]-30000)<1]
# fit nodes = all except held-out coords (munich48.35,rome41.8,tokyo35.55)
held_lat=[48.35,41.80,35.55]
fit=[r for r in fl300 if round(r[0],2) not in [round(x,2) for x in held_lat]]
held=[r for r in fl300 if round(r[0],2) in [round(x,2) for x in held_lat]]
# add korea reference (from alt profile 30000)
Dref=[r[4] for r in alt_rows if abs(r[2]-30000)<1][0]
Rcref=[r[3] for r in alt_rows if abs(r[2]-30000)<1][0]
print(f"\n=== LATITUDE (FL300) ref Rc={Rcref:.2f} Dref={Dref:.4f} ===")
# build fit set: unique by Rc
Rc=np.array([r[3] for r in fit]); Df=np.array([r[4] for r in fit])
flat=Df/Dref
order=np.argsort(Rc); Rc,Df,flat=Rc[order],Df[order],flat[order]
for rc,d,f in zip(Rc,Df,flat): print(f"  Rc={rc:5.2f}  D={d:.4f}  f_lat={f:.3f}")
def sat(rc,a,n): return (1+a*Rcref**n)/(1+a*rc**n)
p0=[0.1,1.3]
popt,_=curve_fit(sat,Rc,flat,p0=p0,maxfev=20000)
a,n=popt; pred=sat(Rc,a,n); mare=np.mean(np.abs(pred-flat)/flat)*100
print(f"\n  SATURATION fit: a={a:.4f} n={n:.4f}  in-sample MARE={mare:.3f}%")
# bootstrap
rng=np.random.default_rng(42); A=[];N=[]
for _ in range(10000):
    idx=rng.integers(0,len(Rc),len(Rc))
    try:
        pp,_=curve_fit(sat,Rc[idx],flat[idx],p0=popt,maxfev=20000); A.append(pp[0]);N.append(pp[1])
    except: pass
A=np.array(A);N=np.array(N)
print(f"  bootstrap B={len(A)}: a={np.median(A):.4f} CI[{np.percentile(A,2.5):.4f},{np.percentile(A,97.5):.4f}]")
print(f"                 n={np.median(N):.4f} CI[{np.percentile(N,2.5):.4f},{np.percentile(N,97.5):.4f}]")
# held-out
print("\n  -- HELD-OUT (out-of-sample) --")
hRc=np.array([r[3] for r in held]); hD=np.array([r[4] for r in held]); hf=hD/Dref
hpred=sat(hRc,a,n); hmare=np.mean(np.abs(hpred-hf)/hf)*100
for rc,f,pr in zip(hRc,hf,hpred): print(f"      Rc={rc:5.2f} f_obs={f:.3f} f_pred={pr:.3f} RE={100*(pr-f)/f:+.1f}%")
print(f"      HELD-OUT MARE={hmare:.2f}%")
print(f"\n  polar/equator ratio: {flat.max():.2f}/{flat.min():.2f} = {flat.max()/flat.min():.2f}")
