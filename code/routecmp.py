import numpy as np
FT2KM=0.0003048
a,n,Rcref=0.023567,1.51857,10.52   # from master fit (recompute precise)
# recompute exact a,n from the 11 nodes to be consistent
from scipy.optimize import curve_fit
latRc=np.array([0.03,0.00,0.48,1.85,3.74,4.86,8.34,10.52,15.91,16.79,17.21])
latD=np.array([2.4250,2.4250,2.4250,2.3975,2.1753,2.0099,1.5734,1.3395,1.0239,0.98808,0.97125])
Dref=1.3395
def sat(rc,a,n): return (1+a*Rcref**n)/(1+a*rc**n)
(a,n),_=curve_fit(sat,latRc,latD/Dref,p0=[0.05,1.4],maxfev=30000)
c=np.array([-0.0232268,0.700697,-4.172282])  # FL200-400 log-quad (recompute)
ft=np.array([20000,25000,30000,35000,40000]);h=ft*FT2KM
Dm=np.array([0.4654,0.8353,1.3395,1.9308,2.5055]); c=np.polyfit(h,np.log(Dm),2)
def Dalt(hk): return np.exp(np.polyval(c,hk))
# routes
R={'ICN-JFK':dict(cruise_ft=35000,Rc=0.01,block=850,cari=47.281),
   'ICN-SIN':dict(cruise_ft=37000,Rc=16.37,block=400,cari=10.301),
   'ICN-FRA':dict(cruise_ft=37000,Rc=1.27,block=700,cari=42.635),
   'ICN-LAX':dict(cruise_ft=36000,Rc=2.89,block=670,cari=35.034),
   'ICN-CJU':dict(cruise_ft=28000,Rc=11.31,block=60,cari=0.69169)}
print(f"fit: a={a:.4f} n={n:.4f}")
print(f"{'route':9} {'cruiseFL':8} {'Rc_mid':6} {'scrRate':7} {'CARIrate':8} {'scrDose':7} {'CARIdose':8} {'doseRE':7}")
rows=[]
for name,d in R.items():
    hk=d['cruise_ft']*FT2KM
    scr_rate=Dalt(hk)*sat(d['Rc'],a,n)      # µSv/h at cruise, midpoint Rc
    bh=d['block']/60
    cari_rate=d['cari']/bh
    scr_dose=scr_rate*bh                      # screen route dose = cruise rate x block time
    re=100*(scr_dose-d['cari'])/d['cari']
    print(f"{name:9} FL{d['cruise_ft']//100:<5} {d['Rc']:6.2f} {scr_rate:7.3f} {cari_rate:8.3f} {scr_dose:7.2f} {d['cari']:8.3f} {re:+6.1f}%")
    rows.append((name,d['Rc'],scr_rate,cari_rate,scr_dose,d['cari'],re))
# annual flag demo: crew flies route round-trip, N trips/yr -> which cross 1 mSv/yr?
print("\nAnnual-dose flag (crew flying route, 2x block h per round trip):")
print(f"{'route':9} {'trips/yr':8} {'ann_screen':10} {'ann_CARI':9} {'flag_scr':8} {'flag_CARI':9} {'match':5}")
trips={'ICN-JFK':40,'ICN-SIN':40,'ICN-FRA':40,'ICN-LAX':40,'ICN-CJU':200}
for name,d in R.items():
    hk=d['cruise_ft']*FT2KM; bh=d['block']/60
    scr_rate=Dalt(hk)*sat(d['Rc'],a,n); scr_dose=scr_rate*bh
    N=trips[name]
    ann_scr=scr_dose*2*N/1000; ann_cari=d['cari']*2*N/1000
    fs='FLAG' if ann_scr>1 else 'ok'; fc='FLAG' if ann_cari>1 else 'ok'
    print(f"{name:9} {N:<8} {ann_scr:10.2f} {ann_cari:9.2f} {fs:8} {fc:9} {'OK' if fs==fc else 'MISMATCH'}")
