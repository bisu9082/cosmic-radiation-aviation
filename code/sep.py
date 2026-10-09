import numpy as np
ft=np.array([10000,20000,30000,35000,40000])
kor=np.array([0.1117,0.4654,1.3395,1.9308,2.5055])   # Rc10.52
pol=np.array([0.16194,0.77555,2.4250,3.6556,5.0011]) # Rc0.03
equ=np.array([0.084649,0.33589,0.97125,1.4019,1.7982]) # Rc17.21
# normalize to FL300
def nz(a): return a/a[2]
print("normalized to FL300 (shape):")
print("  ft:    ",ft)
print("  Korea: ",np.round(nz(kor),4))
print("  Polar: ",np.round(nz(pol),4))
print("  Equat: ",np.round(nz(equ),4))
# separability error: predict pol/equ using Korea shape * f_lat(from FL300)
flat_pol=pol[2]/kor[2]; flat_equ=equ[2]/kor[2]
pred_pol=kor*flat_pol; pred_equ=kor*flat_equ
print(f"\nf_lat(FL300): polar={flat_pol:.3f} equator={flat_equ:.3f}")
print("separability RE (full screen = Dalt(Korea)*f_lat):")
for i,f in enumerate(ft):
    rep=100*(pred_pol[i]-pol[i])/pol[i]; ree=100*(pred_equ[i]-equ[i])/equ[i]
    print(f"  {f:5d}ft: polar RE={rep:+.1f}%   equator RE={ree:+.1f}%")
# over cruise band FL200-400 MARE
m=ft>=20000
print(f"\nseparability MARE (FL200-400): polar={np.mean(np.abs((pred_pol[m]-pol[m])/pol[m]))*100:.1f}%  equator={np.mean(np.abs((pred_equ[m]-equ[m])/equ[m]))*100:.1f}%")
