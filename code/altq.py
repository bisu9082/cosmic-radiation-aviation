import numpy as np
h=np.array([3.048,4.572,6.096,7.620,9.144,10.668,12.192])
D=np.array([0.1117,0.2321,0.4654,0.8353,1.3395,1.9308,2.5055])
def quad(mask,label):
    hh,DD=h[mask],D[mask]
    c=np.polyfit(hh,np.log(DD),2); pred=np.exp(np.polyval(c,hh))
    mare=np.mean(np.abs(pred-DD)/DD)*100
    print(f"[{label}] lnD={c[0]:+.5f}h^2{c[1]:+.5f}h{c[2]:+.5f}  MARE={mare:.2f}% (n={len(hh)})")
    for x,d,p in zip(hh,DD,pred): print(f"    h={x:.3f} D={d:.4f} fit={p:.4f} RE={100*(p-d)/d:+.1f}%")
quad(h>=0,"log-quad FL100-400 (all 7)")
quad(h>=6.0,"log-quad FL200-400 (5)")
