import numpy as np
FT2KM=0.0003048
ft=np.array([10000,15000,20000,25000,30000,35000,40000]); h=ft*FT2KM
D=np.array([0.1117,0.2321,0.4654,0.8353,1.3395,1.9308,2.5055])
c=np.polyfit(h[ft>=20000],np.log(D[ft>=20000]),2)   # FL200-400
c7=np.polyfit(h,np.log(D),2)                          # FL100-400
print("FL200-400 screen & RE:")
for f,hh,d in zip(ft,h,D):
    s=np.exp(np.polyval(c,hh)); s7=np.exp(np.polyval(c7,hh))
    print(f"  {f:5d}ft h={hh:6.3f} CARI={d:.4f}  scr200={s:.4f}({100*(s-d)/d:+.1f}%)  scr100={s7:.4f}({100*(s7-d)/d:+.1f}%)")
