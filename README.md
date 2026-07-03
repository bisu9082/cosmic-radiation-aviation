import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec

# ---- Table 6 data (submitted MS): 11 Rc nodes ----
# id, Rc(GV), D(uSv/h), region, lat, lon
rows = [
 (1, 0.5 ,3.92,"Polar / Svalbard",           78.0,  15.0),
 (2, 1.0 ,3.72,"Sub-Arctic / N. Canada",      62.0, -100.0),
 (3, 2.0 ,3.32,"Sub-polar / Iceland",         64.0, -19.0),
 (4, 3.0 ,2.93,"N. Europe / UK",              54.0,  -2.0),
 (5, 4.5 ,2.42,"C. Europe / France",          47.0,   2.0),
 (6, 6.0 ,2.01,"S. Europe / Croatia",         45.0,  16.0),
 (7, 8.0 ,1.63,"N. Africa / Mediterranean",   33.0,  13.0),
 (8, 9.96,1.40,"Korea (reference)",           37.5, 127.0),
 (9,12.0 ,1.17,"Hong Kong / SE Asia",         22.0, 114.0),
 (10,14.5,1.01,"Philippines / Tropics",       13.0, 122.0),
 (11,17.0,0.88,"Singapore / Equatorial",       1.3, 104.0),
]
ids=[r[0] for r in rows]; Rc=np.array([r[1] for r in rows]); D=np.array([r[2] for r in rows])
region=[r[3] for r in rows]; lat=np.array([r[4] for r in rows]); lon=np.array([r[5] for r in rows])

# ---- warm muted earth-tone sequential colormap (no cold blue) ----
warm = LinearSegmentedColormap.from_list("warm_earth",
        ["#F2E4C9","#E8B45E","#E8943A","#C94F4A","#7A2E2A"])
norm = Normalize(vmin=0.8, vmax=4.0)
colors = warm(norm(D))

# ---- fonts ----
FS_PANEL=28; FS_LABEL=24; FS_TICK=16; FS_ANNO=14
LAND="#EFE7D8"; OCEAN="#FAF8F3"; COAST="#9A9086"; GRID="#D8D0C2"

fig=plt.figure(figsize=(20,13), dpi=200)
gs=GridSpec(2,1, height_ratios=[2.05,1.0], hspace=0.30, figure=fig)

# ===================== Panel (a): world map =====================
ax=fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
ax.set_extent([-140,170,-25,88], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=OCEAN)
ax.add_feature(cfeature.LAND.with_scale("110m"),  facecolor=LAND, edgecolor="none")
ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=COAST, linewidth=0.6)
gl=ax.gridlines(draw_labels=True, linewidth=0.5, color=GRID, alpha=0.9, linestyle="--")
gl.top_labels=False; gl.right_labels=False
gl.xlabel_style={"size":FS_TICK-2}; gl.ylabel_style={"size":FS_TICK-2}

# label offsets (dx,dy in deg) + alignment per priority, tuned to avoid overlap
off={1:(3,2,'left','bottom'),2:(3,-2,'left','top'),3:(-3,2,'right','bottom'),
     4:(-4,-3,'right','top'),5:(3,-3,'left','top'),6:(3,2,'left','bottom'),
     7:(3,-3,'left','top'),8:(3,2,'left','bottom'),9:(4,-1,'left','top'),
     10:(4,1,'left','bottom'),11:(3,-3,'left','top')}
for r in rows:
    i,rc,d,reg,la,lo=r
    ax.scatter(lo,la,s=430,c=[warm(norm(d))],edgecolor="#3A3230",linewidth=1.6,
               zorder=6,transform=ccrs.PlateCarree())
    dx,dy,ha,va=off[i]
    ax.text(lo+dx,la+dy,str(i),fontsize=FS_ANNO+2,fontweight="bold",color="#2A2320",
            ha=ha,va=va,zorder=7,transform=ccrs.PlateCarree())
# highlight reference (Korea)
ax.scatter(127,37.5,s=900,facecolor="none",edgecolor="#2A2320",linewidth=2.2,
           zorder=5,transform=ccrs.PlateCarree())

# colorbar
sm=cm.ScalarMappable(norm=norm,cmap=warm); sm.set_array([])
cb=fig.colorbar(sm,ax=ax,orientation="vertical",fraction=0.022,pad=0.01)
cb.set_label(r"Dose rate $D$ ($\mu$Sv h$^{-1}$)",fontsize=FS_LABEL-2)
cb.ax.tick_params(labelsize=FS_TICK)
ax.text(0.0,1.03,"(a)",transform=ax.transAxes,fontsize=FS_PANEL,fontweight="bold",ha="left")

# ===================== Panel (b): dose vs Rc (key) =====================
ax2=fig.add_subplot(gs[1])
order=np.argsort(Rc)
ax2.plot(Rc[order],D[order],"-",color="#B0A48E",lw=2.0,zorder=1)
ax2.scatter(Rc,D,s=300,c=colors,edgecolor="#3A3230",linewidth=1.5,zorder=3)
for r in rows:
    i,rc,d,reg,la,lo=r
    if i==9:                    # Hong Kong: centered directly below the marker
        ax2.annotate(f"{i}. {reg}",(rc,d),xytext=(0,-11),textcoords="offset points",
                     fontsize=FS_ANNO-1,color="#2A2320",ha="center",va="top")
    elif i==10:                 # Philippines: to the right of marker, slightly above
        ax2.annotate(f"{i}. {reg}",(rc,d),xytext=(9,8),textcoords="offset points",
                     fontsize=FS_ANNO-1,color="#2A2320",ha="left",va="bottom")
    elif i==11:                 # Singapore: to the right of marker
        ax2.annotate(f"{i}. {reg}",(rc,d),xytext=(10,0),textcoords="offset points",
                     fontsize=FS_ANNO-1,color="#2A2320",ha="left",va="center")
    elif i==2:                  # Sub-Arctic: slightly lowered for equal gaps to labels 1 and 3
        ax2.annotate(f"{i}. {reg}",(rc,d),xytext=(8,1),textcoords="offset points",
                     fontsize=FS_ANNO-1,color="#2A2320",ha="left",va="bottom")
    else:                       # above-right
        ax2.annotate(f"{i}. {reg}",(rc,d),xytext=(7,7),textcoords="offset points",
                     fontsize=FS_ANNO-1,color="#2A2320",ha="left",va="bottom")
ax2.set_xlabel(r"Geomagnetic cutoff rigidity $R_c$ (GV)",fontsize=FS_LABEL)
ax2.set_ylabel(r"$D$ ($\mu$Sv h$^{-1}$)",fontsize=FS_LABEL)
ax2.tick_params(labelsize=FS_TICK)
ax2.set_xlim(-0.8,22.5); ax2.set_ylim(0.6,4.35)
ax2.grid(True,color=GRID,alpha=0.6,linewidth=0.6)
ax2.text(0.0,1.06,"(b)",transform=ax2.transAxes,fontsize=FS_PANEL,fontweight="bold",ha="left")
# longitude-invariance note (white box), corner
ax2.text(0.985,0.93,"Dose is governed to first order by latitude ($R_c$);\n"
         "variation with longitude at fixed $R_c$ is second-order.",
         transform=ax2.transAxes,fontsize=FS_ANNO-1,ha="right",va="top",
         bbox=dict(boxstyle="round,pad=0.4",facecolor="white",edgecolor="#B0A48E"))

fig.savefig("/sessions/serene-compassionate-davinci/mnt/outputs/FigR1_global_dose_map.png",
            dpi=200,bbox_inches="tight",facecolor="white")
print("saved FigR1_global_dose_map.png")
