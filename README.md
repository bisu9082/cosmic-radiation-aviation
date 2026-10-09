# Cosmic-radiation-aviation — minimum-data aircrew dose screening (CARI-7)

Reproducible pipeline for the manuscript

> **A minimum-data altitude–latitude formula for rapid screening of aircrew
> cosmic-radiation dose, derived from CARI-7**

All dose values are **real CARI-7 (v4.1.3) executions** (ICRP Publication 103
effective dose, 2004 ISO/Nymmik GCR spectrum, monthly-average mode). No
synthetic or model-fitted dose values are used.

> **v2.0 note (ML tier removed).** Earlier versions of this repository held a
> machine-learning pipeline whose training labels were generated from the dose
> formula itself (circular by construction). That tier has been removed. The
> present version derives the screen directly from CARI-7 reference runs and
> validates it out-of-sample against held-out CARI-7 nodes and full-route CARI-7
> doses. Prior ML artefacts remain in the git history.

## Repository layout

```
code/        analysis and figure scripts (Python 3, numpy/scipy/matplotlib)
  master.py            altitude log-quadratic fit + latitude saturation fit
                       + bootstrap (B=10,000) + annual doses + crossover Rc
  analyze_cari7.py     parse CARI-7 .ANS, altitude/latitude fits
  sep.py               altitude–rigidity separability test (polar vs equator)
  routecmp.py          screen vs CARI-7 full-route dose comparison (5 routes)
  figgen.py            Figures 1, 2, 3 (vector EPS)
  altq.py, table1.py   altitude-fit helpers / Table 1 values
data/
  raw_results_cari7.csv        all CARI-7 (Rc, altitude, HCP, dose) records
  cari7_outputs/               raw CARI-7 program I/O (inputs + .ANS/.OUT)
    KUNODES.{LOC,ANS}  altitude profile + 11 latitude nodes + 3 held-out
    KUSMIN.ANS         solar-minimum (2020, HCP=275 MV) profile
    KUSEP.ANS          polar + equatorial altitude profiles (separability)
    KUROUTES.{BIG,OUT} five real great-circle routes (full-flight doses)
    KUMID.ANS          route great-circle-midpoint rigidities
figures/     Figure1/3/6 as EPS (vector) and PNG
manuscript/  radioprotection_main.tex, myref.bib
```

## Key results (all from CARI-7)

- **Altitude** (FL200–FL400, Incheon, HCP=1141 MV):
  `ln D = -0.0232 h_km^2 + 0.701 h_km - 4.172`, MARE 0.10% (1.05% over FL100–FL400).
- **Latitude** (FL300, 11 nodes, Rc 0–17.2 GV): saturation
  `f_lat(Rc) = (1 + a·Rc_ref^n)/(1 + a·Rc^n)`, a=0.024, n=1.52; in-sample MARE
  3.4%, held-out MARE 1.9% (Munich/Rome/Tokyo).
- **Separability**: altitude shape invariant to ≤1% for Rc≳4 GV; ~5–9% toward the pole.
- **Solar cycle**: +15% full-cycle (HCP 1141→275 MV), altitude-independent.
- **Routes**: on five real routes the hand formula reproduces CARI-7 full-flight
  doses to within ±8% (long-haul) and matches its 1 mSv/yr flag on all five.

## Reproduce

```bash
pip install numpy scipy matplotlib
cd code
python master.py     # prints fits, bootstrap CIs, annual doses, crossover Rc
python sep.py        # separability errors
python routecmp.py   # route screen vs CARI-7
python figgen.py     # regenerates figures
```

CARI-7 itself (FAA CAMI) is required only to regenerate the raw `.ANS`/`.OUT`
files in `data/cari7_outputs/`; those are provided so the analysis runs without it.

## License

MIT (see LICENSE).
