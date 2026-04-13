# Space Weather Submission — Cosmic Radiation Dose Screening

**Manuscript:** *Altitude-based empirical framework for cosmic radiation dose screening in aviation: CARI-7 derivation, machine learning analysis, and cross-validation against published measurements*

**Target journal:** *Space Weather* (AGU / Wiley, IF 3.56, Gold Open Access, APC $3,000)
**Corresponding author:** Ku Kang, Ph.D. — CBRN Defense Research Institute, Ministry of National Defense, ROK
**Date:** April 2026

---

## Project overview

This repository contains all reproducible assets for the Space Weather submission. The study derives and validates a three-tier framework for cosmic radiation dose screening at aviation altitudes under Solar Cycle 25 maximum conditions (HCP = 1,141 MV, November 2025):

- **Tier 1 — Empirical altitude formula:** `D(μSv/h) = 0.0424 × exp(0.335 × h_km)` (MARE 1.7% at ≥ 30,000 ft)
- **Tier 1.5 — Geomagnetic latitude extension:** saturation-type correction `f_lat(Rc) = (1 + a·Rc_ref^n)/(1 + a·Rc^n)` (a = 0.099, n = 1.289, R² = 0.9995)
- **Tier 2 — CARI-7 reference model** (FAA physics-based)
- **Tier 3 — Machine learning ensemble** (Random Forest, Gradient Boosting, Ridge; R² = 0.9919)

External cross-validation against published Korean Peninsula (FL300, SC24) and European (FL370–390, SC23/24 solar minimum) in-flight measurements confirms ±30% international criterion compliance (MARE 1.15% across nine cases).

---

## Folder structure

```
space_weather/
├── README.md                 ← this file
├── upload/                   ← Python source code (GitHub upload targets)
│   ├── ML_Pipeline_CosmicRadiation.py     — Tier 3 ML ensemble (RF, GB, Ridge), feature importance, bootstrap CI, learning curves
│   └── latitude_analysis.py               — Tier 1.5 latitude correction derivation and cross-validation
├── data/                     ← raw inputs and code outputs (CSV)
│   ├── ML0_SyntheticDataset.csv           — N=500, altitude 0.5–15 km, latitude 33–38.5°N, HCP 400–1200 MV
│   ├── ML1_Feature_Importance.csv         — Gini-decrease importances (RF, GB)
│   ├── ML2_Model_Comparison.csv           — R², RMSE, MAE across models
│   ├── ML3_Aircraft_DoseRate_Predictions.csv — operational altitude predictions
│   ├── ML4_LearningCurve.csv              — training/CV R² vs. N
│   ├── ML5_BootstrapCI.csv                — 95% CI from B=1000 bootstrap
│   ├── Stats1–4_*.csv                     — regression, Bland-Altman, precision, uncertainty budget
│   ├── Table1–8_*.csv                     — all manuscript tables as source data
│   ├── 00_Analysis_Summary.csv            — consolidated headline metrics
│   └── latitude_summary.json              — Tier 1.5 fitted parameters and CI
├── figure/                   ← generated figures (PNG)
│   ├── Figure7_ML_FeatureImportance.png
│   ├── FigureS1_LearningCurve.png
│   ├── FigureS2_BootstrapCI.png
│   └── figure6.png
└── latex/                    ← final submission manuscripts
    ├── main.tex
    ├── SI_supplement.tex
    ├── cover_letter.tex
    └── myref.bib
```

---

## Reproducibility

### Environment
- Python ≥ 3.10
- `numpy`, `pandas`, `scikit-learn==1.3.*`, `matplotlib`, `scipy`

### Run the ML pipeline
```bash
cd upload/
python ML_Pipeline_CosmicRadiation.py
```
Regenerates `ML0–ML5*.csv` plus `Figure7_ML_FeatureImportance.png`, `FigureS1_LearningCurve.png`, `FigureS2_BootstrapCI.png`. Random seed is fixed for deterministic output.

### Run the latitude (Tier 1.5) analysis
```bash
python latitude_analysis.py
```
Fits the saturation-type geomagnetic correction against eleven CARI-7 cutoff-rigidity nodes (0.5–17 GV) and reports bootstrapped 95% CIs.

### Key hyperparameters (ML ensemble)
- Random Forest: `n_estimators=200`, `min_samples_leaf=3`, `max_features="sqrt"`
- Gradient Boosting: `n_estimators=200`, `learning_rate=0.05`
- Train/test split: 80/20 stratified; 5-fold CV for learning curves
- Bootstrap resampling: B = 1000

---

## Data description

The synthetic training set (N = 500) samples the operational envelope of aviation cosmic radiation exposure. Targets are generated from CARI-7 reference lookups under SC25-max conditions, not a parametric surrogate, so feature importances reflect the actual CARI-7 dependency structure. External validation uses published in-flight measurements (Kim et al., 2023 — Korean Peninsula; Latocha et al., 2009 — European airspace).

---

## Citation

If you use this code or data, please cite the Space Weather manuscript (submitted, 2026). Full bibliography is in `latex/myref.bib`.

## Contact

Ku Kang, Ph.D. — `bisu9082@gmail.com`
CBRN Defense Research Institute, Ministry of National Defense, Seoul, Republic of Korea
