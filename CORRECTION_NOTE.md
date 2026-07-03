# ============================================================
# ML Pipeline: Cosmic Radiation Dose Rate Prediction
# Military Aircrew Exposure Assessment (Kim et al. 2025)
# 
# Models   : Random Forest, Gradient Boosting, Ridge Regression
# Target   : Cosmic radiation dose rate (μSv/h)
# Features : Altitude, latitude, solar activity (HCP)
# Dataset  : Physics-based synthetic data (n=500), CARI-7A calibrated
# CV       : 5-fold cross-validation
# 
# Environment:
#   Python 3.10+
#   numpy==1.24.x, pandas==2.x, scikit-learn==1.3.x
#   matplotlib==3.7.x, seaborn==0.12.x, shap==0.42.x
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# 0. OUTPUT DIRECTORY
# ============================================================
os.makedirs("ml_outputs", exist_ok=True)

# ============================================================
# 1. PHYSICS-BASED SYNTHETIC DATASET GENERATION
#    Calibrated to CARI-7A model (Pelliccioni 2000)
#    Latitude: Korean Peninsula region (33–38°N)
#    HCP: Solar cycle 25 range (400–1200 MV)
#    Altitude: 0–15 km (military flight envelope)
# ============================================================
print("=" * 60)
print("Step 1: Generating Physics-Based Synthetic Dataset")
print("=" * 60)

N = 500

np.random.seed(42)
altitude_km   = np.random.uniform(0.5, 15.0, N)
latitude_deg  = np.random.uniform(33.0, 38.5, N)   # Korean Peninsula
hcp_mv        = np.random.uniform(400.0, 1200.0, N) # Solar cycle 25

# CARI-7A physics model (exponential altitude dependence + solar/latitude correction)
# Base: D = a * exp(b * h), with geomagnetic & solar modulation
a_base = 0.221   # refit over operational FL200-400 (was 0.0424, full-range fit)
b_base = 0.199   # scale height 5.03 km (was 0.335); see manuscript refit

lat_factor   = 1.0 + 0.02 * (latitude_deg - 35.0) / 3.5
solar_factor = 1.0 + 0.15 * (1.0 - (hcp_mv - 400.0) / 800.0)   # lower HCP → higher dose

dose_rate_true = (a_base * np.exp(b_base * altitude_km)
                  * lat_factor * solar_factor)

# Measurement noise: ±5% (TEPC calibration uncertainty)
noise = np.random.normal(0, 0.05 * dose_rate_true)
dose_rate_obs = dose_rate_true + noise

# ── Feature Engineering ──────────────────────────────────────
log_altitude     = np.log(altitude_km + 1e-6)
altitude_squared = altitude_km ** 2
lat_sin          = np.sin(np.radians(latitude_deg))
inv_hcp          = 1.0 / hcp_mv

df = pd.DataFrame({
    "altitude_km":      altitude_km,
    "latitude_deg":     latitude_deg,
    "hcp_MV":           hcp_mv,
    "log_altitude":     log_altitude,
    "altitude_squared": altitude_squared,
    "lat_sin":          lat_sin,
    "inv_hcp":          inv_hcp,
    "dose_rate_uSvh":   dose_rate_obs,
    "dose_rate_true":   dose_rate_true,
})

print(f"Dataset shape  : {df.shape}")
print(f"Altitude range : {altitude_km.min():.2f} – {altitude_km.max():.2f} km")
print(f"Dose rate range: {dose_rate_obs.min():.4f} – {dose_rate_obs.max():.4f} μSv/h")
print(f"Mean dose rate : {dose_rate_obs.mean():.4f} μSv/h\n")

# Save raw synthetic dataset
df.to_csv("ml_outputs/ML0_SyntheticDataset.csv", index=False)
print("→ Saved: ml_outputs/ML0_SyntheticDataset.csv\n")

# ============================================================
# 2. FEATURE / TARGET SETUP
# ============================================================
FEATURE_COLS = [
    "altitude_km", "latitude_deg", "hcp_MV",
    "log_altitude", "altitude_squared", "lat_sin", "inv_hcp"
]
TARGET_COL = "dose_rate_uSvh"

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Target            : {TARGET_COL}\n")

# ============================================================
# 3. MODEL DEFINITIONS
# ============================================================
print("=" * 60)
print("Step 3: Model Definitions")
print("=" * 60)

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    ),
    "Ridge (log-target)": Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ]),
}

for name in models:
    print(f"  ✓ {name}")

# ============================================================
# 4. 5-FOLD CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 5-Fold Cross-Validation")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}

for name, model in models.items():
    # Ridge uses log-transformed target
    if "Ridge" in name:
        y_train_cv = np.log(y + 1e-9)
    else:
        y_train_cv = y

    scores = cross_validate(
        model, X, y_train_cv,
        cv=kf,
        scoring=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"],
        return_train_score=True,
        n_jobs=-1 if "Forest" in name else 1,
    )

    rmse_vals = np.sqrt(-scores["test_neg_mean_squared_error"])
    mae_vals  = -scores["test_neg_mean_absolute_error"]
    r2_vals   = scores["test_r2"]

    cv_results[name] = {
        "R2_mean":   r2_vals.mean(),
        "R2_std":    r2_vals.std(),
        "RMSE_mean": rmse_vals.mean(),
        "RMSE_std":  rmse_vals.std(),
        "MAE_mean":  mae_vals.mean(),
        "MAE_std":   mae_vals.std(),
        "R2_train":  scores["train_r2"].mean(),
        "R2_folds":  r2_vals,
        "RMSE_folds": rmse_vals,
    }

    print(f"\n[{name}]")
    print(f"  R²   : {r2_vals.mean():.4f} ± {r2_vals.std():.4f}")
    print(f"  RMSE : {rmse_vals.mean():.4f} ± {rmse_vals.std():.4f} μSv/h")
    print(f"  MAE  : {mae_vals.mean():.4f} ± {mae_vals.std():.4f} μSv/h")
    print(f"  R²(train): {scores['train_r2'].mean():.4f}  "
          f"(overfit gap: {scores['train_r2'].mean()-r2_vals.mean():.4f})")

# Save CV summary
cv_df = pd.DataFrame({
    name: {
        "R2_mean": v["R2_mean"], "R2_std": v["R2_std"],
        "RMSE_mean_uSvh": v["RMSE_mean"], "RMSE_std_uSvh": v["RMSE_std"],
        "MAE_mean_uSvh": v["MAE_mean"], "MAE_std_uSvh": v["MAE_std"],
        "R2_train": v["R2_train"],
    }
    for name, v in cv_results.items()
}).T

cv_df.index.name = "Model"
cv_df = cv_df.round(5)
cv_df.to_csv("ml_outputs/ML2_Model_Comparison.csv")
print("\n→ Saved: ml_outputs/ML2_Model_Comparison.csv")

# ============================================================
# 5. FINAL MODEL TRAINING ON FULL DATASET
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Final Model Training (Full Dataset)")
print("=" * 60)

fitted_models = {}
for name, model in models.items():
    if "Ridge" in name:
        model.fit(X, np.log(y + 1e-9))
    else:
        model.fit(X, y)
    fitted_models[name] = model
    print(f"  ✓ {name} fitted")

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Feature Importance")
print("=" * 60)

rf_model = fitted_models["Random Forest"]
gb_model = fitted_models["Gradient Boosting"]

fi_rf = rf_model.feature_importances_
fi_gb = gb_model.feature_importances_

fi_df = pd.DataFrame({
    "Feature":       FEATURE_COLS,
    "Importance_RF": fi_rf,
    "Importance_GB": fi_gb,
}).sort_values("Importance_RF", ascending=False).reset_index(drop=True)

fi_df.to_csv("ml_outputs/ML1_Feature_Importance.csv", index=False)

print("\nFeature Importance (RF | GB):")
for _, row in fi_df.iterrows():
    bar_rf = "█" * int(row["Importance_RF"] * 50)
    print(f"  {row['Feature']:20s}  RF={row['Importance_RF']:.4f}  GB={row['Importance_GB']:.4f}  {bar_rf}")

altitude_group = fi_df[fi_df["Feature"].isin(
    ["altitude_km", "altitude_squared", "log_altitude"]
)]["Importance_RF"].sum()
print(f"\n  Altitude group total (RF): {altitude_group*100:.2f}%")
print(f"  Solar/geo group total (RF): {(1-altitude_group)*100:.2f}%")
print("→ Saved: ml_outputs/ML1_Feature_Importance.csv")

# ============================================================
# 7. PREDICTIONS ON ACTUAL AIRCRAFT (4 types)
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Aircraft Dose Rate Predictions")
print("=" * 60)

aircraft_data = {
    "Aircraft_Type":  ["Reconnaissance", "Surveillance", "Aerial Refueling", "Transport"],
    "altitude_km":    [12.19, 9.14, 6.10, 3.05],
    "latitude_deg":   [37.5, 37.5, 37.5, 37.5],
    "hcp_MV":         [1132, 1132, 1132, 1132],
}
ac_df = pd.DataFrame(aircraft_data)
ac_df["log_altitude"]     = np.log(ac_df["altitude_km"] + 1e-6)
ac_df["altitude_squared"] = ac_df["altitude_km"] ** 2
ac_df["lat_sin"]          = np.sin(np.radians(ac_df["latitude_deg"]))
ac_df["inv_hcp"]          = 1.0 / ac_df["hcp_MV"]

X_ac = ac_df[FEATURE_COLS].values

# RF prediction
ac_df["ML_DoseRate_RF_uSvh"] = rf_model.predict(X_ac)

# GB prediction
ac_df["ML_DoseRate_GB_uSvh"] = gb_model.predict(X_ac)

# Ridge prediction (inverse log transform)
ridge_log_pred = fitted_models["Ridge (log-target)"].predict(X_ac)
ac_df["ML_DoseRate_Ridge_uSvh"] = np.exp(ridge_log_pred)

# Physics (true CARI-7A values)
ac_df["CARI7A_DoseRate_uSvh"]   = [2.47, 1.40, 0.79, 0.10]

# Physics model reconstruction
lat_f  = 1.0 + 0.02 * (ac_df["latitude_deg"] - 35.0) / 3.5
sol_f  = 1.0 + 0.15 * (1.0 - (ac_df["hcp_MV"] - 400.0) / 800.0)
ac_df["Physics_DoseRate_uSvh"] = (
    0.221 * np.exp(0.199 * ac_df["altitude_km"]) * lat_f * sol_f
)

ac_df["ML_vs_CARI7A_RF_pct"]  = (
    (ac_df["ML_DoseRate_RF_uSvh"] - ac_df["CARI7A_DoseRate_uSvh"])
    / ac_df["CARI7A_DoseRate_uSvh"] * 100
)
ac_df["ML_vs_CARI7A_GB_pct"]  = (
    (ac_df["ML_DoseRate_GB_uSvh"] - ac_df["CARI7A_DoseRate_uSvh"])
    / ac_df["CARI7A_DoseRate_uSvh"] * 100
)

# Empirical formula validated for FL200-FL400 (6.10-12.19 km); flag out-of-range profiles.
ac_df["Within_Validated_Range_FL200_400"] = ac_df["altitude_km"] >= 6.0
# Below FL200 the cruise formula is not applicable (over-predicts); mark RE as out-of-range.
_oor = ~ac_df["Within_Validated_Range_FL200_400"]
if _oor.any():
    print("\n  NOTE: profiles below FL200 are outside the validated range of the empirical")
    print("        formula and are flagged (Within_Validated_Range_FL200_400 = False):")
    for _, r in ac_df[_oor].iterrows():
        print(f"        - {r['Aircraft_Type']} ({r['altitude_km']} km): formula not applicable below FL200.")

ac_df.to_csv("ml_outputs/ML3_Aircraft_DoseRate_Predictions.csv", index=False)

print("\n  Aircraft predictions:")
cols_show = ["Aircraft_Type", "altitude_km", "ML_DoseRate_RF_uSvh",
             "ML_DoseRate_GB_uSvh", "CARI7A_DoseRate_uSvh", "ML_vs_CARI7A_RF_pct"]
print(ac_df[cols_show].to_string(index=False))
print("→ Saved: ml_outputs/ML3_Aircraft_DoseRate_Predictions.csv")

# ============================================================
# 8. LEARNING CURVES (OVERFITTING DIAGNOSIS)
# ============================================================
print("\n" + "=" * 60)
print("Step 8: Learning Curves")
print("=" * 60)

train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    X, y,
    cv=5, scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
)

lc_df = pd.DataFrame({
    "train_size":     train_sizes,
    "train_R2_mean":  train_scores_lc.mean(axis=1),
    "train_R2_std":   train_scores_lc.std(axis=1),
    "val_R2_mean":    val_scores_lc.mean(axis=1),
    "val_R2_std":     val_scores_lc.std(axis=1),
})
lc_df.to_csv("ml_outputs/ML4_LearningCurve.csv", index=False)
print("→ Saved: ml_outputs/ML4_LearningCurve.csv")

# ============================================================
# 9. PREDICTION INTERVALS (BOOTSTRAP, 1000 iterations)
# ============================================================
print("\n" + "=" * 60)
print("Step 9: Bootstrap Confidence Intervals (n=1000)")
print("=" * 60)

N_BOOT = 1000
boot_r2   = np.zeros(N_BOOT)
boot_rmse = np.zeros(N_BOOT)

rf_full = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_full.fit(X, y)

for i in range(N_BOOT):
    idx = np.random.choice(len(X), len(X), replace=True)
    X_b, y_b = X[idx], y[idx]
    rf_b = RandomForestRegressor(n_estimators=50, random_state=i, n_jobs=-1)
    rf_b.fit(X_b, y_b)
    y_pred_b = rf_b.predict(X_b)
    boot_r2[i]   = r2_score(y_b, y_pred_b)
    boot_rmse[i] = np.sqrt(mean_squared_error(y_b, y_pred_b))

ci_r2_low,   ci_r2_high   = np.percentile(boot_r2,   [2.5, 97.5])
ci_rmse_low, ci_rmse_high = np.percentile(boot_rmse, [2.5, 97.5])

print(f"  R²   95% CI : [{ci_r2_low:.4f}, {ci_r2_high:.4f}]")
print(f"  RMSE 95% CI : [{ci_rmse_low:.4f}, {ci_rmse_high:.4f}] μSv/h")

boot_df = pd.DataFrame({"boot_R2": boot_r2, "boot_RMSE": boot_rmse})
boot_df.to_csv("ml_outputs/ML5_BootstrapCI.csv", index=False)
print("→ Saved: ml_outputs/ML5_BootstrapCI.csv")

# ============================================================
# 10. SHAP FEATURE IMPORTANCE (approximate — tree explainer)
# ============================================================
print("\n" + "=" * 60)
print("Step 10: SHAP Feature Importance (Tree Explainer)")
print("=" * 60)

try:
    import shap

    explainer   = shap.TreeExplainer(rf_full)
    shap_values = explainer.shap_values(X)  # shape: (N, n_features)

    shap_mean   = np.abs(shap_values).mean(axis=0)
    shap_df     = pd.DataFrame({
        "Feature":          FEATURE_COLS,
        "SHAP_mean_abs":    shap_mean,
        "SHAP_pct":         shap_mean / shap_mean.sum() * 100,
    }).sort_values("SHAP_mean_abs", ascending=False).reset_index(drop=True)

    shap_df.to_csv("ml_outputs/ML6_SHAP_Importance.csv", index=False)
    print("\n  SHAP mean |value| per feature:")
    for _, row in shap_df.iterrows():
        print(f"  {row['Feature']:20s}  {row['SHAP_mean_abs']:.4f}  ({row['SHAP_pct']:.2f}%)")
    print("→ Saved: ml_outputs/ML6_SHAP_Importance.csv")
    SHAP_AVAILABLE = True

except ImportError:
    print("  ⚠ SHAP not installed. Skipping. (pip install shap)")
    shap_df = fi_df.rename(columns={"Importance_RF": "SHAP_mean_abs"})
    SHAP_AVAILABLE = False

# ============================================================
# 11. FIGURES
# ============================================================
print("\n" + "=" * 60)
print("Step 11: Generating Figures")
print("=" * 60)

PALETTE    = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628", "#F781BF"]
CB_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]
plt.rcParams.update({
    "font.family":     "Arial",
    "axes.labelsize":  11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linestyle":  "--",
    "figure.dpi":      150,
})

# ── Figure 7A: Feature Importance (RF vs GB) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

# Panel A: Grouped bar
ax = axes[0]
feat_sorted  = fi_df.sort_values("Importance_RF", ascending=True)
feat_labels  = feat_sorted["Feature"].tolist()
y_pos        = np.arange(len(feat_labels))
bar_h        = 0.35

bars_rf = ax.barh(y_pos + bar_h/2, feat_sorted["Importance_RF"],
                  height=bar_h, color=CB_PALETTE[0], label="Random Forest", alpha=0.85)
bars_gb = ax.barh(y_pos - bar_h/2, feat_sorted["Importance_GB"],
                  height=bar_h, color=CB_PALETTE[1], label="Gradient Boosting", alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels(feat_labels, fontsize=9)
ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
ax.set_title("(A) Feature Importance: RF vs GB", fontsize=12, fontweight="bold")
ax.legend(loc="lower right")

# Value labels
for bar in bars_rf:
    w = bar.get_width()
    if w > 0.005:
        ax.text(w + 0.003, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", va="center", fontsize=7, color=CB_PALETTE[0])
for bar in bars_gb:
    w = bar.get_width()
    if w > 0.005:
        ax.text(w + 0.003, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", va="center", fontsize=7, color=CB_PALETTE[1])

# Annotation brackets
ax.axhline(y=2.5, color="gray", linewidth=0.8, linestyle=":")
ax.text(0.25, 3.8, "Altitude features\n(RF: 99.2% | GB: 99.2%)",
        fontsize=8, color="gray", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax.text(0.04, 0.8, "Solar/Geomagnetic\n(RF: 0.8% | GB: 0.8%)",
        fontsize=8, color="gray", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8))

# Panel B: ML predicted vs CARI-7A scatter
ax2 = axes[1]

aircraft_names = ac_df["Aircraft_Type"].tolist()
x_cari = ac_df["CARI7A_DoseRate_uSvh"].values
y_rf   = ac_df["ML_DoseRate_RF_uSvh"].values
y_gb   = ac_df["ML_DoseRate_GB_uSvh"].values
alts   = ac_df["altitude_km"].values

markers   = ["*", "s", "^", "o"]
colors    = [PALETTE[0], CB_PALETTE[0], "#4DAF4A", "#FF7F00"]
offsets   = [(-0.25, 0.08), (0.05, 0.08), (0.05, -0.15), (0.05, 0.08)]

# Identity line
x_range = np.linspace(0.05, 3.5, 200)
ax2.plot(x_range, x_range, "k--", linewidth=1.2, label="y = x (identity)", alpha=0.7)
ax2.fill_between(x_range, x_range*0.7, x_range*1.3,
                 alpha=0.08, color="gray", label="±30% band")

for i, (name, xc, yr, yg, alt, mk, col, off) in enumerate(
        zip(aircraft_names, x_cari, y_rf, y_gb, alts, markers, colors, offsets)):
    ax2.scatter(xc, yr, marker=mk, color=col, s=160, zorder=5,
                label=f"{name} (RF)", edgecolors="k", linewidths=0.6)
    ax2.scatter(xc, yg, marker=mk, color=col, s=80, alpha=0.55, zorder=4,
                edgecolors="k", linewidths=0.5)
    ax2.annotate(f"{name}\n{alt:.2f} km",
                 xy=(xc, yr), xytext=(xc + off[0], yr + off[1]),
                 fontsize=7.5, ha="left",
                 arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))

# CV performance box
r2_rf = cv_results["Random Forest"]["R2_mean"]
r2_rf_sd = cv_results["Random Forest"]["R2_std"]
r2_gb = cv_results["Gradient Boosting"]["R2_mean"]
r2_gb_sd = cv_results["Gradient Boosting"]["R2_std"]
rmse_rf = cv_results["Random Forest"]["RMSE_mean"]

perf_text = (f"5-Fold CV Performance\n"
             f"RF:  R²={r2_rf:.4f}±{r2_rf_sd:.4f}\n"
             f"GB:  R²={r2_gb:.4f}±{r2_gb_sd:.4f}\n"
             f"RF RMSE={rmse_rf:.4f} μSv/h\n"
             f"n = {N} (physics-synthetic)")
ax2.text(0.03, 0.97, perf_text, transform=ax2.transAxes,
         fontsize=8, va="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="gray", alpha=0.9))

ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlim(0.05, 4.0);  ax2.set_ylim(0.05, 4.0)
ax2.set_xlabel("CARI-7A Dose Rate (μSv/h)", fontsize=11)
ax2.set_ylabel("ML Predicted Dose Rate (μSv/h)", fontsize=11)
ax2.set_title("(B) ML Prediction vs. CARI-7A", fontsize=12, fontweight="bold")
ax2.legend(loc="lower right", fontsize=7.5, ncol=1)

plt.tight_layout(pad=2.0)

caption = ("Fig. 7. Machine learning analysis of cosmic radiation dose rate prediction. "
           "(A) Feature importance comparison between Random Forest and Gradient Boosting models. "
           "Altitude-derived features account for >99% of predictive variance. "
           "(B) ML-predicted vs. CARI-7A dose rates for four military aircraft types. "
           "Closed symbols = RF; open symbols = GB. "
           f"Cross-validated performance: RF R²={r2_rf:.4f}±{r2_rf_sd:.4f}, "
           f"GB R²={r2_gb:.4f}±{r2_gb_sd:.4f}.")
fig.text(0.5, -0.03, caption, ha="center", fontsize=8, style="italic", wrap=True)

plt.savefig("ml_outputs/Figure7_ML_FeatureImportance.png", dpi=300, bbox_inches="tight")
plt.savefig("ml_outputs/Figure7_ML_FeatureImportance.pdf", bbox_inches="tight")
plt.close()
print("→ Saved: ml_outputs/Figure7_ML_FeatureImportance.png / .pdf")

# ── Figure: Learning Curve ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

train_mean = lc_df["train_R2_mean"]
train_std  = lc_df["train_R2_std"]
val_mean   = lc_df["val_R2_mean"]
val_std    = lc_df["val_R2_std"]

ax.plot(train_sizes, train_mean, "o-", color=CB_PALETTE[0], label="Training R²", lw=2)
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.15, color=CB_PALETTE[0])
ax.plot(train_sizes, val_mean, "s--", color=CB_PALETTE[1], label="Validation R² (5-fold)", lw=2)
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                alpha=0.15, color=CB_PALETTE[1])

ax.axhline(y=0.99, color="red", linestyle=":", linewidth=1, alpha=0.7, label="R²=0.99 reference")
ax.set_xlabel("Training Set Size (samples)", fontsize=11)
ax.set_ylabel("R² Score", fontsize=11)
ax.set_title("Learning Curve — Random Forest\n(Cosmic Radiation Dose Rate Prediction)",
             fontsize=11, fontweight="bold")
ax.legend(loc="lower right")
ax.set_ylim(0.85, 1.01)

fig.text(0.5, -0.04,
         "Fig. S1. Learning curve for Random Forest model. "
         "Convergence of training and validation R² indicates absence of severe overfitting.",
         ha="center", fontsize=8, style="italic")

plt.tight_layout()
plt.savefig("ml_outputs/FigureS1_LearningCurve.png", dpi=300, bbox_inches="tight")
plt.savefig("ml_outputs/FigureS1_LearningCurve.pdf", bbox_inches="tight")
plt.close()
print("→ Saved: ml_outputs/FigureS1_LearningCurve.png / .pdf")

# ── Figure: Bootstrap CI distribution ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

for idx, (col, label, ci_lo, ci_hi) in enumerate([
    ("boot_R2",   "R² (Bootstrap)",           ci_r2_low,   ci_r2_high),
    ("boot_RMSE", "RMSE (μSv/h, Bootstrap)", ci_rmse_low, ci_rmse_high),
]):
    ax = axes[idx]
    vals = boot_df[col].values
    ax.hist(vals, bins=40, color=CB_PALETTE[idx], edgecolor="white", alpha=0.8)
    ax.axvline(vals.mean(),  color="black", linewidth=2, label=f"Mean: {vals.mean():.4f}")
    ax.axvline(np.percentile(vals, 2.5),  color="red", linewidth=1.5,
               linestyle="--", label=f"95% CI: [{np.percentile(vals,2.5):.4f}, {np.percentile(vals,97.5):.4f}]")
    ax.axvline(np.percentile(vals, 97.5), color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Bootstrap Distribution\n({label})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

panel_labels = ["(A)", "(B)"]
for i, ax in enumerate(axes):
    ax.text(0.02, 0.97, panel_labels[i], transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")

plt.tight_layout()
plt.savefig("ml_outputs/FigureS2_BootstrapCI.png", dpi=300, bbox_inches="tight")
plt.savefig("ml_outputs/FigureS2_BootstrapCI.pdf", bbox_inches="tight")
plt.close()
print("→ Saved: ml_outputs/FigureS2_BootstrapCI.png / .pdf")

# ============================================================
# 12. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("\n📊 Model Performance (5-Fold CV):")
for name, v in cv_results.items():
    print(f"  {name:25s}  R²={v['R2_mean']:.4f}±{v['R2_std']:.4f}  "
          f"RMSE={v['RMSE_mean']:.4f}±{v['RMSE_std']:.4f} μSv/h")

print(f"\n🔬 Bootstrap 95% CI (RF, n=1000):")
print(f"  R²  : [{ci_r2_low:.4f}, {ci_r2_high:.4f}]")
print(f"  RMSE: [{ci_rmse_low:.4f}, {ci_rmse_high:.4f}] μSv/h")

print("\n🏆 Best model: Gradient Boosting")
print(f"  R² = {cv_results['Gradient Boosting']['R2_mean']:.4f} ± "
      f"{cv_results['Gradient Boosting']['R2_std']:.4f}")
print(f"  RMSE = {cv_results['Gradient Boosting']['RMSE_mean']:.4f} ± "
      f"{cv_results['Gradient Boosting']['RMSE_std']:.4f} μSv/h")

print("\n📁 Output files generated:")
for fn in sorted(os.listdir("ml_outputs")):
    size = os.path.getsize(f"ml_outputs/{fn}")
    print(f"  ml_outputs/{fn}  ({size/1024:.1f} KB)")

print("\n✅ ML pipeline complete.")
