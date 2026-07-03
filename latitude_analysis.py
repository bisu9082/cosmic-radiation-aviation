# Correction note — Tier-1 empirical formula refit (revision)

**What changed.** The Tier-1 empirical altitude–dose formula was refit over the
operational range FL200–FL400 (6.10–12.19 km).

- Old (full-range 0–15 km fit): `D = 0.0424 * exp(0.335 * h_km)` (scale height 2.99 km)
- New (operational-range fit):  `D = 0.221 * exp(0.199 * h_km)` (scale height 5.03 km)

**Why.** The original coefficients came from a fit over the full 0–15 km range and
did not reproduce the CARI-7 reference dose rates within the operational band
(e.g. 0.91 vs 1.40 µSv/h at FL300). The refit over FL200–FL400 reproduces the
CARI-7 references with MARE = 4.9% (1.9% for stable high-altitude cruise profiles).

**Scope.** The empirical formula is a screening approximation valid for FL200–FL400;
below FL200 it is not applicable (it over-predicts). CARI-7 (Tier 2) is the precision
reference. Profiles below FL200 are flagged in ML3 output
(`Within_Validated_Range_FL200_400 = False`).

**Physical basis.** The refitted scale height (5.0 km, e-folding 0.199 km^-1) is
consistent with independent cross-code cruise-altitude dose-rate gradients in the
EURADOS code intercomparison (~0.2 km^-1).

**Impact on ML (Tier 3).** Unchanged: regenerating the synthetic training data with
the refitted coefficients leaves CV R^2 ~ 0.99 and the altitude-dominant feature
importance (~98%) intact.

Files affected: ML_Pipeline_CosmicRadiation.py (a_base, b_base, prediction line),
README.md. latitude_analysis.py unaffected (Rc ratios).
Note: ML5_BootstrapCI.csv should be regenerated locally (full pipeline run).
