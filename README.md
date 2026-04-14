# Laser Cutting Intelligence — Surface Roughness Prediction via Ridge Regression

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/Ridge_Stable_Process_Modeling/blob/main/10_Ridge_Stable_Process_Modeling.ipynb)

> *"When your process variables move together, your model needs a steadying hand — that's what Ridge does."*

---

## 🎯 Business Problem

In precision laser cutting, surface roughness is the primary quality gate. A part with Ra above specification must be deburred, polished, or scrapped — none of which can be recovered in high-mix, low-batch environments. The challenge is not a lack of process data; modern CNC laser systems log dozens of parameters at every cut. The challenge is that the most important process variables are **structurally correlated**: laser power, cutting speed, and assist gas pressure are not set independently. An operator who increases power typically increases speed and gas flow to match — because that is correct engineering practice.

This creates multicollinearity that inflates OLS coefficient estimates and makes them unreliable for process guidance. A model that tells the engineer the wrong direction on a single lever — because its coefficient is noise-amplified — is worse than no model.

**Ridge regression resolves this without discarding any variable.** The L2 penalty shrinks all coefficients proportionally, distributing shared explanatory power across correlated variables rather than arbitrarily amplifying one. The result is a stable, physically interpretable model that can actually guide recipe decisions — before the cut is made.

---

## 📊 Dataset

- **>1,500 cut records** from a CNC laser system controller, complemented by surface quality measurements from the post-cut CMM inspection station
- **Target:** `surface_roughness_ra_um` — Ra surface roughness (µm, continuous)
- **Range:** 0.91 – 5.70 µm  |  **Mean:** 3.18 µm  |  **Spec:** Ra ≤ 3.2 µm  |  **In-spec:** 52.2% of cuts
- **Material mix:** Carbon steel (Ra mean 3.48 µm) · Aluminium (Ra mean 2.86 µm)

| Column | Type | Description |
|---|---|---|
| `laser_power_w` | float | Laser beam power (W) |
| `cutting_speed_mm_s` | float | Torch travel speed (mm/s) |
| `assist_gas_flow_l_min` | float | Assist gas flow rate (L/min) |
| `focal_offset_mm` | float | Focal point offset from surface (mm; 0 = ideal) |
| `material_thickness_mm` | float | Sheet thickness (mm) |
| `material_type` | int | 0 = Carbon steel · 1 = Aluminium |
| `oxygen_pct` | float | Oxygen content in assist gas (%) |
| `shop_temp_c` | float | Ambient temperature in cutting bay (°C) |
| `surface_roughness_ra_um` | float | **Target** — Ra surface roughness (µm) |

### Data Origin (Real-World Perspective)

| Variable(s) | Source System | Notes |
|---|---|---|
| `laser_power_w`, `cutting_speed_mm_s` | CNC Laser Controller / G-Code Log | Program parameters from the cutting recipe file, logged at job start |
| `assist_gas_flow_l_min`, `oxygen_pct` | Gas Console / Flow Controller | Shielding/assist gas parameters — set at job setup or material changeover |
| `focal_offset_mm` | CNC Laser Controller | Focal offset setting from the cutting head configuration, per recipe |
| `material_thickness_mm` | ERP / Production Order | Nominal sheet thickness from the job traveller or material spec |
| `material_type` | ERP / Material Master | Binary flag (steel/aluminium) from the job setup — recorded at job start |
| `shop_temp_c` | Environmental Sensor | Bay-level ambient temperature at cut time |
| `surface_roughness_ra_um` | Post-Cut CMM / Profilometer | **TARGET** — Ra measurement taken at the cut edge after the job completes |

> In real-world operations, joining this dataset requires connecting the CNC job log (parameters), the ERP system (material spec), the environmental data historian (temperature), and the quality inspection station (Ra measurement) on a common job ID and timestamp. The cut parameter log and the Ra measurement are rarely in the same system.

---

## 🤖 Model

**Algorithm:** Ridge Regression (L2 regularisation) — `sklearn.linear_model.Ridge` + `RidgeCV`

The collinear triplet (`laser_power_w`, `cutting_speed_mm_s`, `assist_gas_flow_l_min`) has a Pearson correlation of r = 0.907 between power and speed alone — and VIF scores reaching 7.6 for the power variable. When features are this correlated, OLS spreads the shared explanatory variance across coefficients in a numerically unstable way: small dataset changes can flip a coefficient's sign or magnitude dramatically.

Ridge adds an L2 penalty to the OLS loss:

$$\text{Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p}\beta_j^2$$

Unlike Lasso (Project 09), the L2 norm never reaches exactly zero — it asymptotes toward it. Ridge keeps all 8 variables and assigns each a stable, conservative partial effect. The regularisation path — coefficient trajectories across 80 alpha values — makes this shrinkage transparent and verifiable.

**Why not Lasso here?** All 8 laser parameters have physical justification for inclusion. Lasso's zero-or-keep decision would eliminate variables that are genuinely active but temporarily suppressed by collinearity. Ridge distributes their shared contribution proportionally — which is what the physics demands.

**Alpha tuning:** `RidgeCV` evaluated 80 candidates on a log scale [0.001, 1000], using 5-fold cross-validation. Selected alpha = **4.422**.

**Preprocessing:** `StandardScaler` on all 8 features — mandatory for Ridge, same as Lasso.  
**Split:** 80/20 train/test, `random_state=42`.

---

## 📈 Key Results

| Metric | Ridge | OLS | Operational Meaning |
|---|---|---|---|
| **R²** | **0.722** | 0.722 | 72.2% of Ra variance explained — both models match in accuracy |
| **RMSE** | **0.394 µm** | 0.394 µm | 12.3% of the 3.2 µm spec limit — usable for go/no-go scheduling |
| **MAE** | **0.305 µm** | — | Median absolute miss — adequate for recipe adjustment decisions |
| **Alpha (CV)** | **4.422** | — | RidgeCV 5-fold selected |
| **Train / Test** | **1,200 / 300 cuts** | — | 80/20 split, `random_state=42` |

Ridge and OLS deliver identical test R². This is expected and correct. **Ridge's advantage is not higher R² — it is coefficient stability under multicollinearity**, which makes the model's guidance trustworthy across the operating range rather than numerically brittle.

---

## 🔍 Ridge Coefficients — Stable Estimates Under Collinearity

All coefficients are standardised (per-σ units), making them comparable across features measured in different physical units.

| Feature | Ridge Coef | Direction | Engineering Interpretation |
|---|---|---|---|
| `material_thickness_mm` | +0.385 | ↑ Increases Ra | Dominant driver — more material = more energy required; insufficient energy → rough cut |
| `material_type` | −0.304 | ↓ Reduces Ra | Aluminium's superior thermal conductivity consistently produces lower Ra than steel |
| `laser_power_w` | −0.268 | ↓ Reduces Ra | More power → cleaner melt ejection → smoother edge |
| `assist_gas_flow_l_min` | −0.190 | ↓ Reduces Ra | Gas flow assists melt removal — stable, not artificially amplified by collinearity |
| `oxygen_pct` | +0.052 | ↑ Increases Ra | Higher O₂ increases oxidation on the cut edge, mildly increasing roughness |
| `shop_temp_c` | +0.025 | ↑ Increases Ra | Ambient thermal variation — small but consistent |
| `cutting_speed_mm_s` | +0.016 | ↑ Increases Ra | Faster travel reduces energy density; Ridge assigns a modest, stable estimate |
| `focal_offset_mm` | +0.006 | ↑ Increases Ra | Defocus reduces beam energy density at the workpiece; small effect at moderate offsets |

**The collinearity story:** OLS assigns inflated, unstable estimates to the power–speed–gas triplet — the signs can flip with different training samples. Ridge moderates all three with proportional shrinkage, giving each a physically plausible partial effect the process engineer can act on.

---

## 🔧 Simulation & Scenarios

| Scenario | Configuration | Predicted Ra | Status |
|---|---|---|---|
| **A — 3 mm Aluminium, Optimised** | 3200W · 20 mm/s · 10 L/min · focal 0.0 · O₂ 8% | 1.627 µm | ✅ Pass (margin +1.573 µm) |
| **B — 12 mm Steel, Under-Powered** | 2500W · 35 mm/s · 5 L/min · focal 0.0 · O₂ 25% | 4.174 µm | ❌ Fail (outside spec) |
| **C — 12 mm Steel, Corrected** | 3800W · 18 mm/s · 10 L/min · focal 0.0 · O₂ 25% | 2.935 µm | ✅ Pass (margin +0.265 µm) |

Scenario B vs C quantifies the recipe correction for 12 mm steel: raising power by 1300W, reducing speed by 17 mm/s, and increasing gas flow by 5 L/min recovers **1.24 µm Ra** — from outside spec to within it. The correction is quantified, not guessed.

The 2D response surface (Power × Speed for 12 mm carbon steel) extends this into a full process window map — showing every power/speed combination that meets the Ra ≤ 3.2 µm specification.

---

## 🗂️ Repository Structure

```
Ridge_Stable_Process_Modeling/
├── 10_Ridge_Stable_Process_Modeling.ipynb   ← Notebook (no outputs)
├── laser_cutting_data.csv                    ← 250-row sample dataset (GitHub public)
├── README.md
└── requirements.txt
```

> 📦 **Full Project Pack** — complete dataset (1,500 records), notebook with full outputs, presentation deck (PPTX), and `app.py` Ra simulator available on [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab:** Click the badge above.

**Option 2 — Local:**
```bash
pip install -r requirements.txt
jupyter notebook 10_Ridge_Stable_Process_Modeling.ipynb
```

---

## 💡 Key Learnings

1. **Multicollinearity is not a data quality problem — it is a process reality.** Power, speed, and gas pressure move together in laser cutting because that is correct engineering practice. OLS treats this as a numerical problem to suppress; Ridge treats it as a structure to accommodate. That distinction matters for the coefficient's physical meaning.

2. **Ridge and Lasso solve different problems.** Lasso (Project 09) zeros irrelevant features when some variables genuinely carry no signal. Ridge stabilises coefficients when all features are relevant but correlated. Choosing the wrong tool doesn't break the model — it breaks the interpretability of its guidance.

3. **Matching OLS R² is the goal, not a disappointment.** Ridge achieves 0.722 vs OLS 0.722. This is the correct outcome: Ridge should not improve accuracy over OLS on the same data — it should achieve the same accuracy with more stable, trustworthy coefficients. The win is in the path, not the destination.

4. **The regularisation path is the interpretability tool.** Plotting coefficient trajectories across 80 alpha values reveals how Ridge handles the collinear triplet: power, speed, and gas shrink together smoothly, never reversing sign or exploding. This is what stable process guidance looks like, and it is only visible in the path plot.

5. **VIF above 5 is a Ridge flag, not a modelling failure.** A VIF of 7.6 on laser power means OLS cannot reliably estimate its coefficient in the presence of speed and gas. Ridge transforms this diagnostic from a warning into a design input — the higher the VIF, the more valuable the L2 regularisation.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
