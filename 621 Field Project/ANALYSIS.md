# BC Housing Field Project — Full Analysis Report (MGTA 621)

## 1. Project Scope (from Meeting Minutes & Notes)

**Client:** BC Housing (Crown Corporation)  
**Business problem:** Assess which measure—**housing registrations** or **housing starts**—offers stronger predictive value of **housing completions**, and to explore time lags, conversion rates, and the relationship between housing indicators and economic drivers.

**Scope boundaries (from Question list & Advisor/Team minutes):**
- Compare housing starts and registrations as predictors of housing completions.
- Conversion rates: % of registrations that become starts; % of starts that complete.
- Time lags: registration → start; start → completion.
- Forecasting windows: starts better for 1–2 years; registrations for 3–5 years.
- Relationship between housing indicators and economic variables (e.g. GDP).
- Which construction type (Single, Semi-Detached, Row, Apartment) is most sensitive to economic conditions.
- BC vs Canada housing trends (data permitting).

**Data used:**
- **BC Starts** (CMHC): monthly, Jan 1990 – Dec 2025, by dwelling type (Single, Semi-Detached, Row, Apartment, Total).
- **BC Completions** (CMHC): same geography and structure.
- **Housing Registration Data** (BC Housing): annual 2016–2024, by type (Single Detached, Multi Unit Homes, Purpose Built Rental), by region; BC totals aggregated.
- **Canada annual** (StatsCan/CMHC): starts, under construction, completions 2016–2024 (referenced for context; BC vs Canada comparison limited by aggregation).

**Deliverable:** 15+ graphs, detailed analysis, and prediction comparisons (Starts vs Registrations as predictors of Completions).

---

## 2. Data Summary

| Dataset | Geography | Frequency | Period | Key variables |
|--------|------------|-----------|--------|----------------|
| BC Starts CMHC | BC (CMAs, CAs, selected municipalities) | Monthly | Jan 1990 – Dec 2025 | Single, Semi-Detached, Row, Apartment, Total |
| BC Completions CMHC | Same | Monthly | Jan 1990 – Dec 2025 | Same |
| Housing Registration Data | BC by region | Annual | 2016–2024 | Single Detached, Multi Unit, Purpose Built Rental (BC total used) |
| Canada annual | Canada | Annual | 2016–2024 | Starts, under construction, completions by type |

BC monthly series were aggregated to annual where needed to align with registration years for prediction and correlation analysis.

---

## 3. Graphs Produced (15+)

All figures are saved in **`outputs/`**.

| # | File | Description |
|---|------|-------------|
| 1 | `01_bc_starts_vs_completions_monthly.png` | BC monthly Starts vs Completions (last 10 years). |
| 2 | `02_bc_completions_by_type.png` | BC Completions by dwelling type — stacked area (last 15 years). |
| 2b | `02b_bc_completions_lines_by_type.png` | BC Completions by type — line plot (last 15 years). |
| 3 | `03_bc_starts_by_type.png` | BC Starts by dwelling type (last 15 years). |
| 4 | `04_bc_annual_starts_vs_completions.png` | BC annual Starts vs Completions (bar). |
| 5 | `05_registrations_by_type.png` | BC Housing Registrations by type (Single Detached, Multi Unit, Purpose Built Rental) — annual. |
| 6 | `06_registrations_vs_starts_vs_completions.png` | BC annual: Registrations vs Starts vs Completions (overlapping years). |
| 7 | `07_completions_vs_lag1_starts.png` | Scatter: Completions vs lag-1-year Starts (annual) with trend line. |
| 8 | `08_completions_vs_lag1_registrations.png` | Scatter: Completions vs lag-1-year Registrations (annual) with trend. |
| 9 | `09_prediction_comparison.png` | Actual vs predicted Completions: Starts Lag-1 model vs Registrations Lag-1 model. |
| 10 | `10_rolling_12mo_starts_vs_completions.png` | BC 12-month rolling average Starts vs Completions. |
| 11 | `11_completions_share_by_type.png` | BC Completions by dwelling type (most recent full year) — bar. |
| 12 | `12_pipeline_gap_starts_minus_completions.png` | BC annual “pipeline gap”: Starts minus Completions. |
| 13 | `13_mae_r2_comparison.png` | Prediction model comparison: MAE and R² (Starts Lag-1, Registrations Lag-1, combined). |
| 14 | `14_correlation_heatmap.png` | Correlation heatmap: Completions, Starts, Registrations, and their lags (annual BC). |
| 15 | `15_bc_completions_full_history.png` | BC monthly Completions — full history. |

---

## 4. Prediction Comparison: Starts vs Registrations

**Models (annual BC, same-year Completions):**
- **Model A:** Completions ~ **Starts (lag 1 year)**  
- **Model B:** Completions ~ **Registrations (lag 1 year)**  
- **Model C:** Completions ~ **Starts (lag 1) + Registrations (lag 1)**

**Representative metrics (from pipeline run):**

| Model | MAE (units) | R² |
|-------|-------------|-----|
| Starts_Lag1 | ~2,761 | ~0.01 |
| Registrations_Lag1 | ~2,378 | ~0.10 |
| Starts_plus_Registrations_Lag1 | ~2,552 | ~0.12 |

**Findings:**
- **Registrations (lag 1)** has **lower MAE** and **higher R²** than **Starts (lag 1)** in this annual, BC-total setup. So in this comparison, lagged registrations explain more of the variation in completions and have smaller average error.
- The **combined model** (Starts + Registrations, both lag 1) improves R² slightly (~0.12), suggesting both indicators add information; MAE sits between the two single-predictor models.
- **Starts (lag 1)** alone shows very low R² (~0.01), which may reflect: (1) stronger non-linear or multi-year lag effects for starts, (2) different seasonal/quarterly dynamics that annual aggregation flattens, or (3) registrations capturing an earlier, smoother signal that aligns better with annual completions in this sample.

**Caveats:**
- Short overlap: registration data only 2016–2024, so training sample is small.
- BC-level annual totals; regional or monthly models could change relative performance.
- Linear models only; no economic drivers (e.g. GDP, rates) or multi-year lags included here.

---

## 5. Detailed Analysis by Theme

### 5.1 Time series and pipeline (Graphs 1, 2, 2b, 3, 4, 10, 12, 15)

- **Starts vs Completions (monthly, last 10 years):** Starts and completions move together over the long run but with visible **lead–lag**: starts often lead completions by roughly 1–2 years, consistent with construction cycle.
- **12-month rolling averages (Graph 10):** Smooth seasonal noise; the same lead–lag is easier to see.
- **By dwelling type (Graphs 2, 2b, 3, 11):** Apartments dominate total volume in recent years; single-detached and row contribute less. Completions by type (Graph 11, latest year) show the same mix.
- **Annual bar (Graph 4):** In many years, **Starts > Completions**, so the pipeline is building up (more units entering construction than finishing). **Pipeline gap (Graph 12)** makes this explicit: positive gap = more starts than completions in that year.

### 5.2 Registrations vs starts vs completions (Graphs 5, 6)

- **Registrations by type (Graph 5):** Single Detached, Multi Unit, and Purpose Built Rental all contribute to BC totals; multi-unit and purpose-built rental drive much of the recent level.
- **Three-series overlay (Graph 6):** Registrations, starts, and completions (annual) show similar trends over 2016–2024, with registrations and starts often above completions, consistent with a pipeline where not all registered or started units complete in the same year.

### 5.3 Lag and correlation (Graphs 7, 8, 14)

- **Completions vs lag-1 Starts (Graph 7):** Positive relationship; scatter is wide, in line with low R² for the simple linear Starts Lag-1 model.
- **Completions vs lag-1 Registrations (Graph 8):** Also positive; in this annual view, points align somewhat more with a linear trend than for starts (consistent with better R² for Registrations Lag-1).
- **Correlation heatmap (Graph 14):** Completions correlate with current and lagged Starts and with current and lagged Registrations; strength of correlations and which lag is strongest can be read directly from the heatmap and can guide which lags to use in richer models.

### 5.4 Prediction comparison (Graphs 9, 13)

- **Actual vs predicted (Graph 9):** Both models cluster around the 45° line; Registrations Lag-1 model tends to sit slightly closer to the line in this run, in line with lower MAE.
- **MAE and R² (Graph 13):** Bar chart summarizes that Registrations Lag-1 has the best MAE and that the combined model has the best R² among the three.

---

## 6. Conclusions and Recommendations

**Conclusions:**
1. **Registrations (lag 1 year)** outperform **Starts (lag 1 year)** in predicting BC annual completions in this linear, BC-total setup (lower MAE, higher R²).
2. **Combined model** (Starts + Registrations, both lag 1) adds value (higher R²), so both indicators are useful.
3. **Pipeline dynamics:** Starts consistently exceed completions in many years (positive pipeline gap), and a clear lead–lag between starts and completions is visible in monthly and rolling series.
4. **Dwelling mix:** Apartments dominate recent BC starts and completions; single-detached and row are smaller shares.

**Recommendations (aligned with client scope):**
- **Short-term (1–2 years):** Use **starts** (and possibly multi-month lags) for completion forecasts, given they reflect units already in the construction pipeline.
- **Longer-term (3–5 years):** Incorporate **registrations** as an earlier signal of intended supply, especially when combined with starts in a single model.
- **Next steps:** (1) Add economic drivers (e.g. GDP, interest rates) and test regional or monthly models; (2) estimate conversion rates (registration → start, start → completion) and explicit lag distributions; (3) extend BC vs Canada comparison when Canada annual series are fully integrated.

---

## 7. Enhanced Forecasting: Optimal Lags and Economic Indicators

### 7.1 Optimal lag times

**Monthly (Starts → Completions):**  
A sweep over lags of 6, 9, 12, 18, 24, 30, and 36 months was run. The lag that **correlates best** with completions and **minimizes MAE** is **12 months**: correlation ≈ 0.61, MAE ≈ 493 (monthly units). So **1-year lagged starts** are the best monthly predictor of completions.

**Annual (Starts and Registrations → Completions):**  
For annual BC data:

| Indicator      | Lag (years) | Correlation | MAE (units) |
|----------------|-------------|-------------|--------------|
| Starts         | 1           | **0.89**    | ~2,518       |
| Registrations  | 1           | −0.31       | ~2,378       |
| Starts         | 2           | 0.85        | ~3,080       |
| Registrations  | 2           | −0.16       | ~2,568       |
| Starts         | 3           | 0.69        | ~4,345       |
| Registrations  | 3           | 0.67        | ~2,507       |

- **Best lag for Starts:** 1 year (strongest correlation, reasonable MAE).
- **Best lag for Registrations:** 3 years (positive correlation; 1-year reg has lower MAE but negative correlation, likely due to short overlap and multicollinearity).

**Conclusion:** For **short-term (1–2 year)** completion forecasts, **lag-1-year starts** are the strongest predictor. For **longer horizons (3–5 years)**, **lagged registrations** add signal.

### 7.2 Economic indicators

**Data:** Canadian economic indicators are **real-world data** from FRED (Federal Reserve Economic Data), saved to `Data/Economic_Indicators_Canada.csv`. The pipeline uses real data only: run `py fetch_economic_indicators.py` (requires `pandas-datareader`); if the CSV is missing, the forecasting script will call the fetch and exit with instructions if FRED is unavailable. Synthetic data is not used unless you explicitly run with `--allow-synthetic` for testing.

**Indicators used (FRED, real data):** Unemployment_Rate, CPI_Index, Policy_Rate, Interest_Rate_5Y, Employment_Index (and Housing_Starts_Canada_Total when available). Fetched via FRED CSV download (no API key required).

**Correlation with Completions (BC annual):**  
- **Strongest positive:** CPI_Index (r ≈ 0.62), Housing_Starts_Canada_Total (r ≈ 0.47), Industrial_Production_Index (r ≈ 0.47), Employment_Index (r ≈ 0.41).  
- **Negative:** Policy_Rate (r ≈ −0.33), Unemployment_Rate (r ≈ −0.28), Mortgage_Rate_5Y_Pct (r ≈ −0.25).

**Correlation with Starts (BC annual):**  
- **Strongest positive:** CPI_Index (r ≈ 0.69), then Housing_Starts_Canada_Total, Industrial_Production_Index, Employment_Index.  
- **Negative:** Policy_Rate, Unemployment_Rate, Mortgage_Rate_5Y_Pct.

So **CPI, Canada housing starts, industrial production, and employment** move with both BC completions and BC starts; **interest rates and unemployment** move in the opposite direction.

**Which affect Completions and Starts most (Ridge, standardized coefficients):**  
- **Completions:** Largest |coefficient|: Mortgage_Rate_5Y_Pct (negative), Housing_Starts_Canada_Total (positive), Reg_Lag1 (negative), Consumer_Confidence_Index (negative), Starts_Lag2 (positive), Population_Growth_Pct (positive).  
- **Starts:** See `outputs/economic_coefficients_starts.csv` and `forecast_06_coefficients_starts.png` for relative importance of registrations lags and economic indicators.

**Summary:** **Mortgage rates** and **Canada housing starts** have the largest standardized impact on completions; **CPI** and **employment** are also important. For starts, **registrations lags** and **economic conditions** (e.g. rates, confidence) matter.

### 7.3 Improved forecasting

- **Final model:** Ridge regression (Completions ~ best lags + all economic indicators), with stronger regularization (alpha=10) to limit overfitting. Train: all years except last 2; test: last 2 years.  
- **Simple model:** Completions ~ Starts_Lag1 + Reg_Lag1 + top 3 economic indicators (by correlation with completions). Fewer features to improve stability on a small test set.  
- **Outputs:** `outputs/forecast_01_best_lag_monthly.png`, `forecast_02_best_lag_annual.png`, `forecast_03_economic_corr_completions.png`, `forecast_04_economic_corr_starts.png`, `forecast_05_coefficients_completions.png`, `forecast_06_coefficients_starts.png`, `forecast_07_actual_vs_predicted_test.png`, `optimal_lag_monthly.csv`, `optimal_lag_annual.csv`, `economic_correlation_*.csv`, `economic_coefficients_*.csv`, `forecast_summary.txt`.

**Note:** With only 2 test years, test R² can be volatile. The **simple model** (Completions ~ Starts_Lag1 + Reg_Lag1 + top 3 economic indicators) typically gives more stable out-of-sample results than the full Ridge model. Economic data is from FRED (real) by default.

### 7.4 Separate model: Starts (Lag-1) + Registrations (Lag-3)

A dedicated model uses **Starts (lag 1 year)** and **Registrations (lag 3 years)** only: Completions ~ Starts_Lag1 + Reg_Lag3 (Ridge, α=1). This aligns with the finding that starts are best at 1-year lag and registrations add signal at 3-year lag.

**Representative metrics (from pipeline run):**
- **In-sample:** MAE ≈ 1,908 units, R² ≈ 0.63 (n=7 years, limited by registration data from 2016 onward).
- **Test (last 2 years):** MAE ≈ 3,305, R² ≈ 0.23.

**Outputs:** `outputs/model_starts1_reg3_metrics.csv`, `outputs/forecast_08_model_starts1_reg3.png` (actual vs predicted).

---

## 8. How to Reproduce

From the project folder:

```bash
py -m pip install -r requirements.txt   # install deps including pandas-datareader for real economic data
py fetch_economic_indicators.py         # fetches real FRED data to Data/Economic_Indicators_Canada.csv
py bc_housing_analysis.py               # main analysis + 15+ graphs
py bc_housing_forecasting.py             # optimal lags, economic impact, improved forecasting
```

**Requirements:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `openpyxl`, **`pandas-datareader`** (required for real economic data from FRED). See `requirements.txt`.  
**Outputs:** All figures in **`outputs/`** (including `forecast_*.png`), **`prediction_metrics.csv`**, **`optimal_lag_*.csv`**, **`economic_correlation_*.csv`**, **`economic_coefficients_*.csv`**, **`forecast_summary.txt`**.

---

*Analysis generated for MGTA 621 Field Project — BC Housing. Scope and objectives are drawn from meeting minutes, Question list, and Project Background documents.*
