"""
MGTA 621 — BC Housing: Enhanced Forecasting, Optimal Lags, and Economic Indicators.
- Find best lag (months/years) for Starts and Registrations vs Completions.
- Add many economic indicators; identify which affect Completions and Starts most.
- Improve forecasting with best lags + economics (Ridge, train/test split).
Run: py fetch_economic_indicators.py  then  py bc_housing_forecasting.py
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "Data"
OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)
sns.set_style("whitegrid")
COLORS = sns.color_palette("husl", 10)


def clean_num(s):
    if pd.isna(s) or s == "" or s == "*":
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    return float(str(s).replace(",", "").strip())


def load_bc_starts():
    path = DATA_DIR / "BC Starts CMHC.csv"
    df = pd.read_csv(path, skiprows=2, encoding="latin-1")
    df = df.rename(columns={df.columns[0]: "Period"})
    df = df.dropna(subset=["Period"]).copy()
    df["Period"] = df["Period"].astype(str).str.strip()
    months = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
    def parse_period(p):
        for name, num in months.items():
            if name in p:
                y = re.search(r"\d{4}", p)
                if y:
                    return pd.Timestamp(int(y.group()), num, 1)
        return pd.NaT
    df["Date"] = df["Period"].apply(parse_period)
    df = df.dropna(subset=["Date"]).copy()
    for c in ["Single", "Semi-Detached", "Row", "Apartment", "Total"]:
        if c in df.columns:
            df[c] = df[c].apply(clean_num)
    return df[["Date", "Total"]].rename(columns={"Total": "Starts"})


def load_bc_completions():
    path = DATA_DIR / "BC Completions CMHC.csv"
    df = pd.read_csv(path, skiprows=2, encoding="latin-1")
    df = df.rename(columns={df.columns[0]: "Period"})
    df = df.dropna(subset=["Period"]).copy()
    df["Period"] = df["Period"].astype(str).str.strip()
    months = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 1)}
    def parse_period(p):
        for name, num in months.items():
            if name in p:
                y = re.search(r"\d{4}", p)
                if y:
                    return pd.Timestamp(int(y.group()), num, 1)
        return pd.NaT
    df["Date"] = df["Period"].apply(parse_period)
    df = df.dropna(subset=["Date"]).copy()
    for c in ["Total"]:
        if c in df.columns:
            df[c] = df[c].apply(clean_num)
    return df[["Date", "Total"]].rename(columns={"Total": "Completions"})


def load_registrations():
    path = DATA_DIR / "Housing Registration Data.xlsx"
    sheets = ["Single Detached ", "Multi Unit Homes", "Purpose Built Rental"]
    out = []
    for sh in sheets:
        df = pd.read_excel(path, sheet_name=sh, header=None)
        year_row = df.iloc[0]
        years = [int(clean_num(year_row.get(i))) for i in range(1, 10) if not np.isnan(clean_num(year_row.get(i)))]
        data = df.iloc[1:, 1 : 1 + len(years)]
        data = data.map(lambda x: np.nan if x == "*" or str(x).strip() == "*" else clean_num(x))
        total = data.sum()
        for yr, val in zip(years, total):
            out.append({"Year": yr, "Registrations": val})
    reg = pd.DataFrame(out)
    bc_reg = reg.groupby("Year")["Registrations"].sum().reset_index()
    bc_reg.columns = ["Year", "Registrations"]
    return bc_reg


def load_economic():
    """Load real economic indicators from Data/Economic_Indicators_Canada.csv.
    If missing, runs fetch_economic_indicators (requires pandas_datareader; uses FRED real data)."""
    path = DATA_DIR / "Economic_Indicators_Canada.csv"
    if not path.exists():
        from fetch_economic_indicators import main as fetch_main
        fetch_main()
    df = pd.read_csv(path)
    if "Year" not in df.columns and df.shape[1] > 0:
        df["Year"] = df.index
    return df


# ---------- Optimal lag search ----------
def best_lag_monthly(starts_df, compl_df, max_lag_months=36, lags_to_try=None):
    """Find lag (months) that maximizes |correlation| of Completions with lagged Starts; also report MAE."""
    if lags_to_try is None:
        lags_to_try = [6, 9, 12, 18, 24, 30, 36]
    s = starts_df.set_index("Date")["Starts"]
    c = compl_df.set_index("Date")["Completions"]
    common = s.index.intersection(c.index)
    s, c = s.reindex(common).ffill().bfill(), c.reindex(common).ffill().bfill()
    results = []
    for lag_m in lags_to_try:
        if lag_m > len(s) // 2:
            continue
        lagged = s.shift(lag_m)
        valid = pd.concat([lagged, c], axis=1).dropna()
        if len(valid) < 10:
            continue
        corr = valid["Starts"].corr(valid["Completions"])
        X = valid[["Starts"]]
        y = valid["Completions"]
        m = LinearRegression().fit(X, y)
        pred = m.predict(X)
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        results.append({"lag_months": lag_m, "correlation": corr, "MAE": mae, "R2": r2})
    if not results:
        return None, []
    res_df = pd.DataFrame(results)
    best_by_corr = res_df.loc[res_df["correlation"].abs().idxmax()]
    best_by_mae = res_df.loc[res_df["MAE"].idxmin()]
    return {
        "best_lag_months_by_corr": int(best_by_corr["lag_months"]),
        "best_corr": float(best_by_corr["correlation"]),
        "best_lag_months_by_mae": int(best_by_mae["lag_months"]),
        "best_mae": float(best_by_mae["MAE"]),
        "results": res_df,
    }, res_df


def best_lag_annual(annual_starts, annual_compl, bc_reg, max_lag_years=3):
    """Find best lag (years) for Starts and for Registrations vs Completions (annual)."""
    df = annual_starts[["Year", "Starts"]].merge(
        annual_compl[["Year", "Completions"]], on="Year"
    ).merge(bc_reg, on="Year", how="left").sort_values("Year").reset_index(drop=True)

    for lag_y in range(1, max_lag_years + 1):
        df[f"Starts_Lag{lag_y}"] = df["Starts"].shift(lag_y)
        df[f"Reg_Lag{lag_y}"] = df["Registrations"].shift(lag_y)

    results = []
    for lag_y in range(1, max_lag_years + 1):
        train = df.dropna(subset=["Completions", f"Starts_Lag{lag_y}"])
        if len(train) >= 5:
            corr = train["Completions"].corr(train[f"Starts_Lag{lag_y}"])
            m = LinearRegression().fit(train[[f"Starts_Lag{lag_y}"]], train["Completions"])
            mae = mean_absolute_error(train["Completions"], m.predict(train[[f"Starts_Lag{lag_y}"]]))
            results.append({"Indicator": "Starts", "Lag_Years": lag_y, "Correlation": corr, "MAE": mae})
        train = df.dropna(subset=["Completions", f"Reg_Lag{lag_y}"])
        if len(train) >= 5:
            corr = train["Completions"].corr(train[f"Reg_Lag{lag_y}"])
            m = LinearRegression().fit(train[[f"Reg_Lag{lag_y}"]], train["Completions"])
            mae = mean_absolute_error(train["Completions"], m.predict(train[[f"Reg_Lag{lag_y}"]]))
            results.append({"Indicator": "Registrations", "Lag_Years": lag_y, "Correlation": corr, "MAE": mae})
    return df, pd.DataFrame(results)


def main():
    print("Loading BC data...")
    bc_starts = load_bc_starts()
    bc_compl = load_bc_completions()
    bc_reg = load_registrations()
    annual_starts = bc_starts.copy()
    annual_starts["Year"] = pd.to_datetime(annual_starts["Date"]).dt.year
    annual_starts = annual_starts.groupby("Year")["Starts"].sum().reset_index()
    annual_compl = bc_compl.copy()
    annual_compl["Year"] = pd.to_datetime(annual_compl["Date"]).dt.year
    annual_compl = annual_compl.groupby("Year")["Completions"].sum().reset_index()

    # ---------- Optimal lag: monthly (Starts -> Completions) ----------
    print("Finding best lag (monthly): Starts -> Completions...")
    lag_monthly, lag_monthly_df = best_lag_monthly(bc_starts, bc_compl)
    if lag_monthly:
        print(f"  Best lag by |correlation|: {lag_monthly['best_lag_months_by_corr']} months (corr={lag_monthly['best_corr']:.3f})")
        print(f"  Best lag by MAE: {lag_monthly['best_lag_months_by_mae']} months (MAE={lag_monthly['best_mae']:.0f})")

    # ---------- Optimal lag: annual (Starts and Registrations -> Completions) ----------
    df_annual, lag_annual_df = best_lag_annual(annual_starts, annual_compl, bc_reg)
    # Build proper annual lag results for Starts only and Reg only
    annual_results = []
    for lag_y in range(1, 4):
        col_s = f"Starts_Lag{lag_y}"
        col_r = f"Reg_Lag{lag_y}"
        if col_s in df_annual.columns:
            t = df_annual.dropna(subset=["Completions", col_s])
            if len(t) >= 5:
                c = t["Completions"].corr(t[col_s])
                m = LinearRegression().fit(t[[col_s]], t["Completions"])
                mae = mean_absolute_error(t["Completions"], m.predict(t[[col_s]]))
                annual_results.append({"Indicator": "Starts", "Lag_Years": lag_y, "Correlation": c, "MAE": mae})
        if col_r in df_annual.columns:
            t = df_annual.dropna(subset=["Completions", col_r])
            if len(t) >= 5:
                c = t["Completions"].corr(t[col_r])
                m = LinearRegression().fit(t[[col_r]], t["Completions"])
                mae = mean_absolute_error(t["Completions"], m.predict(t[[col_r]]))
                annual_results.append({"Indicator": "Registrations", "Lag_Years": lag_y, "Correlation": c, "MAE": mae})
    annual_lag_df = pd.DataFrame(annual_results)
    if not annual_lag_df.empty:
        print("Annual lag results (Starts & Registrations -> Completions):")
        print(annual_lag_df.to_string(index=False))

    # ---------- Load economic indicators ----------
    print("Loading economic indicators...")
    econ = load_economic()
    econ_cols = [c for c in econ.columns if c != "Year" and pd.api.types.is_numeric_dtype(econ[c])]
    if not econ_cols:
        econ = None
        print("  No economic columns found; using synthetic run or add Economic_Indicators_Canada.csv")
    else:
        print(f"  Loaded {len(econ_cols)} indicators: {econ_cols}")

    # ---------- Merge annual BC + economic ----------
    df_annual = df_annual.merge(econ, on="Year", how="left") if econ is not None else df_annual
    feature_cols = [c for c in econ_cols if c in df_annual.columns]
    for c in feature_cols:
        df_annual[c] = pd.to_numeric(df_annual[c], errors="coerce")
    df_annual = df_annual.dropna(subset=["Completions", "Starts"], how="all")

    # ---------- Correlation: Completions and Starts vs each economic indicator ----------
    corr_compl = []
    corr_starts = []
    for col in feature_cols:
        valid = df_annual.dropna(subset=["Completions", col])
        if len(valid) >= 5:
            corr_compl.append({"Indicator": col, "Correlation_with_Completions": valid["Completions"].corr(valid[col])})
        valid = df_annual.dropna(subset=["Starts", col])
        if len(valid) >= 5:
            corr_starts.append({"Indicator": col, "Correlation_with_Starts": valid["Starts"].corr(valid[col])})
    corr_compl_df = pd.DataFrame(corr_compl) if corr_compl else pd.DataFrame()
    corr_starts_df = pd.DataFrame(corr_starts) if corr_starts else pd.DataFrame()

    # ---------- Regression: which economic indicators affect Completions / Starts most (Ridge, standardized) ----------
    train_full = df_annual.dropna(subset=["Completions"] + [f"Starts_Lag{k}" for k in [1, 2, 3] if f"Starts_Lag{k}" in df_annual.columns] + [f"Reg_Lag{k}" for k in [1, 2, 3] if f"Reg_Lag{k}" in df_annual.columns] + feature_cols, how="any")
    if len(train_full) < 6 or not feature_cols:
        train_full = df_annual.dropna(how="any", subset=["Completions", "Starts_Lag1"])
        pred_cols = ["Starts_Lag1"]
        feature_cols_used = []
    else:
        pred_cols = [c for c in ["Starts_Lag1", "Starts_Lag2", "Reg_Lag1", "Reg_Lag2"] if c in df_annual.columns]
        feature_cols_used = [c for c in feature_cols if c in train_full.columns]
        pred_cols = pred_cols + feature_cols_used
        pred_cols = [c for c in pred_cols if train_full[c].notna().sum() >= len(train_full) // 2]
    if len(train_full) < 5:
        pred_cols = ["Starts_Lag1"] if "Starts_Lag1" in df_annual.columns else []
        train_full = df_annual.dropna(subset=["Completions", "Starts_Lag1"])

    scaler_y = StandardScaler()
    scaler_X = StandardScaler()
    y_compl = train_full["Completions"].values.reshape(-1, 1)
    X_all = train_full[pred_cols].fillna(train_full[pred_cols].median())
    X_scaled = scaler_X.fit_transform(X_all)
    y_scaled = scaler_y.fit_transform(y_compl).ravel()
    ridge_compl = Ridge(alpha=1.0).fit(X_scaled, y_scaled)
    coef_compl = pd.DataFrame({"Feature": pred_cols, "Std_Coefficient_Completions": ridge_compl.coef_})
    coef_compl = coef_compl.sort_values("Std_Coefficient_Completions", key=abs, ascending=False)

    # Starts ~ Registrations lags + economic
    train_s = df_annual.dropna(subset=["Starts"] + [f"Reg_Lag{k}" for k in [1, 2] if f"Reg_Lag{k}" in df_annual.columns] + feature_cols[:5], how="any")
    if len(train_s) >= 5 and feature_cols:
        pred_cols_s = [c for c in ["Reg_Lag1", "Reg_Lag2"] + feature_cols if c in train_s.columns]
        pred_cols_s = [c for c in pred_cols_s if train_s[c].notna().sum() >= len(train_s) // 2][:10]
        if pred_cols_s:
            X_s = train_s[pred_cols_s].fillna(train_s[pred_cols_s].median())
            y_s = train_s["Starts"].values
            scaler_Xs = StandardScaler()
            scaler_ys = StandardScaler()
            X_s_scaled = scaler_Xs.fit_transform(X_s)
            y_s_scaled = scaler_ys.fit_transform(y_s.reshape(-1, 1)).ravel()
            ridge_starts = Ridge(alpha=1.0).fit(X_s_scaled, y_s_scaled)
            coef_starts = pd.DataFrame({"Feature": pred_cols_s, "Std_Coefficient_Starts": ridge_starts.coef_})
            coef_starts = coef_starts.sort_values("Std_Coefficient_Starts", key=abs, ascending=False)
        else:
            coef_starts = pd.DataFrame()
    else:
        coef_starts = pd.DataFrame()

    # ---------- Train/test split and final forecast ----------
    df_annual = df_annual.sort_values("Year").reset_index(drop=True)
    test_years = 2
    train_mask = df_annual["Year"] < (df_annual["Year"].max() - test_years + 1)
    test_mask = ~train_mask
    use_cols = [c for c in ["Starts_Lag1", "Starts_Lag2", "Reg_Lag1", "Reg_Lag2"] + feature_cols if c in df_annual.columns]
    use_cols = [c for c in use_cols if df_annual[c].notna().sum() >= 10][:12]
    if not use_cols:
        use_cols = ["Starts_Lag1", "Reg_Lag1"] if "Reg_Lag1" in df_annual.columns else ["Starts_Lag1"]
    train_df = df_annual.loc[train_mask].dropna(subset=["Completions"] + use_cols, how="any")
    test_df = df_annual.loc[test_mask].dropna(subset=["Completions"] + use_cols, how="any")
    pred_te, y_te = None, None
    mae_s, r2_s = np.nan, np.nan
    mae_s1r3, r2_s1r3, mae_te_s1r3, r2_te_s1r3 = np.nan, np.nan, np.nan, np.nan
    if len(train_df) >= 5 and len(test_df) >= 1:
        X_tr = train_df[use_cols].fillna(train_df[use_cols].median())
        y_tr = train_df["Completions"]
        X_te = test_df[use_cols].fillna(train_df[use_cols].median())
        y_te = test_df["Completions"]
        scaler_f = StandardScaler()
        X_tr_s = scaler_f.fit_transform(X_tr)
        X_te_s = scaler_f.transform(X_te)
        model_final = Ridge(alpha=10.0).fit(X_tr_s, y_tr)
        pred_te = model_final.predict(X_te_s)
        test_mae = mean_absolute_error(y_te, pred_te)
        test_r2 = r2_score(y_te, pred_te)
        print(f"\nFinal model (Ridge, best lags + economic): Test MAE = {test_mae:.0f}, Test R² = {test_r2:.3f}")
        # Simple model: best lag + top 3 economic (fewer features)
        top_econ = corr_compl_df.nlargest(3, "Correlation_with_Completions")["Indicator"].tolist() if not corr_compl_df.empty else []
        simple_cols = [c for c in ["Starts_Lag1", "Reg_Lag1"] + top_econ if c in df_annual.columns][:5]
        if len(simple_cols) >= 2:
            train_s = df_annual.loc[train_mask].dropna(subset=["Completions"] + simple_cols, how="any")
            test_s = df_annual.loc[test_mask].dropna(subset=["Completions"] + simple_cols, how="any")
            if len(train_s) >= 5 and len(test_s) >= 1:
                X_tr_sm = train_s[simple_cols].fillna(train_s[simple_cols].median())
                y_tr_sm = train_s["Completions"]
                X_te_sm = test_s[simple_cols].fillna(train_s[simple_cols].median())
                y_te_sm = test_s["Completions"]
                scaler_s = StandardScaler()
                model_simple = Ridge(alpha=1.0).fit(scaler_s.fit_transform(X_tr_sm), y_tr_sm)
                pred_simple = model_simple.predict(scaler_s.transform(X_te_sm))
                mae_s = mean_absolute_error(y_te_sm, pred_simple)
                r2_s = r2_score(y_te_sm, pred_simple)
                print(f"Simple model (Starts_Lag1 + top 3 economic): Test MAE = {mae_s:.0f}, Test R² = {r2_s:.3f}")
    else:
        test_mae, test_r2 = np.nan, np.nan

    # ---------- Separate model: Starts Lag-1 + Registrations Lag-3 ----------
    model_s1r3_cols = ["Starts_Lag1", "Reg_Lag3"]
    if all(c in df_annual.columns for c in model_s1r3_cols):
        train_s1r3 = df_annual.dropna(subset=["Completions", "Starts_Lag1", "Reg_Lag3"])
        if len(train_s1r3) >= 5:
            X_s1r3 = train_s1r3[model_s1r3_cols]
            y_s1r3 = train_s1r3["Completions"]
            scaler_s1r3 = StandardScaler()
            X_s1r3_s = scaler_s1r3.fit_transform(X_s1r3)
            model_s1r3 = Ridge(alpha=1.0).fit(X_s1r3_s, y_s1r3)
            pred_s1r3 = model_s1r3.predict(X_s1r3_s)
            mae_s1r3 = mean_absolute_error(y_s1r3, pred_s1r3)
            r2_s1r3 = r2_score(y_s1r3, pred_s1r3)
            print(f"\nStarts Lag-1 + Registrations Lag-3 model: in-sample MAE = {mae_s1r3:.0f}, R² = {r2_s1r3:.3f} (n={len(train_s1r3)})")
            # Test set if available
            test_s1r3 = df_annual.loc[test_mask].dropna(subset=["Completions", "Starts_Lag1", "Reg_Lag3"]) if "test_mask" in dir() else pd.DataFrame()
            if len(test_s1r3) >= 1:
                X_te_s1r3 = test_s1r3[model_s1r3_cols]
                X_te_s1r3_s = scaler_s1r3.transform(X_te_s1r3)
                pred_te_s1r3 = model_s1r3.predict(X_te_s1r3_s)
                mae_te_s1r3 = mean_absolute_error(test_s1r3["Completions"], pred_te_s1r3)
                r2_te_s1r3 = r2_score(test_s1r3["Completions"], pred_te_s1r3)
                print(f"  Test MAE = {mae_te_s1r3:.0f}, Test R² = {r2_te_s1r3:.3f}")
            else:
                mae_te_s1r3, r2_te_s1r3 = np.nan, np.nan
            # Save metrics
            row = {"Model": "Starts_Lag1_plus_Reg_Lag3", "MAE_in_sample": mae_s1r3, "R2_in_sample": r2_s1r3, "n_train": len(train_s1r3)}
            if len(test_s1r3) >= 1:
                row["MAE_test"] = mae_te_s1r3
                row["R2_test"] = r2_te_s1r3
            pd.DataFrame([row]).to_csv(OUT_DIR / "model_starts1_reg3_metrics.csv", index=False)
            # Plot: actual vs predicted (in-sample)
            fig_s1r3, ax_s1r3 = plt.subplots(figsize=(8, 6))
            ax_s1r3.scatter(y_s1r3, pred_s1r3, s=60, color=COLORS[6], label="Starts Lag-1 + Reg Lag-3")
            mn, mx = min(y_s1r3.min(), pred_s1r3.min()), max(y_s1r3.max(), pred_s1r3.max())
            ax_s1r3.plot([mn, mx], [mn, mx], "k--", label="Perfect")
            ax_s1r3.set_xlabel("Actual Completions")
            ax_s1r3.set_ylabel("Predicted Completions")
            ax_s1r3.set_title("Model: Completions ~ Starts (Lag-1) + Registrations (Lag-3)")
            ax_s1r3.legend()
            fig_s1r3.tight_layout()
            fig_s1r3.savefig(OUT_DIR / "forecast_08_model_starts1_reg3.png", dpi=150, bbox_inches="tight")
            plt.close(fig_s1r3)

    # ---------- Save outputs ----------
    if not annual_lag_df.empty:
        annual_lag_df.to_csv(OUT_DIR / "optimal_lag_annual.csv", index=False)
    if lag_monthly_df is not None and len(lag_monthly_df) > 0:
        lag_monthly_df.to_csv(OUT_DIR / "optimal_lag_monthly.csv", index=False)
    if not corr_compl_df.empty:
        corr_compl_df.to_csv(OUT_DIR / "economic_correlation_completions.csv", index=False)
    if not corr_starts_df.empty:
        corr_starts_df.to_csv(OUT_DIR / "economic_correlation_starts.csv", index=False)
    coef_compl.to_csv(OUT_DIR / "economic_coefficients_completions.csv", index=False)
    if not coef_starts.empty:
        coef_starts.to_csv(OUT_DIR / "economic_coefficients_starts.csv", index=False)

    # ---------- Graphs ----------
    # 1) Best lag monthly: correlation and MAE by lag
    if lag_monthly_df is not None and len(lag_monthly_df) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
        lag_monthly_df.plot(x="lag_months", y="correlation", ax=ax1, marker="o", color=COLORS[0])
        ax1.axhline(0, color="gray", ls="--")
        ax1.set_ylabel("Correlation(Completions, Lagged Starts)")
        ax1.set_title("Optimal Lag (Monthly): Starts -> Completions")
        lag_monthly_df.plot(x="lag_months", y="MAE", ax=ax2, marker="s", color=COLORS[1])
        ax2.set_ylabel("MAE")
        ax2.set_xlabel("Lag (months)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "forecast_01_best_lag_monthly.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 2) Best lag annual: bar (MAE by lag and indicator)
    if not annual_lag_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(annual_lag_df))
        ax.bar(x - 0.2, annual_lag_df["MAE"], width=0.35, label="MAE", color=COLORS[0])
        ax_t = ax.twinx()
        ax_t.bar(x + 0.2, annual_lag_df["Correlation"], width=0.35, label="Correlation", color=COLORS[1], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['Indicator']} Lag-{int(row['Lag_Years'])}" for _, row in annual_lag_df.iterrows()], rotation=30, ha="right")
        ax.set_ylabel("MAE")
        ax_t.set_ylabel("Correlation")
        ax.set_title("Optimal Lag (Annual): Starts & Registrations -> Completions")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "forecast_02_best_lag_annual.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 3) Economic indicators: correlation with Completions
    if not corr_compl_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        corr_compl_df = corr_compl_df.sort_values("Correlation_with_Completions", key=abs, ascending=True)
        ax.barh(corr_compl_df["Indicator"], corr_compl_df["Correlation_with_Completions"], color=COLORS[2])
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Correlation with Completions")
        ax.set_title("Economic Indicators: Correlation with BC Completions")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "forecast_03_economic_corr_completions.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 4) Economic indicators: correlation with Starts
    if not corr_starts_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        corr_starts_df = corr_starts_df.sort_values("Correlation_with_Starts", key=abs, ascending=True)
        ax.barh(corr_starts_df["Indicator"], corr_starts_df["Correlation_with_Starts"], color=COLORS[3])
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Correlation with Starts")
        ax.set_title("Economic Indicators: Correlation with BC Starts")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "forecast_04_economic_corr_starts.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 5) Standardized coefficients: which affect Completions most
    if not coef_compl.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        coef_compl = coef_compl.sort_values("Std_Coefficient_Completions", key=abs, ascending=True)
        ax.barh(coef_compl["Feature"], coef_compl["Std_Coefficient_Completions"], color=COLORS[4])
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Standardized Coefficient (Ridge)")
        ax.set_title("Which Predictors Affect Completions Most?")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "forecast_05_coefficients_completions.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 6) Standardized coefficients: which affect Starts most
    if not coef_starts.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        coef_starts = coef_starts.sort_values("Std_Coefficient_Starts", key=abs, ascending=True)
        ax.barh(coef_starts["Feature"], coef_starts["Std_Coefficient_Starts"], color=COLORS[5])
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Standardized Coefficient (Ridge)")
        ax.set_title("Which Predictors Affect Starts Most?")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "forecast_06_coefficients_starts.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 7) Actual vs Predicted (test set)
    if pred_te is not None and y_te is not None and len(pred_te) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_te, pred_te, s=80, color=COLORS[6], label="Test predictions")
        mn, mx = min(y_te.min(), pred_te.min()), max(y_te.max(), pred_te.max())
        ax.plot([mn, mx], [mn, mx], "k--", label="Perfect")
        ax.set_xlabel("Actual Completions")
        ax.set_ylabel("Predicted Completions")
        ax.set_title("Final Model: Test Set — Actual vs Predicted")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "forecast_07_actual_vs_predicted_test.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Summary
    with open(OUT_DIR / "forecast_summary.txt", "w") as f:
        f.write("BC Housing Enhanced Forecasting — Summary\n")
        f.write("=========================================\n")
        if lag_monthly:
            f.write(f"Best monthly lag (Starts->Completions): {lag_monthly['best_lag_months_by_corr']} months (by |corr|), {lag_monthly['best_lag_months_by_mae']} months (by MAE)\n")
        if not annual_lag_df.empty:
            f.write("Best annual lags: see optimal_lag_annual.csv\n")
        f.write(f"Economic indicators loaded: {len(feature_cols)}\n")
        if not corr_compl_df.empty:
            top = corr_compl_df.loc[corr_compl_df["Correlation_with_Completions"].abs().idxmax()]
            f.write(f"Strongest correlation with Completions: {top['Indicator']} (r={top['Correlation_with_Completions']:.3f})\n")
        if not corr_starts_df.empty:
            top = corr_starts_df.loc[corr_starts_df["Correlation_with_Starts"].abs().idxmax()]
            f.write(f"Strongest correlation with Starts: {top['Indicator']} (r={top['Correlation_with_Starts']:.3f})\n")
        f.write(f"Full model (Ridge, all lags + economic): Test MAE = {test_mae:.0f}, Test R² = {test_r2:.3f}\n")
        if not np.isnan(mae_s):
            f.write(f"Simple model (Starts_Lag1 + top 3 economic): Test MAE = {mae_s:.0f}, Test R² = {r2_s:.3f} (preferred for small test set)\n")
        if not np.isnan(mae_s1r3):
            f.write(f"Starts Lag-1 + Registrations Lag-3 model: in-sample MAE = {mae_s1r3:.0f}, R² = {r2_s1r3:.3f}\n")
            if not np.isnan(mae_te_s1r3):
                f.write(f"  Test MAE = {mae_te_s1r3:.0f}, Test R² = {r2_te_s1r3:.3f}\n")
    print("\nDone. Outputs in outputs/: forecast_*.png, optimal_lag_*.csv, economic_*.csv, model_starts1_reg3_metrics.csv, forecast_summary.txt")


if __name__ == "__main__":
    main()
