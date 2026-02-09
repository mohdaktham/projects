"""
MGTA 621 — BC Housing Field Project: Full Analysis Pipeline
Scope (from meeting minutes & proposal):
  - Compare housing registrations vs housing starts as predictors of completions
  - Time lags: registration→start, start→completion
  - Conversion rates; forecasting windows (starts 1–2yr, registrations 3–5yr)
  - BC vs Canada trends; by dwelling type
Deliverable: 10+ graphs, detailed analysis, prediction comparisons.
Run: py bc_housing_analysis.py
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "Data"
OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams["figure.figsize"] = (10, 5)
sns.set_style("whitegrid")
COLORS = sns.color_palette("husl", 8)


def clean_num(s):
    """Parse number from string (handles commas, *, NaN)."""
    if pd.isna(s) or s == "" or s == "*":
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    return float(str(s).replace(",", "").strip())


def load_bc_starts():
    """BC Starts CMHC: monthly by dwelling type."""
    path = DATA_DIR / "BC Starts CMHC.csv"
    df = pd.read_csv(path, skiprows=2, encoding="latin-1")
    df = df.rename(columns={df.columns[0]: "Period"})
    df = df.dropna(subset=["Period"]).copy()
    df["Period"] = df["Period"].astype(str).str.strip()
    # Parse "YYYY Month" -> datetime (first of month)
    months = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
    }
    def parse_period(p):
        for name, num in months.items():
            if name in p:
                y = re.search(r"\d{4}", p)
                if y:
                    return pd.Timestamp(int(y.group()), num, 1)
        return pd.NaT
    df["Date"] = df["Period"].apply(parse_period)
    df = df.dropna(subset=["Date"]).copy()
    cols = ["Single", "Semi-Detached", "Row", "Apartment", "Total"]
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(clean_num)
    df = df[["Date"] + [c for c in cols if c in df.columns]].copy()
    return df


def load_bc_completions():
    """BC Completions CMHC: monthly by dwelling type."""
    path = DATA_DIR / "BC Completions CMHC.csv"
    df = pd.read_csv(path, skiprows=2, encoding="latin-1")
    df = df.rename(columns={df.columns[0]: "Period"})
    df = df.dropna(subset=["Period"]).copy()
    df["Period"] = df["Period"].astype(str).str.strip()
    months = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
    }
    def parse_period(p):
        for name, num in months.items():
            if name in p:
                y = re.search(r"\d{4}", p)
                if y:
                    return pd.Timestamp(int(y.group()), num, 1)
        return pd.NaT
    df["Date"] = df["Period"].apply(parse_period)
    df = df.dropna(subset=["Date"]).copy()
    cols = ["Single", "Semi-Detached", "Row", "Apartment", "Total"]
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(clean_num)
    df = df[["Date"] + [c for c in cols if c in df.columns]].copy()
    return df


def load_canada_annual():
    """Canada annual: starts, under construction, completions."""
    path = DATA_DIR / "Annual Starts Constrxn Completion Canada 16 to24 CMHC StatsCan.csv"
    # Rows 10-27 have the data; header row 10
    df = pd.read_csv(path, skiprows=9, nrows=20)
    df = df.dropna(how="all", axis=1)
    first = df.columns[0]
    df = df.rename(columns={first: "Metric"})
    df["Metric"] = df["Metric"].ffill()
    # Extract type from first column (e.g. "Housing starts", "Total units", ...)
    return df


def load_registrations():
    """Housing Registration Data: annual by type (Single Detached, Multi Unit, Purpose Built Rental)."""
    path = DATA_DIR / "Housing Registration Data.xlsx"
    sheets = ["Single Detached ", "Multi Unit Homes", "Purpose Built Rental"]
    out = []
    for sh in sheets:
        df = pd.read_excel(path, sheet_name=sh, header=None)
        # Row 0: NaN, 2016, 2017, ... ; Row 1: region names
        year_row = df.iloc[0]
        years = [clean_num(year_row.get(i)) for i in range(1, 10)]
        years = [int(y) for y in years if not np.isnan(y)]
        regions = df.iloc[1:, 0]
        data = df.iloc[1:, 1 : 1 + len(years)]
        data = data.map(lambda x: np.nan if x == "*" or str(x).strip() == "*" else clean_num(x))
        total = data.sum()
        name = sh.strip()
        for yr, val in zip(years, total):
            out.append({"Year": yr, "Type": name, "Registrations": val})
    reg = pd.DataFrame(out)
    # BC total registrations by year
    bc_reg = reg.groupby("Year")["Registrations"].sum().reset_index()
    bc_reg.columns = ["Year", "BC_Registrations_Total"]
    return reg, bc_reg


def prepare_annual_bc(starts_df, compl_df):
    """Aggregate BC monthly starts/completions to annual. Keeps Year, Total, Single, Semi-Detached, Row, Apartment."""
    s = starts_df.copy()
    s["Year"] = s["Date"].dt.year
    annual_starts = s.groupby("Year").agg({"Total": "sum", "Single": "sum", "Semi-Detached": "sum", "Row": "sum", "Apartment": "sum"}).reset_index()

    c = compl_df.copy()
    c["Year"] = c["Date"].dt.year
    annual_compl = c.groupby("Year").agg({"Total": "sum", "Single": "sum", "Semi-Detached": "sum", "Row": "sum", "Apartment": "sum"}).reset_index()
    return annual_starts, annual_compl


def run_predictions(annual_starts, annual_compl, bc_reg):
    """Compare predictions: Completions ~ lagged Starts vs ~ lagged Registrations."""
    # Merge: Year, Completions_Total, lagged starts, lagged registrations
    df = annual_compl[["Year", "Total"]].rename(columns={"Total": "Completions"}).copy()
    df = df.merge(annual_starts[["Year", "Total"]].rename(columns={"Total": "Starts"}), on="Year", how="inner")
    df = df.merge(bc_reg.rename(columns={"BC_Registrations_Total": "Registrations"}), on="Year", how="left")
    df = df.sort_values("Year").reset_index(drop=True)

    # Lags 1 and 2 years
    df["Starts_Lag1"] = df["Starts"].shift(1)
    df["Starts_Lag2"] = df["Starts"].shift(2)
    df["Registrations_Lag1"] = df["Registrations"].shift(1)
    df["Registrations_Lag2"] = df["Registrations"].shift(2)
    train = df.dropna(subset=["Completions", "Starts_Lag1", "Registrations_Lag1"]).copy()

    models = {}
    # Model A: Completions ~ Starts_Lag1
    X1 = train[["Starts_Lag1"]]
    y = train["Completions"]
    m1 = LinearRegression().fit(X1, y)
    models["Starts_Lag1"] = {"model": m1, "X": X1, "y": y, "pred": m1.predict(X1)}

    # Model B: Completions ~ Registrations_Lag1
    X2 = train[["Registrations_Lag1"]].dropna()
    idx = X2.index.intersection(y.index)
    if len(idx) >= 5:
        m2 = LinearRegression().fit(train.loc[idx, ["Registrations_Lag1"]], train.loc[idx, "Completions"])
        models["Registrations_Lag1"] = {"model": m2, "X": train.loc[idx, ["Registrations_Lag1"]], "y": train.loc[idx, "Completions"], "pred": m2.predict(train.loc[idx, ["Registrations_Lag1"]])}

    # Model C: Completions ~ Starts_Lag1 + Registrations_Lag1
    X3 = train[["Starts_Lag1", "Registrations_Lag1"]].dropna()
    idx3 = X3.index.intersection(y.index)
    if len(idx3) >= 5:
        m3 = LinearRegression().fit(train.loc[idx3, ["Starts_Lag1", "Registrations_Lag1"]], train.loc[idx3, "Completions"])
        models["Starts_plus_Registrations_Lag1"] = {"model": m3, "X": train.loc[idx3, ["Starts_Lag1", "Registrations_Lag1"]], "y": train.loc[idx3, "Completions"], "pred": m3.predict(train.loc[idx3, ["Starts_Lag1", "Registrations_Lag1"]])}

    return df, train, models


def main():
    print("Loading data...")
    bc_starts = load_bc_starts()
    bc_compl = load_bc_completions()
    reg_by_type, bc_reg = load_registrations()
    annual_starts, annual_compl = prepare_annual_bc(bc_starts, bc_compl)
    df_pred, train_df, models = run_predictions(annual_starts, annual_compl, bc_reg)

    # ---------- GRAPHS (10+) ----------
    figs = []

    # 1) BC Total Starts vs Completions (monthly, recent 10 years)
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    recent = bc_starts["Date"] >= (bc_starts["Date"].max() - pd.DateOffset(years=10))
    ax1.plot(bc_starts.loc[recent, "Date"], bc_starts.loc[recent, "Total"], label="Starts", color=COLORS[0], alpha=0.9)
    recent_c = bc_compl["Date"] >= (bc_compl["Date"].max() - pd.DateOffset(years=10))
    ax1.plot(bc_compl.loc[recent_c, "Date"], bc_compl.loc[recent_c, "Total"], label="Completions", color=COLORS[1], alpha=0.9)
    ax1.set_title("BC Housing: Monthly Starts vs Completions (Last 10 Years)")
    ax1.set_ylabel("Units")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)
    fig1.tight_layout()
    fig1.savefig(OUT_DIR / "01_bc_starts_vs_completions_monthly.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    figs.append("01_bc_starts_vs_completions_monthly.png")

    # 2) BC Completions by dwelling type (stacked area, last 15 years)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    comp = bc_compl[bc_compl["Date"] >= (bc_compl["Date"].max() - pd.DateOffset(years=15))].copy()
    bottom = np.zeros(len(comp))
    for col in ["Single", "Semi-Detached", "Row", "Apartment"]:
        if col in comp.columns:
            ax2.fill_between(comp["Date"], bottom, bottom + comp[col], label=col, alpha=0.7)
            bottom = bottom + comp[col].values
    ax2.plot(comp["Date"], comp["Total"], color="black", linewidth=2, label="Total")
    ax2.set_title("BC Housing Completions by Dwelling Type — Stacked (Last 15 Years)")
    ax2.set_ylabel("Units")
    ax2.legend(loc="upper left")
    ax2.tick_params(axis="x", rotation=45)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "02_bc_completions_by_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    figs.append("02_bc_completions_by_type.png")

    # 2b) Simpler: line plot by type
    fig2b, ax2b = plt.subplots(figsize=(12, 5))
    for col in ["Single", "Semi-Detached", "Row", "Apartment", "Total"]:
        if col in comp.columns:
            ax2b.plot(comp["Date"], comp[col], label=col, linewidth=2 if col == "Total" else 1)
    ax2b.set_title("BC Housing Completions by Dwelling Type (Last 15 Years)")
    ax2b.set_ylabel("Units")
    ax2b.legend()
    ax2b.tick_params(axis="x", rotation=45)
    fig2b.tight_layout()
    fig2b.savefig(OUT_DIR / "02b_bc_completions_lines_by_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig2b)
    figs.append("02b_bc_completions_lines_by_type.png")

    # 3) BC Starts by dwelling type (lines, last 15 years)
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    st = bc_starts[bc_starts["Date"] >= (bc_starts["Date"].max() - pd.DateOffset(years=15))].copy()
    for col in ["Single", "Semi-Detached", "Row", "Apartment", "Total"]:
        if col in st.columns:
            ax3.plot(st["Date"], st[col], label=col, linewidth=2 if col == "Total" else 1)
    ax3.set_title("BC Housing Starts by Dwelling Type (Last 15 Years)")
    ax3.set_ylabel("Units")
    ax3.legend()
    ax3.tick_params(axis="x", rotation=45)
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "03_bc_starts_by_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    figs.append("03_bc_starts_by_type.png")

    # 4) BC Annual Totals: Starts vs Completions (bar)
    fig4, ax4 = plt.subplots(figsize=(11, 5))
    merge_annual = annual_starts[["Year", "Total"]].merge(annual_compl[["Year", "Total"]], on="Year", suffixes=("_Starts", "_Compl"))
    merge_annual = merge_annual.rename(columns={"Total_Starts": "Starts", "Total_Compl": "Completions"})
    x = np.arange(len(merge_annual))
    w = 0.35
    ax4.bar(x - w/2, merge_annual["Starts"], width=w, label="Starts", color=COLORS[0])
    ax4.bar(x + w/2, merge_annual["Completions"], width=w, label="Completions", color=COLORS[1])
    ax4.set_xticks(x)
    ax4.set_xticklabels(merge_annual["Year"].astype(int), rotation=45)
    ax4.set_title("BC Annual Housing Starts vs Completions")
    ax4.set_ylabel("Units")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(OUT_DIR / "04_bc_annual_starts_vs_completions.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    figs.append("04_bc_annual_starts_vs_completions.png")

    # 5) Registrations by type (annual)
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    reg_wide = reg_by_type.pivot(index="Year", columns="Type", values="Registrations")
    reg_wide.plot(kind="bar", ax=ax5, width=0.8)
    ax5.set_title("BC Housing Registrations by Type (Annual)")
    ax5.set_ylabel("Registrations")
    ax5.set_xlabel("Year")
    ax5.tick_params(axis="x", rotation=45)
    ax5.legend(title="Type")
    fig5.tight_layout()
    fig5.savefig(OUT_DIR / "05_registrations_by_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)
    figs.append("05_registrations_by_type.png")

    # 6) BC Registrations vs Starts vs Completions (annual, overlapping years)
    fig6, ax6 = plt.subplots(figsize=(11, 5))
    bc_reg_yr = bc_reg.copy()
    ax6.plot(bc_reg_yr["Year"], bc_reg_yr["BC_Registrations_Total"], marker="o", label="Registrations", color=COLORS[2])
    merge_annual = annual_starts[["Year", "Total"]].merge(annual_compl[["Year", "Total"]], on="Year", suffixes=("_S", "_C"))
    ax6.plot(merge_annual["Year"], merge_annual["Total_S"], marker="s", label="Starts", color=COLORS[0])
    ax6.plot(merge_annual["Year"], merge_annual["Total_C"], marker="^", label="Completions", color=COLORS[1])
    ax6.set_title("BC Annual: Registrations vs Starts vs Completions")
    ax6.set_ylabel("Units")
    ax6.legend()
    ax6.tick_params(axis="x", rotation=45)
    fig6.tight_layout()
    fig6.savefig(OUT_DIR / "06_registrations_vs_starts_vs_completions.png", dpi=150, bbox_inches="tight")
    plt.close(fig6)
    figs.append("06_registrations_vs_starts_vs_completions.png")

    # 7) Lag analysis: Completions vs Lagged Starts (scatter + trend)
    fig7, ax7 = plt.subplots(figsize=(8, 6))
    sc = ax7.scatter(df_pred["Starts_Lag1"], df_pred["Completions"], c=df_pred["Year"], cmap="viridis", s=60)
    ax7.set_xlabel("Starts (Previous Year)")
    ax7.set_ylabel("Completions (Current Year)")
    ax7.set_title("BC: Completions vs Lag-1 Starts (Annual)")
    plt.colorbar(sc, ax=ax7, label="Year")
    z = np.polyfit(df_pred["Starts_Lag1"].dropna(), df_pred.loc[df_pred["Starts_Lag1"].notna(), "Completions"], 1)
    xl = np.array([df_pred["Starts_Lag1"].min(), df_pred["Starts_Lag1"].max()])
    ax7.plot(xl, np.poly1d(z)(xl), "r--", linewidth=2, label="Trend")
    ax7.legend()
    fig7.tight_layout()
    fig7.savefig(OUT_DIR / "07_completions_vs_lag1_starts.png", dpi=150, bbox_inches="tight")
    plt.close(fig7)
    figs.append("07_completions_vs_lag1_starts.png")

    # 8) Completions vs Lagged Registrations (scatter)
    fig8, ax8 = plt.subplots(figsize=(8, 6))
    valid = df_pred.dropna(subset=["Registrations_Lag1", "Completions"])
    if len(valid) >= 4:
        ax8.scatter(valid["Registrations_Lag1"], valid["Completions"], c=valid["Year"], cmap="viridis", s=60)
        ax8.set_xlabel("Registrations (Previous Year)")
        ax8.set_ylabel("Completions (Current Year)")
        ax8.set_title("BC: Completions vs Lag-1 Registrations (Annual)")
        z = np.polyfit(valid["Registrations_Lag1"], valid["Completions"], 1)
        xl = np.array([valid["Registrations_Lag1"].min(), valid["Registrations_Lag1"].max()])
        ax8.plot(xl, np.poly1d(z)(xl), "r--", linewidth=2)
    fig8.tight_layout()
    fig8.savefig(OUT_DIR / "08_completions_vs_lag1_registrations.png", dpi=150, bbox_inches="tight")
    plt.close(fig8)
    figs.append("08_completions_vs_lag1_registrations.png")

    # 9) Prediction comparison: Actual vs Predicted (Starts model vs Registrations model)
    fig9, ax9 = plt.subplots(figsize=(9, 6))
    if "Starts_Lag1" in models:
        m = models["Starts_Lag1"]
        ax9.scatter(m["y"], m["pred"], label="Starts Lag-1 model", alpha=0.8, s=80)
    if "Registrations_Lag1" in models:
        m = models["Registrations_Lag1"]
        ax9.scatter(m["y"], m["pred"], label="Registrations Lag-1 model", alpha=0.8, s=80)
    lims = [min(ax9.get_xlim()[0], ax9.get_ylim()[0]), max(ax9.get_xlim()[1], ax9.get_ylim()[1])]
    ax9.plot(lims, lims, "k--", label="Perfect prediction")
    ax9.set_xlabel("Actual Completions")
    ax9.set_ylabel("Predicted Completions")
    ax9.set_title("Prediction Comparison: Starts vs Registrations (Lag-1)")
    ax9.legend()
    ax9.set_aspect("equal")
    fig9.tight_layout()
    fig9.savefig(OUT_DIR / "09_prediction_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig9)
    figs.append("09_prediction_comparison.png")

    # 10) Time series: 12-month rolling mean Starts vs Completions
    bc_s = bc_starts.set_index("Date")["Total"]
    bc_c = bc_compl.set_index("Date")["Total"]
    roll_s = bc_s.rolling(12).mean()
    roll_c = bc_c.rolling(12).mean()
    fig10, ax10 = plt.subplots(figsize=(12, 5))
    ax10.plot(roll_s.index, roll_s, label="Starts (12-mo roll)", color=COLORS[0])
    ax10.plot(roll_c.index, roll_c, label="Completions (12-mo roll)", color=COLORS[1])
    ax10.set_title("BC Housing: 12-Month Rolling Average — Starts vs Completions")
    ax10.set_ylabel("Units")
    ax10.legend()
    ax10.tick_params(axis="x", rotation=45)
    fig10.tight_layout()
    fig10.savefig(OUT_DIR / "10_rolling_12mo_starts_vs_completions.png", dpi=150, bbox_inches="tight")
    plt.close(fig10)
    figs.append("10_rolling_12mo_starts_vs_completions.png")

    # 11) Share of dwelling types (pie or bar) — recent year
    fig11, ax11 = plt.subplots(figsize=(8, 5))
    last_year = bc_compl["Date"].dt.year.max()
    last = bc_compl[bc_compl["Date"].dt.year == last_year]
    parts = [last["Single"].sum(), last["Semi-Detached"].sum(), last["Row"].sum(), last["Apartment"].sum()]
    labels = ["Single", "Semi-Detached", "Row", "Apartment"]
    ax11.bar(labels, parts, color=COLORS[:4])
    ax11.set_title(f"BC Completions by Dwelling Type ({int(last_year)})")
    ax11.set_ylabel("Units")
    fig11.tight_layout()
    fig11.savefig(OUT_DIR / "11_completions_share_by_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig11)
    figs.append("11_completions_share_by_type.png")

    # 12) Pipeline gap: Starts - Completions (annual) — “under construction” proxy
    fig12, ax12 = plt.subplots(figsize=(11, 5))
    merge_annual = annual_starts[["Year", "Total"]].merge(annual_compl[["Year", "Total"]], on="Year", suffixes=("_S", "_C"))
    merge_annual["Gap"] = merge_annual["Total_S"] - merge_annual["Total_C"]
    ax12.bar(merge_annual["Year"], merge_annual["Gap"], color=COLORS[3], alpha=0.8)
    ax12.axhline(0, color="black", linewidth=0.5)
    ax12.set_title("BC Annual: Starts minus Completions (Pipeline Gap)")
    ax12.set_ylabel("Units (Starts − Completions)")
    ax12.set_xlabel("Year")
    ax12.tick_params(axis="x", rotation=45)
    fig12.tight_layout()
    fig12.savefig(OUT_DIR / "12_pipeline_gap_starts_minus_completions.png", dpi=150, bbox_inches="tight")
    plt.close(fig12)
    figs.append("12_pipeline_gap_starts_minus_completions.png")

    # 13) MAE comparison (bar) — prediction models
    fig13, ax13 = plt.subplots(figsize=(8, 5))
    mae_list = []
    for name, m in models.items():
        maee = mean_absolute_error(m["y"], m["pred"])
        r2 = r2_score(m["y"], m["pred"])
        mae_list.append({"Model": name, "MAE": maee, "R2": r2})
    mae_df = pd.DataFrame(mae_list)
    x = np.arange(len(mae_df))
    ax13.bar(x - 0.2, mae_df["MAE"], width=0.4, label="MAE", color=COLORS[0])
    ax13_twin = ax13.twinx()
    ax13_twin.bar(x + 0.2, mae_df["R2"], width=0.4, label="R²", color=COLORS[1], alpha=0.7)
    ax13.set_xticks(x)
    ax13.set_xticklabels(mae_df["Model"], rotation=30, ha="right")
    ax13.set_ylabel("MAE (Units)")
    ax13_twin.set_ylabel("R²")
    ax13.set_title("Prediction Model Comparison: MAE and R²")
    fig13.tight_layout()
    fig13.savefig(OUT_DIR / "13_mae_r2_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig13)
    figs.append("13_mae_r2_comparison.png")

    # 14) Correlation heatmap (annual BC: Starts, Completions, Registrations, lags)
    fig14, ax14 = plt.subplots(figsize=(8, 6))
    corr_df = df_pred[["Completions", "Starts", "Starts_Lag1", "Starts_Lag2", "Registrations", "Registrations_Lag1", "Registrations_Lag2"]].copy()
    corr_df = corr_df.rename(columns={"Starts_Lag1": "Starts_Lag1", "Starts_Lag2": "Starts_Lag2", "Registrations_Lag1": "Reg_Lag1", "Registrations_Lag2": "Reg_Lag2"})
    c = corr_df.corr()
    sns.heatmap(c, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax14, square=True)
    ax14.set_title("Correlation: Completions, Starts, Registrations & Lags (Annual BC)")
    fig14.tight_layout()
    fig14.savefig(OUT_DIR / "14_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig14)
    figs.append("14_correlation_heatmap.png")

    # 15) Long-term trend: BC Total Completions (full history)
    fig15, ax15 = plt.subplots(figsize=(12, 5))
    ax15.plot(bc_compl["Date"], bc_compl["Total"], color=COLORS[1], alpha=0.8)
    ax15.set_title("BC Housing Completions — Full History (Monthly)")
    ax15.set_ylabel("Units")
    ax15.tick_params(axis="x", rotation=45)
    fig15.tight_layout()
    fig15.savefig(OUT_DIR / "15_bc_completions_full_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig15)
    figs.append("15_bc_completions_full_history.png")

    # ---------- METRICS FOR REPORT ----------
    metrics = []
    for name, m in models.items():
        metrics.append({
            "Model": name,
            "MAE": mean_absolute_error(m["y"], m["pred"]),
            "R2": r2_score(m["y"], m["pred"]),
        })
    metrics_df = pd.DataFrame(metrics)

    # Write short summary
    with open(OUT_DIR / "prediction_metrics.csv", "w") as f:
        f.write(metrics_df.to_csv(index=False))

    print("Saved graphs:", figs)
    print("Metrics:\n", metrics_df.to_string(index=False))
    return figs, metrics_df, df_pred, models


if __name__ == "__main__":
    main()
