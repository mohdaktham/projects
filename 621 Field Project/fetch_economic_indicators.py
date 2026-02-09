"""
Fetch Canadian economic indicators from FRED (pandas_datareader) and save to Data/Economic_Indicators_Canada.csv.
If fetch fails (no API key/network), generates synthetic annual data so the pipeline runs.
Run: py fetch_economic_indicators.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "Data"
OUT_CSV = DATA_DIR / "Economic_Indicators_Canada.csv"

# FRED series IDs for Canada (monthly where available; we aggregate to annual in the main script)
FRED_SERIES = {
    "Unemployment_Rate": "LRUN64TTCAM156S",      # Canada unemployment rate monthly
    "CPI_AllItems": "CANCPIALLMINMEI",            # Canada CPI (2015=100) monthly
    "Policy_Rate": "INTGSBCAM193N",               # Canada long-term gov bond yield (proxy for rates)
    "GDP_Growth": "NAEXKP01CAA661S",             # Canada real GDP (annual) - if exists
    "Interest_Rate_5Y": "IRLTLT01CAM156N",       # Canada long-term gov bond yield
    "Employment_Index": "LFEM64TTCAM647S",      # Canada employment (index)
    "Industrial_Production": "PRINTO01CAM661N",  # Canada industrial production
    "Consumer_Confidence": "CSCICP03CAM665S",   # Canada consumer confidence
    "Housing_Starts_Canada": "HOUSTCAA",          # Canada housing starts (total)
    "Building_Permits_Canada": "PERMITS",         # US - we'll use Canada equivalent if available
}

def fetch_fred():
    """Try to fetch from FRED via pandas_datareader. Returns None on failure."""
    try:
        import pandas_datareader as pdr
    except ImportError:
        print("pandas_datareader not installed. Install with: py -m pip install pandas-datareader")
        return None
    start = "1990-01-01"
    end = "2025-12-31"
    # Try a few key series that often work without API key
    to_try = ["LRUN64TTCAM156S", "CANCPIALLMINMEI", "INTGSBCAM193N", "IRLTLT01CAM156N"]
    out = None
    for sid in to_try:
        try:
            s = pdr.data.DataReader(sid, "fred", start=start, end=end)
            s = s.iloc[:, 0]
            s.name = sid
            if out is None:
                out = pd.DataFrame(s)
            else:
                out = out.join(s, how="outer")
        except Exception as e:
            print(f"FRED series {sid} failed: {e}")
    if out is not None and not out.empty:
        out.index = pd.to_datetime(out.index)
        out = out.resample("AS").mean()  # annual
        out["Year"] = out.index.year
        return out
    return None


def generate_synthetic_economic():
    """Generate synthetic but realistic annual economic data (1990-2024) for pipeline testing."""
    years = np.arange(1990, 2026)
    np.random.seed(42)
    # Trend + noise for each series
    unemployment = 7.5 - 0.08 * (years - 1990) + np.random.randn(len(years)) * 0.5
    unemployment = np.clip(unemployment, 4, 12)
    cpi = 100 * (1.02 ** (years - 2015)) + np.random.randn(len(years)) * 2
    cpi = np.clip(cpi, 80, 150)
    policy_rate = 6 - 0.06 * (years - 1990) + np.random.randn(len(years)) * 0.8
    policy_rate = np.clip(policy_rate, 0.25, 12)
    gdp_growth = 2.5 + 0.02 * np.sin((years - 1990) / 5) + np.random.randn(len(years)) * 1.5
    gdp_growth = np.clip(gdp_growth, -3, 6)
    employment_index = 90 + (years - 1990) * 0.8 + np.random.randn(len(years)) * 2
    employment_index = np.clip(employment_index, 85, 130)
    industrial_prod = 80 + (years - 1990) * 0.6 + np.random.randn(len(years)) * 3
    industrial_prod = np.clip(industrial_prod, 70, 120)
    consumer_conf = 95 + np.random.randn(len(years)) * 10
    consumer_conf = np.clip(consumer_conf, 70, 120)
    # Housing-related: correlate with our BC data conceptually
    housing_starts_can = 180000 + (years - 1990) * 500 + np.random.randn(len(years)) * 15000
    housing_starts_can = np.clip(housing_starts_can, 120000, 280000)
    population_growth = 1.2 + np.random.randn(len(years)) * 0.3
    population_growth = np.clip(population_growth, 0.5, 2.0)
    inflation_yoy = 2.0 + np.random.randn(len(years)) * 1.2
    inflation_yoy = np.clip(inflation_yoy, 0, 6)
    mortgage_rate_5y = 7 - 0.05 * (years - 1990) + np.random.randn(len(years)) * 0.5
    mortgage_rate_5y = np.clip(mortgage_rate_5y, 2, 12)

    df = pd.DataFrame({
        "Year": years,
        "Unemployment_Rate": unemployment,
        "CPI_Index": cpi,
        "Policy_Rate": policy_rate,
        "GDP_Growth_Pct": gdp_growth,
        "Employment_Index": employment_index,
        "Industrial_Production_Index": industrial_prod,
        "Consumer_Confidence_Index": consumer_conf,
        "Housing_Starts_Canada_Total": housing_starts_can,
        "Population_Growth_Pct": population_growth,
        "Inflation_YoY_Pct": inflation_yoy,
        "Mortgage_Rate_5Y_Pct": mortgage_rate_5y,
    })
    return df


def main():
    DATA_DIR.mkdir(exist_ok=True)
    df = fetch_fred()
    if df is not None:
        # Normalize column names to Year + indicators
        df = df.reset_index(drop=True)
        cols = ["Year"] + [c for c in df.columns if c != "Year"]
        df = df[cols]
        df.to_csv(OUT_CSV, index=False)
        print(f"Saved FRED data to {OUT_CSV} ({len(df)} rows)")
    else:
        df = generate_synthetic_economic()
        df.to_csv(OUT_CSV, index=False)
        print(f"Saved synthetic economic data to {OUT_CSV} (replace with real data from Stats Canada / FRED)")
    return df


if __name__ == "__main__":
    main()
