"""
Fetch Canadian economic indicators from FRED and save to Data/Economic_Indicators_Canada.csv.
Uses real-world data only. Tries FRED CSV download (no API key), then pandas_datareader if available.
Run: py fetch_economic_indicators.py
Use --allow-synthetic only for testing without network/FRED.
"""
import sys
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "Data"
OUT_CSV = DATA_DIR / "Economic_Indicators_Canada.csv"

# Friendly name -> FRED series ID (Canada). Same names expected by bc_housing_forecasting.
FRED_SERIES = {
    "Unemployment_Rate": "LRUN64TTCAM156S",
    "CPI_Index": "CANCPIALLMINMEI",
    "Policy_Rate": "INTGSBCAM193N",
    "Interest_Rate_5Y": "IRLTLT01CAM156N",
    "Employment_Index": "LFEM64TTCAM647S",
    "Housing_Starts_Canada_Total": "HOUSTCAA",
}

# FRED CSV download URL (no API key required for public series)
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _fetch_fred_csv_series(series_id):
    """Fetch one FRED series as CSV via standard download URL. Returns Series with DatetimeIndex or None."""
    try:
        url = FRED_CSV_URL.format(series_id=series_id)
        req = urllib.request.Request(url, headers={"User-Agent": "BC-Housing-Analysis/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            df = pd.read_csv(resp, skiprows=1, names=["DATE", "VALUE"])
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE", "VALUE"])
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
        df = df.dropna(subset=["VALUE"])
        if df.empty:
            return None
        return df.set_index("DATE")["VALUE"]
    except Exception:
        return None


def fetch_fred():
    """Fetch from FRED (CSV URL first, then pandas_datareader). Returns DataFrame with friendly column names or None."""
    # Method 1: FRED graph CSV (no API key)
    out = None
    for friendly_name, sid in FRED_SERIES.items():
        s = _fetch_fred_csv_series(sid)
        if s is not None and len(s) > 0:
            s.name = friendly_name
            if out is None:
                out = pd.DataFrame(s)
            else:
                out = out.join(s, how="outer")
    if out is not None and not out.empty:
        out = out[~out.index.duplicated(keep="first")]
        out.index = pd.to_datetime(out.index)
        out = out.resample("YS").mean()
        out["Year"] = out.index.year
        out = out.reset_index(drop=True)
        cols = ["Year"] + [c for c in out.columns if c != "Year"]
        return out[cols]

    # Method 2: pandas_datareader (may fail on some Python/pandas versions)
    try:
        import pandas_datareader as pdr
    except ImportError:
        return None
    except Exception:
        return None
    start, end = "1990-01-01", "2025-12-31"
    out = None
    for friendly_name, sid in FRED_SERIES.items():
        try:
            s = pdr.data.DataReader(sid, "fred", start=start, end=end)
            s = s.iloc[:, 0]
            s.name = friendly_name
            if out is None:
                out = pd.DataFrame(s)
            else:
                out = out.join(s, how="outer")
        except Exception:
            pass
    if out is None or out.empty:
        return None
    out.index = pd.to_datetime(out.index)
    out = out.resample("YS").mean()
    out["Year"] = out.index.year
    out = out.reset_index(drop=True)
    cols = ["Year"] + [c for c in out.columns if c != "Year"]
    return out[cols]


def generate_synthetic_economic():
    """Synthetic data only when --allow-synthetic is passed (for testing)."""
    import numpy as np
    years = np.arange(1990, 2026)
    np.random.seed(42)
    return pd.DataFrame({
        "Year": years,
        "Unemployment_Rate": np.clip(7.5 - 0.08 * (years - 1990) + np.random.randn(len(years)) * 0.5, 4, 12),
        "CPI_Index": np.clip(100 * (1.02 ** (years - 2015)) + np.random.randn(len(years)) * 2, 80, 150),
        "Policy_Rate": np.clip(6 - 0.06 * (years - 1990) + np.random.randn(len(years)) * 0.8, 0.25, 12),
        "GDP_Growth_Pct": np.clip(2.5 + 0.02 * np.sin((years - 1990) / 5) + np.random.randn(len(years)) * 1.5, -3, 6),
        "Employment_Index": np.clip(90 + (years - 1990) * 0.8 + np.random.randn(len(years)) * 2, 85, 130),
        "Industrial_Production_Index": np.clip(80 + (years - 1990) * 0.6 + np.random.randn(len(years)) * 3, 70, 120),
        "Consumer_Confidence_Index": np.clip(95 + np.random.randn(len(years)) * 10, 70, 120),
        "Housing_Starts_Canada_Total": np.clip(180000 + (years - 1990) * 500 + np.random.randn(len(years)) * 15000, 120000, 280000),
        "Population_Growth_Pct": np.clip(1.2 + np.random.randn(len(years)) * 0.3, 0.5, 2.0),
        "Inflation_YoY_Pct": np.clip(2.0 + np.random.randn(len(years)) * 1.2, 0, 6),
        "Mortgage_Rate_5Y_Pct": np.clip(7 - 0.05 * (years - 1990) + np.random.randn(len(years)) * 0.5, 2, 12),
    })


def main():
    allow_synthetic = "--allow-synthetic" in sys.argv
    DATA_DIR.mkdir(exist_ok=True)

    df = fetch_fred()
    if df is not None:
        df.to_csv(OUT_CSV, index=False)
        print(f"Saved real FRED data to {OUT_CSV} ({len(df)} rows, columns: {list(df.columns)})")
        return df

    if allow_synthetic:
        df = generate_synthetic_economic()
        df.to_csv(OUT_CSV, index=False)
        print(f"Saved synthetic economic data (testing only) to {OUT_CSV}")
        return df

    print("Real economic data could not be fetched (check network and FRED access).")
    print("Install: py -m pip install pandas-datareader")
    print("Or add Data/Economic_Indicators_Canada.csv with columns: Year, Unemployment_Rate, CPI_Index, etc.")
    print("To run with synthetic data for testing only, use: py fetch_economic_indicators.py --allow-synthetic")
    sys.exit(1)


if __name__ == "__main__":
    main()
