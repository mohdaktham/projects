"""Quick inspect of Housing Registration Data.xlsx structure."""
import pandas as pd
from pathlib import Path

path = Path(__file__).resolve().parent / "Data" / "Housing Registration Data.xlsx"
xl = pd.ExcelFile(path)
print("Sheet names:", xl.sheet_names)
for name in xl.sheet_names:
    df = pd.read_excel(path, sheet_name=name, header=None)
    print(f"\n--- Sheet: {name} ---")
    print("Shape:", df.shape)
    print(df.head(15).to_string())
