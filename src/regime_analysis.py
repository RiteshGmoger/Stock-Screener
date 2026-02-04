import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "walkforward_results.csv"

if not csv_path.exists():
    csv_path = ROOT / "src" / "walkforward_results.csv"

df = pd.read_csv(csv_path)

def regime(n):
    if n > 2:
        return "Bull"
    elif n < -2:
        return "Bear"
    else:
        return "Flat"

df["regime"] = df["nifty"].apply(regime)
df["edge"] = df["portfolio"] - df["nifty"]

summary = df.groupby("regime").agg({
    "portfolio": "mean",
    "nifty": "mean",
    "edge": "mean"
})

print(summary)

summary.to_csv(ROOT / "regime_summary.csv")
print("✓ Saved: regime_summary.csv")

