import pandas as pd
import numpy as np
from pathlib import Path

# Resolve path safely
ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "walkforward_results.csv"

if not csv_path.exists():
    csv_path = ROOT / "src" / "walkforward_results.csv"

df = pd.read_csv(csv_path)

# Equity curve
equity = (1 + df["portfolio"] / 100).cumprod()

# Drawdown
rolling_max = equity.cummax()
drawdown = (equity - rolling_max) / rolling_max

max_dd = drawdown.min()

print(f"Max Drawdown: {max_dd * 100:.2f}%")

out = pd.DataFrame({
    "equity": equity,
    "drawdown": drawdown
})

out.to_csv(ROOT / "drawdown_curve.csv", index=False)
print("✓ Saved: drawdown_curve.csv")

