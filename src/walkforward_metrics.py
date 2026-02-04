import pandas as pd
import numpy as np

df = pd.read_csv("walkforward_results.csv")

# Basic metrics
portfolio = df["portfolio"]
nifty = df["nifty"]

metrics = {
    "portfolio_mean": portfolio.mean(),
    "portfolio_std": portfolio.std(),
    "portfolio_sharpe": portfolio.mean() / portfolio.std() if portfolio.std() != 0 else 0,
    "nifty_mean": nifty.mean(),
    "nifty_std": nifty.std(),
    "win_rate": (portfolio > nifty).mean(),
    "avg_edge": (portfolio - nifty).mean(),
    "total_edge": (portfolio - nifty).sum(),
}

out = pd.DataFrame(metrics, index=[0])
out.to_csv("walkforward_metrics.csv", index=False)

print("✓ Saved: walkforward_metrics.csv")
print(out.to_string(index=False))

