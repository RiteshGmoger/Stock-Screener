import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("walkforward_results.csv")

df["port_cum"] = (1 + df["portfolio"] / 100).cumprod()
df["nifty_cum"] = (1 + df["nifty"] / 100).cumprod()

plt.figure(figsize=(10, 6))
plt.plot(df["port_cum"], label="Portfolio")
plt.plot(df["nifty_cum"], label="Nifty")
plt.legend()
plt.title("Walk-Forward Equity Curve")
plt.grid(True)

plt.savefig("walkforward_equity_curve.png")
print("✓ Saved: walkforward_equity_curve.png")
