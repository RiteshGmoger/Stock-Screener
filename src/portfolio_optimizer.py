import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yfinance as yf


class VolatilityPortfolio:
    def __init__(self, target_vol=0.15):
        self.target_vol = target_vol

    def build(self, stocks):
        rows = []
        inv_vol_sum = 0.0

        for s in stocks:
            df = yf.download(
                s["ticker"],
                period="60d",
                progress=False,
                auto_adjust=True
            )

            if df.empty or len(df) < 20:
                continue

            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            returns = close.pct_change().dropna()

            # 🔒 FORCE SCALAR
            vol = float(returns.std() * np.sqrt(252))

            if vol <= 0:
                continue

            edge = float((s["score"] - 50) / 1000)

            # 🔒 FORCE SCALAR
            kelly_raw = edge / (vol ** 2)
            kelly = min(max(kelly_raw, 0.0), 0.25)

            inv_vol = 1.0 / (vol + 0.05)
            inv_vol_sum += inv_vol

            rows.append({
                "ticker": s["ticker"],
                "vol": vol,
                "edge": edge,
                "kelly": kelly,
                "inv_vol": inv_vol
            })

        if not rows:
            print("No valid stocks")
            return

        # Base weights
        for r in rows:
            r["weight"] = (r["inv_vol"] / inv_vol_sum) * (1 + r["kelly"]) / 2

        # Normalize
        total_weight = sum(r["weight"] for r in rows)
        for r in rows:
            r["weight"] /= total_weight
            r["weight"] = min(r["weight"], 0.40)

        out = pd.DataFrame(rows)
        out.to_csv("portfolio_weights.csv", index=False)

        print("✓ Saved: portfolio_weights.csv")
        print(out[["ticker", "weight", "vol", "kelly"]])


if __name__ == "__main__":
    portfolio = VolatilityPortfolio()

    example = [
        {"ticker": "RELIANCE.NS", "score": 85},
        {"ticker": "TCS.NS", "score": 82},
        {"ticker": "INFY.NS", "score": 79},
    ]

    portfolio.build(example)

