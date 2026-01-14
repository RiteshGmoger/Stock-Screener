import yfinance as yf
import pandas as pd
from src.indicators import (
    calculate_moving_average,
    calculate_rsi,
    get_ma_signal,
    get_rsi_signal,
    combine_signals,
)

print("=" * 80)
print("TESTING INDICATORS ON REAL DATA")
print("=" * 80)

# --------------------------------------------------
# 1) Download real stock data
# --------------------------------------------------
print("\n[1/5] Downloading RELIANCE.NS data...")
data = yf.download(
    "RELIANCE.NS",
    start="2024-01-01",
    end="2025-01-09",
    progress=False,
)
print(f"✓ Downloaded {len(data)} days of data")

close_prices = data["Close"]
if isinstance(close_prices, pd.DataFrame):
    close_prices = close_prices.iloc[:, 0]

print("\n[2/5] Calculating MA50...")
ma50 = calculate_moving_average(close_prices, window=50)
print("✓ MA50 calculated")
print(ma50.tail(5))

print("\n[3/5] Calculating RSI14...")
rsi14 = calculate_rsi(close_prices, period=14)
print("✓ RSI14 calculated")
print(rsi14.tail(5))

print("\n[4/5] Generating signals for latest day...")

latest_price = float(close_prices.iloc[-1])
latest_ma50 = float(ma50.iloc[-1])
latest_rsi14 = float(rsi14.iloc[-1])

print(f"Latest price: {latest_price:.2f}")
print(f"Latest MA50: {latest_ma50:.2f}")
print(f"Latest RSI14: {latest_rsi14:.2f}")

ma_signal = get_ma_signal(latest_price, latest_ma50)
rsi_signal = get_rsi_signal(latest_rsi14)
combined_score = combine_signals(ma_signal, rsi_signal)

print(f"MA Signal: {ma_signal}")
print(f"RSI Signal: {rsi_signal}")
print(f"Combined Score: {combined_score:+.2f}")

if combined_score >= 0.7:
    print("→ STRONG BUY")
elif combined_score >= 0.3:
    print("→ MODERATE BUY")
elif combined_score <= -0.3:
    print("→ AVOID / SELL")
else:
    print("→ NEUTRAL")

print("\n[5/5] Backtest last 20 days")
print("-" * 80)
print("Date       | Price    | MA50     | RSI    | Score  | Action")
print("-" * 80)

for i in range(-20, 0):
    price = close_prices.iloc[i]
    ma = ma50.iloc[i]
    rsi = rsi14.iloc[i]

    if pd.isna(ma) or pd.isna(rsi):
        continue

    ma_sig = get_ma_signal(float(price), float(ma))
    rsi_sig = get_rsi_signal(float(rsi))
    score = combine_signals(ma_sig, rsi_sig)

    action = "BUY" if score >= 0.3 else "SELL" if score <= -0.3 else "HOLD"
    date = close_prices.index[i].strftime("%Y-%m-%d")

    print(
        f"{date} | {price:8.2f} | {ma:8.2f} | {rsi:6.2f} | {score:+6.2f} | {action}"
    )

print("-" * 80)
print("\n✓ Indicator testing complete")

