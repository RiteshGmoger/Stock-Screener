## How the Screener Works

### Step 1: Download Data
Downloads ~1 year (260 days) of historical price data for each stock using Yahoo Finance.

### Step 2: Calculate Indicators
For each stock, calculates:
- **MA50**: 50-day moving average (trend direction)
- **RSI14**: 14-period RSI (momentum strength)

### Step 3: Generate Signals
Signals are generated from indicators:
- **MA signal**: price compared to MA50 (trend)
- **RSI signal**: momentum strength

Final score:
Score = 0.4 × MA_signal + 0.6 × RSI_signal

Where:
- **MA_signal**: +1 if price > MA50, -1 if price < MA50, 0 if near
- **RSI_signal**: +1 if RSI > 60, 0 if 40–60, -0.5 if < 40

### Step 4: Rank & Output
Stocks are ranked by **Combined_Score** (higher = stronger buy signal).

## Output
- **screener_results.csv** contains:
  - Ticker
  - Price
  - MA50
  - RSI14
  - Combined_Score
  - Rank

## Interpretation
- Score ≥ +0.7 → 🔥 STRONG BUY 
- Score +0.3 to +0.7 → 👍 BUY 
- Score -0.3 to +0.3 → ➖ HOLD 
- Score ≤ -0.3 → ⛔ SELL
