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

## Scoring Logic

The screener converts technical indicators into a **single numerical score**
so stocks can be **ranked objectively** instead of using vague buy/sell labels.

This score combines:
- **Trend** (Moving Average)
- **Momentum** (RSI)

into **one number between -1.0 and +1.0**.

---

### Scoring Formula

Score = 0.4 × MA_Signal + 0.6 × RSI_Signal


- MA contributes **40%** (trend stability)
- RSI contributes **60%** (short-term momentum)

---

### MA Signal (Trend)

MA50 = 50-day moving average

| Condition | MA Signal | Meaning |
|---------|----------|--------|
| Price > MA50 by more than 1% | +1 | Uptrend (bullish) |
| Price within ±1% of MA50 | 0 | Neutral |
| Price < MA50 by more than 1% | -1 | Downtrend (bearish) |

---

### RSI Signal (Momentum)

RSI14 measures buying pressure.

| RSI Value | RSI Signal | Meaning |
|---------|-----------|--------|
| RSI > 60 | +1.0 | Strong buying momentum |
| 40 ≤ RSI ≤ 60 | 0.0 | Balanced / neutral |
| RSI < 40 | -0.5 | Weak momentum (oversold risk) |

---

### Final Score Range

- **+1.0** → strongest possible bullish setup  
- **-1.0** → strongest bearish / avoid setup  

---

### Interpretation of Score

| Score Range | Signal | Meaning |
|------------|-------|--------|
| ≥ +0.7 | 🔥 Strong Buy | Trend + momentum aligned |
| +0.3 to +0.7 | 👍 Moderate Buy | Partial confirmation |
| 0 to +0.3 | 📈 Weak Buy | Low confidence |
| -0.3 to 0 | ➖ Hold | No edge |
| < -0.3 | ⛔ Avoid / Sell | Weak trend or momentum |

---

### Examples

#### Example 1 — Uptrend + Strong Momentum
Price: 500
MA50: 480
RSI: 65

MA signal: +1 (4.2% above MA)
RSI signal: +1.0
Score: 0.4(+1) + 0.6(+1) = 1.0
→ Strong Buy


#### Example 2 — Downtrend + Weak Momentum
Price: 300
MA50: 310
RSI: 35

MA signal: -1
RSI signal: -0.5
Score: 0.4(-1) + 0.6(-0.5) = -0.7
→ Avoid / Sell


#### Example 3 — Neutral Market
Price: 400
MA50: 405
RSI: 50

MA signal: 0
RSI signal: 0.0
Score: 0.0
→ Hold


---

### Why These Weights?

- **RSI (60%)** → captures short-term buying pressure  
- **MA (40%)** → provides structural trend confirmation  

Momentum matters more for timing,  
trend matters more for safety.

Weights are configurable and can be adjusted after backtesting.
