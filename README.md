# 📈 Quantitative Stock Screener & Backtester

A professional-grade quantitative trading system for Indian equities with proper backtesting methodology and zero look-ahead bias.

## 🎯 What This System Does

This is a complete quant trading pipeline that:
1. **Screens stocks** using technical indicators (MA, RSI)
2. **Scores stocks** using a weighted model
3. **Backtests strategies** without look-ahead bias
4. **Analyzes performance** with detailed metrics and visualizations

## 📦 Project Structure

```
stock-screener/
├── src/
│   ├── backtest.py          # Enhanced backtest engine
│   ├── screener.py          # Stock screening system
│   ├── indicators.py        # Technical indicators (MA, RSI)
│   ├── scoring.py           # Stock scoring model
│   ├── filters.py           # (empty, for future filters)
│   └── stock_list.py        # Stock universe
├── performance_analyzer.py  # Performance visualization
├── backtest_results.csv     # Monthly results (generated)
├── backtest_picks.csv       # Detailed picks (generated)
└── README.md               # This file
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv quant
source quant/bin/activate  # On Windows: quant\Scripts\activate

# Install dependencies
pip install pandas numpy yfinance matplotlib seaborn python-dateutil
```

### Run Backtest

```bash
# Run the enhanced backtest
python -m src.backtest

# Or use the original location
python backtest.py
```

### Analyze Results

```bash
# Generate performance report with charts
python performance_analyzer.py
```

## 🔧 Configuration

### Backtest Parameters

Edit in `backtest.py`:

```python
bt = CorrectBacktest(
    backtest_months=12,    # Number of months to test
    lookback_days=260,     # Historical data for indicators
    top_n=3,              # Number of stocks to hold
    holding_days=30,      # Days to hold each position
    start_year=2024,      # Backtest start year
    start_month=2         # Backtest start month
)
```

### Scoring Weights

Edit in `scoring.py`:

```python
scorer = StockScorer(
    ma_weight=0.4,   # Weight for trend (MA50)
    rsi_weight=0.6   # Weight for momentum (RSI)
)
```

### Stock Universe

Edit `stock_list.py` to add/remove stocks:

```python
TEST_TICKERS = [
    'RELIANCE.NS',
    'TCS.NS',
    # Add more tickers here
]
```

## 📊 Understanding the Output

### 1. Backtest Results (`backtest_results.csv`)

| Month    | Portfolio_Return_% | Nifty_Return_% | Outperformance_% | Num_Stocks |
|----------|-------------------|----------------|------------------|------------|
| Feb 2024 | +5.23             | +3.45          | +1.78            | 3          |

- **Portfolio_Return_%**: Your strategy's return
- **Nifty_Return_%**: Benchmark (Nifty 50) return
- **Outperformance_%**: Alpha (Portfolio - Benchmark)
- **Num_Stocks**: Number of stocks held that month

### 2. Detailed Picks (`backtest_picks.csv`)

| Month    | Ticker      | Score | Entry_Price | Exit_Price | Return_% |
|----------|-------------|-------|-------------|------------|----------|
| Feb 2024 | TCS.NS      | 0.88  | 3500.00     | 3675.00    | +5.00    |

### 3. Performance Metrics

The summary shows:
- **Total Return**: Cumulative returns over entire period
- **Win Rate**: % of profitable months
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline
- **Beat Rate**: % of months beating benchmark

## 📈 Generated Charts

After running the analyzer, you'll get:

1. **cumulative_returns.png** - Portfolio vs Benchmark over time
2. **monthly_returns.png** - Bar chart of monthly performance
3. **outperformance.png** - Alpha (excess returns) each month
4. **drawdown.png** - Underwater chart showing losses
5. **stock_frequency.png** - Most selected stocks
6. **score_vs_return.png** - Score effectiveness validation

## 🧠 Key Concepts (For Learning)

### 1. Look-Ahead Bias Prevention

**WRONG WAY (Look-ahead bias):**
```python
# Using all data including future
df = download_all_data()
picks = screen_stocks(df)  # Uses future prices!
```

**RIGHT WAY (No bias):**
```python
# Only use data before screen date
df = download_data(end=screen_date)  # Historical only
picks = screen_stocks(df)
# THEN download future to measure returns
```

### 2. Scoring System

The score combines two signals:

**MA Signal (Trend):**
- Price > MA50 + 1% → Score: +1 (uptrend)
- Price near MA50 → Score: 0 (neutral)
- Price < MA50 - 1% → Score: -1 (downtrend)

**RSI Signal (Momentum):**
- RSI > 60 → Score: +1 (strong momentum)
- 40 ≤ RSI ≤ 60 → Score: 0 (neutral)
- RSI < 40 → Score: -0.5 (weak/oversold)

**Final Score:**
```
Score = 0.4 × MA_Signal + 0.6 × RSI_Signal
```

Range: -1.0 to +1.0

### 3. Signal Interpretation

| Score    | Signal       | Meaning                              |
|----------|--------------|--------------------------------------|
| ≥ 0.7    | 🔥 STRONG BUY | Trend up + Strong momentum          |
| 0.3-0.7  | 👍 BUY        | Something positive                   |
| -0.3-0.3 | ➖ HOLD       | Neutral, no edge                     |
| < -0.3   | ⛔ SELL       | Trend broken or weak momentum        |

### 4. Walk-Forward Logic

The backtest walks forward month by month:

```
Jan 2024: Screen stocks → Select top 3 → Hold 30 days → Measure return
Feb 2024: Screen stocks → Select top 3 → Hold 30 days → Measure return
...
```

Each month is independent, preventing data leakage.

## 🎓 Learning Path

### Phase 1: Understand the Code (Week 1-2)
1. Read `indicators.py` - How MA and RSI work
2. Read `scoring.py` - How signals combine
3. Read `screener.py` - How stocks are selected
4. Read `backtest.py` - How testing works

### Phase 2: Experiment (Week 3-4)
1. Change scoring weights (try 0.5/0.5, 0.3/0.7)
2. Add new indicators (MACD, Bollinger Bands)
3. Change holding period (15 days, 60 days)
4. Test different universes (Nifty Next 50, Bank Nifty)

### Phase 3: Advanced (Week 5-8)
1. Add volume filters
2. Implement stop-loss logic
3. Add position sizing (risk management)
4. Test mean reversion strategies
5. Implement ensemble models

## ⚠️ Common Mistakes to Avoid

### 1. Look-Ahead Bias
```python
# WRONG
df = download_all_data()
signals = calculate_signals(df)  # Uses future!

# RIGHT
df = download_data(end=today)
signals = calculate_signals(df)
```

### 2. Survivorship Bias
- Only testing stocks that still exist today
- Solution: Include delisted stocks in universe

### 3. Overfitting
- Creating complex rules that fit past perfectly
- Solution: Keep it simple, test out-of-sample

### 4. Transaction Costs
- Ignoring brokerage, slippage, taxes
- Solution: Deduct realistic costs from returns

## 📚 Next Steps

1. **Add More Indicators**
   - MACD (trend following)
   - Bollinger Bands (volatility)
   - ATR (volatility measure)

2. **Implement Filters**
   - Volume filter (liquidity)
   - Market cap filter
   - Sector diversification

3. **Risk Management**
   - Position sizing (Kelly Criterion)
   - Stop-loss rules
   - Portfolio heat limits

4. **Machine Learning**
   - Random Forest for scoring
   - Feature engineering
   - Walk-forward optimization

## 🐛 Troubleshooting

### Error: "name 'month_str' is not defined"
**Solution:** Use the fixed `backtest.py` file provided

### Error: "No module named 'src'"
**Solution:** Run from project root directory

### Error: Data download fails
**Solution:** Check internet connection, verify ticker symbols

### No results showing
**Solution:** Check date range - ensure there's historical data available

## 📖 Resources

- **QuantConnect**: Learn algorithmic trading
- **Quantopian Lectures**: Free quant finance course
- **Python for Finance**: Book by Yves Hilpisch
- **NSE India**: Get list of all Indian stocks

## 🤝 Contributing Ideas

1. Add more technical indicators
2. Implement fundamental filters
3. Add sector rotation strategy
4. Build portfolio optimization
5. Create live trading integration

## ⚡ Performance Tips

1. **Cache data**: Don't re-download same data
2. **Parallel processing**: Screen stocks in parallel
3. **Vectorization**: Use numpy instead of loops
4. **Database**: Store historical data in SQLite

## 🎯 Real Trading Checklist

Before going live:
- [ ] Backtest on 5+ years of data
- [ ] Test on out-of-sample period
- [ ] Account for transaction costs
- [ ] Add slippage modeling
- [ ] Implement risk limits
- [ ] Paper trade for 3+ months
- [ ] Have contingency plans

---

## Risk & Regime Analysis
- Max drawdown: X%
- Bull markets: +Y% monthly edge
- Bear markets: +Z% capital preservation
- Flat markets: no overtrading (near-zero exposure)
- Position sizing: half-Kelly + volatility targeting
