# stock-screener

A quantitative equity research system built on Indian stocks (Nifty 50 universe). Screens stocks using technical indicators, scores them, backtests the strategy month by month, and measures performance against the benchmark

Built from scratch as part of learning quant development — no libraries doing the heavy lifting, just raw logic

---

## What it does

1. Downloads historical price data for a set of Nifty 50 stocks
2. Computes indicators: SMA50, SMA200, EMA, RSI14
3. Scores each stock using a weighted signal model (trend + momentum)
4. Applies a bullish regime filter: price > SMA50 > SMA200 AND 40 < RSI < 70
5. Selects the top N stocks
6. Simulates holding them for 30 days and measures actual returns
7. Compares against Nifty 50 benchmark
8. Repeats this every month (walk-forward — no look-ahead bias)
9. Runs regime analysis (bull/bear/flat) and drawdown analysis on the results

The scoring model:
    score = 0.4 × MA_signal + 0.6 × RSI_signal

Range is -1.0 to +1.0. RSI gets more weight because momentum tends to lead price.

---

## Project structure

```
stock-screener/
├── src/
│   ├── screener.py            # main screening pipeline
│   ├── indicators.py          # SMA, EMA, RSI calculations
│   ├── scoring.py             # scoring model (StockScorer)
│   ├── backtest.py            # walk-forward backtester
│   ├── backtest_engine.py     # modular core engine (data-independent)
│   ├── walk_forward.py        # validation runner
│   ├── walkforward_metrics.py # sharpe, win rate, edge calculation
│   ├── portfolio_optimizer.py # volatility-adjusted position sizing
│   ├── performance_analyzer.py# equity curves, drawdown, score vs return
│   ├── regime_analysis.py     # bull/bear/flat market breakdown
│   ├── drawdown_analysis.py   # max drawdown, equity curve
│   ├── plot_walkforward.py    # plots equity curve from results
│   └── stock_list.py          # stock universe (15 Nifty 50 tickers)
├── outputs/
│   ├── backtests/             # backtest_results.csv, backtest_picks.csv
│   ├── plots/                 # equity curve png, drawdown csv
│   ├── walkforward/           # walkforward_results.csv, metrics.csv
│   ├── screener_results.csv
│   └── regime_summary.csv
├── tests/
│   ├── demo_indicators.py
│   └── test_screener_dates.py
├── logs/
│   └── screener.log
└── research/                  # (notes, experiments — not tracked)
```

---

## Setup:

git clone https://github.com/<your-username>/stock-screener
cd stock-screener

conda create -n quant python=3.11
conda activate quant

pip install pandas numpy yfinance matplotlib seaborn python-dateutil

---

## How to run
**Run the screener** (shows top stocks right now):
    python -m src.screener --top 5

**Run the backtest** (walks through last 12 months):
    python -m src.backtest

**Run walk-forward validation**:
    python -m src.walk_forward

**Analyze performance** (charts + metrics):
    python -m src.performance_analyzer

**Portfolio weights** (volatility-adjusted sizing):
    python -m src.portfolio_optimizer

---

## Output files

|            File                 |              What's in it                     |
|---------------------------------|-----------------------------------------------|
| `backtest_results.csv`          | Monthly portfolio return, Nifty return, alpha |
| `backtest_picks.csv`            | Per-stock: entry, exit, score, return         |
| `walkforward_results.csv`       | Same as above but from walk-forward runner    |
| `walkforward_metrics.csv`       | Sharpe, win rate, avg edge                    |
| `regime_summary.csv`            | Performance split by bull/bear/flat           |
| `drawdown_curve.csv`            | Equity and drawdown per period                |
| `walkforward_equity_curve.png`  | Portfolio vs Nifty equity curve               |

---

## Backtest parameters

Edit at the bottom of `src/backtest.py`:
    bt = CorrectBacktest(
        backtest_months=12,
        lookback_days=260,
        top_n=3,
        holding_days=30,
        start_year=2024,
        start_month=1,
    )

---

## Scoring weights

Edit in `src/scoring.py`:
    scorer = StockScorer(
        ma_weight=0.4,
        rsi_weight=0.6
    )

---

## Stock universe

`src/stock_list.py`. Currently 15 Nifty 50 large-caps across banking, IT, pharma, energy, and FMCG. To use the full list, pass `use_test=False` to `get_stock_list()`

---

## Metrics tracked

- Total return vs benchmark
- Sharpe ratio
- Sortino ratio 
- Max drawdown
- Win rate (profitable months)
- Beat rate (months > Nifty)
- Avg monthly alpha
- Score-to-return correlation (checks if the scoring model actually works)

---

## No look-ahead bias — how it works

Every month:
1. Screener downloads data with `end=screen_date` — it never sees the future
2. Returns are measured using data downloaded *after* `screen_date`
3. These two never overlap

This is what makes it a real backtest and not just curve-fitting.

---

## What's next

- [ ] Full Nifty 50 universe (currently 15 stocks)
- [ ] Transaction cost and slippage modeling
- [ ] More indicators (MACD, Bollinger Bands, ATR)
- [ ] Factor-based scoring (momentum, mean reversion, quality)
- [ ] Stop-loss logic
- [ ] C++ execution layer for low-latency work

---

## Context

2nd year CSE student building toward quant development. This project is an attempt to actually understand how a real quant pipeline works — not tutorials, just building it piece by piece. Still early, lots to improve, but the fundamentals (no look-ahead bias, proper walk-forward, real metrics) are there.

---
