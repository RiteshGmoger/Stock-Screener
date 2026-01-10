# Stock Screener

Goal:
Screen 50 stocks and rank the top 10 based on trend and momentum.

Indicators used:
- 50-day Moving Average (trend)
- RSI (momentum)

Scoring logic:
Score = 0.4 * MA_signal + 0.6 * RSI_signal

Filters:
- Average volume > 1M
- Price > 50
- 50-day return > -20%

Status:
Day 1 – Design completed. Coding starts next.

