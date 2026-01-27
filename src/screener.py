import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from src.indicators import (
    calculate_moving_average,
    calculate_rsi,
)

from src.scoring import StockScorer
from src.stock_list import TEST_TICKERS


class StockScreener:
    def __init__(self, tickers, lookback_days=260):
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.data = {}
        self.indicators = {}
        self.results = None
        self.output_file = "screener_results.csv"
        self.scorer = StockScorer(ma_weight=0.4, rsi_weight=0.6)

    def download_data(self):
        print("✦" * 80)
        print("STEP 1: DOWNLOAD DATA".center(80))
        print("✦" * 80)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        for i, ticker in enumerate(self.tickers, 1):
            try:
                print(f"[{i:2d}/{len(self.tickers)}] {ticker:20s}", end=" ")

                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                )

                if df.empty:
                    print("NO DATA")
                    continue

                self.data[ticker] = df
                print(f"✓ {len(df)} rows")

            except Exception as e:
                print(f"{str(e)[:30]}")

        print(f"\n✓ Downloaded {len(self.data)}/{len(self.tickers)} stocks\n")

    def calculate_indicators(self):
        print("✦" * 80)
        print("STEP 2: CALCULATE INDICATORS".center(80))
        print("✦" * 80)

        for i, (ticker, df) in enumerate(self.data.items(), 1):
            try:
                print(f"[{i:2d}/{len(self.data)}] {ticker:20s}", end=" ")

                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                ma50 = calculate_moving_average(close, 50)
                rsi14 = calculate_rsi(close, 14)

                self.indicators[ticker] = {
                    "Close": close,
                    "MA50": ma50,
                    "RSI14": rsi14,
                }

                print("✓")

            except Exception as e:
                print(f"{str(e)[:30]}")

        print(f"\n✓ Indicators calculated for {len(self.indicators)} stocks\n")
        
    def generate_signals(self):
        print("\n" + "✦" * 80)
        print("STEP 3: GENERATE SIGNALS + SCORE")
        print("✦" * 80 + "\n")

        results_list = []

        for ticker, ind in self.indicators.items():
            try:
                close = ind["Close"]
                ma50 = ind["MA50"]
                rsi14 = ind["RSI14"]

                latest_close = close.iloc[-1]
                latest_ma50 = ma50.iloc[-1]
                latest_rsi14 = rsi14.iloc[-1]

                if pd.isna(latest_ma50) or pd.isna(latest_rsi14):
                    continue

                score = self.scorer.calculate_score(
                    latest_close, latest_ma50, latest_rsi14
                )
                signal = self.scorer.get_interpretation(score)

                ma_diff_pct = (latest_close - latest_ma50) / latest_ma50 * 100

                results_list.append({
                    "Ticker": ticker,
                    "Price": round(latest_close, 2),
                    "MA50": round(latest_ma50, 2),
                    "MA_Diff_%": round(ma_diff_pct, 2),
                    "RSI14": round(latest_rsi14, 2),
                    "Score": score,
                    "Signal": signal
                })

                print(f"✓ {ticker:15s} | Score {score:+.2f} | {signal}")

            except Exception as e:
                print(f"✗ {ticker}: {str(e)[:40]}")

        self.results = pd.DataFrame(results_list)
        self.results = self.results.sort_values("Score", ascending=False)
        self.results["Rank"] = range(1, len(self.results) + 1)

        print(f"\n✓ Scored {len(self.results)} stocks\n")

    def export_results(self):
        """Export scored and ranked stocks."""
        print("✦" * 80)
        print("STEP 4: EXPORT RESULTS")
        print("✦" * 80 + "\n")
        
        if self.results is None or self.results.empty:
            print("❌ No results to export\n")
            return
        
        self.results.to_csv(self.output_file, index=False)
        print(f"✓ Exported: {self.output_file}\n")
        
        # Print top 10
        print("TOP 10 STOCKS (by Score):")
        print("✦" * 80)
        cols = ['Rank', 'Ticker', 'Price', 'MA50', 'RSI14', 'Score', 'Signal']
        print(self.results[cols].head(10).to_string(index=False))
        print("✦" * 80 + "\n")

    def run(self):
        self.download_data()
        self.calculate_indicators()
        self.generate_signals()
        self.export_results()
        print("\n✓ SCREENER COMPLETE\n")


if __name__ == "__main__":
    screener = StockScreener(TEST_TICKERS, lookback_days=260)
    screener.run()

