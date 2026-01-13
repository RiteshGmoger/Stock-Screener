"""
Stock Screener: Complete pipeline.
1. Download historical stock data
2. Calculate MA50 and RSI14
3. Generate buy/sell signals
4. Rank and output results
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from indicators import (
    calculate_moving_average,
    calculate_rsi,
    get_signal_score,
)
from stock_list import TEST_TICKERS


class StockScreener:
    def __init__(self, tickers, lookback_days=260):
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.data = {}
        self.indicators = {}
        self.results = None
        self.output_file = "screener_results.csv"

    def download_data(self):
        print("=" * 80)
        print("STEP 1: DOWNLOAD DATA".center(80))
        print("=" * 80)
        print(f"Downloading {self.lookback_days} days of data for {len(self.tickers)} stocks...\n")

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
                    auto_adjust=True
                )

                if df.empty:
                    print("❌ NO DATA")
                    continue

                self.data[ticker] = df
                print(f"✓ {len(df)} rows")

            except Exception as e:
                print(f"❌ ERROR: {str(e)[:40]}")

        print(f"\n✓ Downloaded {len(self.data)}/{len(self.tickers)} stocks\n")

    def calculate_indicators(self):
        print("=" * 80)
        print("STEP 2: CALCULATE INDICATORS".center(80))
        print("=" * 80)

        for i, (ticker, df) in enumerate(self.data.items(), 1):
            try:
                print(f"[{i:2d}/{len(self.data)}] {ticker:20s}", end=" ")

                close = df["Close"]

                ma50 = calculate_moving_average(close, 50)
                rsi14 = calculate_rsi(close, 14)

                self.indicators[ticker] = {
                    "Close": close,
                    "MA50": ma50,
                    "RSI14": rsi14,
                }

                print("✓")

            except Exception as e:
                print(f"❌ {str(e)[:30]}")

        print(f"\n✓ Indicators calculated for {len(self.indicators)} stocks\n")

    def generate_signals(self):
        print("=" * 80)
        print("STEP 3: GENERATE SIGNALS".center(80))
        print("=" * 80)


        results = []

        for i, (ticker, ind) in enumerate(self.indicators.items(), 1):
            try:
                print(f"[{i:2d}/{len(self.indicators)}] {ticker:20s}", end=" ")

                close = ind["Close"]
                ma50 = ind["MA50"]
                rsi14 = ind["RSI14"]

                latest_close = close.iloc[-1].item()
                latest_ma50 = ma50.iloc[-1].item()
                latest_rsi14 = rsi14.iloc[-1].item()


                if pd.isna(latest_ma50) or pd.isna(latest_rsi14):
                    print("❌ NaN")
                    continue

                ma_diff_pct = (latest_close - latest_ma50) / latest_ma50 * 100

                if ma_diff_pct > 1:
                    ma_signal = 1
                elif ma_diff_pct < -1:
                    ma_signal = -1
                else:
                    ma_signal = 0

                if latest_rsi14 > 60:
                    rsi_signal = 1
                elif latest_rsi14 < 40:
                    rsi_signal = -0.5
                else:
                    rsi_signal = 0

                score = get_signal_score(ma_signal, rsi_signal)

                results.append(
                    {
                        "Ticker": ticker,
                        "Price": round(latest_close, 2),
                        "MA50": round(latest_ma50, 2),
                        "RSI14": round(latest_rsi14, 2),
                        "MA_Diff_%": round(ma_diff_pct, 2),
                        "MA_Signal": ma_signal,
                        "RSI_Signal": rsi_signal,
                        "Combined_Score": round(score, 2),
                    }
                )

                print(f"✓ Score {score:+.2f}")

            except Exception as e:
                print(f"❌ {str(e)[:30]}")

        self.results = pd.DataFrame(results)
        self.results = self.results.sort_values(
            "Combined_Score", ascending=False
        ).reset_index(drop=True)
        self.results["Rank"] = self.results.index + 1

        print(f"\n✓ Signals generated for {len(self.results)} stocks\n")

    def export_results(self):
        print("=" * 80)
        print("STEP 4: EXPORT RESULTS".center(80))
        print("=" * 80)

        if self.results is None or self.results.empty:
            print("❌ No results")
            return

        self.results.to_csv(self.output_file, index=False)
        print(f"✓ Exported to {self.output_file}\n")

        print("-" * 80)
        print("TOP 10 STOCKS".center(80))
        print("-" * 80)
        print(self.results[["Rank", "Ticker", "Price", "MA50", "RSI14", "Combined_Score"]]
            .head(10)
            .to_string(index=False)
        )
        print("-" * 80)

    def run(self):
        self.download_data()
        self.calculate_indicators()
        self.generate_signals()
        self.export_results()
        print("\n✓ SCREENER COMPLETE\n")


if __name__ == "__main__":
    screener = StockScreener(TEST_TICKERS, lookback_days=260)
    screener.run()

