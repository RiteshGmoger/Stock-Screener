import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from src.indicators import (
    calculate_moving_average,
    calculate_rsi,
    get_ma_signal,
    get_rsi_signal,
    combine_signals,
)
from src.stock_list import TEST_TICKERS


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
        print("STEP 1: DOWNLOAD DATA")
        print("=" * 80)

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
                )

                if df.empty:
                    print("❌ NO DATA")
                    continue

                self.data[ticker] = df
                print(f"✓ {len(df)} rows")

            except Exception as e:
                print(f"❌ {str(e)[:30]}")

        print(f"\n✓ Downloaded {len(self.data)}/{len(self.tickers)} stocks\n")

    def calculate_indicators(self):
        print("=" * 80)
        print("STEP 2: CALCULATE INDICATORS")
        print("=" * 80)

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
                print(f"❌ {str(e)[:30]}")

        print(f"\n✓ Indicators calculated for {len(self.indicators)} stocks\n")

    def generate_signals(self):
        print("=" * 80)
        print("STEP 3: GENERATE SIGNALS")
        print("=" * 80)

        results = []

        for i, (ticker, ind) in enumerate(self.indicators.items(), 1):
            try:
                print(f"[{i:2d}/{len(self.indicators)}] {ticker:20s}", end=" ")

                close = ind["Close"]
                ma50 = ind["MA50"]
                rsi14 = ind["RSI14"]

                price = float(close.iloc[-1])
                ma = float(ma50.iloc[-1])
                rsi = float(rsi14.iloc[-1])

                if pd.isna(ma) or pd.isna(rsi):
                    print("❌ NaN")
                    continue

                ma_sig = get_ma_signal(price, ma)
                rsi_sig = get_rsi_signal(rsi)
                score = combine_signals(ma_sig, rsi_sig)

                results.append(
                    {
                        "Ticker": ticker,
                        "Price": round(price, 2),
                        "MA50": round(ma, 2),
                        "RSI14": round(rsi, 2),
                        "Combined_Score": round(score, 2),
                    }
                )

                if score >= 0.7:
                    tag = "🔥 STRONG BUY"
                elif score >= 0.3:
                    tag = "👍 BUY"
                elif score <= -0.3:
                    tag = "⛔ SELL"
                else:
                    tag = "➖ HOLD"

                print(f"{tag} ({score:+.2f})")

            except Exception as e:
                print(f"❌ {str(e)[:30]}")

        self.results = pd.DataFrame(results).sort_values(
            "Combined_Score", ascending=False
        )
        self.results["Rank"] = range(1, len(self.results) + 1)

        print(f"\n✓ Signals generated for {len(self.results)} stocks\n")

    def export_results(self):
        print("=" * 80)
        print("STEP 4: EXPORT RESULTS")
        print("=" * 80)

        if self.results is None or self.results.empty:
            print("❌ No results")
            return

        self.results.to_csv(self.output_file, index=False)
        print(f"✓ Exported to {self.output_file}\n")

        print(self.results.head(10).to_string(index=False))

    def run(self):
        self.download_data()
        self.calculate_indicators()
        self.generate_signals()
        self.export_results()
        print("\n✓ SCREENER COMPLETE\n")


if __name__ == "__main__":
    screener = StockScreener(TEST_TICKERS, lookback_days=260)
    screener.run()

