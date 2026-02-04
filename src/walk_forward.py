import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import yfinance as yf

from src.screener import StockScreener
from src.stock_list import get_stock_list


class WalkForwardValidator:
    def __init__(self, train_months=6, test_months=1, top_n=3):
        self.train_months = train_months
        self.test_months = test_months
        self.top_n = top_n
        self.stock_list = get_stock_list()
        self.results = []

        print(
            f"🚀 WalkForwardValidator | "
            f"train={train_months} test={test_months} top={top_n}"
        )

    def _screen_blind(self, screen_date):
        screener = StockScreener(self.stock_list, lookback_days=260)
        screener.data = {}

        for ticker in self.stock_list:
            try:
                df = yf.download(
                    ticker,
                    start=(screen_date - timedelta(days=260)).strftime("%Y-%m-%d"),
                    end=screen_date.strftime("%Y-%m-%d"),
                    progress=False
                )
                if not df.empty:
                    screener.data[ticker] = df
            except Exception:
                continue

        if not screener.data:
            return []

        screener.calculate_indicators()
        scores = []

        for ticker, ind in screener.indicators.items():
            try:
                close = ind["Close"].iloc[-1]
                ma50 = ind["MA50"].iloc[-1]
                rsi = ind["RSI14"].iloc[-1]

                if np.isnan(ma50) or np.isnan(rsi):
                    continue

                score = screener.scorer.calculate_score(close, ma50, rsi)
                scores.append((ticker, score, close))
            except Exception:
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:self.top_n]

    def _measure_returns(self, picks, screen_date):
        returns = []

        for ticker, score, entry in picks:
            try:
                df = yf.download(
                    ticker,
                    start=screen_date.strftime("%Y-%m-%d"),
                    end=(screen_date + timedelta(days=30)).strftime("%Y-%m-%d"),
                    progress=False
                )

                if len(df) < 2:
                    continue

                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                exit_price = close.iloc[-1]
                ret = (exit_price - entry) / entry * 100
                returns.append(ret)

            except Exception:
                continue

        portfolio_ret = float(np.mean(returns)) if returns else 0.0

        nifty_df = yf.download(
            "^NSEI",
            start=screen_date.strftime("%Y-%m-%d"),
            end=(screen_date + timedelta(days=30)).strftime("%Y-%m-%d"),
            progress=False
        )

        if len(nifty_df) >= 2:
            nifty_close = nifty_df["Close"]
            if isinstance(nifty_close, pd.DataFrame):
                nifty_close = nifty_close.iloc[:, 0]

            nifty_ret = float(
                (nifty_close.iloc[-1] - nifty_close.iloc[0])
                / nifty_close.iloc[0]
                * 100
            )
        else:
            nifty_ret = 0.0

        return portfolio_ret, nifty_ret

    def run(self, start_year=2024, months=12):
        date = datetime(start_year, 1, 15)

        for i in range(self.train_months, months):
            screen_date = date + relativedelta(months=i)
            print(f"\n📅 Testing: {screen_date.strftime('%b %Y')}")

            picks = self._screen_blind(screen_date)
            if not picks:
                print("No picks")
                continue

            port_ret, nifty_ret = self._measure_returns(picks, screen_date)

            self.results.append({
                "month": screen_date.strftime("%b %Y"),
                "portfolio": port_ret,
                "nifty": nifty_ret,
                "edge": port_ret - nifty_ret
            })

            print(
                f"Portfolio: {port_ret:+.2f}% | "
                f"Nifty: {nifty_ret:+.2f}% | "
                f"Edge: {port_ret - nifty_ret:+.2f}%"
            )

        df = pd.DataFrame(self.results)
        df.to_csv("walkforward_results.csv", index=False)
        print("\n💾 Saved: walkforward_results.csv")
        return df


if __name__ == "__main__":
    wf = WalkForwardValidator()
    wf.run()

