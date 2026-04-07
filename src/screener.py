"""
    This pulls market data, computes indicators (MA50, MA200, RSI),
    and ranks stocks based on trend + momentum.

    What it does:
        downloads data of stocks(fast)
        calculates indicators
        scores each stock
        outputs ranked results + CSV
"""

import argparse
"""
    lets you run your script like - python -m src.screener --top 5 --date 2026-03-01
    user can control behavior from terminal
"""

"""
    Python shows warnings like - FutureWarning: something will break later
    hides warnings like - FutureWarning from pandas/yfinance
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
"""
    This is for parallel execution
    
    Problem without it -
        download 15 stocks one by one - slow (~30 sec)
    With this -
        download multiple stocks at same time - fast (~3–5 sec)
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import yfinance as yf

from src.indicators import calculate_moving_average, calculate_rsi
from src.scoring    import StockScorer
from src.stock_list import TEST_TICKERS, NIFTY_50_TICKERS

import logging # Better version of print()
import sys # Access to system-level stuff

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ||   %(levelname)s   ||    %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler("screener.log", mode="a"),]
)
logger = logging.getLogger(__name__)


class StockScreener:
    def __init__(self,tickers: list,lookback_days: int  = 400,screen_date: datetime = None):
        """
            tickers       - list of stocks
            lookback_days - number of past days to fetch data
            screen_date: date to run the screener (default: today)
        """
        self.tickers       = tickers
        self.lookback_days = lookback_days
        self.screen_date   = screen_date or datetime.now()
        self.data          = {}
        self.indicators    = {}
        self.results       = None
        self.output_file   = "screener_results.csv"
        self.scorer        = StockScorer(ma_weight=0.4, rsi_weight=0.6)


    def _download_one(self, ticker: str) -> tuple:
        """
            Download price data (OHLCV) for one stock

            Runs inside a thread - each ticker is taken in parallel
            Returns:
                (ticker, df) if success
                (ticker, None) if something fails

            Why using Ticker().history() instead of download():
                download() breaks when used with multiple threads
                All threads end up getting the same stock data

                history() is safer:
                each ticker gets its own object and request,
                so parallel calls don’t interfere with each other

            This keeps the data correct while still being fast
        """
        end_date   = self.screen_date
        start_date = end_date - timedelta(days=self.lookback_days)

        try:
            tk = yf.Ticker(ticker)
            df = tk.history(start=start_date.strftime("%Y-%m-%d"),end=end_date.strftime("%Y-%m-%d"))

            if df.empty:
                logger.warning("%-20s  NO DATA returned", ticker)
                return ticker, None

            logger.info("%-20s  ✓  %d rows downloaded", ticker, len(df))
            return ticker, df

        except Exception as exc:
            logger.error("%-20s  DOWNLOAD ERROR: %s", ticker, str(exc)[:60])
            return ticker, None

    def download_data(self, max_workers: int = 8) -> None:
        """
            Downloads historical stock data for all tickers using multiple threads
            Each ticker is fetched in parallel to reduce total download time
        """
        logger.info("—" * 60)
        logger.info("PARALLEL DOWNLOAD (workers=%d)".center(60), max_workers)
        logger.info("Screen date: %s".center(52), self.screen_date.strftime("%Y-%m-%d"))
        logger.info("—" * 60 + "\n")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            """
                have 8 workers ready to download stocks
                Submit all downloads at once
            """
            futures = {pool.submit(self._download_one, t): t for t in self.tickers}

            for future in as_completed(futures):
                ticker, df = future.result()
                if df is not None:
                    self.data[ticker] = df

        logger.info("Downloaded Stocks - %d / %d tickers\n".center(60),len(self.data), len(self.tickers))
        

    def calculate_indicators(self) -> None:
        """
        Calculate MA50, MA200, RSI14 for every downloaded ticker.

        Why MA200?
            - MA50 tells you short-term trend
            - MA200 tells you LONG-TERM trend
            - price > MA50 > MA200 = both timeframes aligned = highest conviction

        Note on data length:
            MA200 needs at least 200 rows (trading days) to compute.
            With lookback_days=400, we get ~280 trading days — enough.
            With lookback_days=260 (old default), we only got ~175 — too few!
        """
        logger.info("=" * 60)
        logger.info("STEP 2 — INDICATORS  (MA50, MA200, RSI14)")
        logger.info("=" * 60)

        ok = 0
        for ticker, df in self.data.items():
            try:
                # ✅ FIX: yf.Ticker().history() always returns a clean DataFrame
                # with flat column names: Open, High, Low, Close, Volume.
                # df["Close"] is always a proper pandas Series — no MultiIndex,
                # no ambiguity. The old yf.download() bug of all tickers returning
                # the same data is completely gone here.
                close = df["Close"]

                # Safety guard: in case some edge-case returns a DataFrame
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                # Ensure we have a clean 1D float Series for all indicator math
                close = close.squeeze().astype(float).dropna()
                if close.empty:
                    logger.warning("%-20s  no valid data after cleaning", ticker)
                    continue

                ma50  = calculate_moving_average(close, window=50)
                ma200 = calculate_moving_average(close, window=200)
                rsi14 = calculate_rsi(close, period=14)

                self.indicators[ticker] = {
                    "Close": close,
                    "MA50":  ma50,
                    "MA200": ma200,
                    "RSI14": rsi14,
                }
                ok += 1

            except Exception as exc:
                logger.error("%-20s  indicator error: %s", ticker, exc)

        logger.info("Indicators ready: %d / %d\n", ok, len(self.data))


    def generate_signals(self) -> None:
        """
        Score every stock and apply the P2 bullish filter.

        Bullish condition (from roadmap):
            IF price > SMA50 > SMA200 AND 40 < RSI < 70
            THEN SIGNAL = BULLISH

        Notes:
            - Stocks with insufficient data for MA200 get scored
              on MA50 + RSI only (graceful degradation)
            - Results are sorted by Combined_Score descending
        """
        logger.info("=" * 60)
        logger.info("STEP 3 — SIGNALS & SCORING")
        logger.info("=" * 60)

        rows = []

        for ticker, ind in self.indicators.items():
            try:
                close = ind["Close"]
                ma50  = ind["MA50"]
                ma200 = ind["MA200"]
                rsi14 = ind["RSI14"]

                price   = float(close.iloc[-1])
                v_ma50  = float(ma50.iloc[-1])
                v_rsi   = float(rsi14.iloc[-1])

                # MA200: only use if we have enough data (200+ rows)
                last_ma200 = ma200.iloc[-1]
                v_ma200 = float(last_ma200) if not pd.isna(last_ma200) else None

                # Skip if core indicators are not ready
                if pd.isna(v_ma50) or pd.isna(v_rsi):
                    logger.warning("%-20s  skipped — MA50 or RSI not ready", ticker)
                    continue

                # Score (with MA200 if available)
                score    = self.scorer.calculate_score(price, v_ma50, v_rsi, v_ma200)
                signal   = self.scorer.get_interpretation(score)
                ma_diff  = round((price - v_ma50) / v_ma50 * 100, 2)

                # P2 bullish hard filter
                if v_ma200 is not None:
                    bullish = StockScorer.is_bullish(price, v_ma50, v_ma200, v_rsi)
                else:
                    bullish = False  # can't confirm without MA200

                rows.append({
                    "Ticker":         ticker,
                    "Price":          round(price, 2),
                    "MA50":           round(v_ma50, 2),
                    "MA200":          round(v_ma200, 2) if v_ma200 else None,
                    "MA_Diff_%":      ma_diff,
                    "RSI14":          round(v_rsi, 2),
                    "Bullish":        bullish,
                    "Combined_Score": score,
                    "Signal":         signal,
                })

                logger.info(
                    "%-20s  score=%+.2f  %-16s  bullish=%s",
                    ticker, score, signal, bullish
                )

            except Exception as exc:
                logger.error("%-20s  signal error: %s", ticker, exc)

        if not rows:
            logger.error("No stocks passed screening — check data and indicators")
            self.results = pd.DataFrame()
            return

        self.results = (
            pd.DataFrame(rows)
            .sort_values("Combined_Score", ascending=False)
            .reset_index(drop=True)
        )
        self.results["Rank"] = self.results.index + 1
        logger.info("Scored %d stocks\n", len(self.results))


    def export_results(self, top_n: int = 5) -> None:
        """
        Save results to CSV and print top N stocks to terminal.
        """
        logger.info("=" * 60)
        logger.info("STEP 4 — EXPORT")
        logger.info("=" * 60)

        if self.results is None or self.results.empty:
            logger.error("No results to export")
            return

        self.results.to_csv(self.output_file, index=False)
        logger.info("Saved → %s", self.output_file)

        # Print top N
        display_cols = [
            "Rank", "Ticker", "Price", "MA50", "MA200",
            "RSI14", "Combined_Score", "Signal", "Bullish"
        ]
        print("\n" + "=" * 85)
        print(
            f"  TOP {top_n} STOCKS  "
            f"[screened on {self.screen_date.strftime('%Y-%m-%d')}]"
            .center(85, "=")
        )
        print("=" * 85)
        print(self.results[display_cols].head(top_n).to_string(index=False))
        print("=" * 85 + "\n")


    def run(self, top_n: int = 5) -> None:
        logger.info("STOCK SCREENER — %s".center(50) + "\n", self.screen_date.strftime("%Y-%m-%d"))
        self.download_data()
        self.calculate_indicators()
        self.generate_signals()
        self.export_results(top_n=top_n)
        logger.info("SCREENER COMPLETE\n".center(60))


def parse_args() -> argparse.Namespace: # just an object holding values
    parser = argparse.ArgumentParser(
        prog="screener",
        description="Quantitative Stock Screener — NIFTY50",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                Run on today's data (default TEST universe, top 5):
                python -m src.screener

                Run on full NIFTY50, show top 3:
                python -m src.screener --universe NIFTY50 --top 3

                Screen on a specific past date (useful for testing):
                python -m src.screener --date 2026-03-01 --top 5

                Full example:
                python -m src.screener --universe NIFTY50 --top 5 --date 2026-02-24 --lookback 400
            """
    )
    parser.add_argument(
        "--universe",
        choices=["NIFTY50", "TEST"],
        default="TEST",
        help="Stock universe - NIFTY50=15 stocks, TEST=15 stocks (Default: TEST)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="N",
        help="Number of top stocks to display (Default: 5)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Screen date (Default: today). Past dates screen on historical data.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=400,
        metavar="DAYS",
        help="Calendar days of historical data for indicators. Default: 400 (~280 trading days)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tickers = NIFTY_50_TICKERS if args.universe == "NIFTY50" else TEST_TICKERS

    screen_date = datetime.now()
    if args.date:
        try:
            screen_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid date format '%s'. Use YYYY-MM-DD.", args.date)
            sys.exit(1)

    screener = StockScreener(tickers=tickers,lookback_days=args.lookback,
        screen_date=screen_date)
    
    screener.run(top_n=args.top)
