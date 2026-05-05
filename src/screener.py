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
    format="%(asctime)s ││   %(levelname)s   ││    %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler("logs/screener.log", mode="a")]
)
logger = logging.getLogger(__name__)


class StockScreener:
    def __init__(self,tickers: list,lookback_days: int  = 400,screen_date: datetime = None):
        """
            tickers       - list of stocks
            lookback_days - number of past days to fetch data
            screen_date: date to run the screener (default: today)
        """
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.screen_date = screen_date or datetime.now()
        self.data = {}
        self.indicators = {}
        self.results = None
        self.output_file = "outputs/screener_results.csv"
        self.scorer = StockScorer(ma_weight=0.4, rsi_weight=0.6)


    def download_one(self, ticker: str) -> tuple:
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
        start_date = self.screen_date - timedelta(days=self.lookback_days)

        try:
            tk = yf.Ticker(ticker) # yf.download() is NOT thread-safe so we used this
            df = tk.history(start=start_date.strftime("%Y-%m-%d"),end=self.screen_date.strftime("%Y-%m-%d"))

            if df.empty:
                logger.warning("%-20s 🔸  NO DATA returned".center(71), ticker)
                return ticker, None

            logger.info("%-20s 🔹  %d rows downloaded".center(53), ticker, len(df))
            return ticker, df

        except Exception as exc:
            logger.error("%-20s  DOWNLOAD ERROR: %s".center(70), ticker, str(exc)[:60])
            return ticker, None

    def download_data(self, max_workers: int = 8) -> None:
        """
            Downloads historical stock data for all tickers using multiple threads
            Each ticker is fetched in parallel to reduce total download time
        """
        logger.info("─"*71)
        logger.info("PARALLEL DOWNLOAD (workers=%d)".center(69), max_workers)
        logger.info("Screen date: %s".center(61), self.screen_date.strftime("%Y-%m-%d"))
        logger.info("─"*71 + "\n")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            """
                have 8 workers ready to download stocks
                Submit all downloads at once
            """
            task = {pool.submit(self.download_one, t): t for t in self.tickers}
            """
                start downloading this stock
                What pool.submit does:
                    runs download_one(ticker) in another thread
            """

            for task in as_completed(task): # give me results as soon as each finishes - order = RANDOM (based on which finishes first)
                ticker, df = task.result() # ("RELIANCE.NS", dataframe)
                if df is not None:
                    self.data[ticker] = df
        
        logger.info("\n")
        logger.info("Downloaded %d / %d Stocks\n".center(70),len(self.data), len(self.tickers))
        

    def calculate_indicators(self) -> None:
        """
            Compute indicators for each stock using closing prices.

            We calculate:
                MA50  → short-term trend
                MA200 → long-term trend
                RSI14 → momentum

            MA50 reacts faster, MA200 is slower
            If price > MA50 > MA200, both short and long trends are aligned
            which usually means a strong and stable uptrend

            MA200 needs enough data (200+ trading days).
            That’s why we use ~400 calendar days -- gives enough history.
        """
        logger.info("─"*71)
        logger.info("INDICATORS  (MA50, MA200, RSI14)".center(69))
        logger.info("─"*71 + "\n")

        ok = 0
        for ticker, df in self.data.items():
            try:
                """
                    yf.Ticker().history() always returns a clean DataFrame
                    with flat column names : Open, High, Low, Close, Volume.
                    df["Close"] is always a proper pandas Series — no MultiIndex
                """
                close = df["Close"]

                # in case some edge-case return a DataFrame
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0] # take all rows first column(0)

                # Ensure we have a clean 1D float Series for all indicators math
                close = close.squeeze().astype(float).dropna()
                """
                    squeeze():
                        removes extra dimensions
                    ex:
                        [[100], [102]] -> [100, 102]
                        
                    astype(float):
                    ensures all values are numbers
                    ex:
                        "100" -> 100.0
                        
                    dropna():
                        removes missing values:
                        [100, NaN, 102] -> [100, 102]
                """
                if close.empty:
                    logger.warning("%-20s  no valid data after cleaning".center(50), ticker)
                    continue
                if len(close) < 200:
                    logger.warning("%-20s  not enough data (<200 rows)".center(40), ticker)
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
                logger.error("%-20s  indicator error: %s".center(71), ticker, exc)

        logger.info("Indicators ready: %d / %d\n".center(71), ok, len(self.data))


    def generate_signals(self) -> None:
        """
            Score each stock using trend + momentum, then check if its actually worth considering

                Take latest price, MA50, MA200, RSI
                Convert them into a score (how strong the setup is)
                Also apply a strict “bullish” filter

            Bullish rule:
                price > MA50 > MA200  -> trend is clearly up (short + long aligned)
                40 < RSI < 70         -> momentum is healthy but not overheated

            Each stock gets:
                score   → strength of setup
                signal  → BUY / SELL etc.
                bullish → True/False 

            All stocks are sorted by score (best → worst)
        """
        logger.info("─"*71)
        logger.info("│" + "SIGNALS & SCORING".center(69) + "│")
        logger.info("─"*71)

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
                last_ma200 = ma200.iloc[-1]
                v_ma200 = float(last_ma200) if not pd.isna(last_ma200) else None

                if pd.isna(v_ma50) or pd.isna(v_rsi):
                    logger.warning("%-20s  skipped — MA50 or RSI not ready".center(50), ticker)
                    continue

                score    = self.scorer.calculate_score(price, v_ma50, v_rsi, v_ma200)
                signal   = self.scorer.get_interpretation(score)
                ma_diff  = round((price - v_ma50) / v_ma50 * 100, 2)

                if v_ma200 is not None:
                    bullish = StockScorer.is_bullish(price, v_ma50, v_ma200, v_rsi)
                else:
                    bullish = False # cant confirm without MA200

                rows.append({
                    "Ticker":         ticker,
                    "Price":          round(price, 2),
                    "MA50":           round(v_ma50, 2),
                    "MA200":          round(v_ma200, 2) if v_ma200 else None,
                    "MA_Diff_%":      ma_diff,
                    "RSI14":          round(v_rsi, 2),
                    "Bullish":        bullish,
                    "Combined_Score": score,
                    "Signal":         signal
                })

                logger.info("│" + "%-16s  Score = %+.2f    %-12s  Bullish = %-6s".center(49) + "│",ticker, score, signal, bullish)

            except Exception as exc:
                logger.error("%-20s  signal error: %s".center(71), ticker, exc)

        if not rows:
            logger.error("No stocks passed screening — check data and indicators".center(60))
            self.results = pd.DataFrame()
            return
            
        self.results = pd.DataFrame(rows).sort_values("Combined_Score", ascending=False).reset_index(drop=True)
        self.results["Rank"] = self.results.index + 1
        
        logger.info("─"*71 + '\n')
        logger.info("Scored %d stocks\n".center(71), len(self.results))


    def export_results(self, top_n: int = 5) -> None:
        """
            Save results to CSV and print top N stocks to terminal.
        """
        logger.info("─"*71)
        logger.info("EXPORT".center(69))
        logger.info("─"*71)

        if self.results is None or self.results.empty:
            logger.error("No results to export")
            return

        self.results.to_csv(self.output_file, index=False)
        logger.info("Saved -> %s".center(42), self.output_file)

        display_cols = [
            "Rank", "Ticker", "Price", "MA50", "MA200",
            "RSI14", "Combined_Score", "Signal", "Bullish"
        ]
        print("\n" + "─"*98)
        print(f"  TOP {top_n} STOCKS  "f"[screened on {self.screen_date.strftime('%Y-%m-%d')}]".center(98, "─"))
        print("─"*98)
        print("")
        print(self.results[display_cols].head(top_n).to_string(index=False))
        print("")
        print("─"*98 + "\n")


    def run(self, top_n: int = 5) -> None:
        logger.info("STOCK SCREENER - %s".center(50) + "\n", self.screen_date.strftime("%Y-%m-%d"))
        self.download_data()
        self.calculate_indicators()
        self.generate_signals()
        self.export_results(top_n=top_n)
        logger.info("SCREENER COMPLETE\n".center(71))


def parse_args() -> argparse.Namespace: # just an object holding values
    parser = argparse.ArgumentParser(
        prog="screener",
        description="Quantitative Stock Screener - NIFTY50",
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
    parser.add_argument("--universe",
        choices=["NIFTY50", "TEST"],
        default="TEST",
        help="Stock universe - NIFTY50=15 stocks, TEST=15 stocks (Default: TEST)"
    )
    parser.add_argument("--top",
        type=int,
        default=5,
        metavar="N",
        help="Number of top stocks to display (Default: 5)"
    )
    parser.add_argument("--date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Screen date (Default: today). Past dates screen on historical data."
    )
    parser.add_argument("--lookback",
        type=int,
        default=400,
        metavar="DAYS",
        help="Calendar days of historical data for indicators. Default: 400 (~280 trading days)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tickers = NIFTY_50_TICKERS if args.universe == "NIFTY50" else TEST_TICKERS

    screen_date = datetime.now() # Y-M-D 00:52:31.123456
    if args.date:
        try:
            screen_date = datetime.strptime(args.date, "%Y-%m-%d")
            
        except ValueError:
            logger.error("Invalid date format '%s'. Use YYYY-MM-DD.", args.date)
            sys.exit(1)

    screener = StockScreener(tickers=tickers,lookback_days=args.lookback,screen_date=screen_date)
    
    screener.run(top_n=args.top)
