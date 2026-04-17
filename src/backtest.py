"""
backtest.py — Walk-Forward Backtester (P2 Complete)

What this does:
    Runs a month-by-month historical backtest of the screener strategy.
    Each month is independent — no look-ahead bias.

Walk-forward logic:
    For each month i:
        1. Screen stocks using ONLY data up to screen_date_i
        2. Select top N stocks
        3. Hold for holding_days days
        4. Measure actual returns
        5. Compare against Nifty 50 benchmark

No look-ahead bias:
    Step 1 downloads data with end=screen_date.
    Step 3 downloads data AFTER screen_date.
    These two downloads never overlap.

Usage:
    python -m src.backtest
"""

import logging
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import yfinance as yf

from src.screener  import StockScreener
from src.stock_list import get_stock_list

# ------------------------------------------------------------------ #
#  Logging                                                             #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  CorrectBacktest                                                     #
# ------------------------------------------------------------------ #

class CorrectBacktest:
    """
    Walk-forward backtester.

    'Correct' because it has zero look-ahead bias:
        - Screening uses only data available ON the screen date.
        - Returns are measured using only data AFTER the screen date.

    Output files:
        backtest_results.csv  — monthly portfolio vs Nifty
        backtest_picks.csv    — individual stock picks per month
    """

    def __init__(
        self,
        backtest_months: int = 12,
        lookback_days:   int = 260,
        top_n:           int = 3,
        holding_days:    int = 30,
        start_year:      int = 2024,
        start_month:     int = 1,
    ):
        """
        Args:
            backtest_months : How many months to test. Default 12.
            lookback_days   : Historical data window for indicators. Default 260.
            top_n           : Number of stocks to hold per month. Default 3.
            holding_days    : Days to hold before exiting. Default 30.
            start_year      : Year to start backtest from. Default 2024.
            start_month     : Month to start from. Default 1 (January).
        """
        self.backtest_months = backtest_months
        self.lookback_days   = lookback_days
        self.top_n           = top_n
        self.holding_days    = holding_days
        self.start_year      = start_year
        self.start_month     = start_month
        self.stock_list      = get_stock_list()
        self.monthly_results = []   # one dict per month
        self.all_picks       = []   # one dict per stock per month

        logger.info(
            "CorrectBacktest init | months=%d  top_n=%d  hold=%dd  start=%d-%02d",
            backtest_months, top_n, holding_days, start_year, start_month,
        )

    def _screen_on_date(self, screen_date: datetime) -> list:
        """
        Run the screener on a specific historical date.

        Data downloaded: (screen_date - lookback_days) → screen_date
        Future data is NOT available to the screener.

        Returns:
            list of (ticker, score, entry_price) — top N picks
        """
        logger.info("  Screening on %s...", screen_date.strftime("%Y-%m-%d"))

        screener = StockScreener(
            tickers=self.stock_list,
            lookback_days=self.lookback_days,
            screen_date=screen_date,
        )
        screener.download_data(max_workers=6)
        screener.calculate_indicators()
        screener.generate_signals()

        if screener.results is None or screener.results.empty:
            logger.warning("  No picks generated for %s",
                           screen_date.strftime("%b %Y"))
            return []

        top = screener.results.head(self.top_n)
        picks = [
            (row["Ticker"], row["Combined_Score"], row["Price"])
            for _, row in top.iterrows()
        ]

        logger.info(
            "  Top %d picks: %s",
            len(picks),
            [p[0] for p in picks]
        )
        return picks

    # ---------------------------------------------------------------- #
    #  Core: Measure returns after screen_date                         #
    # ---------------------------------------------------------------- #

    def _measure_returns(
        self,
        picks:       list,
        screen_date: datetime,
    ) -> tuple:
        """
        Download price data AFTER screen_date and calculate returns.

        This is the ONLY place we use future data, and it is ONLY
        used for measuring returns — never for signal generation.

        Returns:
            (portfolio_return_pct, nifty_return_pct, list_of_trade_dicts)
        """
        exit_date = screen_date + timedelta(days=self.holding_days)
        trades    = []

        for ticker, score, entry_price in picks:
            try:
                df = yf.download(
                    ticker,
                    start=screen_date.strftime("%Y-%m-%d"),
                    end=exit_date.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                )
                if len(df) < 2:
                    logger.warning("  %s: not enough forward data", ticker)
                    continue

                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                actual_entry = float(close.iloc[0])
                actual_exit  = float(close.iloc[-1])

                if actual_entry == 0:
                    continue

                ret_pct = (actual_exit - actual_entry) / actual_entry * 100

                trades.append({
                    "Ticker":      ticker,
                    "Score":       score,
                    "Entry_Price": round(actual_entry, 2),
                    "Exit_Price":  round(actual_exit,  2),
                    "Return_%":    round(ret_pct, 2),
                })

                logger.info(
                    "  %s  entry=%.2f  exit=%.2f  return=%+.2f%%",
                    ticker, actual_entry, actual_exit, ret_pct
                )

            except Exception as exc:
                logger.warning("  Return calc failed for %s: %s", ticker, exc)

        # Portfolio return = equal-weight average of all picks
        portfolio_ret = (
            float(np.mean([t["Return_%"] for t in trades]))
            if trades else 0.0
        )

        # Nifty 50 benchmark return for the same period
        nifty_ret = self._get_nifty_return(screen_date, exit_date)

        return portfolio_ret, nifty_ret, trades

    def _get_nifty_return(
        self,
        start: datetime,
        end:   datetime,
    ) -> float:
        """Download Nifty 50 return for a date range."""
        try:
            df = yf.download(
                "^NSEI",
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
            if len(df) < 2:
                return 0.0
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return float(
                (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
            )
        except Exception:
            return 0.0

    # ---------------------------------------------------------------- #
    #  Run: main loop                                                   #
    # ---------------------------------------------------------------- #

    def run(self) -> pd.DataFrame:
        """
        Run the full walk-forward backtest.

        Returns:
            pd.DataFrame of monthly results.
            Also saves backtest_results.csv and backtest_picks.csv.
        """
        start = datetime(self.start_year, self.start_month, 15)

        logger.info("=" * 60)
        logger.info("BACKTEST START — %d months from %s",
                    self.backtest_months,
                    start.strftime("%b %Y"))
        logger.info("=" * 60)

        for i in range(self.backtest_months):
            screen_date = start + relativedelta(months=i)
            month_str   = screen_date.strftime("%b %Y")

            logger.info("\n── Month %2d / %d — %s ──",
                        i + 1, self.backtest_months, month_str)

            picks = self._screen_on_date(screen_date)
            if not picks:
                logger.warning("Skipping %s — no picks", month_str)
                continue

            port_ret, nifty_ret, trades = self._measure_returns(
                picks, screen_date
            )

            self.monthly_results.append({
                "Month":               month_str,
                "Portfolio_Return_%":  round(port_ret,              2),
                "Nifty_Return_%":      round(nifty_ret,             2),
                "Outperformance_%":    round(port_ret - nifty_ret,  2),
                "Num_Stocks":          len(trades),
            })

            for t in trades:
                t["Month"] = month_str
                self.all_picks.append(t)

            logger.info(
                "  ► Portfolio: %+.2f%%  Nifty: %+.2f%%  Alpha: %+.2f%%",
                port_ret, nifty_ret, port_ret - nifty_ret,
            )

        return self._save_and_print()

    # ---------------------------------------------------------------- #
    #  Save results and print summary                                   #
    # ---------------------------------------------------------------- #

    def _save_and_print(self) -> pd.DataFrame:
        """Save CSVs and print performance summary."""
        results_df = pd.DataFrame(self.monthly_results)
        picks_df   = pd.DataFrame(self.all_picks)

        results_df.to_csv("backtest_results.csv", index=False)
        picks_df.to_csv("backtest_picks.csv",     index=False)

        logger.info("\nSaved → backtest_results.csv")
        logger.info("Saved → backtest_picks.csv")

        if results_df.empty:
            logger.warning("No results generated — check date range and data")
            return results_df

        # ---- Calculate summary metrics ----
        port  = results_df["Portfolio_Return_%"]
        nifty = results_df["Nifty_Return_%"]
        alpha = results_df["Outperformance_%"]

        cum_port  = float(((1 + port  / 100).prod() - 1) * 100)
        cum_nifty = float(((1 + nifty / 100).prod() - 1) * 100)
        win_rate  = float((port > 0).mean()   * 100)
        beat_rate = float((port > nifty).mean() * 100)
        sharpe    = float(port.mean() / port.std()) if port.std() > 0 else 0.0

        # Max drawdown from equity curve
        equity      = (1 + port / 100).cumprod()
        peak        = equity.cummax()
        dd          = (equity - peak) / peak * 100
        max_dd      = float(dd.min())

        # Sortino (penalise downside only)
        downside    = port[port < 0]
        sortino     = float(port.mean() / downside.std()) if len(downside) > 1 else 0.0

        print("\n" + "═" * 60)
        print(" BACKTEST SUMMARY ".center(60, "═"))
        print("═" * 60)
        print(f"  Months tested       : {len(results_df)}")
        print(f"  Portfolio return    : {cum_port:+.2f}%")
        print(f"  Nifty return        : {cum_nifty:+.2f}%")
        print(f"  Total alpha         : {cum_port - cum_nifty:+.2f}%")
        print(f"  Win rate            : {win_rate:.1f}%  (profitable months)")
        print(f"  Beat benchmark      : {beat_rate:.1f}%  (months > Nifty)")
        print(f"  Sharpe ratio        : {sharpe:.3f}")
        print(f"  Sortino ratio       : {sortino:.3f}")
        print(f"  Max drawdown        : {max_dd:.2f}%")
        print(f"  Avg monthly alpha   : {alpha.mean():+.2f}%")
        print("═" * 60 + "\n")

        return results_df


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    bt = CorrectBacktest(
        backtest_months=12,
        lookback_days=400,
        top_n=3,
        holding_days=30,
        start_year=2024,
        start_month=1,
    )
    bt.run()
