"""
    Walk-Forward Validation for the stock screener strategy

    Tests the screener on a rolling window:
        For each test month, screen using only past data (no future leak)
        Pick top N stocks
        Measure actual returns over the next test_months * 30 days
        Deduct slippage to simulate real trading costs
        Compare against Nifty 50

    Why stricter than backtest:
        screener only sees data available at screen_date
        forward returns are measured after the fact
        no future data ever enters the screening decision

    Difference from backtest:
        backtest.py uses fixed holding_days
        walk_forward uses test_months -> holding period = exactly test_months calendar months
        slippage is modeled here, not in backtest
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import yfinance as yf

from src.screener   import StockScreener
from src.stock_list import get_stock_list

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ││   %(levelname)s   ││    %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler("logs/walk_forward.log", mode="a")]
)
logger = logging.getLogger(__name__)

# India 10Y govt bond yield ~ 7% annualized -> monthly risk-free rate
# used in Sharpe/Sortino to measure excess return above risk-free
INDIA_RF_ANNUAL = 0.07
RF_MONTHLY      = INDIA_RF_ANNUAL / 12   # ~0.00583


class WalkForwardValidator:
    def __init__(self,warmup_months: int = 6,holding_period: int = 1,top_n: int = 3,slippage_pct: float = 0.2):
        """
            warmup_months   : Months to skip at start so MA200 has enough history
                            NOT a training window -- no parameters are optimized
                            Default 6
            holding_period  : Holding period per test window = test_months * 30 days
                            Default 1 -> 30 day hold
            top_n           : Number of stocks to hold each period. Default 3
            slippage_pct    : Round-trip cost % deducted from each trade
                            Covers brokerage + bid-ask spread
                            Nifty50 large caps: 0.2% is conservative and realistic
                            Default 0.2
                            0.1% → aggressive (optimistic)
                            0.2% → safe / realistic 
                            0.3%+ → conservative
        """
        self.warmup_months = warmup_months
        self.holding_period = holding_period
        self.top_n = top_n
        self.slippage_pct = slippage_pct
        self.holding_days = holding_period * 30
        self.stock_list = get_stock_list()
        self.results = []
        self.all_picks = []

        logger.info("─"*71)
        logger.info("│" + "WALK-FORWARD VALIDATOR".center(69) + "│")
        logger.info("─"*71)
        logger.info("│" + "Warmup months".center(34) + ":" + str(warmup_months).center(34) + "│")
        logger.info("│" + "Test months".center(34) + ":" + f"{holding_period} month (calendar)".center(34) + "│")
        logger.info("│" + "Top N".center(34) + ":" + str(top_n).center(34) + "│")
        logger.info("│" + "Slippage".center(34) + ":" + f"{slippage_pct:.2f}% (round-trip)".center(34) + "│")
        logger.info("│" + "Risk-free rate".center(34) + ":" + f"{RF_MONTHLY*100:.2f}% / month  (India 10Y)".center(34) + "│")
        logger.info("─"*71 + "\n")
        

    def screen_blind(self, screen_date: datetime) -> list:
        """
            Run the screener on screen_date using only data available up to that date

            We pass screen_date directly into StockScreener so it fetches
            exactly 400 calendar days of history ending at screen_date
            No future data ever enters the decision.

            Why lookback_days = 400?
                MA200 needs 200+ trading days (~280)
                400 calendar days gives enough buffer for weekends + holidays

            Returns:
                list of (ticker, score, price) for top N picks
        """
        text = f"Screening on {screen_date.strftime('%Y-%m-%d')}"
        logger.info("│" + text.center(69) + "│")
        logger.info("─"*71 + "\n")

        screener = StockScreener(tickers = self.stock_list,lookback_days = 400,screen_date = screen_date)
        screener.download_data(max_workers=6)
        screener.calculate_indicators()
        screener.generate_signals()

        if screener.results is None or screener.results.empty:
            logger.warning("No picks generated for %s".center(69), screen_date.strftime("%b %Y"))
            return []

        top = screener.results.head(self.top_n)
        picks = [(row["Ticker"], row["Combined_Score"], row["Price"])for _, row in top.iterrows()]

        logger.info("─"*71)
        logger.info("│" + "TOP PICKS".center(69) + "│")
        logger.info("─"*71)

        for _, row in top.iterrows():
            line = f"{row['Ticker']:<20} - {row['Signal']:>12}"
            logger.info("│" + line.center(69) + "│")

        logger.info("─"*71)
        return picks


    def _measure_returns(self, picks: list, screen_date: datetime) -> tuple:
        """
            Measure actual returns for each pick over self.holding_days.

            Slippage:
                Each trade has a round-trip cost = entry slippage + exit slippage
                We deduct self.slippage_pct from every trade's gross return
                This simulates brokerage + bid-ask spread

                Example:
                    gross return = +5.0%
                    slippage     = 0.2%
                    net return   = 4.8%  (what you actually make)

            Equal weighting:
                np.mean(returns) = same capital allocated to each stock
                This is intentional and standard for systematic strategies

            Returns:
                (portfolio_return, nifty_return, trades)
        """
        # relativedelta(months=N) gives exact calendar months: Jan 15 -> Feb 15, not Feb 14 or Feb 17
        # timedelta(days=30) is wrong for Feb (28/29 days) and months with 31 days
        exit_date = screen_date + relativedelta(months=self.holding_period)
        trades    = []

        logger.info("│" + "TRADES".center(69) + "│")
        logger.info("─"*71)

        for ticker, score, entry_price in picks:
            try:
                tk = yf.Ticker(ticker)
                df = tk.history(
                    start = screen_date.strftime("%Y-%m-%d"),
                    end   = exit_date.strftime("%Y-%m-%d"),
                )

                if df.empty or len(df) < 2:
                    logger.warning("%-20s  not enough forward data".center(50), ticker)
                    continue

                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna()

                if len(close) < 2:
                    continue

                entry = float(close.iloc[0])
                exit_ = float(close.iloc[-1])

                if entry == 0:
                    continue

                gross_ret = (exit_ - entry) / entry * 100
                net_ret   = gross_ret - self.slippage_pct  # FIX: deduct round-trip slippage

                trades.append({
                    "Ticker":       ticker,
                    "Score":        score,
                    "Entry_Price":  round(entry, 2),
                    "Exit_Price":   round(exit_, 2),
                    "Gross_Return": round(gross_ret, 2),
                    "Return":       round(net_ret, 2),  # net of slippage
                })

                line = (
                    f"{ticker:<12} │ entry={entry:>8.2f} │ exit={exit_:>8.2f}"
                    f" │ gross={gross_ret:+6.2f}% │ net={net_ret:+6.2f}%"
                )
                logger.info("│ " + line.ljust(67) + " │")

            except Exception as exc:
                logger.warning("%-20s  return calc failed: %s".center(60), ticker, exc)

        logger.info("─"*71)

        # equal weighting: each stock gets the same capital allocation
        portfolio_return = float(np.mean([t["Return"] for t in trades])) if trades else 0.0
        nifty_return     = self._get_nifty_return(screen_date, exit_date)

        return portfolio_return, nifty_return, trades


    def _get_nifty_return(self, start: datetime, end: datetime) -> float:
        """
            Download Nifty 50 return for the holding period.
            Used to measure alpha -- did we beat the market?

            Note: Nifty return is NOT slippage-adjusted here
            because buying Nifty via index fund has near-zero cost (~0.01% for Niftybees)
            This makes the comparison fair and slightly conservative for our strategy.

            Example:
                Strategy +5%  vs  Nifty +3%  -> good (beat market)
                Strategy +5%  vs  Nifty +8%  -> bad  (market won)
                Strategy -2%  vs  Nifty -5%  -> good (lost less)
        """
        try:
            tk = yf.Ticker("^NSEI")
            df = tk.history(
                start = start.strftime("%Y-%m-%d"),
                end   = end.strftime("%Y-%m-%d"),
            )

            if len(df) < 2:
                return 0.0

            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            return float((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100)

        except Exception:
            return 0.0


    def _save_and_print(self) -> pd.DataFrame:
        """
            Save results to CSV and print performance summary.

            Sharpe and Sortino are annualized and use India's risk-free rate.

            Annualized Sharpe:
                monthly_sharpe = (mean_return - rf_monthly) / std_return
                annualized     = monthly_sharpe * sqrt(12)

                Why sqrt(12)?
                    Monthly std x sqrt(12) = annual std  (by random walk math)
                    So to annualize Sharpe: multiply by sqrt(12)

                Why rf_monthly?
                    Sharpe measures excess return over risk-free
                    India 10Y bond ~7% annual -> 0.583%/month
                    Ignoring this overstates Sharpe by ~0.5 in India

                Example:
                    mean return = 2.5%/month
                    std         = 3.0%/month
                    rf_monthly  = 0.583%/month

                    sharpe = (2.5 - 0.583) / 3.0 * sqrt(12) ~= 2.21

            Sortino uses only downside months (negative excess) in the denominator
        """
        results_df = pd.DataFrame(self.results)
        picks_df   = pd.DataFrame(self.all_picks)

        results_df.to_csv("outputs/walk_forward_results.csv", index=False)
        picks_df.to_csv("outputs/walk_forward_picks.csv",     index=False)

        print("")
        logger.info("Saved -> walk_forward_results.csv".center(69))
        logger.info("Saved -> walk_forward_picks.csv".center(68))

        if results_df.empty:
            logger.warning("No results generated -- check date range and data".center(71))
            return results_df

        port  = results_df["Portfolio_Return"]
        nifty = results_df["Nifty_Return"]
        alpha = results_df["Outperformance"]

        cump_port  = float(((1 + port  / 100).prod() - 1) * 100)
        cump_nifty = float(((1 + nifty / 100).prod() - 1) * 100)
        win_rate   = float((port > 0).mean()     * 100)
        beat_rate  = float((port > nifty).mean() * 100)

        # FIX: annualized Sharpe with India risk-free rate
        # excess = monthly return (decimal) minus monthly risk-free rate
        excess     = port / 100 - RF_MONTHLY
        # need at least 6 points for std to be meaningful; fewer = random noise
        sharpe_ann = (
            float(excess.mean() / excess.std() * np.sqrt(12))
            if len(excess) >= 6 and excess.std() > 0 else 0.0
        )

        # FIX: annualized Sortino with risk-free rate
        # downside = only months where excess return was negative
        downside_excess = excess[excess < 0]
        sortino_ann     = (
            float(excess.mean() / downside_excess.std() * np.sqrt(12))
            if len(downside_excess) > 1 else 0.0
        )

        equity = (1 + port / 100).cumprod()
        peak   = equity.cummax()
        dd     = (equity - peak) / peak * 100
        max_dd = float(dd.min())

        print("\n" + "─"*98)
        print("│" + "WALK-FORWARD SUMMARY".center(96) + "│")
        print("─"*98)

        mid       = 48
        width     = 96
        val_width = width - mid - 1

        rows = [
            ("Months tested",           str(len(results_df))),
            ("Holding period",          f"{self.holding_period} month(s) (calendar)"),
            ("Slippage (round-trip)",   f"{self.slippage_pct:.2f}%"),
            ("DIVIDER",                 ""),
            ("Portfolio return",        f"{cump_port:+.2f}%"),
            ("Nifty return",            f"{cump_nifty:+.2f}%"),
            ("Total alpha",             f"{(cump_port - cump_nifty):+.2f}%"),
            ("Win rate",                f"{win_rate:.1f}%"),
            ("Beat benchmark",          f"{beat_rate:.1f}%"),
            ("Sharpe (annualized)",     f"{sharpe_ann:.3f}"),
            ("Sortino (annualized)",    f"{sortino_ann:.3f}"),
            ("Max drawdown",            f"{max_dd:.2f}%"),
            ("Avg monthly alpha",       f"{alpha.mean():+.2f}%"),
        ]

        for label, value in rows:
            if label == "DIVIDER":
                print(f"│{'─'*mid}:{'─'*val_width}│")
            else:
                print(f"│{label.center(mid)}:{value.center(val_width)}│")

        print("─"*98 + "\n")

        return results_df


    def run(self, start_year: int = 2024, months: int = 12) -> pd.DataFrame:
        """
            Run walk-forward validation month by month
            Skips first warmup_months so MA200 has enough history before first test

            Example with warmup_months = 6, months = 12:
                Jan-Jun 2024 -> skipped (warmup)
                Jul 2024     -> first test
                ...
                Dec 2024     -> last test  (6 test months total)

            Returns dataframe and saves CSVs
        """
        start = datetime(start_year, 1, 15)

        logger.info("─"*71)
        logger.info("WALK-FORWARD START".center(69))
        logger.info("─"*71)

        total_test_months = months - self.warmup_months

        if total_test_months <= 0:
            logger.error("warmup_months (%d) >= months (%d) — no test"
                "months left Increase months or reduce warmup_months",self.warmup_months, months)
            return pd.DataFrame()

        for i in range(self.warmup_months, months):
            screen_date = start + relativedelta(months=i)
            month_str   = screen_date.strftime("%b %Y")
            month_num   = i - self.warmup_months + 1

            print("")
            logger.info("─"*71)
            text = f"Month {month_num} / {total_test_months} -- {month_str}"
            logger.info("│" + text.center(69) + "│")

            picks = self.screen_blind(screen_date)
            if not picks:
                logger.warning("Skipping %s -- no picks".center(69), month_str)
                continue

            port_return, nifty_return, trades = self._measure_returns(picks, screen_date)

            if not trades:
                logger.warning("Skipping %s -- all trades failed (no forward data)".center(69), month_str)
                continue

            self.results.append({
                "Month":            month_str,
                "Portfolio_Return": round(port_return,  2),
                "Nifty_Return":     round(nifty_return, 2),
                "Outperformance":   round(port_return - nifty_return, 2),
                "Num_Stocks":       len(trades),
            })

            for t in trades:
                t["Month"] = month_str
                self.all_picks.append(t)

            logger.info("│" + "PERFORMANCE".center(69) + "│")
            logger.info("─"*71)

            line1 = f"{'Portfolio (net)':<18} - {port_return:+8.2f}%"
            line2 = f"{'Nifty':<18} - {nifty_return:+8.2f}%"
            line3 = f"{'Alpha':<18} - {(port_return - nifty_return):+8.2f}%"

            logger.info("│" + line1.center(69) + "│")
            logger.info("│" + line2.center(69) + "│")
            logger.info("│" + line3.center(69) + "│")

            logger.info("─"*71)

        return self._save_and_print()


if __name__ == "__main__":
    wf = WalkForwardValidator(warmup_months = 6,holding_period = 1,top_n = 3,slippage_pct = 0.2)
    wf.run(start_year=2024, months=12)
