"""
    Backtest for my stock screener

    Runs the strategy month by month on past data to see if it actually makes money

    How it works:
    Each month,
        run screener using past data only
        pick top stocks
        buy them
        hold for some days
        sell them
        calculate profit
        compare with Nifty

    Important:
        We NEVER use future data while picking stocks
        Future data is only used after buying to check what happened

    Example:
        Jan -> pick stocks -> check Feb returns
        Feb -> pick stocks -> check Mar returns
"""

import logging
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# use this to move month-by-month (timedelta(days=30) is wrong because months have different lengths)
# use timedelta for holding days (exact days)
# use relativedelta for moving month by month (calendar months)
from dateutil.relativedelta import relativedelta

import yfinance as yf

from src.screener  import StockScreener
from src.stock_list import get_stock_list

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ││   %(levelname)s   ││    %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler("logs/backtest.log", mode="a")]
)
logger = logging.getLogger(__name__)


class Backtest:
    def __init__(self,backtest_months: int = 12,lookback_days: int = 400,top_n: int = 3,holding_days: int = 30,start_year: int = 2024,start_month: int = 1):
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
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.holding_days = holding_days
        self.start_year = start_year
        self.start_month = start_month
        self.stock_list = get_stock_list()
        self.monthly_results = []
        self.all_picks = []

        logger.info("─"*71)
        logger.info("BACKTEST INFO".center(69))
        logger.info("─"*71)
        logger.info("Months      :   %d".center(63), backtest_months)
        logger.info("Top N       :   %d".center(63), top_n)
        logger.info("Holding Days  :   %d".center(61), holding_days)
        logger.info(" Start       :   %d-%02d".center(67), start_year, start_month)
        logger.info("─"*71 + "\n")

    def screen_on_date(self, screen_date: datetime) -> list:
        text = f"Using screener on {screen_date.strftime('%Y-%m-%d')}"
        logger.info("│" + text.center(69) + "│")
        logger.info("─"*71 + "\n")

        screener = StockScreener(tickers=self.stock_list,lookback_days=self.lookback_days,screen_date=screen_date)
        screener.download_data(max_workers=6)
        screener.calculate_indicators()
        screener.generate_signals()

        if screener.results is None or screener.results.empty:
            logger.warning("  No picks generated for %s",
                           screen_date.strftime("%b %Y"))
            return []

        top = screener.results.head(self.top_n)
        picks = [(row["Ticker"], row["Combined_Score"], row["Price"])for _, row in top.iterrows()]

        logger.info("─"*71)
        logger.info("│" + "TOP PICKS".center(69) + "│")
        logger.info("─"*71)

        for _ , row in top.iterrows():
            ticker = row["Ticker"]
            label  = row["Signal"]

            line = f"{ticker:<15} - {label:>12}"
            logger.info("│" + line.center(69) + "│")

        logger.info("─"*71)

        return picks


    def measure_returns(self,picks: list,screen_date: datetime) -> tuple:
        exit_date = screen_date + timedelta(days=self.holding_days)
        trades = []

        logger.info("│"+ "TRADES".center(69) +"│")
        logger.info("─"*71)
        for ticker, score, entry_price in picks:
            try:
                df = yf.download(ticker,start=screen_date.strftime("%Y-%m-%d"),end=exit_date.strftime("%Y-%m-%d"),progress=False,auto_adjust=True)
                if len(df) < 2:
                    logger.warning("  %s: not enough forward data", ticker)
                    continue

                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                    
                close = close.dropna()
                if len(close) < 2:
                    return None, None, []

                entry = float(close.iloc[0])
                exit  = float(close.iloc[-1])

                if entry == 0:
                    continue

                ret = (exit - entry) / entry * 100

                trades.append({
                    "Ticker":      ticker,
                    "Score":       score,
                    "Entry_Price": round(entry, 2),
                    "Exit_Price":  round(exit,  2),
                    "Return":      round(ret, 2)
                })

                line = f"{ticker:<14} │ entry ={entry:>8.2f} │ exit ={exit:>8.2f} │ return ={ret:+6.2f}%"
                logger.info("│ " + line.ljust(61) + " │")

            except Exception as exc:
                logger.warning("  Return calc failed for %s: %s".center(69), ticker, exc)
                
        logger.info("─"*71)
    
        portfolio_return = (float(np.mean([t["Return"] for t in trades]))if trades else 0.0)
        nifty_return = self.get_nifty_return(screen_date, exit_date)

        return portfolio_return, nifty_return, trades

    def get_nifty_return(self,start: datetime,end:   datetime) -> float:
        """
            Download Nifty 50 return
            Nifty 50 = index of top 50 companies in India
            tells:
                did you beat the market or not
                
            Example:
                Your strategy	     Nifty	          Meaning
                     +5%              +3%	    good (you beat market)
                     +5%	          +8%	    bad  (market better)
                     -2%	          -5%	    good (you lost less)
        """
        try:
            df = yf.download("^NSEI",start=start.strftime("%Y-%m-%d"),end=end.strftime("%Y-%m-%d"),progress=False,auto_adjust=True)
            
            if len(df) < 2:
                return 0.0
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
                
            return float((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100)
            
        except Exception:
            return 0.0

    def save_and_print(self) -> pd.DataFrame:
        """
            Save CSVs and print performance summary
        """
        results_df = pd.DataFrame(self.monthly_results)
        picks_df   = pd.DataFrame(self.all_picks)

        results_df.to_csv("outputs/backtests/backtest_results.csv",index=False)
        picks_df.to_csv("outputs/backtests/backtest_picks.csv",index=False)

        print("")
        logger.info("Saved → backtest_results.csv".center(69))
        logger.info("Saved → backtest_picks.csv".center(68))

        if results_df.empty:
            logger.warning("No results generated — check date range and data".center(71))
            return results_df

        port  = results_df["Portfolio_Return"]
        nifty = results_df["Nifty_Return"]
        alpha = results_df["Outperformance"]

        cump_port  = float(((1 + port  / 100).prod() - 1) * 100) # compound_returns
        """
            ∏(1+r/100(make it in decimal so /100)​)−1
            final backtest result = total % return of your strategy over the whole period
            so basically, total backtest results all mix it and see is it good or not
        """
        
        cump_nifty = float(((1 + nifty / 100).prod() - 1) * 100)
        win_rate  = float((port > 0).mean()   * 100)
        beat_rate = float((port > nifty).mean() * 100)
        sharpe    = float(port.mean() / port.std()) if port.std() > 0 else 0.0

        equity      = (1 + port / 100).cumprod()
        """
            x = [2, 3, 4]
            x.cumprod() -> [2, 6, 24]
            Because:
                2
                2×3 = 6
                6×4 = 24
                
            port = [2, -1, 3]
            Start with ₹100:
                Day 1 - 100 * 1.02 = 102
                Day 2 - 102 * 0.99 = 100.98
                Day 3 - 100.98 * 1.03 ≈ 104

            equity = [1.02, 1.0098, 1.04]
            Equity = your money over time
            so how ur money growed over each time(return)
        """
        peak        = equity.cummax()
        dd          = (equity - peak) / peak * 100
        max_dd      = float(dd.min())

        downside    = port[port < 0]
        """
            Example:
                port = [2, -1, 3, -4, 1]

                downside = port[port < 0]
                
                [-1, -4]
        """
        sortino     = float(port.mean() / downside.std()) if len(downside) > 1 else 0.0

        print("\n" + "─"*98)
        print("│" + "BACKTEST SUMMARY".center(96) + "│")
        print("─"*98)

        mid   = 48
        width = 96

        months = str(len(results_df))
        port = f"{cump_port:+.2f}%"
        nifty = f"{cump_nifty:+.2f}%"
        alpha_val = f"{(cump_port - cump_nifty):+.2f}%"
        wr = f"{win_rate:.1f}%"
        bb = f"{beat_rate:.1f}%"
        sr = f"{sharpe:.3f}"
        so = f"{sortino:.3f}"
        md = f"{max_dd:.2f}%"
        am = f"{alpha.mean():+.2f}%"

        rows = [
            ("Months tested",     months),
            ("Portfolio return",  port),
            ("Nifty return",      nifty),
            ("Total alpha",       alpha_val),
            ("Win rate",          wr),
            ("Beat benchmark",    bb),
            ("Sharpe ratio",      sr),
            ("Sortino ratio",     so),
            ("Max drawdown",      md),
            ("Avg monthly alpha", am),
        ]

        val_width = width - mid - 1

        for label, value in rows:
            print(f"│{label.center(mid)}:{value.center(val_width)}│")

        print("─"*98 + "\n")

        return results_df


    def run(self) -> pd.DataFrame:
        """
            run backtest month by month and store results
            returns dataframe and saves csv files
        """
        start = datetime(self.start_year, self.start_month, 15)

        logger.info("─"*71)
        logger.info("BACKTEST START - %d months from %s".center(69),self.backtest_months,start.strftime("%b %Y"))
        logger.info("─"*71)

        for i in range(self.backtest_months):
            screen_date = start + relativedelta(months=i) # moves the date forward by i months
            month_str   = screen_date.strftime("%b %Y")

            print("")
            logger.info("─"*71)
            text = "Month %d / %d — %s" % (i + 1, self.backtest_months, month_str)
            logger.info("│" + text.center(69) + "│")
            
            picks = self.screen_on_date(screen_date)
            if not picks:
                logger.warning("Skipping %s — no picks".center(69), month_str)
                continue

            port_return, nifty_return, trades = self.measure_returns(picks, screen_date)

            self.monthly_results.append({
                "Month":               month_str,
                "Portfolio_Return":    round(port_return,2),
                "Nifty_Return":        round(nifty_return,2),
                "Outperformance":      round(port_return - nifty_return,2),
                "Num_Stocks":          len(trades)
            })

            for t in trades:
                t["Month"] = month_str
                self.all_picks.append(t)

            logger.info("│" + "PERFORMANCE".center(69) + "│")
            logger.info("─"*71)

            line1 = f"{'Portfolio':<12} - {port_return:+8.2f}%"
            line2 = f"{'Nifty':<12} - {nifty_return:+8.2f}%"
            line3 = f"{'Alpha':<12} - {(port_return - nifty_return):+8.2f}%"

            logger.info("│" + line1.center(69) + "│")
            logger.info("│" + line2.center(69) + "│")
            logger.info("│" + line3.center(69) + "│")

            logger.info("─"*71)

        return self.save_and_print()



if __name__ == "__main__":
    bt = Backtest(
        backtest_months=12,
        lookback_days=400,
        top_n=3,
        holding_days=30,
        start_year=2024,
        start_month=1,
    )
    bt.run()
