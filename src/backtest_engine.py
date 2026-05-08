from __future__ import annotations # wait until later before checking type hints
# error handling for safe type hints thats all

import logging
from dataclasses import dataclass, field
"""
    dataclass:
        Automatically creates boring class setup code like:
            __init__(),__repr__(),etc.

        Example:
            @dataclass
            class Student:
                name: str
                age: int

            s = Student("Ratan", 20)
            
        instead of manually writing:
            class Student:
                def __init__(self, name, age):
                    self.name = name
                    self.age = age

        Very useful for classes mainly storing data.


    field(init=False):
        Tells dataclass:
            "do not ask user to pass this variable"

        Used for values calculated automatically later

        Example:
            @dataclass
            class Trade:
                entry: float
                exit: float
                pnl: float = field(init=False)

                def __post_init__(self):
                    self.pnl = self.exit - self.entry

            t = Trade(100, 120)

        Here pnl is NOT passed manually
        It is computed automatically after object creation
"""
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INDIA_RF_ANNUAL  = 0.07
INDIA_RF_MONTHLY = INDIA_RF_ANNUAL / 12


def calculate_sharpe(returns: np.ndarray,risk_free: float = 0.0,annualise: bool = False,periods: int = 12) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std())
    if std == 0:
        return 0.0
    sharpe = float((returns.mean() - risk_free) / std)
    if annualise:
        sharpe *= np.sqrt(periods)
    return sharpe

def calculate_max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    equity = np.insert((1 + returns / 100).cumprod(), 0, 1.0) # insert is for insert 1 at index 0 bcz starts fro 1 = 100%
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    return float(drawdown.min())

def calculate_win_rate(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean() * 100)


@dataclass
class Trade:
    ticker: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float

    # Computed automatically — do NOT pass these in
    pnl: float = field(init=False)
    return_pct: float = field(init=False)

    def __post_init__(self):
        # Called automatically after __init__ to compute pnl and return percent
        if self.entry_price <= 0:
            raise ValueError(f"entry_price must be > 0 for {self.ticker}, got {self.entry_price}")
        self.pnl = round(self.exit_price - self.entry_price, 4)
        self.return_pct = round((self.exit_price - self.entry_price) / self.entry_price * 100,4)

    def is_winner(self) -> bool:
        # Returns True if trade was profitable
        return self.return_pct > 0

    def holding_days(self) -> int:
        # Number of calendar days the position was held
        return (self.exit_date - self.entry_date).days

    def __repr__(self) -> str: # python automatically calls print(t1.__repr__())
        return (f"Trade({self.ticker}  "
            f"{self.entry_date.strftime('%Y-%m-%d')} -> "
            f"{self.exit_date.strftime('%Y-%m-%d')}  "
            f"entry = {self.entry_price:.2f}  "
            f"exit = {self.exit_price:.2f}  "
            f"return = {self.return_pct:+.2f}%)"
        )
    """
        __repr__ controls how the object looks when printed.

        Example:
            print(trade)

        Python automatically does something like:
            print(trade.__repr__())

        So __repr__ should RETURN a readable string

        Without __repr__:
            <Trade object at 0x8172ab>

        With __repr__:
            Trade(RELIANCE.NS return=+5.00%)

        Useful for debugging and clean terminal output
    """
                

@dataclass
class BacktestResult:
    period_label: str
    trades: list[Trade]

    num_trades: int = field(init=False)
    total_return: float = field(init=False)
    sharpe_ratio: float = field(init=False)
    max_drawdown: float = field(init=False)
    win_rate: float = field(init=False)

    def __post_init__(self):
        self.num_trades = len(self.trades)

        if self.num_trades == 0:
            self.total_return = 0.0
            self.sharpe_ratio = 0.0
            self.max_drawdown = 0.0
            self.win_rate = 0.0
            return

        returns = np.array([t.return_pct for t in self.trades])

        self.total_return = float(returns.mean())
        self.win_rate     = calculate_win_rate(returns)
        self.max_drawdown = calculate_max_drawdown(returns)

        # with N < 4 std is unstable and the number is noise
        if self.num_trades < 4:
            self.sharpe_ratio = 0.0
        else:
            self.sharpe_ratio = calculate_sharpe(returns)

    def summary(self) -> dict:
        # Return metrics as a flat dict — useful for building DataFrames
        return {
            "period"      : self.period_label,
            "total_return": round(self.total_return, 2),
            "sharpe"      : round(self.sharpe_ratio, 3),
            "max_drawdown": round(self.max_drawdown, 2),
            "win_rate"    : round(self.win_rate, 1),
            "num_trades"  : self.num_trades
        }

    def __repr__(self) -> str:
        return (
            f"BacktestResult({self.period_label}  "
            f"return = {self.total_return:+.2f}%  "
            f"sharpe = {self.sharpe_ratio:.3f}  "
            f"dd = {self.max_drawdown:.2f}%  "
            f"win = {self.win_rate:.1f}%  "
            f"trades = {self.num_trades})"
        )


class BacktestEngine:
    def __init__(self,hold_days: int = 30,slippage_pct: float = 0.06,risk_free_annual: float = INDIA_RF_ANNUAL):
        self.hold_days        = hold_days
        self.slippage_pct     = slippage_pct
        self.risk_free_annual = risk_free_annual
        self.rf_monthly       = risk_free_annual / 12  # derived for monthly Sharpe
        self.results:         List[BacktestResult] = []

        logger.info("BacktestEngine | hold_days = %d  slippage = %.2f%%  rf = %.2f%%/yr",hold_days, slippage_pct, risk_free_annual * 100)

    def run_backtest(self,screener_signals: list,prices_df: pd.DataFrame,period_label: str = "") -> BacktestResult:
        # without this, BacktestEngine(hold_days = 30) had zero effect
        prices_df = prices_df.iloc[:self.hold_days]

        trades = []

        for ticker, score, entry_price, entry_date in screener_signals:
            if ticker not in prices_df.columns:
                logger.warning("  %-20s not found in prices_df - skip", ticker)
                continue

            series = prices_df[ticker].dropna()
            if len(series) < 1:
                logger.warning("  %-20s no forward prices - skip", ticker)
                continue

            actual_entry = entry_price * (1 + self.slippage_pct / 100)
            raw_exit = float(series.iloc[-1])
            actual_exit = raw_exit * (1 - self.slippage_pct / 100)

            exit_date = series.index[-1]
            if hasattr(exit_date, "to_pydatetime"): # pandas Timestamp → normal Python datetime
                exit_date = exit_date.to_pydatetime()

            trade = Trade(
                ticker = ticker,
                entry_date = entry_date,
                entry_price = round(actual_entry, 4),
                exit_date = exit_date,
                exit_price = round(actual_exit,  4)
            )
            trades.append(trade)
            logger.info("  %s", trade)

        result = BacktestResult(period_label = period_label, trades = trades)
        self.results.append(result)
        logger.info("  Result: %s", result)
        return result


    def walk_forward(self, all_windows: list) -> pd.DataFrame:
        """
        Run N independent backtest windows with identical parameters.

        This is the key methodological difference vs a simple backtest:
            Simple backtest  : one long test, parameters chosen after seeing results
            Walk-forward     : N independent out-of-sample tests, same parameters
                               (you can't curve-fit parameters to data you haven't seen)

        Args:
            all_windows : List of (period_label, screener_signals, prices_df).
                          One element per test period.

        Returns:
            pd.DataFrame: Summary metrics for each period.
                          Columns: period, total_return, sharpe, max_drawdown,
                                   win_rate, num_trades
        """
        logger.info("Walk-forward: %d windows", len(all_windows))
        summaries = []

        for period_label, signals, prices_df in all_windows:
            logger.info("\n  -- %s --", period_label)
            result = self.run_backtest(signals, prices_df, period_label)
            summaries.append(result.summary())

        df = pd.DataFrame(summaries)
        logger.info("\nWalk-forward complete: %d periods", len(df))
        return df


    def aggregate(self) -> dict:
        """
        Aggregate metrics across ALL BacktestResult windows.

        Call this after walk_forward() or multiple run_backtest() calls.

        Sharpe here is TEMPORAL (across monthly periods) and annualised:
            formula: (mean_monthly_excess / std_monthly) * sqrt(12)
            where excess = monthly return - monthly risk-free rate

        Returns:
            dict of aggregate metrics.
        """
        if not self.results:
            logger.warning("No results to aggregate — run walk_forward() first")
            return {}

        all_returns = np.array([r.total_return for r in self.results])

        # FIX: annualised Sharpe with risk-free rate
        # was: calculate_sharpe(all_returns)  — raw, no RF, not annualised
        # now: RF-adjusted monthly excess, then annualised by sqrt(12)
        sharpe_ann = calculate_sharpe(
            all_returns,
            risk_free = self.rf_monthly * 100,  # convert to % to match returns units
            annualise = True,
            periods   = 12,
        )

        return {
            "num_periods":     len(self.results),
            "mean_return_%":   round(float(all_returns.mean()), 2),
            "total_return_%":  round(float(((1 + all_returns / 100).prod() - 1) * 100), 2),
            "sharpe_ann":      round(sharpe_ann, 3),
            "max_drawdown_%":  round(calculate_max_drawdown(all_returns), 2),
            "win_rate_%":      round(calculate_win_rate(all_returns), 1),
        }


    def print_summary(self) -> None:
        """Print aggregate summary to terminal — matches backtest.py style."""
        m = self.aggregate()
        if not m:
            return

        mid       = 34
        val_width = 34

        print("\n" + "─"*71)
        print("│" + "BACKTEST ENGINE — SUMMARY".center(69) + "│")
        print("─"*71)

        rows = [
            ("Periods tested",     str(m["num_periods"])),
            ("Mean return",        f"{m['mean_return_%']:+.2f}%"),
            ("Total return",       f"{m['total_return_%']:+.2f}%"),
            ("Sharpe (annualised)", f"{m['sharpe_ann']:.3f}"),
            ("Max drawdown",       f"{m['max_drawdown_%']:.2f}%"),
            ("Win rate",           f"{m['win_rate_%']:.1f}%"),
        ]
        for label, value in rows:
            print("│" + label.center(mid) + ":" + value.center(val_width) + "│")
        print("─"*71 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)-8s | %(message)s")

    print("─" * 71)
    print(" BACKTEST ENGINE SMOKE TEST ".center(71))
    print("─" * 71)


    print("\n[1] Trade dataclass:")
    entry_date = datetime(2024, 1, 15)
    exit_date = datetime(2024, 2, 14)

    t1 = Trade("RELIANCE.NS", entry_date, 2400.0, exit_date, 2520.0)
    t2 = Trade("TCS.NS", entry_date, 3500.0, exit_date, 3360.0)
    t3 = Trade("INFY.NS", entry_date, 1500.0, exit_date, 1575.0)

    print(f"  {t1}")
    print(f"  {t2}")
    print(f"  {t3}")

    assert abs(t1.return_pct - 5.0)   < 0.01, f"Trade 1 return wrong: {t1.return_pct}"
    assert abs(t2.return_pct - (-4.0)) < 0.01, f"Trade 2 return wrong: {t2.return_pct}"
    assert t1.is_winner() is True
    assert t2.is_winner() is False
    assert t1.holding_days() == 30
    print("  Trade tests pass")


    print("\n[2] BacktestResult dataclass:")
    result = BacktestResult("Jan 2024", [t1, t2, t3])
    print(f"  {result}")

    assert result.num_trades == 3
    assert result.win_rate > 0
    assert abs(result.win_rate - 66.67) < 0.1, f"Win rate wrong: {result.win_rate}"
    # sharpe_ratio should be 0.0 because num_trades < 4
    assert result.sharpe_ratio == 0.0, f"Expected 0.0 for N<4, got {result.sharpe_ratio}"
    print("  Summary:", result.summary())
    print("  BacktestResult tests pass")


    print("\n[3] Metric helpers:")
    returns = np.array([5.0, -4.0, 5.0, 2.0, -1.0, 3.0])

    sharpe_raw = calculate_sharpe(returns)
    sharpe_ann = calculate_sharpe(returns, annualise=True)
    max_dd     = calculate_max_drawdown(returns)
    wr         = calculate_win_rate(returns)

    print(f"  Returns:          {returns}")
    print(f"  Sharpe (raw):     {sharpe_raw:.3f}")
    print(f"  Sharpe (annual):  {sharpe_ann:.3f}  (= raw * sqrt(12))")
    print(f"  Max Drawdown:     {max_dd:.2f}%")
    print(f"  Win Rate:         {wr:.1f}%")

    assert sharpe_raw > 0, "Sharpe should be positive for net positive returns"
    assert abs(sharpe_ann - sharpe_raw * np.sqrt(12)) < 0.001, "Annualisation wrong"
    assert max_dd < 0,"Max drawdown should be negative"
    assert abs(wr - 66.7) < 0.1, f"Win rate should be ~66.7%, got {wr}"
    print("  Metric helper tests pass")


    print("\n[4] BacktestEngine.run_backtest():")
    engine = BacktestEngine(hold_days = 5, slippage_pct = 0.06)

    signals = [("RELIANCE.NS", 0.8, 2400.0, entry_date),("TCS.NS",0.7, 3500.0, entry_date),("INFY.NS",0.6, 1500.0, entry_date)]

    dates = pd.date_range(entry_date, periods = 10, freq = "B")
    # create 10 trading/business dates starting from entry_date
    # weekends skipped because freq="B"
    prices = pd.DataFrame({
        "RELIANCE.NS": [2400, 2420, 2450, 2430, 2470,2490, 2510, 2500, 2530, 2550],
        "TCS.NS"     : [3500, 3520, 3480, 3510, 3550,3540, 3560, 3580, 3600, 3620],
        "INFY.NS"    : [1500, 1490, 1510, 1520, 1530,1515, 1540, 1550, 1545, 1560]
    }, index = dates)

    result = engine.run_backtest(signals, prices, "Jan 2024")

    assert result.num_trades == 3
    assert result.total_return != 0.0

    reliance_trade = next(t for t in result.trades if t.ticker == "RELIANCE.NS")
    expected_exit  = 2470 * (1 - 0.06 / 100)
    assert abs(reliance_trade.exit_price - expected_exit) < 0.01, (f"hold_days not enforced: expected exit ~{expected_exit:.2f}, "
        f"got {reliance_trade.exit_price}"
    )

    print(f"  Result: {result}")
    engine.print_summary()
    print("  BacktestEngine tests pass")

    print("\n" + "─" * 71)
    print(" ALL SMOKE TESTS PASSED ".center(71))
    print("─" * 71)
