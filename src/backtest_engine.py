"""
backtest_engine.py — P3 Skeleton (Phase 1 deliverable)

This is the FOUNDATION for the full P3 backtesting engine.

What's here (Phase 1):
    - Trade dataclass         — represents one completed trade
    - BacktestResult dataclass — aggregates metrics for one test window
    - Metric helper functions  — Sharpe, Max Drawdown, Win Rate
    - BacktestEngine class    — skeleton with run_backtest() and walk_forward()
    - Smoke test at bottom    — proves all dataclasses work

What gets built here in Phase 2 (16 Mar – 15 Apr):
    - Full walk-forward validator (12 independent windows)
    - Transaction costs (slippage + brokerage)
    - Annualised Sharpe ratio
    - Equity curve export
    - Full integration with screener output

This file is separate from backtest.py intentionally.
    backtest.py     = P2 (uses screener + yfinance, runs actual historical test)
    backtest_engine.py = P3 (pure engine, receives data, computes metrics)

The separation matters in interviews:
    "I separated the data layer from the computation layer.
     The engine doesn't know or care where prices come from."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ================================================================== #
#  METRIC HELPER FUNCTIONS                                             #
#  Standalone so they can be unit tested independently                #
# ================================================================== #

def calculate_sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """
    Sharpe Ratio = (mean_return - risk_free) / std_dev

    Measures risk-adjusted return.
    Higher = better.
        1.0 is acceptable
        1.5+ is good
        2.0+ is excellent

    Note: This is the RAW monthly Sharpe (not annualised).
          Phase 2 will multiply by sqrt(12) to annualise.

    Args:
        returns   : Array of period returns (%).
        risk_free : Risk-free rate per period (default 0.0).

    Returns:
        float: Sharpe ratio. 0.0 if std_dev is zero.
    """
    if len(returns) == 0:
        return 0.0
    std = float(returns.std())
    if std == 0:
        return 0.0
    return float((returns.mean() - risk_free) / std)


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Max Drawdown = (peak - trough) / peak

    The largest peak-to-trough decline in the equity curve.
    Returned as a NEGATIVE percentage.

    Why it matters:
        A strategy with 30% return but -40% drawdown is psychologically
        impossible to trade. Drawdown kills discipline.
        Target for your screener: max DD < 15%.

    Args:
        returns : Array of period returns (%).

    Returns:
        float: Max drawdown as a negative %. E.g. -12.5 means -12.5% drawdown.
    """
    if len(returns) == 0:
        return 0.0
    equity       = (1 + returns / 100).cumprod()
    running_max  = np.maximum.accumulate(equity)
    drawdown     = (equity - running_max) / running_max * 100
    return float(drawdown.min())


def calculate_win_rate(returns: np.ndarray) -> float:
    """
    Win Rate = num_winning_periods / total_periods × 100

    Args:
        returns : Array of period returns (%).

    Returns:
        float: Win rate as percentage. E.g. 60.0 = 60% win rate.
    """
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean() * 100)


@dataclass
class Trade:
    """
    Represents a single completed trade.

    pnl and return_pct are computed automatically from
    entry_price and exit_price — you don't pass them in.

    Usage:
        trade = Trade(
            ticker      = "RELIANCE.NS",
            entry_date  = datetime(2024, 1, 15),
            entry_price = 2400.0,
            exit_date   = datetime(2024, 2, 14),
            exit_price  = 2520.0,
        )
        print(trade.return_pct)  # → +5.0%
        print(trade.pnl)         # → +120.0
    """
    ticker:      str
    entry_date:  datetime
    entry_price: float
    exit_date:   datetime
    exit_price:  float

    # Computed automatically — do NOT pass these in
    pnl:        float = field(init=False)
    return_pct: float = field(init=False)

    def __post_init__(self):
        """Called automatically after __init__ to compute derived fields."""
        if self.entry_price == 0:
            raise ValueError(f"entry_price cannot be 0 for {self.ticker}")
        self.pnl        = round(self.exit_price - self.entry_price, 4)
        self.return_pct = round(
            (self.exit_price - self.entry_price) / self.entry_price * 100,
            4
        )

    def is_winner(self) -> bool:
        """Returns True if trade was profitable."""
        return self.return_pct > 0

    def holding_days(self) -> int:
        """Number of days the position was held."""
        return (self.exit_date - self.entry_date).days

    def __repr__(self) -> str:
        return (
            f"Trade({self.ticker}  "
            f"{self.entry_date.strftime('%Y-%m-%d')} → "
            f"{self.exit_date.strftime('%Y-%m-%d')}  "
            f"entry={self.entry_price:.2f}  "
            f"exit={self.exit_price:.2f}  "
            f"return={self.return_pct:+.2f}%)"
        )


@dataclass
class BacktestResult:
    """
    Aggregated performance metrics for one backtest window.

    All metrics (total_return, sharpe_ratio, etc.) are computed
    automatically from the list of Trade objects you pass in.

    Usage:
        result = BacktestResult(
            period_label = "Jan 2024",
            trades       = [trade1, trade2, trade3],
        )
        print(result.sharpe_ratio)
        print(result.win_rate)
        print(result.summary())
    """
    period_label: str
    trades:       List[Trade]

    # Computed automatically — do NOT pass these in
    total_return: float = field(init=False)
    sharpe_ratio: float = field(init=False)
    max_drawdown: float = field(init=False)
    win_rate:     float = field(init=False)
    num_trades:   int   = field(init=False)

    def __post_init__(self):
        """Compute all metrics from the trade list."""
        self.num_trades = len(self.trades)

        if self.num_trades == 0:
            self.total_return = 0.0
            self.sharpe_ratio = 0.0
            self.max_drawdown = 0.0
            self.win_rate     = 0.0
            return

        returns = np.array([t.return_pct for t in self.trades])

        # Compounded total return across all trades
        self.total_return = float(((1 + returns / 100).prod() - 1) * 100)
        self.win_rate     = calculate_win_rate(returns)
        self.sharpe_ratio = calculate_sharpe(returns)
        self.max_drawdown = calculate_max_drawdown(returns)

    def summary(self) -> dict:
        """Return metrics as a flat dict (useful for building DataFrames)."""
        return {
            "period":       self.period_label,
            "total_return": round(self.total_return, 2),
            "sharpe":       round(self.sharpe_ratio, 3),
            "max_drawdown": round(self.max_drawdown, 2),
            "win_rate":     round(self.win_rate, 1),
            "num_trades":   self.num_trades,
        }

    def __repr__(self) -> str:
        return (
            f"BacktestResult({self.period_label}  "
            f"return={self.total_return:+.2f}%  "
            f"sharpe={self.sharpe_ratio:.3f}  "
            f"dd={self.max_drawdown:.2f}%  "
            f"win={self.win_rate:.1f}%  "
            f"trades={self.num_trades})"
        )


class BacktestEngine:
    """
    Pure backtesting engine.

    This class RECEIVES prepared data — it does NOT download from yfinance.
    That separation is intentional:
        - Screener handles data acquisition
        - BacktestEngine handles computation
        - Easy to swap data sources without touching the math

    Phase 1 (now):
        run_backtest()  — test one window, returns BacktestResult
        walk_forward()  — test N independent windows
        aggregate()     — summarise across all windows

    Phase 2 (April):
        Transaction cost modelling
        Annualised Sharpe (× √12)
        Equity curve export
        Benchmark comparison per window
    """

    def __init__(
        self,
        hold_days:    int   = 30,
        slippage_pct: float = 0.06,
    ):
        """
        Args:
            hold_days    : Days to hold each position. Default 30.
            slippage_pct : Round-trip transaction cost (brokerage + slippage).
                           0.06% is realistic for Indian equities (Zerodha + impact).
                           Applied as: buy at price×(1 + slippage), sell at price×(1 - slippage).
        """
        self.hold_days    = hold_days
        self.slippage_pct = slippage_pct
        self.results:     List[BacktestResult] = []

        logger.info(
            "BacktestEngine | hold_days=%d  slippage=%.2f%%",
            hold_days, slippage_pct
        )


    def run_backtest(
        self,
        screener_signals: list,
        prices_df:        pd.DataFrame,
        period_label:     str = "",
    ) -> BacktestResult:
        """
        Run one backtest window. Returns BacktestResult.

        Args:
            screener_signals : List of tuples from the screener:
                               (ticker, score, entry_price, entry_date)
                               - ticker      : str  e.g. "RELIANCE.NS"
                               - score       : float  e.g. 0.8
                               - entry_price : float  price at signal date
                               - entry_date  : datetime  signal date

            prices_df        : DataFrame of FORWARD prices (after entry_date).
                               - Index  : dates (DatetimeIndex)
                               - Columns: ticker symbols
                               Example:
                                               RELIANCE.NS   TCS.NS
                                   2024-01-16       2410     3510
                                   2024-01-17       2430     3530
                                   ...

            period_label     : Human-readable label. E.g. "Jan 2024".

        Returns:
            BacktestResult with all metrics.

        Important:
            prices_df must only contain FORWARD prices (after entry_date).
            Using current-day price in prices_df = look-ahead bias.
        """
        trades = []

        for ticker, score, entry_price, entry_date in screener_signals:
            if ticker not in prices_df.columns:
                logger.warning("  %-20s not found in prices_df — skip", ticker)
                continue

            series = prices_df[ticker].dropna()
            if len(series) < 1:
                logger.warning("  %-20s no forward prices — skip", ticker)
                continue

            # Apply buy slippage: you pay slightly more than the quoted price
            actual_entry = entry_price * (1 + self.slippage_pct / 100)

            # Exit at the last available price in the window
            raw_exit    = float(series.iloc[-1])
            # Apply sell slippage: you receive slightly less than quoted
            actual_exit = raw_exit * (1 - self.slippage_pct / 100)

            # exit_date is the last date in the price series
            exit_date = series.index[-1]
            if hasattr(exit_date, "to_pydatetime"):
                exit_date = exit_date.to_pydatetime()

            trade = Trade(
                ticker=ticker,
                entry_date=entry_date,
                entry_price=round(actual_entry, 4),
                exit_date=exit_date,
                exit_price=round(actual_exit, 4),
            )
            trades.append(trade)
            logger.info("  %s", trade)

        result = BacktestResult(period_label=period_label, trades=trades)
        self.results.append(result)
        logger.info("  Result: %s", result)
        return result


    def walk_forward(
        self,
        all_windows: list,
    ) -> pd.DataFrame:
        """
        Run N INDEPENDENT backtest windows. Same parameters for all.

        This is the KEY methodological difference vs a simple backtest:
            - Simple backtest: one long test, parameters chosen after seeing results
            - Walk-forward   : N independent out-of-sample tests, same parameters

        Walk-forward prevents overfitting because you can't curve-fit
        parameters to future data you haven't seen yet.

        Args:
            all_windows : List of (period_label, screener_signals, prices_df)
                          One element per test period.

        Returns:
            pd.DataFrame: Summary metrics for each period.
                          Columns: period, total_return, sharpe, max_drawdown,
                                   win_rate, num_trades
        """
        logger.info("Walk-forward: %d windows", len(all_windows))
        summaries = []

        for period_label, signals, prices_df in all_windows:
            logger.info("\n  ── %s ──", period_label)
            result = self.run_backtest(signals, prices_df, period_label)
            summaries.append(result.summary())

        df = pd.DataFrame(summaries)
        logger.info("\nWalk-forward complete: %d periods", len(df))
        return df


    def aggregate(self) -> dict:
        """
        Aggregate metrics across ALL BacktestResult windows.

        Call this after walk_forward() to get the overall picture.

        Returns:
            dict of aggregate metrics.
        """
        if not self.results:
            logger.warning("No results to aggregate — run walk_forward() first")
            return {}

        all_returns = np.array([r.total_return for r in self.results])

        return {
            "num_periods":     len(self.results),
            "mean_return_%":   round(float(all_returns.mean()), 2),
            "total_return_%":  round(float(((1 + all_returns / 100).prod() - 1) * 100), 2),
            "sharpe_ratio":    round(calculate_sharpe(all_returns), 3),
            "max_drawdown_%":  round(calculate_max_drawdown(all_returns), 2),
            "win_rate_%":      round(calculate_win_rate(all_returns), 1),
        }

    def print_summary(self) -> None:
        """Print aggregate summary to terminal."""
        m = self.aggregate()
        if not m:
            return
        print("\n" + "═" * 55)
        print(" BACKTEST ENGINE — SUMMARY ".center(55, "═"))
        print("═" * 55)
        for k, v in m.items():
            print(f"  {k:<28} {v}")
        print("═" * 55 + "\n")



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    print("=" * 55)
    print(" BACKTEST ENGINE SMOKE TEST ".center(55, "="))
    print("=" * 55)

    # ── Test 1: Trade dataclass ──────────────────────────────────── #
    print("\n[1] Trade dataclass:")
    entry_date = datetime(2024, 1, 15)
    exit_date  = datetime(2024, 2, 14)

    t1 = Trade("RELIANCE.NS", entry_date, 2400.0, exit_date, 2520.0)
    t2 = Trade("TCS.NS",      entry_date, 3500.0, exit_date, 3360.0)
    t3 = Trade("INFY.NS",     entry_date, 1500.0, exit_date, 1575.0)

    print(f"  {t1}")
    print(f"  {t2}")
    print(f"  {t3}")

    assert abs(t1.return_pct - 5.0) < 0.01,  f"Trade 1 return wrong: {t1.return_pct}"
    assert abs(t2.return_pct - (-4.0)) < 0.01, f"Trade 2 return wrong: {t2.return_pct}"
    assert t1.is_winner() is True
    assert t2.is_winner() is False
    assert t1.holding_days() == 30
    print("  ✓ Trade tests pass")

    # ── Test 2: BacktestResult dataclass ─────────────────────────── #
    print("\n[2] BacktestResult dataclass:")
    result = BacktestResult("Jan 2024", [t1, t2, t3])
    print(f"  {result}")

    assert result.num_trades == 3
    assert result.win_rate > 0
    # t1 and t3 are winners → win rate = 2/3 = 66.7%
    assert abs(result.win_rate - 66.67) < 0.1, f"Win rate wrong: {result.win_rate}"
    print("  Summary:", result.summary())
    print("  ✓ BacktestResult tests pass")

    # ── Test 3: Metric helpers ────────────────────────────────────── #
    print("\n[3] Metric helpers:")
    returns = np.array([5.0, -4.0, 5.0, 2.0, -1.0, 3.0])

    sharpe = calculate_sharpe(returns)
    max_dd = calculate_max_drawdown(returns)
    wr     = calculate_win_rate(returns)

    print(f"  Returns:      {returns}")
    print(f"  Sharpe:       {sharpe:.3f}")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"  Win Rate:     {wr:.1f}%")

    assert sharpe > 0,    "Sharpe should be positive for net positive returns"
    assert max_dd < 0,    "Max drawdown should be negative"
    assert abs(wr - 66.7) < 0.1, f"Win rate should be ~66.7%, got {wr}"
    print("  ✓ Metric helper tests pass")

    # ── Test 4: BacktestEngine with fake forward prices ───────────── #
    print("\n[4] BacktestEngine.run_backtest():")
    engine = BacktestEngine(hold_days=30, slippage_pct=0.06)

    signals = [
        ("RELIANCE.NS", 0.8, 2400.0, entry_date),
        ("TCS.NS",      0.7, 3500.0, entry_date),
        ("INFY.NS",     0.6, 1500.0, entry_date),
    ]

    # Fake forward prices (10 trading days)
    dates  = pd.date_range(entry_date, periods=10, freq="B")
    prices = pd.DataFrame({
        "RELIANCE.NS": [2400, 2420, 2450, 2430, 2470,
                        2490, 2510, 2500, 2530, 2550],
        "TCS.NS":      [3500, 3520, 3480, 3510, 3550,
                        3540, 3560, 3580, 3600, 3620],
        "INFY.NS":     [1500, 1490, 1510, 1520, 1530,
                        1515, 1540, 1550, 1545, 1560],
    }, index=dates)

    result = engine.run_backtest(signals, prices, "Jan 2024")

    assert result.num_trades == 3
    assert result.total_return != 0.0
    print(f"  Result: {result}")
    engine.print_summary()
    print("  ✓ BacktestEngine tests pass")

    print("\n" + "=" * 55)
    print(" ALL SMOKE TESTS PASSED ✓ ".center(55, "="))
    print("=" * 55)
