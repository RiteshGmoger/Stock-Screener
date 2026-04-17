"""
    Walk-forward backtester using MA + RSI signals.

    Each month:
        screen stocks using past data
        pick top N
        hold for fixed period
        compute returns with costs
        compare vs Nifty
    
    Runs monthly backtest:
        screen -> pick top stocks -> hold -> compute returns with cost -> compare with Nifty

    Used to check if strategy actually works over time
    Checks if strategy is profitable and consistent
"""

import argparse
import logging
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

# Use non-interactive backend so plots save without needing a display.
# Must be set BEFORE importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import yfinance as yf

# ── Import our own modules ────────────────────────────────────────────────
# Works both as:  python -m src.backtest   (package mode)
# and as:         python backtest.py       (standalone mode)
try:
    from src.screener            import StockScreener
    from src.stock_list          import get_stock_list, TEST_TICKERS, NIFTY_50_TICKERS
    from src.scoring             import StockScorer
    from src.portfolio_optimizer import VolatilityPortfolio
except ModuleNotFoundError:
    from screener            import StockScreener
    from stock_list          import get_stock_list, TEST_TICKERS, NIFTY_50_TICKERS
    from scoring             import StockScorer
    from portfolio_optimizer import VolatilityPortfolio


# ═══════════════════════════════════════════════════════════════════════════ #
#  Logging — same format as screener.py so logs are consistent              #
# ═══════════════════════════════════════════════════════════════════════════ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backtest.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════ #
#  CostModel — realistic execution simulation                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class CostModel:
    """
    Simulates real-world trading costs.

    Two types of costs:

    1. Transaction cost (brokerage + STT + exchange fees)
       Charged as a % on every buy and every sell.
       Typical India retail: 0.05% – 0.15% per leg.
       We default to 0.10% per leg.

    2. Slippage (market impact / bid-ask spread)
       You never get the exact closing price.
         Buying  → you pay a little MORE  (ask side)
         Selling → you receive a little LESS (bid side)
       Typical: 0.03% – 0.10% per leg.
       We default to 0.05% per leg.

    Round-trip cost (entry + exit combined):
        transaction: 0.10% × 2 = 0.20%
        slippage:    0.05% × 2 = 0.10%
        TOTAL:                   0.30%

    That sounds small, but across 12 months × 3 stocks × 2 trades = 72 legs,
    it adds up to ~10% of your gross return being eaten by costs.
    """

    def __init__(self, cost_pct: float = 0.10, slippage_pct: float = 0.05):
        """
        Args:
            cost_pct     : One-way brokerage cost in % (default 0.10 = 0.10%)
            slippage_pct : One-way slippage in %       (default 0.05 = 0.05%)
        """
        # Store as decimals internally
        self.cost     = cost_pct     / 100.0
        self.slippage = slippage_pct / 100.0

        # Descriptive strings for logging
        self.cost_pct_str     = f"{cost_pct:.3f}%"
        self.slippage_pct_str = f"{slippage_pct:.3f}%"

    def entry_price(self, raw_close: float) -> float:
        """
        Realistic buy price: you pay MORE than closing price.
        raw_close × (1 + slippage)

        Example: close=100, slippage=0.05%  →  entry=100.05
        """
        return raw_close * (1.0 + self.slippage)

    def exit_price(self, raw_close: float) -> float:
        """
        Realistic sell price: you receive LESS than closing price.
        raw_close × (1 − slippage)

        Example: close=110, slippage=0.05%  →  exit=109.945
        """
        return raw_close * (1.0 - self.slippage)

    def round_trip_cost_decimal(self) -> float:
        """
        Total % cost to complete one full trade (buy + sell).
        Returned as decimal: 0.002 means 0.2%.
        Deduct this from the gross return to get net return.
        """
        return 2.0 * self.cost

    def round_trip_cost_pct(self) -> float:
        """Same as above, returned as percentage: 0.2 means 0.2%."""
        return self.round_trip_cost_decimal() * 100.0


# ═══════════════════════════════════════════════════════════════════════════ #
#  QuantBacktest — main class                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class QuantBacktest:
    """
    Walk-forward backtester with full quant-level realism.

    Design guarantees:
        ✓  Zero look-ahead bias
               Screening ONLY sees data up to screen_date.
               Return measurement only touches data AFTER screen_date.
               These two downloads never overlap.

        ✓  Realistic execution
               Entry/exit prices include slippage.
               Brokerage costs deducted on every trade.

        ✓  Flexible position sizing
               equal      → 1/N per stock (simple, robust)
               volatility → inverse-vol weighting (riskier stocks get less)

        ✓  Quant-standard performance metrics
               Sharpe, Sortino, Calmar, Max Drawdown, CAGR,
               Win Rate, Beat Rate, Avg Alpha, W/L Ratio.

        ✓  Regime analysis
               Performance in Bull / Flat / Bear markets.
               Tells you if the strategy only works in good markets.

        ✓  Factor attribution
               MA signal vs RSI signal contribution to winning/losing picks.
               Tells you which part of the signal actually adds value.
    """

    def __init__(
        self,
        tickers:         list,
        backtest_months: int   = 12,
        lookback_days:   int   = 400,
        top_n:           int   = 3,
        holding_days:    int   = 30,
        start_year:      int   = 2024,
        start_month:     int   = 1,
        sizing:          str   = "equal",    # "equal" or "volatility"
        cost_pct:        float = 0.10,       # % per leg
        slippage_pct:    float = 0.05,       # % per leg
        max_alloc:       float = 0.40,       # max single-stock weight
    ):
        """
        Args:
            tickers         : List of ticker strings (e.g. ["RELIANCE.NS", ...])
            backtest_months : How many months to run the backtest.
            lookback_days   : Days of historical data per screen cycle.
                              Use >= 400 to ensure MA200 is fully formed.
            top_n           : Stocks to pick per month (1–10).
            holding_days    : Calendar days to hold before next rebalance.
            start_year      : Year to start the first screen.
            start_month     : Month to start (1 = January).
            sizing          : "equal" or "volatility".
            cost_pct        : Brokerage cost per leg in %.
            slippage_pct    : Slippage per leg in %.
            max_alloc       : Max fraction in any single stock (vol sizing).
        """
        # Validate inputs
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if backtest_months < 1:
            raise ValueError("backtest_months must be >= 1")
        if lookback_days < 200:
            logger.warning("lookback_days=%d is low — MA200 may not form correctly. "
                           "Recommend >= 400.", lookback_days)
        if sizing not in ("equal", "volatility"):
            raise ValueError("sizing must be 'equal' or 'volatility'")
        if not (0 < start_month <= 12):
            raise ValueError("start_month must be 1–12")

        self.tickers         = tickers
        self.backtest_months = backtest_months
        self.lookback_days   = lookback_days
        self.top_n           = top_n
        self.holding_days    = holding_days
        self.start_year      = start_year
        self.start_month     = start_month
        self.sizing          = sizing
        self.max_alloc       = max_alloc

        self.cost_model      = CostModel(cost_pct, slippage_pct)
        self.monthly_results = []   # one dict per month
        self.all_picks       = []   # one dict per stock per month

        self._print_init_banner(cost_pct, slippage_pct)

    # ─────────────────────────────────────────────────────────────── #
    #  Step 1: Screen on date (NO look-ahead)                         #
    # ─────────────────────────────────────────────────────────────── #

    def _screen_on_date(self, screen_date: datetime) -> list:
        """
        Run the screener on a historical date.

        Data available to screener: (screen_date - lookback_days) → screen_date
        Future data:                completely hidden from screener ✓

        Uses:
            StockScreener  (screener.py)
            StockScorer    (scoring.py) — to decompose MA and RSI signals

        Returns:
            list of dicts, each representing one stock pick:
            {
                ticker, score, raw_price, signal, bullish,
                ma50, rsi14, ma_signal, rsi_signal
            }
        """
        logger.info("  Screening on %s...", screen_date.strftime("%Y-%m-%d"))

        screener = StockScreener(
            tickers=self.tickers,
            lookback_days=self.lookback_days,
            screen_date=screen_date,
        )
        screener.download_data(max_workers=6)
        screener.calculate_indicators()
        screener.generate_signals()

        if screener.results is None or screener.results.empty:
            logger.warning("  Screener returned no results for %s",
                           screen_date.strftime("%b %Y"))
            return []

        top   = screener.results.head(self.top_n)
        picks = []
        scorer = StockScorer()  # for decomposing signal contributions

        for _, row in top.iterrows():
            # Pull values safely (MA200 can be None/NaN if data is short)
            price = float(row["Price"])
            ma50  = float(row["MA50"])
            rsi14 = float(row["RSI14"])

            ma200 = None
            try:
                v = row["MA200"]
                if v is not None and not pd.isna(v):
                    ma200 = float(v)
            except (KeyError, TypeError, ValueError):
                pass

            # Decompose score into its two parts for factor attribution
            ma_sig  = scorer.get_ma_signal(price, ma50, ma200)
            rsi_sig = scorer.get_rsi_signal(rsi14)

            picks.append({
                "ticker":     str(row["Ticker"]),
                "score":      float(row["Combined_Score"]),
                "raw_price":  price,
                "signal":     str(row["Signal"]),
                "bullish":    bool(row["Bullish"]),
                "ma50":       ma50,
                "ma200":      ma200 or 0.0,
                "rsi14":      rsi14,
                "ma_signal":  float(ma_sig),
                "rsi_signal": float(rsi_sig),
            })

        logger.info("  Top %d picks: %s", len(picks),
                    [p["ticker"] for p in picks])
        return picks

    # ─────────────────────────────────────────────────────────────── #
    #  Step 2: Fetch forward prices (ONLY after screen_date)          #
    # ─────────────────────────────────────────────────────────────── #

    def _fetch_forward_prices(
        self,
        ticker:      str,
        screen_date: datetime,
        exit_date:   datetime,
    ) -> tuple:
        """
        Download close prices in the holding period.
        This is the ONLY place where future data is used,
        and it is ONLY for measuring returns — never for signal generation.

        Applies slippage via CostModel:
            entry = first close × (1 + slippage)   ← you pay more to buy
            exit  = last  close × (1 − slippage)   ← you receive less on sell

        Returns:
            (entry_price, exit_price, days_held)
            or (None, None, 0) on failure
        """
        try:
            df = yf.download(
                ticker,
                start=screen_date.strftime("%Y-%m-%d"),
                end=exit_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )

            if df is None or df.empty:
                logger.warning("    %s: no forward data returned", ticker)
                return None, None, 0

            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()

            if len(close) < 2:
                logger.warning("    %s: only %d bars in forward window",
                               ticker, len(close))
                return None, None, 0

            raw_entry = float(close.iloc[0])
            raw_exit  = float(close.iloc[-1])

            if raw_entry <= 0 or raw_exit <= 0:
                logger.warning("    %s: zero price detected", ticker)
                return None, None, 0

            # Apply slippage
            entry     = self.cost_model.entry_price(raw_entry)
            exit_     = self.cost_model.exit_price(raw_exit)
            days_held = (close.index[-1] - close.index[0]).days

            return entry, exit_, days_held

        except Exception as exc:
            logger.warning("    %s: fetch error — %s", ticker, str(exc)[:80])
            return None, None, 0

    # ─────────────────────────────────────────────────────────────── #
    #  Step 3: Position sizing                                         #
    # ─────────────────────────────────────────────────────────────── #

    def _build_weights(self, picks: list, screen_date: datetime) -> dict:
        """
        Assign portfolio weights to each picked stock.

        equal (default):
            Each stock gets 1/N of capital.
            Simple and surprisingly hard to beat.
            → Use this when you have < 5 stocks or high uncertainty.

        volatility (inverse-vol weighting):
            Low-volatility stocks get MORE weight.
            High-volatility stocks get LESS weight.
            Why: a volatile stock in an equal-weight portfolio
                 dominates the portfolio's total risk.
            This makes the RISK roughly equal, not the CAPITAL.
            Uses recent 60-day volatility via VolatilityPortfolio.

        In both cases weights are:
            ≥ 0.0   (no shorting)
            ≤ max_alloc  (no single stock can dominate)
            sum to 1.0  (fully invested)

        Returns:
            dict  {ticker: weight}
        """
        tickers = [p["ticker"] for p in picks]
        n       = len(tickers)

        if n == 0:
            return {}

        if self.sizing == "equal":
            return {t: 1.0 / n for t in tickers}

        # ── Volatility-based weighting ────────────────────────────
        inv_vols: dict = {}

        for p in picks:
            try:
                # Use data BEFORE screen_date to avoid look-ahead
                end   = screen_date
                start = end - timedelta(days=90)
                df    = yf.download(
                    p["ticker"],
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty or len(df) < 15:
                    inv_vols[p["ticker"]] = 1.0   # fallback weight
                    continue

                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                # Annualised volatility
                vol = float(close.pct_change().dropna().std() * np.sqrt(252))
                vol = max(vol, 0.01)    # avoid division by zero

                # Kelly fraction based on score edge
                # score is in [-1, +1]; scale to small edge estimate
                edge       = float(p["score"]) * 0.05
                kelly_raw  = edge / (vol ** 2)
                kelly      = min(max(kelly_raw, 0.0), 0.25)

                # Inverse-vol + Kelly combined weight numerator
                inv_vols[p["ticker"]] = (1.0 / vol) * (1.0 + kelly)

            except Exception as exc:
                logger.warning("    Vol sizing failed for %s: %s",
                               p["ticker"], str(exc)[:60])
                inv_vols[p["ticker"]] = 1.0

        if not inv_vols:
            return {t: 1.0 / n for t in tickers}

        total = sum(inv_vols.values())
        weights = {t: iv / total for t, iv in inv_vols.items()}

        # Cap at max_alloc
        weights = {t: min(w, self.max_alloc) for t, w in weights.items()}

        # Re-normalise after capping
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {t: w / total_w for t, w in weights.items()}

        return weights

    # ─────────────────────────────────────────────────────────────── #
    #  Step 4: Measure returns for one month                          #
    # ─────────────────────────────────────────────────────────────── #

    def _measure_returns(
        self,
        picks:       list,
        screen_date: datetime,
    ) -> tuple:
        """
        Compute the portfolio's net return for the holding period.

        Process for each stock:
            raw_entry = first close after screen_date
            entry     = raw_entry × (1 + slippage)          ← you pay this
            raw_exit  = last close before exit_date
            exit_     = raw_exit  × (1 − slippage)          ← you receive this
            gross_ret = (exit_ − entry) / entry × 100
            cost_pct  = round_trip brokerage cost            ← deducted
            net_ret   = gross_ret − cost_pct

        Portfolio return = weighted average of all net_ret values.

        Also downloads Nifty 50 for the same period (benchmark).

        Returns:
            (portfolio_return_pct, nifty_return_pct, list_of_trade_dicts)
        """
        exit_date = screen_date + timedelta(days=self.holding_days)
        weights   = self._build_weights(picks, screen_date)
        trades    = []

        for p in picks:
            ticker = p["ticker"]
            weight = weights.get(ticker, 1.0 / len(picks))

            entry, exit_, days_held = self._fetch_forward_prices(
                ticker, screen_date, exit_date
            )
            if entry is None:
                continue

            # Gross return before costs
            gross_ret_pct = (exit_ - entry) / entry * 100.0

            # Deduct round-trip transaction cost
            rt_cost_pct   = self.cost_model.round_trip_cost_pct()
            net_ret_pct   = gross_ret_pct - rt_cost_pct

            trades.append({
                "Ticker":         ticker,
                "Score":          round(p["score"],      3),
                "Signal":         p["signal"],
                "Bullish":        p["bullish"],
                "MA_Signal":      round(p["ma_signal"],  2),
                "RSI_Signal":     round(p["rsi_signal"], 2),
                "MA50":           round(p["ma50"],        2),
                "MA200":          round(p["ma200"],       2),
                "RSI14":          round(p["rsi14"],       2),
                "Entry_Price":    round(entry,            2),
                "Exit_Price":     round(exit_,            2),
                "Days_Held":      days_held,
                "Weight_%":       round(weight * 100,     2),
                "Gross_Return_%": round(gross_ret_pct,    2),
                "Cost_%":         round(rt_cost_pct,      4),
                "Net_Return_%":   round(net_ret_pct,      2),
            })

            logger.info(
                "    %-16s  gross=%+6.2f%%  cost=%.3f%%  net=%+6.2f%%  w=%.1f%%",
                ticker, gross_ret_pct, rt_cost_pct, net_ret_pct, weight * 100
            )

        if not trades:
            return 0.0, 0.0, []

        # ── Weighted portfolio return ─────────────────────────────
        # Re-compute actual weights from what succeeded
        actual_weights  = {t["Ticker"]: weights.get(t["Ticker"], 0.0)
                           for t in trades}
        total_weight    = sum(actual_weights.values())

        if total_weight < 1e-9:
            # Fallback: equal weight
            total_weight = len(trades)
            actual_weights = {t["Ticker"]: 1.0 for t in trades}

        portfolio_ret = sum(
            t["Net_Return_%"] * actual_weights[t["Ticker"]]
            for t in trades
        ) / total_weight

        # ── Benchmark ─────────────────────────────────────────────
        nifty_ret = self._get_benchmark_return(screen_date, exit_date)

        return round(portfolio_ret, 4), round(nifty_ret, 4), trades

    def _get_benchmark_return(self, start: datetime, end: datetime) -> float:
        """
        Nifty 50 (^NSEI) return for the same holding period.
        Used purely for comparison — never for signal generation.
        """
        try:
            df = yf.download(
                "^NSEI",
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
            if df is None or len(df) < 2:
                return 0.0

            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()

            if len(close) < 2:
                return 0.0

            return round(
                float((close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100), 4
            )
        except Exception as exc:
            logger.warning("  Nifty fetch failed: %s", str(exc)[:60])
            return 0.0

    # ─────────────────────────────────────────────────────────────── #
    #  Metrics computation                                             #
    # ─────────────────────────────────────────────────────────────── #

    def _compute_metrics(self, results_df: pd.DataFrame) -> dict:
        """
        Compute all standard quant performance metrics.

        RETURNS:
            Cumulative return  = compounded total over the period
            CAGR               = annualised version of cumulative return

        RISK:
            Sharpe  = mean_return / std_return
                      Higher = better return per unit of total risk.
                      > 1.0 is generally considered good.

            Sortino = mean_return / downside_std
                      Only penalises losing months, not winning months.
                      More relevant than Sharpe for non-normal returns.
                      > 1.5 is considered good.

            Calmar  = CAGR / |max_drawdown|
                      Return per unit of worst-case pain.
                      > 0.5 is generally decent.

            Max DD  = largest peak-to-trough decline on equity curve.
                      The worst you would have felt if unlucky timing.

        WIN/LOSS:
            Win rate  = % of months with positive portfolio return
            Beat rate = % of months where portfolio beat Nifty
            W/L ratio = avg_win / |avg_loss|
                        > 1.0 means wins are larger than losses on average.

        Returns dict with all metrics + equity curve + drawdown series.
        """
        port  = results_df["Portfolio_Return_%"]
        nifty = results_df["Nifty_Return_%"]
        alpha = results_df["Outperformance_%"]
        n     = len(results_df)

        # ── Cumulative returns ─────────────────────────────────────
        cum_port  = float(((1 + port  / 100).prod() - 1) * 100)
        cum_nifty = float(((1 + nifty / 100).prod() - 1) * 100)

        # ── CAGR ──────────────────────────────────────────────────
        years = n / 12.0
        if years > 0 and cum_port > -100:
            cagr = float(((1 + cum_port / 100) ** (1.0 / years) - 1) * 100)
        else:
            cagr = 0.0

        # ── Sharpe (no risk-free rate — simplification for learning) ──
        std_port = float(port.std())
        sharpe   = float(port.mean() / std_port) if std_port > 1e-9 else 0.0

        # ── Sortino ───────────────────────────────────────────────
        downside = port[port < 0]
        std_down = float(downside.std()) if len(downside) > 1 else 0.0
        sortino  = float(port.mean() / std_down) if std_down > 1e-9 else 0.0

        # ── Equity curve and drawdown ──────────────────────────────
        equity    = (1 + port / 100).cumprod()
        peak      = equity.cummax()
        dd_series = (equity - peak) / peak * 100      # always <= 0
        max_dd    = float(dd_series.min())

        # Recovery factor: total return / |max drawdown|
        recovery  = abs(cum_port / max_dd) if max_dd < -1e-9 else 0.0

        # ── Calmar ────────────────────────────────────────────────
        calmar = abs(cagr / max_dd) if max_dd < -1e-9 else 0.0

        # ── Win / Loss ────────────────────────────────────────────
        win_mask    = port > 0
        loss_mask   = port < 0
        win_months  = int(win_mask.sum())
        loss_months = int(loss_mask.sum())
        win_rate    = float(win_months / n * 100) if n > 0 else 0.0
        beat_months = int((alpha > 0).sum())
        beat_rate   = float(beat_months / n * 100) if n > 0 else 0.0

        avg_win  = float(port[win_mask].mean())  if win_months  > 0 else 0.0
        avg_loss = float(port[loss_mask].mean()) if loss_months > 0 else 0.0
        wl_ratio = abs(avg_win / avg_loss)        if avg_loss != 0  else 0.0

        # ── Consecutive stats ─────────────────────────────────────
        max_consec_win  = _max_consecutive(port > 0)
        max_consec_loss = _max_consecutive(port <= 0)

        return {
            # Core
            "n_months":         n,
            "cum_port":         round(cum_port,   2),
            "cum_nifty":        round(cum_nifty,  2),
            "total_alpha":      round(cum_port - cum_nifty, 2),
            "cagr":             round(cagr,        2),
            # Risk
            "sharpe":           round(sharpe,      3),
            "sortino":          round(sortino,      3),
            "calmar":           round(calmar,       3),
            "recovery":         round(recovery,     3),
            "max_dd":           round(max_dd,       2),
            # Win/Loss
            "win_months":       win_months,
            "loss_months":      loss_months,
            "win_rate":         round(win_rate,     1),
            "beat_months":      beat_months,
            "beat_rate":        round(beat_rate,    1),
            "avg_win":          round(avg_win,      2),
            "avg_loss":         round(avg_loss,     2),
            "wl_ratio":         round(wl_ratio,     2),
            "avg_alpha":        round(float(alpha.mean()), 2),
            "max_alpha":        round(float(alpha.max()),  2),
            "min_alpha":        round(float(alpha.min()),  2),
            "max_consec_win":   max_consec_win,
            "max_consec_loss":  max_consec_loss,
            # Series (used for plotting — NOT saved to metrics CSV)
            "equity":           equity,
            "dd_series":        dd_series,
        }

    # ─────────────────────────────────────────────────────────────── #
    #  Regime analysis                                                 #
    # ─────────────────────────────────────────────────────────────── #

    def _regime_breakdown(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Split performance by market regime.

        Regime is defined by Nifty's monthly return:
            Bull : Nifty > +2%   (rising market)
            Bear : Nifty < -2%   (falling market)
            Flat : -2% to +2%    (sideways market)

        Key question this answers:
            "Does my strategy only work when the market is going up?"
            If Beat_Rate in Bear regime is high → strategy is defensive. ✅
            If Beat_Rate in Bear regime is low  → strategy is market-dependent. ⚠️
        """
        df = results_df.copy()

        def classify(n_ret: float) -> str:
            if   n_ret >  2.0: return "Bull"
            elif n_ret < -2.0: return "Bear"
            else:              return "Flat"

        df["Regime"] = df["Nifty_Return_%"].apply(classify)

        summary = (
            df.groupby("Regime")
            .agg(
                Months        = ("Portfolio_Return_%", "count"),
                Avg_Portfolio = ("Portfolio_Return_%", "mean"),
                Avg_Nifty     = ("Nifty_Return_%",     "mean"),
                Avg_Alpha     = ("Outperformance_%",   "mean"),
                Win_Rate_Pct  = ("Portfolio_Return_%",
                                 lambda x: round((x > 0).mean() * 100, 1)),
                Beat_Rate_Pct = ("Outperformance_%",
                                 lambda x: round((x > 0).mean() * 100, 1)),
            )
            .round(2)
        )

        # Make sure all three regimes always appear (even if zero months)
        for reg in ["Bull", "Flat", "Bear"]:
            if reg not in summary.index:
                summary.loc[reg] = [0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return summary.loc[["Bull", "Flat", "Bear"]]

    # ─────────────────────────────────────────────────────────────── #
    #  Factor attribution                                              #
    # ─────────────────────────────────────────────────────────────── #

    def _factor_attribution(self, picks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Decompose pick quality into MA signal vs RSI signal contributions.

        For winning picks (positive return):
            → Are MA_Signal values high? RSI_Signal values high?

        For losing picks (negative return):
            → Which signal was misleading?

        If:
            Winning picks have high MA_Signal but low RSI_Signal
            → The trend (MA) component is the driver of alpha.
            → RSI is adding noise, not value.

        This helps you decide if you should re-weight MA vs RSI in scoring.py
        """
        if picks_df.empty:
            return pd.DataFrame()

        needed_cols = {"Net_Return_%", "MA_Signal", "RSI_Signal", "Score"}
        if not needed_cols.issubset(picks_df.columns):
            return pd.DataFrame()

        df = picks_df.copy()
        df["Outcome"] = df["Net_Return_%"].apply(
            lambda x: "Win ✅" if x > 0 else "Loss ❌"
        )

        summary = (
            df.groupby("Outcome")
            .agg(
                Count       = ("Net_Return_%",  "count"),
                Avg_Return  = ("Net_Return_%",  "mean"),
                Avg_Score   = ("Score",         "mean"),
                Avg_MA_Sig  = ("MA_Signal",     "mean"),
                Avg_RSI_Sig = ("RSI_Signal",    "mean"),
                Pct_Bullish = ("Bullish",
                               lambda x: round(x.mean() * 100, 1)),
            )
            .round(3)
        )

        return summary

    # ─────────────────────────────────────────────────────────────── #
    #  Main loop                                                       #
    # ─────────────────────────────────────────────────────────────── #

    def run(self) -> pd.DataFrame:
        """
        Execute the full walk-forward backtest.

        Iterates month by month from start_year/start_month.
        Each iteration is completely independent — no shared state.

        Returns:
            pd.DataFrame of monthly results (also saved to CSV).
        """
        start = datetime(self.start_year, self.start_month, 15)

        self._print_run_header(start)

        for i in range(self.backtest_months):
            screen_date = start + relativedelta(months=i)
            month_str   = screen_date.strftime("%b %Y")

            print(f"\n{'─' * 70}")
            print(
                f"  Month {i+1:2d} / {self.backtest_months}  "
                f"│  Screen date: {screen_date.strftime('%Y-%m-%d')}  "
                f"│  {month_str}"
            )
            print(f"{'─' * 70}")

            # ── Screen ───────────────────────────────────────────
            picks = self._screen_on_date(screen_date)
            if not picks:
                logger.warning("  Skipping %s — screener returned no picks", month_str)
                continue

            # ── Measure returns ───────────────────────────────────
            port_ret, nifty_ret, trades = self._measure_returns(picks, screen_date)
            if not trades:
                logger.warning("  Skipping %s — no valid forward data", month_str)
                continue

            alpha   = round(port_ret - nifty_ret, 4)
            is_beat = alpha > 0

            self.monthly_results.append({
                "Month":              month_str,
                "Screen_Date":        screen_date.strftime("%Y-%m-%d"),
                "Num_Stocks":         len(trades),
                "Portfolio_Return_%": round(port_ret, 2),
                "Nifty_Return_%":     round(nifty_ret, 2),
                "Outperformance_%":   round(alpha,     2),
                "Sizing_Method":      self.sizing,
            })

            for t in trades:
                t["Month"]       = month_str
                t["Screen_Date"] = screen_date.strftime("%Y-%m-%d")
                self.all_picks.append(t)

            # ── Print month result ────────────────────────────────
            beat_str = "✅ Beat" if is_beat else "❌ Miss"
            print(
                f"\n  Portfolio : {port_ret:>+7.2f}%  │  "
                f"Nifty : {nifty_ret:>+7.2f}%  │  "
                f"Alpha : {alpha:>+7.2f}%  │  {beat_str}"
            )
            print(f"  Picks     : {[t['Ticker'] for t in trades]}")

            # Per-stock detail
            for t in trades:
                print(
                    f"    {t['Ticker']:<18}  "
                    f"score={t['Score']:+.2f}  "
                    f"net={t['Net_Return_%']:>+6.2f}%  "
                    f"w={t['Weight_%']:.1f}%  "
                    f"entry={t['Entry_Price']:.2f}  "
                    f"exit={t['Exit_Price']:.2f}"
                )

        return self._finalize()

    # ─────────────────────────────────────────────────────────────── #
    #  Finalize                                                        #
    # ─────────────────────────────────────────────────────────────── #

    def _finalize(self) -> pd.DataFrame:
        """Compute metrics, print summary, save files, generate plots."""
        results_df = pd.DataFrame(self.monthly_results)
        picks_df   = pd.DataFrame(self.all_picks)

        if results_df.empty:
            print("\n" + "═" * 70)
            print("  ⚠️   No results generated.")
            print("  Check: date range, internet connection, ticker availability.")
            print("═" * 70)
            return results_df

        metrics = self._compute_metrics(results_df)
        regime  = self._regime_breakdown(results_df)
        factors = self._factor_attribution(picks_df) if not picks_df.empty else pd.DataFrame()

        self._print_summary(metrics, regime, factors, picks_df)
        self._save_results(results_df, picks_df, metrics)
        self._plot_results(results_df, picks_df, metrics)

        return results_df

    # ─────────────────────────────────────────────────────────────── #
    #  Save                                                            #
    # ─────────────────────────────────────────────────────────────── #

    def _save_results(
        self,
        results_df: pd.DataFrame,
        picks_df:   pd.DataFrame,
        metrics:    dict,
    ) -> None:
        """Save all results to CSV files."""
        results_df.to_csv("backtest_results.csv", index=False)
        picks_df.to_csv("backtest_picks.csv",     index=False)

        # Strip Series objects before saving to metrics CSV
        metrics_flat = {k: v for k, v in metrics.items()
                        if not isinstance(v, (pd.Series, pd.DataFrame, np.ndarray))}
        pd.DataFrame([metrics_flat]).to_csv("backtest_metrics.csv", index=False)

        print(f"\n  💾  backtest_results.csv  ({len(results_df)} months)")
        print(f"  💾  backtest_picks.csv    ({len(picks_df)} trades)")
        print(f"  💾  backtest_metrics.csv  (summary metrics)")

    # ─────────────────────────────────────────────────────────────── #
    #  Plot                                                            #
    # ─────────────────────────────────────────────────────────────── #

    def _plot_results(
        self,
        results_df: pd.DataFrame,
        picks_df:   pd.DataFrame,
        metrics:    dict,
    ) -> None:
        """
        Generate two plot files:

        backtest_equity.png:
            Top:    Equity curve (portfolio vs Nifty)
            Bottom: Drawdown underwater chart

        backtest_summary.png:
            Left:   Monthly alpha bars (green = beat, red = missed)
            Right:  Stock selection frequency (most-used tickers)
        """
        # Safe style selection
        for style in ["seaborn-v0_8-darkgrid", "seaborn-darkgrid", "ggplot"]:
            try:
                plt.style.use(style)
                break
            except OSError:
                continue

        months = list(range(1, len(results_df) + 1))
        labels = results_df["Month"].tolist()

        port  = results_df["Portfolio_Return_%"].values / 100.0
        nifty = results_df["Nifty_Return_%"].values    / 100.0
        port_cum  = (1 + port ).cumprod() - 1
        nifty_cum = (1 + nifty).cumprod() - 1

        # ── Plot 1: Equity Curve + Drawdown ───────────────────────
        fig  = plt.figure(figsize=(14, 10))
        gs   = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax1  = fig.add_subplot(gs[0])
        ax2  = fig.add_subplot(gs[1], sharex=ax1)

        # Equity curve
        ax1.plot(months, port_cum  * 100,
                 label="Portfolio", lw=2.5, marker="o", ms=5, color="#1565C0")
        ax1.plot(months, nifty_cum * 100,
                 label="Nifty 50",  lw=2.5, marker="s", ms=5,
                 color="#E65100", alpha=0.85)
        ax1.fill_between(months,
                         port_cum * 100, nifty_cum * 100,
                         where=(port_cum >= nifty_cum),
                         alpha=0.15, color="#1565C0", label="_")
        ax1.fill_between(months,
                         port_cum * 100, nifty_cum * 100,
                         where=(port_cum < nifty_cum),
                         alpha=0.15, color="#E65100", label="_")
        ax1.axhline(0, color="black", lw=1.0, ls="--", alpha=0.4)

        # Annotate final values
        if len(months) > 0:
            ax1.annotate(
                f"Portfolio: {port_cum[-1]*100:+.1f}%",
                xy=(months[-1], port_cum[-1] * 100),
                xytext=(months[-1] - 0.5, port_cum[-1] * 100 + 1.5),
                fontsize=9, color="#1565C0", fontweight="bold",
            )
            ax1.annotate(
                f"Nifty: {nifty_cum[-1]*100:+.1f}%",
                xy=(months[-1], nifty_cum[-1] * 100),
                xytext=(months[-1] - 0.5, nifty_cum[-1] * 100 - 2.5),
                fontsize=9, color="#E65100", fontweight="bold",
            )

        ax1.set_ylabel("Cumulative Return (%)", fontsize=11, fontweight="bold")
        ax1.set_title(
            f"Walk-Forward Backtest  │  {len(results_df)} Months  │  "
            f"Top {self.top_n}  │  {self.sizing.title()} Weight  │  "
            f"Sharpe {metrics['sharpe']:.2f}  │  Max DD {metrics['max_dd']:.1f}%",
            fontsize=12, fontweight="bold", pad=12
        )
        ax1.legend(fontsize=11, loc="upper left")
        ax1.set_xlim(0.5, len(months) + 0.5)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # Drawdown
        dd_vals = metrics["dd_series"].values
        ax2.fill_between(months, dd_vals, 0, color="#C62828", alpha=0.35)
        ax2.plot(months, dd_vals, color="#B71C1C", lw=1.5)
        ax2.axhline(
            metrics["max_dd"], color="#B71C1C", ls="--", lw=1.2,
            label=f"Max DD: {metrics['max_dd']:.1f}%"
        )
        ax2.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Month", fontsize=11, fontweight="bold")
        ax2.set_xticks(months)
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax2.legend(fontsize=9, loc="lower right")

        plt.tight_layout()
        fig.savefig("backtest_equity.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  📊  backtest_equity.png")

        # ── Plot 2: Alpha bars + Stock frequency ───────────────────
        has_picks = not picks_df.empty and "Ticker" in picks_df.columns
        fig2, axes = plt.subplots(1, 2 if has_picks else 1, figsize=(14, 6))
        if not has_picks:
            axes = [axes]

        # Monthly alpha bars
        ax = axes[0]
        alpha_vals = results_df["Outperformance_%"].values
        bar_colors = ["#388E3C" if a > 0 else "#C62828" for a in alpha_vals]
        bars = ax.bar(months, alpha_vals, color=bar_colors,
                      alpha=0.85, edgecolor="black", lw=0.5)
        ax.axhline(0, color="black", lw=1.2)
        avg_a = float(np.mean(alpha_vals))
        ax.axhline(avg_a, color="#1565C0", ls="--", lw=2.0,
                   label=f"Avg Alpha: {avg_a:+.2f}%")

        # Value labels on bars
        for bar, val in zip(bars, alpha_vals):
            y_pos = val + 0.1 if val >= 0 else val - 0.3
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:+.1f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold")

        ax.set_xticks(months)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Alpha (%)", fontsize=11, fontweight="bold")
        ax.set_title("Monthly Alpha  (Portfolio − Nifty)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)

        # Stock frequency
        if has_picks:
            ax2b = axes[1]
            freq  = picks_df["Ticker"].value_counts().head(12)
            cmap  = plt.cm.viridis(np.linspace(0.25, 0.85, len(freq)))
            y_pos = range(len(freq))

            ax2b.barh([t.replace(".NS", "") for t in freq.index[::-1]],
                      freq.values[::-1],
                      color=cmap, edgecolor="black", lw=0.5)

            for i, v in enumerate(freq.values[::-1]):
                ax2b.text(v + 0.05, i, str(v),
                          va="center", fontsize=9, fontweight="bold")

            ax2b.set_xlabel("Times Picked", fontsize=11, fontweight="bold")
            ax2b.set_title("Most Frequently Selected Stocks",
                           fontsize=12, fontweight="bold")

        plt.tight_layout()
        fig2.savefig("backtest_summary.png", dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print("  📊  backtest_summary.png")

    # ─────────────────────────────────────────────────────────────── #
    #  Print helpers                                                   #
    # ─────────────────────────────────────────────────────────────── #

    def _print_init_banner(self, cost_pct: float, slippage_pct: float) -> None:
        W = 70
        print("\n" + "═" * W)
        print("  QUANTITATIVE WALK-FORWARD BACKTESTER".center(W))
        print("═" * W)
        print(f"  {'Tickers':<22} {len(self.tickers)}")
        print(f"  {'Months to test':<22} {self.backtest_months}")
        print(f"  {'Top N picks/month':<22} {self.top_n}")
        print(f"  {'Holding period':<22} {self.holding_days} calendar days")
        print(f"  {'Lookback window':<22} {self.lookback_days} calendar days")
        print(f"  {'Position sizing':<22} {self.sizing}")
        print(f"  {'Transaction cost':<22} {cost_pct:.3f}% per leg  "
              f"({cost_pct*2:.3f}% round-trip)")
        print(f"  {'Slippage':<22} {slippage_pct:.3f}% per leg")
        if self.sizing == "volatility":
            print(f"  {'Max allocation':<22} {self.max_alloc*100:.0f}% per stock")
        print("═" * W)

    def _print_run_header(self, start: datetime) -> None:
        print(f"\n  Start     : {start.strftime('%d %b %Y')}")
        print(f"  Universe  : {len(self.tickers)} tickers")
        print(f"  Benchmark : Nifty 50  (^NSEI)")

    def _print_summary(
        self,
        m:        dict,
        regime:   pd.DataFrame,
        factors:  pd.DataFrame,
        picks_df: pd.DataFrame,
    ) -> None:
        """Print comprehensive performance summary to terminal."""
        W = 70

        print("\n\n" + "═" * W)
        print("  BACKTEST PERFORMANCE SUMMARY".center(W, "═"))
        print("═" * W)

        # ── Returns ───────────────────────────────────────────────
        print(f"\n  ── RETURNS {'─'*(W-13)}")
        print(f"  {'Months tested':<32} {m['n_months']}")
        print(f"  {'Portfolio cumulative return':<32} {m['cum_port']:>+8.2f}%")
        print(f"  {'Nifty 50 cumulative return':<32} {m['cum_nifty']:>+8.2f}%")
        print(f"  {'Total alpha generated':<32} {m['total_alpha']:>+8.2f}%")
        print(f"  {'CAGR (annualised)':<32} {m['cagr']:>+8.2f}%")

        # ── Risk ──────────────────────────────────────────────────
        print(f"\n  ── RISK METRICS {'─'*(W-18)}")
        print(f"  {'Sharpe ratio':<32} {m['sharpe']:>8.3f}"
              + ("  ✅" if m['sharpe'] > 1.0 else "  ⚠️"))
        print(f"  {'Sortino ratio':<32} {m['sortino']:>8.3f}"
              + ("  ✅" if m['sortino'] > 1.5 else "  ⚠️"))
        print(f"  {'Calmar ratio':<32} {m['calmar']:>8.3f}"
              + ("  ✅" if m['calmar'] > 0.5 else "  ⚠️"))
        print(f"  {'Recovery factor':<32} {m['recovery']:>8.3f}")
        print(f"  {'Max drawdown':<32} {m['max_dd']:>8.2f}%")

        # ── Win/Loss ──────────────────────────────────────────────
        print(f"\n  ── WIN / LOSS {'─'*(W-16)}")
        print(f"  {'Winning months':<32} "
              f"{m['win_months']:>3} / {m['n_months']}  ({m['win_rate']:.1f}%)")
        print(f"  {'Beat Nifty months':<32} "
              f"{m['beat_months']:>3} / {m['n_months']}  ({m['beat_rate']:.1f}%)")
        print(f"  {'Average winning month':<32} {m['avg_win']:>+8.2f}%")
        print(f"  {'Average losing month':<32} {m['avg_loss']:>+8.2f}%")
        print(f"  {'Win / Loss ratio':<32} {m['wl_ratio']:>8.2f}x"
              + ("  ✅" if m['wl_ratio'] > 1.0 else "  ⚠️"))
        print(f"  {'Average monthly alpha':<32} {m['avg_alpha']:>+8.2f}%")
        print(f"  {'Best alpha month':<32} {m['max_alpha']:>+8.2f}%")
        print(f"  {'Worst alpha month':<32} {m['min_alpha']:>+8.2f}%")
        print(f"  {'Max consecutive wins':<32} {m['max_consec_win']:>8d}")
        print(f"  {'Max consecutive losses':<32} {m['max_consec_loss']:>8d}")

        # ── Stock selection ────────────────────────────────────────
        if not picks_df.empty:
            print(f"\n  ── STOCK SELECTION {'─'*(W-21)}")
            print(f"  {'Total trades executed':<32} {len(picks_df)}")
            print(f"  {'Unique stocks used':<32} {picks_df['Ticker'].nunique()}")
            if "Net_Return_%" in picks_df.columns:
                st_win = float((picks_df["Net_Return_%"] > 0).mean() * 100)
                st_avg = float(picks_df["Net_Return_%"].mean())
                print(f"  {'Individual stock win rate':<32} {st_win:>7.1f}%")
                print(f"  {'Avg individual net return':<32} {st_avg:>+8.2f}%")
            if "Score" in picks_df.columns and "Net_Return_%" in picks_df.columns:
                corr = float(np.corrcoef(
                    picks_df["Score"].values,
                    picks_df["Net_Return_%"].values
                )[0, 1])
                print(f"  {'Score → Return correlation':<32} {corr:>+8.3f}"
                      + ("  ✅ signal works" if corr > 0.1 else "  ⚠️  weak signal"))

        # ── Regime breakdown ──────────────────────────────────────
        if not regime.empty:
            print(f"\n  ── REGIME BREAKDOWN {'─'*(W-22)}")
            header = (f"  {'Regime':<8} {'Months':>6} {'Port%':>8} "
                      f"{'Nifty%':>8} {'Alpha%':>8} {'Win%':>7} {'Beat%':>7}")
            print(header)
            print("  " + "─" * (len(header) - 2))
            for reg, row in regime.iterrows():
                mo = int(row["Months"])
                if mo == 0:
                    continue
                print(
                    f"  {reg:<8} {mo:>6} "
                    f"{row['Avg_Portfolio']:>+8.2f} "
                    f"{row['Avg_Nifty']:>+8.2f} "
                    f"{row['Avg_Alpha']:>+8.2f} "
                    f"{row['Win_Rate_Pct']:>6.1f}% "
                    f"{row['Beat_Rate_Pct']:>6.1f}%"
                )

        # ── Factor attribution ─────────────────────────────────────
        if not factors.empty:
            print(f"\n  ── FACTOR ATTRIBUTION (MA vs RSI) {'─'*(W-36)}")
            print(f"  {'Outcome':<12} {'Count':>6} {'Avg Ret':>9} "
                  f"{'Avg Score':>10} {'Avg MA':>8} {'Avg RSI':>9}")
            print("  " + "─" * 58)
            for outcome, row in factors.iterrows():
                print(
                    f"  {outcome:<12} {int(row['Count']):>6} "
                    f"{row['Avg_Return']:>+9.2f} "
                    f"{row['Avg_Score']:>+10.3f} "
                    f"{row['Avg_MA_Sig']:>+8.3f} "
                    f"{row['Avg_RSI_Sig']:>+9.3f}"
                )

        print("\n" + "═" * W + "\n")


# ═══════════════════════════════════════════════════════════════════════════ #
#  Helper functions                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def _max_consecutive(bool_series: pd.Series) -> int:
    """
    Find the maximum number of consecutive True values in a boolean Series.
    Used for: max_consec_win and max_consec_loss.

    Example:
        [True, True, False, True, True, True] → 3
    """
    max_count = 0
    count     = 0
    for val in bool_series:
        if val:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count


# ═══════════════════════════════════════════════════════════════════════════ #
#  Argument parser                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    All arguments are optional — defaults give a sensible quick run.
    """
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="Quantitative Walk-Forward Backtester — NIFTY50 Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔══════════════════════════════════════════════════════════════╗
║  EXAMPLES                                                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Quick run (defaults):                                       ║
║    python -m src.backtest                                    ║
║                                                              ║
║  12 months, top 5 stocks, volatility sizing:                 ║
║    python -m src.backtest --months 12 --top 5                ║
║                     --sizing volatility                      ║
║                                                              ║
║  Full NIFTY50 universe, start from Jan 2023:                 ║
║    python -m src.backtest --universe NIFTY50                 ║
║                     --year 2023 --month 1 --months 18        ║
║                                                              ║
║  Custom costs (aggressive broker):                           ║
║    python -m src.backtest --cost 0.15 --slippage 0.10        ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  SIZING METHODS                                              ║
║  equal      → 1/N per stock. Simple, robust. (default)       ║
║  volatility → Inverse-vol weight. Riskier stocks get less.   ║
╠══════════════════════════════════════════════════════════════╣
║  NOTES                                                       ║
║  • Use --lookback >= 400 for reliable MA200                  ║
║  • --hold 30 means hold ~1 month before rebalancing          ║
║  • Transaction costs are per leg (buy + sell = 2 × cost)     ║
╚══════════════════════════════════════════════════════════════╝
        """,
    )

    parser.add_argument(
        "--universe",
        choices=["TEST", "NIFTY50"],
        default="TEST",
        help=(
            "Stock universe to screen from. "
            "TEST=15 large-caps (default, faster). "
            "NIFTY50=all 15 tickers in stock_list.py."
        ),
    )
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        metavar="N",
        help="Number of months to backtest. (default: 12)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        metavar="N",
        help="Top N stocks to pick each month. (default: 3)",
    )
    parser.add_argument(
        "--hold",
        type=int,
        default=30,
        metavar="DAYS",
        help="Holding period in calendar days. (default: 30)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=400,
        metavar="DAYS",
        help=(
            "Historical data window for indicators in calendar days. "
            "Use >= 400 to ensure MA200 is fully formed. (default: 400)"
        ),
    )
    parser.add_argument(
        "--sizing",
        choices=["equal", "volatility"],
        default="equal",
        help=(
            "Position sizing method. "
            "equal=1/N per stock. "
            "volatility=inverse-vol weighting. "
            "(default: equal)"
        ),
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.10,
        metavar="PCT",
        help=(
            "Transaction cost per leg in %%. "
            "0.10 means 0.10%% one-way, 0.20%% round-trip. "
            "(default: 0.10)"
        ),
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.05,
        metavar="PCT",
        help=(
            "Slippage per leg in %%. "
            "0.05 means you buy 0.05%% higher and sell 0.05%% lower. "
            "(default: 0.05)"
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        metavar="YYYY",
        help="Start year for the backtest. (default: 2024)",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=1,
        metavar="M",
        help="Start month 1–12. (default: 1 = January)",
    )
    parser.add_argument(
        "--max-alloc",
        type=float,
        default=0.40,
        metavar="FRAC",
        help=(
            "Max fraction of portfolio in any single stock. "
            "Only used with --sizing volatility. "
            "(default: 0.40 = 40%%)"
        ),
    )

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════ #
#  Entry point                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    args = parse_args()

    # Validate month range
    if not (1 <= args.month <= 12):
        logger.error("--month must be between 1 and 12, got %d", args.month)
        sys.exit(1)

    # Validate cost and slippage
    if args.cost < 0 or args.cost > 5:
        logger.error("--cost must be between 0 and 5 (%%)")
        sys.exit(1)
    if args.slippage < 0 or args.slippage > 5:
        logger.error("--slippage must be between 0 and 5 (%%)")
        sys.exit(1)

    # Select ticker universe
    tickers = NIFTY_50_TICKERS if args.universe == "NIFTY50" else TEST_TICKERS

    bt = QuantBacktest(
        tickers         = tickers,
        backtest_months = args.months,
        lookback_days   = args.lookback,
        top_n           = args.top,
        holding_days    = args.hold,
        start_year      = args.year,
        start_month     = args.month,
        sizing          = args.sizing,
        cost_pct        = args.cost,
        slippage_pct    = args.slippage,
        max_alloc       = args.max_alloc,
    )

    bt.run()
