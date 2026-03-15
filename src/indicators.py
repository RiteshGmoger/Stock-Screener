"""
indicators.py — Technical Indicator Library

What's here:
    - calculate_moving_average()   SMA for any window (50, 200, etc.)
    - calculate_ema()              Exponential Moving Average
    - calculate_rsi()              Relative Strength Index (RSI-14)
    - get_ma_signal()              +1 / 0 / -1 based on price vs MA50
    - get_rsi_signal()             +1.0 / 0.0 / -0.5 based on RSI level
    - combine_signals()            Weighted score from MA + RSI signals

What changed from v1:
    - Added calculate_ema()
    - Added signal helper functions (get_ma_signal, get_rsi_signal, combine_signals)
      These were previously only in demo_indicators.py — now they live here properly.
    - calculate_moving_average() still works exactly the same — no breaking changes.
"""

import pandas as pd
import numpy as np


# ------------------------------------------------------------------ #
#  Moving Averages                                                     #
# ------------------------------------------------------------------ #

def calculate_moving_average(price_series: pd.Series, window: int = 50) -> pd.Series:
    """
    Simple Moving Average (SMA).

    Args:
        price_series : Daily close prices as a pandas Series.
        window       : Lookback window. Use 50 for MA50, 200 for MA200.

    Returns:
        pd.Series of SMA values. First (window-1) values will be NaN.

    Example:
        ma50  = calculate_moving_average(close, window=50)
        ma200 = calculate_moving_average(close, window=200)
    """
    return price_series.rolling(window=window).mean()


def calculate_ema(price_series: pd.Series, span: int = 12) -> pd.Series:
    """
    Exponential Moving Average (EMA).

    EMA gives more weight to recent prices than SMA.
    Used in MACD (EMA12 - EMA26).

    Args:
        price_series : Daily close prices.
        span         : EMA span. Common values: 12, 26 (for MACD).

    Returns:
        pd.Series of EMA values.

    Example:
        ema12 = calculate_ema(close, span=12)
        ema26 = calculate_ema(close, span=26)
        macd  = ema12 - ema26
    """
    return price_series.ewm(span=span, adjust=False).mean()


# ------------------------------------------------------------------ #
#  RSI                                                                 #
# ------------------------------------------------------------------ #

def calculate_rsi(price_series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    Formula:
        RS  = avg_gain / avg_loss  (over `period` days)
        RSI = 100 - (100 / (1 + RS))

    Interpretation:
        RSI > 70  → overbought (price ran up too fast)
        RSI < 30  → oversold   (price fell too fast)
        40–60     → neutral momentum

    Note: We use 40/60 as thresholds (not 30/70) because we want
          to catch momentum EARLY, not after it's extreme.

    Args:
        price_series : Daily close prices.
        period       : Lookback period. Standard = 14 days.

    Returns:
        pd.Series of RSI values (0–100). First `period` values = NaN.
    """
    delta    = price_series.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs       = avg_gain / avg_loss
    rsi      = 100 - (100 / (1 + rs))
    return rsi


# ------------------------------------------------------------------ #
#  Signal Helpers                                                      #
# ------------------------------------------------------------------ #

def get_ma_signal(price: float, ma50: float) -> int:
    """
    Generate a trend signal based on price vs MA50.

    Logic:
        price > MA50 + 1%  →  +1  (clear uptrend)
        price < MA50 - 1%  →  -1  (clear downtrend)
        price ≈ MA50       →   0  (neutral / choppy)

    The 1% buffer avoids noise around the moving average.

    Args:
        price : Latest closing price.
        ma50  : Latest 50-day moving average value.

    Returns:
        int: +1, 0, or -1
    """
    if pd.isna(price) or pd.isna(ma50) or ma50 == 0:
        return 0
    diff_pct = (price - ma50) / ma50 * 100
    if diff_pct > 1:
        return 1
    elif diff_pct < -1:
        return -1
    return 0


def get_rsi_signal(rsi: float) -> float:
    """
    Generate a momentum signal based on RSI value.

    Logic:
        RSI > 60  →  +1.0  (buyers are dominant, momentum is real)
        RSI < 40  →  -0.5  (sellers pushed price, momentum is weak)
        40–60     →   0.0  (neutral — no edge)

    Why -0.5 and not -1.0 for weak RSI?
        Low RSI can also mean oversold bounce potential.
        We penalise but don't fully disqualify.

    Args:
        rsi : RSI value (0–100).

    Returns:
        float: +1.0, 0.0, or -0.5
    """
    if pd.isna(rsi):
        return 0.0
    rsi = max(0.0, min(100.0, rsi))
    if rsi > 60:
        return 1.0
    elif rsi < 40:
        return -0.5
    return 0.0


def combine_signals(
    ma_signal:  int,
    rsi_signal: float,
    ma_weight:  float = 0.4,
    rsi_weight: float = 0.6,
) -> float:
    """
    Combine MA and RSI signals into a single score.

    Formula:
        score = (ma_weight × MA_signal) + (rsi_weight × RSI_signal)

    Default weights: MA=0.4, RSI=0.6
    RSI gets more weight because momentum leads price direction.

    Range: [-1.0, +1.0]

    Args:
        ma_signal  : Output of get_ma_signal()  (+1, 0, -1)
        rsi_signal : Output of get_rsi_signal() (+1.0, 0.0, -0.5)
        ma_weight  : Weight for MA signal  (default 0.4)
        rsi_weight : Weight for RSI signal (default 0.6)

    Returns:
        float: Combined score rounded to 2 decimal places.
    """
    return round(ma_weight * ma_signal + rsi_weight * rsi_signal, 2)
