import pandas as pd
import numpy as np


def calculate_moving_average(price_series, window=50):
    """
    Calculate moving average (SMA).
    """
    return price_series.rolling(window=window).mean()


def calculate_rsi(price_series, period=14):
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = price_series.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def get_ma_signal(current_price, ma_value, threshold_pct=1.0):
    if pd.isna(ma_value):
        return 0

    pct_diff = (current_price - ma_value) / ma_value * 100

    if pct_diff > threshold_pct:
        return +1
    elif pct_diff < -threshold_pct:
        return -1
    else:
        return 0


def get_rsi_signal(rsi_value):
    if pd.isna(rsi_value):
        return 0

    if rsi_value > 60:
        return +1.0
    elif rsi_value < 40:
        return -0.5
    else:
        return 0


def combine_signals(ma_signal, rsi_signal, ma_weight=0.4, rsi_weight=0.6):
    score = (ma_weight * ma_signal) + (rsi_weight * rsi_signal)
    
    return max(-1.0, min(1.0, score))

