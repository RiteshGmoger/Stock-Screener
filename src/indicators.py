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

