import pandas as pd
import numpy as np


def calculate_moving_average(price_series, window=50):
    """
    Calculate moving average (MA).
    """
    return price_series.rolling(window=window).mean()


def calculate_rsi(price_series, period=14):
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = price_series.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean() #check buyers
    avg_loss = loss.rolling(window=period).mean() #check sellers

    rs = avg_gain / avg_loss
    """
    if we checking sellers it would be avg_loss / avg_gain
    even we dont use this we can still see buyers and sellers both

    avg_gain = 6
    avg_loss = 2
    RS = 3
    Buyers are 3× stronger
    
    avg_gain = 2
    avg_loss = 6
    RS = 0.33
    Sellers are 3× stronger
    """
    rsi = 100 - (100 / (1 + rs))
    """
    if rs = 0 : rsi = 0
       rs = INF : rsi = 100
       
    making rs b/w (0 - 100)
    """

    return rsi

