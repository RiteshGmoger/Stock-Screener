"""
    It takes raw price data and converts it into:
        trend (using moving averages)
        momentum (using RSI)
        final decision score (combine signals)

    Basically:
        price → indicators → signals → score

    Whats inside:
        Moving averages (SMA, EMA) to track trend
        RSI to measure momentum
        Simple rules to convert these into buy/sell signals
        A function to combine everything into one score

    This is the main logic used by the screener and backtest
"""

import pandas as pd
import numpy as np

def calculate_moving_average(price_series: pd.Series, window: int = 50) -> pd.Series:
    """
        Takes last N prices (like 50) and computes their average

        At the beginning, there aren’t enough values,
        so it returns NaN until it has N prices

        After that, it keeps moving forward
        each step drops the oldest price, adds the newest,
        and calculates a new average
                
        Example:
            prices = [100, 102, 101, 105, 110]
            window = 3

            Output:
            [NaN, NaN, 101, 102.67, 105.33]

            Explanation:
            - First 2 values are NaN (not enough data for window=3)
            - Then,
                 avg(100,102,101) = 101
                 avg(102,101,105) = 102.67
                 avg(101,105,110) = 105.33
    """
    return price_series.rolling(window=window).mean()


def calculate_ema(price_series: pd.Series, span: int = 12) -> pd.Series:
    """
        Instead of taking equal average like SMA, this gives more importance
        to recent prices and less to older ones

        At the start, it begins from the first price, then keeps updating
        step by step using the previous EMA and the new price

        It moves forward continuously - it doesn’t drop old values suddenly,
        older prices just slowly lose importance

        Example:
            Given:
                prices = [100, 102, 101, 105, 110]
            span = 3
            
            Formula for weight:
                alpha (α) = 2 / (span + 1)

            So:
                α = 2 / (3 + 1) = 0.5
            50% -> today’s price
            50% -> previous EMA
            
            Day 1:
            EMA₁ = 100 (start = first value)

            Day 2 (price = 102):
            EMA₂ = (102 × 0.5) + (100 × 0.5)
                 = 51 + 50
                 = 101
                 
            Day 3 (price = 101):
            EMA₃ = (101 × 0.5) + (101 × 0.5)
                 = 50.5 + 50.5
                 = 101
                 
            Day 4 (price = 105):
            EMA₄ = (105 × 0.5) + (101 × 0.5)
                 = 52.5 + 50.5
                 = 103
                 
            Day 5 (price = 110):
            EMA₅ = (110 × 0.5) + (103 × 0.5)
                 = 55 + 51.5
                 = 106.5
                 
            FINAL OUTPUT:
                [100, 101, 101, 103, 106.5]
    """
    return price_series.ewm(span=span, adjust=False).mean()


def calculate_rsi(price_series: pd.Series, period: int = 14) -> pd.Series:
    """
        Take price differences (delta)
            delta = today price - yesterday price

        Split into gains and losses
            gain = delta if positive, else 0
            loss = -delta if negative, else 0

        Take average over last N days (usually 14)
            avg_gain = average of gains
            avg_loss = average of losses

        Compare them
            RS = avg_gain / avg_loss

        Convert to scale 0–100
            RSI = 100 - (100 / (1 + RS))

        Meaning:
        - More gains -- RSI goes up
        - More losses -- RSI goes down

        First N values are NaN (not enough data)

        Example:
            prices = [100, 102, 101, 105]

        Delta:
            [NaN, +2, -1, +4]

        Gain and Loss:
            gain = [0, 2, 0, 4]
            loss = [0, 0, 1, 0]

        Avg Gain and Loss:
            avg_gain = (2 + 0 + 4) / 3 = 2
            avg_loss = (0 + 1 + 0) / 3 = 0.33

        Relative Strength(RS):
            RS = 2 / 0.33 ≈ 6

        Relative Strength Index(RSI):
            RSI = 100 - (100 / (1 + 6))
                ≈ 100 - (100 / 7)
                ≈ 85.7

         High RSI = strong upward movement
    """
    delta    = price_series.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs       = avg_gain / avg_loss
    
    """
        RS can range from 0 to infinity, so we convert it into a clean 0–100 scale
        to make it easier to interpret
    """
    rsi      = 100 - (100 / (1 + rs))
    return rsi
