"""
scoring.py — Stock Scoring Model

Combines two signals into a single score [-1.0, +1.0]:
    1. Trend    : MA50 + MA200 (price structure)
    2. Momentum : RSI14

What changed from v1:
    - get_ma_signal() now accepts optional ma200 argument
    - If ma200 is provided, golden/death cross gives a confidence bonus
    - Added is_bullish() — hard boolean filter used in P2 screener
      Rule: price > SMA50 > SMA200 AND 40 < RSI < 70 → BULLISH
    - calculate_score() now accepts optional ma200 argument
    - Backward compatible: all old calls still work without ma200
"""

import pandas as pd


class StockScorer:
    """
    Scores a stock from -1.0 (avoid) to +1.0 (strong buy).

    Usage:
        scorer = StockScorer()

        # Basic (P1 style, MA50 + RSI only):
        score = scorer.calculate_score(price=500, ma50=480, rsi14=65)

        # With SMA200 (P2 style, full signal):
        score = scorer.calculate_score(price=500, ma50=480, rsi14=65, ma200=450)
    """

    def __init__(self, ma_weight: float = 0.4, rsi_weight: float = 0.6):
        """
        Args:
            ma_weight  : Weight for trend signal. Default 0.4.
            rsi_weight : Weight for momentum signal. Default 0.6.
                         Must sum to 1.0.
        """
        if abs(ma_weight + rsi_weight - 1.0) > 0.001:
            raise ValueError(
                f"Weights must sum to 1.0, got {ma_weight + rsi_weight:.3f}"
            )
        self.ma_weight  = ma_weight
        self.rsi_weight = rsi_weight

    # ---------------------------------------------------------------- #
    #  Signal 1: Trend (MA50 + optional MA200)                         #
    # ---------------------------------------------------------------- #

    def get_ma_signal(
        self,
        price: float,
        ma50:  float,
        ma200: float = None,
    ) -> float:
        """
        Trend signal based on price vs MA50, with optional MA200 confirmation.

        Base signal (MA50 only):
            price > MA50 + 1%  →  +1.0  (uptrend)
            price < MA50 - 1%  →  -1.0  (downtrend)
            price ≈ MA50       →   0.0  (neutral)

        MA200 bonus (if ma200 is provided):
            Golden cross (MA50 > MA200) while price is above MA50:
                adds +0.5 → capped at +1.0
            Death cross (MA50 < MA200) while price is below MA50:
                subtracts -0.5 → floored at -1.0

        Why cap at ±1.0?
            Weights are calibrated for [-1, +1].
            Letting signal go above 1 would distort the combined score.

        Args:
            price : Latest closing price.
            ma50  : 50-day SMA value.
            ma200 : 200-day SMA value (optional).

        Returns:
            float: signal in range [-1.0, +1.0]
        """
        if pd.isna(price) or pd.isna(ma50) or ma50 == 0:
            return 0.0

        diff_pct = (price - ma50) / ma50 * 100

        # Base signal from MA50
        if diff_pct > 1:
            base = 1.0
        elif diff_pct < -1:
            base = -1.0
        else:
            base = 0.0

        # MA200 confirmation bonus / penalty
        if ma200 is not None and not pd.isna(ma200) and ma200 > 0:
            if base > 0 and ma50 > ma200:
                # Golden cross structure: price > MA50 > MA200
                # This is the strongest uptrend signal in technical analysis
                base = min(base + 0.5, 1.0)
            elif base < 0 and ma50 < ma200:
                # Death cross structure: price < MA50 < MA200
                # Confirmed downtrend
                base = max(base - 0.5, -1.0)

        return base

    # ---------------------------------------------------------------- #
    #  Signal 2: Momentum (RSI14)                                      #
    # ---------------------------------------------------------------- #

    def get_rsi_signal(self, rsi14: float) -> float:
        """
        Momentum signal based on RSI.

        RSI > 60  →  +1.0  (active buying pressure, momentum confirmed)
        40–60     →   0.0  (no momentum edge, stand aside)
        RSI < 40  →  -0.5  (selling pressure, partial penalty)

        Why not -1.0 for low RSI?
            Low RSI can mean oversold bounce is coming.
            We penalise but don't fully disqualify.

        Args:
            rsi14 : RSI value (0–100).

        Returns:
            float: +1.0, 0.0, or -0.5
        """
        if pd.isna(rsi14):
            return 0.0
        rsi14 = max(0.0, min(100.0, float(rsi14)))
        if rsi14 > 60:
            return 1.0
        elif rsi14 < 40:
            return -0.5
        return 0.0

    # ---------------------------------------------------------------- #
    #  Combined score                                                   #
    # ---------------------------------------------------------------- #

    def calculate_score(
        self,
        price:  float,
        ma50:   float,
        rsi14:  float,
        ma200:  float = None,
    ) -> float:
        """
        Final score = ma_weight × MA_signal + rsi_weight × RSI_signal

        Range: [-1.0, +1.0]

        Args:
            price : Latest close.
            ma50  : 50-day SMA.
            rsi14 : RSI (14 period).
            ma200 : 200-day SMA (optional but recommended for P2).

        Returns:
            float: Rounded to 2 decimal places.
        """
        ma_sig  = self.get_ma_signal(price, ma50, ma200)
        rsi_sig = self.get_rsi_signal(rsi14)
        score   = self.ma_weight * ma_sig + self.rsi_weight * rsi_sig
        return round(score, 2)

    # ---------------------------------------------------------------- #
    #  P2 hard filter: bullish condition                                #
    # ---------------------------------------------------------------- #

    @staticmethod
    def is_bullish(
        price: float,
        ma50:  float,
        ma200: float,
        rsi14: float,
    ) -> bool:
        """
        Hard boolean rule for P2 screener signal condition:

            IF price > SMA50 > SMA200 AND 40 < RSI < 70
            THEN SIGNAL = BULLISH

        This is a pre-filter. Use this BEFORE scoring to check if a
        stock even qualifies as bullish.

        Why 40 < RSI < 70 (not the usual 30/70)?
            - RSI > 70 means overbought — you're buying too late
            - RSI > 40 means momentum has already turned positive
            - This range catches stocks with healthy, sustainable momentum

        Args:
            price : Latest close.
            ma50  : 50-day SMA.
            ma200 : 200-day SMA.
            rsi14 : RSI value.

        Returns:
            bool: True if all conditions met.
        """
        if any(pd.isna(v) for v in [price, ma50, ma200, rsi14]):
            return False
        return (price > ma50 > ma200) and (40 < rsi14 < 70)

    # ---------------------------------------------------------------- #
    #  Interpretation label                                             #
    # ---------------------------------------------------------------- #

    def get_interpretation(self, score: float) -> str:
        """
        Convert numeric score to human-readable label.

        Score ≥ 0.7  → STRONG BUY  (trend + momentum both aligned)
        Score ≥ 0.3  → BUY         (one signal strong, other neutral)
        Score > -0.3 → HOLD        (no edge — professionals do nothing here)
        Score ≤ -0.3 → SELL        (trend broken or momentum weak)
        """
        if score >= 0.7:
            return "🔥 STRONG BUY"
        elif score >= 0.3:
            return "👍 BUY"
        elif score > -0.3:
            return "➖ HOLD"
        else:
            return "⛔ SELL"


# ------------------------------------------------------------------ #
#  Self-test (run with: python -m src.scoring)                        #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    scorer = StockScorer()

    # --- existing tests (must not break) ---
    assert scorer.calculate_score(500, 480, 65)  ==  1.0,  "Test 1 failed"
    assert scorer.calculate_score(300, 310, 35)  == -0.7,  "Test 2 failed"
    assert scorer.calculate_score(400, 405, 50)  == -0.4,  "Test 3 failed"

    # --- new: MA200 golden cross bonus ---
    # price=500 > ma50=480 > ma200=450 → golden cross → bonus applied
    score_golden = scorer.calculate_score(500, 480, 65, ma200=450)
    assert score_golden == 1.0, f"Golden cross test failed: {score_golden}"

    # --- new: MA200 death cross penalty ---
    # price=300 < ma50=310 < ma200=350 → death cross → penalty applied
    score_death = scorer.calculate_score(300, 310, 35, ma200=350)
    assert score_death == -0.7, f"Death cross test failed: {score_death}"

    # --- new: is_bullish filter ---
    # All conditions met: price > ma50 > ma200, RSI in 40-70
    assert StockScorer.is_bullish(500, 480, 450, 55) is True,  "Bullish test 1 failed"
    # RSI too high (overbought)
    assert StockScorer.is_bullish(500, 480, 450, 72) is False, "Bullish test 2 failed"
    # MA50 < MA200 (not golden cross)
    assert StockScorer.is_bullish(500, 480, 490, 55) is False, "Bullish test 3 failed"

    print("✓ All scoring tests pass")
