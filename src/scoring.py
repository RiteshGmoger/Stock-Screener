"""
    We take two things:
        trend     - from MA50 and MA200
        momentum  - from RSI14

    and combine them into one number between -1.0 and +1.0

    What’s new here:
        MA200 is now used for confirmation (golden/death cross idea)
        added a simple bullish filter to avoid weak setups
        scoring still works even if MA200 is not available

    This is the core decision layer used by the screener and backtest
"""

import pandas as pd

class StockScorer:
    """
        Scores a stock from -1.0 (avoid) to +1.0 (strong buy)

        # Basic (P1 style, MA50 + RSI only):
        score = scorer.calculate_score(price=500, ma50=480, rsi14=65)

        # With SMA200 (P2 style, full signal):
        score = scorer.calculate_score(price=500, ma50=480, rsi14=65, ma200=450)
    """

    def __init__(self, ma_weight: float = 0.4, rsi_weight: float = 0.6):
        """
            We combine two signals:
                MA → tells us trend
                RSI → tells us momentum

            We give RSI a bit more weight (0.6) because momentum usually shows up
            before trend fully forms.
            MA still matters (0.4), but it reacts slower, so slightly less importance.=
            
            so basically,
                RSI gets more weight (0.6) since it reacts faster to price moves
                MA is slower, so we keep it at 0.4.

            Weights should add up to 1 so the final score stays clean and predictable
        """
        
        if abs(ma_weight + rsi_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {ma_weight + rsi_weight:.3f}")
        self.ma_weight  = ma_weight
        self.rsi_weight = rsi_weight


    def get_ma_signal(self,price: float,ma50:  float,ma200: float = None) -> float:
        """
            We use moving averages to understand trend

            We compare price with MA50 (last 50-day average)
                If price is clearly above MA50 → uptrend → +1
                If price is clearly below MA50 → downtrend → -1
                If price is too close → nothing clear → 0

            We use a small 1% gap so tiny fluctuations don’t give fake signals

            MA200 is a slower average (long-term trend)
                If price > MA50 AND MA50 > MA200  
                    strong uptrend (short-term + long-term aligned)  

                If price < MA50 AND MA50 < MA200  
                    strong downtrend  
                    reduce score more

                If MA50 and MA200 don’t agree, we don’t touch the signal.

            We never let signal go above +1 or below -1  
            because later we combine this with RSI.=

            Example:
                Case 1 (strong uptrend):
                    price = 500
                    ma50  = 480
                    ma200 = 450

                    price > ma50 → uptrend → +1  
                    ma50 > ma200 → confirmed → stays strong (+1)

                Case 2 (weak uptrend):
                    price = 500
                    ma50  = 480
                    ma200 = 490

                    price > ma50 → uptrend  
                    but ma50 < ma200 → long-term still weak  
                    no extra confidence

                Case 3 (strong downtrend):
                    price = 300
                    ma50  = 320
                    ma200 = 350

                    price < ma50 → downtrend → -1  
                    ma50 < ma200 → confirmed → stays strong (-1)


            so basically,
                MA50 tells “what is happening now”  
                MA200 tells “is this real or not”
        """
        if pd.isna(price) or pd.isna(ma50) or ma50 == 0:
            return 0.0

        diff_pct = (price - ma50) / ma50 * 100

        if diff_pct > 1:
            base = 1.0
        elif diff_pct < -1:
            base = -1.0
        else:
            base = 0.0

        if ma200 is not None and not pd.isna(ma200) and ma200 > 0:
            if base > 0 and ma50 > ma200:
                """
                    Golden cross structure: price > MA50 > MA200
                    This is the strongest uptrend signal in technical analysis
                """
                # Bcz i dont waant  value to go above 1
                base = min(base + 0.5, 1.0)
            elif base < 0 and ma50 < ma200:
                """
                    Death cross structure: price < MA50 < MA200
                    Confirmed downtrend
                """
                base = max(base - 0.5, -1.0)

        return base


    def get_rsi_signal(self, rsi14: float) -> float:
        """
            This converts RSI into a simple momentum signal:
            +1.0 (strong momentum), 0.0 (neutral), -0.5 (weak momentum)

            Check RSI level:
                If RSI > 60
                    buyers are in control
                    return +1.0 (momentum is strong)

                If RSI < 40
                    sellers dominated recently
                    return -0.5 (momentum is weak, but not fully bearish)

                If 40 ≤ RSI ≤ 60
                    no clear direction
                    return 0.0 (neutral)

            Why only -0.5 for low RSI?
                Low RSI can also mean stock is oversold
                it might bounce up soon
                so we reduce score, but don’t kill it completely

            Example:
                RSI = 72
                strong buying → +1.0

                RSI = 55
                no edge → 0.0

                RSI = 35
                weak / selling pressure → -0.5
        """
        if pd.isna(rsi14):
            return 0.0
            
        rsi14 = max(0.0, min(100.0, float(rsi14)))
        
        if rsi14 > 60:
            return 1.0
        elif rsi14 < 40:
            return -0.5
        return 0.0
        

    def calculate_score(self,price:  float,ma50:   float,rsi14:  float,ma200:  float = None) -> float:
        """
            This combines trend (MA) and momentum (RSI) into one final score

            MA tells direction (up/down),
            RSI tells strength (how strong the move is)

            We give weights to both and add them:
                score = (MA × 0.4) + (RSI × 0.6)

            RSI has more weight because momentum usually leads price
            Basiclly we give more importance to RSI then MA

            Result is between -1 and +1:
                closer to +1 → strong buy
                closer to -1 → strong sell
                near 0       → no clear edge

            Example:

                ma_signal = +1
                rsi_signal = +1.0
                
                score = (1 × 0.4) + (1 × 0.6)
                      = 0.4 + 0.6
                      = 1.0  → strong buy

                ma_signal = +1
                rsi_signal = 0.0
                
                score = (1 × 0.4) + (0 × 0.6)
                      = 0.4 → buy

                ma_signal = -1
                rsi_signal = -0.5
                
                score = (-1 × 0.4) + (-0.5 × 0.6)
                      = -0.4 - 0.3
                      = -0.7 → sell

                ma_signal = 0
                rsi_signal = 0
                
                score = 0 → no signal
        """
        ma_sig  = self.get_ma_signal(price, ma50, ma200)
        rsi_sig = self.get_rsi_signal(rsi14)
        score   = self.ma_weight * ma_sig + self.rsi_weight * rsi_sig
        return round(score, 2)


    @staticmethod
    def is_bullish(price: float,ma50:  float,ma200: float,rsi14: float) -> bool:
        """
            Simple filter to catch strong, clean trends

            We only care about stocks that are already moving properly,
            not random sideways or messy ones

            So,
                price > MA50 > MA200  → trend is aligned (short + long term both up)
                40 < RSI < 70         → momentum is there but not too late

            Why,
                If RSI > 70, it’s already overbought, you’re probably late
                If RSI < 40, momentum is weak, nothing interesting yet

            So this basically picks stocks that are:
                already trending
                but not exhausted yet

            We run this before scoring so we don’t waste time
            ranking bad or noisy stocks

            price  - latest price
            ma50   - short term trend
            ma200  - long term trend
            rsi14  - momentum

            Returns,
                True  → looks good, worth considering
                False → ignore it
        """
        if any(pd.isna(v) for v in [price, ma50, ma200, rsi14]):
            return False
        """
            If ANY value is missing -> reject this stock
        """
        
        return (price > ma50 > ma200) and (40 < rsi14 < 70)


    def get_interpretation(self, score: float) -> str:
        """
            Score ≥ 0.7  → STRONG BUY  (trend + momentum both aligned)
            Score ≥ 0.3  → BUY         (one signal strong, other neutral)
            Score > -0.3 → HOLD        (no edge — professionals do nothing here)
            Score ≤ -0.3 → SELL        (trend broken or momentum weak)
        """
        
        if score >= 0.7:
            return "STRONG BUY"
        elif score >= 0.3:
            return "BUY"
        elif score > -0.3:
            return "HOLD"
        elif score > -0.7:
            return "WEAK SELL"
        else:
            return "SELL"


if __name__ == "__main__":
    scorer = StockScorer()

    """
        quick checks so we don’t break scoring logic

        using simple fake values to verify behavior
        if any of this fails -- something is off
    """

    # basic scoring (no MA200)
    assert scorer.calculate_score(500, 480, 65) == 1.0
    assert scorer.calculate_score(300, 310, 35) == -0.7
    assert scorer.calculate_score(400, 405, 50) == -0.4


    """
        MA200 confirmation

        strong uptrend → price > ma50 > ma200
        should stay capped at +1.0
    """
    score_golden = scorer.calculate_score(500, 480, 65, ma200=450)
    assert score_golden == 1.0, f"Golden cross test failed: {score_golden}"


    """
        strong downtrend → price < ma50 < ma200
        penalty applied but stays in range
    """
    score_death = scorer.calculate_score(300, 310, 35, ma200=350)
    assert score_death == -0.7, f"Death cross test failed: {score_death}"


    """
        bullish filter
        we only accept clean trends with healthy momentum
    """

    # valid case -- should pass
    assert StockScorer.is_bullish(500, 480, 450, 55) is True

    # RSI too high -- late entry -- reject
    assert StockScorer.is_bullish(500, 480, 450, 72) is False

    # check trend aligned or not
    assert StockScorer.is_bullish(500, 480, 490, 55) is False


    print("✓ All scoring tests pass")
