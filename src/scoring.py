import pandas as pd


class StockScorer:
    """
    Combines:
    - Trend (MA50)
    - Momentum (RSI14)

    Output:
    score ∈ [-1.0, +1.0]
    """

    def __init__(self, ma_weight=0.4, rsi_weight=0.6):
        total = ma_weight + rsi_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")

        self.ma_weight = ma_weight
        self.rsi_weight = rsi_weight

    def get_ma_signal(self, price, ma50):
        """
        MA signal:
        +1 → price clearly above MA (uptrend)
         0 → price near MA
        -1 → price clearly below MA (downtrend)
        """
        if pd.isna(price) or pd.isna(ma50) or ma50 == 0:
            return 0

        diff_pct = (price - ma50) / ma50 * 100

        if diff_pct > 1:
            return 1
        elif diff_pct < -1:
            return -1
        else:
            return 0

    def get_rsi_signal(self, rsi14):
        """
        RSI signal:
        +1.0 → strong momentum
         0.0 → neutral
        -0.5 → weak / oversold
        """
        if pd.isna(rsi14):
            return 0.0

        rsi14 = max(0, min(100, rsi14))
        """
        RSI is momentum, not direction.
        -  High RSI (>60) → buyers are actively pushing price now
        -  Low RSI (<40) → sellers pushed price earlier, momentum is weak now
        """
        if rsi14 > 60:
            return 1.0
        elif rsi14 < 40:
            return -0.5
        else:
            return 0.0

    def calculate_score(self, price, ma50, rsi14):
        """
        Final score:
        score = 0.4 * MA_signal + 0.6 * RSI_signal
        """
        ma_sig = self.get_ma_signal(price, ma50)
        rsi_sig = self.get_rsi_signal(rsi14)

        score = (self.ma_weight * ma_sig) + (self.rsi_weight * rsi_sig)
        return round(score, 2)

    def get_interpretation(self, score):
        if score >= 0.7:
            return "🔥 STRONG BUY"
        elif score >= 0.3:
            return "👍 BUY"
        elif score > -0.3:
            return "➖ HOLD"
        else:
            return "⛔ SELL"          
        """
        🔥 score >= 0.7 → STRONG BUY
            What this means mentally:
                Trend is clearly up
                Momentum is strong
                Both indicators agree
                Buyers are dominant now

            This is:
                high confidence
                momentum + structure aligned
                rare but powerful

            don’t get many of these — and that’s good.

        👍 0.3 ≤ score < 0.7 → BUY
            Mental meaning:
                Something is right, but not everything
                Maybe:
                trend is up, momentum is okay
                or momentum is strong, trend just turned up
                Buyers have edge, but not domination

            This is:
                acceptable risk
                normal trading opportunity
                most trades live here

        ➖ -0.3 < score < 0.3 → HOLD
                This is the most important zone.
                Mental meaning:
                Market is confused
                Buyers and sellers are balanced
                No edge
                Professionals do NOTHING here.
                Not trading is a decision.
                This zone protects you from overtrading.

        ⛔ score ≤ -0.3 → SELL
            Mental meaning:
                Trend is broken OR
                Momentum is weak AND
                Buyers are not in control
                This does not mean:
                “Short immediately”
            
            It means:
                “Don’t be long here”
                Capital preservation > making money.
        """


if __name__ == "__main__":
    scorer = StockScorer()

    assert scorer.calculate_score(500, 480, 65) == 1.0
    assert scorer.calculate_score(300, 310, 35) == -0.7
    assert scorer.calculate_score(400, 405, 50) == -0.4

    print("✓ All scoring tests pass")

