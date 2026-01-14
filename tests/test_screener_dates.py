import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.screener import StockScreener
from src.stock_list import TEST_TICKERS


print("\n" + "=" * 80)
print("TEST: SCREENER ON TODAY'S DATA")
print("=" * 80)

screener = StockScreener(TEST_TICKERS, lookback_days=260)
screener.run()

print("✓ Screener ran successfully")

if screener.results is not None and not screener.results.empty:
    print(f"✓ Ranked {len(screener.results)} stocks")
    print(
        f"✓ Top pick: "
        f"{screener.results.iloc[0]['Ticker']} "
        f"(score: {screener.results.iloc[0]['Combined_Score']})"
    )
else:
    print("No results generated")

