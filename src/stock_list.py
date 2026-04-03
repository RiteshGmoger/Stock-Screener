"""
stock_list.py — Universe of stocks to screen

Defines two lists:
    NIFTY_50_TICKERS : 15 hand-picked Nifty 50 large-caps across sectors
    TEST_TICKERS     : Same as above — used for fast testing

Why keep them the same size for now?
    During development, TEST = small and fast.
    Once you expand to the full 50-stock Nifty universe, swap NIFTY_50_TICKERS
    to the complete list. TEST can stay small for CI/unit tests.
"""

NIFTY_50_TICKERS = [
    'RELIANCE.NS',     # Energy giant
    'TCS.NS',          # IT service
    'INFY.NS',         # IT service
    'HDFCBANK.NS',     # Banking
    'ICICIBANK.NS',    # Banking
    'AXISBANK.NS',     # Banking
    'KOTAKBANK.NS',    # Banking
    'LT.NS',           # Engineering, construction
    'WIPRO.NS',        # IT service
    'HCLTECH.NS',      # IT service
    'BAJAJFINSV.NS',   # Financial services
    'MARUTI.NS',       # Automobiles
    'BHARTIARTL.NS',   # Telecom
    'SUNPHARMA.NS',    # Pharma
    'DRREDDY.NS',      # Pharma
]

TEST_TICKERS = NIFTY_50_TICKERS[:15]


def get_stock_list(use_test: bool = True):
    """
    Returns list of tickers for screening.
    use_test=True → smaller list (faster, safer)
    use_test=False → full universe
    """
    return TEST_TICKERS if use_test else NIFTY_50_TICKERS
