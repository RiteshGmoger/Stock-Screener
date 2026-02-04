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
