import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from src.screener import StockScreener
from src.stock_list import TEST_TICKERS


class Backtest:
    """
    Backtesting engine that prevents look-ahead bias by:
    1. Only using data available BEFORE screen date
    2. Selecting stocks based on historical signals
    3. Measuring forward returns after selection
    """

    def __init__(self, backtest_months=12, lookback_days=260, top_n=3, 
                 holding_days=30, start_year=2024, start_month=2):
        """
        Args:
            backtest_months: Number of months to backtest
            lookback_days: Days of historical data for indicators
            top_n: Number of top-scored stocks to hold
            holding_days: Days to hold each position
            start_year: Year to start backtest
            start_month: Month to start backtest
        """
        self.backtest_months = backtest_months
        self.lookback_days = lookback_days
        self.top_n = top_n
        self.holding_days = holding_days
        self.start_year = start_year
        self.start_month = start_month
        
        self.stock_list = TEST_TICKERS
        self.results = []
        self.monthly_picks = []  # Track stock picks each month
        self.errors = []  # Track errors for debugging

        print("\n" + "✦" * 80)
        print(" QUANTITATIVE BACKTEST ENGINE ".center(80, "✦"))
        print("✦" * 80)
        print(f"\n📊 Configuration:")
        print(f"   • Backtest Period: {backtest_months} months")
        print(f"   • Portfolio Size: Top {top_n} stocks")
        print(f"   • Holding Period: {holding_days} days")
        print(f"   • Universe: {len(self.stock_list)} stocks")
        print(f"   • Lookback: {lookback_days} days")
        print(f"   • Start: {start_year}-{start_month:02d}")
        print("\n" + "─" * 80 + "\n")

    def _get_screen_date(self, year, month):
        """Get the 15th of each month as screening date"""
        try:
            return datetime(year, month, 15)
        except ValueError:
            # Handle invalid dates (e.g., if we go past Dec)
            return None

    def _get_month_range(self):
        """Generate list of (year, month) tuples for backtesting"""
        start = datetime(self.start_year, self.start_month, 1)
        months = []
        
        for i in range(self.backtest_months):
            current = start + relativedelta(months=i)
            months.append((current.year, current.month))
        
        return months

    def _download_data_blind(self, ticker, screen_date):
        """
        Download ONLY data available before screen_date.
        This prevents look-ahead bias.
        """
        start = screen_date - timedelta(days=self.lookback_days)
        
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=screen_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True
            )
            return df if not df.empty else None
        except Exception as e:
            self.errors.append(f"Download failed for {ticker}: {str(e)[:50]}")
            return None

    def _download_future(self, ticker, screen_date):
        """
        Download forward-looking data for return calculation.
        This is ONLY used for measuring returns, never for selection.
        """
        end = screen_date + timedelta(days=self.holding_days + 10)  # Buffer for weekends
        
        try:
            df = yf.download(
                ticker,
                start=screen_date.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True
            )
            return df if len(df) >= 2 else None
        except Exception as e:
            self.errors.append(f"Future download failed for {ticker}: {str(e)[:50]}")
            return None

    def _calc_return(self, entry, exit):
        """Calculate percentage return"""
        # Handle pandas Series - extract scalar value
        if isinstance(entry, pd.Series):
            entry = entry.iloc[0] if len(entry) > 0 else entry
        if isinstance(exit, pd.Series):
            exit = exit.iloc[0] if len(exit) > 0 else exit
            
        # Now check for invalid values
        if pd.isna(entry) or pd.isna(exit) or entry == 0:
            return 0
        return ((exit - entry) / entry) * 100

    def _get_benchmark_return(self, screen_date):
        """Calculate Nifty 50 benchmark return"""
        try:
            end = screen_date + timedelta(days=self.holding_days + 10)
            nifty = yf.download(
                "^NSEI",
                start=screen_date.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True
            )
            
            if len(nifty) < 2:
                return 0
            
            return self._calc_return(nifty["Close"].iloc[0], nifty["Close"].iloc[-1])
        
        except Exception as e:
            self.errors.append(f"Benchmark download failed: {str(e)[:50]}")
            return 0

    def _screen_stocks(self, screen_date):
        """
        Screen stocks using ONLY data before screen_date.
        Returns: List of (ticker, score, entry_price) tuples
        """
        screener = StockScreener(self.stock_list, self.lookback_days)
        screener.data = {}
        
        # Download historical data for each stock
        downloaded = 0
        for ticker in self.stock_list:
            df = self._download_data_blind(ticker, screen_date)
            if df is not None:
                screener.data[ticker] = df
                downloaded += 1
        
        if downloaded == 0:
            print(f"   ⚠️  No data available for screening")
            return []
        
        # Calculate indicators
        screener.calculate_indicators()
        
        # Score stocks
        scored = []
        for ticker, ind in screener.indicators.items():
            try:
                close = ind["Close"].iloc[-1]
                ma = ind["MA50"].iloc[-1]
                rsi = ind["RSI14"].iloc[-1]
                
                # Ensure close is scalar
                if isinstance(close, pd.Series):
                    close = close.iloc[0]
                if isinstance(ma, pd.Series):
                    ma = ma.iloc[0]
                if isinstance(rsi, pd.Series):
                    rsi = rsi.iloc[0]
                
                if pd.isna(ma) or pd.isna(rsi):
                    continue
                
                score = screener.scorer.calculate_score(close, ma, rsi)
                scored.append((ticker, score, close))
                
            except Exception as e:
                self.errors.append(f"Scoring failed for {ticker}: {str(e)[:30]}")
                continue
        
        # Sort by score and return top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.top_n]

    def _calculate_portfolio_returns(self, picks, screen_date):
        """
        Calculate forward returns for selected stocks.
        This measures performance AFTER selection.
        """
        returns = []
        pick_details = []
        
        for ticker, score, entry_price in picks:
            # Download future data
            future_df = self._download_future(ticker, screen_date)
            
            if future_df is None or len(future_df) < 2:
                # If we can't get future data, skip this stock
                continue
            
            # Get exit price (30 days later or last available)
            exit_price = future_df["Close"].iloc[-1]
            
            # Ensure it's a scalar value (not a Series)
            if isinstance(exit_price, pd.Series):
                exit_price = exit_price.iloc[0]
            
            ret = self._calc_return(entry_price, exit_price)
            returns.append(ret)
            
            pick_details.append({
                'ticker': ticker,
                'score': score,
                'entry': round(entry_price, 2),
                'exit': round(exit_price, 2),
                'return': round(ret, 2)
            })
        
        return returns, pick_details

    def _display_summary(self):
        """Display beautiful summary statistics"""
        if not self.results:
            print("\n❌ No results to display\n")
            return
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "✦" * 80)
        print(" BACKTEST RESULTS ".center(80, "✦"))
        print("✦" * 80 + "\n")
        
        # Monthly results table
        print("📊 Monthly Performance:\n")
        print(df.to_string(index=False))
        
        # Summary statistics
        print("\n" + "✦" * 80)
        print(" PERFORMANCE METRICS ".center(80))
        print("✦" * 80 + "\n")
        
        portfolio_returns = df['Portfolio_Return_%'].values
        nifty_returns = df['Nifty_Return_%'].values
        outperformance = df['Outperformance_%'].values
        
        # Calculate metrics
        total_port = np.sum(portfolio_returns)
        total_nifty = np.sum(nifty_returns)
        avg_port = np.mean(portfolio_returns)
        avg_nifty = np.mean(nifty_returns)
        
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns) * 100
        beat_rate = (outperformance > 0).sum() / len(outperformance) * 100
        
        port_std = np.std(portfolio_returns)
        sharpe = (avg_port / port_std) if port_std > 0 else 0
        
        max_dd = self._calculate_max_drawdown(portfolio_returns)
        
        print(f"Portfolio Statistics:")
        print(f"  • Total Return:        {total_port:+.2f}%")
        print(f"  • Average Monthly:     {avg_port:+.2f}%")
        print(f"  • Win Rate:            {win_rate:.1f}%")
        print(f"  • Volatility (Std):    {port_std:.2f}%")
        print(f"  • Sharpe Ratio:        {sharpe:.2f}")
        print(f"  • Max Drawdown:        {max_dd:.2f}%")
        
        print(f"\nBenchmark Statistics:")
        print(f"  • Total Return:        {total_nifty:+.2f}%")
        print(f"  • Average Monthly:     {avg_nifty:+.2f}%")
        
        print(f"\nRelative Performance:")
        print(f"  • Total Outperformance: {total_port - total_nifty:+.2f}%")
        print(f"  • Avg Outperformance:   {avg_port - avg_nifty:+.2f}%")
        print(f"  • Beat Benchmark:       {beat_rate:.1f}% of months")
        
        print("\n" + "═" * 80)
        
        # Show best and worst months
        best_idx = np.argmax(portfolio_returns)
        worst_idx = np.argmin(portfolio_returns)
        
        print(f"\n🏆 Best Month:  {df.iloc[best_idx]['Month']} ({portfolio_returns[best_idx]:+.2f}%)")
        print(f"📉 Worst Month: {df.iloc[worst_idx]['Month']} ({portfolio_returns[worst_idx]:+.2f}%)")
        
        print("\n" + "═" * 80 + "\n")

    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from monthly returns"""
        cumulative = np.cumprod(1 + np.array(returns) / 100)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        return np.min(drawdown)

    def _export_results(self):
        """Export results to CSV files"""
        print("💾 Exporting results...\n")
        
        # Export monthly results
        df = pd.DataFrame(self.results)
        df.to_csv("backtest_results.csv", index=False)
        print(f"   ✅ Saved: backtest_results.csv")
        
        # Export detailed picks
        picks_data = []
        for month_data in self.monthly_picks:
            for pick in month_data['picks']:
                picks_data.append({
                    'Month': month_data['month'],
                    'Ticker': pick['ticker'],
                    'Score': pick['score'],
                    'Entry_Price': pick['entry'],
                    'Exit_Price': pick['exit'],
                    'Return_%': pick['return']
                })
        
        if picks_data:
            picks_df = pd.DataFrame(picks_data)
            picks_df.to_csv("backtest_picks.csv", index=False)
            print(f"   ✅ Saved: backtest_picks.csv")
        
        # Export errors if any
        if self.errors:
            with open("backtest_errors.log", "w") as f:
                f.write("\n".join(self.errors))
            print(f"   ⚠️  Saved: backtest_errors.log ({len(self.errors)} errors)")
        
        print("\n✅ Backtest Complete!\n")
        
    def run(self):
        """Execute the backtest"""
        print("🚀 Starting Backtest...\n")
        print("─" * 80)
        
        month_range = self._get_month_range()
        
        for idx, (year, month) in enumerate(month_range, 1):
            screen_date = self._get_screen_date(year, month)
            
            if screen_date is None:
                continue
            
            label = screen_date.strftime("%b %Y")
            
            print(f"\n[{idx:2d}/{self.backtest_months}] 📅 {label}")
            print(f"      Screen Date: {screen_date.date()}")
            
            # Screen stocks
            print(f"      🔍 Screening {len(self.stock_list)} stocks...")
            picks = self._screen_stocks(screen_date)
            
            if not picks:
                print(f"      ❌ No valid picks this month")
                self.results.append({
                    'Month': label,
                    'Portfolio_Return_%': 0,
                    'Nifty_Return_%': 0,
                    'Outperformance_%': 0,
                    'Num_Stocks': 0
                })
                continue
            
            # Show picks
            print(f"      ✅ Selected {len(picks)} stocks:")
            for i, (t, s, p) in enumerate(picks, 1):
                print(f"         {i}. {t:15s} Score: {s:+.2f}  Price: ₹{p:,.2f}")
            
            # Calculate returns
            print(f"      💰 Calculating returns...")
            returns, pick_details = self._calculate_portfolio_returns(picks, screen_date)
            
            # Store pick details
            self.monthly_picks.append({
                'month': label,
                'picks': pick_details
            })
            
            # Portfolio metrics
            portfolio_return = np.mean(returns) if returns else 0
            benchmark_return = self._get_benchmark_return(screen_date)
            outperformance = portfolio_return - benchmark_return
            
            # Store results
            self.results.append({
                'Month': label,
                'Portfolio_Return_%': round(float(portfolio_return), 2),
                'Nifty_Return_%': round(float(benchmark_return), 2),
                'Outperformance_%': round(float(outperformance), 2),
                'Num_Stocks': len(returns)
            })
            
            # Print results
            print(f"      📊 Results:")
            print(f"         Portfolio: {portfolio_return:+7.2f}%")
            print(f"         Nifty 50:  {benchmark_return:+7.2f}%")
            print(f"         Alpha:     {outperformance:+7.2f}%")
            print("─" * 80)
        
        # Display summary
        self._display_summary()
        
        # Export results
        self._export_results()


if __name__ == "__main__":
    # Run backtest for last 12 months
    bt = Backtest(
        backtest_months=12,
        lookback_days=260,
        top_n=3,
        holding_days=30,
        start_year=2024,
        start_month=2
    )
    bt.run()
