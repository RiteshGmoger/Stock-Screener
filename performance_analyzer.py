"""
Performance Analyzer: Visualize and analyze backtest results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PerformanceAnalyzer:
    """Analyze and visualize backtest performance"""
    
    def __init__(self, results_file="backtest_results.csv", 
                 picks_file="backtest_picks.csv"):
        """
        Args:
            results_file: Path to monthly results CSV
            picks_file: Path to detailed picks CSV
        """
        try:
            self.results = pd.read_csv(results_file)
            print(f"✅ Loaded {len(self.results)} months of results")
        except FileNotFoundError:
            print(f"❌ File not found: {results_file}")
            self.results = None
            return
        
        try:
            self.picks = pd.read_csv(picks_file)
            print(f"✅ Loaded {len(self.picks)} stock picks")
        except FileNotFoundError:
            print(f"⚠️  Picks file not found: {picks_file}")
            self.picks = None
    
    def plot_cumulative_returns(self, save=True):
        """Plot cumulative returns: Portfolio vs Benchmark"""
        if self.results is None:
            return
        
        portfolio_returns = self.results['Portfolio_Return_%'].values / 100
        nifty_returns = self.results['Nifty_Return_%'].values / 100
        
        # Calculate cumulative returns
        portfolio_cumulative = (1 + portfolio_returns).cumprod() - 1
        nifty_cumulative = (1 + nifty_returns).cumprod() - 1
        
        # Plot
        plt.figure(figsize=(14, 7))
        months = range(1, len(self.results) + 1)
        
        plt.plot(months, portfolio_cumulative * 100, 
                label='Portfolio', linewidth=2.5, marker='o', markersize=6)
        plt.plot(months, nifty_cumulative * 100, 
                label='Nifty 50', linewidth=2.5, marker='s', markersize=6, alpha=0.7)
        
        plt.fill_between(months, portfolio_cumulative * 100, 
                        nifty_cumulative * 100, alpha=0.2)
        
        plt.xlabel('Month', fontsize=12, fontweight='bold')
        plt.ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
        plt.title('Portfolio vs Benchmark: Cumulative Returns', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add final values as text
        final_port = portfolio_cumulative[-1] * 100
        final_nifty = nifty_cumulative[-1] * 100
        plt.text(len(months) * 0.02, final_port, 
                f'Portfolio: {final_port:.1f}%', 
                fontsize=10, fontweight='bold')
        plt.text(len(months) * 0.02, final_nifty, 
                f'Nifty 50: {final_nifty:.1f}%', 
                fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        if save:
            plt.savefig('cumulative_returns.png', dpi=300, bbox_inches='tight')
            print("📊 Saved: cumulative_returns.png")
        plt.show()
    
    def plot_monthly_returns(self, save=True):
        """Plot monthly returns comparison"""
        if self.results is None:
            return
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(self.results))
        width = 0.35
        
        portfolio = self.results['Portfolio_Return_%'].values
        nifty = self.results['Nifty_Return_%'].values
        
        # Color bars based on positive/negative
        port_colors = ['green' if r > 0 else 'red' for r in portfolio]
        nifty_colors = ['darkgreen' if r > 0 else 'darkred' for r in nifty]
        
        bars1 = ax.bar(x - width/2, portfolio, width, label='Portfolio', 
                      color=port_colors, alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, nifty, width, label='Nifty 50', 
                      color=nifty_colors, alpha=0.6, edgecolor='black')
        
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
        ax.set_title('Monthly Returns: Portfolio vs Benchmark', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.results['Month'], rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save:
            plt.savefig('monthly_returns.png', dpi=300, bbox_inches='tight')
            print("📊 Saved: monthly_returns.png")
        plt.show()
    
    def plot_outperformance(self, save=True):
        """Plot monthly outperformance (alpha)"""
        if self.results is None:
            return
        
        plt.figure(figsize=(14, 7))
        
        outperformance = self.results['Outperformance_%'].values
        months = self.results['Month'].values
        
        colors = ['green' if x > 0 else 'red' for x in outperformance]
        
        plt.bar(range(len(months)), outperformance, color=colors, 
               alpha=0.7, edgecolor='black')
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        plt.xlabel('Month', fontsize=12, fontweight='bold')
        plt.ylabel('Outperformance (%)', fontsize=12, fontweight='bold')
        plt.title('Monthly Alpha (Portfolio - Benchmark)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xticks(range(len(months)), months, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add average line
        avg_alpha = np.mean(outperformance)
        plt.axhline(y=avg_alpha, color='blue', linestyle='--', 
                   linewidth=2, label=f'Avg: {avg_alpha:.2f}%')
        plt.legend(fontsize=11)
        
        plt.tight_layout()
        if save:
            plt.savefig('outperformance.png', dpi=300, bbox_inches='tight')
            print("📊 Saved: outperformance.png")
        plt.show()
    
    def plot_drawdown(self, save=True):
        """Plot underwater (drawdown) chart"""
        if self.results is None:
            return
        
        returns = self.results['Portfolio_Return_%'].values / 100
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        
        plt.figure(figsize=(14, 7))
        plt.fill_between(range(len(drawdown)), drawdown, 0, 
                        color='red', alpha=0.3)
        plt.plot(drawdown, color='darkred', linewidth=2)
        
        plt.xlabel('Month', fontsize=12, fontweight='bold')
        plt.ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        plt.title('Portfolio Drawdown Over Time', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        plt.axhline(y=max_dd, color='red', linestyle='--', 
                   label=f'Max DD: {max_dd:.2f}%')
        plt.scatter(max_dd_idx, max_dd, color='red', s=100, zorder=5)
        plt.legend(fontsize=11)
        
        plt.tight_layout()
        if save:
            plt.savefig('drawdown.png', dpi=300, bbox_inches='tight')
            print("📊 Saved: drawdown.png")
        plt.show()
    
    def plot_stock_frequency(self, save=True, top_n=10):
        """Plot most frequently selected stocks"""
        if self.picks is None:
            print("⚠️  No picks data available")
            return
        
        stock_counts = self.picks['Ticker'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 7))
        colors = sns.color_palette("viridis", len(stock_counts))
        
        plt.barh(range(len(stock_counts)), stock_counts.values, 
                color=colors, edgecolor='black')
        plt.yticks(range(len(stock_counts)), stock_counts.index)
        plt.xlabel('Number of Times Selected', fontsize=12, fontweight='bold')
        plt.ylabel('Stock Ticker', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Most Selected Stocks', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(stock_counts.values):
            plt.text(v + 0.1, i, str(v), fontweight='bold', va='center')
        
        plt.tight_layout()
        if save:
            plt.savefig('stock_frequency.png', dpi=300, bbox_inches='tight')
            print("📊 Saved: stock_frequency.png")
        plt.show()
    
    def plot_score_vs_return(self, save=True):
        """Scatter plot: Score vs Actual Return"""
        if self.picks is None:
            print("⚠️  No picks data available")
            return
        
        plt.figure(figsize=(12, 7))
        
        scores = self.picks['Score'].values
        returns = self.picks['Return_%'].values
        
        # Color by return (positive/negative)
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        plt.scatter(scores, returns, c=colors, alpha=0.6, s=100, edgecolors='black')
        
        # Add trend line
        z = np.polyfit(scores, returns, 1)
        p = np.poly1d(z)
        plt.plot(scores, p(scores), "b--", alpha=0.8, linewidth=2, 
                label=f'Trend: y={z[0]:.1f}x+{z[1]:.1f}')
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        plt.xlabel('Stock Score', fontsize=12, fontweight='bold')
        plt.ylabel('Actual Return (%)', fontsize=12, fontweight='bold')
        plt.title('Score vs Actual Return: Validation Check', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Calculate correlation
        correlation = np.corrcoef(scores, returns)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=11, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        if save:
            plt.savefig('score_vs_return.png', dpi=300, bbox_inches='tight')
            print("📊 Saved: score_vs_return.png")
        plt.show()
    
    def generate_report(self):
        """Generate complete performance report"""
        print("\n" + "═" * 80)
        print(" GENERATING PERFORMANCE REPORT ".center(80, "═"))
        print("═" * 80 + "\n")
        
        self.plot_cumulative_returns()
        self.plot_monthly_returns()
        self.plot_outperformance()
        self.plot_drawdown()
        
        if self.picks is not None:
            self.plot_stock_frequency()
            self.plot_score_vs_return()
        
        print("\n✅ Report generation complete!")
        print("═" * 80 + "\n")
    
    def print_statistics(self):
        """Print detailed statistics"""
        if self.results is None:
            return
        
        print("\n" + "═" * 80)
        print(" DETAILED STATISTICS ".center(80, "═"))
        print("═" * 80 + "\n")
        
        portfolio = self.results['Portfolio_Return_%'].values
        nifty = self.results['Nifty_Return_%'].values
        alpha = self.results['Outperformance_%'].values
        
        print("📊 Return Statistics:")
        print(f"   Portfolio Mean:     {np.mean(portfolio):+.2f}%")
        print(f"   Portfolio Median:   {np.median(portfolio):+.2f}%")
        print(f"   Portfolio Std Dev:  {np.std(portfolio):.2f}%")
        print(f"   Portfolio Min:      {np.min(portfolio):+.2f}%")
        print(f"   Portfolio Max:      {np.max(portfolio):+.2f}%")
        
        print(f"\n📊 Benchmark Statistics:")
        print(f"   Nifty Mean:         {np.mean(nifty):+.2f}%")
        print(f"   Nifty Std Dev:      {np.std(nifty):.2f}%")
        
        print(f"\n📊 Risk-Adjusted Metrics:")
        sharpe = (np.mean(portfolio) / np.std(portfolio)) if np.std(portfolio) > 0 else 0
        print(f"   Sharpe Ratio:       {sharpe:.3f}")
        
        # Sortino (downside deviation)
        downside = portfolio[portfolio < 0]
        sortino = (np.mean(portfolio) / np.std(downside)) if len(downside) > 0 else 0
        print(f"   Sortino Ratio:      {sortino:.3f}")
        
        # Win/Loss stats
        wins = (portfolio > 0).sum()
        losses = (portfolio < 0).sum()
        win_rate = wins / len(portfolio) * 100
        
        avg_win = np.mean(portfolio[portfolio > 0]) if wins > 0 else 0
        avg_loss = np.mean(portfolio[portfolio < 0]) if losses > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        print(f"\n📊 Win/Loss Analysis:")
        print(f"   Winning Months:     {wins} ({win_rate:.1f}%)")
        print(f"   Losing Months:      {losses}")
        print(f"   Average Win:        {avg_win:.2f}%")
        print(f"   Average Loss:       {avg_loss:.2f}%")
        print(f"   Win/Loss Ratio:     {win_loss_ratio:.2f}")
        
        # Alpha stats
        beat_benchmark = (alpha > 0).sum()
        beat_rate = beat_benchmark / len(alpha) * 100
        
        print(f"\n📊 Alpha Analysis:")
        print(f"   Average Alpha:      {np.mean(alpha):+.2f}%")
        print(f"   Beat Benchmark:     {beat_benchmark}/{len(alpha)} months ({beat_rate:.1f}%)")
        print(f"   Max Positive Alpha: {np.max(alpha):+.2f}%")
        print(f"   Max Negative Alpha: {np.min(alpha):+.2f}%")
        
        if self.picks is not None:
            print(f"\n📊 Stock Selection Analysis:")
            print(f"   Total Picks:        {len(self.picks)}")
            print(f"   Unique Stocks:      {self.picks['Ticker'].nunique()}")
            
            pick_returns = self.picks['Return_%'].values
            print(f"   Avg Pick Return:    {np.mean(pick_returns):+.2f}%")
            print(f"   Pick Win Rate:      {(pick_returns > 0).sum() / len(pick_returns) * 100:.1f}%")
            
            # Score effectiveness
            correlation = np.corrcoef(self.picks['Score'], pick_returns)[0, 1]
            print(f"   Score-Return Corr:  {correlation:.3f}")
        
        print("\n" + "═" * 80 + "\n")


if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    analyzer.print_statistics()
    analyzer.generate_report()
