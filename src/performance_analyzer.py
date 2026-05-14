"""
    performance_analyzer.py

    Reads backtest output CSVs and produces:
        six charts saved to outputs/plots/
        a detailed statistics report printed to terminal

    Expected inputs:
        outputs/backtests/backtest_results.csv(monthly summary)
        outputs/backtests/backtest_picks.csv(individual trades)

    Column contract (must match what backtest.py writes):
        results  : Month, Portfolio_Return, Nifty_Return, Outperformance, Num_Stocks
        picks    : Ticker, Score, Entry_Price, Exit_Price, Return, Month
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

PLOT_DIR = "outputs/plots"


class PerformanceAnalyzer:

    def __init__(self,
        results_file: str = "outputs/backtests/backtest_results.csv",
        picks_file: str = "outputs/backtests/backtest_picks.csv"
    ):
        os.makedirs(PLOT_DIR, exist_ok = True)

        try:
            self.results = pd.read_csv(results_file)
            print(f"Loaded {len(self.results)} months from {results_file}")
        except FileNotFoundError:
            print(f"Results file not found: {results_file}")
            self.results = None

        try:
            self.picks = pd.read_csv(picks_file)
            print(f"Loaded {len(self.picks)} picks from {picks_file}")
        except FileNotFoundError:
            print(f"Picks file not found: {picks_file}")
            self.picks = None



    def _save(self, filename: str) -> None:
        path = os.path.join(PLOT_DIR, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved  {path}")


    def _equity_curve(self) -> np.ndarray:
        return (1 + self.results["Portfolio_Return"] / 100).cumprod().values

    def _nifty_curve(self) -> np.ndarray:
        return (1 + self.results["Nifty_Return"] / 100).cumprod().values



    def plot_cumulative_returns(self, save: bool = True) -> None:
        """Cumulative return of portfolio vs Nifty 50."""
        if self.results is None:
            return

        port_cum  = self._equity_curve()  - 1
        nifty_cum = self._nifty_curve()   - 1
        months    = range(1, len(self.results) + 1)

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(months, port_cum  * 100, label="Portfolio",
                linewidth=2.5, marker="o", markersize=5)
        ax.plot(months, nifty_cum * 100, label="Nifty 50",
                linewidth=2.5, marker="s", markersize=5, alpha=0.75)
        ax.fill_between(months, port_cum * 100, nifty_cum * 100, alpha=0.15)
        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.4)

        ax.set_xlabel("Month",                  fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)",  fontsize=12, fontweight="bold")
        ax.set_title("Cumulative Returns  —  Portfolio vs Nifty 50",
                     fontsize=14, fontweight="bold", pad=16)
        ax.legend(fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

        plt.tight_layout()
        if save:
            self._save("cumulative_returns.png")
        plt.show()


    def plot_monthly_returns(self, save: bool = True) -> None:
        """Side-by-side bar chart of monthly portfolio vs Nifty returns."""
        if self.results is None:
            return

        portfolio = self.results["Portfolio_Return"].values
        nifty     = self.results["Nifty_Return"].values
        x         = np.arange(len(self.results))
        width     = 0.35

        fig, ax = plt.subplots(figsize=(14, 7))

        port_colors  = ["#2ecc71" if r >= 0 else "#e74c3c" for r in portfolio]
        nifty_colors = ["#27ae60" if r >= 0 else "#c0392b" for r in nifty]

        ax.bar(x - width / 2, portfolio, width,
               color=port_colors,  alpha=0.85, edgecolor="black", label="Portfolio")
        ax.bar(x + width / 2, nifty,     width,
               color=nifty_colors, alpha=0.65, edgecolor="black", label="Nifty 50")

        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Month",         fontsize=12, fontweight="bold")
        ax.set_ylabel("Return (%)",    fontsize=12, fontweight="bold")
        ax.set_title("Monthly Returns  —  Portfolio vs Nifty 50",
                     fontsize=14, fontweight="bold", pad=16)
        ax.set_xticks(x)
        ax.set_xticklabels(self.results["Month"], rotation=45, ha="right")
        ax.legend(fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

        plt.tight_layout()
        if save:
            self._save("monthly_returns.png")
        plt.show()


    def plot_outperformance(self, save: bool = True) -> None:
        """Bar chart of monthly alpha (portfolio minus benchmark)."""
        if self.results is None:
            return

        alpha  = self.results["Outperformance"].values
        months = self.results["Month"].values
        colors = ["#2ecc71" if x >= 0 else "#e74c3c" for x in alpha]

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.bar(range(len(months)), alpha, color=colors, alpha=0.8, edgecolor="black")
        ax.axhline(0, color="black", linewidth=1.5)

        avg_alpha = np.mean(alpha)
        ax.axhline(avg_alpha, color="#2980b9", linewidth=2, linestyle="--",
                   label=f"Avg alpha  {avg_alpha:+.2f}%")

        ax.set_xlabel("Month",                  fontsize=12, fontweight="bold")
        ax.set_ylabel("Outperformance (%)",      fontsize=12, fontweight="bold")
        ax.set_title("Monthly Alpha  —  Portfolio minus Benchmark",
                     fontsize=14, fontweight="bold", pad=16)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha="right")
        ax.legend(fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

        plt.tight_layout()
        if save:
            self._save("outperformance.png")
        plt.show()


    def plot_drawdown(self, save: bool = True) -> None:
        """Underwater / drawdown chart of portfolio equity curve."""
        if self.results is None:
            return

        equity      = self._equity_curve()
        running_max = np.maximum.accumulate(equity)
        drawdown    = (equity - running_max) / running_max * 100

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.fill_between(range(len(drawdown)), drawdown, 0, color="#e74c3c", alpha=0.30)
        ax.plot(drawdown, color="#c0392b", linewidth=2)

        max_dd     = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        ax.axhline(max_dd, color="#e74c3c", linestyle="--",
                   label=f"Max drawdown  {max_dd:.2f}%")
        ax.scatter(max_dd_idx, max_dd, color="#c0392b", s=90, zorder=5)

        ax.set_xlabel("Month",          fontsize=12, fontweight="bold")
        ax.set_ylabel("Drawdown (%)",   fontsize=12, fontweight="bold")
        ax.set_title("Portfolio Drawdown",
                     fontsize=14, fontweight="bold", pad=16)
        ax.legend(fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

        plt.tight_layout()
        if save:
            self._save("drawdown.png")
        plt.show()


    def plot_stock_frequency(self, save: bool = True, top_n: int = 10) -> None:
        """Horizontal bar chart of most frequently selected stocks."""
        if self.picks is None:
            print("No picks data available for stock frequency chart")
            return

        counts = self.picks["Ticker"].value_counts().head(top_n)
        colors = sns.color_palette("viridis", len(counts))

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.barh(range(len(counts)), counts.values,
                color=colors, edgecolor="black")
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index, fontsize=11)

        for i, v in enumerate(counts.values):
            ax.text(v + 0.05, i, str(v), va="center", fontweight="bold")

        ax.set_xlabel("Times Selected",             fontsize=12, fontweight="bold")
        ax.set_ylabel("Ticker",                     fontsize=12, fontweight="bold")
        ax.set_title(f"Top {top_n} Most Selected Stocks",
                     fontsize=14, fontweight="bold", pad=16)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        plt.tight_layout()
        if save:
            self._save("stock_frequency.png")
        plt.show()


    def plot_score_vs_return(self, save: bool = True) -> None:
        """Scatter plot: model score vs actual trade return with trend line."""
        if self.picks is None:
            print("No picks data available for score vs return chart")
            return

        scores  = self.picks["Score"].values
        returns = self.picks["Return"].values
        colors  = ["#2ecc71" if r >= 0 else "#e74c3c" for r in returns]

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.scatter(scores, returns, c=colors, alpha=0.65, s=90, edgecolors="black")

        z = np.polyfit(scores, returns, 1)
        x_line = np.linspace(scores.min(), scores.max(), 200)
        ax.plot(x_line, np.poly1d(z)(x_line), color="#2980b9", linewidth=2,
                linestyle="--", label=f"Trend  y = {z[0]:.2f}x + {z[1]:.2f}")

        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=1)

        corr = np.corrcoef(scores, returns)[0, 1]
        ax.text(0.05, 0.95, f"Correlation  {corr:.3f}",
                transform=ax.transAxes, fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel("Model Score",        fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual Return (%)",  fontsize=12, fontweight="bold")
        ax.set_title("Score vs Actual Return  —  Validation",
                     fontsize=14, fontweight="bold", pad=16)
        ax.legend(fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

        plt.tight_layout()
        if save:
            self._save("score_vs_return.png")
        plt.show()


    # statistics report

    def print_statistics(self) -> None:
        if self.results is None:
            return

        portfolio = self.results["Portfolio_Return"].values
        nifty     = self.results["Nifty_Return"].values
        alpha     = self.results["Outperformance"].values

        W = 80
        SEP = "─" * W

        def row(label: str, value: str, indent: int = 3) -> None:
            pad = " " * indent
            print(f"{pad}{label:<28}{value}")

        print()
        print(SEP)
        print("PERFORMANCE REPORT".center(W))
        print(SEP)

        # return statistics
        print()
        print("  Return Statistics")
        print()
        row("Portfolio mean",    f"{np.mean(portfolio):+.2f}%")
        row("Portfolio median",  f"{np.median(portfolio):+.2f}%")
        row("Portfolio std dev", f"{np.std(portfolio):.2f}%")
        row("Portfolio min",     f"{np.min(portfolio):+.2f}%")
        row("Portfolio max",     f"{np.max(portfolio):+.2f}%")
        print()
        row("Nifty mean",        f"{np.mean(nifty):+.2f}%")
        row("Nifty std dev",     f"{np.std(nifty):.2f}%")

        # risk-adjusted metrics
        sharpe = np.mean(portfolio) / np.std(portfolio) if np.std(portfolio) > 0 else 0.0

        downside = portfolio[portfolio < 0]
        sortino  = (np.mean(portfolio) / np.std(downside)
                    if len(downside) > 1 else 0.0)

        equity      = (1 + portfolio / 100).cumprod()
        running_max = np.maximum.accumulate(equity)
        drawdown    = (equity - running_max) / running_max * 100
        max_dd      = float(drawdown.min())

        print()
        print("  Risk-Adjusted Metrics")
        print()
        row("Sharpe ratio",  f"{sharpe:.3f}")
        row("Sortino ratio", f"{sortino:.3f}")
        row("Max drawdown",  f"{max_dd:.2f}%")

        # win / loss
        wins     = int((portfolio > 0).sum())
        losses   = int((portfolio < 0).sum())
        win_rate = wins / len(portfolio) * 100

        avg_win  = np.mean(portfolio[portfolio > 0]) if wins  > 0 else 0.0
        avg_loss = np.mean(portfolio[portfolio < 0]) if losses > 0 else 0.0
        wl_ratio = abs(avg_win / avg_loss)           if avg_loss != 0 else 0.0

        print()
        print("  Win / Loss")
        print()
        row("Winning months",   f"{wins}  ({win_rate:.1f}%)")
        row("Losing months",    f"{losses}")
        row("Average win",      f"{avg_win:.2f}%")
        row("Average loss",     f"{avg_loss:.2f}%")
        row("Win / loss ratio", f"{wl_ratio:.2f}")

        # alpha
        beat_n    = int((alpha > 0).sum())
        beat_rate = beat_n / len(alpha) * 100

        print()
        print("  Alpha")
        print()
        row("Average alpha",      f"{np.mean(alpha):+.2f}%")
        row("Beat benchmark",     f"{beat_n} / {len(alpha)} months  ({beat_rate:.1f}%)")
        row("Max positive alpha", f"{np.max(alpha):+.2f}%")
        row("Max negative alpha", f"{np.min(alpha):+.2f}%")

        # stock selection
        if self.picks is not None:
            pick_returns = self.picks["Return"].values
            pick_wr      = (pick_returns > 0).sum() / len(pick_returns) * 100
            corr         = np.corrcoef(self.picks["Score"].values, pick_returns)[0, 1]

            print()
            print("  Stock Selection")
            print()
            row("Total picks",          f"{len(self.picks)}")
            row("Unique stocks",        f"{self.picks['Ticker'].nunique()}")
            row("Avg pick return",      f"{np.mean(pick_returns):+.2f}%")
            row("Pick win rate",        f"{pick_wr:.1f}%")
            row("Score / return corr",  f"{corr:.3f}")

        print()
        print(SEP)
        print()
        

    def generate_report(self) -> None:
        """Run all plots and print the statistics summary."""
        W = 80
        print()
        print("─" * W)
        print("GENERATING PERFORMANCE REPORT".center(W))
        print("─" * W)
        print()

        self.plot_cumulative_returns()
        self.plot_monthly_returns()
        self.plot_outperformance()
        self.plot_drawdown()

        if self.picks is not None:
            self.plot_stock_frequency()
            self.plot_score_vs_return()

        self.print_statistics()

        print("Report complete".center(W))
        print("─" * W)
        print()


if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    analyzer.generate_report()
