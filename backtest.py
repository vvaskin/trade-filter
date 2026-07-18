"""Reusable backtesting helpers for single-cluster mean reversion."""

import numpy as np
import pandas as pd


def print_stats(stats):
    print(f"Total return: {stats['total_return_pct']:.2f}%")
    print(f"Sharpe ratio: {stats['sharpe']:.2f}")
    print(f"Max drawdown: {stats['max_drawdown_pct']:.2f}%")


def prepare_weekly_data(
    prices: pd.DataFrame,
    start_date=None,
    end_date=None,
    freq: str = "W-FRI",
):
    """
    A very simple function to prepare data for a custom time period and custom frequency.
    """
    prices = prices.sort_index().copy()
    if start_date is not None:
        prices = prices.loc[start_date:]
    if end_date is not None:
        prices = prices.loc[:end_date]

    weekly_prices = prices.resample(freq).last()
    weekly_simple_returns = weekly_prices.pct_change().dropna()
    weekly_log_returns = np.log(weekly_prices / weekly_prices.shift()).dropna()
    return weekly_log_returns, weekly_simple_returns


def run_benchmark(
    prices: pd.DataFrame,
    start_date=None,
    end_date=None,
    I: float = 10_000,
    freq: str = "W-FRI",
    periods_per_year: int = 52,
):
    """
    Equal-weight buy-and-hold benchmark on the same stocks and period, used to test against our own strategy.
    """
    weekly_log_returns, weekly_simple_returns = prepare_weekly_data(
        prices, start_date, end_date, freq
    )

    benchmark_return = weekly_simple_returns.shift(-1).mean(axis=1).dropna()
    benchmark_equity = (1 + benchmark_return).cumprod() * I

    stats = {
        "total_return_pct": (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1) * 100,
        "sharpe": benchmark_return.mean() / benchmark_return.std() * np.sqrt(periods_per_year),
        "max_drawdown_pct": (benchmark_equity / benchmark_equity.cummax() - 1).min() * 100,
    }

    return {
        "equity": benchmark_equity,
        "benchmark_return": benchmark_return,
        "stats": stats,
    }


def build_mean_reversion_positions(
    weekly_log_returns: pd.DataFrame,
    gross_exposure: float = 10_000,
    tau: float = 0.0,
):
    """
    Build dollar-neutral mean-reversion positions.

    A stock is eligible only when the absolute value of its
    residual weekly return is at least tau.

    Active weeks have:
        long exposure  = +gross_exposure / 2
        short exposure = -gross_exposure / 2

    If no long or no short signals are available, the portfolio
    holds cash for that week.
    """
    if tau < 0:
        raise ValueError("tau must be non-negative.")

    if gross_exposure <= 0:
        raise ValueError("gross_exposure must be positive.")

    # Calculate residuals
    cluster_return = weekly_log_returns.mean(axis=1)
    residuals = weekly_log_returns.sub(cluster_return, axis=0)

    # Threshold residuals based on tau
    eligible_residuals = residuals.where(
        residuals.abs() >= tau,
        0.0,
    )

    # Buy negative residuals and short positive residuals.
    signals = -eligible_residuals

    # Store long and short signal strengths as positive numbers.
    long_scores = signals.clip(lower=0.0)
    short_scores = (-signals).clip(lower=0.0)

    long_total = long_scores.sum(axis=1)
    short_total = short_scores.sum(axis=1)

    # --- Ensuring dollar neutrality ---

    # An active week is one where both long and short signals are available. Otherwise strategy holds cash.
    active_week = long_total.gt(0) & short_total.gt(0)

    # Allocate half of the gross exposure to each side.
    long_positions = long_scores.div(
        long_total.replace(0, np.nan),
        axis=0,
    ) * (gross_exposure / 2)

    short_positions = -short_scores.div(
        short_total.replace(0, np.nan),
        axis=0,
    ) * (gross_exposure / 2)

    dollar_positions = (
        long_positions + short_positions
    ).fillna(0.0)

    # Do not take a directional portfolio when one side is unavailable.
    dollar_positions.loc[~active_week] = 0.0

    return dollar_positions


def run_backtest(
    prices: pd.DataFrame,
    start_date=None,
    end_date=None,
    I: float = 10_000,
    cost_bps: float = 10,
    tau: float = 0.0,
    freq: str = "W-FRI",
    periods_per_year: int = 52,
):
    """
    Single-cluster mean reversion algorithm backtest.
    """
    weekly_log_returns, weekly_simple_returns = prepare_weekly_data(
        prices, start_date, end_date, freq
    )

    # -- Algorithm slightly modified from the paper --

    dollar_positions = build_mean_reversion_positions(
        weekly_log_returns=weekly_log_returns, gross_exposure=I, tau=tau
    )

    forward_returns = weekly_simple_returns.shift(-1)
    dollar_positions, forward_returns = dollar_positions.align(
        forward_returns, join="inner", axis=0
    )
    # -- End of algorithm --

    # Calcuating PnL without transaction costs
    strategy_pnl = (dollar_positions * forward_returns).sum(axis=1).dropna()
    strategy_return = strategy_pnl / I

    # Calculating transaction costs
    turnover = dollar_positions.diff().abs().sum(axis=1)
    cost = (turnover / I) * (cost_bps / 10_000)
    strategy_return_net = (strategy_return - cost.fillna(0))

    equity = (1 + strategy_return_net).cumprod() * I

    stats = {
        "total_return_pct": (equity.iloc[-1] / equity.iloc[0] - 1) * 100,
        "sharpe": strategy_return_net.mean() / strategy_return_net.std() * np.sqrt(periods_per_year),
        "max_drawdown_pct": (equity / equity.cummax() - 1).min() * 100,
    }

    return {
        "equity": equity,
        "strategy_return": strategy_return_net,
        "dollar_positions": dollar_positions,
        "stats": stats,
    }

def backtest_positions(
    target_positions,
    forward_returns,
    initial_capital=10_000,
    cost_bps=10,
    periods_per_year=52,
):
    """
    Backtest a supplied target-position matrix. These matrices come from machine learning models in `meta-labelling.ipynb`.
    """
    pos = target_positions.sort_index().copy()

    fwd = forward_returns.reindex(
        index=pos.index,
        columns=pos.columns,
    )

    # Check
    if fwd.isna().any().any():
        raise ValueError("Missing forward returns for some positions.")

    # Gross PnL and return
    gross_pnl = (pos * fwd).sum(axis=1)
    gross_return = gross_pnl / initial_capital

    position_changes = pos.diff()

    position_changes.iloc[0] = pos.iloc[0] # Enforce first row to not be NaN

    turnover = position_changes.abs().sum(axis=1)

    # Transaction costs
    cost_rate = cost_bps / 10_000
    transaction_costs = turnover * cost_rate
    cost_return = transaction_costs / initial_capital

    # Net performance (whith transaction cost)
    net_pnl = gross_pnl - transaction_costs
    net_return = net_pnl / initial_capital

    net_equity = (
        initial_capital * (1 + net_return).cumprod()
    )

    # Include the initial capital as the first high-water mark
    running_peak = net_equity.cummax().clip(
        lower=initial_capital
    )
    drawdown = net_equity / running_peak - 1

    # Sharpe ratio
    weekly_volatility = net_return.std(ddof=1)

    if np.isclose(weekly_volatility, 0):
        sharpe = np.nan
    else:
        sharpe = (
            net_return.mean()
            / weekly_volatility
            * np.sqrt(periods_per_year)
        )

    stats = {
        "total_return": (
            net_equity.iloc[-1] / initial_capital - 1
        ),
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "total_turnover": turnover.sum(),
        "average_gross_exposure": (
            pos.abs().sum(axis=1).mean()
        ),
    }

    return {
        "net_return": net_return,
        "equity": net_equity,
        "stats": stats,
    }
