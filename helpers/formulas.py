import numpy as np
import pandas as pd

from helpers.utils import df_check


def winrate(wins: int, losses: int) -> float:
    total = wins + losses
    return wins / total if total > 0 else 0.0


def avg_metrics(
    pl_series: pd.Series = pd.Series(dtype=float),
    risk_series: pd.Series = pd.Series(dtype=float),
    rr_series: pd.Series = pd.Series(dtype=float),
) -> tuple[float, float, float, float]:
    """
    Calculate average win, average loss (absolute), average risk, and average risk-reward ratio
    from given series.

    Args:
        pl_series (pd.Series): Series of profit/loss values. Defaults to empty.
        risk_series (pd.Series): Series of risk values. Defaults to empty.
        rr_series (pd.Series): Series of risk-reward ratios. Defaults to empty.

    Returns:
        tuple: (avg_win, avg_loss, avg_risk, avg_rr) as floats.

    Notes:
        - The function handles missing or empty series by returning 0.0 for NaN values.
    """
    # Calculate average win
    avg_win = pl_series[pl_series > 0].mean()
    avg_win = 0.0 if pd.isna(avg_win) else avg_win

    # Calculate average loss (absolute value)
    avg_loss = abs(pl_series[pl_series < 0].mean())
    avg_loss = 0.0 if pd.isna(avg_loss) else avg_loss

    # Calculate average risk
    avg_risk = risk_series.mean()
    avg_risk = 0.0 if pd.isna(avg_risk) else avg_risk

    # Calculate average risk-reward ratio
    avg_rr = rr_series.mean()
    avg_rr = 0.0 if pd.isna(avg_rr) else avg_rr

    return float(avg_win), float(avg_loss), float(avg_risk), float(avg_rr)


def best_worst_trade(pl_series: pd.Series) -> tuple[float, float]:
    best_trade_value = pl_series.max() or 0.0
    worst_trade_value = pl_series.min() or 0.0
    return float(best_trade_value), float(worst_trade_value)


def max_drawdown_from_pct_returns(
    perTrade_returns=None, cumulative_returns=None
) -> float:
    """
    Calculate max drawdown from a series of percentage returns or cumulative percentage returns.

    Parameters:
    - perTrade_returns (list or pd.Series): Raw percentage returns per period (e.g., 0.01 = 1%).
    - cumulative_returns (list or pd.Series): Cumulative percentage returns (e.g., 0.0299 = 2.99%).

    Returns:
    - float: Max drawdown as a positive decimal (e.g., 0.05 for 5%).

    Raises:
    - ValueError: If both or neither inputs are provided, or if any peak value is -100% (causing division-by-zero).
    """
    if (perTrade_returns is None and cumulative_returns is None) or (
        perTrade_returns is not None and cumulative_returns is not None
    ):
        raise ValueError(
            "Provide exactly one of perTrade_returns or cumulative_returns"
        )
    if perTrade_returns is not None:
        if not isinstance(perTrade_returns, pd.Series):
            perTrade_returns = pd.Series(perTrade_returns)
        returns_curve = (1 + perTrade_returns).cumprod() - 1  # right way to compounds
    else:
        if not isinstance(cumulative_returns, pd.Series):
            cumulative_returns = pd.Series(cumulative_returns)
        returns_curve = cumulative_returns

    peak = returns_curve.cummax()
    # Check for division-by-zero (1 + peak == 0)
    if (1 + peak).eq(0).any():
        raise ValueError("Cannot compute drawdown: peak value of -100% detected")

    drawdown = (returns_curve - peak) / (1 + peak)
    return -drawdown.min()  # make it positive dd value


def max_drawdown_from_equity(equity_balances=None) -> float:
    """
    Calculate max drawdown from a series of equity balances.

    Parameters:
    - equity_balances (list or pd.Series): Series of portfolio values (e.g., [1000, 1020, 980, ...]).

    Returns:
    - float: Max drawdown as a positive decimal (e.g., 0.05 for 5%).

    Raises:
    - ValueError: If equity_balances is None, empty, or contains zero/negative values.
    """
    if equity_balances is None or (
        isinstance(equity_balances, (list, pd.Series)) and len(equity_balances) == 0
    ):
        raise ValueError("equity_balances must be provided and non-empty")

    if not isinstance(equity_balances, pd.Series):
        equity_balances = pd.Series(equity_balances)

    # Check for zero or negative balances
    if (equity_balances <= 0).any():
        raise ValueError("equity_balances cannot contain zero or negative values")

    peak = equity_balances.cummax()
    drawdown = (equity_balances - peak) / peak
    return -drawdown.min()


def expectency(pl_series: pd.Series, wins: int, losses: int) -> float:
    wr = winrate(wins, losses)
    lr = 1 - wr
    avg_win, avg_loss, _, _ = avg_metrics(pl_series=pl_series)
    # Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss), where Avg Loss is positive
    expectency = (wr * avg_win) - (lr * avg_loss)
    return expectency


###################################################### i'm here
def durations(df: pd.DataFrame) -> tuple[float, float]:
    try:
        # Specify the expected format to avoid warnings
        df["entry_time"] = pd.to_datetime(
            df["entry_time"], format="%H:%M:%S", errors="coerce"
        )
        df["exit_time"] = pd.to_datetime(
            df["exit_time"], format="%H:%M:%S", errors="coerce"
        )
    except ValueError:
        return 0.0, 0.0

    # Check if all values are NaT (i.e., conversion failed for all rows)
    if df["entry_time"].isna().all() or df["exit_time"].isna().all():
        return 0.0, 0.0

    df["minutes"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60

    # Filter only the rows where 'outcome' is "WIN" and 'minutes' > 0
    only_wins = df[(df["minutes"] > 0) & (df["outcome"] == "WIN")]["minutes"]
    min_duration = only_wins.min() if not only_wins.empty else 0.0
    max_duration = only_wins.max() if not only_wins.empty else 0.0
    return float(min_duration), float(max_duration)


def consecutive_losses(df: pd.DataFrame) -> int:
    df_check(df, ["outcome"])
    if df["outcome"].empty:
        return 0

    current_streak = 0
    max_streak = 0
    for outcome in df["outcome"]:
        if outcome == "LOSS":
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak
