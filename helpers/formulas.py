import numpy as np
import pandas as pd
from helpers.utils import df_check


def winrate(wins: float, losses: float) -> float:
    total = wins + losses
    return wins / total if total > 0 else 0.0


def avg_metrics(
    pl_series: pd.Series = pd.Series(dtype=float),
    risk_series: pd.Series = pd.Series(dtype=float),
    rr_series: pd.Series = pd.Series(dtype=float)
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


def avg_rr(df: pd.DataFrame) -> float:
    df_check(df, ["pl_by_rr"])
    if df["pl_by_rr"].empty:
        return 0.0
    valid_data = df["pl_by_rr"].dropna()
    if valid_data.empty:
        return 0.0
    return float(valid_data.mean())


def best_worst_trade(df: pd.DataFrame) -> tuple[float, float]:
    df_check(df, ["pl_by_percentage"])
    if df["pl_by_percentage"].empty:
        return 0.0, 0.0

    pl_series = pl_raw(df)
    best_trade_value = pl_series.max() or 0.0
    worst_trade_value = pl_series.min() or 0.0
    return float(best_trade_value), float(worst_trade_value)


def max_drawdown(df: pd.DataFrame) -> float:
    """
    Calculate the maximum drawdown of a series of periodic returns.
    """
    pl_series = pl_raw(df)
    # 1) Build the wealth index
    wealth_index = (1 + pl_series).cumprod()
    # 2) Compute the running peak
    running_max = wealth_index.cummax()
    # 3) Compute drawdowns
    drawdown = (wealth_index - running_max) / running_max
    # 4) Return the worst (most negative) drawdown
    return drawdown.min()


def expectency(df: pd.DataFrame) -> float:
    df_check(df, ["outcome"])
    if df["outcome"].empty:
        return 0.0

    wins = (df["outcome"] == "WIN").sum()
    losses = (df["outcome"] == "LOSS").sum()
    wr = (wins / (wins + losses)) if (wins + losses) > 0 else 0.0
    lr = 1 - wr
    avg_w = avg_wl(df)[0]
    avg_l = avg_wl(df)[1]

    # Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss), where Avg Loss is positive
    expectency = (wr * avg_w) - (lr * avg_l)
    return expectency


def durations(df: pd.DataFrame) -> tuple[float, float]:
    # Calculates the min and max duration of winning trades
    if (
        df is None
        or df.empty
        or "entry_time" not in df
        or "exit_time" not in df
        or df["entry_time"].empty
        or df["exit_time"].empty
    ):
        return 0.0, 0.0

    try:
        # Specify the expected format to avoid warnings
        df["entry_time"] = pd.to_datetime(df["entry_time"], format="%H:%M:%S", errors="coerce")
        df["exit_time"] = pd.to_datetime(df["exit_time"], format="%H:%M:%S", errors="coerce")
    except ValueError:
        return 0.0, 0.0

    # Check if all values are NaT (i.e., conversion failed for all rows)
    if df["entry_time"].isna().all() or df["exit_time"].isna().all():
        return 0.0, 0.0

    df["duration_minutes"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 60

    # Filter only the rows where 'outcome' is "WIN" and 'duration_minutes' > 0
    only_wins = df[(df["duration_minutes"] > 0) & (df["outcome"] == "WIN")]["duration_minutes"]
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


def stats_table(df: pd.DataFrame) -> dict:
    """
    Returns a dictionary of statistics.
    """
    if df is None or df.empty:
        print("Warning: No data to process for statistics.")

    # Calculate metrics using the helper functions
    total_trades = len(df) if df is not None else 0
    pl_values = pl_raw(df)
    total_pl = pl_values.sum()


    table = {
        "Total Trades": total_trades,
        "Total P/L": total_pl,
        "Winrate": winrate(df)[0],
        "Winning Trades": winning_trades(df),
        "Losing Trades": lossing_trades(df),
        "Avg Win": avg_wl(df)[0],
        "Avg Loss": avg_wl(df)[1],
        "Risk": avg_risk(df),
        "Avg RR": avg_rr(df),
        "Best Trade": best_worst_trade(df)[0],
        "Worst Trade": best_worst_trade(df)[1],
        "Max Drawdown": max_drawdown(df),
        "Expectancy": expectency(df),
        "Min Duration (mins)": durations(df)[0],
        "Max Duration (mins)": durations(df)[1],
        "Max Consecutive Losses": consecutive_losses(df),
    }
    return table


def term_stats(stats: dict) -> None:
    """
    Prints the trading statistics from a dictionary to the terminal.
    """
    if not stats:
        print("No statistics available to display.")
        return

    print("\n--- Trading Statistics ---")
    for key, value in stats.items():
        print(f"{key:<20}: {value}")  # Use f-string formatting for alignment
    print("-------------------------\n")
    return
