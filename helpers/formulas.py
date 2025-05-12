import numpy as np
import pandas as pd
from helpers.utils import df_check, strict_percentage_convert

# Handle all possible ways
def pl_raw(df: pd.DataFrame) -> pd.Series:
    """Converts profit/loss percentages to a float Series, handling strings and numeric values."""
    df_check(df, ["pl_by_percentage"])
    if df["pl_by_percentage"].empty:
        return pd.Series(dtype=float)

    pl_series = df["pl_by_percentage"].apply(strict_percentage_convert)
    return pd.Series(pl_series, dtype=float)


def risk_raw(df: pd.DataFrame) -> pd.Series:
    """Converts risk percentages to a float Series, handling strings and numeric values."""
    df_check(df, ["risk_by_percentage"])
    if df["risk_by_percentage"].empty:
        return pd.Series(dtype=float)

    risk_series = df["risk_by_percentage"].apply(strict_percentage_convert)
    return pd.Series(risk_series, dtype=float)


def winrate(df: pd.DataFrame) -> tuple[float, float]:
    df_check(df, ["outcome"])
    if df["outcome"].empty:
        return 0.0, 0.0

    outcomes = df["outcome"]
    wins = (outcomes == "WIN").sum()
    losses = (outcomes == "LOSS").sum()

    wr = (wins / (wins + losses)) if (wins + losses) > 0 else 0.0
    wr_with_be = (wins / (len(df["outcome"]))) if (wins + losses) > 0 else 0.0

    return wr, wr_with_be


def winning_trades(df: pd.DataFrame) -> float:
    df_check(df, ["outcome"])
    if df["outcome"].empty:
        return 0.0
    outcomes = df["outcome"]
    wins = (outcomes == "WIN").sum()
    return wins


def breakevens_trades(df: pd.DataFrame) -> float:
    df_check(df, ["outcome"])
    if df["outcome"].empty:
        return 0.0
    outcomes = df["outcome"]
    be = (outcomes == "BE").sum()
    return be


def lossing_trades(df: pd.DataFrame) -> float:
    df_check(df, ["outcome"])
    if df["outcome"].empty:
        return 0.0
    outcomes = df["outcome"]
    losses = (outcomes == "LOSS").sum()
    return losses


def avg_wl(df: pd.DataFrame) -> tuple[float, float]:
    df_check(df, ["pl_by_percentage"])
    if df["pl_by_percentage"].empty:
        return 0.0, 0.0

    pl_series = pl_raw(df)
    avg_win = pl_series[pl_series > 0].mean()
    avg_loss = abs(pl_series[pl_series < 0].mean())  # <-- Make loss positive here

    avg_win = 0.0 if pd.isna(avg_win) else avg_win
    avg_loss = 0.0 if pd.isna(avg_loss) else avg_loss

    return float(avg_win), float(avg_loss)


def avg_risk(df: pd.DataFrame) -> float:
    df_check(df, ["risk_by_percentage"])
    if df["risk_by_percentage"].empty:
        return 0.0
    risk_series = risk_raw(df)
    avg_risk = risk_series.mean() or 0.0
    return float(avg_risk)


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
