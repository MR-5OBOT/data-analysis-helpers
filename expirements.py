import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helpers.data_cleaning import *
from helpers.data_preprocessing import *
from helpers.formulas import *
from helpers.utils import *
from helpers.visualizations import *


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


url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQL7L-HMzezpuFCDOuS0wdUm81zbX4iVOokaFUGonVR1XkhS6CeDl1gHUrW4U0Le4zihfpqSDphTu4I/pub?gid=212787870&single=true&output=csv"
# df = pd.read_csv("./data/TV_paper_test_data.csv")
df = pd.read_csv(url)

# risk_series = clean_numeric_series(df["risk_by_percentage"])
pl_series = clean_numeric_series(df["pl_by_percentage"])
# rr_series = clean_numeric_series(df["pl_by_rr"])

print(f"{max_drawdown_from_pct_returns(perTrade_returns=pl_series) * 100:.4f}%")
