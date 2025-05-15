import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helpers.data_cleaning import *
from helpers.data_preprocessing import *
from helpers.formulas import *
from helpers.utils import *
from helpers.visualizations import *


def max_drawdown_from_pct_returns(pct_returns=None, cumulative_equity=None) -> float:
    """
    Calculate max drawdown from a series of percentage raw returns or cumulative returns.

    Parameters:
    - percentage_returns (list or pd.Series): Percentage returns per trade (e.g., 0.01 = 1%)
    - cumulative_returns (list or pd.Series): Cumulative percentage_returns (same as percentage_returns)

    Returns:
    - float: Max drawdown as a positive decimal (e.g., 0.15 for 15%)

    Raises:
    - ValueError: If both or neither inputs are provided
    """
    if (pct_returns is None and cumulative_equity is None) or (
        pct_returns is not None and cumulative_equity is not None
    ):
        raise ValueError("Provide exactly one of pct_returns or cumulative_equity")

    if pct_returns is not None:
        if not isinstance(pct_returns, pd.Series):
            pct_returns = pd.Series(pct_returns)
        equity_curve = (1 + pct_returns).cumprod()
    else:
        if not isinstance(cumulative_equity, pd.Series):
            cumulative_equity = pd.Series(cumulative_equity)
        equity_curve = cumulative_equity

    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return -drawdown.min()


url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQL7L-HMzezpuFCDOuS0wdUm81zbX4iVOokaFUGonVR1XkhS6CeDl1gHUrW4U0Le4zihfpqSDphTu4I/pub?gid=212787870&single=true&output=csv"
# df = pd.read_csv("./data/TV_paper_test_data.csv")
df = pd.read_csv(url)

# risk_series = clean_numeric_series(df["risk_by_percentage"])
pl_series = clean_numeric_series(df["pl_by_percentage"])
# rr_series = clean_numeric_series(df["pl_by_rr"])

print(f"{max_drawdown(pl_series):.2f}%")
