import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helpers.data_cleaning import *
from helpers.data_preprocessing import *
from helpers.formulas import *
from helpers.utils import *
from helpers.visualizations import *

url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQL7L-HMzezpuFCDOuS0wdUm81zbX4iVOokaFUGonVR1XkhS6CeDl1gHUrW4U0Le4zihfpqSDphTu4I/pub?gid=212787870&single=true&output=csv"
# df = pd.read_csv("./data/")
df = pd.read_csv(url)

risk_series = clean_numeric_series(df["risk_by_percentage"])
pl_series = clean_numeric_series(df["pl_by_percentage"])
rr_series = clean_numeric_series(df["pl_by_rr"])


wins = len(df[df["outcome"] == "WIN"])
losses = len(df[df["outcome"] == "LOSS"])


def expectency(pl_series: pd.Series, wins: int, losses: int) -> float:
    wr = winrate(wins, losses)
    lr = 1 - wr
    avg_win, avg_loss, _, _ = avg_metrics(pl_series=pl_series)
    # Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss), where Avg Loss is positive
    expectency = (wr * avg_win) - (lr * avg_loss)
    return expectency


print(f"{expectency(pl_series, wins, losses) * 100:.4f}%")
# print(f"{max_drawdown_from_pct_returns(perTrade_returns=pl_series) * 100:.4f}%")
