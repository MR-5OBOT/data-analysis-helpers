import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from helpers.formulas import *
from helpers.utils import *
from helpers.visualizations import *
from helpers.data_preprocessing import *
from helpers.data_cleaning import *


url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQL7L-HMzezpuFCDOuS0wdUm81zbX4iVOokaFUGonVR1XkhS6CeDl1gHUrW4U0Le4zihfpqSDphTu4I/pub?gid=212787870&single=true&output=csv"
# df = pd.read_csv("./data/TV_paper_test_data.csv")
df = pd.read_csv(url)

risk_series = clean_numeric_series(df["risk_by_percentage"])
pl_series = clean_numeric_series(df["pl_by_percentage"])
rr_series = clean_numeric_series(df["pl_by_rr"])
print(f"AVG WIN: {avg_metrics(pl_series=pl_series)[0] * 100:.2f}%")
print(f"AVG LOSS: {avg_metrics(pl_series=pl_series)[1] * 100:.2f}%")
print(f"AVG RISK: {avg_metrics(risk_series=risk_series)[2] * 100:.2f}%")
print(f"AVG RR: {avg_metrics(rr_series=rr_series)[3]:.2f}")
