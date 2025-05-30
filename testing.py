import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from helpers.visualizations import *
from helpers.data_cleaning import *

url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQL7L-HMzezpuFCDOuS0wdUm81zbX4iVOokaFUGonVR1XkhS6CeDl1gHUrW4U0Le4zihfpqSDphTu4I/pub?gid=212787870&single=true&output=csv"
df = pd.read_csv(url)

pl = clean_numeric_series(df["pl_by_percentage"])
risk = clean_numeric_series(df["risk_by_percentage"])

plot_distribution(pl, title="PL Distribution", xlabel="P/L by %")
plot_distribution(risk, title="Risk Distribution", xlabel="Risk by %")
plt.show()
