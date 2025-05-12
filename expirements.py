import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from helpers.utils import *
from helpers.visualizations import *
from helpers.data_preprocessing import *
from helpers.data_cleaning import *


url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQL7L-HMzezpuFCDOuS0wdUm81zbX4iVOokaFUGonVR1XkhS6CeDl1gHUrW4U0Le4zihfpqSDphTu4I/pub?gid=212787870&single=true&output=csv"
# df = pd.read_csv("./data/TV_paper_test_data.csv")
df = pd.read_csv(url)

# pl = pl_raw(df)
# print(pl_raw)
# print(datetime_to_time(s))
# print(df.head())

test_inputs = df["pl_by_percentage"]

# Convert to a pandas Series
series = pd.Series(test_inputs)

# Clean the series
cleaned_series = clean_numeric_series(series)

# Print the cleaned series
print(cleaned_series)

