import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from helpers.utils import *
from helpers.visualizations import *
from helpers.formulas import *
from helpers.data_preprocessing import *


url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQL7L-HMzezpuFCDOuS0wdUm81zbX4iVOokaFUGonVR1XkhS6CeDl1gHUrW4U0Le4zihfpqSDphTu4I/pub?gid=212787870&single=true&output=csv"
df = pd.read_csv("./data/TV_paper_test_data.csv")

s = df["Time"]
print(datetime_to_time(s))
# print(df.head())
