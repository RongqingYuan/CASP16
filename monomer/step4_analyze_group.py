import sys
import seaborn as sns
import matplotlib
import os
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
import factor_analyzer
from sklearn.preprocessing import MinMaxScaler
from sympy import *
from matplotlib import pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity

csv_path = "./csv_raw/"  # read raw csv files
csv_list = [txt for txt in os.listdir(csv_path) if txt.endswith(".csv")]

# csv_file = csv_path + csv_list[3]
# print("Processing {}".format(csv_file))
# data = pd.read_csv(csv_file, index_col=0)

# read all data and concatenate them into one big dataframe
data = pd.DataFrame()
for csv_file in csv_list:
    print("Processing {}".format(csv_file))
    data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
    print(data_tmp.shape)
    if data_tmp.shape[1] == 35:
        print("something wrong with {}".format(csv_file))
        sys.exit(0)
    data = pd.concat([data, data_tmp], axis=0)
# print the first 5 rows
print(data.head())
print(data.shape)
# remove the "GR#" column and "#" column
data = data.drop(["GR#", "#"], axis=1)
# "DipDiff", "BBscore", "SCscore" does not contribute to the prediction, so we remove them as well
data = data.drop(["DipDiff", "BBscore", "SCscore"], axis=1)
# remove the "RANK" column
data = data.drop(["RANK"], axis=1)
# drop these 3 columns: NP_P, NP, err
data = data.drop(["NP_P", "NP", "err"], axis=1)
