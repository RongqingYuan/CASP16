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

# read all data and concatenate them into one big dataframe
data_whole = pd.DataFrame()
data_domain = pd.DataFrame()
for csv_file in csv_list:
    print("Processing {}".format(csv_file))
    data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
    print(data_tmp.shape)
    if data_tmp.shape[1] == 35:
        print("something wrong with {}".format(csv_file))
        sys.exit(0)
    if len(csv_file.split("-")) == 2:
        data_domain = pd.concat([data_domain, data_tmp], axis=0)
    elif len(csv_file.split("-")) == 1:
        data_whole = pd.concat([data_whole, data_tmp], axis=0)
    else:
        print("something wrong with {}".format(csv_file))
        sys.exit(0)
# print the first 5 rows
print(data_whole.head())
print(data_whole.shape)
# remove the "GR#" column and "#" column
data_whole = data_whole.drop(["GR#", "#"], axis=1)
# "DipDiff", "BBscore", "SCscore" does not contribute to the prediction, so we remove them as well
data_whole = data_whole.drop(["DipDiff", "BBscore", "SCscore"], axis=1)
# remove the "RANK" column
data_whole = data_whole.drop(["RANK"], axis=1)
# drop these 3 columns: NP_P, NP, err
data_whole = data_whole.drop(["NP_P", "NP", "err"], axis=1)

# fill the N/A values with nan
# fill the "-" values with nan
data_whole.replace("N/A", np.nan, inplace=True)
data_whole.replace("-", np.nan, inplace=True)
data_whole = data_whole.astype(float)

# some columns need to be taken inverse because in this case,
# smaller values are better, so we need to invert them for the sake of consistency
# MP_ramfv is not in the list. fv stands for favored and it is not in the list
inverse_columns = ["RMS_CA", "RMS_ALL", "RMSD[L]", "MolPrb_Score",
                   "FlexE", "MP_clash", "MP_rotout", "MP_ramout"]  # also we don't need err, err is removed previously
# take the negation of the columns
data_whole[inverse_columns] = -data_whole[inverse_columns]
data_whole = (data_whole - data_whole.mean()) / data_whole.std()
print(data_whole.shape)

data_whole = data_whole[((data_whole >= -2) | data_whole.isna()).all(axis=1)]

# save to csv_tmp path to see if anything goes wrong
# data_whole.to_csv(csv_tmp_path + monomer[:-4] + ".csv")

# after removing the outliers, we need to do z-score normalization again
data_whole = (data_whole - data_whole.mean()) / data_whole.std()
print(data_whole.shape)
