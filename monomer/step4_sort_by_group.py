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
from copy import deepcopy, copy
import time

csv_raw_path = "./monomer_data_aug_30/raw/EU/"  # read raw csv files
csv_path = "./monomer_data_aug_30/processed/EU/"  # read raw csv files
if "whole" in csv_path:
    use_domain = False
else:
    use_domain = True
csv_list = [txt for txt in os.listdir(csv_path) if txt.endswith(".csv")]

out_path = "./by_target/"
if not os.path.exists(out_path):
    os.makedirs(out_path)
# read all data and concatenate them into one big dataframe
data_whole = pd.DataFrame()
data_domain = pd.DataFrame()
for csv_file in csv_list:
    print("Processing {}".format(csv_file))
    data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
    # print(data_tmp.shape)
    if data_tmp.shape[1] == 35:
        print("something wrong with {}".format(csv_file))
        sys.exit(0)
    # if len(csv_file.split("-")) == 2:
    #     data_domain = pd.concat([data_domain, data_tmp], axis=0)
    # elif len(csv_file.split("-")) == 1:
    #     data_whole = pd.concat([data_whole, data_tmp], axis=0)
    # else:
    #     print("something wrong with {}".format(csv_file))
    #     sys.exit(0)
    data_whole = pd.concat([data_whole, data_tmp], axis=0)


# # print the first 5 rows
# print(data_whole.head())
# print(data_whole.shape)
# # remove the "GR#" column and "#" column
# data_whole = data_whole.drop(["GR#", "#"], axis=1)
# # "DipDiff", "BBscore", "SCscore" does not contribute to the prediction, so we remove them as well
# data_whole = data_whole.drop(["DipDiff", "BBscore", "SCscore"], axis=1)
# # remove the "RANK" column
# data_whole = data_whole.drop(["RANK"], axis=1)
# # drop these 3 columns: NP_P, NP, err
# data_whole = data_whole.drop(["NP_P", "NP", "err"], axis=1)

# # fill the N/A values with nan
# # fill the "-" values with nan
# data_whole.replace("N/A", np.nan, inplace=True)
# data_whole.replace("-", np.nan, inplace=True)
# data_whole = data_whole.astype(float)

# # some columns need to be taken inverse because in this case,
# # smaller values are better, so we need to invert them for the sake of consistency
# # MP_ramfv is not in the list. fv stands for favored and it is not in the list
# inverse_columns = ["RMS_CA", "RMS_ALL", "RMSD[L]", "MolPrb_Score",
#                    "FlexE", "MP_clash", "MP_rotout", "MP_ramout"]  # also we don't need err, err is removed previously
# # take the negation of the columns
# data_whole[inverse_columns] = -data_whole[inverse_columns]

# measures is all the columns
measures = data_whole.columns
print(measures)


# measure_of_interest = "GDT_TS"
# measure_of_interest_1 = "RMS_CA"

# get the column for the measure of interests and measure of interests 1
# data_whole = data_whole[[measure_of_interest, measure_of_interest_1]]
if use_domain:
    data_whole.index = data_whole.index.str.replace(
        r'(\w+)(TS\d+_\d+)-(\w+)', r'\1-\3\2')

data_whole['prefix'] = data_whole.index.str.split('_').str[0]
data_whole['group'] = data_whole.index.str.split('_').str[1]
print(data_whole)
print(data_whole.shape)
data_whole_grouped = data_whole.groupby('prefix').max()
print(data_whole_grouped.shape)
print(data_whole_grouped.head())
print(measures)
time.sleep(5)
for measure_of_interest in measures:
    print(data_whole_grouped.head())
    print(data_whole_grouped.shape)
    data_whole_mean = pd.DataFrame(data_whole_grouped[measure_of_interest])
    print(data_whole_mean.shape)

    data_whole_mean.index = data_whole_mean.index.str.split('TS').map(tuple)
    print(len(data_whole_mean.index))
    data_whole_mean.index = pd.MultiIndex.from_tuples(
        data_whole_mean.index, names=['target', 'group'])
    data_whole_mean = data_whole_mean.stack().unstack('target')
    data_whole_mean.index = [f'{b}-{c}' for b, c in data_whole_mean.index]

    print(data_whole_mean.head())
    print(data_whole_mean.shape)

    if use_domain:
        end = "EU"
    else:
        end = "whole"
    data_whole_mean.to_csv(out_path +
                           './groups_by_targets_for-{}-{}.csv'.format(measure_of_interest, end))

    # # save the data
    # # normalize the data with the z-score again
    # data_whole_mean = (data_whole_mean - data_whole_mean.mean()
    #                    ) / data_whole_mean.std()
    # # fill nan with 0
    # data_whole_mean.fillna(0, inplace=True)

    # data_whole_mean.to_csv(out_path +
    #                        './individual_score_processed-{}-{}.csv'.format(measure_of_interest, end))
