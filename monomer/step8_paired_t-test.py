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
import scipy.stats as stats

csv_path = "./monomer_data/raw/"  # read raw csv files
csv_path = "./monomer_data/raw_data/EU/"

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

# measures is all the columns
measures = data_whole.columns
print(measures)


# TODO
# let's keep it as it is for now
# this part will convert format like T1270TS274_2-D2 to T1270-D2TS274_2
use_EU = True
if use_EU:
    data_whole.index = data_whole.index.str.replace(
        r'(\w+)(TS\d+_\d+)-(\w+)', r'\1-\3\2')
data_whole.index = data_whole.index.str.extract(
    r'(?P<Level1>.+TS)(?P<Level2>\d+)_(?P<Level3>\d+)')
print(data_whole.head())
data_whole.index = pd.MultiIndex.from_tuples(
    data_whole.index, names=['target', 'group', 'submission_id'])
print(data_whole.head())
measure_of_interest = "GDT_TS"
data_whole = data_whole[measure_of_interest]
data_whole = pd.DataFrame(data_whole)
data_whole = data_whole.stack().unstack('group')
# add submission_id to the index
# print the keys of data_whole
data_whole.index = [f'{target}_{submission_id}' for target,
                    submission_id, measure in data_whole.index]

# get a new column called submission_id and target
data_whole['target'] = data_whole.index.str.split('_').str[0]
data_whole['submission_id'] = data_whole.index.str.split('_').str[1]
# remove any submission_id = 6
data_whole = data_whole[data_whole['submission_id'] != '6']
#  for each target, fill the missing values with the mean of the target
#  if the whole group is missing, fill with 0
data_whole_by_target = data_whole.groupby('target')


def fill_missing(group):
    # 计算组内每列的均值（不包括 target 和 submission_id 列）
    means = group.drop(columns=['submission_id']).mean()

    # 用均值填补缺失值
    filled_group = group.fillna(means)

    # 如果整组数据都是缺失值，则用0填补
    filled_group = filled_group.fillna(0)

    return filled_group


data_whole_by_target = data_whole_by_target.apply(
    fill_missing).reset_index(drop=True)


# TODO
# these two lines are also working
# grouped_means = data_whole.groupby('target').transform('mean')
# data_whole_by_target = data_whole.fillna(grouped_means)

print(data_whole_by_target.head())
data_whole_by_target.to_csv('./tmp/test.csv')
wanted_group = ["052", "022", "456", "051",
                "319", "287", "208", "028", "019", "294", "465", "110", "345", "139"]

# now we have a good data shape we can run the paired t-test

points = {}
for group_1 in wanted_group:
    for group_2 in wanted_group:
        if group_1 == group_2:
            continue
        data_1 = data_whole_by_target[group_1]
        data_1 = pd.DataFrame(data_1)
        data_2 = data_whole_by_target[group_2]
        data_2 = pd.DataFrame(data_2)
        # run the paired t-test
        t_stat, p_val = stats.ttest_rel(data_1, data_2)
        print(
            f"Group {group_1} vs Group {group_2}: t_stat={t_stat}, p_val={p_val}")
        if group_1 not in points:
            points[group_1] = 0
        if t_stat > 0 and p_val/2 < 0.01:
            points[group_1] += 1

# sort the points
points = dict(sorted(points.items(), key=lambda x: x[1], reverse=True))
print(points)
sys.exit(0)

data_whole['prefix'] = data_whole.index.str.split('_').str[0]
data_whole['group'] = data_whole.index.str.split('_').str[1]
print(data_whole.shape)
print(data_whole.head())

data_whole_grouped = data_whole.groupby('prefix').min()
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
        end = "domain"
    else:
        end = "whole"
    data_whole_mean.to_csv(out_path +
                           './individual_score_raw-{}-{}.csv'.format(measure_of_interest, end))

    # save the data
    # normalize the data with the z-score again
    data_whole_mean = (data_whole_mean - data_whole_mean.mean()
                       ) / data_whole_mean.std()
    # fill nan with 0
    data_whole_mean.fillna(0, inplace=True)

    data_whole_mean.to_csv(out_path +
                           './individual_score-{}-{}.csv'.format(measure_of_interest, end))
