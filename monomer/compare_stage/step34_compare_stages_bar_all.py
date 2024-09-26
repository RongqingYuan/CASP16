import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import ward, fcluster
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform


measure = "GDT_TS"
measure = "GDT_HA"

# model = "first"
model = "best"

# mode = "easy"
# mode = "medium"
# mode = "hard"
mode = "all"
impute_value = -2

T1_score_path = "../group_by_target_EU/raw/"
T1_score_file = f"groups_by_targets_for-raw-{measure}-{model}-{mode}.csv"
T1_data = pd.read_csv(T1_score_path + T1_score_file, index_col=0)
T1_data = T1_data.filter(regex='T1')

all_score_path = "./group_by_target/raw/"
all_score_file = f"groups_by_targets_for-raw-{measure}-{model}-{mode}.csv"
data = pd.read_csv(all_score_path + all_score_file, index_col=0)
T0_data = data.filter(regex='T0')
T2_data = data.filter(regex='T2')

out_path = "./compare/"
if not os.path.exists(out_path):
    os.makedirs(out_path)
print(T0_data.shape)
print(T1_data.shape)
print(T2_data.shape)
# breakpoint()
T0_data = T0_data.T
T1_data = T1_data.T
T2_data = T2_data.T

stages = ["T0", "T1"]
stages = ["T1", "T2"]
top_n = 20
if stages == ["T0", "T1"]:
    # 提取xxx部分
    T0_data.index = T0_data.index.str.extract(r'T0([-\w]+)')[0]
    # there is a -D*. In order to match the "-" in the regex, we need to add "-" in the regex
    T1_data.index = T1_data.index.str.extract(r'T1([-\w]+)')[0]
    # then remove the "-all" in T1_data.index
    T1_data.index = T1_data.index.str.replace("-all", "")
    breakpoint()
    # 确保索引对齐
    common_index = T0_data.index.intersection(T1_data.index)
    df0_aligned = T0_data.loc[common_index]
    df1_aligned = T1_data.loc[common_index]
    # 计算差值
    df0_aligned = df0_aligned.T
    df1_aligned = df1_aligned.T

    targets = df0_aligned.columns
    diff_valid_numbers = []
    diff_valid_dict = {}
    for target in targets:
        df0_aligned_target = df0_aligned[target]
        df1_aligned_target = df1_aligned[target]
        # take the intersection of the two dataframes where both are not null
        # 布尔索引，选择两个列都不为 NaN 的行
        valid_rows = df0_aligned_target.notna() & df1_aligned_target.notna()
        # 取出交集部分
        df0_aligned_valid = df0_aligned_target[valid_rows]
        df1_aligned_valid = df1_aligned_target[valid_rows]
        # 计算差值
        # df_diff_valid = df1_aligned_valid - df0_aligned_valid
        # get top_n means for df0_aligned_valid and df1_aligned_valid
        df0_aligned_valid = list(df0_aligned_valid)
        df1_aligned_valid = list(df1_aligned_valid)
        diff_valid_number = len(df0_aligned_valid)
        df0_aligned_valid = df0_aligned_valid[:top_n]
        df1_aligned_valid = df1_aligned_valid[:top_n]
        df0_aligned_valid_mean = sum(
            df0_aligned_valid) / len(df0_aligned_valid)
        df1_aligned_valid_mean = sum(
            df1_aligned_valid) / len(df1_aligned_valid)
        diff_valid = df1_aligned_valid_mean - df0_aligned_valid_mean
        diff_valid_dict[target] = diff_valid
        diff_valid_numbers.append(diff_valid_number)
        # breakpoint()
    # sort the diff_valid_dict by value
    diff_valid_dict = dict(
        sorted(diff_valid_dict.items(), key=lambda item: item[1], reverse=True))
    print(diff_valid_dict)
    print(sorted(diff_valid_numbers, reverse=True))

    # plot the diff_valid_dict
    plt.figure(figsize=(16, 12))
    plt.bar(diff_valid_dict.keys(), diff_valid_dict.values())
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("{}, {} - {}".format(measure,
               stages[1], stages[0]), fontsize=20)
    # draw a horizontal line at y=0
    plt.axhline(0, color='black', linestyle='-')
    plt.title("Mean difference between {} of {} and {} for top {} of {} models".format(
        measure, stages[1], stages[0], top_n, model), fontsize=20)
    fig_file = f"T1_T0_diffs_{measure}_top_{top_n}_{model}.png"
    plt.savefig(out_path + fig_file, dpi=300)

elif stages == ["T1", "T2"]:
    # 提取xxx部分
    T1_data.index = T1_data.index.str.extract(r'T1([-\w]+)')[0]
    # there is a -D*. In order to match the "-" in the regex, we need to add "-" in the regex
    T2_data.index = T2_data.index.str.extract(r'T2([-\w]+)')[0]
    T1_data.index = T1_data.index.str.replace("-all", "")

    # 确保索引对齐
    common_index = T1_data.index.intersection(T2_data.index)
    df0_aligned = T1_data.loc[common_index]
    df1_aligned = T2_data.loc[common_index]
    # 计算差值
    df0_aligned = df0_aligned.T
    df1_aligned = df1_aligned.T

    targets = df0_aligned.columns
    diff_valid_numbers = []
    diff_valid_dict = {}
    for target in targets:
        df0_aligned_target = df0_aligned[target]
        df1_aligned_target = df1_aligned[target]
        # take the intersection of the two dataframes where both are not null
        # 布尔索引，选择两个列都不为 NaN 的行
        valid_rows = df0_aligned_target.notna() & df1_aligned_target.notna()
        # 取出交集部分
        df0_aligned_valid = df0_aligned_target[valid_rows]
        df1_aligned_valid = df1_aligned_target[valid_rows]
        # 计算差值
        # df_diff_valid = df1_aligned_valid - df0_aligned_valid
        # get top_n means for df0_aligned_valid and df1_aligned_valid
        df0_aligned_valid = list(df0_aligned_valid)
        df1_aligned_valid = list(df1_aligned_valid)
        diff_valid_number = len(df0_aligned_valid)
        df0_aligned_valid = df0_aligned_valid[:top_n]
        df1_aligned_valid = df1_aligned_valid[:top_n]
        df0_aligned_valid_mean = sum(
            df0_aligned_valid) / len(df0_aligned_valid)
        df1_aligned_valid_mean = sum(
            df1_aligned_valid) / len(df1_aligned_valid)
        diff_valid = df1_aligned_valid_mean - df0_aligned_valid_mean
        diff_valid_dict[target] = diff_valid
        diff_valid_numbers.append(diff_valid_number)
        # breakpoint()
    # sort the diff_valid_dict by value
    diff_valid_dict = dict(
        sorted(diff_valid_dict.items(), key=lambda item: item[1], reverse=True))
    print(diff_valid_dict)
    print(sorted(diff_valid_numbers, reverse=True))
    # plot the diff_valid_dict
    plt.figure(figsize=(24, 12))
    plt.bar(diff_valid_dict.keys(), diff_valid_dict.values())
    plt.ylabel("{} of {} - {}".format(measure,
                                      stages[1], stages[0]), fontsize=20)
    # draw a horizontal line at y=0
    plt.axhline(0, color='black', linestyle='-')
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Mean difference between {} of {} and {} for top {} of {} models".format(
        measure, stages[1], stages[0], top_n, model), fontsize=20)
    fig_file = f"T2_T1_diffs_{measure}_top_{top_n}_{model}.png"
    plt.savefig(out_path + fig_file, dpi=300)
