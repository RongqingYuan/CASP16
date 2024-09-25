import argparse
import os
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys


def bootstrap_t_test(measures, model, mode,
                     score_path, output_path="./bootstrap_EU/",
                     impute_value=-2, p_value_threshold=0.05, weight=None, bootstrap_rounds=1000, top_n=25):
    # measure_type = "CASP16"
    if isinstance(measures, str):
        if measures == "CASP15":
            measures = ['LDDT', 'CAD_AA', 'SphGr',
                        'MP_clash', 'RMS_CA',
                        'GDT_HA', 'QSE', 'reLLG_const']
            measure_type = "CASP15"
        elif measures == "CASP16":
            measures = ['GDT_HA', 'GDC_SC', 'AL0_P', 'SphGr',
                        'CAD_AA', 'QSE', 'MolPrb_Score', 'reLLG_const']
            measure_type = "CASP16"
        else:
            print("measures should be a list of strings, or 'CASP15' / 'CASP16'")
            return 1
    else:
        measure_type = "custom"
    try:
        measures = list(measures)
    except:
        raise ValueError("measures must be iterable")
    if weight is None:
        weight = [1/len(measures)] * len(measures)
    equal_weight = len(set(weight)) == 1
    assert len(measures) == len(weight)
    score_path = score_path + f"impute={impute_value}/"
    T1_data = pd.DataFrame()
    groups = None
    for i in range(len(measures)):
        measure = measures[i]
        score_file = f"group_by_target-{measure}-{model}-{mode}.csv"
        score_matrix = pd.read_csv(score_path + score_file, index_col=0)
        score_matrix = score_matrix.filter(regex='T1')
        score_matrix = score_matrix.reindex(
            sorted(score_matrix.columns), axis=1)
        score_matrix = score_matrix.T
        score_matrix.columns = score_matrix.columns.str.replace(
            "TS", "")
        # rename each column to include the measure name
        groups = score_matrix.columns
        score_matrix.columns = [
            f"{col}-{measure}" for col in score_matrix.columns]
        T1_data = pd.concat([T1_data, score_matrix], axis=1)
    missing_values = T1_data.isnull().sum().sum()
    if missing_values > 0:
        raise ValueError(
            "There are missing values in the T1_data, please fill them before running the bootstrap_t_test function")
    # in this senario there just should not be any missing values

    if mode == "all":
        pass
    elif mode == "easy":
        T1_data = T1_data[easy_group]
    elif mode == "medium":
        T1_data = T1_data[medium_group]
    elif mode == "hard":
        T1_data = T1_data[hard_group]

    # # remove columns with more than 80% missing values
    # T1_data = T1_data.loc[:, T1_data.isnull().mean() < 0.8]
    if "raw" in score_file:
        T1_data.fillna(50, inplace=True)
    else:
        T1_data.fillna(-2, inplace=True)
    length = len(groups)
    points = {}

    measure_points_dict = {}
    for measure in measures:
        measure_points = {}
        for i in range(length):
            group_i = groups[i]
            measure_points[group_i] = 0
        for i in range(length):
            for j in range(length):
                group_1_id = groups[i]
                group_2_id = groups[j]
                if group_1_id == group_2_id:
                    continue
                group_1 = group_1_id + "-" + measure
                group_2 = group_2_id + "-" + measure
                group_1_data = T1_data[group_1]
                group_2_data = T1_data[group_2]
                t_stat, p_value = stats.ttest_rel(group_1_data, group_2_data)
                if t_stat > 0 and p_value/2 < p_value_threshold:
                    measure_points[group_1_id] += 1
        measure_points_dict[measure] = measure_points
        print(f"{measure} finished.")
    for measure in measures:
        measure_points = measure_points_dict[measure]
        for group in measure_points:
            if group not in points:
                points[group] = 0
            points[group] += measure_points[group] * \
                weight[measures.index(measure)]
    points = dict(sorted(points.items(), key=lambda x: x[1], reverse=True))
    dict_file = f"{measure_type}_{model}_{mode}_p={p_value_threshold}_equal_weight={equal_weight}_impute={impute_value}_ranking_t_test.txt"
    with open(output_path + dict_file, 'w') as f:
        f.write(str(points))
    # this is to get a re-ordered list of groups
    groups = list(points.keys())
    # can plot the points here
    plt.figure(figsize=(48, 24))
    bottom = [0 for i in range(length)]
    for key in measure_points_dict:
        points = []
        measure_points = measure_points_dict[key]
        for group in groups:
            points.append(measure_points[group])
        for i in range(length):
            points[i] = points[i] * weight[measures.index(key)]
        plt.bar(groups, points, bottom=bottom, label=key, width=0.8)
        bottom = [bottom[i] + points[i] for i in range(length)]
    plt.xticks(np.arange(length), groups, rotation=90, fontsize=24, ha='right')
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24)
    if equal_weight:
        plt.title(
            f"bootstrap result of t-test points for {measure_type} EUs with equal weight", fontsize=30)
        score_png_file = f"t_test_points_{measure_type}_{model}_{mode}_p={p_value_threshold}_impute={impute_value}_equal_weight.png"
        plt.savefig(output_path +
                    score_png_file,
                    dpi=300)
    else:
        plt.title(
            f"weighted bootstrap result of t-test points for {measure_type} EUs", fontsize=30)
        score_png_file = f"t_test_points_{measure_type}_{model}_{mode}_p={p_value_threshold}_impute={impute_value}_custom_weight.png"

        plt.savefig(output_path + score_png_file,
                    dpi=300)

    points = {}
    length = len(groups)
    win_matrix = [[0 for i in range(length)] for j in range(length)]
    # get the target list, it is the first element when split T1_data.index
    targets = T1_data.index.map(lambda x: x.split("-")[0])
    T1_data["target"] = targets
    for r in range(bootstrap_rounds):
        # T1_data_bootstrap = T1_data.sample(frac=1, replace=True)
        # T1_data_bootstrap = T1_data.groupby('target', group_keys=False).apply(lambda x: x.sample(n=1))
        grouped = T1_data.groupby('target')
        T1_data_bootstrap = grouped.apply(lambda x: x.sample(
            n=1)).sample(n=len(grouped), replace=True)
        T1_data_bootstrap = T1_data_bootstrap.sort_index()
        bootstrap_points = {}
        for measure in measures:
            measure_points = {}
            for i in range(length):
                group_i = groups[i]
                measure_points[group_i] = 0
            for i in range(length):
                for j in range(length):
                    group_1_id = groups[i]
                    group_2_id = groups[j]
                    if group_1_id == group_2_id:
                        continue
                    group_1 = group_1_id + "-" + measure
                    group_2 = group_2_id + "-" + measure
                    group_1_data = T1_data_bootstrap[group_1]
                    group_2_data = T1_data_bootstrap[group_2]
                    t_stat, p_value = stats.ttest_rel(
                        group_1_data, group_2_data)
                    if t_stat > 0 and p_value/2 < p_value_threshold:
                        measure_points[group_1_id] += 1
            measure_points_dict[measure] = measure_points
        for measure in measures:
            measure_points = measure_points_dict[measure]
            for group in measure_points:
                if group not in bootstrap_points:
                    bootstrap_points[group] = 0
                bootstrap_points[group] += measure_points[group] * \
                    weight[measures.index(measure)]
        for i in range(length):
            for j in range(length):
                if i == j:
                    continue
                if bootstrap_points[groups[i]] > bootstrap_points[groups[j]]:
                    win_matrix[i][j] += 1

        print("Round: {}".format(r))

    plt.figure(figsize=(30, 25))
    ax = sns.heatmap(win_matrix, annot=False,
                     cmap='Greys', cbar=True, square=True, )
    #  linewidths=1, linecolor='black')
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
    ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment='center')
    ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center')
    plt.xticks(np.arange(length), groups, rotation=45, fontsize=10, ha='right')
    plt.yticks(np.arange(length), groups, rotation=0, fontsize=10)
    plt.title(
        "{} bootstrap result of t-test points for {} EUs".format(measure_type, mode), fontsize=20)
    png_file = f"win_matrix_{measure_type}_{model}_{mode}_p={p_value_threshold}_n={bootstrap_rounds}_equal_weight={equal_weight}_impute={impute_value}_bootstrap_t_test.png"
    plt.savefig(output_path + png_file,
                dpi=300)
    npy_file = f"win_matrix_{measure_type}_{model}_{mode}_p={p_value_threshold}_n={bootstrap_rounds}_equal_weight={equal_weight}_impute={impute_value}_bootstrap_t_test.npy"
    np.save(output_path + npy_file,
            win_matrix)

    top_n = 25
    top_n_id = groups[:top_n]
    win_matrix = np.array(win_matrix)
    win_matrix_top_n = win_matrix[:top_n, :top_n]

    win_matrix_top_n = win_matrix_top_n / bootstrap_rounds
    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(win_matrix_top_n, annot=True, fmt=".2f",
                     cmap='Greys', cbar=True, square=True,
                     #  linewidths=1, linecolor='black',
                     )

    for text in ax.texts:
        value = float(text.get_text())  # 获取注释的值
        if value >= 0.95:
            text.set_color('red')  # >0.95 的文字为红色
        elif value < 0.95 and value >= 0.75:
            text.set_color('white')  # 0.75-0.95 的文字为白色
        else:
            text.set_color('black')  # 其他文字为黑色
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, float(1/4), float(1/2),
                    float(3/4), 1])
    cbar.set_ticklabels([0, float(1/4), float(1/2),
                        float(3/4), 1])
    cbar.ax.tick_params(labelsize=12)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
    ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment='center')
    ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center')
    plt.xticks(np.arange(top_n), top_n_id, rotation=45, fontsize=12)
    plt.yticks(np.arange(top_n), top_n_id, rotation=0, fontsize=12)
    if equal_weight:
        plt.title("{} t-test bootstrap result for {} EUs top {} groups with equal weight".format(
            measure_type, mode, top_n), fontsize=18)
    else:
        plt.title("{} t-test bootstrap result for {} EUs top {} groups with custom weight".format(
            measure_type, mode, top_n), fontsize=18)

    fig_file = f"win_matrix_{measure_type}_{model}_{mode}_p={p_value_threshold}_n={bootstrap_rounds}_equal_weight={equal_weight}_impute={impute_value}_top_{top_n}_bootstrap_t_test.png"
    plt.savefig(output_path + fig_file, dpi=300)


hard_group = [
    "T1207-D1",
    "T1210-D1",
    "T1210-D2",
    "T1220s1-D1",
    "T1220s1-D2",
    "T1226-D1",
    "T1228v1-D3",
    "T1228v1-D4",
    "T1228v2-D3",
    "T1228v2-D4",
    "T1239v1-D4",
    "T1239v2-D4",
    "T1271s1-D1",
    "T1271s3-D1",
    "T1271s8-D1",
]

medium_group = [
    "T1210-D3",
    "T1212-D1",
    "T1218-D1",
    "T1218-D2",
    "T1227s1-D1",
    "T1228v1-D1",
    "T1228v2-D1",
    "T1230s1-D1",
    "T1237-D1",
    "T1239v1-D1",
    "T1239v1-D3",
    "T1239v2-D1",
    "T1239v2-D3",
    "T1243-D1",
    "T1244s1-D1",
    "T1244s2-D1",
    "T1245s2-D1",
    "T1249v1-D1",
    "T1257-D3",
    "T1266-D1",
    "T1270-D1",
    "T1270-D2",
    "T1267s1-D1",
    "T1267s1-D2",
    "T1267s2-D1",
    "T1269-D1",
    "T1269-D2",
    "T1269-D3",
    "T1271s2-D1",
    "T1271s4-D1",
    "T1271s5-D1",
    "T1271s5-D2",
    "T1271s7-D1",
    "T1271s8-D2",
    "T1272s2-D1",
    "T1272s6-D1",
    "T1272s8-D1",
    "T1272s9-D1",
    "T1279-D2",
    "T1284-D1",
    "T1295-D1",
    "T1295-D3",
    "T1298-D1",
    "T1298-D2",
]

easy_group = [
    "T1201-D1",
    "T1201-D2",
    "T1206-D1",
    "T1208s1-D1",
    "T1208s2-D1",
    "T1218-D3",
    "T1228v1-D2",
    "T1228v2-D2",
    "T1231-D1",
    "T1234-D1",
    "T1235-D1",
    "T1239v1-D2",
    "T1239v2-D2",
    "T1240-D1",
    "T1240-D2",
    "T1245s1-D1",
    "T1246-D1",
    "T1257-D1",
    "T1257-D2",
    "T1259-D1",
    "T1271s6-D1",
    "T1274-D1",
    "T1276-D1",
    "T1278-D1",
    "T1279-D1",
    "T1280-D1",
    "T1292-D1",
    "T1294v1-D1",
    "T1294v2-D1",
    "T1295-D2",
    # "T1214",
]


# # 假设 T1_data 中有一列 'target_column' 用于分层

# def stratified_sample(df, stratify_column, frac=1, replace=True):
#     # 按照 stratify_column 的值进行分组，然后在每个组内进行采样
#     return df.groupby(stratify_column, group_keys=False).apply(
#         lambda x: x.sample(frac=frac, replace=replace)
#     )


# # 进行分层采样
# # T1_data_bootstrap = stratified_sample(T1_data, stratify_column='target', frac=1, replace=True)


# measure = "RMSD[L]"
# measure = "RMS_CA"
# measure = "GDT_HA"
# measure = "GDT_TS"

# score_path = "./group_by_target_EU/"
# mode = "hard"
# mode = "medium"
# mode = "easy"
# mode = "all"
# model = "first"
# model = "best"
# bootstrap_rounds = 1000

# measures = ['GDT_TS', 'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
#             'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
#             'SphGr',
#             'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
#             'MP_rotout', 'MP_ramout', 'MP_ramfv', 'reLLG_lddt', 'reLLG_const']


CASP15_measures = ['LDDT', 'CAD_AA', 'SphGr',
                   'MP_clash', 'RMS_CA',
                   'GDT_HA', 'QSE', 'reLLG_const']

CASP16_measures = ['GDT_HA', 'GDC_SC', 'AL0_P', 'SphGr',
                   'CAD_AA', 'QSE', 'MolPrb_Score', 'reLLG_const']
impute_value = 0
equal_weight = True
equal_weight = False
if equal_weight:
    weights = [1/8] * 8
else:
    # weights = [1/16, 1/16, 1/16,
    #            1/12, 1/12,
    #            1/4, 1/4, 1/4]

    weight = [1/4, 1/8,
              1/8, 1/8,
              1/8,
              1/8, 1/16,
              1/4]

parser = argparse.ArgumentParser(
    description="options for bootstrapping sum of z-scores")
parser.add_argument("--score_path", type=str, default="./group_by_target_EU/")
parser.add_argument("--measures", type=str, default="CASP16")
parser.add_argument("--model", type=str, default="best")
parser.add_argument("--mode", type=str, default="all")
parser.add_argument("--output_path", type=str, default="./bootstrap_EU/")
parser.add_argument("--impute_value", type=int, default=-2)
parser.add_argument("--p_value_threshold", type=float, default=0.05)
parser.add_argument("--weight", type=float, nargs='+', default=None)
parser.add_argument("--bootstrap_rounds", type=int, default=1000)
parser.add_argument("--equal_weight", action="store_true")
parser.add_argument("--top_n", type=int, default=25)
args = parser.parse_args()

score_path = args.score_path
measures = args.measures
model = args.model
mode = args.mode
output_path = args.output_path
impute_value = args.impute_value
p_value_threshold = args.p_value_threshold
weight = args.weight
bootstrap_rounds = args.bootstrap_rounds
equal_weight = args.equal_weight
top_n = args.top_n

if equal_weight:
    weight = [1/8] * 8
else:
    weight = [1/4, 1/8,
              1/8, 1/8,
              1/8,
              1/8, 1/16,
              1/4]
if not os.path.exists(output_path):
    os.makedirs(output_path)
bootstrap_t_test(measures, model, mode,
                 score_path, output_path=output_path,
                 impute_value=impute_value, p_value_threshold=p_value_threshold, weight=weight, bootstrap_rounds=bootstrap_rounds, top_n=top_n)
