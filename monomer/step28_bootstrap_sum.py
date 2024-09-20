import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys


score_path = "./group_by_target_EU/"
measure = "RMSD[L]"
measure = "RMS_CA"
measure = "GDT_HA"
measure = "GDT_TS"
# measure = sys.argv[1]

mode = "hard"
mode = "medium"
mode = "easy"
mode = "all"
model = "first"
model = "best"

measures = ['GDT_TS', 'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
            'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
            'SphGr',
            'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
            'MP_rotout', 'MP_ramout', 'MP_ramfv', 'reLLG_lddt', 'reLLG_const']
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


wanted_measures = ['LDDT', 'CAD_AA', 'SphGr',
                   'MP_clash', 'RMS_CA',
                   'GDT_HA', 'QSE', 'reLLG_const']

wanted_measures = ['GDT_HA', 'GDC_SC', 'AL0_P', 'SphGr',
                   'CAD_AA', 'QSE', 'MolPrb_Score', 'reLLG_const']
equal_weight = False
equal_weight = True
if equal_weight:
    weights = [1/8] * 8
else:
    weights = [1/16, 1/16, 1/16,
               1/12, 1/12,
               1/4, 1/4, 1/4]
bootstrap_rounds = 1000


def bootstrap_sum(measures, model, mode,
                  score_path, output_path="./bootstrap_EU/",
                  weight=None, bootstrap_rounds=1000):
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
    measures = list(measures)
    if weight is None:
        weight = [1/len(measures)] * len(measures)
    # if all weight are the same, then equal_weight is True
    equal_weight = len(set(weight)) == 1
    assert len(measures) == len(weight)
    # score_file = "groups_by_targets_for-raw-{}-EU.csv".format(measure)
    # score_path = "/home2/s439906/project/CASP16/monomer/by_EU/"
    # score_file = "sum_CASP15_score-{}-{}-equal-weight-False.csv".format(
    #     model, mode)
    measure = measures[0]
    score_file = "group_by_target-{}-{}-{}.csv".format(
        measure, model, mode)
    data_tmp = pd.read_csv(score_path + score_file, index_col=0)
    data_columns = data_tmp.columns

    target_count = {}
    for EU in data_columns:
        target = EU.split("-")[0]
        if target not in target_count:
            target_count[target] = 0
        target_count[target] += 1
    target_weight = {key: 1/value for key, value in target_count.items()}
    EU_weight = {EU: target_weight[EU.split("-")[0]]
                 for EU in data_columns}
    EU_weight = pd.Series(EU_weight)
    # breakpoint()
    data = pd.DataFrame()
    measure_score_dict = {}
    for i in range(len(measures)):
        measure = measures[i]
        score_file = "group_by_target-{}-{}-{}.csv".format(
            measure, model, mode)
        score_matrix = pd.read_csv(score_path + score_file, index_col=0)
        score_matrix = score_matrix.reindex(
            sorted(score_matrix.columns), axis=1)
        weight_i = weight[i]
        # multiply the score_matrix by the weight
        score_matrix = score_matrix * weight_i
        score_matrix = score_matrix * EU_weight
        score_matrix_sum = score_matrix.sum(axis=1)
        measure_score_dict[measure] = dict(score_matrix_sum)
        data = pd.concat([data, score_matrix], axis=0)

    data = data.T
    columns = data.columns
    grouped_columns = data.groupby(columns, axis=1)
    grouped_data = grouped_columns.sum()
    groups = grouped_data.columns
    length = len(groups)

    #####
    sum_scores = {}
    for measure in measures:
        measure_sum_score = pd.Series(measure_score_dict[measure])
        for group in groups:
            if group not in sum_scores:
                sum_scores[group] = 0
            sum_scores[group] += measure_sum_score[group]
    sum_scores = dict(
        sorted(sum_scores.items(), key=lambda x: x[1], reverse=True))

    groups = list(sum_scores.keys())
    groups_plt = [group[2:] for group in groups]

    plt.figure(figsize=(30, 15))
    bottom = [0 for i in range(length)]
    for key in measure_score_dict:
        measure_points = measure_score_dict[key]
        points = [measure_points[group] for group in groups]
        plt.bar(groups_plt, points, bottom=bottom, label=key, width=0.8)
        bottom = [bottom[i] + points[i] for i in range(length)]
    plt.xticks(np.arange(length), groups, rotation=45, fontsize=10, ha='right')
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    if equal_weight:
        plt.title(
            f"Bootstrap result of t-test points for {measure_type} EUs with equal weight", fontsize=18)
        plt.savefig(output_path +
                    f"sum_points_{measure_type}_{model}_{mode}_equal_weight.png",
                    dpi=300)
    else:
        plt.title(
            f"Bootstrap result of t-test points for {measure_type} EUs with custom weight", fontsize=18)
        plt.savefig(output_path +
                    f"sum_points_{measure_type}_{model}_{mode}_custom_weight.png",
                    dpi=300)
    #####
    # use the above code to get a initial ranking of the groups.
    # then generate new groups list using the ranking
    # then do bootstrapping

    # sum each column
    sum = grouped_data.sum(axis=0)
    # convert the sum to a dictionary and sort it
    scores = dict(sum)
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    with open(output_path + "{}_{}_{}_n={}_equal_weight={}_ranking_sum.txt".format(
            measure_type, model, mode,  bootstrap_rounds, equal_weight), 'w') as f:
        f.write(str(scores))
    groups = list(scores.keys())
    length = len(groups)
    win_matrix = [[0 for i in range(length)] for j in range(length)]
    # get the target list, it is the first element when split T1_data.index
    targets = grouped_data.index.map(lambda x: x.split("-")[0])
    grouped_data["target"] = targets

    # breakpoint()

    # now do the bootstrapping
    for r in range(bootstrap_rounds):
        grouped = grouped_data.groupby('target')
        data_bootstrap = grouped.apply(lambda x: x.sample(
            n=1)).sample(n=len(grouped), replace=True)
        data_bootstrap = data_bootstrap.sort_index()  # not necessary,just to look nice
        data_bootstrap = data_bootstrap.drop(columns='target')
        # breakpoint()
        sum = data_bootstrap.sum(axis=0)
        scores = dict(sum)
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        # bootstrap_points = {}
        for i in range(length):
            for j in range(length):
                if i == j:
                    continue
                if scores[groups[i]] > scores[groups[j]]:
                    win_matrix[i][j] += 1
        print("Round: {}".format(r))
    # breakpoint()
    # plot the win matrix as heatmap using seaborn
    # only the color should be plotted, do not show the numbers

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
        "{} bootstrap result of summ points for {} EUs".format(measure_type, mode), fontsize=20)
    plt.savefig(output_path +
                "win_matrix_{}_{}_{}_n={}_sum_equal_weight={}_bootstrap_sum.png".format(measure_type, model, mode, bootstrap_rounds,  equal_weight), dpi=300)
    # save the win matrix as a numpy array

    np.save(output_path + "win_matrix_{}_{}_{}_n={}_sum_equal_weight={}_bootstrap_sum.npy".format(
        measure_type, model, mode, bootstrap_rounds, equal_weight), win_matrix)


# measure = measures[0]
weights = [1/16, 1/16, 1/16,
           1/12, 1/12,
           1/4, 1/4, 1/4]
bootstrap_sum("CASP16", model, mode,
              score_path, output_path="./bootstrap_EU/",
              weight=None, bootstrap_rounds=bootstrap_rounds)
