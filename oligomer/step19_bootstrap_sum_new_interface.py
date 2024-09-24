import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt

# hard_group = [

# ]

# medium_group = [

# ]

# easy_group = [

# ]


# csv_path = "./monomer_data_aug_30/processed/EU/"
# csv_path = "./monomer_data_Sep_10/processed/EU/"
# csv_path = "./monomer_data_Sep_10/raw_data/EU/"
# # csv_path = "./monomer_data_Sep_17/raw_data/"
# csv_path = "./oligomer_data_Sep_17/raw_data/"
# csv_list = [txt for txt in os.listdir(
#     csv_path) if txt.endswith(".csv") and (txt.startswith("T1") or txt.startswith("H1"))]

# model = "first"
# model = "best"

# mode = "hard"
# mode = "medium"
# mode = "easy"
# mode = "all"

# if mode == "hard":
#     csv_list = [csv for csv in csv_list if csv.split(
#         ".")[0] in hard_group]

# elif mode == "medium":
#     csv_list = [csv for csv in csv_list if csv.split(
#         ".")[0] in medium_group]

# elif mode == "easy":
#     csv_list = [csv for csv in csv_list if csv.split(
#         ".")[0] in easy_group]

# elif mode == "all":
#     pass


# # print(csv_list.__len__())
# # breakpoint()
# # read all data and concatenate them into one big dataframe
# feature = "GDT_TS"
# features = ['GDT_TS', 'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
#             'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
#             'SphGr',
#             'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
#             'MP_rotout', 'MP_ramout', 'MP_ramfv', 'reLLG_lddt', 'reLLG_const']

# inverse_columns = ["RMS_CA", "RMS_ALL", "err",
#                    "RMSD[L]", "MolPrb_Score", "FlexE", "MP_clash", "MP_rotout", "MP_ramout"]


def bootstrap_sum(measures, model, mode,
                  interface_score_path, EU_score_path, output_path="./bootstrap_new_interface/",
                  weight=None, bootstrap_rounds=1000, impute_value=-2):
    EU_measures = ["tm_score", "lddt"]
    # interface_measures = ["qs_global_max", "ics_max", "ips_max", "dockq_max"]
    interface_measures = ["qs_global_mean", "ics_mean",
                          "ips_mean", "dockq_mean"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
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
        measure_type = "CASP16"
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

    # measure = measures[0]
    # score_file = "group_by_target-{}-{}-{}.csv".format(
    #     measure, model, mode)
    # data_tmp = pd.read_csv(score_path + score_file, index_col=0)

    # breakpoint()

    # data_columns = data_tmp.columns
    # target_count = {}
    # for EU in data_columns:
    #     target = EU.split("-")[0]
    #     if target not in target_count:
    #         target_count[target] = 0
    #     target_count[target] += 1
    # target_weight = {key: 1/value for key, value in target_count.items()}
    # EU_weight = {EU: target_weight[EU.split("-")[0]]
    #              for EU in data_columns}
    # EU_weight = pd.Series(EU_weight)

    measure = measures[0]
    score_file = f"group_by_target-{measure}-{model}-{mode}-impute_value={impute_value}.csv"
    data_tmp = pd.read_csv(interface_score_path + score_file, index_col=0)
    data_columns = data_tmp.columns

    interface_count = {}
    for EU in data_columns:
        target = EU.split("_")[0]
        if target not in interface_count:
            interface_count[target] = 0
        interface_count[target] += 1
    target_weight = {key: 1/(value ** (2/3))
                     for key, value in interface_count.items()}
    interface_weight = {EU: target_weight[EU.split("_")[0]]
                        for EU in data_columns}
    interface_weight = pd.Series(interface_weight)
    breakpoint()
    data = pd.DataFrame()
    measure_score_dict = {}
    for i in range(len(measures)):
        measure = measures[i]
        if measure in EU_measures:
            score_file = f"group_by_target-{measure}-{model}-{mode}-impute_value={impute_value}.csv"
            score_matrix = pd.read_csv(EU_score_path + score_file, index_col=0)
            score_matrix = score_matrix.reindex(
                sorted(score_matrix.columns), axis=1)
            weight_i = weight[i]
            # multiply the score_matrix by the weight
            score_matrix = score_matrix * weight_i
            # score_matrix = score_matrix * EU_weight
            score_matrix_sum = score_matrix.sum(axis=1)
            measure_score_dict[measure] = dict(score_matrix_sum)
            data = pd.concat([data, score_matrix], axis=0)
        elif measure in interface_measures:
            score_file = f"group_by_target-{measure}-{model}-{mode}-impute_value={impute_value}.csv"
            score_matrix = pd.read_csv(
                interface_score_path + score_file, index_col=0)
            score_matrix = score_matrix.reindex(
                sorted(score_matrix.columns), axis=1)
            weight_i = weight[i]
            # multiply the score_matrix by the weight
            score_matrix = score_matrix * weight_i
            score_matrix = score_matrix * interface_weight
            score_matrix_sum = score_matrix.sum(axis=1)
            measure_score_dict[measure] = dict(score_matrix_sum)

    data = data.T
    columns = data.columns
    grouped_columns = data.groupby(columns, axis=1)
    grouped_data = grouped_columns.sum()
    groups = grouped_data.columns
    length = len(groups)
    # breakpoint()

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
    plt.xticks(np.arange(length), groups_plt,
               rotation=45, fontsize=10, ha='right')
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    if equal_weight:
        plt.title(
            f"z-score sum for {measure_type} EUs with equal weight", fontsize=18)
        png_file = f"sum_points_{measure_type}_{model}_{mode}_impute_value={impute_value}_equal_weight.png"
        plt.savefig(output_path + png_file, dpi=300)
    else:
        plt.title(
            f"z-score sum for {measure_type} EUs with custom weight", fontsize=18)
        png_file = f"sum_points_{measure_type}_{model}_{mode}_impute_value={impute_value}_custom_weight.png"
        plt.savefig(output_path + png_file, dpi=300)
    breakpoint()
    #####

    # breakpoint()

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


interface_score_path = "./group_by_target_per_interface/"
EU_score_path = "./group_by_target_EU_new/"
output_path = "./bootstrap_new_interface/"
# model = "first"
model = "best"
# mode = "hard"
# mode = "medium"
# mode = "easy"
mode = "all"

features = ["QSglob", "QSbest", "ICS(F1)", "lDDT", "DockQ_Avg",
            "IPS(JaccCoef)", "TMscore"]
measures = ["qs_global", "qs_best", "ics", "ips", "dockq_ave", "tm_score"]
measures = ["qs_global", "qs_best", "ics", "ips", "dockq_ave", "tm_score"]
measures = ["ics", "ips", "dockq_ave", "tm_score", "lddt"]
measures = ["qs_global", "ics", "ips", "dockq_ave", "tm_score", "lddt"]
measures = ["qs_global_max", "ics_max",
            "ips_max", "dockq_max", "tm_score", "lddt"]
measures = ["qs_global_mean", "ics_mean",
            "ips_mean", "dockq_mean", "tm_score", "lddt"]
# measures = ["tm_score"]
# measures = ["qs_global"]


equal_weight = False
equal_weight = True
if equal_weight:
    weights = [1/8] * 8
else:
    weights = [1/16, 1/16, 1/16,
               1/12, 1/12,
               1/4, 1/4, 1/4]
bootstrap_rounds = 1000
impute_value = -2

bootstrap_sum(measures, model, mode,
              interface_score_path, EU_score_path, output_path=output_path,
              weight=None, bootstrap_rounds=bootstrap_rounds, impute_value=impute_value)
