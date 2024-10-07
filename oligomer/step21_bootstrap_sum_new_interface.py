import argparse
import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt


def bootstrap_sum(measures, model, mode,
                  interface_score_path, EU_score_path, output_path="./bootstrap_new_interface/",
                  weight=None, bootstrap_rounds=1000, impute_value=-2, top_n=25):
    EU_measures = ["tm_score", "lddt"]
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
    equal_weight = len(set(weight)) == 1
    assert len(measures) == len(weight)

    # measure = measures[0]
    # score_file = f"group_by_target-{measure}-{model}-{mode}-impute_value={impute_value}.csv"
    # data_tmp = pd.read_csv(interface_score_path + score_file, index_col=0)
    # data_columns = data_tmp.columns
    # interface_count = {}
    # for EU in data_columns:
    #     target = EU.split("_")[0]
    #     if target not in interface_count:
    #         interface_count[target] = 0
    #     interface_count[target] += 1
    # target_weight = {key: 1/(value ** (2/3))
    #                  for key, value in interface_count.items()}
    # interface_weight = {EU: target_weight[EU.split("_")[0]]
    #                     for EU in data_columns}
    # interface_weight = pd.Series(interface_weight)

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
            score_matrix = score_matrix * weight_i

            # # get only T columns
            # score_matrix = score_matrix.loc[:,
            #                                 score_matrix.columns.str.startswith("T")]
            score_matrix_sum = score_matrix.sum(axis=1)
            measure_score_dict[measure] = dict(score_matrix_sum)
            data = pd.concat([data, score_matrix], axis=0)
        elif measure in interface_measures:
            score_file = f"sum_weighted_EU_{measure}-{model}-{mode}-impute_value={impute_value}.csv"
            score_matrix = pd.read_csv(
                interface_score_path + score_file, index_col=0)
            # drop the sum column
            score_matrix = score_matrix.drop(columns="sum")
            score_matrix = score_matrix.reindex(
                sorted(score_matrix.columns), axis=1)
            weight_i = weight[i]
            score_matrix = score_matrix * weight_i

            # # get only T columns
            # score_matrix = score_matrix.loc[:,
            #                                 score_matrix.columns.str.startswith("T")]
            score_matrix_sum = score_matrix.sum(axis=1)
            measure_score_dict[measure] = dict(score_matrix_sum)
            data = pd.concat([data, score_matrix], axis=0)

    # T1249_data = pd.DataFrame()
    # for i in range(len(measures)):
    #     measure = measures[i]
    #     if measure in EU_measures:
    #         score_file = f"group_by_target-T1249o-{measure}-{model}-{mode}-impute_value={impute_value}.csv"
    #         score_matrix = pd.read_csv(EU_score_path + score_file, index_col=0)
    #         score_matrix = score_matrix.reindex(
    #             sorted(score_matrix.columns), axis=1)
    #         weight_i = weight[i]
    #         # multiply the score_matrix by the weight
    #         score_matrix = score_matrix * weight_i
    #         # score_matrix = score_matrix * EU_weight
    #         score_matrix_sum = score_matrix.sum(axis=1)
    #         measure_score_dict[measure] = dict(score_matrix_sum)
    #         data = pd.concat([data, score_matrix], axis=0)
    #     elif measure in interface_measures:
    #         measure = measure.replace("_mean", "")
    #         if measure == "dockq":
    #             measure = "dockq_ave"
    #         score_file = f"group_by_target-T1249o-{measure}-{model}-{mode}-impute_value={impute_value}.csv"
    #         score_matrix = pd.read_csv(
    #             EU_score_path + score_file, index_col=0)
    #         score_matrix = score_matrix.reindex(
    #             sorted(score_matrix.columns), axis=1)
    #         weight_i = weight[i]
    #         # multiply the score_matrix by the weight
    #         score_matrix = score_matrix * weight_i
    #         # score_matrix = score_matrix * interface_weight
    #         score_matrix_sum = score_matrix.sum(axis=1)
    #         measure_score_dict[measure] = dict(score_matrix_sum)
    #         data = pd.concat([data, score_matrix], axis=0)

    # drop whole column if any value is nan
    data = data.dropna(axis=1, how='any')
    data = data.T
    columns = data.columns
    grouped_columns = data.groupby(columns, axis=1)
    grouped_data = grouped_columns.sum()
    groups = grouped_data.columns
    length = len(groups)
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
    plt.figure(figsize=(48, 24))
    bottom = [0 for i in range(length)]
    convert = {"qs_global_mean": "QSglob_mean",
               "ics_mean": "ICS_mean",
               "ips_mean": "IPS_mean",
               "dockq_mean": "DockQ_mean",
               "tm_score": "TMscore",
               "lddt": "lDDT"}
    for key in measure_score_dict:
        measure_points = measure_score_dict[key]
        points = [measure_points[group] for group in groups]
        plt.bar(groups_plt, points, bottom=bottom,
                label=convert[key], width=0.8)
        bottom = [bottom[i] + points[i] for i in range(length)]
    plt.xticks(np.arange(length), groups_plt,
               rotation=90, fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=32)
    if equal_weight:
        plt.title(
            f"z-score sum for {measure_type} oligomer {mode} EUs, {model} models with equal weight", fontsize=32)
        png_file = f"sum_points_{measure_type}_{model}_{mode}_impute_value={impute_value}_equal_weight.png"
        plt.savefig(output_path + png_file, dpi=300)
    else:
        plt.title(
            f"z-score sum for {measure_type} oligomer {mode} EUs, {model} models with custom weight", fontsize=32)
        png_file = f"sum_points_{measure_type}_{model}_{mode}_impute_value={impute_value}_custom_weight.png"
        plt.savefig(output_path + png_file, dpi=300)

    top_n_group = groups[:top_n]
    top_n_group_plt = groups_plt[:top_n]
    plt.figure(figsize=(16, 12))
    bottom = [0 for i in range(top_n)]
    for key in measure_score_dict:
        measure_points = measure_score_dict[key]
        points = [measure_points[group] for group in top_n_group]
        plt.bar(top_n_group_plt, points, bottom=bottom,
                label=convert[key], width=0.8)
        bottom = [bottom[i] + points[i] for i in range(top_n)]
    plt.xticks(np.arange(top_n), top_n_group_plt,
               rotation=45, fontsize=20, ha='right')
    plt.yticks(fontsize=20)
    if min(bottom) > 0:
        plt.ylim(-2, max(bottom)+5)
    plt.legend(fontsize=20)
    if equal_weight:
        plt.title(
            f"z-score sum for {measure_type} oligomer {mode} EUs, {model} models with equal weight", fontsize=20)
        png_file = f"sum_points_{measure_type}_{model}_{mode}_impute_value={impute_value}_top_{top_n}_equal_weight.png"
        plt.savefig(output_path + png_file, dpi=300)
    else:
        plt.title(
            f"z-score sum for {measure_type} oligomer {mode} EUs, {model} models with custom weight", fontsize=20)
        png_file = f"sum_points_{measure_type}_{model}_{mode}_impute_value={impute_value}_top_{top_n}_custom_weight.png"
        plt.savefig(output_path + png_file, dpi=300)

    # use the above code to get a initial ranking of the groups.
    # then generate new groups list using the ranking
    # then do bootstrapping

    sum = grouped_data.sum(axis=0)
    scores = dict(sum)
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    txt_file = f"{measure_type}_{model}_{mode}_n={bootstrap_rounds}_equal_weight={equal_weight}_impute_value={impute_value}_ranking_sum.txt"
    with open(output_path + txt_file, "w") as f:
        f.write(str(scores))
    groups = list(scores.keys())
    length = len(groups)
    win_matrix = [[0 for i in range(length)] for j in range(length)]
    targets = grouped_data.index.map(lambda x: x.split("-")[0])
    grouped_data["target"] = targets

    # now do the bootstrapping
    for r in range(bootstrap_rounds):
        grouped = grouped_data.groupby('target')
        data_bootstrap = grouped.apply(lambda x: x.sample(
            n=1)).sample(n=len(grouped), replace=True)
        data_bootstrap = data_bootstrap.sort_index()  # not necessary,just to look nice
        data_bootstrap = data_bootstrap.drop(columns='target')
        sum = data_bootstrap.sum(axis=0)
        scores = dict(sum)
        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        for i in range(length):
            for j in range(length):
                if i == j:
                    continue
                if scores[groups[i]] > scores[groups[j]]:
                    win_matrix[i][j] += 1
        print("Round: {}".format(r))

    plt.figure(figsize=(30, 25))
    ax = sns.heatmap(win_matrix, annot=False,
                     cmap='Greys', cbar=True, square=True,
                     #  linewidths=1, linecolor='black'
                     )
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
    ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment='center')
    ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center')
    plt.xticks(np.arange(length), groups, rotation=45, fontsize=10)
    plt.yticks(np.arange(length), groups, rotation=0, fontsize=10)
    plt.title(
        "{} bootstrap result of sum points for {} EUs, {} models".format(measure_type, mode, model), fontsize=16)
    png_file = f"win_matrix_{measure_type}_{model}_{mode}_n={bootstrap_rounds}_equal_weight={equal_weight}_impute={impute_value}_bootstrap_sum.png"
    plt.savefig(output_path + png_file, dpi=300)

    # save the win matrix as a numpy array
    npy_file = f"win_matrix_{measure_type}_{model}_{mode}_n={bootstrap_rounds}_equal_weight={equal_weight}_impute={impute_value}_bootstrap_sum.npy"
    np.save(output_path + npy_file, win_matrix)

    top_n_id = groups[:top_n]
    top_n_id = [i.replace("TS", "") for i in top_n_id]
    win_matrix = np.array(win_matrix)
    win_matrix_top_n = win_matrix[:top_n, :top_n]
    win_matrix_top_n = win_matrix_top_n / bootstrap_rounds
    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(win_matrix_top_n, annot=True, fmt=".2f",
                     cmap='Greys', cbar=True, square=True,
                     #   linewidths=1, linecolor='black',
                     )

    for text in ax.texts:
        value = float(text.get_text())  # get text from the heatmap
        if value >= 0.95:
            text.set_color('red')
        elif value < 0.95 and value >= 0.75:
            text.set_color('white')
        else:
            text.set_color('black')

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
    plt.xticks(np.arange(top_n), top_n_id,
               rotation=45, fontsize=16, ha='right')
    plt.yticks(np.arange(top_n), top_n_id, rotation=0, fontsize=16)
    if equal_weight:
        plt.title("oligomer bootstrap result for {} targets, {} models, top {} groups".format(
            mode, model, top_n), fontsize=16)
    else:
        plt.title("oligomer bootstrap result for {} targets, {} models, top {} groups".format(
            mode, model, top_n), fontsize=16)
    png_top_file = f"win_matrix_{measure_type}_{model}_{mode}_n={bootstrap_rounds}_equal_weight={equal_weight}_impute={impute_value}_top_{top_n}_bootstrap_sum.png"
    plt.savefig(output_path + png_top_file, dpi=300)


parser = argparse.ArgumentParser(
    description="options for bootstrapping sum of z-scores")
parser.add_argument("--measures", type=list, default=["qs_global_mean", "ics_mean",
                                                      "ips_mean", "dockq_mean", "tm_score", "lddt"])
parser.add_argument("--model", type=str, default="best")
parser.add_argument("--mode", type=str, default="all")
parser.add_argument("--interface_score_path", type=str,
                    default="./group_by_target_per_interface/")
parser.add_argument("--EU_score_path", type=str,
                    default="./group_by_target_EU_new/")
parser.add_argument("--output_path", type=str,
                    default="./bootstrap_new_interface/")
parser.add_argument("--weight", type=list, default=None)
parser.add_argument("--bootstrap_rounds", type=int, default=1000)
parser.add_argument("--impute_value", type=int, default=-2)
parser.add_argument("--top_n", type=int, default=25)
parser.add_argument("--equal_weight", action="store_true")

args = parser.parse_args()
measures = args.measures
model = args.model
mode = args.mode
interface_score_path = args.interface_score_path
EU_score_path = args.EU_score_path
output_path = args.output_path
weight = args.weight
bootstrap_rounds = args.bootstrap_rounds
impute_value = args.impute_value
top_n = args.top_n
equal_weight = args.equal_weight


if equal_weight:
    weights = [1/6] * 6
else:
    weights = [1/6, 1/6, 1/6,
               1/6, 1/6,
               1/6]

bootstrap_sum(measures, model, mode,
              interface_score_path, EU_score_path, output_path=output_path,
              weight=None, bootstrap_rounds=bootstrap_rounds, impute_value=impute_value, top_n=top_n)
