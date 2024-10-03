import os
import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def bootstrap_sum(measures, model, mode,
                  score_path, output_path="./bootstrap_EU/",
                  impute_value=-2, weight=None, bootstrap_rounds=1000,  top_n=25):
    if isinstance(measures, str):
        if measures == "CASP15":
            measures = ['LDDT', 'CAD_AA', 'SphGr',
                        'MolPrb_Score',
                        'GDT_HA', 'QSE', 'reLLG_lddt']
            measure_type = "CASP15"
        elif measures == "CASP16":
            measures = ['GDT_HA', 'GDC_SC',
                        'AL0_P', 'SphGr',
                        'CAD_AA', 'QSE', 'LDDT',
                        'MolPrb_Score',
                        'reLLG_const']
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
    measure = measures[0]
    score_path = score_path + f"impute={impute_value}/"
    score_file = f"group_by_target-{measure}-{model}-{mode}.csv"
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
    data = pd.DataFrame()
    measure_score_dict = {}
    for i in range(len(measures)):
        measure = measures[i]
        score_file = f"group_by_target-{measure}-{model}-{mode}.csv"
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
    plt.figure(figsize=(50, 25))
    bottom = [0 for i in range(length)]
    for key in measure_score_dict:
        measure_points = measure_score_dict[key]
        points = [measure_points[group] for group in groups]
        plt.bar(groups_plt, points, bottom=bottom, label=key, width=0.8)
        bottom = [bottom[i] + points[i] for i in range(length)]
    plt.xticks(np.arange(length), groups_plt,
               rotation=90, fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=28)
    if equal_weight:
        plt.title(
            f"sum z-score for {measure_type} monomer, {model} models, {mode} EUs with equal weight", fontsize=30)
        plt.savefig(output_path +
                    f"sum_score_{measure_type}_{model}_{mode}_impute={impute_value}_equal_weight.png",
                    dpi=300)
    else:
        plt.title(
            f"weighted sum z-score for {measure_type} monomer, {model} models, {mode} EUs", fontsize=30)
        plt.savefig(output_path +
                    f"sum_score_{measure_type}_{model}_{mode}_impute={impute_value}_custom_weight.png",
                    dpi=300)

    top_n_group = groups[:top_n]
    top_n_group_plt = groups_plt[:top_n]
    plt.figure(figsize=(16, 12))
    bottom = [0 for i in range(top_n)]
    for key in measure_score_dict:
        measure_points = measure_score_dict[key]
        points = [measure_points[group] for group in top_n_group]
        plt.bar(top_n_group_plt, points, bottom=bottom,
                label=key, width=0.8)
        bottom = [bottom[i] + points[i] for i in range(top_n)]
    plt.xticks(np.arange(top_n), top_n_group_plt,
               rotation=45, fontsize=20, ha='right')
    plt.yticks(fontsize=20)
    # set y tick min to -15
    if min(bottom) > 0:
        plt.ylim(-2, max(bottom)+5)
    # plt.ylim(-5, max(bottom)+5)
    plt.legend(fontsize=20)
    if equal_weight:
        plt.title(
            f"sum z-score for {measure_type} monomer, {model} models, {mode} EUs with equal weight", fontsize=20)
        png_file = f"sum_points_{measure_type}_{model}_{mode}_impute_value={impute_value}_top_{top_n}_equal_weight.png"
        plt.savefig(output_path + png_file, dpi=300)
    else:
        plt.title(
            f"weighted sum z-score for {measure_type} monomer, {model} models, {mode} EUs", fontsize=20)
        png_file = f"sum_points_{measure_type}_{model}_{mode}_impute_value={impute_value}_top_{top_n}_custom_weight.png"
        plt.savefig(output_path + png_file, dpi=300)

    sum = grouped_data.sum(axis=0)
    scores = dict(sum)
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    dict_file = f"{measure_type}_{model}_{mode}_n={bootstrap_rounds}_equal_weight={equal_weight}_impute={impute_value}_ranking_sum.txt"
    with open(output_path + dict_file, "w")as f:
        f.write(str(scores))
    groups = list(scores.keys())
    length = len(groups)
    win_matrix = [[0 for i in range(length)] for j in range(length)]
    # get the target list, it is the first element when split T1_data.index
    targets = grouped_data.index.map(lambda x: x.split("-")[0])
    grouped_data["target"] = targets

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
    plt.xticks(np.arange(length), groups, rotation=45, fontsize=10)
    plt.yticks(np.arange(length), groups, rotation=0, fontsize=10)
    plt.title(
        "{} bootstrap result of sum points for {} EUs".format(measure_type, mode), fontsize=20)
    png_file = f"win_matrix_{measure_type}_{model}_{mode}_n={bootstrap_rounds}_equal_weight={equal_weight}_impute={impute_value}_bootstrap_sum.png"
    plt.savefig(output_path + png_file, dpi=300)
    npy_file = f"win_matrix_{measure_type}_{model}_{mode}_n={bootstrap_rounds}_equal_weight={equal_weight}_impute={impute_value}_bootstrap_sum.npy"
    np.save(output_path + npy_file,  win_matrix)

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
        value = float(text.get_text())  # 获取注释的值
        if value >= 0.95:
            text.set_color('red')  # >0.95 的文字为红色
        elif value < 0.95 and value >= 0.75:
            text.set_color('white')  # 0.75-0.95 的文字为白色
        else:
            text.set_color('black')  # 其他文字为黑色

    cbar = ax.collections[0].colorbar
    # also set 0, int(bootstrap_rounds/4), int(bootstrap_rounds/2), int(bootstrap_rounds*3/4), bootstrap_rounds
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
               rotation=45, fontsize=18, ha='right')
    plt.yticks(np.arange(top_n), top_n_id, rotation=0, fontsize=18)
    if equal_weight:
        plt.title("bootstrap result of {} score for {} models, {} targets, top {} groups, equal weight".format(
            measure_type, model, mode, top_n), fontsize=16)
    else:
        plt.title("bootstrap result of {} score for {} models, {} targets, top {} groups".format(
            measure_type, model, mode, top_n), fontsize=16)
    png_top_file = f"win_matrix_{measure_type}_{model}_{mode}_n={bootstrap_rounds}_equal_weight={equal_weight}_impute={impute_value}_top_{top_n}_bootstrap_sum.png"
    plt.savefig(output_path + png_top_file, dpi=300)


# score_path = "./group_by_target_EU/"
# output_path = "./bootstrap_EU/"
# # measure = "RMSD[L]"
# # measure = "RMS_CA"
# # measure = "GDT_HA"
# # measure = "GDT_TS"

# # mode = "hard"
# # mode = "medium"
# # mode = "easy"
# mode = "all"
# # model = "first"
# model = "best"

# measures = ['GDT_TS', 'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
#             'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
#             'SphGr',
#             'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
#             'MP_rotout', 'MP_ramout', 'MP_ramfv', 'reLLG_lddt', 'reLLG_const']

# CASP15_measures = ['LDDT', 'CAD_AA', 'SphGr',
#                    'MP_clash', 'RMS_CA',
#                    'GDT_HA', 'QSE', 'reLLG_const']

# CASP16_measures = ['GDT_HA', 'GDC_SC',
#                    'AL0_P', 'SphGr',
#                    'CAD_AA',
#                    'QSE', 'MolPrb_Score',
#                    'reLLG_const']
# measures = ['GDT_HA', 'GDC_SC',
#             'AL0_P', 'SphGr',
#             'CAD_AA', 'QSE', 'LDDT',
#             'MolPrb_Score',
#             'reLLG_const']
# equal_weight = True
# equal_weight = False


parser = argparse.ArgumentParser(
    description="options for bootstrapping sum of z-scores")
parser.add_argument("--score_path", type=str, default="./group_by_target_EU/")
parser.add_argument("--measures", type=str, default="CASP15")
parser.add_argument("--model", type=str, default="best")
parser.add_argument("--mode", type=str, default="all")
parser.add_argument("--output_path", type=str, default="./bootstrap_EU/")
parser.add_argument("--impute_value", type=int, default=-2)
parser.add_argument("--weight", type=float, nargs='+', default=None)
parser.add_argument("--bootstrap_rounds", type=int, default=1000)
parser.add_argument("--top_n", type=int, default=25)
parser.add_argument("--equal_weight", action="store_true")

args = parser.parse_args()
score_path = args.score_path
measures = args.measures
model = args.model
mode = args.mode
output_path = args.output_path
impute_value = args.impute_value
weight = args.weight
bootstrap_rounds = args.bootstrap_rounds
top_n = args.top_n
equal_weight = args.equal_weight
if not os.path.exists(output_path):
    os.makedirs(output_path)
if equal_weight:
    weight = [1/9] * 9
else:
    # weight = [1/16, 1/16, 1/16,
    #           1/12, 1/12,
    #           1/4, 1/4, 1/4]

    # weight = [1/6, 1/16,
    #           1/16, 1/8,
    #           1/8, 1/6, 1/16,
    #           1/16,
    #           1/6]
    weight = [1/16, 1/16, 1/16,
              1/12,
              1/6, 1/6, 1/6]

# measures = ['GDT_HA', 'GDC_SC',
#             'AL0_P', 'SphGr',
#             'CAD_AA', 'QSE', 'LDDT',
#             'MolPrb_Score',
#             'reLLG_const']
bootstrap_sum(measures=measures, model=model, mode=mode,
              score_path=score_path, output_path=output_path,
              impute_value=impute_value, weight=weight, bootstrap_rounds=bootstrap_rounds, top_n=top_n)
