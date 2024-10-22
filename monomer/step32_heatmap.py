import os
import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec


def plot_heatmap(measures, model, mode,
                 score_path, output_path,
                 impute_value=-2, weight=None, bootstrap_rounds=1000,  top_n=25):
    if isinstance(measures, str):
        if measures == "CASP15":
            measures = ['LDDT', 'CAD_AA', 'SphGr',
                        'MP_clash', 'RMS_CA',
                        'GDT_HA', 'QSE', 'reLLG_const']
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
    # equal_weight = len(set(weight)) == 1
    assert len(measures) == len(weight)

    # to get the mask
    measure = measures[0]
    raw_path = score_path + "raw/"
    raw_file = raw_path + \
        f"groups_by_targets_for-raw-{measure}-{model}-{mode}.csv"
    raw_data = pd.read_csv(raw_file, index_col=0)
    mask = raw_data.isna()

    # get the same shape of data
    score_path = score_path + f"impute={impute_value}/"
    score_file = f"group_by_target-{measure}-{model}-{mode}.csv"
    data_tmp = pd.read_csv(score_path + score_file, index_col=0)
    heatmap_data = pd.DataFrame(
        0, index=data_tmp.index, columns=data_tmp.columns)
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
    for i in range(len(measures)):
        measure = measures[i]
        score_file = f"group_by_target-{measure}-{model}-{mode}.csv"
        score_matrix = pd.read_csv(score_path + score_file, index_col=0)
        weight_i = weight[i]
        score_matrix = score_matrix * weight_i
        score_matrix = score_matrix * EU_weight
        heatmap_data = heatmap_data + score_matrix
    sum = heatmap_data.sum(axis=1)
    sorted_indices = sum.sort_values(ascending=True).index
    sorted_heatmap_data = heatmap_data.loc[sorted_indices].reset_index(
        drop=True)
    sorted_sum = sum.loc[sorted_indices].reset_index(drop=True)
    sorted_mask = pd.DataFrame(
        mask, index=heatmap_data.index).loc[sorted_indices].reset_index(drop=True)

    # use mask to mask the data. will be used for heatmap
    masked_data = sorted_heatmap_data.copy()
    masked_data[sorted_mask] = np.nan

    # set up the colormap
    cmap = plt.cm.YlGn
    cmap = ListedColormap(cmap(np.linspace(0, 1, 256)))
    cmap.set_bad(color='gray')  # set the masked area to gray

    # set up the figure and gridspec
    fig = plt.figure(figsize=(42, 30))
    gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.1)

    # plot the heatmap
    ax0 = fig.add_subplot(gs[0])
    sns.heatmap(masked_data, cmap=cmap, cbar=True, ax=ax0)
    ax0.set_yticklabels(
        [f'{i}' for i in sorted_indices], rotation=0)  # use the same order as the row sum
    ax0.set_xticklabels(sorted_heatmap_data.columns, rotation=45, ha='right')
    ax0.tick_params(axis='x', labelsize=16)
    ax0.tick_params(axis='y', labelsize=16)
    cbar = ax0.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    ax0.set_title(
        "Heatmap for z-scores for monomers", fontsize=20)

    # plot the row sum
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    y_pos = range(len(sorted_sum))
    y_pos = [i+0.5 for i in y_pos]  # change the position of the bars
    ax1.barh(y_pos, sorted_sum, color='tan')
    ax1.set_yticks(range(len(sorted_sum)))
    ax1.set_yticklabels(
        [f'{i}' for i in sorted_indices], rotation=0)  # use the same order as the heatmap

    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.invert_yaxis()  # flip the y-axis
    ax1.set_xlabel("Sum", fontsize=16)
    ax1.set_title("Group sum z-scores", fontsize=20)

    # save the figure
    plt.tight_layout()
    plt.show()
    plt.savefig(output_path + f"heatmap_{measure_type}_{model}_{mode}_impute={impute_value}_equal_weight.png",
                dpi=300)


parser = argparse.ArgumentParser(
    description="options for bootstrapping sum of z-scores")
parser.add_argument("--score_path", type=str, default="./group_by_target_EU/")
parser.add_argument("--measures", type=str, default="CASP16")
parser.add_argument("--model", type=str, default="best")
parser.add_argument("--mode", type=str, default="all")
parser.add_argument("--output_path", type=str, default="./heatmap/")
parser.add_argument("--impute_value", type=int, default=-2)
parser.add_argument("--weight", type=float, nargs='+', default=None)
parser.add_argument("--bootstrap_rounds", type=int, default=1000)
parser.add_argument("--top_n", type=int, default=25)
parser.add_argument("--equal_weight", action="store_true")
parser.add_argument("--stage", type=str, default="1")

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
stage = args.stage
if not os.path.exists(output_path):
    os.makedirs(output_path)
if equal_weight:
    weight = [1/9] * 9
else:
    # weight = [1/16, 1/16, 1/16,
    #           1/12, 1/12,
    #           1/4, 1/4, 1/4]

    weight = [1/6, 1/16,
              1/16, 1/8,
              1/8, 1/6, 1/16,
              1/16,
              1/6]


# measures = [
#     "ICS(F1)",
#     "IPS",
#     "QSglob",
#     # "QSbest",
#     # "GDT_TS",
#     # "RMSD",
#     "GlobDockQ",
#     "TMscore",
#     "lDDT",
#     # "BestDockQ"
# ]
measures = ['GDT_HA', 'GDC_SC',
            'AL0_P', 'SphGr',
            'CAD_AA', 'QSE', 'LDDT',
            'MolPrb_Score',
            'reLLG_const']
plot_heatmap(measures=measures, model=model, mode=mode,
             score_path=score_path, output_path=output_path,
             impute_value=impute_value, weight=weight, bootstrap_rounds=bootstrap_rounds, top_n=top_n)
