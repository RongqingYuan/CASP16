import os
import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec


def plot_heatmap(measures, model, mode,
                 score_path, interface_score_path, output_path,
                 impute_value=-2, weight=None):
    EU_measures = ["tm_score", "lddt", "dockq_ave", "dockq_wave",
                   "ics", "ips", "qs_best", "qs_global"]
    interface_measures = ["qs_global_mean", "ics_mean",
                          "ips_mean", "dockq_mean", "qs_best_mean", "qs_global_mean_inclNone", "ics_mean_inclNone",
                          "ips_mean_inclNone", "dockq_mean_inclNone", "qs_best_mean_inclNone"]

    measures = list(measures)
    if measures == ['qs_global_mean', 'ics_mean', 'ips_mean', 'dockq_mean', 'tm_score', 'lddt']:
        measure_type = "CASP16"
    elif measures == ['qs_best', 'ics', 'ips', 'tm_score', 'lddt']:
        measure_type = "CASP15"
    else:
        measure_type = "custom"
    if weight is None:
        weight = [1/len(measures)] * len(measures)
        print("weight is None, use equal weight")
    assert len(measures) == len(weight)

    # to get the mask
    measure = "tm_score"
    raw_path = score_path
    raw_file = raw_path + \
        f"{measure}-{model}-{mode}-impute={impute_value}_raw.csv"
    raw_data = pd.read_csv(raw_file, index_col=0)
    mask = raw_data.isna()
    # get the same shape of data
    # score_path = score_path + f"impute={impute_value}/"
    score_file = f"{measure}-{model}-{mode}-impute={impute_value}.csv"
    data_tmp = pd.read_csv(score_path + score_file, index_col=0)
    heatmap_data = pd.DataFrame(
        0, index=data_tmp.index, columns=data_tmp.columns)

    for i in range(len(measures)):
        measure = measures[i]
        if measure in EU_measures:
            score_path = score_path
            score_file = f"{measure}-{model}-{mode}-impute={impute_value}.csv"
            score_matrix = pd.read_csv(score_path + score_file, index_col=0)
            weight_i = weight[i]
            score_matrix = score_matrix * weight_i
            heatmap_data = heatmap_data + score_matrix
        elif measure in interface_measures:
            score_file = f"{measure}-{model}-{mode}-impute={impute_value}_unweighted_EU.csv"
            score_matrix = pd.read_csv(
                interface_score_path + score_file, index_col=0)
            weight_i = weight[i]
            score_matrix = score_matrix * weight_i
            heatmap_data = heatmap_data + score_matrix
    # drop all columns any value is nans
    heatmap_data = heatmap_data.dropna(axis=1, how='any')

    # data_columns = heatmap_data.columns
    # target_count = {}
    # for EU in data_columns:
    #     target = EU.split("v")[0]
    #     if target not in target_count:
    #         target_count[target] = 0
    #     target_count[target] += 1
    # target_weight = {key: 1/value for key, value in target_count.items()}
    # EU_weight = {EU: target_weight[EU.split("v")[0]]
    #              for EU in data_columns}
    # EU_weight = pd.Series(EU_weight)
    # heatmap_data = heatmap_data * EU_weight

    # save heatmap_data to csv
    heatmap_data.to_csv(output_path +
                        f"heatmap_{measure_type}_{model}_{mode}_impute={impute_value}_equal_weight.csv")
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
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.3)

    # plot the heatmap
    ax0 = fig.add_subplot(gs[0])
    sns.heatmap(masked_data, cmap=cmap, cbar=True, ax=ax0)
    ax0.set_yticklabels(
        [f'{i}' for i in sorted_indices], rotation=0)  # use the same order as the row sum
    ax0.set_xticklabels(sorted_heatmap_data.columns, rotation=90)

    # set x tick font size
    ax0.tick_params(axis='x', labelsize=16)
    # set y tick font size
    ax0.tick_params(axis='y', labelsize=16)
    ax0.set_title(
        "Heatmap for z-scores of oligomers", fontsize=20)

    # plot the row sum
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    y_pos = range(len(sorted_sum))
    y_pos = [i+0.5 for i in y_pos]  # change the position of the bars
    ax1.barh(y_pos, sorted_sum, color='tan')
    # ax1.margins(y=0.5)
    ax1.set_yticks(range(len(sorted_sum)))
    ax1.set_yticklabels(
        [f'{i}' for i in sorted_indices], rotation=0)  # use the same order as the heatmap
    # ax1.spines['bottom'].set_position(('outward', 10))  # 将 x 轴向下移动 10 点
    # ymin, ymax = ax1.get_ylim()  # 获取当前的 y 轴范围
    # ax1.set_ylim(ymin - 1, ymax-1)  # 为最底部条形预留空间

    # set x tick font size
    ax1.tick_params(axis='x', labelsize=16)
    # set y tick font size
    ax1.tick_params(axis='y', labelsize=16)
    ax1.invert_yaxis()  # flip the y-axis
    ax1.set_xlabel("Sum", fontsize=16)
    ax1.set_title("Group sum z-scores", fontsize=20)

    # save the figure
    plt.tight_layout()
    plt.show()
    plt.savefig(output_path + f"heatmap_{measure_type}_{model}_{mode}_impute={impute_value}_equal_weight.png",
                dpi=300)

    # 假设 heatmap_data 是一个 DataFrame
    # 计算每行的和
    heatmap_data['sum'] = heatmap_data.sum(axis=1)
    # 对 sum 列进行排序
    heatmap_data = heatmap_data.sort_values(by='sum', ascending=False)
    # 仅保留 sum 列
    sum_column = heatmap_data[['sum']]
    # 将 sum 列保存到 CSV 文件
    sum_column.to_csv(output_path +
                      f"group_sum_{measure_type}_{model}_{mode}_impute={impute_value}_equal_weight.csv")


parser = argparse.ArgumentParser(
    description="options for bootstrapping sum of z-scores")
parser.add_argument("--measures", nargs='+',
                    default=[
                        "tm_score",
                        "lddt",
                        # "qs_best_mean",
                        # "ics_mean",
                        # "ips_mean",
                        # "dockq_mean"
                        "qs_best_mean_inclNone",
                        "ics_mean_inclNone",
                        "ips_mean_inclNone",
                        "dockq_mean_inclNone",
                    ]
                    )
parser.add_argument("--EU_score_path", type=str,
                    default="./group_by_target_EU_new/")
parser.add_argument("--interface_score_path", type=str,
                    default="./group_by_target_per_interface/")
parser.add_argument("--model", type=str, default="best")
parser.add_argument("--mode", type=str, default="all")
parser.add_argument("--output_path", type=str,
                    default="./heatmap_interface/"
                    # default="./heatmap_prot_nucl_interface/"
                    # default="./heatmap_prot_prot_interface/"
                    )
parser.add_argument("--impute_value", type=int, default=-2)
parser.add_argument("--weight", type=float, nargs='+',
                    default=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

args = parser.parse_args()
measures = args.measures
score_path = args.EU_score_path
interface_score_path = args.interface_score_path
model = args.model
mode = args.mode
output_path = args.output_path
impute_value = args.impute_value
weight = args.weight

if not os.path.exists(output_path):
    os.makedirs(output_path)


plot_heatmap(measures=measures, model=model, mode=mode,
             score_path=score_path, interface_score_path=interface_score_path, output_path=output_path,
             impute_value=impute_value, weight=None)
