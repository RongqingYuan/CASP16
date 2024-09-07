import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import ward, fcluster
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

score_path = "./by_target/"
measure = "GDT_HA"
measure = "GDT_TS"
score_file = "groups_by_targets_for-{}-EU.csv".format(measure)
score_file = "groups_by_targets_for-raw-{}-EU.csv".format(measure)

data = pd.read_csv(score_path + score_file, index_col=0)
T0_data = data.filter(regex='T0')
T1_data = data.filter(regex='T1')
T2_data = data.filter(regex='T2')

print(T0_data.shape)
print(T1_data.shape)
print(T2_data.shape)

T0_data = T0_data.T
T1_data = T1_data.T
T2_data = T2_data.T

stages = ["T1", "T2"]
stages = ["T0", "T1"]

if stages == ["T0", "T1"]:
    # 提取xxx部分
    T0_data.index = T0_data.index.str.extract(r'T0([-\w]+)')[0]
    # there is a -D*. In order to match the "-" in the regex, we need to add "-" in the regex
    T1_data.index = T1_data.index.str.extract(r'T1([-\w]+)')[0]
    # 确保索引对齐
    common_index = T0_data.index.intersection(T1_data.index)
    df0_aligned = T0_data.loc[common_index]
    df1_aligned = T1_data.loc[common_index]
    # 计算差值
    df_diff = df1_aligned - df0_aligned
    # breakpoint()

    # 结果
    print(df1_aligned.shape)
    print(df0_aligned.shape)
    print(df_diff.shape)

    df_diff.fillna(0, inplace=True)
    print(df_diff.shape)

    # 对行进行 Ward 聚类
    row_linkage = sch.linkage(df_diff, method='ward')

    # 对列进行 Ward 聚类
    col_linkage = sch.linkage(df_diff.T, method='ward')

    # # 绘制热图和树状图
    # sns.clustermap(score_matrix, row_linkage=row_linkage,
    #                col_linkage=col_linkage, cmap="viridis", figsize=(40, 30), dendrogram_ratio=(0.15, 0.15))
    # plt.savefig("./by_target_png/clustermap-{}.png".format(measure), dpi=300)

    # 生成clustermap
    g = sns.clustermap(df_diff,
                       row_linkage=row_linkage,
                       col_linkage=col_linkage,
                       cmap="coolwarm",
                       figsize=(40, 15),
                       dendrogram_ratio=(0.15, 0.15),
                       vmin=-80,  # 设定最小值为 -n
                       vmax=80)

    # 调整x轴和y轴的字体大小
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=12)  # 调整x轴字体大小
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=12)  # 调整y轴字体大小
    # rotate x-axis labels
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45)
    # align to the right
    plt.setp(g.ax_heatmap.get_xticklabels(), ha='right')
    # set the title
    g.fig.suptitle("Clustermap-{}".format(measure), fontsize=30)
    plt.savefig(
        "./stages_png/clustermap_{}-{}-{}.png".format(stages[0], stages[1], measure), dpi=300)

elif stages == ["T1", "T2"]:
    # 提取xxx部分
    T1_data.index = T1_data.index.str.extract(r'T1([-\w]+)')[0]
    # there is a -D*. In order to match the "-" in the regex, we need to add "-" in the regex
    T2_data.index = T2_data.index.str.extract(r'T2([-\w]+)')[0]
    # 确保索引对齐
    common_index = T1_data.index.intersection(T2_data.index)
    df0_aligned = T1_data.loc[common_index]
    df1_aligned = T2_data.loc[common_index]
    # 计算差值
    df_diff = df1_aligned - df0_aligned
    # breakpoint()

    # 结果
    print(df1_aligned.shape)
    print(df0_aligned.shape)
    print(df_diff.shape)

    df_diff.fillna(0, inplace=True)
    print(df_diff.shape)

    # 对行进行 Ward 聚类
    row_linkage = sch.linkage(df_diff, method='ward')

    # 对列进行 Ward 聚类
    col_linkage = sch.linkage(df_diff.T, method='ward')

    # # 绘制热图和树状图
    # sns.clustermap(score_matrix, row_linkage=row_linkage,
    #                col_linkage=col_linkage, cmap="viridis", figsize=(40, 30), dendrogram_ratio=(0.15, 0.15))
    # plt.savefig("./by_target_png/clustermap-{}.png".format(measure), dpi=300)

    # 生成clustermap
    g = sns.clustermap(df_diff,
                       row_linkage=row_linkage,
                       col_linkage=col_linkage,
                       cmap="coolwarm",
                       figsize=(40, 30),
                       dendrogram_ratio=(0.15, 0.15),
                       vmin=-80,  # 设定最小值为 -n
                       vmax=80)

    # 调整x轴和y轴的字体大小
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=12)  # 调整x轴字体大小
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=12)  # 调整y轴字体大小
    # rotate x-axis labels
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45)
    # align to the right
    plt.setp(g.ax_heatmap.get_xticklabels(), ha='right')
    # set the title
    g.fig.suptitle("Clustermap-{}".format(measure), fontsize=30)
    plt.savefig(
        "./stages_png/clustermap_{}-{}-{}.png".format(stages[0], stages[1], measure), dpi=300)
