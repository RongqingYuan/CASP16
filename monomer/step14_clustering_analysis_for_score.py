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


def pairwise_distance_with_nan(data):
    n_samples = data.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            valid_mask = ~np.isnan(data[i]) & ~np.isnan(
                data[j])  # Only consider valid (non-NaN) values
            if np.any(valid_mask):
                dist = np.linalg.norm(
                    data[i, valid_mask] - data[j, valid_mask])
            else:
                dist = 100000  # Assign a very large distance if no valid comparison exists
                # dist = np.inf  # Assign a very large distance if no valid comparison exists
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


score_matrix = pd.read_csv(score_path + score_file, index_col=0)
score_matrix = score_matrix.T
print(score_matrix.shape)
# kick out columns that has more than 90% of nan
score_matrix = score_matrix.loc[:, score_matrix.isnull().mean() < .9]
print(score_matrix.shape)
score_matrix = score_matrix.T
# run ward clustering
# # impute the missing values to -5, means they are doing terribly bad
# score_matrix.fillna(0, inplace=True)

# impute the missing values to the mean value of the column
# score_matrix.fillna(score_matrix.mean(), inplace=True)

# keep columns that start with T1
score_matrix = score_matrix.filter(regex='T1')
print(score_matrix.shape)
score_array = np.array(score_matrix)
score_array_T = np.array(score_matrix.T)

score_matrix.fillna(50, inplace=True)
# Calculate pairwise distances
dist_matrix = pairwise_distance_with_nan(score_array)
# Convert the distance matrix into a format suitable for clustering
# Convert distance matrix to condensed form for clustering
dist_array = squareform(dist_matrix, checks=False)
# # Perform hierarchical clustering using the distance matrix
# linkage_matrix = linkage(dist_array, method='complete')  # You can also try 'single', 'average', etc.
# 对行进行 Ward 聚类
row_linkage = sch.linkage(dist_array, method='ward')

dist_matrix_T = pairwise_distance_with_nan(score_array_T)
dist_array_T = squareform(dist_matrix_T, checks=False)
col_linkage = sch.linkage(dist_array_T, method='ward')
# 对列进行 Ward 聚类
# col_linkage = sch.linkage(dist_array.T, method='ward')

# # 绘制热图和树状图
# sns.clustermap(score_matrix, row_linkage=row_linkage,
#                col_linkage=col_linkage, cmap="viridis", figsize=(40, 30), dendrogram_ratio=(0.15, 0.15))
# plt.savefig("./by_target_png/clustermap-{}.png".format(measure), dpi=300)


# 生成clustermap
g = sns.clustermap(score_matrix,
                   #    row_linkage=row_linkage,
                   col_linkage=col_linkage,
                   cmap="coolwarm",
                   figsize=(40, 30),
                   dendrogram_ratio=(0.15, 0.15))

# 调整x轴和y轴的字体大小
plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=12)  # 调整x轴字体大小
plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=12)  # 调整y轴字体大小
# set the title
g.fig.suptitle("Clustermap-{}".format(measure), fontsize=30)
plt.savefig("./by_target_png/clustermap-{}.png".format(measure), dpi=300)


breakpoint()


score_matrix = pd.read_csv(score_path + score_file, index_col=0)
score_matrix = score_matrix.T
print(score_matrix.shape)
# kick out columns that has more than 90% of nan
score_matrix = score_matrix.loc[:, score_matrix.isnull().mean() < .9]
print(score_matrix.shape)
score_matrix = score_matrix.T
# run ward clustering
# # impute the missing values to -5, means they are doing terribly bad
# score_matrix.fillna(0, inplace=True)

# # impute the missing values to the mean value of the column
# score_matrix.fillna(score_matrix.mean(), inplace=True)
score_matrix.fillna(50, inplace=True)

# keep columns that start with T1
score_matrix = score_matrix.filter(regex='T1')
print(score_matrix.shape)

# 对行进行 Ward 聚类
row_linkage = sch.linkage(score_matrix, method='ward')

# 对列进行 Ward 聚类
col_linkage = sch.linkage(score_matrix.T, method='ward')

# # 绘制热图和树状图
# sns.clustermap(score_matrix, row_linkage=row_linkage,
#                col_linkage=col_linkage, cmap="viridis", figsize=(40, 30), dendrogram_ratio=(0.15, 0.15))
# plt.savefig("./by_target_png/clustermap-{}.png".format(measure), dpi=300)


# 生成clustermap
g = sns.clustermap(score_matrix,
                   row_linkage=row_linkage,
                   col_linkage=col_linkage,
                   cmap="coolwarm",
                   figsize=(40, 30),
                   dendrogram_ratio=(0.15, 0.15))

# 调整x轴和y轴的字体大小
plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=12)  # 调整x轴字体大小
plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=12)  # 调整y轴字体大小
# set the title
g.fig.suptitle("Clustermap-{}".format(measure), fontsize=30)
plt.savefig("./by_target_png/clustermap-{}.png".format(measure), dpi=300)
