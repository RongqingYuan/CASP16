import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import ward, fcluster
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns


score_path = "./by_target/"
measure = "GDT_HA"
measure = "GDT_TS"
score_file = "groups_by_targets_for-{}-EU.csv".format(measure)
score_file = "groups_by_targets_for-raw-{}-EU.csv".format(measure)


score_matrix = pd.read_csv(score_path + score_file, index_col=0)

# run ward clustering
print(score_matrix.shape)
# # impute the missing values to -5, means they are doing terribly bad
# score_matrix.fillna(0, inplace=True)

# # impute the missing values to the mean value of the column
# score_matrix.fillna(score_matrix.mean(), inplace=True)
score_matrix.fillna(50, inplace=True)


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
