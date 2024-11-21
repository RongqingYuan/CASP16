import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


features = ['GDT_TS', 'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
            'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
            'SphGr',
            'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
            'MP_rotout', 'MP_ramout', 'MP_ramfv', 'reLLG_lddt', 'reLLG_const']

mode = "all"
# mode = "hard"
# mode = "medium"
# mode = "easy"

model = "first"
model = "best"

path = "./sum/"
path = "./sum_EU/"
impute_value = -2
data_all = pd.DataFrame()
for feature in features:
    file = "./sum_EU/" + "impute={}/".format(impute_value) + \
        "sum_unweighted_{}-{}-{}.csv".format(feature, model, mode)
    data = pd.read_csv(file, index_col=0)

    # drop sum column
    data = data.drop(columns=['sum'])
    # breakpoint()
    data = data.stack()
    # breakpoint()
    data = pd.DataFrame(data)
    data.columns = [feature]
    # data.index = data.index.map(lambda x: "-".join(x))
    data.to_csv(
        "./by_EU/" + "sum_{}-{}-{}_expanded.csv".format(feature, model, mode))
    # breakpoint()
    data_all = pd.concat([data_all, data], axis=1)

breakpoint()

data_all.to_csv(
    "./by_EU/" + "sum_all-{}-{}_expanded.csv".format(model, mode))
# drop rows with all -2 values
data_all_exist = data_all[~(data_all == -2.0).all(axis=1)]
# 交换 MolPrb_Score 和 CAD_SS 两列
cols = list(data_all_exist.columns)
# 获取 MolPrb_Score 和 CAD_SS 的索引位置
idx1, idx2 = cols.index('MolPrb_Score'), cols.index('CAD_SS')
# 交换列的位置
cols[idx1], cols[idx2] = cols[idx2], cols[idx1]
# 重新排列 DataFrame 的列
data_all_exist_new = data_all_exist[cols]

# swap these two measure and the data: MolPrb_Score and CAD_SS
# get the correlation matrix of the data
correlation_matrix = data_all_exist_new.corr()
# plot the correlation matrix
plt.figure(figsize=(18, 16), dpi=300)
cmap = sns.diverging_palette(240, 10, as_cmap=True)  # 240°是蓝色，10°是红色
ax = sns.heatmap(correlation_matrix, annot=True,
                 cmap=cmap, cbar=True, center=0, square=True)
ax.yaxis.set_tick_params(labelsize=20)  # 设置y轴字体大小
ax.xaxis.set_tick_params(labelsize=20)  # 设置x轴字体大小
plt.title("Correlation Matrix", fontsize=22)
# set the font of scale bar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

plt.savefig("./by_EU/" + "correlation_matrix.png", dpi=300)


# # NMF data_all_exist
# N = 3
# model = NMF(n_components=N, init='random', random_state=0)
# W = model.fit_transform(data_all_exist_new)
# H = model.components_

# breakpoint()


N = 5
# Load_Matrix_rotated = FactorAnalyzer(
#     rotation='varimax', n_factors=N, method='principal')
# Load_Matrix_rotated = FactorAnalyzer(
#     rotation='promax', n_factors=N, method='principal')
# Load_Matrix_rotated = FactorAnalyzer(
#     rotation='promax', n_factors=N, method='principal')
# Load_Matrix_rotated = FactorAnalyzer(
#     rotation='varimax', n_factors=N, method='minres')
# Load_Matrix_rotated = FactorAnalyzer(
#     rotation='promax', n_factors=N, method='minres')

Load_Matrix_rotated = FactorAnalyzer(
    rotation='promax', n_factors=N, method='principal', is_corr_matrix=False)
Load_Matrix_rotated.fit(data_all_exist)
f_contribution_var_rotated = Load_Matrix_rotated.get_factor_variance()
matrices_var_rotated = pd.DataFrame()
matrices_var_rotated["eigenvalue"] = f_contribution_var_rotated[0]
matrices_var_rotated["variance contributed"] = f_contribution_var_rotated[1]
matrices_var_rotated["cumulative variance contributed"] = f_contribution_var_rotated[2]
print("loading matrix data after rotation")
print(matrices_var_rotated)
print("loading matrix after rotation")
print(Load_Matrix_rotated.loadings_)

Load_Matrix = Load_Matrix_rotated.loadings_
df = pd.DataFrame(np.abs(Load_Matrix), index=data_all.columns)
plt.figure(figsize=(12, 9), dpi=300)
ax = sns.heatmap(df, annot=True, cmap="BuPu", cbar=True)
ax.yaxis.set_tick_params(labelsize=9)  # 设置y轴字体大小
plt.title("Factor Analysis (abs)", fontsize="xx-large")
plt.ylabel("factors", fontsize="xx-large")  # 设置y轴标签
# 保存图片
plt.savefig("./by_EU/" + "FA_abs.png", dpi=300)


df = pd.DataFrame(Load_Matrix, index=data_all.columns)
plt.figure(figsize=(12, 9), dpi=300)
cmap = sns.diverging_palette(240, 10, as_cmap=True)  # 240°是蓝色，10°是红色
ax = sns.heatmap(df, annot=True, cmap=cmap, center=0,
                 cbar=True, annot_kws={"fontsize": 15})
# ax = sns.heatmap(df, annot=True, cmap="BuPu", cbar=True)
ax.yaxis.set_tick_params(labelsize=15)  # 设置y轴字体大小
plt.title("Loading Matrix", fontsize=15)
plt.xlabel("factors", fontsize=15)  # 设置x轴标签
# plt.xlabel("factors", fontsize="xx-large")  # 设置x轴标签
# set the x-axis to be factor1, factor2, ...
ax.set_xticklabels(["Factor " + str(i + 1) for i in range(N)], fontsize=15)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.savefig("./by_EU/" + "FA_{}.png".format(N), dpi=300)


M = Load_Matrix_rotated.loadings_
N_ = np.dot(M.T, M)
N_inv = np.linalg.inv(N_)
F_mat = np.dot(N_inv, M.T)
# breakpoint()
F_mat_T = F_mat.T
F_mat_T = pd.DataFrame(F_mat_T, index=data_all.columns)

plt.figure(figsize=(8, 6), dpi=300)
cmap = sns.diverging_palette(240, 10, as_cmap=True)  # 240°是蓝色，10°是红色
ax = sns.heatmap(F_mat_T, annot=True, cmap=cmap, center=0, cbar=True)
# ax = sns.heatmap(df, annot=True, cmap="BuPu", cbar=True)
ax.yaxis.set_tick_params(labelsize=10)  # 设置y轴字体大小
plt.title("Weight Matrix", fontsize=10)
plt.xlabel("factors", fontsize=10)  # 设置x轴标签
# plt.xlabel("factors", fontsize="xx-large")  # 设置x轴标签
# set the x-axis to be factor1, factor2, ...
ax.set_xticklabels(["factor" + str(i + 1) for i in range(N)], fontsize=10)
plt.savefig("./by_EU/" + "weight_FA_{}.png".format(N), dpi=300)


regression_score = data_all@F_mat.T
high_resolution_score = regression_score[2]
chemical_score = regression_score[1]
low_resolution_score = regression_score[0]
score_df = pd.DataFrame(
    {"low_resolution_score": low_resolution_score, "chemical_score": chemical_score, "high_resolution_score": high_resolution_score}, index=data_all.index)

# group by the first level of the index
score_df = score_df.groupby(level=0).sum()
score_df.to_csv(
    "./by_EU/" + "factor_score_all-{}-{}_expanded.csv".format(model, mode))
# sort the score_df by the first level of the index
score_df = score_df.sort_values(by="low_resolution_score", ascending=False)
score_df.to_csv(
    "./by_EU/" + "factor_score_all-{}-{}_expanded_sorted.csv".format(model, mode))
# breakpoint()
# score_df.index = score_df.index.str.extract(
#     r'(T\w+)TS(\w+)_(\w+)-(D\w+)').apply(lambda x: (f"{x[0]}-{x[3]}", f"TS{x[1]}", x[2]), axis=1)

# plot the score_df
# plot first column vs second column
plt.figure(figsize=(10, 15))
plt.scatter(score_df["low_resolution_score"],
            score_df["chemical_score"])
plt.xlabel("low_resolution_score", fontsize=15)
plt.ylabel("chemical_score", fontsize=15)
plt.title("low_resolution_score vs chemical_score", fontsize=15)
# for i in range(score_df.shape[0]):
#     plt.text(score_df["low_resolution_score"].iloc[i],
#              score_df["chemical_score"].iloc[i], score_df.index[i])
plt.savefig("./by_EU/" + "low_resolution_score_vs_chemical_score.png", dpi=300)

# plot first column vs third column
plt.figure(figsize=(10, 15))
plt.scatter(score_df["low_resolution_score"],
            score_df["high_resolution_score"])
plt.xlabel("low_resolution_score", fontsize=15)
plt.ylabel("high_resolution_score", fontsize=15)
plt.title("low_resolution_score vs high_resolution_score", fontsize=15)
# for i in range(score_df.shape[0]):
#     plt.text(score_df["low_resolution_score"].iloc[i],
#              score_df["high_resolution_score"].iloc[i], score_df.index[i])
plt.savefig(
    "./by_EU/" + "low_resolution_score_vs_high_resolution_score.png", dpi=300)

# plot second column vs third column
plt.figure(figsize=(10, 15))
plt.scatter(score_df["chemical_score"],
            score_df["high_resolution_score"])
plt.xlabel("chemical_score", fontsize=15)
plt.ylabel("high_resolution_score", fontsize=15)
plt.title("chemical_score vs high_resolution_score", fontsize=15)
for i in range(score_df.shape[0]):
    plt.text(score_df["chemical_score"].iloc[i],
             score_df["high_resolution_score"].iloc[i], score_df.index[i])

plt.savefig("./by_EU/" + "chemical_score_vs_high_resolution_score.png", dpi=300)
