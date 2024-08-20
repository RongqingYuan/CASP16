import seaborn as sns
import matplotlib
import os
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
import factor_analyzer
from sklearn.preprocessing import MinMaxScaler
from sympy import *
from matplotlib import pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity

csv_path = "./csv/"
csv_list = [txt for txt in os.listdir(csv_path) if txt.endswith(".csv")]

csv_file = csv_path + csv_list[3]
print("Processing {}".format(csv_file))
data = pd.read_csv(csv_file, index_col=0)

# print the first 5 rows
print(data.head())

NP_P_remove = False
NP_remove = False
FlexE_remove = False
# NP_P column has issues. if it is all NaN, we will drop it
if data["NP_P"].isnull().all():
    data = data.drop("NP_P", axis=1)
    print("NP_P column is all NaN for {}, so we drop it".format(csv_file))
    NP_P_remove = True
# NP column has issues. if it is all NaN, we will drop it
if data["NP"].isnull().all():
    data = data.drop("NP", axis=1)
    print("NP column is all NaN for {}, so we drop it".format(csv_file))
    NP_remove = True
# FlexE column has issues. if it is all NaN, we will drop it
if data["FlexE"].isnull().all():
    data = data.drop("FlexE", axis=1)
    print("FlexE column is all NaN for {}, so we drop it".format(csv_file))
    FlexE_remove = True

# if every value in NP_P is the same, we will drop it
if not NP_P_remove and data["NP_P"].nunique() == 1:
    data = data.drop("NP_P", axis=1)
    print("NP_P column has only one unique value for {}, so we drop it".format(csv_file))
# if every value in NP is the same, we will drop it
if not NP_remove and data["NP"].nunique() == 1:
    data = data.drop("NP", axis=1)
    print("NP column has only one unique value for {}, so we drop it".format(csv_file))
# if every value in FlexE is the same, we will drop it
if not FlexE_remove and data["FlexE"].nunique() == 1:
    data = data.drop("FlexE", axis=1)
    print("FlexE column has only one unique value for {}, so we drop it".format(csv_file))


print(data.head())

# run kmo test and bartlett test first
# to see if the data is suitable for factor analysis
kmo_all, kmo_model = calculate_kmo(data)
print("KMO: {}".format(kmo_model))
chi_square_value, p_value = calculate_bartlett_sphericity(data)
print("Chi square value: {}".format(chi_square_value))
print("P value: {}".format(p_value))

# this part is almost identical to principal component analysis
Load_Matrix = FactorAnalyzer(
    rotation=None, n_factors=len(data.T), method='principal')
Load_Matrix.fit(data)
f_contribution_var = Load_Matrix.get_factor_variance()
matrices_var = pd.DataFrame()
matrices_var["eigenvalue before rotation"] = f_contribution_var[0]
matrices_var["variance contributed before rotation"] = f_contribution_var[1]
matrices_var["cumulative variance contributed before rotation"] = f_contribution_var[2]
print(matrices_var)
print(Load_Matrix.loadings_)  # 旋转前的成分矩阵
N = 0
eigenvalues = 1
for c in matrices_var["eigenvalue before rotation"]:
    if c >= eigenvalues:
        N += 1
    else:
        s = matrices_var["cumulative variance contributed before rotation"][N-1]
        print("Use {} factors, cumulative variance contributed is {}".format(N, s))
        break

# it is used to see how many factors are appropriate, generally it is taken to the left and right of the smooth place, of course, it is also necessary to combine the contribution rate
# matplotlib.rcParams["font.family"] = "SimHei"
ev, v = Load_Matrix.get_eigenvalues()
print('Eigenvalue:', ev)
plt.figure(figsize=(8, 6.5))
plt.scatter(range(1, data.shape[1] + 1), ev)
plt.plot(range(1, data.shape[1] + 1), ev)
plt.title('Eigenvalues vs number of factors',
          fontdict={'weight': 'normal', 'size': 25})
plt.xlabel('number of factors', fontdict={'weight': 'normal', 'size': 15})
plt.ylabel('Eigenvalues', fontdict={'weight': 'normal', 'size': 15})
plt.grid()
plt.savefig('Eigenvalues_vs_number_of_factors.png')


Load_Matrix_rotated = FactorAnalyzer(
    rotation='varimax', n_factors=N, method='principal')
# Load_Matrix_rotated = FactorAnalyzer(
#     rotation='promax', n_factors=N, method='principal')
# Load_Matrix_rotated = FactorAnalyzer(
#     rotation='varimax', n_factors=N, method='minres')
# Load_Matrix_rotated = FactorAnalyzer(
#     rotation='promax', n_factors=N, method='minres')
Load_Matrix_rotated.fit(data)
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
df = pd.DataFrame(np.abs(Load_Matrix), index=data.columns)
plt.figure(figsize=(12, 9), dpi=300)
ax = sns.heatmap(df, annot=True, cmap="BuPu", cbar=True)
ax.yaxis.set_tick_params(labelsize=9)  # 设置y轴字体大小
plt.title("Factor Analysis (abs)", fontsize="xx-large")
plt.ylabel("factors", fontsize="xx-large")  # 设置y轴标签
# 保存图片
plt.savefig("FA_abs.png")
plt.show()  # 显示图片


df = pd.DataFrame(Load_Matrix, index=data.columns)
plt.figure(figsize=(12, 9), dpi=300)
cmap = sns.diverging_palette(240, 10, as_cmap=True)  # 240°是蓝色，10°是红色
ax = sns.heatmap(df, annot=True, cmap=cmap, center=0, cbar=True)


# ax = sns.heatmap(df, annot=True, cmap="BuPu", cbar=True)
ax.yaxis.set_tick_params(labelsize=9)  # 设置y轴字体大小
plt.title("Factor Analysis", fontsize="xx-large")
plt.ylabel("factors", fontsize="xx-large")  # 设置y轴标签
# 保存图片
plt.savefig("FA.png")
plt.show()  # 显示图片

# 计算因子得分（回归方法）（系数矩阵的逆乘以因子载荷矩阵）
f_corr = data.corr()  # 皮尔逊相关系数
X1 = np.mat(f_corr)
X1 = np.linalg.inv(X1)
factor_score_weight = np.dot(X1, Load_Matrix_rotated.loadings_)
factor_score_weight = pd.DataFrame(factor_score_weight)
col = []
for i in range(N):
    col.append("factor" + str(i + 1))
factor_score_weight.columns = col
factor_score_weight.index = f_corr.columns
print("Factor score:\n", factor_score_weight)


print(Load_Matrix_rotated.get_communalities())
print(Load_Matrix_rotated.transform(data))
# print every row of the factor score

shape = Load_Matrix_rotated.transform(data).shape

print(Load_Matrix_rotated.get_factor_variance())
variance, proportional_variance, cumulative_variance = Load_Matrix_rotated.get_factor_variance()


print(np.dot(Load_Matrix_rotated.transform(data), proportional_variance))
