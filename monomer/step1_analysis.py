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

csv_file = csv_path + csv_list[1]
print("Processing {}".format(csv_file))
data = pd.read_csv(csv_file, index_col=0)

# print the first 5 rows
print(data.head())

# NP_P column has issues. if it is all NaN, we will drop it
if data["NP_P"].isnull().all():
    data = data.drop("NP_P", axis=1)
    print("NP_P column is all NaN for {}, so we drop it".format(csv_file))
# NP column has issues. if it is all NaN, we will drop it
if data["NP"].isnull().all():
    data = data.drop("NP", axis=1)
    print("NP column is all NaN for {}, so we drop it".format(csv_file))
# FlexE column has issues. if it is all NaN, we will drop it
if data["FlexE"].isnull().all():
    data = data.drop("FlexE", axis=1)
    print("FlexE column is all NaN for {}, so we drop it".format(csv_file))

# if every value in NP_P is the same, we will drop it
if data["NP_P"].nunique() == 1:
    data = data.drop("NP_P", axis=1)
    print("NP_P column has only one unique value for {}, so we drop it".format(csv_file))
# if every value in NP is the same, we will drop it
if data["NP"].nunique() == 1:
    data = data.drop("NP", axis=1)
    print("NP column has only one unique value for {}, so we drop it".format(csv_file))
# # if every value in FlexE is the same, we will drop it
# if data["FlexE"].nunique() == 1:
#     data = data.drop("FlexE", axis=1)
#     print("FlexE column has only one unique value for {}, so we drop it".format(csv_file))


print(data.head())

# run kmo test and bartlett test first
kmo_all, kmo_model = calculate_kmo(data)
print("KMO: {}".format(kmo_model))
chi_square_value, p_value = calculate_bartlett_sphericity(data)
print("Chi square value: {}".format(chi_square_value))
print("P value: {}".format(p_value))

Load_Matrix = FactorAnalyzer(
    rotation=None, n_factors=len(data.T), method='principal')
Load_Matrix.fit(data)
f_contribution_var = Load_Matrix.get_factor_variance()
matrices_var = pd.DataFrame()
matrices_var["旋转前特征值"] = f_contribution_var[0]
matrices_var["旋转前方差贡献率"] = f_contribution_var[1]
matrices_var["旋转前方差累计贡献率"] = f_contribution_var[2]
print(matrices_var)
print(Load_Matrix.loadings_)  # 旋转前的成分矩阵
N = 0
eigenvalues = 1
for c in matrices_var["旋转前特征值"]:
    if c >= eigenvalues:
        N += 1
    else:
        s = matrices_var["旋转前方差累计贡献率"][N-1]
        print("\n选择了" + str(N) + "个因子累计贡献率为" + str(s)+"\n")
        break

# 主要用来看取多少因子合适，一般是取到平滑处左右，当然还要需要结合贡献率
# matplotlib.rcParams["font.family"] = "SimHei"
ev, v = Load_Matrix.get_eigenvalues()
print('\n相关矩阵特征值：', ev)
plt.figure(figsize=(8, 6.5))
plt.scatter(range(1, data.shape[1] + 1), ev)
plt.plot(range(1, data.shape[1] + 1), ev)
plt.title('特征值和因子个数的变化', fontdict={'weight': 'normal', 'size': 25})
plt.xlabel('因子', fontdict={'weight': 'normal', 'size': 15})
plt.ylabel('特征值', fontdict={'weight': 'normal', 'size': 15})
plt.grid()
plt.savefig('特征值和因子个数的变化.png')


# Load_Matrix_rotated = FactorAnalyzer(
#     rotation='varimax', n_factors=N, method='principal')
Load_Matrix_rotated = FactorAnalyzer(
    rotation='promax', n_factors=N, method='minres')
Load_Matrix_rotated.fit(data)
f_contribution_var_rotated = Load_Matrix_rotated.get_factor_variance()
matrices_var_rotated = pd.DataFrame()
matrices_var_rotated["特征值"] = f_contribution_var_rotated[0]
matrices_var_rotated["方差贡献率"] = f_contribution_var_rotated[1]
matrices_var_rotated["方差累计贡献率"] = f_contribution_var_rotated[2]
print("旋转后的载荷矩阵的贡献率")
print(matrices_var_rotated)
print("旋转后的成分矩阵")
print(Load_Matrix_rotated.loadings_)

Load_Matrix = Load_Matrix_rotated.loadings_
df = pd.DataFrame(np.abs(Load_Matrix), index=data.columns)
plt.figure(figsize=(8, 8))
ax = sns.heatmap(df, annot=True, cmap="BuPu", cbar=False)
ax.yaxis.set_tick_params(labelsize=15)  # 设置y轴字体大小
plt.title("因子分析", fontsize="xx-large")
plt.ylabel("因子", fontsize="xx-large")  # 设置y轴标签
# 保存图片
plt.savefig("因子分析.png")
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
print("因子得分：\n", factor_score_weight)


print(Load_Matrix_rotated.get_communalities())
print(Load_Matrix_rotated.transform(data))
