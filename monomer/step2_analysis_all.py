import sys
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
import scipy.stats as stats

csv_path = "./monomer_data/processed/whole/"
csv_path = "./monomer_data/processed/EU/"
csv_path = "./monomer_data_aug_28/processed/whole/"
csv_path = "./monomer_data_aug_30/processed/EU/"
# csv_path = "./monomer_data/EU/"
png_dir = "./new_png/"
score_dir = "./new_score/"
if not os.path.exists(png_dir):
    os.makedirs(png_dir)

if not os.path.exists(score_dir):
    os.makedirs(score_dir)

csv_list = [txt for txt in os.listdir(csv_path) if txt.endswith(".csv")]
csv_list = [txt for txt in os.listdir(
    csv_path) if txt.endswith(".csv") and txt.startswith("T1")]

# breakpoint()
# csv_file = csv_path + csv_list[3]
# print("Processing {}".format(csv_file))
# data = pd.read_csv(csv_file, index_col=0)

# read all data and concatenate them into one big dataframe
data = pd.DataFrame()
for csv_file in csv_list:
    print("Processing {}".format(csv_file))
    data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
    print(data_tmp.shape)
    if data_tmp.shape[1] == 35:
        print("something wrong with {}".format(csv_file))
        sys.exit(0)
    data = pd.concat([data, data_tmp], axis=0)
# print the first 5 rows
print(data.head())
print(data.shape)
# sys.exit(0)

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


# drop these 3 columns: NP_P, NP, err
data = data.drop(["NP_P", "NP", "err"], axis=1)

# drop these two columns:Z-M1-GDT Z-MA-GDT
data = data.drop(["Z-M1-GDT", "Z-MA-GDT"], axis=1)
# save the data to csv file
data.to_csv("./monomer_data_aug_28/all/fa_processed_all.csv")

# get the mean and std of the data
print("Mean of the data")
print(data.mean())
print("Std of the data")
print(data.std())
# z-score normalization
data = (data - data.mean()) / data.std()
print("Mean of the data")
print(data.mean())
print("Std of the data")
print(data.std())
# z-score normalization
# breakpoint()
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
N = 3
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
plt.savefig(png_dir+'Eigenvalues_vs_number_of_factors.png', dpi=300)


# Load_Matrix_rotated = FactorAnalyzer(
#     rotation='varimax', n_factors=N, method='principal')
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
plt.savefig(png_dir+"FA_abs.png")
plt.show()  # 显示图片


df = pd.DataFrame(Load_Matrix, index=data.columns)
plt.figure(figsize=(12, 9), dpi=300)
cmap = sns.diverging_palette(240, 10, as_cmap=True)  # 240°是蓝色，10°是红色
ax = sns.heatmap(df, annot=True, cmap=cmap, center=0, cbar=True)


# ax = sns.heatmap(df, annot=True, cmap="BuPu", cbar=True)
ax.yaxis.set_tick_params(labelsize=10)  # 设置y轴字体大小
plt.title("Factor Analysis", fontsize="xx-large")
# plt.ylabel("factors", fontsize="xx-large")  # 设置y轴标签
plt.xlabel("factors", fontsize="xx-large")  # 设置x轴标签
# set the x-axis to be factor1, factor2, ...
ax.set_xticklabels(["factor" + str(i + 1) for i in range(N)], fontsize=10)
# 保存图片
plt.savefig(png_dir + "FA_{}.png".format(N))
# plt.show()  # 显示图片
# sys.exit(0)
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
raw_score = Load_Matrix_rotated.transform(data)
score = np.dot(Load_Matrix_rotated.transform(data), proportional_variance)
# save the score to csv file
print(np.dot(Load_Matrix_rotated.transform(data), proportional_variance).shape)
# get the row labels
print(data.index)
print(raw_score.shape)
print(proportional_variance)
print(raw_score.T[0].shape)
print(data.index.shape)
high_resolution_score = raw_score.T[0]
chemical_score = raw_score.T[1]
low_resolution_score = raw_score.T[2]
# get a new dataframe using data.index as the index, high_resolution_score, chemical_score, low_resolution_score as the columns
score_df = pd.DataFrame(
    {"high_resolution_score": high_resolution_score, "chemical_score": chemical_score, "low_resolution_score": low_resolution_score}, index=data.index)
score_df.to_csv("./tmp/" + "score_df_0.csv")

# score_df.index = score_df.index.str.extract(
#     r'(?P<Level1>.+TS)(?P<Level2>\d+)_(?P<Level3>\d+)')
# score_df.index = score_df.index.str.extract(r'(T\d+)(TS)(\d+)(?:_(\d+))?(?:-(D\d+))?') # does not work

score_df.index = score_df.index.str.extract(
    r'(T\w+)TS(\w+)_(\w+)-(D\w+)').apply(lambda x: (f"{x[0]}-{x[3]}", f"TS{x[1]}", x[2]), axis=1)

print(score_df.shape)
print(score_df.head())
score_df.to_csv("./tmp/" + "score_df_1.csv")
score_df.index = pd.MultiIndex.from_tuples(
    score_df.index, names=['target', 'group', 'submission_id'])
# grouped = score_df.groupby(["target", "group"])
grouped = score_df.groupby(["group", "target"])
print(grouped.head())
print(grouped.head())
print(type(grouped))
print(grouped["high_resolution_score"].max())
print(type(grouped["high_resolution_score"].max()))
print(pd.DataFrame(grouped["high_resolution_score"].max()))


grouped = pd.DataFrame(grouped["low_resolution_score"].max())
grouped.to_csv("./tmp/" + "grouped.csv")
# grouped = pd.DataFrame(grouped["low_resolution_score"].max())
# grouped = pd.DataFrame(grouped["chemical_score"].max())
# sum the scores for each group
# grouped["group"] = grouped.index.get_level_values("group")
grouped_by_target = grouped.stack().unstack('target')
sum_grouped_by_target = grouped_by_target.sum(axis=1)
grouped_by_target["sum"] = sum_grouped_by_target
grouped_by_target = grouped_by_target.sort_values(by="sum", ascending=False)
grouped_by_target.to_csv("./tmp/" + "grouped_reset.csv")
print(grouped_by_target.head())
# breakpoint()
print(grouped_by_target.head())
# sort grouped by value
# grouped_by_target = grouped_by_target.sort_values(by="low_resolution_score", ascending=False)
# grouped = grouped.sort_values(by="low_resolution_score", ascending=False)
# grouped = grouped.sort_values(by="chemical_score", ascending=False)
print(grouped_by_target)
for k, v in grouped_by_target.iterrows():
    print(k, v)

# breakpoint()
print(score_df.index)
measure_of_interest = "chemical_score"
measure_of_interest = "low_resolution_score"
measure_of_interest = "high_resolution_score"
score_df = pd.DataFrame(score_df[measure_of_interest])
score_df.to_csv("./tmp/" + "score_df_2.csv")
data_whole = score_df.stack().unstack('group')
data_whole.index = [f'{target}_{submission_id}' for target,
                    submission_id, measure in data_whole.index]
print(data_whole.head())
# get a new column called submission_id and target
data_whole['target'] = data_whole.index.str.split('_').str[0]
data_whole['submission_id'] = data_whole.index.str.split('_').str[1]
# remove any submission_id = 6
data_whole = data_whole[data_whole['submission_id'] != '6']
data_whole_by_target = data_whole.groupby('target').max()
data_whole_by_target.to_csv("./tmp/" + "data_whole_by_target.csv")

wanted_group = ["052", "022", "456", "051",
                "319", "287", "208", "028", "019", "294", "465", "110", "345", "139"]
wanted_group = ["TS"+group for group in wanted_group]

points = {}

wanted_group = [
    group for group in data_whole_by_target.columns if group.startswith("TS")]

for group_1 in wanted_group:
    for group_2 in wanted_group:
        if group_1 == group_2:
            continue
        data_1 = data_whole_by_target[group_1]
        data_1 = pd.DataFrame(data_1)
        # fill missing values with 0
        data_1.fillna(0, inplace=True)
        data_2 = data_whole_by_target[group_2]
        data_2 = pd.DataFrame(data_2)
        # fill missing values with 0
        data_2.fillna(0, inplace=True)
        try:
            t_stat, p_val = stats.ttest_rel(data_1, data_2)
        except:
            breakpoint()
        print(
            f"Group {group_1} vs Group {group_2}: t_stat={t_stat}, p_val={p_val}")
        if group_1 not in points:
            points[group_1] = 0
        if t_stat > 0 and p_val/2 < 0.05:
            points[group_1] += 1
points = dict(sorted(points.items(), key=lambda item: item[1], reverse=True))
print(points)

print(grouped_by_target)
# drop the seconde level of the index
grouped_by_target = grouped_by_target.droplevel(1)
print(grouped_by_target)

# for k, v in grouped_by_target.iterrows():
#     print(k, v)

# convert the sum column to a dictionary
sum_dict = grouped_by_target["sum"].to_dict()
print(sum_dict)

sys.exit(0)


def fill_missing(group):
    # 计算组内每列的均值（不包括 target 和 submission_id 列）
    means = group.drop(columns=['submission_id']).mean()

    # 用均值填补缺失值
    filled_group = group.fillna(means)

    # 如果整组数据都是缺失值，则用0填补
    filled_group = filled_group.fillna(0)

    return filled_group


data_whole_by_target = data_whole_by_target.apply(
    fill_missing).reset_index(drop=True)
sys.exit(0)

score_dict = {}
for i in range(shape[0]):
    if i % 1000 == 0:
        print(i)
    score_dict[data.index[i]] = score[i]


group_dict = {}  # {group_id: [score1, score2, ...]}
group_dict_domain = {}  # {group_id: [score1, score2, ...]}
group_target_dict = {}  # {stage_target_group: [score1, score2, ...]}
group_target_domain_dict = {}  # {stage_target_group: [score1, score2, ...]}
group_first_dict = {}
group_first_domain_dict = {}

for k, v in score_dict.items():
    info = k.split("TS")
    target_info = info[0]
    group_info = info[1]

    if len(target_info) == 5:
        protein_type = target_info[0]
        stage = target_info[1]
        target_id = target_info[2:]
    else:
        protein_type = target_info[0]
        stage = target_info[1]
        target_id = target_info[2:]
        print(k)

    group_info = group_info.split("-")
    if len(group_info) == 1:  # it is not a domain
        group_id = group_info[0].split("_")[0]
        submission_rank = group_info[0].split("_")[1]
        if group_id not in group_dict:
            group_dict[group_id] = []
        group_dict[group_id].append(float(v))
        stage_target_group = stage + "_" + target_id + "_" + group_id
        if stage_target_group not in group_target_dict:
            group_target_dict[stage_target_group] = []
        group_target_dict[stage_target_group].append(float(v))
        if submission_rank == "1":
            if group_id not in group_first_dict:
                group_first_dict[group_id] = []
            group_first_dict[group_id].append(float(v))

    elif len(group_info) == 2:  # domain
        group_id = group_info[0].split("_")[0]
        submission_rank = group_info[0].split("_")[1]
        domain = group_info[1]
        if group_id not in group_dict_domain:
            group_dict_domain[group_id] = []
        group_dict_domain[group_id].append(float(v))

        stage_target_group = stage + "_" + target_id + "_" + group_id
        if stage_target_group not in group_target_domain_dict:
            group_target_domain_dict[stage_target_group] = []
        group_target_domain_dict[stage_target_group].append(float(v))

        if submission_rank == "1":
            if group_id not in group_first_domain_dict:
                group_first_domain_dict[group_id] = []
            group_first_domain_dict[group_id].append(float(v))

# print(group_dict.__len__())
# print(group_dict_domain.__len__())
# for k, v in group_dict.items():
#     print(k)
# print("Domain")
# print("#####")
# for k, v in group_dict_domain.items():
#     print(k)

# print(group_target_dict)


group_target_domain_best_dict = {}
# {group_id: [best_score_target1, best_score_target2, ...]}
group_scores_domain_dict = {}

for k, v in group_target_domain_dict.items():
    group_id = k.split("_")[2]
    group_target_domain_best_dict[k] = max(v)
    if group_id not in group_scores_domain_dict:
        group_scores_domain_dict[group_id] = []
    group_scores_domain_dict[group_id].append(max(v))

print(group_target_domain_best_dict)
print(group_scores_domain_dict.__len__())
for k, v in group_scores_domain_dict.items():
    print(k)
    print(len(v))

group_domains_score_dict = {}
for k, v in group_scores_domain_dict.items():
    group_domains_score_dict[k] = sum(v)
    print(len(v))

# sort the dictionary by value
group_domains_score_dict = dict(
    sorted(group_domains_score_dict.items(), key=lambda item: item[1], reverse=True))
with open(score_dir + "domain_score_best_model.txt", "w") as f:
    for k, v in group_domains_score_dict.items():
        # 3 digits
        f.write("Group {}: {}\n".format(k, round(v, 3)))

# sort the dictionary by value
group_first_domain_dict = dict(
    sorted(group_first_domain_dict.items(), key=lambda item: sum(item[1]), reverse=True))
with open(score_dir + "domain_score_first_model.txt", "w") as f:
    for k, v in group_first_domain_dict.items():
        # 3 digits
        f.write("Group {}: {}\n".format(k, round(sum(v), 3)))

# {'456': 52.47606938219763, '022': 50.6724223587766, '051': 48.73931303606964, '345': 48.28194568763339, '110': 47.71679543723419, '052': 46.8975765339637, '462': 45.710855410626614, '241': 44.74558722074533, '294': 43.855970611855035, '147': 41.09445883711779, '019': 41.08698706682758, '148': 40.51538394275573, '221': 40.2248087885984, '312': 39.469134772318355, '419': 37.687393286012885, '208': 37.615120255053895, '015': 37.15989732821828, '304': 37.04620612803516, '267': 36.13535508918436, '028': 34.87433664764452, '264': 34.03526046432325, '331': 31.67392497759025, '319': 31.455604182552012, '425': 29.02148174464271, '301': 28.399242292818368, '314': 27.783778290668625, '284': 27.763582052786905, '287': 26.71726906703062, '164': 26.383913132354174, '375': 26.214845284836873, '298': 25.52631457252373, '075': 25.006934472501932, '122': 24.920217409144076, '293': 24.754981478363906, '235': 23.68968661658768, '269': 22.589412826209564, '465': 22.476812700915094, '475': 21.979739084968724, '079': 21.967712024049405, '163': 21.73381146710162, '272': 20.872987380672345, '286': 19.545451184260294, '369': 18.173555519771128, '311': 17.082711698787023, '262': 15.30233176780295, '023': 15.057760876456797, '393': 14.49786631767398, '290': 13.350465239660016, '145': 12.934003386776393, '423': 12.630258963837901, '171': 12.23612574607281, '091': 11.971355056536567, '388': 11.704657479932232, '322': 11.588424321912392, '274': 11.241658588262776, '204': 11.233727528015073, '494': 10.893696089005626, '196': 10.135482069140751, '450': 10.049474051120782, '014': 9.42633564723013, '059': 9.14008265844257, '397': 9.04575103141413, '198': 7.048101344791274, '218': 6.2810544732665265, '489': 6.203965828692558, '191': 6.136138424447427, '380': 5.790099873562435, '031': 5.701016775855994, '481': 5.649941875032565, '358': 5.087828548960176, '219': 4.957992552039616, '167': 4.475760950069823, '323': 3.608044016550222, '468': 3.3105477549994045, '325': 2.6899676577688556, '338': 2.5630876696177904, '159': 2.5349973983245255, '085': 1.5289398861129344, '276': 1.3078224753604597, '189': 1.2861426745153284, '033': 1.0910284453650996, '376': 1.0523919511927815, '351': 0.7619487734307631, '008': 0.7311567340882534, '231': 0.5999491156372105, '143': 0.3729567717888606, '174': 0.33347629003937546, '225': 0.28631994414115725, '386': 0.28470543419013344, '049': 0.2747199323098795, '271': 0.25011441051622557, '017': 0.043597385795303256, '112': -0.06232390713420108, '355': -0.06888844249273791, '400': -0.129219005443159, '187': -0.7586178597340996, '281': -0.9328895855333783, '132': -1.2657692083328769, '040': -1.8398811434965614, '117': -2.8439726650753845, '357': -3.855013663217005, '337': -10.921967785032171, '300': -13.279211201471558, '114': -14.246781439489908, '212': -22.506397321098152, '139': -25.86372494967308, '361': -35.48521109616379, '261': -36.158921415672765, '120': -43.59309011372384}

group_target_best_dict = {}
# {group_id: [best_score_target1, best_score_target2, ...]}
group_scores_dict = {}
for k, v in group_target_dict.items():
    group_id = k.split("_")[2]
    group_target_best_dict[k] = max(v)
    if group_id not in group_scores_dict:
        group_scores_dict[group_id] = []
    group_scores_dict[group_id].append(max(v))

print(group_target_best_dict)
print(group_scores_dict.__len__())
for k, v in group_scores_dict.items():
    print(k)
    print(len(v))

group_score_dict = {}
for k, v in group_scores_dict.items():
    group_score_dict[k] = sum(v)
    print(len(v))

# sort the dictionary by value
group_score_dict = dict(
    sorted(group_score_dict.items(), key=lambda item: item[1], reverse=True))
with open(score_dir + "score_best_model.txt", "w") as f:
    for k, v in group_score_dict.items():
        # 3 digits
        f.write("Group {}: {}\n".format(k, round(v, 3)))

# sort the dictionary by value
group_first_dict = dict(
    sorted(group_first_dict.items(), key=lambda item: sum(item[1]), reverse=True))
with open(score_dir + "score_first_model.txt", "w") as f:
    for k, v in group_first_dict.items():
        # 3 digits
        f.write("Group {}: {}\n".format(k, round(sum(v), 3)))
