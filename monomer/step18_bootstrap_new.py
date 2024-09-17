import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys


score_path = "./group_by_target_EU/"
measure = "RMSD[L]"
measure = "RMS_CA"
measure = "GDT_HA"
measure = "GDT_TS"
measure = sys.argv[1]

mode = "hard"
mode = "medium"
mode = "easy"
mode = "all"
model = "first"
model = "best"

measures = ['GDT_TS', 'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
            'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
            'SphGr',
            'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
            'MP_rotout', 'MP_ramout', 'MP_ramfv', 'reLLG_lddt', 'reLLG_const']
hard_group = [
    "T1207-D1",
    "T1210-D1",
    "T1210-D2",
    "T1220s1-D1",
    "T1220s1-D2",
    "T1226-D1",
    "T1228v1-D3",
    "T1228v1-D4",
    "T1228v2-D3",
    "T1228v2-D4",
    "T1239v1-D4",
    "T1239v2-D4",
    "T1271s1-D1",
    "T1271s3-D1",
    "T1271s8-D1",
]

medium_group = [
    "T1210-D3",
    "T1212-D1",
    "T1218-D1",
    "T1218-D2",
    "T1227s1-D1",
    "T1228v1-D1",
    "T1228v2-D1",
    "T1230s1-D1",
    "T1237-D1",
    "T1239v1-D1",
    "T1239v1-D3",
    "T1239v2-D1",
    "T1239v2-D3",
    "T1243-D1",
    "T1244s1-D1",
    "T1244s2-D1",
    "T1245s2-D1",
    "T1249v1-D1",
    "T1257-D3",
    "T1266-D1",
    "T1270-D1",
    "T1270-D2",
    "T1267s1-D1",
    "T1267s1-D2",
    "T1267s2-D1",
    "T1269-D1",
    "T1269-D2",
    "T1269-D3",
    "T1271s2-D1",
    "T1271s4-D1",
    "T1271s5-D1",
    "T1271s5-D2",
    "T1271s7-D1",
    "T1271s8-D2",
    "T1272s2-D1",
    "T1272s6-D1",
    "T1272s8-D1",
    "T1272s9-D1",
    "T1279-D2",
    "T1284-D1",
    "T1295-D1",
    "T1295-D3",
    "T1298-D1",
    "T1298-D2",
]

easy_group = [
    "T1201-D1",
    "T1201-D2",
    "T1206-D1",
    "T1208s1-D1",
    "T1208s2-D1",
    "T1218-D3",
    "T1228v1-D2",
    "T1228v2-D2",
    "T1231-D1",
    "T1234-D1",
    "T1235-D1",
    "T1239v1-D2",
    "T1239v2-D2",
    "T1240-D1",
    "T1240-D2",
    "T1245s1-D1",
    "T1246-D1",
    "T1257-D1",
    "T1257-D2",
    "T1259-D1",
    "T1271s6-D1",
    "T1274-D1",
    "T1276-D1",
    "T1278-D1",
    "T1279-D1",
    "T1280-D1",
    "T1292-D1",
    "T1294v1-D1",
    "T1294v2-D1",
    "T1295-D2",
    # "T1214",
]


# # 假设 T1_data 中有一列 'target_column' 用于分层

# def stratified_sample(df, stratify_column, frac=1, replace=True):
#     # 按照 stratify_column 的值进行分组，然后在每个组内进行采样
#     return df.groupby(stratify_column, group_keys=False).apply(
#         lambda x: x.sample(frac=frac, replace=replace)
#     )


# # 进行分层采样
# # T1_data_bootstrap = stratified_sample(T1_data, stratify_column='target', frac=1, replace=True)


def bootstrap(measure, mode, model, p_value_threshold=0.05):
    # score_file = "groups_by_targets_for-raw-{}-EU.csv".format(measure)
    score_file = "group_by_target-{}-{}-{}.csv".format(measure, model, mode)
    score_matrix = pd.read_csv(score_path + score_file, index_col=0)
    # score_matrix = score_matrix.T
    T1_data = score_matrix.filter(regex='T1')

    # print(score_matrix.shape)
    # print(score_matrix.head())
    # print(T1_data.shape)
    # print(T1_data.head())

    # T1_data_tmp = T1_data.copy()
    # T1_data_tmp.fillna(0, inplace=True)
    # sum = T1_data_tmp.sum(axis=1)
    # T1_data_tmp['sum'] = sum
    # T1_data_tmp = T1_data_tmp.sort_values(by='sum', ascending=False)

    if mode == "all":
        pass
    elif mode == "easy":
        T1_data = T1_data[easy_group]
    elif mode == "medium":
        T1_data = T1_data[medium_group]
    elif mode == "hard":
        T1_data = T1_data[hard_group]

    T1_data = T1_data.T
    # print(T1_data.shape)
    # print(T1_data.head())

    # remove columns with more than 80% missing values
    T1_data = T1_data.loc[:, T1_data.isnull().mean() < 0.8]
    if "raw" in score_file:
        T1_data.fillna(50, inplace=True)
    else:
        T1_data.fillna(-2, inplace=True)
    # print(T1_data.shape)
    # print(T1_data.head())
    groups = T1_data.columns
    points = {}
    length = len(groups)

    for i in range(length):
        for j in range(length):
            group_1 = groups[i]
            group_2 = groups[j]
            if group_1 == group_2:
                continue
            group_1_data = T1_data[group_1]
            group_2_data = T1_data[group_2]
            t_stat, p_value = stats.ttest_rel(group_1_data, group_2_data)
            # print("Group 1: {}, Group 2: {}, t-stat: {}, p-value: {}".format(group_1, group_2, t_stat, p_value))
            if t_stat > 0 and p_value/2 < p_value_threshold:
                if group_1 not in points:
                    points[group_1] = 0
                points[group_1] += 1

    points = dict(sorted(points.items(), key=lambda x: x[1], reverse=True))
    with open("./bootstrap_EU/{}_{}_{}_p={}_ranking_step18.txt".format(measure, model, mode, p_value_threshold), 'w') as f:
        f.write(str(points))
    # use the above t-test code to get a initial ranking of the groups.
    # then generate new groups list using the ranking
    # then do bootstrapping
    groups = list(points.keys())
    # breakpoint()

    points = {}
    length = len(groups)
    win_matrix = [[0 for i in range(length)] for j in range(length)]
    bootstrap_rounds = 1000
    # get the target list, it is the first element when split T1_data.index
    targets = T1_data.index.map(lambda x: x.split("-")[0])
    T1_data["target"] = targets
    # breakpoint()
    for r in range(bootstrap_rounds):
        # T1_data_bootstrap = T1_data.sample(frac=1, replace=True)
        # T1_data_bootstrap = T1_data.groupby('target', group_keys=False).apply(lambda x: x.sample(n=1))
        grouped = T1_data.groupby('target')
        T1_data_bootstrap = grouped.apply(lambda x: x.sample(
            n=1)).sample(n=len(grouped), replace=True)
        # sort the T1_data_bootstrap rows by the index
        T1_data_bootstrap = T1_data_bootstrap.sort_index()

        for i in range(length):
            for j in range(length):
                group_1 = groups[i]
                group_2 = groups[j]
                if group_1 == group_2:
                    continue
                group_1_data = T1_data_bootstrap[group_1]
                group_2_data = T1_data_bootstrap[group_2]
                # print all value in group_1_data
                # for k in range(group_1_data.shape[0]):
                #     print(group_1_data.iloc[k])
                # run paired t-test
                t_stat, p_value = stats.ttest_rel(group_1_data, group_2_data)
                # print("Group 1: {}, Group 2: {}, t-stat: {}, p-value: {}".format(group_1, group_2, t_stat, p_value))
                if t_stat > 0 and p_value/2 < 0.05:
                    if group_1 not in points:
                        points[group_1] = 0
                    points[group_1] += 1
                    win_matrix[i][j] += 1
        print("Round: {}".format(r))
        # breakpoint()

    # breakpoint()
    points = dict(sorted(points.items(), key=lambda x: x[1], reverse=True))
    print(points)
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
    plt.title("Bootstrap result of {} for {} targets".format(
        measure, mode), fontsize=20)
    plt.savefig(
        "./bootstrap_EU/win_matrix_bootstrap_{}_{}_{}_p={}_n={}_step18.png".format(measure, model, mode, p_value_threshold, bootstrap_rounds), dpi=300)
    # save the win matrix as a numpy array

    np.save("./bootstrap_EU/win_matrix_bootstrap_{}_{}_{}_p={}_n={}_step18.npy".format(
        measure, model, mode, p_value_threshold, bootstrap_rounds), win_matrix)


bootstrap(measure, mode, model, p_value_threshold=0.05)
sys.exit(0)

# group 1 is the first group in the ranked points
group_1 = list(points.keys())[0]
# group 2 is the second group in the ranked points
group_2 = list(points.keys())[1]
# now do bootstrapping
rounds = 10000
group_1_wins = []
group_2_wins = []
no_difference = []
for i in range(rounds):
    sampled_data = T1_data.sample(frac=1, replace=True)
    # print(sampled_data.shape)
    group_1_data = sampled_data[group_1]
    group_2_data = sampled_data[group_2]
    t_stat, p_value = stats.ttest_rel(group_1_data, group_2_data)
    if t_stat > 0 and p_value/2 < 0.01:
        group_1_wins.append(1)
    elif t_stat < 0 and p_value/2 < 0.01:
        group_2_wins.append(1)
    else:
        no_difference.append(1)
print("Score: {}".format(measure))
print("Group 1: {}, Group 2: {}".format(group_1, group_2))
print("Group 1 wins: {}, Group 2 wins: {}".format(
    len(group_1_wins), len(group_2_wins)))
print("No difference: {}".format(len(no_difference)))


################
sys.exit(0)


def jackknife_resampling(data):
    n = len(data)
    jackknife_samples = []
    for i in range(n):
        sample = np.delete(data, i)
        jackknife_samples.append(sample.mean())
    return np.array(jackknife_samples)


def jackknife_t_test(group_1, group_2):
    # Perform jackknife resampling
    jack_1 = jackknife_resampling(group_1)
    jack_2 = jackknife_resampling(group_2)
    # Perform t-test on jackknife samples
    t_statistic, p_value = stats.ttest_ind(jack_1, jack_2)
    return t_statistic, p_value


# Example usage
np.random.seed(42)  # for reproducibility
group_1 = np.random.normal(loc=0, scale=1, size=30)
group_2 = np.random.normal(loc=0.5, scale=1, size=30)
t_stat, p_val = jackknife_t_test(group_1, group_2)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")
# Interpret the results
alpha = 0.05
if p_val < alpha:
    print("Reject the null hypothesis: There is a significant difference between the groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the groups.")
