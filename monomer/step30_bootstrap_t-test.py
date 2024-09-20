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
# measure = sys.argv[1]

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


def bootstrap_t_test(measures, model, mode, score_path, output_path="./bootstrap_EU/", p_value_threshold=0.05, weight=None, bootstrap_rounds=1000):
    measure_type = "CASP16"
    try:
        measures = list(measures)
    except:
        raise ValueError("measures must be iterable")
    if weight is None:
        weight = [1/len(measures)] * len(measures)
    equal_weight = len(set(weight)) == 1
    assert len(measures) == len(weight)

    T1_data = pd.DataFrame()
    groups = None
    for i in range(len(measures)):
        measure = measures[i]
        score_file = "group_by_target-{}-{}-{}.csv".format(
            measure, model, mode)
        score_matrix = pd.read_csv(score_path + score_file, index_col=0)
        score_matrix = score_matrix.filter(regex='T1')
        score_matrix = score_matrix.reindex(
            sorted(score_matrix.columns), axis=1)
        score_matrix = score_matrix.T
        score_matrix.columns = score_matrix.columns.str.replace(
            "TS", "")
        # rename each column to include the measure name
        groups = score_matrix.columns
        score_matrix.columns = [
            f"{col}-{measure}" for col in score_matrix.columns]
        T1_data = pd.concat([T1_data, score_matrix], axis=1)
    missing_values = T1_data.isnull().sum().sum()
    if missing_values > 0:
        raise ValueError(
            "There are missing values in the T1_data, please fill them before running the bootstrap_t_test function")
    # in this senario there just should not be any missing values

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

    # # remove columns with more than 80% missing values
    # T1_data = T1_data.loc[:, T1_data.isnull().mean() < 0.8]
    if "raw" in score_file:
        T1_data.fillna(50, inplace=True)
    else:
        T1_data.fillna(-2, inplace=True)
    length = len(groups)
    points = {}

    measure_points_dict = {}
    for measure in measures:
        measure_points = {}
        for i in range(length):
            group_i = groups[i]
            measure_points[group_i] = 0
        for i in range(length):
            for j in range(length):
                group_1_id = groups[i]
                group_2_id = groups[j]
                if group_1_id == group_2_id:
                    continue
                group_1 = group_1_id + "-" + measure
                group_2 = group_2_id + "-" + measure
                group_1_data = T1_data[group_1]
                group_2_data = T1_data[group_2]
                t_stat, p_value = stats.ttest_rel(group_1_data, group_2_data)
                if t_stat > 0 and p_value/2 < p_value_threshold:
                    measure_points[group_1_id] += 1
        measure_points_dict[measure] = measure_points
        print(f"{measure} finished.")
    for measure in measures:
        measure_points = measure_points_dict[measure]
        for group in measure_points:
            if group not in points:
                points[group] = 0
            points[group] += measure_points[group] * \
                weight[measures.index(measure)]
    points = dict(sorted(points.items(), key=lambda x: x[1], reverse=True))
    with open(output_path + f"{measure_type}_{model}_{mode}_p={p_value_threshold}_equal_weight={equal_weight}_ranking_t_test.txt", 'w') as f:
        f.write(str(points))
    # this is to get a re-ordered list of groups
    groups = list(points.keys())

    # can plot the points here
    plt.figure(figsize=(30, 15))
    bottom = [0 for i in range(length)]
    for key in measure_points_dict:
        measure_points = measure_points_dict[key]
        points = [measure_points[group] for group in groups]
        plt.bar(groups, points, bottom=bottom, label=key, width=0.8)
        bottom = [bottom[i] + points[i] for i in range(length)]
    plt.xticks(np.arange(length), groups, rotation=45, fontsize=10, ha='right')
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    if equal_weight:
        plt.title(
            f"Bootstrap result of t-test points for {measure_type} EUs with equal weight", fontsize=18)
        plt.savefig(output_path +
                    f"t_test_points_{measure_type}_{model}_{mode}_p={p_value_threshold}_equal_weight.png",
                    dpi=300)
    else:
        plt.title(
            f"Bootstrap result of t-test points for {measure_type} EUs with custom weight", fontsize=18)
        plt.savefig(output_path +
                    f"t_test_points_{measure_type}_{model}_{mode}_p={p_value_threshold}_custom_weight.png",
                    dpi=300)
    # use the above t-test code to get a initial ranking of the groups.
    # then generate new groups list using the ranking
    # then do bootstrapping

    # breakpoint()
    points = {}
    length = len(groups)
    win_matrix = [[0 for i in range(length)] for j in range(length)]
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
        # breakpoint()
        T1_data_bootstrap = T1_data_bootstrap.sort_index()
        # breakpoint()
        bootstrap_points = {}
        # for i in range(length):
        #     group_i = groups[i]
        #     bootstrap_points[group_i] = 0

        for measure in measures:
            measure_points = {}
            for i in range(length):
                group_i = groups[i]
                measure_points[group_i] = 0
            for i in range(length):
                for j in range(length):
                    group_1_id = groups[i]
                    group_2_id = groups[j]
                    if group_1_id == group_2_id:
                        continue
                    group_1 = group_1_id + "-" + measure
                    group_2 = group_2_id + "-" + measure
                    group_1_data = T1_data_bootstrap[group_1]
                    group_2_data = T1_data_bootstrap[group_2]
                    t_stat, p_value = stats.ttest_rel(
                        group_1_data, group_2_data)
                    if t_stat > 0 and p_value/2 < p_value_threshold:
                        measure_points[group_1_id] += 1
            measure_points_dict[measure] = measure_points
            # print(f"{measure} finished.")
        for measure in measures:
            measure_points = measure_points_dict[measure]
            for group in measure_points:
                if group not in bootstrap_points:
                    bootstrap_points[group] = 0
                bootstrap_points[group] += measure_points[group] * \
                    weight[measures.index(measure)]
        for i in range(length):
            for j in range(length):
                if i == j:
                    continue
                if bootstrap_points[groups[i]] > bootstrap_points[groups[j]]:
                    win_matrix[i][j] += 1

        print("Round: {}".format(r))
    # points = dict(sorted(points.items(), key=lambda x: x[1], reverse=True))
    # print(points)
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
    plt.xticks(np.arange(length), groups, rotation=45, fontsize=10, ha='right')
    plt.yticks(np.arange(length), groups, rotation=0, fontsize=10)
    plt.title(
        "{} bootstrap result of t-test points for {} EUs".format(measure_type, mode), fontsize=20)
    plt.savefig(output_path +
                f"win_matrix_{measure_type}_{model}_{mode}_p={p_value_threshold}_n={bootstrap_rounds}_equal_weight={equal_weight}_bootstrap_t_test.png",
                dpi=300)

    np.save(output_path +
            f"win_matrix_{measure_type}_{model}_{mode}_p={p_value_threshold}_n={bootstrap_rounds}_equal_weight={equal_weight}_bootstrap_t_test.npy",
            win_matrix)


# # bootstrap_t_test(measure, model, mode, p_value_threshold=0.05)
# bootstrap_t_test(measures, model, mode, score_path,
#                  weight=None, bootstrap_rounds=1000)
wanted_measures = ['LDDT', 'CAD_AA', 'SphGr',
                   'MP_clash', 'RMS_CA',
                   'GDT_HA', 'QSE', 'reLLG_const']

wanted_measures = ['GDT_HA', 'GDC_SC', 'AL0_P', 'SphGr',
                   'CAD_AA', 'QSE', 'MolPrb_Score', 'reLLG_const']
equal_weight = False
equal_weight = True
if equal_weight:
    weights = [1/8] * 8
else:
    weights = [1/16, 1/16, 1/16,
               1/12, 1/12,
               1/4, 1/4, 1/4]


bootstrap_t_test(wanted_measures, model, mode,
                 score_path, output_path="./bootstrap_EU/",
                 weight=None, bootstrap_rounds=50)

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
