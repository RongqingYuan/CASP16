import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os


hard_group = [
]

medium_group = [
]

easy_group = [
]


csv_path = "./monomer_data_aug_30/processed/EU/"
csv_path = "./monomer_data_Sep_10/processed/EU/"
csv_path = "./monomer_data_Sep_10/raw_data/EU/"
# csv_path = "./monomer_data_Sep_17/raw_data/"
csv_path = "./oligomer_data_Sep_17/raw_data/"
sum_path = "./sum_raw_EU/"

if not os.path.exists(sum_path):
    os.makedirs(sum_path)

csv_list = [txt for txt in os.listdir(
    csv_path) if txt.endswith(".csv") and (txt.startswith("T1") or txt.startswith("H1"))]

model = "first"
model = "best"

mode = "hard"
mode = "medium"
mode = "easy"
mode = "all"

if mode == "hard":
    csv_list = [csv for csv in csv_list if csv.split(
        ".")[0] in hard_group]

elif mode == "medium":
    csv_list = [csv for csv in csv_list if csv.split(
        ".")[0] in medium_group]

elif mode == "easy":
    csv_list = [csv for csv in csv_list if csv.split(
        ".")[0] in easy_group]

elif mode == "all":
    pass


# feature = "GDT_TS"
# features = ['GDT_TS', 'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
#             'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
#             'SphGr',
#             'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
#             'MP_rotout', 'MP_ramout', 'MP_ramfv', 'reLLG_lddt', 'reLLG_const']

# inverse_columns = ["RMS_CA", "RMS_ALL", "err",
#                    "RMSD[L]", "MolPrb_Score", "FlexE", "MP_clash", "MP_rotout", "MP_ramout"]

features = ["QSglob", "QSbest", "ICS(F1)", "lDDT", "DockQ_Avg",
            "IPS(JaccCoef)", "TMscore"]
feature = features[1]


def get_group_by_target(csv_path, csv_list, feature, model, mode):
    data = pd.DataFrame()
    data_raw = pd.DataFrame()
    for csv_file in csv_list:
        data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
        data_tmp = pd.DataFrame(data_tmp[feature])
        print("Processing {}".format(csv_file), data_tmp.shape)
        # breakpoint()
        data_tmp.index = data_tmp.index.str.extract(
            r'(\w+)TS(\w+)_(\w+)').apply(lambda x: (f"{x[0]}", f"TS{x[1]}", x[2][0]), axis=1)
        # breakpoint()
        data_tmp.index = pd.MultiIndex.from_tuples(
            data_tmp.index, names=['target', 'group', 'submission_id'])
        # # get all data with submission_id == 6
        # data_tmp = data_tmp.loc[(slice(None), slice(None), "6"), :]
        # drop all data with submission_id == 6
        if model == "best":
            data_tmp = data_tmp.loc[(slice(None), slice(None), [
                "1", "2", "3", "4", "5"]), :]
        elif model == "first":
            data_tmp = data_tmp.loc[(slice(None), slice(None),
                                    "1"), :]
        # grouped = data_tmp.groupby(["group", "target"])
        # grouped = pd.DataFrame(grouped[feature].max())
        # grouped.index = grouped.index.droplevel(1)

        grouped = data_tmp.groupby(["group"])
        grouped = pd.DataFrame(grouped[feature].max())
        # grouped.index = grouped.index.droplevel(1)
        # sort grouped
        grouped = grouped.sort_values(by=feature, ascending=False)
        initial_z = (grouped - grouped.mean()) / grouped.std()
        new_z_score = pd.DataFrame(
            index=grouped.index, columns=grouped.columns)
        filtered_data = grouped[feature][initial_z[feature] >= -2]
        new_mean = filtered_data.mean(skipna=True)
        new_std = filtered_data.std(skipna=True)
        new_z_score[feature] = (grouped[feature] - new_mean) / new_std
        new_z_score = new_z_score.fillna(-2.0)
        new_z_score = new_z_score.where(new_z_score > -2, -2)

        # breakpoint()

        # I actually don't understand why this is necessary... but need to keep it in mind.
        # grouped = grouped.apply(lambda x: (x - x.mean()) / x.std())

        new_z_score = new_z_score.rename(
            columns={feature: csv_file.split(".")[0]})
        data = pd.concat([data, new_z_score], axis=1)
        grouped = grouped.rename(
            columns={feature: csv_file.split(".")[0]})
        data_raw = pd.concat([data_raw, grouped], axis=1)
    # impute data again with -2
    # breakpoint()
    data = data.fillna(-2.0)
    data.to_csv("./group_by_target_EU/" +
                "group_by_target-{}-{}-{}.csv".format(feature, model, mode))

    data_raw.to_csv("./group_by_target_EU/" +
                    "group_by_target_raw-{}-{}-{}.csv".format(feature, model, mode))

    data["sum"] = data.sum(axis=1)
    data = data.sort_values(by="sum", ascending=False)
    data.to_csv("./sum/" + "sum_{}-{}-{}.csv".format(feature, model, mode))

    return data, data_raw


data, data_raw = get_group_by_target(csv_path, csv_list, feature, model, mode)
# print the nan rate of the data in columns and rows
print(data_raw.isna().sum(axis=0)/data.shape[0])
print(data_raw.isna().sum(axis=1)/data.shape[1])
# remove rows with more than 50% nan values
data_raw = data_raw[data_raw.isna().sum(axis=1) < 0.5 * data_raw.shape[1]]
print(data_raw.isna().sum(axis=0)/data.shape[0])
print(data_raw.isna().sum(axis=1)/data.shape[1])
# for each row, get its non-nan values in intersection with "TS145", and calculate the sum of the non-nan values
# then divide the sum of the non-nan values by the sum of "TS145" to get the normalized sum
# then plot the normalized sum
data_raw = data_raw.T

groups = data_raw.columns
baseline_group = pd.DataFrame(data_raw["TS145"])
baseline_dict = {}
for group in groups:
    data_raw_group = pd.DataFrame(data_raw[group])
    # get the intersection of non nan values in data_raw_group and baseline_group
    non_nan_baseline = baseline_group[pd.notna(baseline_group["TS145"])].index
    non_nan_group = data_raw_group[pd.notna(data_raw_group[group])].index
    # 计算两者的交集
    intersection_index = non_nan_baseline.intersection(non_nan_group)
    # 根据 intersection_index 获取对应的值
    baseline_values = baseline_group.loc[intersection_index, "TS145"]
    group_values = data_raw_group.loc[intersection_index, group]

    # sum them up
    sum_baseline = baseline_values.sum()
    sum_group = group_values.sum()
    ratio = sum_group / sum_baseline
    baseline_dict[group] = ratio
    # # 输出结果或进行进一步操作
    # print(
    #     f"Group {group} has {len(intersection_index)} intersecting non-NaN values.")
    # print("Baseline values:", baseline_values.values)
    # print(f"{group} values:", group_values.values)

# sort the baseline_dict by its values
baseline_dict = dict(sorted(baseline_dict.items(),
                     key=lambda x: x[1], reverse=True))
groups = [key[2:] for key in baseline_dict.keys()]
values = list(baseline_dict.values())
plt.figure(figsize=(30, 15))
highlight_group = "145"
bar_colors = ['C0' if group != highlight_group else 'C1' for group in groups]
plt.bar(groups, values, color=bar_colors)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.title(
    "sum of {} for {} model in {} targets, compared with baseline (group {}) in CASP16".format(feature, model, mode, highlight_group), fontsize=15)
plt.ylabel("sum of {}".format(feature))
plt.axhline(y=1, color='k')
# there is one group 145, we need to write something on top of its bar
for group, value in zip(groups, values):
    if group == highlight_group:
        plt.text(group, value + 0.02, str("ColabFold"),
                 ha='center', fontsize=10, color='C1')
# first 5 groups
first_5_groups = groups[:5]
first_5_values = values[:5]
for group, value in zip(first_5_groups, first_5_values):
    plt.text(group, value + 0.01, str(value.round(2)),
             ha='center', fontsize=7, color='C0')
plt.savefig(
    sum_path + "sum_intersect_{}-{}-{}_with_colabfold_baseline.png".format(feature, model, mode), dpi=300)
###############
