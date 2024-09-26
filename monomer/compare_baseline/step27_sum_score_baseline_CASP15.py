import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


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


# csv_path = "./monomer_data_aug_30/processed/EU/"
# csv_path = "./monomer_data_Sep_10/processed/EU/"
# csv_path = "./monomer_data_Sep_10/raw_data/EU/"
csv_path = "./monomer_data_CASP15/raw_data/"
# baseline_path = "./monomer_data_regular_CASP15/raw_data/"
csv_list = [txt for txt in os.listdir(
    csv_path) if txt.endswith(".csv") and txt.startswith("T1")]
print(csv_list.__len__())
out_path = "./group_by_target_CASP15/"
sum_path = "./sum_CASP15/"
png_path = "./png_CASP15/"

# out_path = "./group_by_target/"
# sum_path = "./sum/"
if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(sum_path):
    os.makedirs(sum_path)
if not os.path.exists(png_path):
    os.makedirs(png_path)

model = "first"
model = "best"

mode = "easy"
mode = "medium"
mode = "hard"
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


# print(csv_list.__len__())
# breakpoint()
# read all data and concatenate them into one big dataframe
feature = "GDT_TS"
feature = "GDT_HA"
features = ['GDT_TS',
            'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
            'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
            'SphGr',
            'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
            'MP_rotout', 'MP_ramout', 'MP_ramfv', 'reLLG_lddt', 'reLLG_const']

inverse_columns = ["RMS_CA", "RMS_ALL", "err",
                   "RMSD[L]", "MolPrb_Score", "FlexE", "MP_clash", "MP_rotout", "MP_ramout"]


def get_group_by_target(csv_list, csv_path, feature, model, mode):
    data = pd.DataFrame()
    data_raw = pd.DataFrame()  # store raw data for analysis
    for csv_file in csv_list:
        print("Processing {}".format(csv_file))
        data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
        data_tmp = pd.DataFrame(data_tmp[feature])
        # there is a BUG here. something wrong with group 999 processing but I don't have time to fix it.
        # so just leave it here as it is.
        data_tmp = data_tmp.replace("-", float(0))
        # breakpoint()
        if feature in inverse_columns:
            data_tmp[feature] = -data_tmp[feature]
        data_tmp.index = data_tmp.index.str.extract(
            r'(T\w+)TS(\w+)_(\w+)-(D\w+)').apply(lambda x: (f"{x[0]}-{x[3]}", f"TS{x[1]}", x[2]), axis=1)
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
        grouped = data_tmp.groupby(["group"])
        grouped = pd.DataFrame(grouped[feature].max())
        # grouped.index = grouped.index.droplevel(1)
        # if there is any value in this column that is a string, convert it to float
        grouped[feature] = grouped[feature].astype(float)
        try:
            # # find if there is any string in the dataframe
            # for i in range(grouped.shape[0]):
            #     for j in range(grouped.shape[1]):
            #         if isinstance(grouped.iloc[i, j], str):
            #             print(grouped.iloc[i, j])
            grouped = grouped.sort_values(by=feature, ascending=False)
            # print all the value in grouped
            # for i in range(grouped.shape[0]):print(grouped.iloc[i])
            initial_z = (grouped - grouped.mean()) / grouped.std()
        except:
            breakpoint()
        new_z_score = pd.DataFrame(
            index=grouped.index, columns=grouped.columns)
        filtered_data = grouped[feature][initial_z[feature] >= -2]
        new_mean = filtered_data.mean(skipna=True)
        new_std = filtered_data.std(skipna=True)
        new_z_score[feature] = (grouped[feature] - new_mean) / new_std
        new_z_score = new_z_score.fillna(-2.0)
        new_z_score = new_z_score.where(new_z_score > -2, -2)

        # I actually don't understand why this is necessary... but need to keep it in mind.
        # grouped = grouped.apply(lambda x: (x - x.mean()) / x.std())

        new_z_score = new_z_score.rename(
            columns={feature: csv_file.split(".")[0]})
        grouped = grouped.rename(
            columns={feature: csv_file.split(".")[0]})
        data = pd.concat([data, new_z_score], axis=1)
        data_raw = pd.concat([data_raw, grouped], axis=1)

    # impute data again with -2
    data = data.fillna(-2.0)
    # sort columns by alphabetical order
    data = data.reindex(sorted(data.columns), axis=1)
    data_columns = data.columns
    data.to_csv(
        out_path + "group_by_target_baseline-{}-{}-{}.csv".format(feature, model, mode))

    target_count = {}
    for EU in data_columns:
        target = EU.split("-")[0]
        if target not in target_count:
            target_count[target] = 0
        target_count[target] += 1
    # use the inverse of the target_count as the weight
    target_weight = {key: 1/value for key, value in target_count.items()}
    # assign EU_weight based on the target_weight
    EU_weight = {EU: target_weight[EU.split("-")[0]]
                 for EU in data_columns}

    data["sum"] = data.sum(axis=1)
    data = data.sort_values(by="sum", ascending=False)
    data.to_csv(sum_path
                + "sum_unweighted_baseline_{}-{}-{}.csv".format(feature, model, mode))

    data.drop(columns=["sum"], inplace=True)
    # assign the weight to the data
    data = data * pd.Series(EU_weight)
    data["sum"] = data.sum(axis=1)
    data = data.sort_values(by="sum", ascending=False)
    data.to_csv(sum_path
                + "sum_baseline_{}-{}-{}.csv".format(feature, model, mode))

    # get the lowest value in the data_raw, and impute nan with the lowest value
    # data_raw = data_raw.fillna(data_raw.min().min())
    data_raw = data_raw.reindex(sorted(data_raw.columns), axis=1)
    data_raw["sum"] = data_raw.sum(axis=1)
    data_raw = data_raw.sort_values(by="sum", ascending=False)
    data_raw.to_csv(sum_path
                    + "sum_raw_unweighted_baseline_{}-{}-{}.csv".format(feature, model, mode))
    data_raw.drop(columns=["sum"], inplace=True)
    data_raw_weighted = data_raw * pd.Series(EU_weight)
    data_raw_weighted["sum"] = data_raw_weighted.sum(axis=1)
    data_raw_weighted = data_raw_weighted.sort_values(
        by="sum", ascending=False)
    data_raw_weighted.to_csv(sum_path
                             + "sum_raw_baseline_{}-{}-{}.csv".format(feature, model, mode))
    # breakpoint()
    return data, data_raw


data, data_raw = get_group_by_target(
    csv_list, csv_path, feature, model, mode)
###############
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
baseline_group = pd.DataFrame(data_raw["TS446"])
baseline_dict = {}
for group in groups:
    data_raw_group = pd.DataFrame(data_raw[group])
    # get the intersection of non nan values in data_raw_group and baseline_group
    non_nan_baseline = baseline_group[pd.notna(baseline_group["TS446"])].index
    non_nan_group = data_raw_group[pd.notna(data_raw_group[group])].index
    # 计算两者的交集
    intersection_index = non_nan_baseline.intersection(non_nan_group)
    # 根据 intersection_index 获取对应的值
    baseline_values = baseline_group.loc[intersection_index, "TS446"]
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
plt.figure(figsize=(35, 15))
highlight_group = "446"
bar_colors = ['C0' if group != highlight_group else 'C1' for group in groups]
plt.bar(groups, values, color=bar_colors)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.title(
    "sum of {} for {} model in {} targets, compared with baseline (group {}) in CASP15".format(feature, model, mode, highlight_group), fontsize=20)
plt.ylabel("sum of {}".format(feature), fontsize=20)
plt.axhline(y=1, color='k')
# there is one group 145, we need to write something on top of its bar
for group, value in zip(groups, values):
    # breakpoint()
    if group == highlight_group:
        # breakpoint()
        plt.text(group, value + 0.03, str("ColabFold"),
                 ha='center', fontsize=20, color='C1')
# first 5 groups
first_5_groups = groups[:5]
first_5_values = values[:5]
for group, value in zip(first_5_groups, first_5_values):
    plt.text(group, value + 0.01, str(value.round(2)),
             ha='center', fontsize=10, color='C0')
plt.savefig(
    png_path + "sum_intersect_{}-{}-{}_with_colabfold_baseline.png".format(feature, model, mode), dpi=300)
###############
# breakpoint()
# for feature in features:
#     get_group_by_target(csv_list, csv_path, feature, model, mode)
#     print("Finished processing {}".format(feature))
data_sum = data["sum"]
# plot the data_sum
# remove the first 2 char in the index
data_sum.index = data_sum.index.str[2:]
plt.figure(figsize=(35, 15))
highlight_index = "446"
bar_colors = ['C0' if index !=
              highlight_index else 'C1' for index in data_sum.index]

plt.bar(data_sum.index, data_sum.values, color=bar_colors)
plt.xticks(rotation=90, fontsize=15)
plt.yticks(fontsize=20)
plt.title(
    "sum z-score of {} for {} model in {} targets in CASP15".format(feature, model, mode), fontsize=24)
plt.ylabel(f"sum {feature} z-score", fontsize=20)
plt.axhline(y=0, color='k')
# there is one group 145, we need to write something on top of its bar

for index, value in zip(data_sum.index, data_sum.values):
    if index == highlight_index:
        plt.text(index, value + 2.5, str("ColabFold"),
                 ha='center', fontsize=15, color='C1')


plt.savefig(
    png_path + "sum_z-score_{}-{}-{}.png".format(feature, model, mode), dpi=300)
# plt.axvline(x="145", color='r')

sys.exit(0)

# divide by the sum of 145
data_raw_sum = data_raw["sum"]
data_raw_sum.index = data_raw_sum.index.str[2:]
# want to normalize the data_raw_sum by the sum of group 145
data_raw_sum = data_raw_sum / data_raw_sum["446"]
plt.figure(figsize=(40, 20))
highlight_index = "446"
bar_colors = ['C0' if index !=
              highlight_index else 'C1' for index in data_raw_sum.index]
plt.bar(data_raw_sum.index, data_raw_sum.values, color=bar_colors)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.title(
    "sum of {} for {} model in {} targets, compared with baseline (group {})".format(feature, model, mode, highlight_index))
plt.ylabel("sum of {}".format(feature))
plt.axhline(y=1, color='k')
# there is one group 145, we need to write something on top of its bar
for index, value in zip(data_raw_sum.index, data_raw_sum.values):
    if index == highlight_index:
        plt.text(index, value + 2.0, str("ColabFold"),
                 ha='center', fontsize=10, color='C1')
plt.savefig(
    png_path + "sum_{}-{}-{}_compared_with_baseline.png".format(feature, model, mode), dpi=300)
# breakpoint()
