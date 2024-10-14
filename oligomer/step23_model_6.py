import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os
import pandas as pd
import numpy as np
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, Normalize


def group_by_target(results_dir, result_files, out_dir,
                    feature, model, mode, impute_value):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data = pd.DataFrame()
    data_weighted = pd.DataFrame()
    data_raw = pd.DataFrame()
    print("Processing {}".format(feature))
    for result_file in result_files:
        print("Processing {}".format(result_file))
        result_path = results_dir + result_file
        # it is actually a tsv file, us pd.read_csv to read it
        data_tmp = pd.read_csv(result_path, sep="\t", index_col=0)

        nr_interf_group_in_ref = data_tmp["nr_interf_group_in_ref"]
        if nr_interf_group_in_ref.nunique() == 1:
            # print("All values are the same in nr_interf_group_in_ref")
            pass
        else:
            print("Not all values are the same in nr_interf_group_in_ref")
        nr_refinterf_num = data_tmp["nr_refinterf_#"]
        if nr_refinterf_num.nunique() == 1:
            # print("All values are the same in nr_refinterf_#")
            # convert to string
            nr_refinterf_num = nr_refinterf_num.astype(str)
            pass
        else:
            print("Not all values are the same in nr_refinterf_#")

        nr_interf_group_interfsize_in_ref = data_tmp["nr_interf_group_interfsize_in_ref"]
        if nr_interf_group_interfsize_in_ref.nunique() == 1:
            # print("All values are the same in nr_interf_group_interfsize_in_ref")
            # convert to string
            nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.astype(
                str)
            pass
        else:
            print("Not all values are the same in nr_interf_group_interfsize_in_ref")

        data_tmp = pd.DataFrame(data_tmp[feature]).astype(str)
        data_tmp_split = data_tmp[feature].str.split(';', expand=True)
        data_tmp_split.columns = [
            f'interface_{i+1}' for i in range(data_tmp_split.shape[1])]

        # data_tmp = pd.concat([data_tmp, data_tmp_split], axis=1)
        # breakpoint()

        # # check if there is any "-" in the data
        # has_dash = data_tmp[feature].str.contains('-')
        # has_dash = data_tmp[feature].str.contains('None')
        # print(has_dash.sum())

        # fill "-" with nan
        data_tmp_split = data_tmp_split.replace("-", np.nan)
        # convert everything to float
        data_tmp_split = data_tmp_split.astype(float)

        data_tmp_split.index = data_tmp_split.index.str.extract(
            r'(\w+)TS(\w+)_(\w+)').apply(lambda x: (f"{x[0]}", f"TS{x[1]}", x[2][0]), axis=1)
        data_tmp_split.index = pd.MultiIndex.from_tuples(
            data_tmp_split.index, names=['target', 'group', 'submission_id'])
        if model == "best":
            data_tmp_split = data_tmp_split.loc[(slice(None), slice(None), [
                "1", "2", "3", "4", "5"]), :]
        elif model == "first":
            data_tmp_split = data_tmp_split.loc[(slice(None), slice(None),
                                                 "1"), :]
        elif model == "msa":
            data_tmp_split = data_tmp_split.loc[(slice(None), slice(None),
                                                 "6"), :]

        # # get the first nr_refinterf_num
        # nr_refinterf_num = nr_refinterf_num.iloc[0]
        # nr_interface_weights = nr_refinterf_num.split(";")
        # nr_interface_weights = [float(weight)
        #                         for weight in nr_interface_weights]

        # # get the first nr_interf_group_interfsize_in_ref
        # nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.iloc[0]
        # nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.split(
        #     ";")

        # nr_interface_size_weights = []
        # for i in range(len(nr_interf_group_interfsize_in_ref)):
        #     size_info = nr_interf_group_interfsize_in_ref[i].split(",")
        #     size_weight = 0
        #     for j in range(len(size_info)):
        #         sizes = size_info[j].split("/")
        #         for size in sizes:
        #             size_weight += int(size)
        #     nr_interface_size_weights.append(size_weight)
        # # take log 10 of the nr_interface_size_weights
        # nr_interface_size_weights = np.log10(nr_interface_size_weights)

        # EU_weight = data_tmp_split.shape[1] ** (1/3)

        for i in range(data_tmp_split.shape[1]):
            grouped = data_tmp_split.groupby(["group"])
            feature_name = f"interface_{i+1}"
            grouped = pd.DataFrame(grouped[feature_name].max())
            grouped = grouped.sort_values(by=feature_name, ascending=False)
            initial_z = (grouped - grouped.mean()) / grouped.std()
            new_z_score = pd.DataFrame(
                index=grouped.index, columns=grouped.columns)
            filtered_data = grouped[feature_name][initial_z[feature_name] >= -2]
            new_mean = filtered_data.mean(skipna=True)
            new_std = filtered_data.std(skipna=True)
            new_z_score[feature_name] = (
                grouped[feature_name] - new_mean) / new_std
            new_z_score = new_z_score.fillna(impute_value)
            new_z_score = new_z_score.where(
                new_z_score > impute_value, impute_value)

            new_z_score = new_z_score.rename(
                columns={feature_name: result_file.split(".")[0]+"_"+feature_name})
            data = pd.concat([data, new_z_score], axis=1)
            grouped = grouped.rename(
                columns={feature_name: result_file.split(".")[0]+"_"+feature_name})
            data_raw = pd.concat([data_raw, grouped], axis=1)

    weight_list = []
    for result_file in result_files:
        print("Processing {}".format(result_file))
        result_path = results_dir + result_file
        # it is actually a tsv file, us pd.read_csv to read it
        data_tmp = pd.read_csv(result_path, sep="\t", index_col=0)

        nr_interf_group_in_ref = data_tmp["nr_interf_group_in_ref"]
        if nr_interf_group_in_ref.nunique() == 1:
            # print("All values are the same in nr_interf_group_in_ref")
            pass
        else:
            print("Not all values are the same in nr_interf_group_in_ref")
        nr_refinterf_num = data_tmp["nr_refinterf_#"]
        if nr_refinterf_num.nunique() == 1:
            # print("All values are the same in nr_refinterf_#")
            # convert to string
            nr_refinterf_num = nr_refinterf_num.astype(str)
            pass
        else:
            print("Not all values are the same in nr_refinterf_#")

        nr_interf_group_interfsize_in_ref = data_tmp["nr_interf_group_interfsize_in_ref"]
        if nr_interf_group_interfsize_in_ref.nunique() == 1:
            # print("All values are the same in nr_interf_group_interfsize_in_ref")
            # convert to string
            nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.astype(
                str)
            pass
        else:
            print("Not all values are the same in nr_interf_group_interfsize_in_ref")

        data_tmp = pd.DataFrame(data_tmp[feature]).astype(str)
        data_tmp_split = data_tmp[feature].str.split(';', expand=True)
        data_tmp_split.columns = [
            f'interface_{i+1}' for i in range(data_tmp_split.shape[1])]

        # data_tmp = pd.concat([data_tmp, data_tmp_split], axis=1)
        # breakpoint()

        # # check if there is any "-" in the data
        # has_dash = data_tmp[feature].str.contains('-')
        # has_dash = data_tmp[feature].str.contains('None')
        # print(has_dash.sum())

        # fill "-" with nan
        data_tmp_split = data_tmp_split.replace("-", np.nan)
        data_tmp_split = data_tmp_split.astype(float)

        data_tmp_split.index = data_tmp_split.index.str.extract(
            r'(\w+)TS(\w+)_(\w+)').apply(lambda x: (f"{x[0]}", f"TS{x[1]}", x[2][0]), axis=1)
        data_tmp_split.index = pd.MultiIndex.from_tuples(
            data_tmp_split.index, names=['target', 'group', 'submission_id'])
        if model == "best":
            data_tmp_split = data_tmp_split.loc[(slice(None), slice(None), [
                "1", "2", "3", "4", "5"]), :]
        elif model == "first":
            data_tmp_split = data_tmp_split.loc[(slice(None), slice(None),
                                                 "1"), :]
        elif model == "msa":
            data_tmp_split = data_tmp_split.loc[(slice(None), slice(None),
                                                 "6"), :]

        nr_refinterf_num = nr_refinterf_num.iloc[0]
        nr_interface_weights = nr_refinterf_num.split(";")
        nr_interface_weights = [float(weight)
                                for weight in nr_interface_weights]
        nr_interface_weights = np.array(nr_interface_weights)
        # nr_interface_weights = nr_interface_weights / nr_interface_weights.sum()

        nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.iloc[0]
        nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.split(
            ";")

        nr_interface_size_weights = []
        for i in range(len(nr_interf_group_interfsize_in_ref)):
            size_info = nr_interf_group_interfsize_in_ref[i].split(",")
            size_weight = 0
            for j in range(len(size_info)):
                sizes = size_info[j].split("/")
                for size in sizes:
                    size_weight += int(size)
            nr_interface_size_weights.append(size_weight/2)
        nr_interface_size_weights = np.log10(nr_interface_size_weights)
        nr_interface_size_weights = np.array(nr_interface_size_weights)
        # nr_interface_size_weights = nr_interface_size_weights / nr_interface_size_weights.sum()

        # elementwise multiplication
        interface_weight = nr_interface_weights * nr_interface_size_weights
        # normalize the weights
        interface_weight = interface_weight / interface_weight.sum()
        print(interface_weight)
        # append each weight to the weight_list
        for weight in interface_weight:
            weight_list.append(weight)
        EU_weight = data_tmp_split.shape[1] ** (1/3)
        # EU_weight = 1

        # if result_file.startswith("H1236"):
        #     breakpoint()

        # get a target_score df that has the same row as data_raw to make sure no nan values
        target_score = pd.DataFrame(index=data_raw.index)
        for i in range(data_tmp_split.shape[1]):
            grouped = data_tmp_split.groupby(["group"])
            feature_name = f"interface_{i+1}"
            grouped = pd.DataFrame(grouped[feature_name].max())
            grouped = grouped.sort_values(by=feature_name, ascending=False)
            initial_z = (grouped - grouped.mean()) / grouped.std()
            new_z_score = pd.DataFrame(
                index=grouped.index, columns=grouped.columns)
            filtered_data = grouped[feature_name][initial_z[feature_name] >= -2]
            new_mean = filtered_data.mean(skipna=True)
            new_std = filtered_data.std(skipna=True)
            new_z_score[feature_name] = (
                grouped[feature_name] - new_mean) / new_std
            new_z_score = new_z_score.fillna(impute_value)
            new_z_score = new_z_score.where(
                new_z_score > impute_value, impute_value)
            target_score = pd.concat([target_score, new_z_score], axis=1)
            target_score = target_score.fillna(impute_value)
            # breakpoint()
        # breakpoint()

        # continue
        # breakpoint()
        # # first weight by the number of interfaces
        # target_score = target_score * nr_interface_weights
        # # then weight by the size of the interfaces
        # target_score = target_score * nr_interface_size_weights

        target_score = target_score * interface_weight
        # then take the average of the data
        data_weighted[result_file.split(".")[0]] = target_score.sum(axis=1)
        # then multiply by the EU_weight
        data_weighted[result_file.split(".")[0]] = data_weighted[result_file.split(
            ".")[0]] * EU_weight

    data = data.fillna(impute_value)
    data_file = f"group_by_target-{feature}-{model}-{mode}-impute_value={impute_value}.csv"
    data.to_csv(out_dir + data_file)
    data_raw_file = f"group_by_target_raw-{feature}-{model}-{mode}-impute_value={impute_value}.csv"
    data_raw.to_csv(out_dir + data_raw_file)
    data["sum"] = data.sum(axis=1)
    data = data.sort_values(by="sum", ascending=False)
    sum_data_file = f"sum_{feature}-{model}-{mode}-impute_value={impute_value}.csv"
    data.to_csv(out_dir + sum_data_file)

    data_weighted["sum"] = data_weighted.sum(axis=1)
    data_weighted = data_weighted.sort_values(by="sum", ascending=False)
    sum_data_weighted_file = f"sum_weighted_EU_{feature}-{model}-{mode}-impute_value={impute_value}.csv"
    data_weighted.to_csv(out_dir + sum_data_weighted_file)

    return data_raw, weight_list

# results_dir = "/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/model_results/"
# results_dir = "/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/nr_interfaces/"

# out_dir = "./group_by_target_per_interface/"
# model = "best"
# model = "first"
# mode = "all"
# impute_value = -2


# group_by_target(results_dir, result_files, out_dir,
#                 "dockq_max", model, mode, impute_value)
# group_by_target(results_dir, result_files, out_dir,
#                 "qs_best_max", model, mode, impute_value)
# group_by_target(results_dir, result_files, out_dir,
#                 "qs_global_max", model, mode, impute_value)
# group_by_target(results_dir, result_files, out_dir,
#                 "ics_max", model, mode, impute_value)
# group_by_target(results_dir, result_files, out_dir,
#                 "ips_max", model, mode, impute_value)


# group_by_target(results_dir, result_files, out_dir,
#                 "dockq_mean", model, mode, impute_value)
# group_by_target(results_dir, result_files, out_dir,
#                 "qs_best_mean", model, mode, impute_value)
# group_by_target(results_dir, result_files, out_dir,
#                 "qs_global_mean", model, mode, impute_value)
# group_by_target(results_dir, result_files, out_dir,
#                 "ics_mean", model, mode, impute_value)
# group_by_target(results_dir, result_files, out_dir,
#                 "ips_mean", model, mode, impute_value)

# features = ['dockq_mean', 'qs_best_mean', 'qs_global_mean',
#             'ics_mean', 'ips_mean']

parser = argparse.ArgumentParser(
    description="options for interface score processing")
# parser.add_argument("--features", type=list,
#                     default=['qs_global_mean', 'dockq_mean', 'qs_best_mean',
#                              'ics_mean', 'ips_mean'])
parser.add_argument("--features", type=list,
                    default=['qs_global_mean', 'dockq_mean', 'qs_best_mean',
                             'ics_mean', 'ips_mean'])
parser.add_argument("--results_dir", type=str,
                    default="/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/nr_interfaces/")
parser.add_argument("--out_dir", type=str,
                    default="./group_by_target_per_interface_6/")
parser.add_argument("--model", type=str, default="msa")
parser.add_argument("--mode", type=str, default="all")
parser.add_argument("--impute_value", type=int, default=-2)

args = parser.parse_args()
results_dir = args.results_dir
out_dir = args.out_dir
model = args.model
mode = args.mode
impute_value = args.impute_value
features = args.features
result_files = [result for result in os.listdir(
    results_dir) if result.endswith(".results")]
removed_targets = [
    "T1219",
    "T1269",
    "H1265",
    "T1295",
    "T1246",
]
removed_targets = [
    # "T1219",
    "T1246",
    # "T1269",
    "T1269v1o_",
    # "T1295",
    "T1295o.",
    # "T1249",
    # "H1265",
    "H1265_",
    "T2270o",
]


to_remove = []
for result_file in result_files:
    for removed_target in removed_targets:
        if result_file.startswith(removed_target):
            to_remove.append(result_file)
            break
for remove in to_remove:
    result_files.remove(remove)
result_files = sorted(result_files)
print(result_files.__len__())
# breakpoint()
for feature in features:
    data_raw, weight_list = group_by_target(results_dir, result_files, out_dir,
                                            feature, model, mode, impute_value)
    colab_path = "/home2/s439906/project/CASP16/oligomer/group_by_target_per_interface/" + \
        f"group_by_target_raw-{feature}-first-all-impute_value=-2.csv"
    colab_data = pd.read_csv(colab_path, index_col=0)
    colab_data = pd.DataFrame(colab_data.loc["TS145"]).T
    all_data = pd.concat([data_raw, colab_data], axis=0) * weight_list
    all_data = all_data.T
    all_data['target'] = all_data.index.str.split(
        'interface').str[0].str.split('_').str[0]
    all_data = all_data.groupby('target').sum()
    # change 0 to nan
    all_data = all_data.replace(0, np.nan)
    # remove columns with more than 0.5 nan values
    # compute non NaN values
    all_data = all_data.loc[:, all_data.isna().mean() <= 0.5]
    all_data = all_data.T
    colab_data = pd.DataFrame(all_data.loc["TS145"]).T
    colab_data_clean = colab_data.dropna(axis=1)
    # 只保留 data_df 中同时存在于 colab_data_clean 中的列
    common_columns = all_data.columns.intersection(colab_data_clean.columns)
    # 提取包含共同列的 data_df 和 colab_data_clean
    data_df_filtered = all_data[common_columns]
    colab_data_filtered = colab_data_clean[common_columns]
    data_df_filled = data_df_filtered.fillna(0)
    ts145_values = data_df_filled.loc["TS145"]
    normalized_data = data_df_filled.div(ts145_values, axis=1)
    # max_value = np.max(normalized_data)
    # norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=max_value)

    colors = ["#d73027", "#ffffff", "#4575b4"]  # 红色, 白色, 蓝色
    max_value = 15
    cmap = LinearSegmentedColormap.from_list("red_white_blue", colors)
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=15)

    plt.figure(figsize=(20, 7.5))
    sns.heatmap(normalized_data, annot=True,
                fmt=".2f", cmap=cmap, norm=norm, square=True, vmax=max_value)
    plt.yticks(rotation=0, fontsize=12)
    plt.xticks(rotation=90, fontsize=12)
    plt.xlabel("Target", fontsize=12)
    plt.title(f"{feature} values with baseline_first", fontsize=16)
    plt.savefig(
        "./png/heatmap-{}-colab_first.png".format(feature), dpi=300)

    greater_than_1_count = (normalized_data >= 1.01).sum(axis=1)
    less_than_1_count = (normalized_data < 0.99).sum(axis=1)
    zero_count = (normalized_data == 0).sum(axis=1)
    print(greater_than_1_count)
    greater_than_1_count = greater_than_1_count.sort_values(ascending=False)
    print(dict(greater_than_1_count))
    less_than_1_count = less_than_1_count.sort_values(ascending=False)
    print(dict(less_than_1_count))
    zero_count = zero_count.sort_values(ascending=False)
    print(dict(zero_count))

    print(less_than_1_count)
    # get how many non-zero values in each row
    non_zero_count = normalized_data.astype(bool).sum(axis=1)
    # sum each row
    normalized_data["sum"] = normalized_data.sum(axis=1)
    # then divide sum by non_zero_count
    normalized_data["mean"] = normalized_data["sum"] / non_zero_count

    breakpoint()
    groups = all_data.columns
    colab_data = pd.DataFrame(all_data['TS145'])
    colab_data_clean = colab_data.dropna(axis=1)
    # 只保留 data_df 中同时存在于 colab_data_clean 中的列
    common_columns = data_raw.columns.intersection(colab_data_clean.columns)
    # 提取包含共同列的 data_df 和 colab_data_clean
    data_df_filtered = data_raw[common_columns]
    breakpoint()
    colab_data_filtered = colab_data_clean[common_columns]

    merged_data = pd.concat([data_df_filtered, colab_data_filtered], axis=0)

    win_matrix = np.zeros((len(groups), len(groups)))
    breakpoint()
    score_dict = {}
    for group in groups:
        score_dict[group] = 0
    for i in range(len(groups)):
        for j in range(len(groups)):
            group1 = groups[i]
            group2 = groups[j]
            # calculate the non NaN values
            group1_clean = all_data[group1].dropna()
            group2_clean = all_data[group2].dropna()
            # calculate the common index
            common_index = group1_clean.index.intersection(group2_clean.index)
            group1_data = group1_clean.loc[common_index]
            group2_data = group2_clean.loc[common_index]
            group1_score = group1_data.sum()
            group2_score = group2_data.sum()
            ratio = group1_score / group2_score
            win_matrix[i, j] = ratio
            if ratio > 1:
                score_dict[group1] += 1
            # breakpoint()

    breakpoint()
