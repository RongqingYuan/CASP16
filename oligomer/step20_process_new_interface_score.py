import argparse
import sys
import os
import pandas as pd
import numpy as np


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
            nr_interface_size_weights.append(size_weight)
        nr_interface_size_weights = np.log10(nr_interface_size_weights)
        nr_interface_size_weights = np.array(nr_interface_size_weights)
        # nr_interface_size_weights = nr_interface_size_weights / nr_interface_size_weights.sum()

        # elementwise multiplication
        interface_weight = nr_interface_weights * nr_interface_size_weights
        # normalize the weights
        interface_weight = interface_weight / interface_weight.sum()
        print(interface_weight)
        EU_weight = data_tmp_split.shape[1] ** (1/2)
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
        data_weighted[result_file.split(".")[0]] = target_score.mean(axis=1)
        # then multiply by the EU_weight
        data_weighted[result_file.split(".")[0]] = data_weighted[result_file.split(
            ".")[0]] * EU_weight

    # breakpoint()
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
parser.add_argument("--features", type=list,
                    default=['dockq_mean', 'qs_best_mean', 'qs_global_mean',
                             'ics_mean', 'ips_mean'])
parser.add_argument("--results_dir", type=str,
                    default="/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/nr_interfaces/")
parser.add_argument("--out_dir", type=str,
                    default="./group_by_target_per_interface/")
parser.add_argument("--model", type=str, default="best")
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
removed_targets = ["T1219",
                   "T1269",
                   "H1265",
                   "T1295",
                   "T1249"]
to_remove = []
for result_file in result_files:
    for removed_target in removed_targets:
        if result_file.startswith(removed_target):
            to_remove.append(result_file)
            break
for remove in to_remove:
    result_files.remove(remove)
result_files = sorted(result_files)
# print(result_files.__len__())


for feature in features:
    group_by_target(results_dir, result_files, out_dir,
                    feature, model, mode, impute_value)
