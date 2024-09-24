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

        # get the first nr_refinterf_num
        nr_refinterf_num = nr_refinterf_num.iloc[0]
        nr_interface_weights = nr_refinterf_num.split(";")
        nr_interface_weights = [float(weight)
                                for weight in nr_interface_weights]

        # get the first nr_interf_group_interfsize_in_ref
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
        # take log 10 of the nr_interface_size_weights
        nr_interface_size_weights = np.log10(nr_interface_size_weights)

        EU_weight = data_tmp_split.shape[1] ** (1/3)
        # if result_file.startswith("H1236"):
        #     breakpoint()
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

        # get the first nr_refinterf_num
        nr_refinterf_num = nr_refinterf_num.iloc[0]
        nr_interface_weights = nr_refinterf_num.split(";")
        nr_interface_weights = [float(weight)
                                for weight in nr_interface_weights]
        # normalize nr_interface_weights to sum to 1
        nr_interface_weights = np.array(nr_interface_weights)
        # nr_interface_weights = nr_interface_weights / nr_interface_weights.sum()

        # get the first nr_interf_group_interfsize_in_ref
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
        # take log 10 of the nr_interface_size_weights
        nr_interface_size_weights = np.log10(nr_interface_size_weights)
        # normalize nr_interface_size_weights to sum to 1
        nr_interface_size_weights = np.array(nr_interface_size_weights)
        # nr_interface_size_weights = nr_interface_size_weights / \
        #     nr_interface_size_weights.sum()
        EU_weight = data_tmp_split.shape[1] ** (1/3)
        # if result_file.startswith("H1236"):
        #     breakpoint()

        # get a target_score df that has the same row as data_raw
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
        # first weight by the number of interfaces
        target_score = target_score * nr_interface_weights
        # then weight by the size of the interfaces
        target_score = target_score * nr_interface_size_weights
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
results_dir = "/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/nr_interfaces/"
result_files = [result for result in os.listdir(
    results_dir) if result.endswith(".results")]
removed_targets = ["T1219",
                   "T1269",
                   "H1265",
                   "T1295",
                   "T1249"]
print(result_files.__len__())

to_remove = []
for result_file in result_files:
    for removed_target in removed_targets:
        if result_file.startswith(removed_target):
            to_remove.append(result_file)
            break
for remove in to_remove:
    result_files.remove(remove)

# sort result_files by alphabet
result_files = sorted(result_files)
print(result_files.__len__())
print(result_files)

print('H1265_v2.nr_interface.results'.startswith('H1265'))
out_dir = "./group_by_target_per_interface/"
model = "best"
mode = "all"
impute_value = -2


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

group_by_target(results_dir, result_files, out_dir,
                "dockq_mean", model, mode, impute_value)
group_by_target(results_dir, result_files, out_dir,
                "qs_best_mean", model, mode, impute_value)
group_by_target(results_dir, result_files, out_dir,
                "qs_global_mean", model, mode, impute_value)
group_by_target(results_dir, result_files, out_dir,
                "ics_mean", model, mode, impute_value)
group_by_target(results_dir, result_files, out_dir,
                "ips_mean", model, mode, impute_value)
sys.exit(0)
group_by_target(results_dir, result_files, out_dir, "dockq_ave", model, mode)
group_by_target(results_dir, result_files, out_dir, "tm_score", model, mode)
group_by_target(results_dir, result_files, out_dir, "lddt", model, mode)


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
