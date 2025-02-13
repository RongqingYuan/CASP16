import argparse
import sys
import os
import pandas as pd
import numpy as np


def group_by_target(results_dir, result_files, out_dir,
                    feature, model, mode, impute_value):
    interface_weight_dict = {}
    EU_weight_dict = {}
    data = pd.DataFrame()
    data_weighted = pd.DataFrame()
    data_raw = pd.DataFrame()
    print("Processing {}".format(feature))
    for result_file in result_files:
        data = pd.DataFrame()
        data_weighted = pd.DataFrame()
        data_raw = pd.DataFrame()
        # print("Processing {}".format(result_file))
        result_path = results_dir + result_file
        data = pd.read_csv(result_path, index_col=0)
        data_tmp = pd.DataFrame(data[feature]).astype(str)
        data_tmp_split = data_tmp[feature].str.split(',', expand=True)
        data_tmp_split.columns = [
            f'interface_{i+1}' for i in range(data_tmp_split.shape[1])]
        # fill "-" with nan
        data_tmp_split = data_tmp_split.replace("-", np.nan)
        # fill "None" with nan
        data_tmp_split = data_tmp_split.replace("None", np.nan)
        # convert everything to float
        data_tmp_split = data_tmp_split.astype(float)

        if feature.startswith("prot_nucl_per"):
            nr_interf_group_interfsize_in_ref = data["prot_nucl_interfaces_sizes"]
            if nr_interf_group_interfsize_in_ref.nunique() == 1:
                nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.astype(
                    str)
                pass
            else:
                print("Not all values are the same in nr_interf_group_interfsize_in_ref")
        elif feature.startswith("prot_per"):
            nr_interf_group_interfsize_in_ref = data["prots_interfaces_size"]
            if nr_interf_group_interfsize_in_ref.nunique() == 1:
                nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.astype(
                    str)
                pass
            else:
                print("Not all values are the same in nr_interf_group_interfsize_in_ref")
        else:
            print("feature name is wrong")

        nr_refinterf_num = ";".join(data_tmp_split.shape[1] * ["1"])
        nr_interface_weights = nr_refinterf_num.split(";")
        nr_interface_weights = [float(weight)
                                for weight in nr_interface_weights]
        nr_interface_weights = np.array(nr_interface_weights)
        print('target: ', result_file, nr_interface_weights)

        # nr_interf_group_interfsize_in_ref = ";".join(
        #     data_tmp_split.shape[1] * ["1"])
        # nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.split(
        #     ";")
        # nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.iloc[0]

        for i in range(data_tmp_split.shape[0]):
            if nr_interf_group_interfsize_in_ref.iloc[i] == "-":
                print(nr_interf_group_interfsize_in_ref.iloc[i])
                continue
            else:
                break
        nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.iloc[i]
        if nr_interf_group_interfsize_in_ref == "nan":
            # in this case it simply means that there is no protein interface
            # just use some random value
            print(f"no protein interface for target {result_file}")
            nr_interf_group_interfsize_in_ref = '1/1'
        nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.split(
            ",")

        nr_interface_size_weights = []
        # for i in range(len(nr_interf_group_interfsize_in_ref)):
        #     size_info = nr_interf_group_interfsize_in_ref[i].split(",")
        #     size_weight = 0
        #     for j in range(len(size_info)):
        #         sizes = size_info[j].split("/")
        #         for size in sizes:
        #             size_weight += int(size)
        #     nr_interface_size_weights.append(size_weight/2)

        for i in range(len(nr_interf_group_interfsize_in_ref)):
            size_info = nr_interf_group_interfsize_in_ref[i].split("/")
            print(size_info)
            size_weight = 0
            for size in size_info:
                size_weight += int(size)
            nr_interface_size_weights.append(size_weight/2)
        nr_interface_size_weights = np.log10(nr_interface_size_weights)
        nr_interface_size_weights = np.array(nr_interface_size_weights)
        # nr_interface_size_weights = nr_interface_size_weights / nr_interface_size_weights.sum()

        # nr_interface_size_weights = [float(weight)
        #                              for weight in nr_interf_group_interfsize_in_ref]
        nr_interface_size_weights = np.array(nr_interface_size_weights)
        # elementwise multiplication
        interface_weight = nr_interface_weights * nr_interface_size_weights
        interface_weight = interface_weight / interface_weight.sum()
        interface_weight_dict[result_file] = interface_weight
        # print(interface_weight)
        # EU_weight = data_tmp_split.shape[1] ** (1/3)
        EU_weight = data_tmp_split.shape[1]
        EU_weight_dict[result_file] = EU_weight

        target_score = data_tmp_split * interface_weight
        data_weighted[result_file.split(".")[0]] = target_score.sum(axis=1)
        # data_weighted[result_file.split(".")[0]] = data_weighted[result_file.split(".")[0]] * EU_weight
        # save data_tmp_split to csv
        data_weighted.to_csv(out_dir + result_file.split(".")[0]
                             + "_" + feature + ".csv")

    with open(out_dir + "interface_weight.txt", "w") as f:
        for key in interface_weight_dict:
            target = key.split(".")[0]
            f.write(f"{target}\t{interface_weight_dict[key]}\n")
    with open(out_dir + "EU_weight.txt", "w") as f:
        for key in EU_weight_dict:
            target = key.split(".")[0]
            f.write(f"{target}\t{EU_weight_dict[key]}\n")
    return 0
    for result_file in result_files:
        print("Processing {}".format(result_file))
        result_path = results_dir + result_file
        data_tmp = pd.read_csv(result_path, index_col=0)
        data_tmp = pd.DataFrame(data_tmp[feature]).astype(str)
        data_tmp_split = data_tmp[feature].str.split(',', expand=True)
        data_tmp_split.columns = [
            f'interface_{i+1}' for i in range(data_tmp_split.shape[1])]
        # fill "-" with nan
        data_tmp_split = data_tmp_split.replace("-", np.nan)
        # fill "None" with nan
        data_tmp_split = data_tmp_split.replace("None", np.nan)
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
        data_tmp = pd.read_csv(result_path, index_col=0)
        if feature.startswith("prot_nucl_per"):
            nr_interf_group_interfsize_in_ref = data_tmp["prot_nucl_interfaces_sizes"]
            if nr_interf_group_interfsize_in_ref.nunique() == 1:
                nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.astype(
                    str)
                pass
            else:
                print("Not all values are the same in nr_interf_group_interfsize_in_ref")
        elif feature.startswith("prot_per"):
            nr_interf_group_interfsize_in_ref = data_tmp["prots_interfaces_size"]
            if nr_interf_group_interfsize_in_ref.nunique() == 1:
                nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.astype(
                    str)
                pass
            else:
                print("Not all values are the same in nr_interf_group_interfsize_in_ref")
        data_tmp = pd.DataFrame(data_tmp[feature]).astype(str)
        data_tmp_split = data_tmp[feature].str.split(',', expand=True)
        data_tmp_split.columns = [
            f'interface_{i+1}' for i in range(data_tmp_split.shape[1])]
        # fill "-" with nan
        data_tmp_split = data_tmp_split.replace("-", np.nan)
        # fill "None" with nan
        data_tmp_split = data_tmp_split.replace("None", np.nan)
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

        nr_refinterf_num = ";".join(data_tmp_split.shape[1] * ["1"])
        nr_interface_weights = nr_refinterf_num.split(";")
        nr_interface_weights = [float(weight)
                                for weight in nr_interface_weights]
        nr_interface_weights = np.array(nr_interface_weights)

        # nr_interf_group_interfsize_in_ref = ";".join(
        #     data_tmp_split.shape[1] * ["1"])
        # nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.split(
        #     ";")
        # nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.iloc[0]
        for i in range(data_tmp_split.shape[0]):
            if nr_interf_group_interfsize_in_ref.iloc[i] == "-":
                print(nr_interf_group_interfsize_in_ref.iloc[i])
                continue
            else:
                break
        nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.iloc[i]
        if nr_interf_group_interfsize_in_ref == "nan":
            # in this case it simply means that there is no protein interface
            # just use some random value
            nr_interf_group_interfsize_in_ref = '1/1'
        nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.split(
            ",")

        nr_interface_size_weights = []
        # for i in range(len(nr_interf_group_interfsize_in_ref)):
        #     size_info = nr_interf_group_interfsize_in_ref[i].split(",")
        #     size_weight = 0
        #     for j in range(len(size_info)):
        #         sizes = size_info[j].split("/")
        #         for size in sizes:
        #             size_weight += int(size)
        #     nr_interface_size_weights.append(size_weight/2)

        for i in range(len(nr_interf_group_interfsize_in_ref)):
            size_info = nr_interf_group_interfsize_in_ref[i].split("/")
            print(size_info)
            size_weight = 0
            for size in size_info:
                size_weight += int(size)
            nr_interface_size_weights.append(size_weight/2)
        nr_interface_size_weights = np.log10(nr_interface_size_weights)
        nr_interface_size_weights = np.array(nr_interface_size_weights)
        # nr_interface_size_weights = nr_interface_size_weights / nr_interface_size_weights.sum()

        # nr_interface_size_weights = [float(weight)
        #                              for weight in nr_interf_group_interfsize_in_ref]
        nr_interface_size_weights = np.array(nr_interface_size_weights)
        # elementwise multiplication
        interface_weight = nr_interface_weights * nr_interface_size_weights
        interface_weight = interface_weight / interface_weight.sum()
        print(interface_weight)
        EU_weight = data_tmp_split.shape[1] ** (1/3)
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

        # continue
        # breakpoint()
        # # first weight by the number of interfaces
        # target_score = target_score * nr_interface_weights
        # # then weight by the size of the interfaces
        # target_score = target_score * nr_interface_size_weights
        target_score = target_score * interface_weight
        data_weighted[result_file.split(".")[0]] = target_score.sum(axis=1)
        data_weighted[result_file.split(".")[0]] = data_weighted[result_file.split(
            ".")[0]] * EU_weight

    # don't sort data_raw
    data_raw_file = f"{feature}-{model}-{mode}-impute_value={impute_value}-raw.csv"
    data_raw.to_csv(out_dir + data_raw_file)

    # sort columns by alphabetical order
    data = data.reindex(sorted(data.columns), axis=1)
    # sort rows by alphabetical order
    data = data.sort_index()
    data = data.fillna(impute_value)
    data_file = f"{feature}-{model}-{mode}-impute_value={impute_value}.csv"
    data.to_csv(out_dir + data_file)

    data["sum"] = data.sum(axis=1)
    data = data.sort_values(by="sum", ascending=False)
    sum_data_file = f"{feature}-{model}-{mode}-impute_value={impute_value}-sum.csv"
    data.to_csv(out_dir + sum_data_file)

    # sort by column_name
    data_weighted = data_weighted.reindex(
        sorted(data_weighted.columns), axis=1)
    # sort by row_name
    data_weighted = data_weighted.sort_index()
    data_weighted["sum"] = data_weighted.sum(axis=1)
    # data_weighted = data_weighted.sort_values(by="sum", ascending=False)
    sum_data_weighted_file = f"{feature}-{model}-{mode}-impute_value={impute_value}-weighted-sum.csv"
    data_weighted.to_csv(out_dir + sum_data_weighted_file)


# results_dir = "/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/model_results/"
# results_dir = "/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/nr_interfaces/"

parser = argparse.ArgumentParser(
    description="options for interface score processing")
parser.add_argument("--input", type=str, default="./M_target_csv_v1/")
parser.add_argument("--type", type=str, default="pn")
# parser.add_argument("--features", type=list,
#                     default=[
#                         # "prot_nucl_qs_global",

#                         # "prot_nucl_per_interface_qs_best",
#                         # "prot_nucl_per_interface_ics_trimmed",
#                         # "prot_nucl_per_interface_ips_trimmed",

#                         "prot_per_interface_qs_best",
#                         "prot_per_interface_ics_trimmed",
#                         "prot_per_interface_ips_trimmed",
#                     ]
#                     )
# parser.add_argument("--output_dir", type=str, default="./step1_pp/")
parser.add_argument("--score_type", type=str, default="all")
parser.add_argument("--model", type=str, default="best")
parser.add_argument("--mode", type=str, default="all")
parser.add_argument("--impute_value", type=int, default=-2)

args = parser.parse_args()
input_dir = args.input
model = args.model
mode = args.mode
impute_value = args.impute_value

if args.type == "pp":
    features = [
        "prot_per_interface_qs_best",
        "prot_per_interface_ics_trimmed",
        "prot_per_interface_ips_trimmed",
    ]
    output_dir = "./step1_pp/"
elif args.type == "pn":
    features = [
        "prot_nucl_per_interface_qs_best",
        "prot_nucl_per_interface_ics_trimmed",
        "prot_nucl_per_interface_ips_trimmed",
    ]
    output_dir = "./step1_pn/"
else:
    print("wrong type")
    sys.exit(1)
result_files = [result for result in os.listdir(
    input_dir) if result.endswith(".csv")]

bad_targets = ["M1268", "M1297"]
remove = []
for file in result_files:
    for target in bad_targets:
        if file.startswith(target):
            remove.append(file)
for target in remove:
    result_files.remove(target)


result_files = sorted(result_files)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for feature in features:
    group_by_target(input_dir, result_files, output_dir,
                    feature, model, mode, impute_value)
