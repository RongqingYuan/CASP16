import argparse
import sys
import os
import pandas as pd
import numpy as np


def get_EU_raw_score(result_file, feature):
    # print("Processing {}".format(feature))
    # print("Processing {}".format(result_file))
    data = pd.read_csv(result_file, sep="\t", index_col=0)
    data = pd.DataFrame(data[feature])
    min_feature = data[feature].replace(
        ['N/A', '-'], pd.NA).dropna().astype(float).min()
    data[feature] = data[feature].replace(
        ['N/A', '-'], min_feature)
    data[feature] = data[feature].astype(float)
    return data


def get_interface_raw_score(result_file, feature):
    # print("Processing {}".format(feature))
    # print("Processing {}".format(result_file))
    data = pd.read_csv(result_file, sep="\t", index_col=0)
    nr_interf_group_in_ref = data["nr_interf_group_in_ref"].iloc[0]
    nr_interf_group_interfsize_in_ref = data["nr_interf_group_interfsize_in_ref"].iloc[0]
    interface_data = pd.DataFrame(data[feature]).astype(str)

    nr_intefaces = str(nr_interf_group_in_ref).split(";")
    all_interfaces = []
    for nr_interface in nr_intefaces:
        nr_interface = nr_interface.split(",")
        for interface in nr_interface:
            all_interfaces.append(interface)

    nr_inteface_sizes = str(nr_interf_group_interfsize_in_ref).split(";")
    all_interface_sizes = []
    for nr_interface_size in nr_inteface_sizes:
        nr_interface_size = nr_interface_size.split(",")
        for interface_size in nr_interface_size:
            size1 = int(interface_size.split("/")[0])
            size2 = int(interface_size.split("/")[1])
            size = (size1 + size2) / 2
            all_interface_sizes.append(size)

    expanded_interface_data = interface_data[feature].apply(
        lambda row: [val for part in row.split(';') for val in part.split(',')])
    all_interface_scores = pd.DataFrame(expanded_interface_data.tolist(), columns=[
        f'interface_{i}' for i in range(len(expanded_interface_data[0]))], index=interface_data.index)

    assert all_interface_scores.shape[1] == len(all_interfaces)
    assert all_interface_scores.shape[1] == len(all_interface_sizes)
    size_weight = np.log10(all_interface_sizes)
    if "H1204" in result_file:
        # breakpoint()
        ...
    # normalize the size_weight
    size_weight = size_weight / size_weight.sum()
    # each column has a value called None. Impute None with the minimum value of that column
    # Impute None values with the minimum value of each column
    all_interface_scores = all_interface_scores.apply(
        pd.to_numeric, errors='coerce')
    # all_interface_scores = all_interface_scores.apply(
    #     lambda x: x.fillna(x.min()), axis=0)
    all_interface_scores = all_interface_scores.apply(
        lambda x: x.fillna(0), axis=0)

    weighted_interface_scores = all_interface_scores.astype(
        float).multiply(size_weight, axis=1)
    weighted_sum_score = weighted_interface_scores.sum(axis=1).to_frame()
    weighted_sum_score.columns = [feature.split("_")[0]]
    return weighted_sum_score

    # for result_file in result_files:
    #     data = pd.read_csv(result_path, sep="\t", index_col=0)
    #     data = pd.read_csv(result_path, sep="\t", index_col=0)
    #     data = pd.DataFrame(data[feature])
    #     breakpoint()
    #     ############################
    #     nr_interf_group_in_ref = data_tmp["nr_interf_group_in_ref"]
    #     if nr_interf_group_in_ref.nunique() == 1:
    #         pass
    #     else:
    #         print("Not all values are the same in nr_interf_group_in_ref")
    #     nr_refinterf_num = data_tmp["nr_refinterf_#"]
    #     if nr_refinterf_num.nunique() == 1:
    #         nr_refinterf_num = nr_refinterf_num.astype(str)
    #         pass
    #     else:
    #         print("Not all values are the same in nr_refinterf_#")
    #     nr_interf_group_interfsize_in_ref = data_tmp["nr_interf_group_interfsize_in_ref"]
    #     if nr_interf_group_interfsize_in_ref.nunique() == 1:
    #         nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.astype(
    #             str)
    #         pass
    #     else:
    #         print("Not all values are the same in nr_interf_group_interfsize_in_ref")
    #     ############################

    #     data_tmp = pd.DataFrame(data_tmp[feature]).astype(str)
    #     data_tmp_split = data_tmp[feature].str.split(';', expand=True)
    #     data_tmp_split.columns = [
    #         f'interface_{i+1}' for i in range(data_tmp_split.shape[1])]
    #     # fill "-" with nan
    #     data_tmp_split = data_tmp_split.replace("-", np.nan)
    #     data_tmp_split = data_tmp_split.astype(float)

    #     data_tmp_split.index = data_tmp_split.index.str.extract(
    #         r'(\w+)TS(\w+)_(\w+)').apply(lambda x: (f"{x[0]}", f"TS{x[1]}", x[2][0]), axis=1)
    #     data_tmp_split.index = pd.MultiIndex.from_tuples(
    #         data_tmp_split.index, names=['target', 'group', 'submission_id'])
    #     if model == "best":
    #         data_tmp_split = data_tmp_split.loc[(slice(None), slice(None), [
    #             "1", "2", "3", "4", "5"]), :]
    #     elif model == "first":
    #         data_tmp_split = data_tmp_split.loc[(slice(None), slice(None),
    #                                              "1"), :]
    #     elif model == "sixth":
    #         data_tmp_split = data_tmp_split.loc[(slice(None), slice(None),
    #                                              "6"), :]
    #     for i in range(data_tmp_split.shape[1]):
    #         grouped = data_tmp_split.groupby(["group"])
    #         feature_name = f"interface_{i+1}"
    #         grouped = pd.DataFrame(grouped[feature_name].max())
    #         grouped = grouped.sort_values(by=feature_name, ascending=False)
    #         initial_z = (grouped - grouped.mean()) / grouped.std()
    #         new_z_score = pd.DataFrame(
    #             index=grouped.index, columns=grouped.columns)
    #         filtered_data = grouped[feature_name][initial_z[feature_name] >= -2]
    #         new_mean = filtered_data.mean(skipna=True)
    #         new_std = filtered_data.std(skipna=True)
    #         new_z_score[feature_name] = (
    #             grouped[feature_name] - new_mean) / new_std
    #         new_z_score = new_z_score.fillna(impute_value)
    #         new_z_score = new_z_score.where(
    #             new_z_score > impute_value, impute_value)

    #         new_z_score = new_z_score.rename(
    #             columns={feature_name: result_file.split(".")[0]+"_"+feature_name})
    #         data = pd.concat([data, new_z_score], axis=1)
    #         grouped = grouped.rename(
    #             columns={feature_name: result_file.split(".")[0]+"_"+feature_name})
    #         data_raw = pd.concat([data_raw, grouped], axis=1)

    # for result_file in result_files:
    #     print("Processing {}".format(result_file))
    #     result_path = results_dir + result_file
    #     data_tmp = pd.read_csv(result_path, sep="\t", index_col=0)

    #     ############################
    #     nr_interf_group_in_ref = data_tmp["nr_interf_group_in_ref"]
    #     if nr_interf_group_in_ref.nunique() == 1:
    #         pass
    #     else:
    #         print("Not all values are the same in nr_interf_group_in_ref")
    #     nr_refinterf_num = data_tmp["nr_refinterf_#"]
    #     if nr_refinterf_num.nunique() == 1:
    #         nr_refinterf_num = nr_refinterf_num.astype(str)
    #         pass
    #     else:
    #         print("Not all values are the same in nr_refinterf_#")
    #     nr_interf_group_interfsize_in_ref = data_tmp["nr_interf_group_interfsize_in_ref"]
    #     if nr_interf_group_interfsize_in_ref.nunique() == 1:
    #         nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.astype(
    #             str)
    #         pass
    #     else:
    #         print("Not all values are the same in nr_interf_group_interfsize_in_ref")
    #     ############################

    #     data_tmp = pd.DataFrame(data_tmp[feature]).astype(str)
    #     data_tmp_split = data_tmp[feature].str.split(';', expand=True)
    #     data_tmp_split.columns = [
    #         f'interface_{i+1}' for i in range(data_tmp_split.shape[1])]
    #     data_tmp_split = data_tmp_split.replace("-", np.nan)
    #     data_tmp_split = data_tmp_split.astype(float)

    #     nr_refinterf_num = nr_refinterf_num.iloc[0]
    #     nr_interface_weights = nr_refinterf_num.split(";")
    #     nr_interface_weights = [float(weight)
    #                             for weight in nr_interface_weights]
    #     nr_interface_weights = np.array(nr_interface_weights)
    #     # nr_interface_weights = nr_interface_weights / nr_interface_weights.sum()

    #     nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.iloc[0]
    #     nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.split(
    #         ";")

    #     nr_interface_size_weights = []
    #     for i in range(len(nr_interf_group_interfsize_in_ref)):
    #         size_info = nr_interf_group_interfsize_in_ref[i].split(",")
    #         size_weight = 0
    #         for j in range(len(size_info)):
    #             sizes = size_info[j].split("/")
    #             for size in sizes:
    #                 size_weight += int(size)
    #         nr_interface_size_weights.append(size_weight/2)
    #     nr_interface_size_weights = np.log10(nr_interface_size_weights)
    #     nr_interface_size_weights = np.array(nr_interface_size_weights)
    #     # nr_interface_size_weights = nr_interface_size_weights / nr_interface_size_weights.sum()

    #     # elementwise multiplication
    #     interface_weight = nr_interface_weights * nr_interface_size_weights
    #     interface_weight = interface_weight / interface_weight.sum()
    #     print(interface_weight)
    #     EU_weight = data_tmp_split.shape[1] ** (1/3)

    #     # # get a target_score df that has the same row as data_raw to make sure no nan values
    #     # target_score = pd.DataFrame(index=data_raw.index)
    #     # for i in range(data_tmp_split.shape[1]):
    #     #     grouped = data_tmp_split.groupby(["group"])
    #     #     feature_name = f"interface_{i+1}"
    #     #     grouped = pd.DataFrame(grouped[feature_name].max())
    #     #     grouped = grouped.sort_values(by=feature_name, ascending=False)
    #     #     initial_z = (grouped - grouped.mean()) / grouped.std()
    #     #     new_z_score = pd.DataFrame(
    #     #         index=grouped.index, columns=grouped.columns)
    #     #     filtered_data = grouped[feature_name][initial_z[feature_name] >= -2]
    #     #     new_mean = filtered_data.mean(skipna=True)
    #     #     new_std = filtered_data.std(skipna=True)
    #     #     new_z_score[feature_name] = (
    #     #         grouped[feature_name] - new_mean) / new_std
    #     #     new_z_score = new_z_score.fillna(impute_value)
    #     #     new_z_score = new_z_score.where(
    #     #         new_z_score > impute_value, impute_value)
    #     #     target_score = pd.concat([target_score, new_z_score], axis=1)
    #     #     target_score = target_score.fillna(impute_value)

    #     # # get a target_score df that has the same row as data_raw to make sure no nan values
    #     # target_score = pd.DataFrame(index=data_raw.index)
    #     # for i in range(data_tmp_split.shape[1]):
    #     #     grouped = data_tmp_split.groupby(["group"])
    #     #     feature_name = f"interface_{i+1}"
    #     #     grouped = pd.DataFrame(grouped[feature_name].max())
    #     #     # grouped = grouped.sort_values(by=feature_name, ascending=False)
    #     #     target_score = pd.concat([target_score, grouped], axis=1)
    #     #     # target_score = target_score.fillna(impute_value)

    #     # template_score = pd.DataFrame(index=data_raw.index)
    #     data_tmp_split = data_tmp_split.fillna(0)
    #     data_tmp_split = data_tmp_split * interface_weight
    #     if not os.path.exists(out_dir + "weighted_raw/"):
    #         os.makedirs(out_dir + "weighted_raw/")
    #     target_score_tmp = data_tmp_split.sum(axis=1).to_frame()
    #     # change the column name to the result_file name
    #     target_score_tmp.columns = [feature]
    #     if data_int_weighted is None:
    #         data_int_weighted = target_score_tmp
    #     else:
    #         data_int_weighted = pd.concat(
    #             [data_int_weighted, target_score_tmp])
    #     # target_score_tmp.to_csv(
    #     #     out_dir + "weighted_raw/" + result_file.split(".")[0] + "_" + feature + ".csv")

    #     # data_int_weighted[result_file.split(".")[0]] = target_score_tmp[
    #     #     result_file.split(".")[0]]
    #     data_tmp_split.index = data_tmp_split.index.str.extract(
    #         r'(\w+)TS(\w+)_(\w+)').apply(lambda x: (f"{x[0]}", f"TS{x[1]}", x[2][0]), axis=1)
    #     data_tmp_split.index = pd.MultiIndex.from_tuples(
    #         data_tmp_split.index, names=['target', 'group', 'submission_id'])
    #     if model == "best":
    #         data_tmp_split = data_tmp_split.loc[(slice(None), slice(None), [
    #             "1", "2", "3", "4", "5"]), :]
    #     elif model == "first":
    #         data_tmp_split = data_tmp_split.loc[(slice(None), slice(None),
    #                                              "1"), :]
    #     elif model == "sixth":
    #         data_tmp_split = data_tmp_split.loc[(slice(None), slice(None),
    #                                              "6"), :]
    #     target_score = data_tmp_split.sum(axis=1).to_frame()
    #     print(type(target_score))
    #     grouped = target_score.groupby(["group"])
    #     target_score = pd.DataFrame(grouped.max())
    #     target_score.columns = [result_file.split(".")[0]]
    #     initial_z = (target_score - target_score.mean()) / \
    #         target_score.std()
    #     new_z_score = pd.DataFrame(
    #         index=target_score.index, columns=target_score.columns)
    #     filtered_data = target_score[initial_z >= -2]
    #     new_mean = filtered_data.mean(skipna=True)
    #     new_std = filtered_data.std(skipna=True)
    #     new_z_score = (target_score - new_mean) / new_std
    #     new_z_score = new_z_score.fillna(impute_value)
    #     new_z_score = new_z_score.where(
    #         new_z_score > impute_value, impute_value)
    #     new_z_score = new_z_score.reindex(
    #         data_raw.index, fill_value=impute_value)
    #     new_z_score = new_z_score.where(
    #         new_z_score > impute_value, impute_value)
    #     data_unweighted[result_file.split(
    #         ".")[0]] = new_z_score[result_file.split(".")[0]]
    #     new_z_score = new_z_score * EU_weight
    #     print(new_z_score.shape)
    #     data_weighted[result_file.split(
    #         ".")[0]] = new_z_score[result_file.split(".")[0]]
    #     # take the weighted sum of the interface scores
    #     # target_score = target_score * interface_weight
    #     # data_weighted[result_file.split(".")[0]] = target_score.sum(axis=1)
    #     # # fill with 0 here if there are any
    #     # data_weighted = data_weighted.fillna(0)

    #     # # breakpoint()
    #     # initial_z = (data_weighted - data_weighted.mean()) / \
    #     #     data_weighted.std()
    #     # new_z_score = pd.DataFrame(
    #     #     index=data_weighted.index, columns=data_weighted.columns)
    #     # filtered_data = data_weighted[initial_z >= -2]
    #     # new_mean = filtered_data.mean(skipna=True)
    #     # new_std = filtered_data.std(skipna=True)
    #     # new_z_score = (data_weighted - new_mean) / new_std
    #     # new_z_score = new_z_score.fillna(impute_value)
    #     # new_z_score = new_z_score.where(
    #     #     new_z_score > impute_value, impute_value)
    #     # data_weighted = new_z_score
    #     # data_weighted.reindex(data_raw.index, fill_value=impute_value)
    #     # # then multiply by the EU_weight
    #     # data_weighted[result_file.split(".")[0]] = data_weighted[result_file.split(
    #     #     ".")[0]] * EU_weight
    #     # breakpoint()
    # # data_int_weighted = data_int_weighted.reindex(data_raw.index)
    # data_int_weighted.to_csv(out_dir + f"{feature}_per_model.csv")

    # data = data.fillna(impute_value)
    # data = data.reindex(sorted(data.columns), axis=1)
    # data = data.sort_index()
    # data_file = f"{feature}-{model}-{mode}-impute={impute_value}.csv"
    # data.to_csv(out_dir + data_file)

    # data["sum"] = data.sum(axis=1)
    # data = data.sort_values(by="sum", ascending=False)
    # sum_data_file = f"{feature}-{model}-{mode}-impute={impute_value}_sum.csv"
    # data.to_csv(out_dir + sum_data_file)

    # data_raw = data_raw.reindex(sorted(data_raw.columns), axis=1)
    # data_raw = data_raw.sort_index()
    # data_raw_file = f"{feature}-{model}-{mode}-impute={impute_value}_raw.csv"
    # data_raw.to_csv(out_dir + data_raw_file)

    # data_weighted["sum"] = data_weighted.sum(axis=1)
    # data_weighted = data_weighted.reindex(
    #     sorted(data_weighted.columns), axis=1)
    # data_weighted = data_weighted.sort_index()
    # # data_weighted = data_weighted.sort_values(by="sum", ascending=False)
    # sum_data_weighted_file = f"{feature}-{model}-{mode}-impute={impute_value}_weighted_EU.csv"
    # data_weighted.to_csv(out_dir + sum_data_weighted_file)

    # data_unweighted["sum"] = data_unweighted.sum(axis=1)
    # data_unweighted = data_unweighted.reindex(
    #     sorted(data_unweighted.columns), axis=1)
    # data_unweighted = data_unweighted.sort_index()
    # # data_unweighted = data_unweighted.sort_values(by="sum", ascending=False)
    # sum_data_unweighted_file = f"{feature}-{model}-{mode}-impute={impute_value}_unweighted_EU.csv"
    # data_unweighted.to_csv(out_dir + sum_data_unweighted_file)


parser = argparse.ArgumentParser(
    description="options for data processing")
parser.add_argument("--EU_measures", nargs="+",
                    default=[
                        # "qs_global",
                        # "qs_best",
                        # "ics",
                        # "ips",
                        # "dockq_ave",
                        # "dockq_wave",
                        "lddt",
                        "tm",
                    ])
parser.add_argument("--interface_measures", nargs="+",
                    default=[
                        "ips_per_interface",
                        "ics_per_interface",
                        "qs_best_per_interface",
                        "dockq_per_interface",
                        # "qs_global_mean_inclNone",
                    ])
parser.add_argument("--EU_dir", type=str,
                    default="/data/data1/conglab/qcong/CASP16/stage1_oligomer_inputs/step16/")
parser.add_argument("--interface_dir", type=str,
                    # default="/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/nr_interfaces/"
                    default="/data/data1/conglab/qcong/CASP16/stage1_oligomer_inputs/step16/"
                    )
parser.add_argument("--out_dir", type=str,
                    default="./raw_score/")
parser.add_argument("--model", type=str, default="best")
parser.add_argument("--mode", type=str, default="all")
parser.add_argument("--impute_value", type=int, default=-2)
parser.add_argument("--stage", type=str, default="all")
parser.add_argument("--bad_targets", nargs="+", default=[
    "T1246",
    # "T1249",
    "T1269v1o.",
    # "T1269v1o_",
    "T1295o.",
    "H1265_",
    "T2246",
    # "T2270o",
])

args = parser.parse_args()
EU_measures = args.EU_measures
interface_measures = args.interface_measures
EU_dir = args.EU_dir
interface_dir = args.interface_dir
out_dir = args.out_dir
model = args.model
mode = args.mode
impute_value = args.impute_value
stage = args.stage
bad_targets = args.bad_targets

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if stage == "1":
    bad_targets.extend([
        "T0",
        "T2",
        "H0",
        "H2"
    ])
elif stage == "0":
    bad_targets.extend([
        "T1",
        "T2",
        "H1",
        "H2"
    ])
elif stage == "2":
    bad_targets.extend([
        "T0",
        "T1",
        "H0",
        "H1"
    ])
elif stage == "all":
    pass
else:
    print("Invalid stage")
    sys.exit(1)

result_files = [result for result in os.listdir(
    EU_dir) if (result.endswith(".results") or result.endswith(".result")) and not result.endswith(".nr_interface.results_v2")]
to_remove = []
for result_file in result_files:
    for removed_target in bad_targets:
        if result_file.startswith(removed_target):
            to_remove.append(result_file)
            break
for remove in to_remove:
    result_files.remove(remove)
result_files = sorted(result_files)


result_files = [result for result in os.listdir(
    # interface_dir) if result.endswith(".nr_interface.results_v2")]
    interface_dir) if result.endswith(".result")]
to_remove = []
for result_file in result_files:
    for removed_target in bad_targets:
        if result_file.startswith(removed_target):
            to_remove.append(result_file)
            break
for remove in to_remove:
    result_files.remove(remove)
result_files = sorted(result_files)

assert interface_dir == EU_dir


for result_file in result_files:
    data_all = None
    result_file = interface_dir + result_file
    for measure in interface_measures:
        interface_df = get_interface_raw_score(result_file, measure)
        if data_all is None:
            data_all = interface_df
        else:
            assert data_all.index.equals(interface_df.index)
            data_all = pd.concat([data_all, interface_df], axis=1)
    for measure in EU_measures:
        EU_df = get_EU_raw_score(result_file, measure)
        if data_all is None:
            data_all = EU_df
        else:
            assert data_all.index.equals(EU_df.index)
            data_all = pd.concat([data_all, EU_df], axis=1)

    if data_all.isna().sum().sum() > 0:
        print(f"Warning: {result_file} has missing values")
    # set the first column to Model
    # data_all = data_all.rename(columns={data_all.columns[0]: "Model"})
    data_all.to_csv(out_dir + result_file.split("/")
                    [-1].split(".")[0] + ".csv")
    # BUG 1 some monomer files do not have the correct model 1. We need to manually make the smallest model id to be 1
    lines = []
    with open(out_dir + result_file.split("/")[-1].split(".")[0] + ".csv", "r") as f:
        found_list = []
        for line in f:
            if line.startswith("model"):
                lines.append(line)
            else:
                words = line.split(",")
                model = words[0]
                group, model_id = model.split("_")
                if group not in found_list:
                    found_list.append(group)
                    if not model_id.startswith("1"):
                        model_id = "1" + model_id[1:]
                        print(f"making changes to {group} {model_id}")
                    new_model = group + "_" + model_id
                    words[0] = new_model
                    new_line = ",".join(words)
                    lines.append(new_line)
                else:
                    lines.append(line)
    with open(out_dir + result_file.split("/")[-1].split(".")[0] + ".csv", "w") as f:
        for line in lines:
            f.write(line)
    data = pd.read_csv(out_dir + result_file.split("/")[-1].split(".")[0] + ".csv",
                       index_col=0)
    data = data[~data.index.str.contains('_6')]
    z_score_dir = "./z_score/"
    if not os.path.exists(z_score_dir):
        os.makedirs(z_score_dir)
    initial_z = (data - data.mean()) / data.std(ddof=0)
    new_z_score = pd.DataFrame(index=data.index, columns=data.columns)
    for column in data.columns:
        filtered_data = data[column][initial_z[column] >= -2]
        new_mean = filtered_data.mean(skipna=True)
        new_std = filtered_data.std(skipna=True, ddof=0)
        new_z_score[column] = (data[column] - new_mean) / new_std
    new_z_score = new_z_score.fillna(impute_value)
    new_z_score = new_z_score.where(new_z_score > impute_value, impute_value)
    new_z_score.to_csv(z_score_dir + result_file.split("/")
                       [-1].split(".")[0] + ".csv")


# removed_targets = [
#     "T1219",
#     "T1269",
#     "H1265",
#     "T1295",
#     "T1246",
# ]

# removed_targets = [
#     # "T1219",
#     "T1246",
#     # "T1269",
#     "T1269v1o_",
#     # "T1295",
#     "T1295o.",
#     # "T1249",
#     # "H1265",
#     "H1265_",
#     "T2270o",
# ]

# parser = argparse.ArgumentParser(
#     description="options for interface score processing")
# parser.add_argument("--measures", type=list,
#                     # default=['dockq_mean', 'qs_best_mean', 'qs_global_mean',
#                     #          'ics_mean', 'ips_mean']
#                     default=['dockq_mean_inclNone', 'qs_best_mean_inclNone', 'qs_global_mean_inclNone',
#                              'ics_mean_inclNone', 'ips_mean_inclNone']
#                     )
# parser.add_argument("--results_dir", type=str,
#                     default="/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/nr_interfaces/")
# parser.add_argument("--out_dir", type=str,
#                     default="./group_by_target_per_interface/")
# parser.add_argument("--model", type=str, default="best")
# parser.add_argument("--mode", type=str, default="all")
# parser.add_argument("--impute_value", type=int, default=-2)
# parser.add_argument("--stage", type=str, default="1")
# parser.add_argument("--bad_targets", nargs="+", default=[
#     "T1246",
#     "T1249",
#     "T1269v1o.",
#     # "T1269v1o_",
#     "T1295o.",
#     "H1265_",
#     "T2270o",
# ])

# args = parser.parse_args()
# features = args.measures
# results_dir = args.results_dir
# out_dir = args.out_dir
# model = args.model
# mode = args.mode
# impute_value = args.impute_value
# stage = args.stage
# bad_targets = args.bad_targets

# result_files = [result for result in os.listdir(
#     results_dir) if result.endswith(".nr_interface.results_v2")]
# # removed_targets = [
# #     "T1219",
# #     "T1269",
# #     "H1265",
# #     "T1295",
# #     "T1246",
# # ]
# # removed_targets = [
# #     # "T1219",
# #     "T1246",
# #     # "T1269",
# #     "T1269v1o_",
# #     # "T1295",
# #     "T1295o.",
# #     # "T1249",
# #     # "H1265",
# #     "H1265_",
# #     "T2270o",
# # ]
# if stage == "1":
#     bad_targets.extend([
#         "T0",
#         "T2",
#         "H0",
#         "H2"
#     ])
# elif stage == "0":
#     bad_targets.extend([
#         "T1",
#         "T2",
#         "H1",
#         "H2"
#     ])
# elif stage == "2":
#     bad_targets.extend([
#         "T0",
#         "T1",
#         "H0",
#         "H1"
#     ])
# else:
#     print("Invalid stage")
#     sys.exit(1)

# to_remove = []
# for result_file in result_files:
#     for removed_target in bad_targets:
#         if result_file.startswith(removed_target):
#             to_remove.append(result_file)
#             break
# for remove in to_remove:
#     result_files.remove(remove)
# result_files = sorted(result_files)
# for feature in features:
#     get_interface_raw_score(results_dir, result_files, out_dir,
#                             feature, model, mode, impute_value)
