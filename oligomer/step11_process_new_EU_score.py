import argparse
import sys
import os
import pandas as pd
import numpy as np


def group_by_target(results_dir, result_files, out_dir, feature, model, mode, impute_value):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data = pd.DataFrame()
    data_raw = pd.DataFrame()
    print("Processing {}".format(feature))
    for result_file in result_files:
        print("Processing {}".format(result_file))
        result_path = results_dir + result_file
        data_tmp = pd.read_csv(result_path, sep="\t", index_col=0)
        data_tmp = data_tmp.replace("-", np.nan)
        data_tmp[feature] = data_tmp[feature].astype(float)

        data_tmp = pd.DataFrame(data_tmp[feature])
        data_tmp.index = data_tmp.index.str.extract(
            r'(\w+)TS(\w+)_(\w+)').apply(lambda x: (f"{x[0]}", f"TS{x[1]}", x[2][0]), axis=1)
        data_tmp.index = pd.MultiIndex.from_tuples(
            data_tmp.index, names=['target', 'group', 'submission_id'])
        if model == "best":
            data_tmp = data_tmp.loc[(slice(None), slice(None), [
                "1", "2", "3", "4", "5"]), :]
        elif model == "first":
            data_tmp = data_tmp.loc[(slice(None), slice(None),
                                    "1"), :]
        elif model == "sixth":
            data_tmp = data_tmp.loc[(slice(None), slice(None),
                                    "6"), :]
        grouped = data_tmp.groupby(["group"])
        grouped = pd.DataFrame(grouped[feature].max())
        grouped = grouped.sort_values(by=feature, ascending=False)
        initial_z = (grouped - grouped.mean()) / grouped.std()
        new_z_score = pd.DataFrame(
            index=grouped.index, columns=grouped.columns)
        filtered_data = grouped[feature][initial_z[feature] >= -2]
        new_mean = filtered_data.mean(skipna=True)
        new_std = filtered_data.std(skipna=True)
        new_z_score[feature] = (grouped[feature] - new_mean) / new_std
        new_z_score = new_z_score.fillna(impute_value)
        new_z_score = new_z_score.where(
            new_z_score > impute_value, impute_value)
        new_z_score = new_z_score.rename(
            columns={feature: result_file.split(".")[0]})
        data = pd.concat([data, new_z_score], axis=1)
        grouped = grouped.rename(
            columns={feature: result_file.split(".")[0]})
        data_raw = pd.concat([data_raw, grouped], axis=1)

    data = data.fillna(impute_value)
    data = data.reindex(sorted(data.columns), axis=1)
    data = data.sort_index()
    data_file = f"{feature}-{model}-{mode}-impute={impute_value}.csv"
    data.to_csv(out_dir + data_file)

    data["sum"] = data.sum(axis=1)
    data = data.sort_values(by="sum", ascending=False)
    sum_data_file = f"{feature}-{model}-{mode}-impute={impute_value}_sum.csv"
    data.to_csv(out_dir + sum_data_file)

    data_raw = data_raw.reindex(sorted(data_raw.columns), axis=1)
    data_raw = data_raw.sort_index()
    data_raw_file = f"{feature}-{model}-{mode}-impute={impute_value}_raw.csv"
    data_raw.to_csv(out_dir + data_raw_file)


parser = argparse.ArgumentParser(
    description="options for data processing")
parser.add_argument("--measures", type=list,
                    default=["qs_global", "qs_best", "ics", "ips", "dockq_ave", "tm_score", "lddt"])
parser.add_argument("--results_dir", type=str,
                    default="/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/model_results/")
parser.add_argument("--out_dir", type=str,
                    default="./group_by_target_EU_new/")
parser.add_argument("--model", type=str, default="best")
parser.add_argument("--mode", type=str, default="all")
parser.add_argument("--impute_value", type=int, default=-2)
parser.add_argument("--stage", type=str, default="1")
parser.add_argument("--bad_targets", nargs="+", default=[
    "T1246",
    "T1249",
    "T1269v1o_",
    "T1295o.",
    "H1265_",
    "T2270o",
])

args = parser.parse_args()
results_dir = args.results_dir
out_dir = args.out_dir
model = args.model
mode = args.mode
impute_value = args.impute_value
measures = args.measures
stage = args.stage
bad_targets = args.bad_targets

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
else:
    print("Invalid stage")
    sys.exit(1)

result_files = [result for result in os.listdir(
    results_dir) if result.endswith(".results") and not result.endswith(".nr_interface.results")]
to_remove = []
for result_file in result_files:
    for removed_target in bad_targets:
        if result_file.startswith(removed_target):
            to_remove.append(result_file)
            break
for remove in to_remove:
    result_files.remove(remove)
result_files = sorted(result_files)
# print(result_files)
# print(len(result_files))
for measure in measures:
    group_by_target(results_dir, result_files, out_dir,
                    measure, model, mode, impute_value)
