import argparse
import sys
import os
import pandas as pd
import numpy as np


def group_by_target(results_dir, v1_file, v2_file, out_dir, feature, model, mode, stage, impute_value=-2):
    print("Processing {}".format(feature))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # data = pd.DataFrame()
    # data_raw = pd.DataFrame()
    v1_path = results_dir + v1_file
    # it is actually a tsv file, us pd.read_csv to read it
    data_v1 = pd.read_csv(v1_path, sep="\t", index_col=0)
    data_v1 = data_v1.replace("-", np.nan)
    data_v1[feature] = data_v1[feature].astype(float)
    # fill "-" with nan
    data_v1 = pd.DataFrame(data_v1[feature])
    data_v1.index = data_v1.index.str.extract(
        r'(\w+)TS(\w+)_(\w+)').apply(lambda x: (f"{x[0]}", f"TS{x[1]}", x[2][0]), axis=1)
    data_v1.index = pd.MultiIndex.from_tuples(
        data_v1.index, names=['target', 'group', 'submission_id'])
    if model == "best":
        data_v1 = data_v1.loc[(slice(None), slice(None), [
            "1", "2", "3", "4", "5"]), :]
    elif model == "first":
        data_v1 = data_v1.loc[(slice(None), slice(None),
                               "1"), :]
    data_v1_grouped = data_v1.groupby(["target", "group"])
    data_v1_grouped = pd.DataFrame(data_v1_grouped[feature].max())
    print(data_v1_grouped)
    # get target = T1249v1 rows
    if stage == "1":
        target_v1_model_v1 = data_v1_grouped.loc["T1249v1", :]
    elif stage == "2":
        target_v1_model_v1 = data_v1_grouped.loc["T2249v1", :]
    # rename column to v1_file
    target_v1_model_v1 = target_v1_model_v1.rename(
        columns={feature: f"{feature}_v1"})
    print(target_v1_model_v1)
    if stage == "1":
        target_v1_model_v2 = data_v1_grouped.loc["T1249v2", :]
    elif stage == "2":
        target_v1_model_v2 = data_v1_grouped.loc["T2249v2", :]

    target_v1_model_v2 = target_v1_model_v2.rename(
        columns={feature: f"{feature}_v1"})

    v2_path = results_dir + v2_file
    # it is actually a tsv file, us pd.read_csv to read it
    data_v2 = pd.read_csv(v2_path, sep="\t", index_col=0)
    data_v2 = data_v2.replace("-", np.nan)
    data_v2[feature] = data_v2[feature].astype(float)
    # fill "-" with nan
    data_v2 = pd.DataFrame(data_v2[feature])
    data_v2.index = data_v2.index.str.extract(
        r'(\w+)TS(\w+)_(\w+)').apply(lambda x: (f"{x[0]}", f"TS{x[1]}", x[2][0]), axis=1)
    data_v2.index = pd.MultiIndex.from_tuples(
        data_v2.index, names=['target', 'group', 'submission_id'])
    if model == "best":
        data_v2 = data_v2.loc[(slice(None), slice(None), [
            "1", "2", "3", "4", "5"]), :]
    elif model == "first":
        data_v2 = data_v2.loc[(slice(None), slice(None),
                               "1"), :]
    data_v2_grouped = data_v2.groupby(["target", "group"])
    data_v2_grouped = pd.DataFrame(data_v2_grouped[feature].max())
    # get target = T1249v1 rows
    if stage == "1":
        target_v2_model_v1 = data_v2_grouped.loc["T1249v1", :]
    elif stage == "2":
        target_v2_model_v1 = data_v2_grouped.loc["T2249v1", :]
    # rename column to v1_file
    target_v2_model_v1 = target_v2_model_v1.rename(
        columns={feature: f"{feature}_v2"})
    if stage == "1":
        target_v2_model_v2 = data_v2_grouped.loc["T1249v2", :]
    elif stage == "2":
        target_v2_model_v2 = data_v2_grouped.loc["T2249v2", :]
    target_v2_model_v2 = target_v2_model_v2.rename(
        columns={feature: f"{feature}_v2"})

    case_1 = pd.concat([target_v1_model_v1, target_v2_model_v2], axis=1)
    case_2 = pd.concat([target_v1_model_v2, target_v2_model_v1], axis=1)
    # this is a combinatorial problem, we need to choose the best one with restrictions
    # fortunately there are only 2 combinations...
    case_1["sum"] = case_1.sum(axis=1)
    case_2["sum"] = case_2.sum(axis=1)

    # choose the best one for each group
    best_df = pd.DataFrame()
    for group in case_1.index:
        if case_1.loc[group, "sum"] > case_2.loc[group, "sum"]:
            best_df = pd.concat([best_df, case_1.loc[group, :].to_frame().T])
        else:
            best_df = pd.concat([best_df, case_2.loc[group, :].to_frame().T])

    best_df = best_df.drop(columns=["sum"])
    best_df = best_df.rename(columns={f"{feature}_v1": v1_file.split(".")[
                             0], f"{feature}_v2": v2_file.split(".")[0]})
    best_df.to_csv(out_dir +
                   "group_by_target-raw-T1249o-{}-{}-{}-impute_value={}.csv".format(feature, model, mode, impute_value))
    print(best_df)
    for target in best_df.columns:
        data = best_df[target]
        data = pd.DataFrame(data)
        initial_z = (data - data.mean()) / data.std()
        new_z_score = pd.DataFrame(
            index=data.index, columns=data.columns)
        filtered_data = data[initial_z >= -2]
        new_mean = filtered_data.mean(skipna=True)
        new_std = filtered_data.std(skipna=True)
        new_z_score[target] = (data - new_mean) / new_std
        new_z_score = new_z_score.fillna(impute_value)
        new_z_score = new_z_score.where(
            new_z_score > impute_value, impute_value)
        best_df[target] = new_z_score
    best_df.to_csv(out_dir +
                   "group_by_target-T1249o-{}-{}-{}-impute_value={}.csv".format(feature, model, mode, impute_value))

    print(best_df)

    template_raw_file = out_dir + \
        f"{feature}-{model}-{mode}-impute={impute_value}_raw.csv"
    template_EU_file = out_dir + \
        f"{feature}-{model}-{mode}-impute={impute_value}.csv"
    template_raw = pd.read_csv(template_raw_file, index_col=0)
    template_EU = pd.read_csv(template_EU_file, index_col=0)
    print(template_raw)
    print(template_EU)
    template_raw = pd.concat([template_raw, best_df], axis=1)
    template_EU = pd.concat([template_EU, best_df], axis=1)
    # impute template_EU with impute_value
    template_EU = template_EU.fillna(impute_value)
    # sort row and column alphabetically
    template_EU = template_EU.reindex(sorted(template_EU.columns), axis=1)
    template_EU = template_EU.sort_index()
    template_raw = template_raw.reindex(sorted(template_raw.columns), axis=1)
    template_raw = template_raw.sort_index()
    # drop TS314 row in the dataframe
    # this is a special case for T1249o
    template_raw = template_raw.drop(index='TS314')
    template_EU = template_EU.drop(index='TS314')
    template_raw.to_csv(template_raw_file)
    template_EU.to_csv(template_EU_file)

    # breakpoint()
    # filtered_data = best_df[feature][initial_z[feature] >= -2]
    # new_mean = filtered_data.mean(skipna=True)
    # new_std = filtered_data.std(skipna=True)
    # new_z_score[feature] = (best_df[feature] - new_mean) / new_std
    # new_z_score = new_z_score.fillna(-2.0)
    # new_z_score = new_z_score.where(new_z_score > -2, -2)
    # new_z_score = new_z_score.rename(
    #     columns={feature:     v1_file.split(".")[0]})
    # best_df = best_df.rename(
    #     columns={feature:     v1_file.split(".")[0]})
    # breakpoint()
    # grouped = grouped.sort_values(by=feature, ascending=False)
    # initial_z = (grouped - grouped.mean()) / grouped.std()
    # new_z_score = pd.DataFrame(
    #     index=grouped.index, columns=grouped.columns)
    # filtered_data = grouped[feature][initial_z[feature] >= -2]
    # new_mean = filtered_data.mean(skipna=True)
    # new_std = filtered_data.std(skipna=True)
    # new_z_score[feature] = (grouped[feature] - new_mean) / new_std
    # new_z_score = new_z_score.fillna(-2.0)
    # new_z_score = new_z_score.where(new_z_score > -2, -2)
    # new_z_score = new_z_score.rename(
    #     columns={feature:     v1_file.split(".")[0]})
    # data = pd.concat([data, new_z_score], axis=1)
    # grouped = grouped.rename(
    #     columns={feature:     v1_file.split(".")[0]})
    # data_raw = pd.concat([data_raw, grouped], axis=1)

    # data = data.fillna(-2.0)
    # data.to_csv(out_dir +
    #             "group_by_target-{}-{}-{}.csv".format(feature, model, mode))

    # data_raw.to_csv(out_dir +
    #                 "group_by_target_raw-{}-{}-{}.csv".format(feature, model, mode))

    # data["sum"] = data.sum(axis=1)
    # data = data.sort_values(by="sum", ascending=False)
    # data.to_csv(out_dir + "sum_{}-{}-{}.csv".format(feature, model, mode))


results_dir = "/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/model_results/"
v1_file, v2_file = 'T1249v1o.results', 'T1249v2o.results'
out_dir = "./score_EU_v2/"
model = "best"
mode = "all"
impute_value = -2
stage = "1"

parser = argparse.ArgumentParser(description="options for sum z-score")
parser.add_argument("--measures",  nargs='+',
                    default=["tm_score", "lddt", "qs_global",
                             "qs_best", "ics", "ips", "dockq_ave"]
                    )
parser.add_argument("--results_dir", type=str,
                    help="path to the results directory",
                    default="/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/model_results/")
parser.add_argument("--v1_file", type=str,
                    help="v1 file name", default="T1249v1o.results")
parser.add_argument("--v2_file", type=str,
                    help="v2 file name", default="T1249v2o.results")
parser.add_argument("--out_dir", type=str,
                    help="output directory", default="./group_by_target_EU_new/")
parser.add_argument("--model", type=str,
                    help="model to use", default="best")
parser.add_argument("--mode", type=str,
                    help="mode to use", default="all")
parser.add_argument("--impute_value", type=int,
                    help="impute value", default=-2)
parser.add_argument("--stage", type=str, default="1")

args = parser.parse_args()
results_dir = args.results_dir
v1_file = args.v1_file
v2_file = args.v2_file
out_dir = args.out_dir
model = args.model
mode = args.mode
impute_value = args.impute_value
stage = args.stage

if stage == "1":
    ...
elif stage == "2":
    v1_file = "T2249v1o.results"
    v2_file = "T2249v2o.results"

for measure in args.measures:
    group_by_target(results_dir, v1_file, v2_file,
                    out_dir, measure, model, mode, stage, impute_value=impute_value)

# group_by_target(results_dir, v1_file, v2_file,
#                 out_dir, "tm_score", model, mode, impute_value=impute_value)
# group_by_target(results_dir, v1_file, v2_file,
#                 out_dir, "lddt", model, mode, impute_value=impute_value)
# group_by_target(results_dir, v1_file, v2_file,
#                 out_dir, "qs_global", model, mode, impute_value=impute_value)
# group_by_target(results_dir, v1_file, v2_file,
#                 out_dir, "qs_best", model, mode, impute_value=impute_value)
# group_by_target(results_dir, v1_file, v2_file,
#                 out_dir, "ics", model, mode, impute_value=impute_value)
# group_by_target(results_dir, v1_file, v2_file,
#                 out_dir, "ips", model, mode, impute_value=impute_value)
# group_by_target(results_dir, v1_file, v2_file,
#                 out_dir, "dockq_ave", model, mode, impute_value=impute_value)
