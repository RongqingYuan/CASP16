import sys
import os
import pandas as pd
import numpy as np

results_dir = "/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/model_results/"
# result_files = [result for result in os.listdir(
#     results_dir) if result.endswith(".results")]
# result_files = [
#     result_file for result_file in result_files if "v" in result_file]

v1_file, v2_file = 'T1249v1o.results', 'T1249v2o.results'
out_dir = "./group_by_target_EU_new/"
model = "best"
mode = "all"


def group_by_target(results_dir, v1_file, v2_file, out_dir, feature, model, mode):
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
    # get target = T1249v1 rows
    target_v1_model_v1 = data_v1_grouped.loc["T1249v1", :]
    # rename column to v1_file
    target_v1_model_v1 = target_v1_model_v1.rename(
        columns={feature: f"{feature}_v1"})
    target_v1_model_v2 = data_v1_grouped.loc["T1249v2", :]
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
    target_v2_model_v1 = data_v2_grouped.loc["T1249v1", :]
    # rename column to v1_file
    target_v2_model_v1 = target_v2_model_v1.rename(
        columns={feature: f"{feature}_v2"})
    target_v2_model_v2 = data_v2_grouped.loc["T1249v2", :]
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
                   "group_by_target-raw-T1249o-{}-{}-{}.csv".format(feature, model, mode))
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
        new_z_score = new_z_score.fillna(-2.0)
        new_z_score = new_z_score.where(new_z_score > -2, -2)
        best_df[target] = new_z_score
    best_df.to_csv(out_dir +
                   "group_by_target-T1249o-{}-{}-{}.csv".format(feature, model, mode))

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


group_by_target(results_dir, v1_file, v2_file,
                out_dir, "tm_score", model, mode)
group_by_target(results_dir, v1_file, v2_file,
                out_dir, "lddt", model, mode)
sys.exit(0)
group_by_target(results_dir, result_files, out_dir, "qs_best", model, mode)
group_by_target(results_dir, result_files, out_dir, "ics", model, mode)
group_by_target(results_dir, result_files, out_dir, "ips", model, mode)
group_by_target(results_dir, result_files, out_dir, "dockq_ave", model, mode)


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
