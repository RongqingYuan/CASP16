import os
import sys
import pandas as pd
import numpy as np


def group_by_target(results_dir, v1_file, v2_file, out_dir,
                    feature, model, mode, impute_value):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data = pd.DataFrame()
    data_weighted = pd.DataFrame()
    data_raw = pd.DataFrame()
    target_v1_model_v1 = None
    target_v1_model_v2 = None
    target_v2_model_v1 = None
    target_v2_model_v2 = None
    template_raw_file = out_dir + \
        f"{feature}-{model}-{mode}-impute={impute_value}.csv"
    template_EU_file = out_dir + \
        f"{feature}-{model}-{mode}-impute={impute_value}_weighted_EU.csv"
    for file in [v1_file, v2_file]:
        # the nomenclature is bad here, ignore it because it is just a small script.
        v1_path = results_dir + file
        v1_data = pd.read_csv(v1_path, sep="\t", index_col=0)
        ############################
        nr_interf_group_in_ref = v1_data["nr_interf_group_in_ref"]
        if nr_interf_group_in_ref.nunique() == 1:
            pass
        else:
            print("Not all values are the same in nr_interf_group_in_ref")
        nr_refinterf_num = v1_data["nr_refinterf_#"]
        if nr_refinterf_num.nunique() == 1:
            nr_refinterf_num = nr_refinterf_num.astype(str)
            pass
        else:
            print("Not all values are the same in nr_refinterf_#")
        nr_interf_group_interfsize_in_ref = v1_data["nr_interf_group_interfsize_in_ref"]
        if nr_interf_group_interfsize_in_ref.nunique() == 1:
            nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.astype(
                str)
            pass
        else:
            print("Not all values are the same in nr_interf_group_interfsize_in_ref")
        ############################
        v1_data = pd.DataFrame(v1_data[feature]).astype(str)
        v1_data_split = v1_data[feature].str.split(';', expand=True)
        v1_data_split.columns = [
            f'interface_{i+1}' for i in range(v1_data_split.shape[1])]
        # fill "-" with nan
        v1_data_split = v1_data_split.replace("-", np.nan)
        v1_data_split = v1_data_split.astype(float)
        v1_data_split.index = v1_data_split.index.str.extract(
            r'(\w+)TS(\w+)_(\w+)').apply(lambda x: (f"{x[0]}", f"TS{x[1]}", x[2][0]), axis=1)
        v1_data_split.index = pd.MultiIndex.from_tuples(
            v1_data_split.index, names=['target', 'group', 'submission_id'])
        if model == "best":
            v1_data_split = v1_data_split.loc[(slice(None), slice(None), [
                "1", "2", "3", "4", "5"]), :]
        elif model == "first":
            v1_data_split = v1_data_split.loc[(slice(None), slice(None),
                                               "1"), :]
        elif model == "sixth":
            v1_data_split = v1_data_split.loc[(slice(None), slice(None),
                                               "6"), :]

        if file == v1_file:
            for i in range(v1_data_split.shape[1]):
                grouped = v1_data_split.groupby(["target", "group"])
                feature_name = f"interface_{i+1}"
                grouped = pd.DataFrame(grouped[feature_name].max())
                print(grouped)
                target_v1_model_v1 = grouped.loc["T1249v1", :]
                # add _v1 to the column name of target_v1_model_v1
                target_v1_model_v1 = target_v1_model_v1.rename(
                    columns={feature_name: f"{v1_file.split('.')[0]}_{feature_name}"})
                print(target_v1_model_v1)
                target_v1_model_v2 = grouped.loc["T1249v2", :]
                # add _v2 to the column name of target_v1_model_v2
                target_v1_model_v2 = target_v1_model_v2.rename(
                    columns={feature_name: f"{v2_file.split('.')[0]}_{feature_name}"})
                print(target_v1_model_v2)
        elif file == v2_file:
            for i in range(v1_data_split.shape[1]):
                grouped = v1_data_split.groupby(["target", "group"])
                feature_name = f"interface_{i+1}"
                grouped = pd.DataFrame(grouped[feature_name].max())
                target_v2_model_v1 = grouped.loc["T1249v1", :]
                # add _v1 to the column name of target_v2_model_v1
                target_v2_model_v1 = target_v2_model_v1.rename(
                    columns={feature_name: f"{v1_file.split('.')[0]}_{feature_name}"})
                print(target_v2_model_v1)
                target_v2_model_v2 = grouped.loc["T1249v2", :]
                # add _v2 to the column name of target_v2_model_v2
                target_v2_model_v2 = target_v2_model_v2.rename(
                    columns={feature_name: f"{v2_file.split('.')[0]}_{feature_name}"})
                print(target_v2_model_v2)

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
    # sort the best_df by alphabetical order
    best_df = best_df.reindex(sorted(best_df.columns), axis=1)
    best_df = best_df.sort_index()
    template_raw_df = pd.read_csv(template_raw_file, index_col=0)
    print(best_df)
    print(template_raw_df)
    template_raw_df = pd.concat([template_raw_df, best_df], axis=1)
    # # sort the template_raw_df by alphabetical order
    # template_raw_df = template_raw_df.reindex(sorted(template_raw_df.columns), axis=1)
    # template_raw_df = template_raw_df.sort_index()
    # template_raw_df.to_csv(template_raw_file)

    # grouped = grouped.sort_values(by=feature_name, ascending=False)
    # initial_z = (grouped - grouped.mean()) / grouped.std()
    # new_z_score = pd.DataFrame(
    #     index=grouped.index, columns=grouped.columns)
    # filtered_data = grouped[feature_name][initial_z[feature_name] >= -2]
    # new_mean = filtered_data.mean(skipna=True)
    # new_std = filtered_data.std(skipna=True)
    # new_z_score[feature_name] = (
    #     grouped[feature_name] - new_mean) / new_std
    # new_z_score = new_z_score.fillna(impute_value)
    # new_z_score = new_z_score.where(
    #     new_z_score > impute_value, impute_value)
    # new_z_score = new_z_score.rename(
    #     columns={feature_name: v1_file.split(".")[0]+"_"+feature_name})
    # data = pd.concat([data, new_z_score], axis=1)
    # grouped = grouped.rename(
    #     columns={feature_name: v1_file.split(".")[0]+"_"+feature_name})
    # data_raw = pd.concat([data_raw, grouped], axis=1)
    template_EU_df = pd.read_csv(template_EU_file, index_col=0)
    # drop sum column
    template_EU_df = template_EU_df.drop(columns=["sum"])
    for result_file in [v1_file, v2_file]:
        print("Processing {}".format(result_file))
        result_path = results_dir + result_file
        data_tmp = pd.read_csv(result_path, sep="\t", index_col=0)

        ############################
        nr_interf_group_in_ref = data_tmp["nr_interf_group_in_ref"]
        if nr_interf_group_in_ref.nunique() == 1:
            pass
        else:
            print("Not all values are the same in nr_interf_group_in_ref")
        nr_refinterf_num = data_tmp["nr_refinterf_#"]
        if nr_refinterf_num.nunique() == 1:
            nr_refinterf_num = nr_refinterf_num.astype(str)
            pass
        else:
            print("Not all values are the same in nr_refinterf_#")
        nr_interf_group_interfsize_in_ref = data_tmp["nr_interf_group_interfsize_in_ref"]
        if nr_interf_group_interfsize_in_ref.nunique() == 1:
            nr_interf_group_interfsize_in_ref = nr_interf_group_interfsize_in_ref.astype(
                str)
            pass
        else:
            print("Not all values are the same in nr_interf_group_interfsize_in_ref")
        ############################

        data_tmp = pd.DataFrame(data_tmp[feature]).astype(str)
        data_tmp_split = data_tmp[feature].str.split(';', expand=True)
        data_tmp_split.columns = [
            f'interface_{i+1}' for i in range(data_tmp_split.shape[1])]
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
            nr_interface_size_weights.append(size_weight/2)
        nr_interface_size_weights = np.log10(nr_interface_size_weights)
        nr_interface_size_weights = np.array(nr_interface_size_weights)
        # nr_interface_size_weights = nr_interface_size_weights / nr_interface_size_weights.sum()

        # elementwise multiplication
        interface_weight = nr_interface_weights * nr_interface_size_weights
        interface_weight = interface_weight / interface_weight.sum()
        EU_weight = data_tmp_split.shape[1] ** (1/3)

        target = result_file.split(".")[0]
        data_raw = pd.DataFrame(best_df[f"{target}_interface_1"])
        # there is only one interface in the best_df so we can just take the first one for simplicity
        # get a target_score df that has the same row as data_raw to make sure no nan values
        target_score = pd.DataFrame(index=template_EU_df.index)
        for i in range(data_raw.shape[1]):
            grouped = data_raw
            feature_name = f"{target}_interface_{i+1}"
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
        print(target_score)
        # take the weighted sum of the interface scores
        target_score = target_score * interface_weight
        data_weighted[result_file.split(".")[0]] = target_score.sum(axis=1)
        # then multiply by the EU_weight
        data_weighted[result_file.split(".")[0]] = data_weighted[result_file.split(
            ".")[0]] * EU_weight
    print(data_weighted)
    template_EU_df = pd.concat([template_EU_df, data_weighted], axis=1)
    print(template_EU_df)
    # remove rows with any nan values # group 314 only have this target,
    # and it will cause nan values in original data when concat
    template_raw_df = template_raw_df.dropna(axis=0, how='any')
    template_EU_df = template_EU_df.dropna(axis=0, how='any')
    # sort the template_raw_df by alphabetical order
    template_raw_df = template_raw_df.reindex(
        sorted(template_raw_df.columns), axis=1)
    template_raw_df = template_raw_df.sort_index()
    template_raw_df.to_csv(template_raw_file)

    # sort the template_EU_df by alphabetical order
    template_EU_df = template_EU_df.reindex(
        sorted(template_EU_df.columns), axis=1)
    template_EU_df = template_EU_df.sort_index()
    # sort the template_EU_df by alphabetical order
    template_EU_df["sum"] = template_EU_df.sum(axis=1)
    template_EU_df.to_csv(template_EU_file)
    template_EU_df.to_csv(
        out_dir + f"{feature}-{model}-{mode}-impute={impute_value}_sum.csv")
    # sys.exit(0)
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


results_dir = "/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_oligo_interfaces_20240917/nr_interfaces/"
v1_file, v2_file = 'T1249v1o.nr_interface.results', 'T1249v2o.nr_interface.results'
out_dir = "./group_by_target_per_interface/"
model = "best"
mode = "all"
impute_value = -2
group_by_target(results_dir, v1_file, v2_file, out_dir,
                "dockq_mean", model, mode, impute_value)
group_by_target(results_dir, v1_file, v2_file, out_dir,
                "qs_best_mean", model, mode, impute_value)
group_by_target(results_dir, v1_file, v2_file, out_dir,
                "qs_global_mean", model, mode, impute_value)
group_by_target(results_dir, v1_file, v2_file, out_dir,
                "ics_mean", model, mode, impute_value)
group_by_target(results_dir, v1_file, v2_file, out_dir,
                "ips_mean", model, mode, impute_value)
