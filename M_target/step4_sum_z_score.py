import argparse
import pandas as pd
import numpy as np
import os


def get_group_by_target(csv_list, csv_path, out_path, sum_path,
                        feature, model, mode, impute_value=-2):
    inverse_columns = ["RMS_CA", "RMS_ALL", "err",
                       "RMSD[L]", "MolPrb_Score", "FlexE", "MP_clash", "MP_rotout", "MP_ramout", "RMSD"]
    if not os.path.exists(out_path + f"impute={impute_value}/"):
        os.makedirs(out_path + f"impute={impute_value}/")
    if not os.path.exists(out_path + "raw/"):
        os.makedirs(out_path + "raw/")
    if not os.path.exists(sum_path + f"impute={impute_value}/"):
        os.makedirs(sum_path + f"impute={impute_value}/")

    data = pd.DataFrame()
    data_raw = pd.DataFrame()
    for csv_file in csv_list:
        print(f"Processing {csv_file}")
        data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
        data_tmp = pd.DataFrame(data_tmp[feature])
        # if there is "-" in the value, replace it with 0
        data_tmp = data_tmp.replace("-", float(0))
        data_tmp = data_tmp.astype(float)
        if feature in inverse_columns:
            data_tmp[feature] = -data_tmp[feature]
        data_tmp.index = data_tmp.index.str.extract(
            r'(M\w+)TS(\w+)_(\w+)-(D\w+)').apply(lambda x: (f"{x[0]}-{x[3]}", f"TS{x[1]}", x[2]), axis=1)
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
        grouped[feature] = grouped[feature].astype(float)
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
            columns={feature: csv_file.split(".")[0]})
        data = pd.concat([data, new_z_score], axis=1)

        grouped = grouped.rename(columns={feature: csv_file.split(".")[0]})
        data_raw = pd.concat([data_raw, grouped], axis=1)
    # impute data again with impute_value, because some group may not submit all targets
    data = data.fillna(impute_value)

    # nan_values = data.isnull().sum().sum()
    # print(f"Number of nan values: {nan_values}")

    data = data.reindex(sorted(data.columns), axis=1)
    data = data.sort_index()
    data_columns = data.columns
    data.to_csv(
        out_path + f"impute={impute_value}/" + f"group_by_target-{feature}-{model}-{mode}.csv")
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
    # actually, the EU_weight here is not useful at all because it is all 1
    data["sum"] = data.sum(axis=1)
    data = data.sort_values(by="sum", ascending=False)
    data.to_csv(
        sum_path + f"impute={impute_value}/" + f"sum_unweighted_{feature}-{model}-{mode}.csv")

    data.drop(columns=["sum"], inplace=True)
    data = data * pd.Series(EU_weight)
    data["sum"] = data.sum(axis=1)
    data = data.sort_values(by="sum", ascending=False)
    data.to_csv(
        sum_path + f"impute={impute_value}/" + f"sum_{feature}-{model}-{mode}.csv")
    # sort columns by alphabetical order
    data_raw = data_raw.reindex(sorted(data_raw.columns), axis=1)
    # sort rows by alphabetical order
    data_raw = data_raw.sort_index()
    data_raw_csv = f"groups_by_targets_for-raw-{feature}-{model}-{mode}.csv"
    data_raw.to_csv
    data_raw.to_csv(
        out_path + "raw/" + data_raw_csv)


parser = argparse.ArgumentParser(description="options for sum z-score")

parser.add_argument('--csv_path', type=str,
                    default="./hybrid_data_Oct_11/raw_data/")
parser.add_argument('--sum_path', type=str, default="./sum_EU/")
parser.add_argument('--out_path', type=str, default="./group_by_target_EU/")
parser.add_argument('--model', type=str, help='first or best', default='best')
parser.add_argument('--mode', type=str,
                    help='easy, medium, hard or all', default='all')
parser.add_argument('--impute_value', type=int, default=-2)
parser.add_argument('--stage', type=str, default="1")

args = parser.parse_args()
csv_path = args.csv_path
sum_path = args.sum_path
out_path = args.out_path
model = args.model
mode = args.mode
impute_value = args.impute_value
stage = args.stage
if stage == "1":
    csv_list = [txt for txt in os.listdir(
        csv_path) if txt.endswith(".csv") and txt.startswith("M1")]
elif stage == "0":
    csv_list = [txt for txt in os.listdir(
        csv_path) if txt.endswith(".csv") and txt.startswith("M0")]
elif stage == "2":
    csv_list = [txt for txt in os.listdir(
        csv_path) if txt.endswith(".csv") and txt.startswith("M2")]
# csv_list = [txt for txt in os.listdir(
#     csv_path) if txt.endswith(".csv") and txt.startswith("M1")]
csv_list = sorted(csv_list)

bad_targets = ["M1268", "M1297"]
remove = []
for file in csv_list:
    for target in bad_targets:
        if file.startswith(target):
            remove.append(file)
for target in remove:
    csv_list.remove(target)

for csv in csv_list:
    print(csv)

if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(sum_path):
    os.makedirs(sum_path)
features = ["ICS(F1)", "IPS", "QSglob", "QSbest", "lDDT",
            "GDT_TS", "RMSD", "TMscore", "GlobDockQ", "BestDockQ"]
for feature in features:
    get_group_by_target(csv_list, csv_path, out_path, sum_path,
                        feature, model, mode, impute_value=impute_value)
    print("Finished processing {}".format(feature))
