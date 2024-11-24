import argparse
import pandas as pd
import os


def get_group_by_target(csv_list, csv_path, out_path,
                        feature, model, mode, impute_value=-2):
    inverse_columns = ["RMS_CA", "RMS_ALL", "err",
                       "RMSD[L]", "MolPrb_Score", "FlexE", "MP_clash", "MP_rotout", "MP_ramout"]

    # if not os.path.exists(out_path + f"impute={impute_value}/"):
    #     os.makedirs(out_path + f"impute={impute_value}/")
    # if not os.path.exists(out_path + "raw/"):
    #     os.makedirs(out_path + "raw/")
    # if not os.path.exists(sum_path + f"impute={impute_value}/"):
    #     os.makedirs(sum_path + f"impute={impute_value}/")

    data = pd.DataFrame()
    data_raw = pd.DataFrame()
    for csv_file in csv_list:
        print(f"Processing {csv_file}")
        data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
        data_tmp = pd.DataFrame(data_tmp[feature])
        # if there is "-" in the value, replace it with 0
        data_tmp = data_tmp.replace("-", float(0))

        if feature in inverse_columns:
            data_tmp[feature] = -data_tmp[feature]

        data_tmp.index = data_tmp.index.str.extract(
            r'(T\w+)TS(\w+)_(\w+)-(D\w+)').apply(lambda x: (f"{x[0]}-{x[3]}", f"TS{x[1]}", x[2]), axis=1)
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

    # # impute data again with impute_value, because some group may not submit all targets
    # data = data.fillna(impute_value)
    # # nan_values = data.isnull().sum().sum()
    # # print(f"Number of nan values: {nan_values}")

    # data = data.reindex(sorted(data.columns), axis=1)
    # data = data.sort_index()
    # data_csv = f"{feature}-{model}-{mode}-impute={impute_value}.csv"
    # data.to_csv(out_path + data_csv)

    # data_columns = data.columns
    # target_count = {}
    # for EU in data_columns:
    #     target = EU.split("-")[0]
    #     if target not in target_count:
    #         target_count[target] = 0
    #     target_count[target] += 1
    # # use the inverse of the target_count as the weight
    # target_weight = {key: 1/value for key, value in target_count.items()}
    # # assign EU_weight based on the target_weight
    # EU_weight = {EU: target_weight[EU.split("-")[0]]
    #              for EU in data_columns}

    # data["sum"] = data.sum(axis=1)
    # data = data.sort_values(by="sum", ascending=False)
    # data_sum_unweighted_csv = f"{feature}-{model}-{mode}-impute={impute_value}-sum_unweighted.csv"
    # data.to_csv(out_path + data_sum_unweighted_csv)

    # data.drop(columns=["sum"], inplace=True)
    # data = data * pd.Series(EU_weight)
    # data["sum"] = data.sum(axis=1)
    # data = data.sort_values(by="sum", ascending=False)
    # data_sum_csv = f"{feature}-{model}-{mode}-impute={impute_value}-sum.csv"
    # data.to_csv(out_path + data_sum_csv)

    data_raw = data_raw.reindex(sorted(data_raw.columns), axis=1)
    data_raw = data_raw.sort_index()
    data_raw_csv = f"{feature}-{model}-{mode}-raw.csv"
    data_raw.to_csv(out_path + data_raw_csv)


parser = argparse.ArgumentParser(description="options for sum z-score")

parser.add_argument('--csv_path', type=str,
                    default="./monomer_data_newest/raw_data/")
parser.add_argument('--out_path', type=str, default="./group_by_target_EU/")
parser.add_argument('--model', type=str,
                    help='first, best or sixth', default='best')
parser.add_argument('--mode', type=str,
                    help='easy, medium, hard or all', default='all')
parser.add_argument('--impute_value', type=int, default=-2)

args = parser.parse_args()
csv_path = args.csv_path
out_path = args.out_path
model = args.model
mode = args.mode
impute_value = args.impute_value


csv_list = [txt for txt in os.listdir(
    csv_path) if txt.endswith(".csv") and txt.startswith("T")]
csv_list = sorted(csv_list)


hard_group = [
    "T1207-D1",
    "T1210",
    "T1220s1",
    "T1226-D1",
    "T1228-D3-all",
    "T1271s1-D1",
]

medium_group = [
    "T1201",
    "T1212-D1",
    "T1218-D1",
    "T1218-D2",
    "T1227s1-D1",
    "T1228-D1-all",
    "T1228-D4-all",
    "T1230s1-D1",
    "T1237-D1",
    "T1239-D1-all",
    "T1239-D3-all",
    "T1243-D1",
    "T1244s1-D1",
    "T1245s2-D1",
    "T1249v1-D1",
    "T1257",
    "T1266-D1",
    "T1267s1-D1",
    "T1267s1-D2",
    "T1267s2-D1",
    "T1269-D1",
    "T1269-D2",
    "T1269-D3",
    "T1270-D1",
    "T1270-D2",
    "T1271s2-D1",
    "T1271s3-D1",
    "T1271s4-D1",
    "T1271s5-D1",
    "T1271s5-D2",
    "T1271s7-D1",
    "T1271s8-D1",
    "T1271s8-D2",
    "T1272s2-D1",
    "T1272s6-D1",
    "T1272s8-D1",
    "T1272s9-D1",
    "T1279-D2",
    "T1284-D1",
    "T1295-D1",
    "T1295-D3",
    "T1298-D1",
    "T1298-D2",
]

easy_group = [
    "T1206-D1",
    "T1208s1-D1",
    "T1208s2-D1",
    "T1218-D3",
    "T1228-D2-all",
    "T1231-D1",
    "T1234-D1",
    "T1235-D1",
    "T1239-D2-all",
    "T1239-D4-all",
    "T1240-D1",
    "T1240-D2",
    "T1245s1-D1",
    "T1246-D1",
    "T1259-D1",
    "T1271s6-D1",
    "T1274-D1",
    "T1276-D1",
    "T1278-D1",
    "T1279-D1",
    "T1280-D1",
    "T1292-D1",
    "T1294-D1-all",
    "T1295-D2",
    "T1299-D1",
]

if mode == "hard":
    csv_list = [csv for csv in csv_list if csv.split(
        ".")[0] in hard_group]
    print(len(csv_list))
elif mode == "medium":
    csv_list = [csv for csv in csv_list if csv.split(
        ".")[0] in medium_group]
    print(len(csv_list))
elif mode == "easy":
    csv_list = [csv for csv in csv_list if csv.split(
        ".")[0] in easy_group]
    print(len(csv_list))
elif mode == "all":
    pass

for csv in csv_list:
    print(csv)

if not os.path.exists(out_path):
    os.makedirs(out_path)
features = ['GDT_TS',
            'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
            'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
            'SphGr',
            'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
            # 'MP_rotout',
            # 'MP_ramout',
            # 'MP_ramfv',
            'reLLG_lddt',
            'reLLG_const']
for feature in features:
    get_group_by_target(csv_list, csv_path, out_path,
                        feature, model, mode, impute_value=impute_value)
    print("Finished processing {}".format(feature))
