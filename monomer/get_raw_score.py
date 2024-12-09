import argparse
import pandas as pd
import os


def get_group_by_target(csv_list, csv_path, out_path,
                        feature, model, mode, impute_value=-2):
    inverse_columns = ["RMS_CA", "RMS_ALL", "err",
                       "RMSD[L]", "MolPrb_Score", "FlexE", "MP_clash", "MP_rotout", "MP_ramout"]

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
        # initial_z = (grouped - grouped.mean()) / grouped.std()
        # new_z_score = pd.DataFrame(
        #     index=grouped.index, columns=grouped.columns)
        # filtered_data = grouped[feature][initial_z[feature] >= -2]
        # new_mean = filtered_data.mean(skipna=True)
        # new_std = filtered_data.std(skipna=True)
        # new_z_score[feature] = (grouped[feature] - new_mean) / new_std
        # new_z_score = new_z_score.fillna(impute_value)
        # new_z_score = new_z_score.where(
        #     new_z_score > impute_value, impute_value)
        # new_z_score = new_z_score.rename(
        #     columns={feature: csv_file.split(".")[0]})
        # data = pd.concat([data, new_z_score], axis=1)

        grouped = grouped.rename(columns={feature: csv_file.split(".")[0]})
        data_raw = pd.concat([data_raw, grouped], axis=1)

    data_raw = data_raw.reindex(sorted(data_raw.columns), axis=1)
    data_raw = data_raw.sort_index()
    data_raw_csv = f"{feature}-{model}-{mode}-raw.csv"
    data_raw.to_csv(out_path + data_raw_csv)


parser = argparse.ArgumentParser(description="options for sum z-score")
parser.add_argument('--csv_path', type=str,
                    default="./monomer_data_newest/raw_data/")
parser.add_argument('--out_path', type=str, default="./score_T1/")
parser.add_argument('--model', type=str,
                    help='first, best or sixth', default='best')
parser.add_argument('--mode', type=str,
                    help='easy, medium, hard or all', default='all')
parser.add_argument('--phase', type=str, default='1')
parser.add_argument('--impute_value', type=int, default=-2)

args = parser.parse_args()
csv_path = args.csv_path
out_path = args.out_path
model = args.model
mode = args.mode
phase = args.phase
impute_value = args.impute_value


if phase == "0,1,2":
    csv_list = [txt for txt in os.listdir(
        csv_path) if txt.endswith(".csv") and txt.startswith("T")]
elif phase == "0":
    csv_list = [txt for txt in os.listdir(
        csv_path) if txt.endswith(".csv") and txt.startswith("T0")]
elif phase == "1":
    csv_list = [txt for txt in os.listdir(
        csv_path) if txt.endswith(".csv") and txt.startswith("T1")]
elif phase == "2":
    csv_list = [txt for txt in os.listdir(
        csv_path) if txt.endswith(".csv") and txt.startswith("T2")]
csv_list = sorted(csv_list)


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
            'reLLG_const'
            ]
for feature in features:
    get_group_by_target(csv_list, csv_path, out_path,
                        feature, model, mode, impute_value=impute_value)
    print("Finished processing {}".format(feature))
