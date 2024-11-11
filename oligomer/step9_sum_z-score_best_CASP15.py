import pandas as pd
import os


csv_path = "./monomer_data_aug_30/processed/EU/"
csv_path = "./monomer_data_Sep_10/processed/EU/"
csv_path = "./monomer_data_Sep_10/raw_data/EU/"
# csv_path = "./monomer_data_Sep_17/raw_data/"
csv_path = "./oligomer_data_CASP15/raw_data/"

csv_list = [txt for txt in os.listdir(
    csv_path) if txt.endswith(".csv") and (txt.startswith("T1") or txt.startswith("H1"))]

model = "first"
model = "best"

mode = "hard"
mode = "medium"
mode = "easy"
mode = "all"

# print(csv_list.__len__())
# breakpoint()
# read all data and concatenate them into one big dataframe
feature = "GDT_TS"
features = ['GDT_TS', 'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
            'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
            'SphGr',
            'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
            'MP_rotout', 'MP_ramout', 'MP_ramfv', 'reLLG_lddt', 'reLLG_const']

inverse_columns = ["RMS_CA", "RMS_ALL", "err",
                   "RMSD[L]", "MolPrb_Score", "FlexE", "MP_clash", "MP_rotout", "MP_ramout"]
features = [
    "QSglob",
    "QSbest",
    "ICS(F1)",
    # "lDDT",
    # "DockQ_Avg",
    # "IPS(JaccCoef)",
    # "TMscore",
]

group_by_target_path = "./group_by_target_EU_CASP15/"
sum_path = "./sum_CASP15/"
if not os.path.exists(group_by_target_path):
    os.makedirs(group_by_target_path)
if not os.path.exists(sum_path):
    os.makedirs(sum_path)


def get_group_by_target(csv_path, csv_list, feature, model, mode):
    data = pd.DataFrame()
    data_raw = pd.DataFrame()
    for csv_file in csv_list:
        data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
        data_tmp = pd.DataFrame(data_tmp[feature])
        # convert to float
        data_tmp[feature] = data_tmp[feature].astype(float)
        # breakpoint()
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
    data.to_csv("./group_by_target_EU_CASP15/" +
                "group_by_target-{}-{}-{}.csv".format(feature, model, mode))

    data_raw.to_csv("./group_by_target_EU_CASP15/" +
                    "group_by_target_raw-{}-{}-{}.csv".format(feature, model, mode))

    data["sum"] = data.sum(axis=1)
    data = data.sort_values(by="sum", ascending=False)
    data.to_csv("./sum_CASP15/" +
                "sum_{}-{}-{}.csv".format(feature, model, mode))


for feature in features:
    get_group_by_target(csv_path, csv_list, feature, model, mode)
    print("Finished processing {}".format(feature))
