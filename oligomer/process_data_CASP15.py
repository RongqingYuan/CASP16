import os
import numpy as np
import pandas as pd
import argparse

oligomer_path = "/data/data1/conglab/qcong/CASP16/oligomers/"
oligomer_path = "/home2/s439906/data/CASP16/oligomers_Sep_8/"
oligomer_path = "/home2/s439906/data/CASP16/oligomers_Sep_17/"
oligomer_path = "/home2/s439906/data/CASP16/oligomers_Sep_17_merge_v/"
oligomer_path = "/home2/s439906/data/CASP16/oligo/"
oligomer_out_raw_path = "./oligomer_data_CASP15/raw_data/"
oligomer_out_path = "./oligomer_data_CASP15/processed/"

parser = argparse.ArgumentParser(
    description="options for data processing")
parser.add_argument("--oligomer_path", type=str,
                    default="/home2/s439906/data/CASP16/oligo/")
parser.add_argument("--oligomer_output_path", type=str,
                    default="./oligomer_data_CASP15/")

args = parser.parse_args()
oligomer_path = args.oligomer_path
oligomer_output_path = args.oligomer_output_path

oligomer_out_path = oligomer_output_path + "/processed/"
oligomer_out_raw_path = oligomer_output_path + "/raw_data/"
oligomer_list = [txt for txt in os.listdir(
    oligomer_path) if txt.endswith(".txt")]
if not os.path.exists(oligomer_out_path):
    os.makedirs(oligomer_out_path)
if not os.path.exists(oligomer_out_raw_path):
    os.makedirs(oligomer_out_raw_path)
all_data = {}

# read the oligomer list
# this is too messy, we need to completely re-write this part compared to the monomer part
for oligomer in oligomer_list:
    oligomer_file = oligomer_path + oligomer
    data = []
    feature_number = 0
    with open(oligomer_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                feature_number = len(line.split())
                break

    with open(oligomer_file, "r") as f:
        flag = 0  # to indicate if we have gone through the header line
        for line in f:
            # tmp = line.split('[')
            # if len(tmp) == 1:
            #     tmp = tmp[0].split()
            #     if len(tmp) > 5 and flag == 1:
            #         # there is something wrong with the data, print it out
            #         print("Error: some data is missing for {}".format(oligomer))
            #     if len(tmp) > 5 and flag == 0:
            #         data.append(tmp)
            #         flag = 1

            #     continue
            # part_1 = tmp[0].split()
            # tmp = tmp[1]
            # tmp = tmp.split(']')
            # contact_score = tmp[0]
            # part_2 = tmp[1].split()[:5]  # the 5th is not used. a placeholder
            # contact_score = contact_score.split(',')
            # try:
            #     contact_score = [float(score) for score in contact_score]
            #     mean_score = str(np.mean(contact_score))
            # except ValueError:
            #     print("ValueError: {} for contact score. Probably due to missing value. Will assign 0".format(
            #         oligomer))
            #     mean_score = "0"  # probably the predictor didn't even get a complex
            # line = part_1 + [mean_score] + part_2
            # # if type(contact_score) == list:
            # #     line = part_1 + contact_score + part_2
            # # else:
            # #     line = part_1 + [contact_score] + part_2
            line = line.split()
            if len(line) < 4:
                continue
            elif len(line) >= feature_number:
                line = line[:feature_number]
            else:
                print(
                    "Error: the length of the data is not consistent for {}".format(oligomer))
                breakpoint()
            data.append(line)
    # print(data)
    # breakpoint()
    all_data[oligomer] = data
    # print(oligomer)
    length = len(data[0])
    for i in range(1, len(data)):
        if len(data[i]) != length:
            breakpoint()
            print(
                "Error: the length of the data is not consistent for {}".format(oligomer))

    # convert the data to dataframe, the first row is the column names, the first column is the index
    data = pd.DataFrame(data)
    # set the first row as the column names
    data.columns = data.iloc[0]
    # drop the first row
    data = data.drop(0)
    # set the "Model" column as the index
    data = data.set_index("Model")
    # this is good raw data
    data.replace("N/A", np.nan, inplace=True)
    data.replace("-", np.nan, inplace=True)
    data.to_csv(oligomer_out_raw_path + oligomer[:-4] + ".csv")

    data = data.drop(["#", "Gr.Code", "Stoich.", "Symm.", "Group"], axis=1)

    # print(data.shape)
    # print(data.head())

    # impute the N/A values with the mean value of the column
    # impute the - values with the mean value of the column
    # data = data.apply(pd.to_numeric)
    data.replace("N/A", np.nan, inplace=True)
    data.replace("-", np.nan, inplace=True)
    # data = data.drop(["QS_Interfaces"], axis=1)
    # data = data.drop(["SymmRMSD"], axis=1)
    try:
        data = data.drop(["SymmGr.RMSD"], axis=1)
    except KeyError:
        print("KeyError: SymmGr.RMSD not found for {}, will continue".format(oligomer))

    # convert the data type to float
    data = data.astype(float)
    # inverse_columns = ["SymmRMSD"]
    # data[inverse_columns] = -data[inverse_columns]
    initial_z = (data - data.mean()) / data.std()
    new_z_score = pd.DataFrame(index=data.index, columns=data.columns)
    for column in data.columns:
        filtered_data = data[column][initial_z[column] >= -2]
        new_mean = filtered_data.mean(skipna=True)
        new_std = filtered_data.std(skipna=True)
        new_z_score[column] = (data[column] - new_mean) / new_std
    new_z_score = new_z_score.fillna(-2.0)
    new_z_score = new_z_score.where(new_z_score > -2, -2.0)

    # data = (data - data.mean()) / data.std()
    # data = data[((data >= -2) | data.isna()).all(axis=1)]
    # data = (data - data.mean()) / data.std()
    # data = data.fillna(0.0)

    # try:
    #     data = data.astype(float)
    # except ValueError:
    #     print("ValueError: ", oligomer)
    #     sys.exit()

    # try:
    #     data = (data - data.mean()) / data.std()
    # except ValueError:
    #     print("ValueError: ", oligomer)
    #     sys.exit()
    # # normalize the data; the first column is the index so we don't normalize it
    # data.iloc[:, 1:] = (data.iloc[:, 1:] - data.iloc[:,
    #                     1:].mean()) / data.iloc[:, 1:].std()
    # print(data.head())

    # save the normalized data to csv file
    new_z_score.to_csv(oligomer_out_path + oligomer[:-4] + ".csv")


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
    # "ICS(F1)",
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
