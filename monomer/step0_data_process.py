import os
import numpy as np
import pandas as pd
import sys
import time

# monomer_path = "/data/data1/conglab/qcong/CASP16/monomers/"
monomer_path = "/home2/s439906/data/CASP16/monomers/"
monomer_list = [txt for txt in os.listdir(
    monomer_path) if txt.endswith(".txt")]

evaluation_unit = []
whole_structure = []

for file_name in monomer_list:
    t_code = file_name.split('-')[0]
    same_protein = [f for f in monomer_list if f.startswith(t_code)]

    if '-D' in file_name:
        if len(same_protein) == 1 and '-D1' in file_name:
            evaluation_unit.append(file_name)
            whole_structure.append(file_name)
            # both evaluation unit and whole structure
        else:
            evaluation_unit.append(file_name)
    else:
        whole_structure.append(file_name)

print("evaluation unit: ", evaluation_unit)
print("whole structure: ", whole_structure)
monomer_data_raw_EU_path = "./monomer_data_aug_30/raw_data/EU/"
monomer_data_raw_whole_path = "./monomer_data_aug_30/raw_data/whole/"
monomer_data_raw_all_path = "./monomer_data_aug_30/raw_data/all/"

monomer_data_EU_path = "./monomer_data_aug_30/processed/EU/"
monomer_data_whole_path = "./monomer_data_aug_30/processed/whole/"
monomer_data_all_path = "./monomer_data_aug_30/processed/all/"


if not os.path.exists(monomer_data_raw_EU_path):
    os.makedirs(monomer_data_raw_EU_path)
if not os.path.exists(monomer_data_raw_whole_path):
    os.makedirs(monomer_data_raw_whole_path)
if not os.path.exists(monomer_data_raw_all_path):
    os.makedirs(monomer_data_raw_all_path)
if not os.path.exists(monomer_data_EU_path):
    os.makedirs(monomer_data_EU_path)
if not os.path.exists(monomer_data_whole_path):
    os.makedirs(monomer_data_whole_path)
if not os.path.exists(monomer_data_all_path):
    os.makedirs(monomer_data_all_path)

csv_tmp_path = "./csv_tmp/"
if not os.path.exists(csv_tmp_path):
    os.makedirs(csv_tmp_path)

# read the monomer list
for monomer in monomer_list:
    monomer_file = monomer_path + monomer
    data = []
    with open(monomer_file, "r") as f:
        for line in f:
            line = line.split()
            if len(line) > 1:
                data.append(line)

    # convert the data to dataframe, the first row is the column names, the first column is the index
    data = pd.DataFrame(data)
    # set the first row as the column names
    data.columns = data.iloc[0]
    # drop the first row
    data = data.drop(0)
    # set the "Model" column as the index
    # if there is a header called MODEL, change it to Model
    if "MODEL" in data.columns:
        data = data.rename(columns={"MODEL": "Model"})
    data = data.set_index("Model")

    # save it as complete raw data, in case we need it later
    data.to_csv(monomer_data_raw_all_path + monomer[:-4] + ".csv")
    if monomer in evaluation_unit:
        data.to_csv(monomer_data_raw_EU_path + monomer[:-4] + ".csv")
    if monomer in whole_structure:
        data.to_csv(monomer_data_raw_whole_path + monomer[:-4] + ".csv")

    # remove the "GR#" column and "#" column
    data = data.drop(["GR#", "#"], axis=1)
    # "DipDiff", "BBscore", "SCscore" does not contribute to the prediction, so we remove them as well
    data = data.drop(["DipDiff", "BBscore", "SCscore"], axis=1)
    # remove the "RANK" column
    data = data.drop(["RANK"], axis=1)

    # # check the data shape
    # print(data.shape)
    # # print the first 5 rows
    # print(data.head())

    # fill the N/A values with nan
    # fill the "-" values with nan
    # data = data.apply(pd.to_numeric)
    data.replace("N/A", np.nan, inplace=True)
    data.replace("-", np.nan, inplace=True)
    # convert the data type to float
    data = data.astype(float)

    # some columns need to be taken inverse because in this case, smaller values are better, so we need to invert them for the sake of consistency
    inverse_columns = ["RMS_CA", "RMS_ALL", "err",
                       "RMSD[L]", "MolPrb_Score", "FlexE", "MP_clash", "MP_rotout", "MP_ramout"]  # MP_ramfv is not in the list. fv stands for favored and it is not in the list
    # take the negation of the columns
    data[inverse_columns] = -data[inverse_columns]
    # normalize the data with the z-score

    # data_tmp = (data - data.mean()) / data.std()
    initial_z = (data - data.mean()) / data.std()

    # remove z-score less than -2
    filtered_data = data[((initial_z >= -2) | initial_z.isna()).all(axis=1)]
    # filtered_data = data[initial_z >= -2]
    filtered_data.to_csv(csv_tmp_path + monomer[:-4] + ".csv")

    # remove outliers: i.e. if any value is smaller than -2, we directly remove the whole row
    # data = data[(data > -2).all(axis=1)]
    # filtered_data = data_tmp[((data_tmp >= -2) | data_tmp.isna()).all(axis=1)]
    # new_mean = filtered_data.mean(skipna=True)
    # new_std = filtered_data.std(skipna=True)
    new_mean = filtered_data.mean()
    new_std = filtered_data.std()
    print("new mean: ", new_mean)
    print("new std: ", new_std)
    # # save to csv_tmp path to see if anything goes wrong
    # data.to_csv(csv_tmp_path + monomer[:-4] + ".csv")

    # after removing the outliers, we need to do z-score normalization again
    data = (data - new_mean) / new_std

    # if in one column, every value is the same, we set the column to 0
    for col in data.columns:
        if data[col].nunique(dropna=False) == 1:
            data[col] = 0

    # impute NaN values with the mean value of the column
    # print data head
    # sleep for 5 seconds

    # # print only the first row
    # print(data.head(1))
    # # print only the first column
    # print(data.iloc[:, 0])

    # impute nan with 0.0
    data = data.fillna(-2.0)
    # fill anything less than -2 with -2
    data = data.where(data > -2, -2)

    # print(data.head())
    # time.sleep(5)

    try:
        data = data.astype(float)
    except ValueError:
        print("ValueError: ", monomer)
        sys.exit()

    # try:
    #     data = (data - data.mean()) / data.std()
    # except ValueError:
    #     print("ValueError: ", monomer)
    #     sys.exit()
    # # normalize the data; the first column is the index so we don't normalize it
    # data.iloc[:, 1:] = (data.iloc[:, 1:] - data.iloc[:,
    #                     1:].mean()) / data.iloc[:, 1:].std()
    # print(data.head())
    # save the normalized data to csv file
    # data.to_csv(csv_path + monomer[:-4] + ".csv")
    data.to_csv(monomer_data_all_path + monomer[:-4] + ".csv")
    if monomer in evaluation_unit:
        data.to_csv(monomer_data_EU_path + monomer[:-4] + ".csv")
    if monomer in whole_structure:
        data.to_csv(monomer_data_whole_path + monomer[:-4] + ".csv")
    # else:
    #     print("Error when saving the data: ", monomer)
    #     sys.exit()
