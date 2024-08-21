import os
import numpy as np
import pandas as pd
import sys
import time


monomer_path = "/data/data1/conglab/qcong/CASP16/monomers/"
monomer_list = [txt for txt in os.listdir(
    monomer_path) if txt.endswith(".txt")]
csv_path = "./csv_transpose/"
csv_raw_path = "./csv_transpose_raw/"
csv_tmp_path = "./csv_transpose_tmp/"
if not os.path.exists(csv_path):
    os.makedirs(csv_path)
if not os.path.exists(csv_raw_path):
    os.makedirs(csv_raw_path)
if not os.path.exists(csv_tmp_path):
    os.makedirs(csv_tmp_path)

all_data = {}
# read the monomer list
for monomer in monomer_list:
    monomer_file = monomer_path + monomer
    data = []
    with open(monomer_file, "r") as f:
        for line in f:
            line = line.split()
            if len(line) > 1:
                data.append(line)
    all_data[monomer] = data

    # convert the data to dataframe, the first row is the column names, the first column is the index
    data = pd.DataFrame(data)
    # set the first row as the column names
    data.columns = data.iloc[0]
    # drop the first row
    data = data.drop(0)
    # set the "Model" column as the index
    data = data.set_index("Model")
    # save it as complete raw data, in case we need it later
    data.to_csv(csv_raw_path + monomer[:-4] + ".csv")

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
    data = (data - data.mean()) / data.std()

    # remove outliers: i.e. if any value is smaller than -2, we directly remove the whole row
    # data = data[(data > -2).all(axis=1)]
    data = data[((data >= -2) | data.isna()).all(axis=1)]
    # save to csv_tmp path to see if anything goes wrong
    data.to_csv(csv_tmp_path + monomer[:-4] + ".csv")

    # after removing the outliers, we need to do z-score normalization again
    data = (data - data.mean()) / data.std()

    # if in one column, every value is the same, we set the column to 0
    for col in data.columns:
        # Check if all values are the same, including NaN
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
    data = data.fillna(0.0)

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

    data.to_csv(csv_path + monomer[:-4] + ".csv")
