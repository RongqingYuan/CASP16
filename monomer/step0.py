# use packages to deal with extentions
import os
import numpy as np
import pandas as pd
import sys

monomer_path = "/data/data1/conglab/qcong/CASP16/monomers/"
monomer_list = [txt for txt in os.listdir(
    monomer_path) if txt.endswith(".txt")]


csv_path = "./csv/"
csv_raw_path = "./csv_raw/"
if not os.path.exists(csv_path):
    os.makedirs(csv_path)
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
    # remove the "GR#" column and "#" column
    data = data.drop("GR#", axis=1)
    data = data.drop("#", axis=1)

    # # check the data shape
    # print(data.shape)
    # # print the first 5 rows
    # print(data.head())

    # save the data to a csv file
    data.to_csv(csv_raw_path + monomer[:-4] + ".csv")

    # impute the N/A values with the mean value of the column
    # impute the - values with the mean value of the column
    # data = data.apply(pd.to_numeric)
    data.replace("N/A", np.nan, inplace=True)
    data.replace("-", np.nan, inplace=True)
    data = data.fillna(data.mean())

    # # print only the first row
    # print(data.head(1))
    # # print only the first column
    # print(data.iloc[:, 0])

    # convert the data type to float
    data = data.astype(float)

    try:
        data = data.astype(float)
    except ValueError:
        print("ValueError: ", monomer)
        sys.exit()

    try:
        data = (data - data.mean()) / data.std()
    except ValueError:
        print("ValueError: ", monomer)
        sys.exit()
    # # normalize the data; the first column is the index so we don't normalize it
    # data.iloc[:, 1:] = (data.iloc[:, 1:] - data.iloc[:,
    #                     1:].mean()) / data.iloc[:, 1:].std()
    # print(data.head())

    # normalize the data with the z-score
    data = (data - data.mean()) / data.std()
    # save the normalized data to csv file
    data.to_csv(csv_path + monomer[:-4] + ".csv")
