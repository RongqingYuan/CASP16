# use packages to deal with extentions
import os
import numpy as np
import pandas as pd
import sys
import time

oligomer_path = "/data/data1/conglab/qcong/CASP16/oligomers/"
oligomer_list = [txt for txt in os.listdir(
    oligomer_path) if txt.endswith(".txt")]


csv_path = "./csv/"
csv_raw_path = "./csv_raw/"
if not os.path.exists(csv_path):
    os.makedirs(csv_path)
if not os.path.exists(csv_raw_path):
    os.makedirs(csv_raw_path)
all_data = {}

# read the oligomer list
# this is too messy, we need to completely re-write this part compared to the monomer part
for oligomer in oligomer_list:
    oligomer_file = oligomer_path + oligomer
    data = []
    with open(oligomer_file, "r") as f:
        flag = 0  # to indicate if we have gone through the header line
        for line in f:
            tmp = line.split('[')
            if len(tmp) == 1:
                tmp = tmp[0].split()
                if len(tmp) > 5 and flag == 1:
                    # there is something wrong with the data, print it out
                    print("Error: some data is missing for {}".format(oligomer))
                if len(tmp) > 5 and flag == 0:
                    data.append(tmp)
                    flag = 1

                continue
            part_1 = tmp[0].split()
            tmp = tmp[1]
            tmp = tmp.split(']')
            contact_score = tmp[0]
            part_2 = tmp[1].split()[:5]  # the 5th is not used. a placeholder
            contact_score = contact_score.split(',')
            try:
                contact_score = [float(score) for score in contact_score]
                mean_score = str(np.mean(contact_score))
            except ValueError:
                print("ValueError: {} Probably due to missing value".format(oligomer))
                mean_score = "0"  # probably the predictor didn't even get a complex
            line = part_1 + [mean_score] + part_2
            # if type(contact_score) == list:
            #     line = part_1 + contact_score + part_2
            # else:
            #     line = part_1 + [contact_score] + part_2
            data.append(line)
    # print(data)
    all_data[oligomer] = data
    # print(oligomer)
    length = len(data[0])
    for i in range(1, len(data)):
        if len(data[i]) != length:
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
    data.to_csv(csv_raw_path + oligomer[:-4] + ".csv")  # this is good raw data

    # remove the "GR#" column and "#" column
    data = data.drop(["#", "Gr.Code", "Stoich.", "Symm."], axis=1)
    # # check the data shape
    # print(data.shape)
    # # print the first 5 rows
    # print(data.head())

    # save the data to a csv file
    data.to_csv(csv_raw_path + oligomer[:-4] + ".csv")

    # impute the N/A values with the mean value of the column
    # impute the - values with the mean value of the column
    # data = data.apply(pd.to_numeric)
    data.replace("N/A", np.nan, inplace=True)
    data.replace("-", np.nan, inplace=True)
    data = data.fillna(data.mean())
    data = data.drop(["QS_Interfaces", "SymmGr.RMSD"], axis=1)

    # # print only the first row
    # print(data.head(1))
    # # print only the first column
    # print(data.iloc[:, 0])

    # convert the data type to float
    data = data.astype(float)

    try:
        data = data.astype(float)
    except ValueError:
        print("ValueError: ", oligomer)
        sys.exit()

    try:
        data = (data - data.mean()) / data.std()
    except ValueError:
        print("ValueError: ", oligomer)
        sys.exit()
    # # normalize the data; the first column is the index so we don't normalize it
    # data.iloc[:, 1:] = (data.iloc[:, 1:] - data.iloc[:,
    #                     1:].mean()) / data.iloc[:, 1:].std()
    # print(data.head())

    # normalize the data with the z-score
    data = (data - data.mean()) / data.std()
    # save the normalized data to csv file
    data.to_csv(csv_path + oligomer[:-4] + ".csv")
