import os
import numpy as np
import pandas as pd

# oligomer_path = "/data/data1/conglab/qcong/CASP16/oligomers/"
# oligomer_path = "/home2/s439906/data/CASP16/oligomers_Sep_8/"
# oligomer_path = "/home2/s439906/data/CASP16/oligomers_Sep_17/"
# oligomer_path = "/home2/s439906/data/CASP16/oligomers_Sep_17_merge_v/"
# oligomer_path = "/home2/s439906/data/CASP16/oligomer_server_Nov_16/"
oligomer_path = "/home2/s439906/data/CASP16/oligomer_web_Nov_17/"

oligomer_list = [txt for txt in os.listdir(
    oligomer_path) if txt.endswith(".txt") and ("T1" in txt or "H1" in txt)]

# for oligomer in oligomer_list:
#     if oligomer.startswith("T1269"):
#         # pop out the bad target
#         oligomer_list.remove(oligomer)
# for oligomer in oligomer_list:
#     if oligomer.startswith("T1219"):
#         # pop out the bad target
#         oligomer_list.remove(oligomer)
for oligomer in oligomer_list:
    if oligomer.startswith("T1295o."):
        # pop out the bad target
        oligomer_list.remove(oligomer)
for oligomer in oligomer_list:
    if oligomer.startswith("H1265_"):
        # pop out the bad target
        oligomer_list.remove(oligomer)
oligomer_out_raw_path = "./oligomer_data_Nov_17/raw_data/"
oligomer_out_path = "./oligomer_data_Nov_17/processed/"
if not os.path.exists(oligomer_out_path):
    os.makedirs(oligomer_out_path)
if not os.path.exists(oligomer_out_raw_path):
    os.makedirs(oligomer_out_raw_path)
for file in os.listdir(oligomer_out_path):
    os.remove(oligomer_out_path + file)
for file in os.listdir(oligomer_out_raw_path):
    os.remove(oligomer_out_raw_path + file)

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
                print("ValueError: {} for contact score. Probably due to missing value. Will assign 0".format(
                    oligomer))
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
    # this is good raw data
    data.to_csv(oligomer_out_raw_path + oligomer[:-4] + ".csv")

    data = data.drop(["#", "Gr.Code", "Stoich.", "Symm."], axis=1)

    # print(data.shape)
    # print(data.head())

    # impute the N/A values with the mean value of the column
    # impute the - values with the mean value of the column
    # data = data.apply(pd.to_numeric)
    data.replace("N/A", np.nan, inplace=True)
    data.replace("-", np.nan, inplace=True)
    data = data.drop(["QS_Interfaces"], axis=1)
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
