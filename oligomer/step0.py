# use packages to deal with extentions
import os
import numpy as np
import pandas as pd
import sys

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
        for line in f:
            line = line.split()
            if len(line) > 10:
                data.append(line)

    print(data)
    print(data[0].__len__())
    print(data[1].__len__())
    print(data[0])
    print(data[1])
    print(data[1][19])
    print(data[1][-6])

    # deal with the symmetric group...
    if data[0].__len__() < data[1].__len__():
        for i in range(1, len(data)):

            if not data[i][-1].startswith("c") and not data[i][-1].startswith("-"):
                print("something other symmetric group appears for {}".format(oligomer))
                print(data[i])
                sys.exit(0)  # not sure about the symmetric group...

            c_count = [c for c in data[i] if c.startswith("c")].__len__()
            d_count = [d for d in data[i] if d.startswith("d")].__len__()
            t_count = [t for t in data[i] if t.startswith("t")].__len__()
            sym_count = c_count + d_count + t_count
            if sym_count > 1:
                tmp = data[i][-sym_count:]
                sym_group_rmsd = " ".join(tmp)
                data[i] = data[i][:-sym_count]
                data[i].append(sym_group_rmsd)

    # deal with contact...
    if data[0].__len__() < data[1].__len__():
        # from data[19] to data[-6] is the data we need to concat into one column
        for i in range(1, len(data)):
            # print(data[i][19:-5])
            # get the slice from data[19] to data[-5]
            tmp = data[i][-5:]
            contact_score = " ".join(data[i][19:-5])
            # print(contact_score)
            data[i] = data[i][:19]
            data[i].append(contact_score)
            data[i].extend(tmp)

    assert data[0].__len__() == data[1].__len__()

    all_data[oligomer] = data

    # convert the data to dataframe, the first row is the column names, the first column is the index
    data = pd.DataFrame(data)
    # set the first row as the column names
    data.columns = data.iloc[0]
    # drop the first row
    data = data.drop(0)
    # set the "Model" column as the index
    data = data.set_index("Model")
    data.to_csv(csv_raw_path + oligomer[:-4] + ".csv")  # this is good raw data

    print(data)
    # data = data.drop(, axis=1)
    data = data.drop(["#", "Gr.Code", "Stoich.", "Symm."], axis=1)
    # in QSglob_perInterface, the values are in list, we need to get all of them and take the mean
    data.replace("N/A", np.nan, inplace=True)
    data.replace("-", np.nan, inplace=True)
    data = data.fillna(data.mean())
    data["QSglob_perInterface"] = data["QSglob_perInterface"].apply(
        lambda x: np.mean([float(i) for i in str(x).split("[")[-1].split("]")[0].split(",")]))
    data = data.drop(["QS_Interfaces", "SymmGr.RMSD"], axis=1)

    print(data)
    # # check the data shape
    # print(data.shape)
    # # print the first 5 rows
    # print(data.head())

    # save the data to a csv file

    # impute the N/A values with the mean value of the column
    # impute the - values with the mean value of the column
    # data = data.apply(pd.to_numeric)

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

    # normalize the data with the z-score
    data = (data - data.mean()) / data.std()
    # save the normalized data to csv file
    data.to_csv(csv_path + oligomer[:-4] + ".csv")

    # break
