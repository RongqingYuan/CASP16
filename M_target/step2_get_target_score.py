import os
import numpy as np
import pandas as pd
import sys
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=str, default="1")
args = parser.parse_args()
stage = args.stage


# monomer_path = "/home2/s439906/data/CASP16/monomers/"
# monomer_path = "/home2/s439906/data/CASP16/monomers_Sep_8/"
# monomer_path = "/home2/s439906/data/CASP16/monomers_Sep_10/"
monomer_path = "/home2/s439906/data/CASP16/hybrid_Oct_10/"
monomer_list = [txt for txt in os.listdir(
    monomer_path) if txt.endswith(".txt")]
monomer_data_EU_path = "./hybrid_data_Oct_11/processed/"
monomer_data_raw_EU_path = "./step2/"

if stage == "1":
    ...
elif stage == "0":
    monomer_path = "/home2/s439906/data/CASP16/monomers_Oct_3_T0/"
    monomer_list = [txt for txt in os.listdir(
        monomer_path) if txt.endswith(".txt") and "D" in txt]
    monomer_data_EU_path = "./monomer_data_T0_EU/processed/"
    monomer_data_raw_EU_path = "./monomer_data_T0_EU/raw_data/"
elif stage == "2":
    monomer_path = "/home2/s439906/data/CASP16/monomers_Oct_3_T2/"
    monomer_list = [txt for txt in os.listdir(
        monomer_path) if txt.endswith(".txt") and "D" in txt]
    monomer_data_EU_path = "./monomer_data_T2_EU/processed/"
    monomer_data_raw_EU_path = "./monomer_data_T2_EU/raw_data/"


if not os.path.exists(monomer_data_EU_path):
    os.makedirs(monomer_data_EU_path)
if not os.path.exists(monomer_data_raw_EU_path):
    os.makedirs(monomer_data_raw_EU_path)


time_1 = time.time()
for monomer in monomer_list:
    monomer_file = monomer_path + monomer
    data = []
    with open(monomer_file, "r") as f:
        for line in f:
            line = line.split()
            if len(line) > 1:
                data.append(line)
    data = pd.DataFrame(data)
    data.columns = data.iloc[0]
    data = data.drop(0)
    # BUG 1 if there is a header called MODEL, change it to Model
    # set the "Model" column as the index
    if "MODEL" in data.columns:
        data = data.rename(columns={"MODEL": "model"})
    elif "#Model" in data.columns:
        data = data.rename(columns={"#Model": "model"})
    elif "Model" in data.columns:
        data = data.rename(columns={"Model": "model"})
    try:
        data = data.set_index("model")
    except KeyError:
        print("KeyError: ", monomer)
        sys.exit()

    # # BUG 2 another strange bug. Some index line does not have -D* as the suffix. We need to fill them out
    # # here this part I just want to be consistent with the previous code and the following code
    # if "-D" in monomer:
    #     domain_id = monomer.split(".")[0].split("-")[1]
    #     data.index = data.index.map(
    #         lambda x: x + "-" + domain_id if "-D" not in x else x)
    # elif "-D" not in monomer:
    #     domain_id = "D0"  # temporary solution
    #     data.index = data.index.map(
    #         lambda x: x + "-" + domain_id if "-D" not in x else x)

    # data = data[["ICS(F1)", "IPS", "QSglob", "QSbest",
    #              "lDDT", "GDT_TS", "RMSD", "TMscore", "GlobDockQ", "BestDockQ"]]
    data = data[["lDDT",   "TMscore", "GlobDockQ"]]
    # fill - with np.nan
    data = data.replace("-", np.nan)
    data = data.astype(float)
    for column in data.columns:
        # save the raw data of that column
        data_column = data[column]
        data_column.to_csv(monomer_data_raw_EU_path +
                           monomer[:-4] + "_" + column + ".csv")
    # data.to_csv(monomer_data_raw_EU_path + monomer[:-4] + ".csv")
    # initial_z = (data - data.mean()) / data.std()
    # new_z_score = pd.DataFrame(index=data.index, columns=data.columns)
    # for column in data.columns:
    #     filtered_data = data[column][initial_z[column] >= -2]
    #     new_mean = filtered_data.mean(skipna=True)
    #     new_std = filtered_data.std(skipna=True)
    #     new_z_score[column] = (data[column] - new_mean) / new_std
    # new_z_score = new_z_score.fillna(-2.0)
    # new_z_score = new_z_score.where(new_z_score > -2, -2)
    # new_z_score.to_csv(monomer_data_EU_path + monomer[:-4] + ".csv")

time_2 = time.time()
print("Time used: ", time_2 - time_1)
