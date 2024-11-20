import os
import numpy as np
import pandas as pd
import sys
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=str, default="1")
parser.add_argument('--input_path', type=str,
                    # default="/home2/s439906/data/CASP16/monomers_EU_merge_v/"
                    default="/home2/s439906/data/CASP16/monomer_inputs/"
                    )
parser.add_argument('--output_path', type=str,
                    default="./monomer_data_Sep_15_EU/")

args = parser.parse_args()
stage = args.stage
input_path = args.input_path
output_path = args.output_path

# monomer_path = "/home2/s439906/data/CASP16/monomers_Sep_8/"
# monomer_path = "/home2/s439906/data/CASP16/monomers_Sep_10/"


monomer_path = input_path
monomer_list = [txt for txt in os.listdir(
    monomer_path) if txt.endswith(".txt")]
monomer_data_EU_path = output_path + \
    "processed/" if output_path[-1] == "/" else output_path + "/processed/"
monomer_data_raw_EU_path = output_path + \
    "raw_data/" if output_path[-1] == "/" else output_path + "/raw_data/"

if stage == "1":
    monomer_list = [txt for txt in os.listdir(
        monomer_path) if txt.endswith(".txt") and txt.startswith("T1")]
elif stage == "0":
    monomer_path = "/home2/s439906/data/CASP16/monomers_Oct_3_T0/"
    monomer_list = [txt for txt in os.listdir(
        monomer_path) if txt.endswith(".txt")]
    monomer_data_EU_path = "./monomer_data_T0_EU/processed/"
    monomer_data_raw_EU_path = "./monomer_data_T0_EU/raw_data/"
elif stage == "2":
    monomer_path = "/home2/s439906/data/CASP16/monomers_Oct_3_T2/"
    monomer_list = [txt for txt in os.listdir(
        monomer_path) if txt.endswith(".txt")]
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
        data = data.rename(columns={"MODEL": "Model"})
    try:
        data = data.set_index("Model")
    except KeyError:
        print("KeyError: ", monomer)
        sys.exit()
    # BUG 1.1 some monomer files have "MolProb" instead of "MolPrb_Score"
    # set the "MolPrb_Score" column as the index
    if "MolProb" in data.columns:
        data = data.rename(columns={"MolProb": "MolPrb_Score"})

    # BUG 2 another strange bug. Some index line does not have -D* as the suffix. We need to fill them out
    if "-D" in monomer:
        domain_id = monomer.split(".")[0].split("-")[1]
        data.index = data.index.map(
            lambda x: x + "-" + domain_id if "-D" not in x else x)
    elif "-D" not in monomer:
        domain_id = "D0"  # temporary solution
        data.index = data.index.map(
            lambda x: x + "-" + domain_id if "-D" not in x else x)

    data.to_csv(monomer_data_raw_EU_path + monomer[:-4] + ".csv")
    data = data.drop(["GR#", "#"], axis=1)
    try:
        data = data.drop(["DipDiff", "BBscore", "SCscore"], axis=1)
    except KeyError:
        continue  # CASP15 does not have these columns
    data = data.drop(["RANK"], axis=1)

    data.replace("N/A", np.nan, inplace=True)
    data.replace("-", np.nan, inplace=True)
    data = data.astype(float)
    inverse_columns = ["RMS_CA", "RMS_ALL", "err",
                       "RMSD[L]", "MolPrb_Score", "FlexE", "MP_clash", "MP_rotout", "MP_ramout"]
    data[inverse_columns] = -data[inverse_columns]
    initial_z = (data - data.mean()) / data.std()

    new_z_score = pd.DataFrame(index=data.index, columns=data.columns)
    for column in data.columns:
        filtered_data = data[column][initial_z[column] >= -2]
        new_mean = filtered_data.mean(skipna=True)
        new_std = filtered_data.std(skipna=True)
        new_z_score[column] = (data[column] - new_mean) / new_std
    new_z_score = new_z_score.fillna(-2.0)
    new_z_score = new_z_score.where(new_z_score > -2, -2)
    new_z_score.to_csv(monomer_data_EU_path + monomer[:-4] + ".csv")

time_2 = time.time()
# print("Time used: ", time_2 - time_1)
