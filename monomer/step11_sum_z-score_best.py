import pandas as pd
import numpy as np
import sys
import os


hard_group = [
    "T1207-D1",
    "T1210-D1",
    "T1210-D2",
    "T1220s1-D1",
    "T1220s1-D2",
    "T1226-D1",
    "T1228v1-D3",
    "T1228v1-D4",
    "T1228v2-D3",
    "T1228v2-D4",
    "T1239v1-D4",
    "T1239v2-D4",
    "T1271s1-D1",
    "T1271s3-D1",
    "T1271s8-D1",
]

medium_group = [
    "T1210-D3",
    "T1212-D1",
    "T1218-D1",
    "T1218-D2",
    "T1227s1-D1",
    "T1228v1-D1",
    "T1228v2-D1",
    "T1230s1-D1",
    "T1237-D1",
    "T1239v1-D1",
    "T1239v1-D3",
    "T1239v2-D1",
    "T1239v2-D3",
    "T1243-D1",
    "T1244s1-D1",
    "T1244s2-D1",
    "T1245s2-D1",
    "T1249v1-D1",
    "T1257-D3",
    "T1266-D1",
    "T1270-D1",
    "T1270-D2",
    "T1267s1-D1",
    "T1267s1-D2",
    "T1267s2-D1",
    "T1269-D1",
    "T1269-D2",
    "T1269-D3",
    "T1271s2-D1",
    "T1271s4-D1",
    "T1271s5-D1",
    "T1271s5-D2",
    "T1271s7-D1",
    "T1271s8-D2",
    "T1272s2-D1",
    "T1272s6-D1",
    "T1272s8-D1",
    "T1272s9-D1",
    "T1279-D2",
    "T1284-D1",
    "T1295-D1",
    "T1295-D3",
    "T1298-D1",
    "T1298-D2",
]

easy_group = [
    "T1201-D1",
    "T1201-D2",
    "T1206-D1",
    "T1208s1-D1",
    "T1208s2-D1",
    "T1218-D3",
    "T1228v1-D2",
    "T1228v2-D2",
    "T1231-D1",
    "T1234-D1",
    "T1235-D1",
    "T1239v1-D2",
    "T1239v2-D2",
    "T1240-D1",
    "T1240-D2",
    "T1245s1-D1",
    "T1246-D1",
    "T1257-D1",
    "T1257-D2",
    "T1259-D1",
    "T1271s6-D1",
    "T1274-D1",
    "T1276-D1",
    "T1278-D1",
    "T1279-D1",
    "T1280-D1",
    "T1292-D1",
    "T1294v1-D1",
    "T1294v2-D1",
    "T1295-D2",
    # "T1214",
]


csv_path = "./monomer_data_aug_30/processed/EU/"
csv_path = "./monomer_data_Sep_10/processed/EU/"
csv_list = [txt for txt in os.listdir(
    csv_path) if txt.endswith(".csv") and txt.startswith("T1")]

model = "first"
model = "best"

mode = "easy"
mode = "medium"
mode = "hard"
mode = "all"

if mode == "hard":
    csv_list = [csv for csv in csv_list if csv.split(
        ".")[0] in hard_group]

elif mode == "medium":
    csv_list = [csv for csv in csv_list if csv.split(
        ".")[0] in medium_group]

elif mode == "easy":
    csv_list = [csv for csv in csv_list if csv.split(
        ".")[0] in easy_group]

elif mode == "all":
    pass
print(csv_list.__len__())
# breakpoint()
# read all data and concatenate them into one big dataframe
data = pd.DataFrame()
feature = "GDT_TS"

for csv_file in csv_list:
    print("Processing {}".format(csv_file))
    data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
    data_tmp = pd.DataFrame(data_tmp[feature])
    print(data_tmp.shape)
    data_tmp.index = data_tmp.index.str.extract(
        r'(T\w+)TS(\w+)_(\w+)-(D\w+)').apply(lambda x: (f"{x[0]}-{x[3]}", f"TS{x[1]}", x[2]), axis=1)
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
    grouped = data_tmp.groupby(["group", "target"])
    grouped = pd.DataFrame(grouped[feature].max())
    grouped.index = grouped.index.droplevel(1)

    # I actually don't understand why this is necessary... but need to keep it in mind.
    grouped = grouped.apply(lambda x: (x - x.mean()) / x.std())

    grouped = grouped.rename(columns={feature: csv_file.split(".")[0]})
    data = pd.concat([data, grouped], axis=1)


data["sum"] = data.sum(axis=1)
data = data.sort_values(by="sum", ascending=False)
data.to_csv("./sum/" + "sum_{}_{}.csv".format(model, mode))
