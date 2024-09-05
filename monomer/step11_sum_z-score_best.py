import pandas as pd
import numpy as np
import sys
import os


csv_path = "./monomer_data_aug_30/processed/EU/"
csv_list = [txt for txt in os.listdir(
    csv_path) if txt.endswith(".csv") and txt.startswith("T1")]


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
    # data_tmp = data_tmp.loc[(slice(None), slice(None), [
    #                          "1", "2", "3", "4", "5"]), :]
    data_tmp = data_tmp.loc[(slice(None), slice(None),
                             "1"), :]
    grouped = data_tmp.groupby(["group", "target"])
    grouped = pd.DataFrame(grouped[feature].max())
    grouped.index = grouped.index.droplevel(1)
    # grouped = grouped.apply(lambda x: (x - x.mean()) / x.std())

    # change the column name to the target name
    grouped = grouped.rename(columns={feature: csv_file.split(".")[0]})
    data = pd.concat([data, grouped], axis=1)


# sum each row
data["sum"] = data.sum(axis=1)

# sort the data by the sum
data = data.sort_values(by="sum", ascending=False)
data.to_csv("./tmp/" + "sum_best.csv")
