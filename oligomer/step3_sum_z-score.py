import pandas as pd
import numpy as np
import sys
import os


csv_path = "./oligomer_data/processed/"
csv_list = [txt for txt in os.listdir(
    csv_path) if txt.endswith(".csv") and txt.startswith("T1") or txt.startswith("H1")]

feature = "ICS(F1)"
feature = "QSglob"
model = "best"
model = "first"


def get_group_by_target(csv_list, feature, model):
    data = pd.DataFrame()
    for csv_file in csv_list:
        data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
        data_tmp = pd.DataFrame(data_tmp[feature])
        print("Processing {}".format(csv_file), data_tmp.shape)
        data_tmp.index = data_tmp.index.str.extract(
            r'(\w+)TS(\w+)_(\w+)').apply(lambda x: (f"{x[0]}", f"TS{x[1]}", x[2][0]), axis=1)
        # note:for the submission_id, we need to get the first character as the number because it is like "1o" now
        data_tmp.index = pd.MultiIndex.from_tuples(
            data_tmp.index, names=['target', 'group', 'submission_id'])
        if model == "best":
            data_tmp = data_tmp.loc[(slice(None), slice(None), [
                                    "1", "2", "3", "4", "5"]), :]
        elif model == "first":
            data_tmp = data_tmp.loc[(slice(None), slice(None), "1"), :]
        grouped = data_tmp.groupby(["group", "target"])
        grouped = pd.DataFrame(grouped[feature].max())
        grouped.index = grouped.index.droplevel(1)
        grouped = grouped.apply(lambda x: (x - x.mean()) / x.std())
        grouped = grouped.rename(columns={feature: csv_file.split(".")[0]})
        # print(grouped.head())
        data = pd.concat([data, grouped], axis=1)

    data["sum"] = data.sum(axis=1)
    data = data.sort_values(by="sum", ascending=False)
    data.to_csv("./group_by_target/" + "sum_{}_{}.csv".format(model, feature))

    return 0


features = ["QSglob", "ICS(F1)", "lDDT", "DockQ_Avg",
            "IPS(JaccCoef)", "TMscore"]
models = ["best", "first"]

for feature in features:
    for model in models:
        get_group_by_target(csv_list, feature, model)
        print("Done for {} {}".format(feature, model))
