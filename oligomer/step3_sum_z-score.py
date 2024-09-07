import pandas as pd
import numpy as np
import sys
import os


csv_path = "./oligomer_data/processed/"
csv_list = [txt for txt in os.listdir(
    csv_path) if txt.endswith(".csv") and txt.startswith("T1") or txt.startswith("H1")]

# breakpoint()
# csv_file = csv_path + csv_list[3]
# print("Processing {}".format(csv_file))
# data = pd.read_csv(csv_file, index_col=0)

# read all data and concatenate them into one big dataframe
data = pd.DataFrame()
for csv_file in csv_list:
    print("Processing {}".format(csv_file))
    data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
    data_tmp = pd.DataFrame(data_tmp["QSglob"])
    print(data_tmp.shape)
    if data_tmp.shape[1] == 35:
        print("something wrong with {}".format(csv_file))
        sys.exit(0)
    data_tmp.index = data_tmp.index.str.extract(
        r'(\w+)TS(\w+)_(\w+)').apply(lambda x: (f"{x[0]}", f"TS{x[1]}", x[2]), axis=1)

    # print(data_tmp.shape)
    # print(data_tmp.head())
    data_tmp.index = pd.MultiIndex.from_tuples(
        data_tmp.index, names=['target', 'group', 'submission_id'])
    # save the target name
    # target_name = data_tmp.index.get_level_values("target")[0]
    # grouped = score_df.groupby(["target", "group"])
    grouped = data_tmp.groupby(["group", "target"])
    # print(grouped.head())
    # print(type(grouped))

    feature = "ICS(F1)"
    feature = "DockQ_Avg"
    feature = "QSglob"
    grouped = pd.DataFrame(grouped[feature].max())
    # print(grouped.head())
    # grouped = pd.DataFrame(grouped["low_resolution_score"].max())
    # grouped = pd.DataFrame(grouped["chemical_score"].max())
    # sum the scores for each group
    print(grouped.head())
    result = grouped.groupby("group")
    print(type(result))
    # convert the groupby object to a dataframe
    result = result.apply(lambda x: x)
    print(result.head())
    print(type(result))
    result.index = result.index.droplevel(1)
    # breakpoint()
    # normalize the scores using z-score
    # print(result.head())
    # result = result.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    result = result.apply(lambda x: (x - x.mean()) / x.std())
    # print(result.head())
    # result.to_csv("./tmp/" + "grouped_results.csv")s
    # breakpoint()
    # sort grouped by value
    # grouped = grouped.sort_values(by="low_resolution_score", ascending=False)
    # grouped = grouped.sort_values(by="chemical_score", ascending=False)
    # print(grouped)
    # print(result.head())
    # change the column name to the target name
    result = result.rename(columns={"GDT_TS": csv_file.split(".")[0]})
    grouped.index = grouped.index.droplevel(1)
    grouped = grouped.apply(lambda x: (x - x.mean()) / x.std())
    grouped = grouped.rename(columns={"GDT_TS": csv_file.split(".")[0]})
    print(grouped.head())
    data = pd.concat([data, grouped], axis=1)


# sum each row
data["sum"] = data.sum(axis=1)

# sort the data by the sum
data = data.sort_values(by="sum", ascending=False)
data.to_csv("./sum/" + "sum_best.csv")
