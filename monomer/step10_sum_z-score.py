import pandas as pd
import numpy as np
import sys
# read ./monomer_data_aug_28/all/fa_processed_all.csv


data = pd.read_csv(
    "./monomer_data_aug_28/all/fa_processed_all.csv", index_col=0)
print(data.head())

data.index = data.index.str.extract(
    r'(T\w+)TS(\w+)_(\w+)-(D\w+)').apply(lambda x: (f"{x[0]}-{x[3]}", f"TS{x[1]}", x[2]), axis=1)

print(data.shape)
print(data.head())
data.index = pd.MultiIndex.from_tuples(
    data.index, names=['target', 'group', 'submission_id'])
# grouped = score_df.groupby(["target", "group"])
grouped = data.groupby(["group", "target"])
print(grouped.head())
print(type(grouped))

feature = "GDT_TS"
grouped = pd.DataFrame(grouped[feature].max())
print(grouped.head())
# grouped = pd.DataFrame(grouped["low_resolution_score"].max())
# grouped = pd.DataFrame(grouped["chemical_score"].max())
# sum the scores for each group
grouped = grouped.groupby("group").sum()
print(grouped)
# sort grouped by value
grouped = grouped.sort_values(by=feature, ascending=False)
# grouped = grouped.sort_values(by="low_resolution_score", ascending=False)
# grouped = grouped.sort_values(by="chemical_score", ascending=False)
print(grouped)
for k, v in grouped.iterrows():
    print(k, v)
sys.exit()


print(data.index)
measure_of_interest = "low_resolution_score"
measure_of_interest = "chemical_score"
measure_of_interest = "high_resolution_score"
score_df = pd.DataFrame(data[measure_of_interest])
score_df.to_csv("./tmp/" + "score_df_2.csv")
data_whole = score_df.stack().unstack('group')
data_whole.index = [f'{target}_{submission_id}' for target,
                    submission_id, measure in data_whole.index]
print(data_whole.head())
# get a new column called submission_id and target
data_whole['target'] = data_whole.index.str.split('_').str[0]
data_whole['submission_id'] = data_whole.index.str.split('_').str[1]
# remove any submission_id = 6
data_whole = data_whole[data_whole['submission_id'] != '6']
data_whole_by_target = data_whole.groupby('target').max()
data_whole_by_target.to_csv("./tmp/" + "data_whole_by_target.csv")

wanted_group = ["052", "022", "456", "051",
                "319", "287", "208", "028", "019", "294", "465", "110", "345", "139"]
wanted_group = ["TS"+group for group in wanted_group]

points = {}
