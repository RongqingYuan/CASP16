import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


csv_path = "./oligomer_data_Sep_17/processed/"
csv_list = [txt for txt in os.listdir(
    csv_path) if txt.endswith(".csv") and txt.startswith("T1") or txt.startswith("H1")]


group_by_target_path = "./group_by_target_EU/"
if not os.path.exists(group_by_target_path):
    os.makedirs(group_by_target_path)
feature = "ICS(F1)"
feature = "QSglob"
model = "first"
model = "best"


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
        grouped = data_tmp.groupby(["group"])
        # grouped = data_tmp.groupby(["group", "target"])
        grouped = pd.DataFrame(grouped[feature].max())
        # grouped.index = grouped.index.droplevel(1)
        grouped = grouped.apply(lambda x: (x - x.mean()) / x.std())
        grouped = grouped.rename(columns={feature: csv_file.split(".")[0]})
        # print(grouped.head())
        try:
            data = pd.concat([data, grouped], axis=1)
        except ValueError:
            print("ValueError: the data is not consistent for {}".format(csv_file))
            breakpoint()
    data["sum"] = data.sum(axis=1)
    data = data.sort_values(by="sum", ascending=False)
    data.to_csv("./group_by_target_EU/" +
                "sum_{}_{}.csv".format(model, feature))
    # drop sum column
    data_sum = data["sum"]
    data_sum_dict = data_sum.to_dict()
    data = data.drop("sum", axis=1, inplace=False)
    return data, data_sum_dict


features = ["QSglob", "ICS(F1)", "lDDT", "DockQ_Avg",
            "IPS(JaccCoef)", "TMscore"]
models = ["best", "first"]
model = models[0]
data_all = pd.DataFrame()
score_sum_dict = {}
for feature in features:
    data, data_sum_dict = get_group_by_target(csv_list, feature, model)
    print(data.head())
    data_all = pd.concat([data_all, data], axis=1)
    print("Done for {} {}".format(feature, model))
    score_sum_dict[feature] = data_sum_dict
data_all["sum"] = data_all.sum(axis=1)
data_all = data_all.sort_values(by="sum", ascending=False)
data_all.to_csv("./group_by_target_EU/sum_{}_all.csv".format(model))


score_sum = {}
for key in score_sum_dict:
    score = score_sum_dict[key]
    for group in score:
        if group not in score_sum:
            score_sum[group] = 0
        score_sum[group] += score[group]
score_sum = dict(sorted(score_sum.items(), key=lambda x: x[1], reverse=True))
print(score_sum)

teams = list(score_sum.keys())
ind = np.arange(len(teams))

# plot the bar chart
fig, ax = plt.subplots(figsize=(30, 15))
bottom = [0 for i in range(len(teams))]
for key in score_sum_dict:
    score = score_sum_dict[key]
    score = [score[team] for team in teams]
    ax.bar(ind, score, label=key, bottom=bottom, width=0.8)
    bottom = [bottom[i] + score[i] for i in range(len(teams))]

ax.set_xticks(ind)
teams = [team[2:] for team in teams]
ax.set_xticklabels(teams, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=12)
plt.xlabel("Teams", fontsize=12)
plt.ylabel("Sum of z-score", fontsize=12)
plt.title("Sum of z-score for each team", fontsize=12)
plt.savefig("./png/sum_z-score.png", dpi=300)
