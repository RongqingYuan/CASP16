import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys


models = ["best", "first"]

features = ["QSglob"]
features = ["QSglob", "ICS(F1)", "lDDT", "DockQ_Avg",
            "IPS(JaccCoef)", "TMscore"]
features = ["QSglob", "ICS(F1)", "lDDT", "DockQ_Avg",
            "IPS(JaccCoef)"]
features = ["QSglob", "ICS(F1)",
            "IPS(JaccCoef)"]

feature = features[2]
feature = "TMscore"
feature = "DockQ_Avg"
model = "first"
model = "best"

data_file = "sum_{}_{}.csv".format(model, feature)
data_path = "./group_by_target/"
data = pd.read_csv(data_path + data_file, index_col=0)
data = data.drop("sum", axis=1)
data = data.T
# breakpoint()
data.fillna(-2, inplace=True)
# breakpoint()
groups = data.columns
# breakpoint()

points = {}
length = len(groups)
for i in range(length):
    for j in range(length):
        if i == j:
            continue
        group_1 = groups[i]
        group_2 = groups[j]
        data_1 = data[group_1]
        data_2 = data[group_2]
        t, p = stats.ttest_rel(data_1, data_2)
        if group_1 not in points:
            points[group_1] = 0
        if t > 0 and p/2 < 0.05:
            points[group_1] += 1
points = dict(sorted(points.items(), key=lambda x: x[1], reverse=True))
print(points)
groups = list(points.keys())

bootstrap_points = {}
bootstrap_win_matrix = [[0 for i in range(length)] for j in range(length)]
bootstrap_rounds = 1000

for r in range(bootstrap_rounds):
    data_bootstrap = data.sample(frac=1, replace=True, axis=0)
    for i in range(length):
        for j in range(length):
            if i == j:
                continue
            group_1 = groups[i]
            group_2 = groups[j]
            data_1 = data_bootstrap[group_1]
            data_2 = data_bootstrap[group_2]
            t, p = stats.ttest_rel(data_1, data_2)
            if t > 0 and p/2 < 0.05:
                bootstrap_win_matrix[i][j] += 1
    print("Round: {}".format(r))

plt.figure(figsize=(30, 25))
ax = sns.heatmap(bootstrap_win_matrix, annot=False,
                 cmap='Greys', cbar=True, square=True, )
#  linewidths=1, linecolor='black')
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(2)
ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment='center')
ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center')
plt.xticks(np.arange(length), groups, rotation=45, ha='right', fontsize=10)
plt.yticks(np.arange(length), groups, rotation=0, fontsize=10)
plt.title("Bootstrap result of {} for {} targets".format(
    feature, model), fontsize=20)
plt.savefig(
    "./bootstrap/win_matrix_bootstrap_{}_{}_n={}.png".format(feature, model, bootstrap_rounds), dpi=300)
sys.exit(0)


def paired_t_test_for_groups(model, feature):
    data_file = "sum_{}_{}.csv".format(model, feature)
    data_path = "./group_by_target/"
    data = pd.read_csv(data_path + data_file, index_col=0)
    data = data.drop("sum", axis=1)
    data = data.T
    # impute the missing values with -2
    data.fillna(-2, inplace=True)
    groups = data.columns
    points = {}
    for group_1 in groups:
        for group_2 in groups:
            if group_1 == group_2:
                continue
            data_1 = data[group_1]
            data_2 = data[group_2]
            # if group_1 == "TS022":
            #     breakpoint()
            # breakpoint()
            t, p = stats.ttest_rel(data_1, data_2)
            if group_1 not in points:
                points[group_1] = 0
            if t > 0 and p/2 < 0.05:
                points[group_1] += 1
    points = dict(sorted(points.items(), key=lambda x: x[1], reverse=True))
    return points


points_dict = {}
for feature in features:
    model = "best"
    points = paired_t_test_for_groups(model, feature)
    points_dict[f"{model}_{feature}"] = points


# sum all points
points_sum = {}
for key in points_dict:
    points = points_dict[key]
    for group in points:
        if group not in points_sum:
            points_sum[group] = 0
        points_sum[group] += points[group]

points_sum = dict(sorted(points_sum.items(), key=lambda x: x[1], reverse=True))
print(points_sum)

teams = list(points_sum.keys())
ind = np.arange(len(teams))

fig, ax = plt.subplots(figsize=(30, 15))
bottom = [0 for i in range(len(teams))]
for key in points_dict:
    points = points_dict[key]
    points = [points[team] for team in teams]
    ax.bar(ind, points, bottom=bottom, label=key, width=0.8)
    bottom = [bottom[i] + points[i] for i in range(len(teams))]
ax.set_xticks(ind)
teams = [team[2:] for team in teams]
ax.set_xticklabels(teams)
ax.legend(fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Number of wins", fontsize=12)
plt.title("Number of wins for each team", fontsize=12)
plt.savefig("./png/sum_points.png", dpi=300)

breakpoint()
# plot the points as bar chart

# now bootstrap the data and compared the first 2 groups
# read the data
model = "best"
feature = "QSglob"
feature = "TMscore"
data_file = "sum_{}_{}.csv".format(model, feature)
data_path = "./group_by_target/"
data = pd.read_csv(data_path + data_file, index_col=0)
data = data.drop("sum", axis=1)
data = data.T

# get the groups
groups = list(points_sum.keys())
# get the first 2 groups
group_1 = groups[0]
group_2 = groups[2]

bootstrapping_rounds = 1000
group_1_win = 0
group_2_win = 0
no_diff = 0
for i in range(bootstrapping_rounds):
    new_data = data.sample(frac=1, replace=True, axis=0)
    data_1 = new_data[group_1]
    data_2 = new_data[group_2]
    t, p = stats.ttest_rel(data_1, data_2)
    if t > 0 and p/2 < 0.05:
        group_1_win += 1
    elif t < 0 and p/2 < 0.05:
        group_2_win += 1
    else:
        no_diff += 1
    if i % 100 == 0:
        print(f"Round: {i}")


print(f"Group 1: {group_1} wins {group_1_win} times")
print(f"Group 2: {group_2} wins {group_2_win} times")
print(f"No significant difference: {no_diff} times")

sys.exit(0)
# try bootstrapping over the targets
times = 100
for i in range(times):
    # sample new data
    points = {}
    new_data = data.sample(frac=1, replace=True, axis=0)
    # print(new_data)
    for group_1 in groups:
        for group_2 in groups:
            if group_1 == group_2:
                continue
            data_1 = new_data[group_1]
            data_2 = new_data[group_2]
            # breakpoint()
            t, p = stats.ttest_rel(data_1, data_2)
            if group_1 not in points:
                points[group_1] = 0
            if t > 0 and p/2 < 0.005:
                points[group_1] += 1
    breakpoint()
    points = dict(sorted(points.items(), key=lambda x: x[1], reverse=True))
    print(points)
