import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
csv_path = "./csv/"
csv_list = [txt for txt in os.listdir(csv_path) if txt.endswith(".csv")]

# csv_file = csv_path + csv_list[3]
# print("Processing {}".format(csv_file))
# data = pd.read_csv(csv_file, index_col=0)

# read all data and concatenate them into one big dataframe

use_domain = False  # can change
use_domain = True  # can change
score_type = "best"
score_type = "first"
data = pd.DataFrame()


for csv_file in csv_list:
    print("Processing {}".format(csv_file))
    parse_csv_file = csv_file.split("-")
    if len(parse_csv_file) == 2 and use_domain:
        data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
        print(data_tmp.shape)
        if data_tmp.shape[1] == 35:
            print("something wrong with {}".format(csv_file))
            sys.exit(0)
        data = pd.concat([data, data_tmp], axis=0)
    if len(parse_csv_file) == 1 and not use_domain:
        data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
        print(data_tmp.shape)
        if data_tmp.shape[1] == 35:
            print("something wrong with {}".format(csv_file))
            sys.exit(0)
        data = pd.concat([data, data_tmp], axis=0)

# print the first 5 rows
print(data.head())
print(data.shape)
# sys.exit(0)

NP_P_remove = False
NP_remove = False
FlexE_remove = False
# NP_P column has issues. if it is all NaN, we will drop it
if data["NP_P"].isnull().all():
    data = data.drop("NP_P", axis=1)
    print("NP_P column is all NaN for {}, so we drop it".format(csv_file))
    NP_P_remove = True
# NP column has issues. if it is all NaN, we will drop it
if data["NP"].isnull().all():
    data = data.drop("NP", axis=1)
    print("NP column is all NaN for {}, so we drop it".format(csv_file))
    NP_remove = True
# FlexE column has issues. if it is all NaN, we will drop it
if data["FlexE"].isnull().all():
    data = data.drop("FlexE", axis=1)
    print("FlexE column is all NaN for {}, so we drop it".format(csv_file))
    FlexE_remove = True

# if every value in NP_P is the same, we will drop it
if not NP_P_remove and data["NP_P"].nunique() == 1:
    data = data.drop("NP_P", axis=1)
    print("NP_P column has only one unique value for {}, so we drop it".format(csv_file))
# if every value in NP is the same, we will drop it
if not NP_remove and data["NP"].nunique() == 1:
    data = data.drop("NP", axis=1)
    print("NP column has only one unique value for {}, so we drop it".format(csv_file))
# if every value in FlexE is the same, we will drop it
if not FlexE_remove and data["FlexE"].nunique() == 1:
    data = data.drop("FlexE", axis=1)
    print("FlexE column has only one unique value for {}, so we drop it".format(csv_file))


print(data.head())
# there are in total 31 plots, we need to set a big figure size
fig, axes = plt.subplots(6, 6, figsize=(30, 25))
model_list = data.index
# iterate over all possible columns
count = 0
for column in data.columns:
    score_by_target = {}
    score_by_group = {}
    first_score_by_group = {}
    best_score_by_group = {}
    score = data[column]
    # check if model_list and score have the same length
    assert len(model_list) == len(score)
    for model, value in zip(model_list, score):
        # parse model
        # T0208s1TS147_4-D1 or T2235TS262_1-D1 or T0218TS312_4
        target_info = model.split("TS")[0]
        if target_info not in score_by_target:
            score_by_target[target_info] = {}
        group_info = model.split("TS")[1]
        if use_domain == True:
            # print(group_info)
            group_info = group_info.split("-")
            # print(group_info)
            domain = group_info[1]
            # print(domain)
            group_info = group_info[0]

            group = group_info.split("_")[0]
            submission_id = group_info.split("_")[1]
            if group not in score_by_target[target_info]:
                score_by_target[target_info][group] = []
            score_by_target[target_info][group].append(value)
            if submission_id == "1":
                if group not in first_score_by_group:
                    first_score_by_group[group] = []
                first_score_by_group[group].append(value)
        else:
            group = group_info.split("_")[0]
            submission_id = group_info.split("_")[1]
            if group not in score_by_target[target_info]:
                score_by_target[target_info][group] = []
            score_by_target[target_info][group].append(value)
            if submission_id == "1":
                if group not in first_score_by_group:
                    first_score_by_group[group] = []
                first_score_by_group[group].append(value)

    for target in score_by_target:
        for group in score_by_target[target]:
            if group not in best_score_by_group:
                best_score_by_group[group] = []
            best_score_by_group[group].append(
                max(score_by_target[target][group]))

    first_accumulate_score_by_group = {}
    # print(first_score_by_group)
    best_accumulate_score_by_group = {}
    for group in first_score_by_group:
        first_accumulate_score_by_group[group] = sum(
            first_score_by_group[group])
    for group in best_score_by_group:
        best_accumulate_score_by_group[group] = sum(best_score_by_group[group])

    # print("First Accumulate Score by Group")
    # print(first_accumulate_score_by_group)
    # print("Best Accumulate Score by Group")
    # print(best_accumulate_score_by_group)
    sorted_first_accumulate_score_by_group = dict(
        sorted(first_accumulate_score_by_group.items(), key=lambda item: item[1], reverse=True))
    sorted_best_accumulate_score_by_group = dict(
        sorted(best_accumulate_score_by_group.items(), key=lambda item: item[1], reverse=True))

    # first_x = list(sorted_first_accumulate_score_by_group.keys())
    # first_y = list(sorted_first_accumulate_score_by_group.values())

    # I am too lazy :(

    if score_type == "first":
        best_x = list(sorted_first_accumulate_score_by_group.keys())
        best_y = list(sorted_first_accumulate_score_by_group.values())
    elif score_type == "best":
        best_x = list(sorted_best_accumulate_score_by_group.keys())
        best_y = list(sorted_best_accumulate_score_by_group.values())

    # print(best_x.__len__())
    # print(best_y.__len__())

    # plot the values in best_accumulate_score_by_group, use boxplot
    # axes[count // 6, count % 6].boxplot(first_y, positions=first_x, patch_artist=True)
    # axes[count // 6, count % 6].boxplot(best_y, positions=best_x, patch_artist=True)

    # add scatter points on the boxplot
    scatter_x = np.full_like(best_x, 1, dtype=int)
    # print(scatter_x)
    axes[count // 6, count %
         6].scatter(scatter_x, best_y, color="red", marker="o")
    axes[count // 6, count %
         6].boxplot(best_y, positions=[1])
    # axes[count // 6, count % 6].set_title(column)
    axes[count // 6, count % 6].set_xticklabels(column, rotation=45)
    axes[count // 6, count % 6].set_ylabel("Accumulate Score")
    count += 1
# remove the empty plots
for i in range(count, 36):
    fig.delaxes(axes.flatten()[i])
# set title for the whole figure
if use_domain:
    fig.suptitle(
        "Accumulate {} domain score by group for all targets".format(score_type), fontsize=20)
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # save some space for the title
    plt.savefig("./accumulate_{}_domain_score.png".format(score_type), dpi=300)
else:
    fig.suptitle(
        "Accumulate {} whole score by group for all targets".format(score_type), fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # save some space for the title
    plt.savefig("./accumulate_{}_whole_score.png".format(score_type), dpi=300)
