import os
import sys
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# three types of input: pp, pn, all
parser.add_argument('--type', type=str, default="all")
parser.add_argument('--model', type=str, default="best")
parser.add_argument('--output', type=str, default="./step5/")
args = parser.parse_args()
type = args.type
model = args.model
output = args.output
if not os.path.exists(output):
    os.makedirs(output)
all_scores = ["prot_per_interface_qs_best",
              "prot_per_interface_ics_trimmed",
              "prot_per_interface_ips_trimmed",
              "prot_nucl_per_interface_qs_best",
              "prot_nucl_per_interface_ics_trimmed",
              "prot_nucl_per_interface_ips_trimmed",
              "lDDT",
              "TMscore",
              "GlobDockQ"]
if type == "pp":
    input = "./step3_pp/"
    scores = ["prot_per_interface_qs_best",
              "prot_per_interface_ics_trimmed",
              "prot_per_interface_ips_trimmed",
              "lDDT",
              "TMscore",
              "GlobDockQ"]
elif type == "pn":
    input = "./step3_pn/"
    scores = ["prot_nucl_per_interface_qs_best",
              "prot_nucl_per_interface_ics_trimmed",
              "prot_nucl_per_interface_ips_trimmed",
              "lDDT",
              "TMscore",
              "GlobDockQ"]
elif type == "all":
    input = "./step3_all/"
    scores = ["prot_per_interface_qs_best",
              "prot_per_interface_ics_trimmed",
              "prot_per_interface_ips_trimmed",
              "prot_nucl_per_interface_qs_best",
              "prot_nucl_per_interface_ics_trimmed",
              "prot_nucl_per_interface_ips_trimmed",
              "lDDT",
              "TMscore",
              "GlobDockQ"]

else:
    print("type must be pp, pn or all")
    sys.exit()
targets = [csv for csv in os.listdir(input) if csv.endswith("_zscore.csv")]
targets.sort()
print(targets, len(targets))
no_pp_targets = ["M1212", "M1221", "M1224", "M1276", "M1282"]

data = pd.DataFrame()
pp_qs_best_df = pd.DataFrame()
pp_ics_df = pd.DataFrame()
pp_ips_df = pd.DataFrame()
pn_qs_best_df = pd.DataFrame()
pn_ics_df = pd.DataFrame()
pn_ips_df = pd.DataFrame()
lDDT_df = pd.DataFrame()
TMscore_df = pd.DataFrame()
GlobDockQ_df = pd.DataFrame()
score_df = pd.DataFrame()
score_dict = {}

for target in targets:
    data_tmp = pd.read_csv(input + target, index_col=0)
    data_tmp['weighted_sum'] = data_tmp.sum(axis=1)
    data_tmp.index = data_tmp.index.str.extract(
        r'(\w+)TS(\w+)_(\w+)').apply(lambda x: (f"{x[0]}", f"TS{x[1]}", x[2][0]), axis=1)
    data_tmp.index = pd.MultiIndex.from_tuples(
        data_tmp.index, names=['target', 'group', 'model'])
    # no more duplicated index to worry about
    # if data_tmp.index.duplicated().any():
    #     print(f"Duplicated index in {csv_file}")
    #     data_tmp = data_tmp.groupby(
    #         level=data_tmp.index.names).max()
    if model == "best":
        data_tmp = data_tmp.loc[(slice(None), slice(None), [
            "1", "2", "3", "4", "5"]), :]
    elif model == "first":
        data_tmp = data_tmp.loc[(slice(None), slice(None),
                                 "1"), :]
    elif model == "sixth":
        data_tmp = data_tmp.loc[(slice(None), slice(None),
                                 "6"), :]
    else:
        print("model must be best, first or sixth")
        sys.exit()
    grouped = data_tmp.groupby(["group"])
    # take the max weighted_sum as well as the corresponding components
    z_score_max = data_tmp.loc[grouped['weighted_sum'].idxmax()]
    z_score_max = z_score_max.groupby(["group"])
    z_score = pd.DataFrame(z_score_max['weighted_sum'].max())
    z_score = z_score.rename(
        columns={"weighted_sum": target.split("_")[0]})
    data = pd.concat([data, z_score], axis=1)

    # pp_qs_best_df = pd.DataFrame(
    #     z_score_max["prot_per_interface_qs_best"].max())
    # pp_qs_best_df = pp_qs_best_df.rename(
    #     columns={"prot_per_interface_qs_best": target.split("_")[0]})
    # pp_ics_df = pd.DataFrame(
    #     z_score_max["prot_per_interface_ics_trimmed"].max())
    # pp_ics_df = pp_ics_df.rename(
    #     columns={"prot_per_interface_ics_trimmed": target.split("_")[0]})
    # pp_ips_df = pd.DataFrame(
    #     z_score_max["prot_per_interface_ips_trimmed"].max())
    # pp_ips_df = pp_ips_df.rename(
    #     columns={"prot_per_interface_ips_trimmed": target.split("_")[0]})
    # pn_qs_best_df = pd.DataFrame(
    #     z_score_max["prot_nucl_per_interface_qs_best"].max())
    # pn_qs_best_df = pn_qs_best_df.rename(
    #     columns={"prot_nucl_per_interface_qs_best": target.split("_")[0]})
    # pn_ics_df = pd.DataFrame(
    #     z_score_max["prot_nucl_per_interface_ics_trimmed"].max())
    # pn_ics_df = pn_ics_df.rename(
    #     columns={"prot_nucl_per_interface_ics_trimmed": target.split("_")[0]})
    # pn_ips_df = pd.DataFrame(
    #     z_score_max["prot_nucl_per_interface_ips_trimmed"].max())
    # pn_ips_df = pn_ips_df.rename(
    #     columns={"prot_nucl_per_interface_ips_trimmed": target.split("_")[0]})
    # lDDT_df = pd.DataFrame(
    #     z_score_max["lDDT"].max())
    # lDDT_df = lDDT_df.rename(
    #     columns={"lDDT": target.split("_")[0]})
    # TMscore_df = pd.DataFrame(
    #     z_score_max["TMscore"].max())
    # TMscore_df = TMscore_df.rename(
    #     columns={"TMscore": target.split("_")[0]})
    # GlobDockQ_df = pd.DataFrame(
    #     z_score_max["GlobDockQ"].max())
    # GlobDockQ_df = GlobDockQ_df.rename(
    #     columns={"GlobDockQ": target.split("_")[0]})
    # # print(pp_qs_best_df)

    if type == "pp":
        if target.split("_")[0] in no_pp_targets:
            continue
        else:
            pp_qs_best = pd.DataFrame(
                z_score_max["prot_per_interface_qs_best"].max())
            pp_qs_best = pp_qs_best.rename(
                columns={"prot_per_interface_qs_best": target.split("_")[0]})
            pp_qs_best_df = pd.concat([pp_qs_best_df, pp_qs_best], axis=1)
            pp_ics = pd.DataFrame(
                z_score_max["prot_per_interface_ics_trimmed"].max())
            pp_ics = pp_ics.rename(
                columns={"prot_per_interface_ics_trimmed": target.split("_")[0]})
            pp_ics_df = pd.concat([pp_ics_df, pp_ics], axis=1)
            pp_ips = pd.DataFrame(
                z_score_max["prot_per_interface_ips_trimmed"].max())
            pp_ips = pp_ips.rename(
                columns={"prot_per_interface_ips_trimmed": target.split("_")[0]})
            pp_ips_df = pd.concat([pp_ips_df, pp_ips], axis=1)
            lDDT = pd.DataFrame(
                z_score_max["lDDT"].max())
            lDDT = lDDT.rename(
                columns={"lDDT": target.split("_")[0]})
            lDDT_df = pd.concat([lDDT_df, lDDT], axis=1)
            TMscore = pd.DataFrame(
                z_score_max["TMscore"].max())
            TMscore = TMscore.rename(
                columns={"TMscore": target.split("_")[0]})
            TMscore_df = pd.concat([TMscore_df, TMscore], axis=1)
            GlobDockQ = pd.DataFrame(
                z_score_max["GlobDockQ"].max())
            GlobDockQ = GlobDockQ.rename(
                columns={"GlobDockQ": target.split("_")[0]})
            GlobDockQ_df = pd.concat([GlobDockQ_df, GlobDockQ], axis=1)

    elif type == "pn":
        pn_qs_best = pd.DataFrame(
            z_score_max["prot_nucl_per_interface_qs_best"].max())
        pn_qs_best = pn_qs_best.rename(
            columns={"prot_nucl_per_interface_qs_best": target.split("_")[0]})
        pn_qs_best_df = pd.concat([pn_qs_best_df, pn_qs_best], axis=1)
        pn_ics = pd.DataFrame(
            z_score_max["prot_nucl_per_interface_ics_trimmed"].max())
        pn_ics = pn_ics.rename(
            columns={"prot_nucl_per_interface_ics_trimmed": target.split("_")[0]})
        pn_ics_df = pd.concat([pn_ics_df, pn_ics], axis=1)
        pn_ips = pd.DataFrame(
            z_score_max["prot_nucl_per_interface_ips_trimmed"].max())
        pn_ips = pn_ips.rename(
            columns={"prot_nucl_per_interface_ips_trimmed": target.split("_")[0]})
        pn_ips_df = pd.concat([pn_ips_df, pn_ips], axis=1)
        lDDT = pd.DataFrame(
            z_score_max["lDDT"].max())
        lDDT = lDDT.rename(
            columns={"lDDT": target.split("_")[0]})
        lDDT_df = pd.concat([lDDT_df, lDDT], axis=1)
        TMscore = pd.DataFrame(
            z_score_max["TMscore"].max())
        TMscore = TMscore.rename(
            columns={"TMscore": target.split("_")[0]})
        TMscore_df = pd.concat([TMscore_df, TMscore], axis=1)
        GlobDockQ = pd.DataFrame(
            z_score_max["GlobDockQ"].max())
        GlobDockQ = GlobDockQ.rename(
            columns={"GlobDockQ": target.split("_")[0]})
        GlobDockQ_df = pd.concat([GlobDockQ_df, GlobDockQ], axis=1)

    elif type == "all":
        pp_qs_best = pd.DataFrame(
            z_score_max["prot_per_interface_qs_best"].max())
        pp_qs_best = pp_qs_best.rename(
            columns={"prot_per_interface_qs_best": target.split("_")[0]})
        pp_qs_best_df = pd.concat([pp_qs_best_df, pp_qs_best], axis=1)
        pp_ics = pd.DataFrame(
            z_score_max["prot_per_interface_ics_trimmed"].max())
        pp_ics = pp_ics.rename(
            columns={"prot_per_interface_ics_trimmed": target.split("_")[0]})
        pp_ics_df = pd.concat([pp_ics_df, pp_ics], axis=1)
        pp_ips = pd.DataFrame(
            z_score_max["prot_per_interface_ips_trimmed"].max())
        pp_ips = pp_ips.rename(
            columns={"prot_per_interface_ips_trimmed": target.split("_")[0]})
        pp_ips_df = pd.concat([pp_ips_df, pp_ips], axis=1)
        pn_qs_best = pd.DataFrame(
            z_score_max["prot_nucl_per_interface_qs_best"].max())
        pn_qs_best = pn_qs_best.rename(
            columns={"prot_nucl_per_interface_qs_best": target.split("_")[0]})
        pn_qs_best_df = pd.concat([pn_qs_best_df, pn_qs_best], axis=1)
        pn_ics = pd.DataFrame(
            z_score_max["prot_nucl_per_interface_ics_trimmed"].max())
        pn_ics = pn_ics.rename(
            columns={"prot_nucl_per_interface_ics_trimmed": target.split("_")[0]})
        pn_ics_df = pd.concat([pn_ics_df, pn_ics], axis=1)
        pn_ips = pd.DataFrame(
            z_score_max["prot_nucl_per_interface_ips_trimmed"].max())
        pn_ips = pn_ips.rename(
            columns={"prot_nucl_per_interface_ips_trimmed": target.split("_")[0]})
        pn_ips_df = pd.concat([pn_ips_df, pn_ips], axis=1)
        lDDT = pd.DataFrame(
            z_score_max["lDDT"].max())
        lDDT = lDDT.rename(
            columns={"lDDT": target.split("_")[0]})
        lDDT_df = pd.concat([lDDT_df, lDDT], axis=1)
        TMscore = pd.DataFrame(
            z_score_max["TMscore"].max())
        TMscore = TMscore.rename(
            columns={"TMscore": target.split("_")[0]})
        TMscore_df = pd.concat([TMscore_df, TMscore], axis=1)
        GlobDockQ = pd.DataFrame(
            z_score_max["GlobDockQ"].max())
        GlobDockQ = GlobDockQ.rename(
            columns={"GlobDockQ": target.split("_")[0]})
        GlobDockQ_df = pd.concat([GlobDockQ_df, GlobDockQ], axis=1)
# print(pp_qs_best_df.shape)
# print(pp_ics_df.shape)
# print(pp_ips_df.shape)
# print(pn_qs_best_df.shape)
# print(pn_ics_df.shape)
# print(pn_ips_df.shape)
# print(lDDT_df.shape)
# print(TMscore_df.shape)
# print(GlobDockQ_df.shape)
top_n = 20
if type == "pp":
    score_dict["pp_qs_best"] = pp_qs_best_df
    score_dict["pp_ics"] = pp_ics_df
    score_dict["pp_ips"] = pp_ips_df
    score_dict["lDDT"] = lDDT_df
    score_dict["TMscore"] = TMscore_df
    score_dict["GlobDockQ"] = GlobDockQ_df
    for key in score_dict.keys():
        score = score_dict[key]
        score_sum = score.sum(axis=1)
        score_df[key] = score_sum
    score_df['sum'] = score_df.sum(axis=1)
    score_df = score_df.sort_values(by='sum', ascending=False)
    largest_value = score_df['sum'].max()
    print(score_df)
    score_df = score_df.head(top_n)
    groups = score_df.index.to_list()
    pp_qs_best = score_df["pp_qs_best"].to_list()
    pp_ics = score_df["pp_ics"].to_list()
    pp_ips = score_df["pp_ips"].to_list()
    lDDT = score_df["lDDT"].to_list()
    TMscore = score_df["TMscore"].to_list()
    GlobDockQ = score_df["GlobDockQ"].to_list()
    all_scores = [pp_qs_best, pp_ics, pp_ips, lDDT, TMscore, GlobDockQ]
    positive_series = [[val if val > 0 else 0 for val in series]
                       for series in all_scores]
    negative_series = [[val if val < 0 else 0 for val in series]
                       for series in all_scores]
    fig = plt.figure(figsize=(7, 6), dpi=300)

    bottom_positive = np.zeros(len(groups))
    bottom_negative = np.zeros(len(groups))

    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

    for i, series in enumerate(positive_series):
        plt.bar(groups, series, bottom=bottom_positive,
                color=colors[i], alpha=0.9)
        bottom_positive += series

    for i, series in enumerate(negative_series):
        plt.bar(groups, series, bottom=bottom_negative,
                color=colors[i], alpha=0.5)
        bottom_negative += series

    plt.title(
        f'models: {model}, top_n: {top_n}, type: {type}', fontsize=12, pad=20)
    plt.legend(score_dict.keys(), loc='upper right', fontsize=10)

    plt.xticks(rotation=90, fontsize=12)
    # y range : -5 to 95
    # if model == "best":
    #     plt.ylim(-5, 105)
    # elif model == "first":
    #     plt.ylim(-10, 75)
    plt.ylim(-10, largest_value+5)
    plt.ylabel('cumulative z-score', fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    # plt.show()
    fig.savefig(output + f"{model}_top_n={top_n}_{type}.png")

elif type == "pn":
    score_dict["pn_qs_best"] = pn_qs_best_df
    score_dict["pn_ics"] = pn_ics_df
    score_dict["pn_ips"] = pn_ips_df
    score_dict["lDDT"] = lDDT_df
    score_dict["TMscore"] = TMscore_df
    score_dict["GlobDockQ"] = GlobDockQ_df
    for key in score_dict.keys():
        score = score_dict[key]
        score_sum = score.sum(axis=1)
        score_df[key] = score_sum
    score_df['sum'] = score_df.sum(axis=1)
    score_df = score_df.sort_values(by='sum', ascending=False)
    largest_value = score_df['sum'].max()
    print(score_df)
    score_df = score_df.head(top_n)
    groups = score_df.index.to_list()
    pn_qs_best = score_df["pn_qs_best"].to_list()
    pn_ics = score_df["pn_ics"].to_list()
    pn_ips = score_df["pn_ips"].to_list()
    lDDT = score_df["lDDT"].to_list()
    TMscore = score_df["TMscore"].to_list()
    GlobDockQ = score_df["GlobDockQ"].to_list()
    all_scores = [pn_qs_best, pn_ics, pn_ips, lDDT, TMscore, GlobDockQ]
    positive_series = [[val if val > 0 else 0 for val in series]
                       for series in all_scores]
    negative_series = [[val if val < 0 else 0 for val in series]
                       for series in all_scores]
    fig = plt.figure(figsize=(7, 6), dpi=300)

    bottom_positive = np.zeros(len(groups))
    bottom_negative = np.zeros(len(groups))

    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

    for i, series in enumerate(positive_series):
        plt.bar(groups, series, bottom=bottom_positive,
                color=colors[i], alpha=0.9)
        bottom_positive += series

    for i, series in enumerate(negative_series):
        plt.bar(groups, series, bottom=bottom_negative,
                color=colors[i], alpha=0.5)
        bottom_negative += series

    plt.title(
        f'models: {model}, top_n: {top_n}, type: {type}', fontsize=12, pad=20)

    plt.legend(score_dict.keys(), loc='upper right', fontsize=10)

    plt.xticks(rotation=90, fontsize=12)
    # y range : -5 to 95
    # if model == "best":
    #     plt.ylim(-5, 105)
    # elif model == "first":
    #     plt.ylim(-10, 75)
    plt.ylim(-10, largest_value+5)
    plt.ylabel('cumulative z-score', fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    # plt.show()
    fig.savefig(output + f"{model}_top_n={top_n}_{type}.png")

elif type == "all":
    score_dict["pp_qs_best"] = pp_qs_best_df
    score_dict["pp_ics"] = pp_ics_df
    score_dict["pp_ips"] = pp_ips_df
    score_dict["pn_qs_best"] = pn_qs_best_df
    score_dict["pn_ics"] = pn_ics_df
    score_dict["pn_ips"] = pn_ips_df
    score_dict["lDDT"] = lDDT_df
    score_dict["TMscore"] = TMscore_df
    score_dict["GlobDockQ"] = GlobDockQ_df

    for key in score_dict.keys():
        score = score_dict[key]
        score_sum = score.sum(axis=1)
        score_df[key] = score_sum
    score_df['sum'] = score_df.sum(axis=1)
    score_df = score_df.sort_values(by='sum', ascending=False)
    largest_value = score_df['sum'].max()
    print(score_df)
    score_df = score_df.head(top_n)
    groups = score_df.index.to_list()
    pp_qs_best = score_df["pp_qs_best"].to_list()
    pp_ics = score_df["pp_ics"].to_list()
    pp_ips = score_df["pp_ips"].to_list()
    pn_qs_best = score_df["pn_qs_best"].to_list()
    pn_ics = score_df["pn_ics"].to_list()
    pn_ips = score_df["pn_ips"].to_list()
    lDDT = score_df["lDDT"].to_list()
    TMscore = score_df["TMscore"].to_list()
    GlobDockQ = score_df["GlobDockQ"].to_list()
    all_scores = [pp_qs_best, pp_ics, pp_ips, pn_qs_best,
                  pn_ics, pn_ips, lDDT, TMscore, GlobDockQ]
    positive_series = [[val if val > 0 else 0 for val in series]
                       for series in all_scores]
    negative_series = [[val if val < 0 else 0 for val in series]
                       for series in all_scores]
    fig = plt.figure(figsize=(7, 6), dpi=300)

    bottom_positive = np.zeros(len(groups))
    bottom_negative = np.zeros(len(groups))

    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']

    for i, series in enumerate(positive_series):
        plt.bar(groups, series, bottom=bottom_positive,
                color=colors[i], alpha=0.9)
        bottom_positive += series

    for i, series in enumerate(negative_series):
        plt.bar(groups, series, bottom=bottom_negative,
                color=colors[i], alpha=0.5)
        bottom_negative += series

    plt.title(
        f'models: {model}, top_n: {top_n}, type: {type}', fontsize=12, pad=20)

    plt.legend(score_dict.keys(), loc='upper right', fontsize=10)

    plt.xticks(rotation=90, fontsize=12)
    # y range : -5 to 95
    # if model == "best":
    #     plt.ylim(-5, 105)
    # elif model == "first":
    #     plt.ylim(-10, 75)
    plt.ylim(-10, largest_value+5)
    plt.ylabel('cumulative z-score', fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    # plt.show()
    fig.savefig(output + f"{model}_top_n={top_n}_{type}.png")
