import os
import sys
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
# three types of input: pp, pn, all
parser.add_argument('--type', type=str, default="all")
parser.add_argument('--model', type=str, default="best")
args = parser.parse_args()
type = args.type
model = args.model

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
pp_score_dir = "./step1_pp/"
pn_score_dir = "./step1_pn/"
pp_weight_dict = {}
pn_weight_dict = {}
with open(pp_score_dir + "EU_weight.txt", "r") as f:
    for line in f:
        line = line.split()
        pp_weight_dict[line[0]] = int(line[1]) ** (1/3)
with open(pn_score_dir + "EU_weight.txt", "r") as f:
    for line in f:
        line = line.split()
        pn_weight_dict[line[0]] = int(line[1]) ** (1/3)
data = pd.DataFrame()
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
        columns={"weighted_sum": target.split("_zscore.csv")[0]})
    data = pd.concat([data, z_score], axis=1)
    # print(z_score)
    # breakpoint()
print(data)
