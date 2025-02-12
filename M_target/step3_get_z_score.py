import os
import sys
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
# three types of input: pp, pn, all
parser.add_argument('--type', type=str, default="all")
args = parser.parse_args()
type = args.type
input_dir = "/home2/s439906/data/CASP16/hybrid_Oct_10/"
if type == "pp":
    output = "./step3_pp/"
    scores = ["prot_per_interface_qs_best",
              "prot_per_interface_ics_trimmed",
              "prot_per_interface_ips_trimmed",
              "lDDT",
              "TMscore",
              "GlobDockQ"]
elif type == "pn":
    output = "./step3_pn/"
    scores = ["prot_nucl_per_interface_qs_best",
              "prot_nucl_per_interface_ics_trimmed",
              "prot_nucl_per_interface_ips_trimmed",
              "lDDT",
              "TMscore",
              "GlobDockQ"]
elif type == "all":
    output = "./step3_all/"
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
if not os.path.exists(output):
    os.makedirs(output)
pp_score_dir = "./step1_pp/"
pn_score_dir = "./step1_pn/"
target_score_dir = "./step2/"
pp_scores = ["prot_per_interface_qs_best",
             "prot_per_interface_ics_trimmed",
             "prot_per_interface_ips_trimmed"]
pn_scores = ["prot_nucl_per_interface_qs_best",
             "prot_nucl_per_interface_ics_trimmed",
             "prot_nucl_per_interface_ips_trimmed"]
target_scores = ["lDDT", "TMscore", "GlobDockQ"]
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


targets = [txt.split('.')[0]
           for txt in os.listdir(
    input_dir) if txt.endswith(".txt") and "M0" not in txt]
# remove M1268 and M1297 from targets
targets.remove("M1268")
targets.remove("M1297")
# sort targets
targets.sort()
print(targets, len(targets))
no_pp_targets = ["M1212", "M1221", "M1224", "M1276", "M1282"]


all_df = pd.DataFrame()

for target in targets:
    target_df = pd.DataFrame()
    z_score_df = pd.DataFrame()
    for score in scores:
        if score in pp_scores:
            file = pp_score_dir + target + "_" + score + ".csv"
        elif score in pn_scores:
            file = pn_score_dir + target + "_" + score + ".csv"
        elif score in target_scores:
            file = target_score_dir + target + "_" + score + ".csv"
        df_score = pd.read_csv(file)
        df_score.set_index("model", inplace=True)
        df_score.columns = [score]  # set column name as score name
        target_df = pd.concat([target_df, df_score], axis=1)

        initial_z = (df_score - df_score.mean()) / df_score.std(ddof=0)
        new_z_score = pd.DataFrame(
            index=df_score.index, columns=df_score.columns)
        filtered_data = df_score[initial_z[score] >= -2]
        # breakpoint()
        new_mean = filtered_data.mean(skipna=True)
        new_std = filtered_data.std(skipna=True, ddof=0)
        new_z_score[score] = (df_score - new_mean) / new_std
        new_z_score = new_z_score.fillna(-2.0)
        new_z_score = new_z_score.where(new_z_score > -2, -2)
        if score in pp_scores:
            new_z_score = new_z_score * pp_weight_dict[target]
        elif score in pn_scores:
            new_z_score = new_z_score * pn_weight_dict[target]
        z_score_df = pd.concat([z_score_df, new_z_score], axis=1)

    if type == "pn":
        z_score_weight = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    elif type == "pp":
        z_score_weight = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    elif type == "all":
        if target in no_pp_targets:
            z_score_weight = [0, 0, 0, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
        else:
            z_score_weight = [1/12, 1/12, 1/12,
                              1/12, 1/12, 1/12, 1/6, 1/6, 1/6]
    # multiply each column with the weight
    z_score_df = z_score_df * z_score_weight

    print(target_df)
    print(z_score_df)
    if type == "pp" and target in no_pp_targets:
        continue
    target_df.to_csv(output + target + "_raw"+".csv")
    z_score_df.to_csv(output + target + "_zscore"+".csv")

    # initial_z = (target_df - target_df.mean()) / target_df.std(ddof=0)
    # new_z_score = pd.DataFrame(
    #     index=target_df.index, columns=target_df.columns)
    # for column in target_df.columns:
    #     filtered_data = target_df[column][initial_z[column] >= -2]
    #     new_mean = filtered_data.mean(skipna=True)
    #     new_std = filtered_data.std(skipna=True, ddof=0)
    #     new_z_score[column] = (target_df[column] - new_mean) / new_std
    # new_z_score = new_z_score.fillna(-2.0)
    # new_z_score = new_z_score.where(new_z_score > -2, -2)
    # new_z_score.to_csv(output + target + "_zscore"+".csv")
