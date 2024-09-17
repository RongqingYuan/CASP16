import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
import os
import json

score_path = "./group_by_target/"
file_path = "./bootstrap/"
score_path = "./group_by_target_EU/"
file_path = "./bootstrap_EU/"
# measure = sys.argv[1]
measure = "GDT_TS"
mode = "hard"
# mode = "medium"
# mode = "easy"
mode = "all"

model = "first"
model = "best"
p_value_threshold = 0.005
bootstrap_rounds = 1000

measures = ['GDT_TS', 'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
            'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
            'SphGr',
            'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
            'MP_rotout', 'MP_ramout', 'MP_ramfv', 'reLLG_lddt', 'reLLG_const']
hard_group = [
    "T1207-D1",
    "T1210-D1",
    "T1210-D2",
    "T1220s1-D1",
    "T1220s1-D2",
    "T1226-D1",
    "T1228v1-D3",
    "T1228v1-D4",
    "T1228v2-D3",
    "T1228v2-D4",
    "T1239v1-D4",
    "T1239v2-D4",
    "T1271s1-D1",
    "T1271s3-D1",
    "T1271s8-D1",
]

medium_group = [
    "T1210-D3",
    "T1212-D1",
    "T1218-D1",
    "T1218-D2",
    "T1227s1-D1",
    "T1228v1-D1",
    "T1228v2-D1",
    "T1230s1-D1",
    "T1237-D1",
    "T1239v1-D1",
    "T1239v1-D3",
    "T1239v2-D1",
    "T1239v2-D3",
    "T1243-D1",
    "T1244s1-D1",
    "T1244s2-D1",
    "T1245s2-D1",
    "T1249v1-D1",
    "T1257-D3",
    "T1266-D1",
    "T1270-D1",
    "T1270-D2",
    "T1267s1-D1",
    "T1267s1-D2",
    "T1267s2-D1",
    "T1269-D1",
    "T1269-D2",
    "T1269-D3",
    "T1271s2-D1",
    "T1271s4-D1",
    "T1271s5-D1",
    "T1271s5-D2",
    "T1271s7-D1",
    "T1271s8-D2",
    "T1272s2-D1",
    "T1272s6-D1",
    "T1272s8-D1",
    "T1272s9-D1",
    "T1279-D2",
    "T1284-D1",
    "T1295-D1",
    "T1295-D3",
    "T1298-D1",
    "T1298-D2",
]

easy_group = [
    "T1201-D1",
    "T1201-D2",
    "T1206-D1",
    "T1208s1-D1",
    "T1208s2-D1",
    "T1218-D3",
    "T1228v1-D2",
    "T1228v2-D2",
    "T1231-D1",
    "T1234-D1",
    "T1235-D1",
    "T1239v1-D2",
    "T1239v2-D2",
    "T1240-D1",
    "T1240-D2",
    "T1245s1-D1",
    "T1246-D1",
    "T1257-D1",
    "T1257-D2",
    "T1259-D1",
    "T1271s6-D1",
    "T1274-D1",
    "T1276-D1",
    "T1278-D1",
    "T1279-D1",
    "T1280-D1",
    "T1292-D1",
    "T1294v1-D1",
    "T1294v2-D1",
    "T1295-D2",
    # "T1214",
]

dict_file = "{}_{}_{}_p={}_ranking_step18.txt".format(
    measure, model, mode, p_value_threshold)
dict_obj = None

with open(file_path + dict_file, "r") as f:
    for line in f:
        line = line.strip()
        # breakpoint()
        dict_obj = eval(line)

win_matrix_file = "win_matrix_bootstrap_{}_{}_{}_p={}_n={}_step18.npy".format(
    measure, model, mode, p_value_threshold, str(bootstrap_rounds))

win_matrix = np.load(file_path + win_matrix_file)
# breakpoint()

top_n = 25
# the id is top n in the dict_obj
top_n_id = list(dict_obj.keys())[:top_n]

# the win_matrix is the top n by n in the win_matrix
win_matrix_top_n = win_matrix[:top_n, :top_n]

total_count = top_n * (top_n - 1) / 2 * bootstrap_rounds

valid_count = 0
for i in range(top_n):
    for j in range(top_n):
        valid_count += win_matrix_top_n[i, j]

print(valid_count, total_count, valid_count / total_count)
print()

bootstrap_stats = {}
top_n = 25
for measure in measures:
    dict_file = "{}_{}_{}_p={}_ranking_step18.txt".format(
        measure, model, mode, p_value_threshold)
    dict_obj = None

    with open(file_path + dict_file, "r") as f:
        for line in f:
            line = line.strip()
            # breakpoint()
            dict_obj = eval(line)

    win_matrix_file = "win_matrix_bootstrap_{}_{}_{}_p={}_n={}_step18.npy".format(
        measure, model, mode, p_value_threshold, str(bootstrap_rounds))

    win_matrix = np.load(file_path + win_matrix_file)
    # breakpoint()

    # the id is top n in the dict_obj
    top_n_id = list(dict_obj.keys())[:top_n]

    # the win_matrix is the top n by n in the win_matrix
    win_matrix_top_n = win_matrix[:top_n, :top_n]

    total_count = top_n * (top_n - 1) / 2 * bootstrap_rounds
    valid_count = 0
    for i in range(top_n):
        for j in range(top_n):
            valid_count += win_matrix_top_n[i, j]

    print(valid_count, total_count, valid_count /
          total_count, measure, model, mode)
    bootstrap_stats[measure] = valid_count / total_count

# sort the bootstrap_stats and plot
bootstrap_stats = dict(sorted(bootstrap_stats.items(),
                              key=lambda x: x[1], reverse=True))
print(bootstrap_stats)
if not os.path.exists("./bootstrap_analysis_EU/"):
    os.makedirs("./bootstrap_analysis_EU/")
# plot the bootstrap_stats
plt.figure(figsize=(12.5, 7.5))
plt.bar(bootstrap_stats.keys(), bootstrap_stats.values())
plt.xticks(rotation=45, ha='right')
plt.ylabel("Winning Rate")
plt.title("Bootstrap Winning Rate for {} {} top {} groups".format(
    model, mode, top_n))

plt.savefig("./bootstrap_analysis_EU/" + "bootstrap_winning_rate_{}_{}_{}_p={}_top_{}.png".format(
    model, mode, bootstrap_rounds, p_value_threshold, top_n), dpi=300)
