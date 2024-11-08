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
out_path = "./bootstrap_analysis_EU/"
# measure = sys.argv[1]
measure = "GDT_TS"


measure = "GDT_HA"
# measure = "GDC_SC"
# measure = "AL0_P"
# measure = "SphGr"
# measure = "CAD_AA"
# measure = "QSE"
# measure = "MolPrb_Score"
# measure = "reLLG_const"


mode = "hard"
mode = "medium"
mode = "easy"
mode = "all"
model = "first"
model = "best"


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
measure_type = "CASP15"
equal_weight = False

measure_type = "CASP16"
equal_weight = True

dict_file = "{}_{}_{}_n={}_sum_equal_weight_{}.txt".format(
    measure_type, model, mode,  bootstrap_rounds, equal_weight)
dict_obj = None

with open(file_path + dict_file, "r") as f:
    for line in f:
        line = line.strip()
        dict_obj = eval(line)

win_matrix_file = "win_matrix_bootstrap_{}_{}_{}_n={}_sum_equal_weight_{}.npy".format(
    measure_type, model, mode, bootstrap_rounds, equal_weight)
win_matrix = np.load(file_path + win_matrix_file)


top_n = 25
top_n_id = list(dict_obj.keys())[:top_n]
win_matrix_top_n = win_matrix[:top_n, :top_n]
win_matrix_top_n = win_matrix_top_n / bootstrap_rounds
plt.figure(figsize=(24, 18))
ax = sns.heatmap(win_matrix_top_n, annot=True, fmt=".2f",
                 cmap='Greys', cbar=True, square=True,
                 #  linewidths=1, linecolor='black',
                 )
# set the largest scale bar to bootstrap_rounds
cbar = ax.collections[0].colorbar
# cbar.set_ticks([0, bootstrap_rounds])
# cbar.set_ticklabels([0, bootstrap_rounds])
# also set 0, int(bootstrap_rounds/4), int(bootstrap_rounds/2), int(bootstrap_rounds*3/4), bootstrap_rounds
cbar.set_ticks([0, float(1/4), float(1/2),
                float(3/4), 1])
cbar.set_ticklabels([0, float(1/4), float(1/2),
                     float(3/4), 1])
cbar.ax.tick_params(labelsize=10)
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(2)
ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment='center')
ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center')
plt.xticks(np.arange(top_n), top_n_id, rotation=45, fontsize=10)
plt.yticks(np.arange(top_n), top_n_id, rotation=0, fontsize=10)
if equal_weight:
    plt.title("Bootstrap result of {} score for {} top {} groups, with equal weight".format(
        measure_type, mode, top_n), fontsize=20)
else:
    plt.title("Bootstrap result of {} score for {} top {} groups".format(
        measure_type, mode, top_n), fontsize=20)
plt.savefig(out_path + "win_matrix_bootstrap_{}_{}_{}_n={}_sum_equal_weight_{}_top_{}_sum.png".format(
    measure_type, model, mode, bootstrap_rounds, equal_weight, top_n), dpi=300)
