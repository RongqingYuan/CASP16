import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
colab_file = "/home2/s439906/data/CASP16/ColabFoldBaseline_CASP16scores.csv"
out_path = "./baseline_data_Sep_15_EU/"
raw_out_path = "./baseline_data_Sep_15_EU/raw_data/"


if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(raw_out_path):
    os.makedirs(raw_out_path)


EUs = ["T1201",
       "T1206-D1",
       "T1207-D1",
       "T1208s1-D1",
       "T1208s2-D1",
       "T1210",
       "T1212-D1",
       "T1218-D1",
       "T1218-D2",
       "T1218-D3",
       "T1220s1",
       "T1226-D1",
       "T1227s1-D1",
       "T1228v1-D1", "T1228v1-D1_1", "T1228v2-D1", "T1228v2-D1_1",
       "T1228v1-D2", "T1228v1-D2_1", "T1228v2-D2", "T1228v2-D2_1",
       "T1228v1-D3", "T1228v1-D3_1", "T1228v2-D3", "T1228v2-D3_1",
       "T1228v1-D4", "T1228v1-D4_1", "T1228v2-D4", "T1228v2-D4_1",
       "T1230s1-D1",
       "T1231-D1",
       "T1234-D1",
       "T1235-D1",
       "T1237-D1",
       "T1239v1-D1", "T1239v1-D1_1",
       "T1239v1-D2", "T1239v1-D2_1",
       "T1239v1-D3", "T1239v1-D3_1",
       "T1239v1-D4", "T1239v1-D4_1",
       "T1240-D1",
       "T1240-D2",
       "T1243-D1",
       "T1244s1-D1",
       "T1244s2-D1",
       "T1245s1-D1",
       "T1246-D1",
       "T1249v1-D1",
       "T1257",
       "T1259-D1",
       "T1266-D1",
       "T1267s1-D1",
       "T1267s1-D2",
       "T1267s2-D1",
       "T1269-D1",
       "T1269-D2",
       "T1269-D3",
       "T1270-D1",
       "T1270-D2",
       "T1271s1-D1",
       "T1271s2-D1",
       "T1271s3-D1",
       "T1271s4-D1",
       "T1271s5-D1",
       "T1271s5-D2",
       "T1271s6-D1",
       "T1271s7-D1",
       "T1271s8-D1",
       "T1271s8-D2",
       "T1272s2-D1",
       "T1272s6-D1",
       "T1272s8-D1",
       "T1272s9-D1",
       "T1274-D1",
       "T1276-D1",
       "T1278-D1",
       "T1279-D1",
       "T1279-D2",
       "T1280-D1",
       "T1284-D1",
       "T1292-D1",
       "T1294v1-D1", "T1294v2-D1",
       "T1295-D1",
       "T1295-D2",
       "T1295-D3",
       "T1298-D1",
       "T1298-D2",
       ]


valid_lines = []
EU_dict = {}
header_line = None
with open(colab_file, "r") as f:
    for line in f:
        if line.startswith("Model"):
            valid_lines.append(line)
            header_line = line
            continue
        if line.startswith("T0") or line.startswith("T2"):
            continue
        if line.startswith("T1"):
            model = line.split(",")[0]
            target = model.split("TS")[0]

            length = len(model.split("-"))
            if length == 2:
                domain = "-" + model.split("-")[1]
            elif length == 1:
                domain = ""  # a special notation for the convience of comparison
            EU = target + domain
            if EU in EUs:
                valid_lines.append(line)
                if EU not in EU_dict:
                    EU_dict[EU] = []
                EU_dict[EU].append(line)


with open(out_path + "/ColabFoldBaseline_CASP16scores_curated.csv", "w") as f:
    for line in valid_lines:
        f.write(line)

for EU in EU_dict:
    with open(raw_out_path + EU + ".csv", "w") as f:
        f.write(header_line)
        for line in EU_dict[EU]:
            if len(EU.split("-")) == 2:
                f.write(line)
            else:
                model = line.split(",")[0]
                model = model + "-D0"
                line = ",".join(line.split(",")[1:])
                f.write(model + "," + line)
