import pandas as pd
import os


path = "/home2/s439906/data/CASP16/stoichiometry/"
# out_path = "/home2/s439906/data/CASP16/"
out_path = "./"
files = [file for file in os.listdir(path) if file.endswith(".csv")]

data = pd.DataFrame()
data_dict = {}
for file in files:
    data_temp = pd.read_csv(path + file)
    stoichiometry = data_temp["AssemblyData"].tolist()
    for stoich in stoichiometry:
        if stoich not in data_dict:
            data_dict[stoich] = 0
        data_dict[stoich] += 1
    print("finished:", file)

# print(data_dict.keys())
# sort it
data_dict = dict(
    sorted(data_dict.items(), key=lambda item: item[1], reverse=True))

# convert to dataframe
data = pd.DataFrame(data_dict.items(), columns=["stoichiometry", "count"])
data.to_csv(out_path + "stoich_bg_distribution.csv", index=False)
