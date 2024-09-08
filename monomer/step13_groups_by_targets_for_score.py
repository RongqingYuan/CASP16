import sys
import seaborn as sns
import matplotlib
import os
import pandas as pd
import numpy as np
import time

csv_raw_path = "./monomer_data_aug_30/raw/EU/"  # read raw csv files
csv_path = "./monomer_data_aug_30/raw_data/EU/"  # read raw csv files
csv_path = "./monomer_data_aug_30/processed/EU/"  # read raw csv files
if "whole" in csv_path:
    use_domain = False
else:
    use_domain = True
csv_list = [txt for txt in os.listdir(csv_path) if txt.endswith(".csv")]

out_path = "./by_target/"
if not os.path.exists(out_path):
    os.makedirs(out_path)
data_whole = pd.DataFrame()
for csv_file in csv_list:
    data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
    print("Processing {}".format(csv_file), data_tmp.shape)
    # if data_tmp.shape[1] == 35:
    #     print("something wrong with {}".format(csv_file))
    #     sys.exit(0)
    data_whole = pd.concat([data_whole, data_tmp], axis=0)

measures = data_whole.columns
if use_domain:
    data_whole.index = data_whole.index.str.replace(
        r'(\w+)(TS\d+_\d+)-(\w+)', r'\1-\3\2')

data_whole['prefix'] = data_whole.index.str.split('_').str[0]
data_whole['group'] = data_whole.index.str.split('_').str[1]
print(data_whole.head())
print(data_whole.shape)


# get all group 6 rows
data_whole_6 = data_whole[data_whole['group'] == '6']
# drop all group 6 rows
data_whole = data_whole[data_whole['group'] != '6']
# breakpoint()
data_whole_grouped = data_whole.groupby('prefix').max()
print(data_whole_grouped.shape)
print(data_whole_grouped.head())
print(measures)

breakpoint()

for measure_of_interest in measures:
    print(data_whole_grouped.head())
    print(data_whole_grouped.shape)
    data_whole_mean = pd.DataFrame(data_whole_grouped[measure_of_interest])
    print(data_whole_mean.shape)

    data_whole_mean.index = data_whole_mean.index.str.split('TS').map(tuple)
    print(len(data_whole_mean.index))
    data_whole_mean.index = pd.MultiIndex.from_tuples(
        data_whole_mean.index, names=['target', 'group'])
    data_whole_mean = data_whole_mean.stack().unstack('target')
    data_whole_mean.index = [f'{b}-{c}' for b, c in data_whole_mean.index]

    print(data_whole_mean.head())
    print(data_whole_mean.shape)

    if use_domain:
        end = "EU"
    else:
        end = "whole"
    if "raw" not in csv_path:
        data_whole_mean.to_csv(out_path +
                               './groups_by_targets_for-{}-{}.csv'.format(measure_of_interest, end))
    else:
        data_whole_mean.to_csv(out_path +
                               './groups_by_targets_for-raw-{}-{}.csv'.format(measure_of_interest, end))
    # # save the data
    # # normalize the data with the z-score again
    # data_whole_mean = (data_whole_mean - data_whole_mean.mean()
    #                    ) / data_whole_mean.std()
    # # fill nan with 0
    # data_whole_mean.fillna(0, inplace=True)

    # data_whole_mean.to_csv(out_path +
    #                        './individual_score_processed-{}-{}.csv'.format(measure_of_interest, end))
