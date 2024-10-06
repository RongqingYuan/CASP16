import argparse
import os
import pandas as pd


def to_csv(input_dir, files, output_dir):
    for file in files:
        file_path = os.path.join(input_dir, file)
        data = pd.read_csv(file_path, sep='\t')
        data.to_csv(os.path.join(output_dir, file.replace(
            '.results', '.csv')), index=False)


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='/data/data1/conglab/jzhan6/CASP16/targetPDBs/Targets_hybrid_20240922/results/')
parser.add_argument('--output_dir', type=str,
                    default="./M_target_csv/")
args = parser.parse_args()
data_dir = args.data_dir
output_dir = args.output_dir

files = [file for file in os.listdir(
    data_dir) if file.endswith('.results')]

bad_targets = ["M1268", "M1297_v1"]
remove = []
for file in files:
    for target in bad_targets:
        if file.startswith(target):
            remove.append(file)
for target in remove:
    files.remove(target)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for file in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, file))
to_csv(data_dir, files, output_dir)
