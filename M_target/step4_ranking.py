import os
import sys
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
# three types of input: pp, pn, all
parser.add_argument('--type', type=str, default="all")
args = parser.parse_args()
type = args.type
if type == "pp":
    input = "./step3_pp/"
elif type == "pn":
    input = "./step3_pn/"
elif type == "all":
    input = "./step3_all/"
else:
    print("type must be pp, pn or all")
    sys.exit()

for file in os.listdir(input):
    if file.endswith(".csv"):
        data = pd.read_csv(input + file)
