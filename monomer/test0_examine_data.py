
import sys
import seaborn as sns
import matplotlib
import os
import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
import factor_analyzer
from sklearn.preprocessing import MinMaxScaler
from sympy import *
from matplotlib import pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
import scipy.stats as stats
csv_path = "./monomer_data_aug_28/processed/EU/"
# csv_path = "./monomer_data/EU/"
png_dir = "./new_png/"
score_dir = "./new_score/"
if not os.path.exists(png_dir):
    os.makedirs(png_dir)

if not os.path.exists(score_dir):
    os.makedirs(score_dir)

csv_list = [txt for txt in os.listdir(csv_path) if txt.endswith(".csv")]
csv_list = [txt for txt in os.listdir(
    csv_path) if txt.endswith(".csv") and txt.startswith("T1")]


# open the csv_list[0]

data = pd.read_csv(csv_path + csv_list[0], index_col=0)
# calculate the variance of the data
print(data.var())
