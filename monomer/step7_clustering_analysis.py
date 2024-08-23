from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

csv_path = "./csv/"
csv_list = [txt for txt in os.listdir(csv_path) if txt.endswith(".csv")]
measure = "GDT_TS"
data = pd.read_csv("individual_score-{}.csv".format(measure), index_col=0)

# print the first 5 rows
print(data.head())
print(data.shape)
# print the sum of each column
print(data.sum())
labels = data.index
print(labels)
# transpose_data = data.T
# sys.exit(0)

# clustering analysis use pca

pca = PCA(n_components=4)
data_pca = pca.fit_transform(data)
print(data_pca.shape)
# how much variance is explained by the first two components
print(pca.explained_variance_ratio_)


# plot the 1d data
plt.figure(figsize=(8, 8))
plt.scatter(data_pca[:, 0], np.zeros(data_pca.shape[0]))
plt.xlabel("PC1")
plt.title("PCA")
plt.savefig("PCA_1d.png", dpi=300)

# plot the data
plt.figure(figsize=(8, 8))
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA_{}".format(measure))
# add the label to the points
for i in range(data_pca.shape[0]):
    plt.text(data_pca[i, 0], data_pca[i, 1], labels[i])
plt.savefig("PCA_{}.png".format(measure), dpi=300)


# run other clustering algorithms like tsne
tsne = TSNE(n_components=2, perplexity=20)
data_tsne = tsne.fit_transform(data)
plt.figure(figsize=(8, 8))
plt.scatter(data_tsne[:, 0], data_tsne[:, 1])
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")
plt.title("t-SNE")
plt.savefig("t-SNE.png", dpi=300)
