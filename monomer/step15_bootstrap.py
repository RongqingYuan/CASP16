import sys


from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

score_path = "./by_target/"
measure = "GDT_HA"
measure = "RMSD[L]"
measure = "RMS_CA"
measure = "GDT_TS"
score_file = "groups_by_targets_for-{}-EU.csv".format(measure)
score_file = "groups_by_targets_for-raw-{}-EU.csv".format(measure)


score_matrix = pd.read_csv(score_path + score_file, index_col=0)
# score_matrix = score_matrix.T
T1_data = score_matrix.filter(regex='T1')
print(score_matrix.shape)
print(score_matrix.head())
print(T1_data.shape)
print(T1_data.head())
T1_data = T1_data.T
print(T1_data.shape)
print(T1_data.head())
# drop columns with more than 80% values missing
T1_data = T1_data.loc[:, T1_data.isnull().mean() < 0.8]
# fill na with 0
T1_data.fillna(50, inplace=True)
print(T1_data.shape)
print(T1_data.head())
# fill na with 0
groups = T1_data.columns
# run paired t-test for group pairs
total_count = 0
significant_count = 0
points = {}
# create a 2d array to see if one wins over the other
length = len(groups)
win_matrix = [[0 for i in range(length)] for j in range(length)]

for i in range(length):
    for j in range(length):
        group_1 = groups[i]
        group_2 = groups[j]
        if group_1 == group_2:
            continue
        group_1_data = T1_data[group_1]
        group_2_data = T1_data[group_2]
        # run paired t-test
        import scipy.stats as stats
        t_stat, p_value = stats.ttest_rel(group_1_data, group_2_data)
        print("Group 1: {}, Group 2: {}, t-stat: {}, p-value: {}".format(
            group_1, group_2, t_stat, p_value))
        total_count += 1
        if p_value/2 < 0.05:
            significant_count += 1
        if t_stat > 0 and p_value/2 < 0.05:
            if group_1 not in points:
                points[group_1] = 0
            points[group_1] += 1
            win_matrix[i][j] = 1
print("Total count: {}, Significant count: {}".format(
    total_count, significant_count))
# sort the points ascending = False
points = dict(sorted(points.items(), key=lambda x: x[1], reverse=True))
print(points)
# plot the win matrix, if 1 use grey, if 0 use white
plt.imshow(win_matrix, cmap='Greys', interpolation='nearest')
plt.show()
plt.savefig("./tmp/win_matrix.png")


# group 1 is the first group in the ranked points
group_1 = list(points.keys())[0]
# group 2 is the second group in the ranked points
group_2 = list(points.keys())[1]
# now do bootstrapping
rounds = 5000
group_1_wins = []
group_2_wins = []
no_difference = []
for i in range(rounds):
    sampled_data = T1_data.sample(frac=1, replace=True)
    group_1_data = sampled_data[group_1]
    group_2_data = sampled_data[group_2]
    t_stat, p_value = stats.ttest_rel(group_1_data, group_2_data)
    if t_stat > 0 and p_value/2 < 0.05:
        group_1_wins.append(1)
    elif t_stat < 0 and p_value/2 < 0.05:
        group_2_wins.append(1)
    else:
        no_difference.append(1)
print("Group 1 wins: {}, Group 2 wins: {}".format(
    len(group_1_wins), len(group_2_wins)))
print("No difference: {}".format(len(no_difference)))


################
sys.exit(0)


def jackknife_resampling(data):
    n = len(data)
    jackknife_samples = []
    for i in range(n):
        sample = np.delete(data, i)
        jackknife_samples.append(sample.mean())
    return np.array(jackknife_samples)


def jackknife_t_test(group_1, group_2):
    # Perform jackknife resampling
    jack_1 = jackknife_resampling(group_1)
    jack_2 = jackknife_resampling(group_2)
    # Perform t-test on jackknife samples
    t_statistic, p_value = stats.ttest_ind(jack_1, jack_2)
    return t_statistic, p_value


# Example usage
np.random.seed(42)  # for reproducibility
group_1 = np.random.normal(loc=0, scale=1, size=30)
group_2 = np.random.normal(loc=0.5, scale=1, size=30)
t_stat, p_val = jackknife_t_test(group_1, group_2)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")
# Interpret the results
alpha = 0.05
if p_val < alpha:
    print("Reject the null hypothesis: There is a significant difference between the groups.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference between the groups.")
