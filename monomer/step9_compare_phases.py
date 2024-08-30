import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.stats as stats
processed_data_path = "./monomer_data_aug_28/raw_data/EU/"
processed_data = [txt for txt in os.listdir(
    processed_data_path) if txt.endswith(".csv")]

# For each csv file starts with T1XXX, we want to see if there is a corresponding T2XXX file.

first_phase = "T0"
second_phase = "T1"

T1_files = []
for monomer_data in processed_data:
    if monomer_data.startswith(first_phase):
        monomer_data_T2 = monomer_data.replace(first_phase, second_phase)
        if monomer_data_T2 not in processed_data:
            print("Warning: {} does not have a corresponding file".format(monomer_data))
        else:
            print("Found a corresponding file for {}".format(monomer_data))
            T1_files.append(monomer_data)
    else:
        continue

print("There are {} {} files".format(len(T1_files), first_phase))
quantile_diffs = []

top_n = 100
quantile = 0.5
measure = "GDT_TS"


top_n_files = []
top_n_diffs = []
for T1_file in T1_files:
    T2_file = T1_file.replace(first_phase, second_phase)
    T1_data = pd.read_csv(processed_data_path + T1_file, index_col=0)
    T2_data = pd.read_csv(processed_data_path + T2_file, index_col=0)
    T1_measure = T1_data[measure]
    T2_measure = T2_data[measure]

    # TODO two ways: 1. compare the quantile 2. take top n and run t-test

    # get 75% quantile
    # this part is base on quantile
    quantile_75_T1 = T1_measure.quantile(quantile)
    quantile_75_T2 = T2_measure.quantile(quantile)
    quantile_diff = quantile_75_T2 - quantile_75_T1
    print("{} {} {}".format(T1_file, quantile_75_T1, quantile_75_T2))
    quantile_diffs.append(quantile_diff)

    # if quantile_75_T1 > quantile_75_T2:
    #     print("{} has a higher 75% quantile than {}".format(T1_file, T2_file))
    #     print("T1: {}, T2: {}".format(quantile_75_T1, quantile_75_T2))
    #     T1_measure = list(T1_measure)
    #     T2_measure = list(T2_measure)
    #     T1_mean = sum(T1_measure) / len(T1_measure)
    #     T2_mean = sum(T2_measure) / len(T2_measure)
    #     mean_diff = T1_mean - T2_mean
    #     print("T1 mean: {}, T2 mean: {}, mean difference: {}".format(
    #         T1_mean, T2_mean, mean_diff))
    # else:
    #     print()

    # also we can run unpaired t-test
    # this part is based on t-test
    # t-test does not require equal sample size, but let's do it anyway
    T1_measure = list(T1_measure)
    T2_measure = list(T2_measure)
    len_T1 = len(T1_measure)
    len_T2 = len(T2_measure)
    max_len = max(len_T1, len_T2)
    min_len = max_len // 2
    min_len = 50  # try this as well because we focus on the top performers
    min_len = min(len_T1, len_T2)
    min_len = top_n  # try this as well because we focus on the top performers
    T1_measure = T1_measure[:min_len]
    T2_measure = T2_measure[:min_len]
    T1_mean = sum(T1_measure) / len(T1_measure)
    T2_mean = sum(T2_measure) / len(T2_measure)
    mean_diff = T2_mean - T1_mean
    top_n_files.append(T1_file)
    top_n_diffs.append(mean_diff)
    # t_stat, p_value = stats.ttest_ind(T1_measure, T2_measure)
    # if p_value < 0.05 and t_stat > 0:
    #     # print("T-test result: T1 has a higher mean than T2 for {}".format(T1_file))
    #     # calculate the mean difference
    #     T1_mean = sum(T1_measure) / len(T1_measure)
    #     T2_mean = sum(T2_measure) / len(T2_measure)
    #     mean_diff = T1_mean - T2_mean
    #     top_n_files.append(T1_file)
    #     top_n_diffs.append(mean_diff)
    #     with open("T1_higher.txt", "a") as f:
    #         f.write(str(T1_file) + " " + str(t_stat) +
    #                 " " + str(p_value) + " " + str(mean_diff) + "\n")


# sort quantile_diffs with T1_files correspondingly, largest difference first
quantile_diffs, T1_files = zip(*sorted(zip(quantile_diffs, T1_files)))
# invert the order
quantile_diffs = quantile_diffs[::-1]
T1_files = T1_files[::-1]

# plot bar plot of quantile_diffs. y-axis is the difference in 75% quantile, x is each pair of T1 and T2
plt.figure(figsize=(12, 6))
plt.bar(range(len(quantile_diffs)), quantile_diffs)
# set y axis to be +N to -N, N is the absolute value of the largest difference
N = max(abs(min(quantile_diffs)), abs(max(quantile_diffs))) + 1
plt.ylim(-N, N)
plt.xticks(range(len(quantile_diffs)), T1_files, rotation=90, fontsize=10)
plt.xlabel("{} and {} pairs".format(first_phase, second_phase), fontsize=12)
plt.ylabel("{}% quantile {} - {}".format(
    quantile*100,  second_phase, first_phase), fontsize=12)
# draw a line at 0
plt.axhline(0, color='black', linestyle='-')
plt.title("Difference for {}% quantile of {} between {} and {}".format(
    quantile*100, measure, second_phase, first_phase), fontsize=14)
plt.tight_layout()
plt.savefig("./stages_png/" +
            "{}_{}_{}_quantile_diffs_{}.png".format(second_phase, first_phase, quantile*100, measure), dpi=300)

# sort top_n_diffs with T1_files correspondingly
top_n_diffs, top_n_files = zip(*sorted(zip(top_n_diffs, top_n_files)))
# invert the order
top_n_diffs = top_n_diffs[::-1]
top_n_files = top_n_files[::-1]

# plot bar plot of top_n_diffs. y-axis is the difference in mean, x is each pair of T1 and T2
plt.figure(figsize=(12, 6))
plt.bar(range(len(top_n_diffs)), top_n_diffs)
# set y axis to be +N to -N, N is the absolute value of the largest difference
N = max(abs(min(top_n_diffs)), abs(max(top_n_diffs))) + 1
plt.ylim(-N, N)
plt.xticks(range(len(top_n_diffs)), top_n_files, rotation=90, fontsize=10)
plt.xlabel("{} and {} pairs".format(first_phase, second_phase), fontsize=12)
plt.ylabel("top_{} {} - {}".format(top_n,
                                   second_phase, first_phase), fontsize=12)
# draw a line at 0
plt.axhline(0, color='black', linestyle='-')
plt.title("Difference for mean between {} of {} and {} for top {}".format(
    second_phase, measure, first_phase, top_n), fontsize=14)
plt.tight_layout()
plt.savefig("./stages_png/"+"{}_{}_top_{}_mean_diffs_{}.png".format(second_phase,
            first_phase, top_n, measure), dpi=300)
