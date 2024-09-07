import pandas as pd
import numpy as np
import scipy.stats as stats
data_path = "./tmp/"
data_file = "sum_best.csv"


# run the t-test
data = pd.read_csv(data_path + data_file, index_col=0)
# drop  sum
data = data.drop("sum", axis=1)
data = data.T

# breakpoint()

groups = data.columns

points = {}

for group_1 in groups:
    for group_2 in groups:
        if group_1 == group_2:
            continue
        data_1 = data[group_1]
        data_2 = data[group_2]
        # breakpoint()
        t, p = stats.ttest_rel(data_1, data_2)
        if group_1 not in points:
            points[group_1] = 0
        if t > 0 and p/2 < 0.005:
            points[group_1] += 1

# sort the points
points = dict(sorted(points.items(), key=lambda x: x[1], reverse=True))
print(points)


# try bootstrapping over the targets

times = 100
for i in range(times):
    # sample new data
    points = {}
    new_data = data.sample(frac=1, replace=True, axis=0)
    # print(new_data)
    for group_1 in groups:
        for group_2 in groups:
            if group_1 == group_2:
                continue
            data_1 = new_data[group_1]
            data_2 = new_data[group_2]
            # breakpoint()
            t, p = stats.ttest_rel(data_1, data_2)
            if group_1 not in points:
                points[group_1] = 0
            if t > 0 and p/2 < 0.005:
                points[group_1] += 1
    breakpoint()
    points = dict(sorted(points.items(), key=lambda x: x[1], reverse=True))
    print(points)
