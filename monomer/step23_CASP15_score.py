
import matplotlib.pyplot as plt
import pandas as pd
model = "first"
model = "best"

mode = "hard"
mode = "medium"
mode = "easy"
mode = "all"
features = ['GDT_TS', 'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
            'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
            'SphGr',
            'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
            'MP_rotout', 'MP_ramout', 'MP_ramfv', 'reLLG_lddt', 'reLLG_const']

wanted_features = ['LDDT', 'CAD_AA', 'SphGr',
                   'MP_clash', 'RMS_CA',
                   'GDT_HA', 'QSE', 'reLLG_const']

wanted_features = ['GDT_HA', 'GDC_SC', 'AL0_P', 'SphGr',
                   'CAD_AA', 'QSE', 'MolPrb_Score', 'reLLG_const']
# sum_path = "./sum/"
# sum_path = "./sum_EU/"
equal_weight = False
equal_weight = True
if equal_weight:
    weights = [1/8] * 8
else:
    weights = [1/16, 1/16, 1/16,
               1/12, 1/12,
               1/4, 1/4, 1/4]
data_sum = pd.DataFrame()
for i in range(len(wanted_features)):
    feature = wanted_features[i]
    weight = weights[i]
    file = "./sum_EU/" + "sum_{}-{}-{}.csv".format(feature, model, mode)
    data = pd.read_csv(file, index_col=0)
    # only get the sum column
    data = data["sum"]
    data = pd.DataFrame(data) * weight
    data.columns = [feature]
    data_sum = pd.concat([data_sum, data], axis=1)
data_sum["sum"] = data_sum.sum(axis=1)
# sort the sum column
data_sum = data_sum.sort_values(by="sum", ascending=False)
data_sum.to_csv(
    "./by_EU/" + "sum_CASP16_score-{}-{}-equal-weight-{}.csv".format(model, mode, equal_weight))

# plot the bar chart

fig = plt.figure(figsize=(32, 16))
bottom = [0] * len(data_sum)
print(bottom)
for i in range(len(wanted_features)):
    feature = wanted_features[i]
    plt.bar(data_sum.index, data_sum[feature], label=feature, bottom=bottom)
    bottom = [a+b for a, b in zip(bottom, data_sum[feature])]

plt.xticks(rotation=45, fontsize=10, ha='right')
plt.ylabel("Sum of score")
plt.title("Sum of score for {} model in {} targets".format(model, mode))
# draw a line at y=0
plt.axhline(y=0, color='k')

plt.legend()
plt.savefig(
    "./by_EU/" + "sum_CASP16_score-{}-{}-equal-weight-{}.png".format(model, mode, equal_weight
                                                                     ), dpi=300)
