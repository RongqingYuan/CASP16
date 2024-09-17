import matplotlib.pyplot as plt
import pandas as pd


file_path = "./bootstrap_EU/"
out_path = "./bootstrap_analysis_EU/"
measure = "GDT_TS"
mode = "hard"
mode = "medium"
mode = "easy"
mode = "all"
model = "first"
model = "best"
p_value_threshold = 0.05

measures = ['GDT_TS', 'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
            'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
            'SphGr',
            'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
            'MP_rotout', 'MP_ramout', 'MP_ramfv', 'reLLG_lddt', 'reLLG_const']

measures_used = ['GDT_HA', 'GDC_SC', 'AL0_P', 'SphGr',
                 'CAD_AA', 'QSE', 'MolPrb_Score', 'reLLG_const']


def get_points(measure, model, mode, p):
    with open(file_path + "{}_{}_{}_p={}_ranking_step18.txt".format(measure, model, mode, p_value_threshold), "r") as f:
        for line in f:
            point_dict = eval(line)
            break
    return point_dict


data_all = pd.DataFrame()
for measure in measures_used:
    points = get_points(measure, model, mode, p_value_threshold)
    data = pd.DataFrame(list(points.items()), columns=['group', measure])
    data.set_index('group', inplace=True)
    data_all = pd.concat([data_all, data], axis=1)
# fill na with 0
data_all = data_all.fillna(0)
sum = data_all.sum(axis=1)
data_all["sum"] = sum
data_all = data_all.sort_values(by="sum", ascending=False)
data_all.to_csv(out_path + "bootstrap_points_{}_{}_p={}.csv".format(model,
                mode, p_value_threshold))
print(data_all)
data_all.index = data_all.index.str.replace("TS", "")
bottom = [0] * len(data_all)
plt.figure(figsize=(35, 15))
for measure in measures_used:
    plt.bar(data_all.index, data_all[measure], label=measure, bottom=bottom)
    bottom = [a+b for a, b in zip(bottom, data_all[measure])]
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.ylabel("points in bootstrap")
plt.title("Points in bootstrap for {} model in {} targets".format(model, mode))
plt.legend()
plt.savefig(out_path + "bootstrap_points_{}_{}_p={}.png".format(model,
            mode, p_value_threshold), dpi=300)
