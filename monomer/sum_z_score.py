from fractions import Fraction
import argparse
import pandas as pd
import os


def get_group_by_target(csv_list, csv_path, out_path,
                        model, mode, phase, impute_value, weights, EU_weight):
    data = pd.DataFrame()
    # get another df for GDT_HA    GDC_SC  reLLG_const       QSE     AL0_P     SphGr    CAD_AA      LDDT  MolPrb_Score
    GDT_HA_df = pd.DataFrame()
    GDC_SC_df = pd.DataFrame()
    reLLG_const_df = pd.DataFrame()
    QSE_df = pd.DataFrame()
    AL0_P_df = pd.DataFrame()
    SphGr_df = pd.DataFrame()
    CAD_AA_df = pd.DataFrame()
    LDDT_df = pd.DataFrame()
    MolPrb_Score_df = pd.DataFrame()

    for csv_file in csv_list:
        target = csv_file.split(".")[0]
        data_tmp = pd.read_csv(csv_path + csv_file, index_col=0)
        if data_tmp.empty:
            print("Error: No data found in provided CSV files.")
            return
        if data_tmp.shape[1] == len(weights):
            data_weighted_sum = data_tmp.dot(weights)
            data_weighted_sum = pd.DataFrame(data_weighted_sum)
            data_weighted_sum = data_weighted_sum.rename(
                columns={0: target})
            data_tmp['weighted_sum'] = data_weighted_sum
        data_tmp.index = data_tmp.index.str.extract(
            r'(T\w+)TS(\w+)_(\w+)-(D\w+)').apply(lambda x: (f"{x[0]}-{x[3]}", f"TS{x[1]}", x[2]), axis=1)
        data_tmp.index = pd.MultiIndex.from_tuples(
            data_tmp.index, names=['target', 'group', 'model'])

        # no more duplicated index to worry about

        # if data_tmp.index.duplicated().any():
        #     print(f"Duplicated index in {csv_file}")
        #     data_tmp = data_tmp.groupby(
        #         level=data_tmp.index.names).max()

        if model == "best":
            data_tmp = data_tmp.loc[(slice(None), slice(None), [
                "1", "2", "3", "4", "5"]), :]
        elif model == "first":
            data_tmp = data_tmp.loc[(slice(None), slice(None),
                                     "1"), :]
        elif model == "sixth":
            data_tmp = data_tmp.loc[(slice(None), slice(None),
                                     "6"), :]
        grouped = data_tmp.groupby(["group"])

        # take the max weighted_sum as well as the corresponding components
        z_score_max = data_tmp.loc[grouped['weighted_sum'].idxmax()]
        z_score_max = z_score_max.groupby(["group"])
        z_score = pd.DataFrame(z_score_max['weighted_sum'].max())
        z_score = z_score.rename(columns={"weighted_sum": target})
        data = pd.concat([data, z_score], axis=1)

        GDT_HA = pd.DataFrame(z_score_max['GDT_HA'].max())
        GDT_HA = GDT_HA.rename(columns={"GDT_HA": target})
        GDT_HA_df = pd.concat([GDT_HA_df, GDT_HA], axis=1)

        GDC_SC = pd.DataFrame(z_score_max['GDC_SC'].max())
        GDC_SC = GDC_SC.rename(columns={"GDC_SC": target})
        GDC_SC_df = pd.concat([GDC_SC_df, GDC_SC], axis=1)

        reLLG_const = pd.DataFrame(z_score_max['reLLG_const'].max())
        reLLG_const = reLLG_const.rename(columns={"reLLG_const": target})
        reLLG_const_df = pd.concat([reLLG_const_df, reLLG_const], axis=1)

        QSE = pd.DataFrame(z_score_max['QSE'].max())
        QSE = QSE.rename(columns={"QSE": target})
        QSE_df = pd.concat([QSE_df, QSE], axis=1)

        AL0_P = pd.DataFrame(z_score_max['AL0_P'].max())
        AL0_P = AL0_P.rename(columns={"AL0_P": target})
        AL0_P_df = pd.concat([AL0_P_df, AL0_P], axis=1)

        SphGr = pd.DataFrame(z_score_max['SphGr'].max())
        SphGr = SphGr.rename(columns={"SphGr": target})
        SphGr_df = pd.concat([SphGr_df, SphGr], axis=1)

        CAD_AA = pd.DataFrame(z_score_max['CAD_AA'].max())
        CAD_AA = CAD_AA.rename(columns={"CAD_AA": target})
        CAD_AA_df = pd.concat([CAD_AA_df, CAD_AA], axis=1)

        LDDT = pd.DataFrame(z_score_max['LDDT'].max())
        LDDT = LDDT.rename(columns={"LDDT": target})
        LDDT_df = pd.concat([LDDT_df, LDDT], axis=1)

        MolPrb_Score = pd.DataFrame(z_score_max['MolPrb_Score'].max())
        MolPrb_Score = MolPrb_Score.rename(columns={"MolPrb_Score": target})
        MolPrb_Score_df = pd.concat([MolPrb_Score_df, MolPrb_Score], axis=1)
    data_copy = data.copy()

    data = data.fillna(impute_value)
    data = data.reindex(sorted(data.columns), axis=1)
    data = data.sort_index()
    data_csv = f"sum-{model}-{mode}-{phase}-impute={impute_value}_unweighted.csv"
    data.to_csv(out_path + data_csv)
    print(data.shape)

    # data_columns = data.columns
    # target_count = {}
    # for EU in data_columns:
    #     target = EU.split("-")[0]
    #     if target not in target_count:
    #         target_count[target] = 0
    #     target_count[target] += 1
    # target_weight = {key: 1/value for key, value in target_count.items()}
    # EU_weight = {EU: target_weight[EU.split("-")[0]]
    #              for EU in data_columns}
    # for EU in data_columns:
    #     print(EU, EU_weight[EU])

    data = data * pd.Series(EU_weight)
    data["sum"] = data.sum(axis=1)
    data = data.sort_values(by="sum", ascending=False)
    data_sum_csv = f"sum-{model}-{mode}-{phase}-impute={impute_value}-EU_weighted_sum.csv"
    data.to_csv(out_path + data_sum_csv)

    # # top_n = 28
    # # # get the top n index
    # # top_n_index = data.index[:top_n]

    # data_copy = data_copy * pd.Series(EU_weight)
    # # data_copy["sum"] = data_copy.sum(axis=1)
    # data_missing_csv = f"sum-{model}-{mode}-{phase}-impute={impute_value}-EU_weighted_with_missing.csv"
    # data_copy.to_csv(out_path + data_missing_csv)

    def df2csv(df, name, out_path=out_path,
               model=model, mode=mode, phase=phase, impute_value=impute_value, EU_weight=EU_weight):
        df = df.fillna(impute_value)
        df = df.reindex(sorted(df.columns), axis=1)
        df = df.sort_index()
        df_csv = f"{name}-{model}-{mode}-{phase}-impute={impute_value}_unweighted.csv"
        df.to_csv(out_path + df_csv)

        df = df * pd.Series(EU_weight)
        # df["sum"] = df.sum(axis=1)
        # df = df.sort_values(by="sum", ascending=False)
        df_sum_csv = f"{name}-{model}-{mode}-{phase}-impute={impute_value}-EU_weighted_sum.csv"
        df.to_csv(out_path + df_sum_csv)

    df2csv(GDT_HA_df, "GDT_HA")
    df2csv(GDC_SC_df, "GDC_SC")
    df2csv(reLLG_const_df, "reLLG_const")
    df2csv(QSE_df, "QSE")
    df2csv(AL0_P_df, "AL0_P")
    df2csv(SphGr_df, "SphGr")
    df2csv(CAD_AA_df, "CAD_AA")
    df2csv(LDDT_df, "LDDT")
    df2csv(MolPrb_Score_df, "MolPrb_Score")


parser = argparse.ArgumentParser(description="options for sum z-score")

parser.add_argument('--csv_path', type=str,
                    default="./monomer_data_newest/processed/")
parser.add_argument('--out_path', type=str, default="./score_all/")
parser.add_argument('--model', type=str,
                    help='first, best or sixth', default='best')
parser.add_argument('--mode', type=str,
                    help='easy, medium, difficult or all', default='all')
parser.add_argument("--phase", type=str, default="0,1,2")
parser.add_argument('--impute_value', type=int, default=-2)
parser.add_argument("--weights", type=float, nargs='+',
                    default=[
                        1/6, 1/16, 1/6,
                        1/6,
                        1/16, 1/8,
                        1/8, 1/16,
                        1/16,
                    ])

args = parser.parse_args()
csv_path = args.csv_path
out_path = args.out_path
model = args.model
mode = args.mode
phase = args.phase
impute_value = args.impute_value
weights = args.weights

if phase != "0,1,2" and mode != "all":
    print("Warning: phase is not 0,1,2, mode is not all. Current implementation does not support this combination.")
    exit(1)

csv_list = [txt for txt in os.listdir(
    csv_path) if txt.endswith(".csv") and txt.startswith("T")]
csv_list = sorted(csv_list)

metadata_file = "./monomer_EU_info"
# difficulty_group = {"easy": [], "medium": [], "difficult": []}
EU_weight = {}
with open(metadata_file, "r") as f:
    for line in f:
        if line.startswith("T"):
            words = line.strip().split()
            protein = words[0]
            if protein == "T1249":
                protein = "T1249v1"
            if protein == "T2249":
                protein = "T2249v1"
            domain = words[1]
            difficulty = words[2]
            weight = float(Fraction(words[4]))
            EU = f"{protein}-{domain}"
            EU_weight[EU] = weight
            # difficulty_group[difficulty].append(EU)

difficulty_group = {"easy": [], "medium": [], "difficult": []}
difficulty_file = "./step7_result"
with open(difficulty_file, "r") as f:
    for line in f:
        if line.startswith("T"):
            words = line.strip().split()
            protein = words[0]
            if protein == "T1249-D1":
                protein = "T1249v1-D1"
            if protein == "T2249-D1":
                protein = "T2249v1-D1"
            difficulty = words[2]
            # EU = f"{protein}-{domain}"
            EU = protein
            if difficulty == "easy":
                difficulty_group[difficulty].append(EU)
            elif difficulty == "medium":
                difficulty_group[difficulty].append(EU)
            elif difficulty == "hard":
                difficulty_group["difficult"].append(EU)

if mode == "difficult":
    csv_list = [csv for csv in csv_list if csv.split(
        ".")[0] in difficulty_group["difficult"]]
    for csv in difficulty_group["difficult"]:
        if csv + ".csv" not in csv_list:
            print(f"Warning: {csv} not found in csv_list")
    EU_weight = {EU.split(".")[0]: EU_weight[EU.split(".")[0]]
                 for EU in csv_list}
elif mode == "medium":
    csv_list = [csv for csv in csv_list if csv.split(
        ".")[0] in difficulty_group["medium"]]
    for csv in difficulty_group["medium"]:
        if csv + ".csv" not in csv_list:
            print(f"Warning: {csv} not found in csv_list")
    EU_weight = {EU.split(".")[0]: EU_weight[EU.split(".")[0]]
                 for EU in csv_list}
elif mode == "easy":
    csv_list = [csv for csv in csv_list if csv.split(
        ".")[0] in difficulty_group["easy"]]
    for csv in difficulty_group["easy"]:
        if csv + ".csv" not in csv_list:
            print(f"Warning: {csv} not found in csv_list")
    EU_weight = {EU.split(".")[0]: EU_weight[EU.split(".")[0]]
                 for EU in csv_list}
elif mode == "all":
    pass
# breakpoint()
if phase == "0":
    csv_list = [csv for csv in csv_list if csv.split(".")[0].startswith("T0")]
    EU_weight = {EU.split(".")[0]: EU_weight[EU.split(".")[0]]
                 for EU in csv_list}
elif phase == "1":
    csv_list = [csv for csv in csv_list if csv.split(".")[0].startswith("T1")]
    EU_weight = {EU.split(".")[0]: EU_weight[EU.split(".")[0]]
                 for EU in csv_list}
elif phase == "2":
    csv_list = [csv for csv in csv_list if csv.split(".")[0].startswith("T2")]
    EU_weight = {EU.split(".")[0]: EU_weight[EU.split(".")[0]]
                 for EU in csv_list}
elif phase == "0,1,2":
    pass

assert len(csv_list) == len(EU_weight)
if not os.path.exists(out_path):
    os.makedirs(out_path)
get_group_by_target(csv_list, csv_path, out_path,
                    model, mode, phase, impute_value, weights, EU_weight)


# features = ['GDT_TS',
#             'GDT_HA', 'GDC_SC', 'GDC_ALL', 'RMS_CA', 'RMS_ALL', 'AL0_P',
#             'AL4_P', 'ALI_P', 'LGA_S', 'RMSD[L]', 'MolPrb_Score', 'LDDT',
#             'SphGr',
#             'CAD_AA', 'RPF', 'TMscore', 'FlexE', 'QSE', 'CAD_SS', 'MP_clash',
#             # 'MP_rotout',
#             # 'MP_ramout',
#             # 'MP_ramfv',
#             'reLLG_lddt',
#             'reLLG_const']
