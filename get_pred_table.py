import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import random
import matminer.featurizers.composition.composite as composite

magpie_preset = composite.ElementProperty.from_preset("magpie")
import pymatgen.core as mg
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
import argparse

plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["xtick.direction"] = "in"  # 目盛り線の向き、内側"in"か外側"out"かその両方"inout"か
plt.rcParams["ytick.direction"] = "in"  # 目盛り線の向き、内側"in"か外側"out"かその両方"inout"か
plt.rcParams["xtick.major.width"] = 1.2  # x軸主目盛り線の線幅
plt.rcParams["ytick.major.width"] = 1.2  # y軸主目盛り線の線幅
plt.rcParams["xtick.major.size"] = 3  # x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 3  # y軸主目盛り線の長さ
# plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.grid"] = False  # True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.3
plt.rcParams["legend.markerscale"] = 2
plt.rcParams["legend.fancybox"] = False  # Trueを指定すると凡例の枠の角が丸くなる
plt.rcParams["legend.framealpha"] = 1  # 判例の透明度
plt.rcParams["legend.edgecolor"] = "black"


def get_formula_to_feature(formula: str) -> list:
    try:
        mg_comp = mg.Composition(formula)
        return magpie_preset.featurize(mg_comp)

    except Exception as e:
        # print(f"Error: {e}")
        return []


def identify_material_and_dopants(formula: str) -> dict:
    composition = mg.Composition(formula).fractional_composition

    # Determine base material and dopant compositions
    base_material = {el: amt for el, amt in composition.items() if amt >= 0.1}
    dopants = {el: amt for el, amt in composition.items() if amt < 0.1}

    # Normalize base material to sum to 1
    total_base = sum(base_material.values())
    total_dopants = sum(dopants.values())
    ratio = total_dopants / total_base

    base_material_normalized = {
        el: amt / total_base for el, amt in base_material.items()
    }
    dopants_normalized = {el: amt / total_base for el, amt in dopants.items()}

    return {"base_material": base_material_normalized, "dopants": dopants_normalized}


def get_feature(formula: str) -> list:
    try:
        comp_dict = identify_material_and_dopants(formula)
        feature = []

        base_mat_str = ""
        for el, frac in comp_dict["base_material"].items():
            base_mat_str += el.symbol + str(frac)

        feature = get_formula_to_feature(base_mat_str)

        if len(comp_dict["dopants"]) > 0:
            dopants_str = ""
            sum_frac = 0
            for el, frac in comp_dict["dopants"].items():
                dopants_str += el.symbol + str(frac)
                sum_frac += frac
            dopants_feature = sum_frac * np.array(get_formula_to_feature(dopants_str))
        else:
            dopants_feature = [0] * len(feature)

        feature.extend(list(dopants_feature))

        return feature

    except Exception as e:
        # print(f"Error: {e}")
        return []


def main(args):
    data_path = args.data_path
    df_data = pd.read_csv(data_path)
    df_data = df_data[df_data["e_above_hull"] == 0]
    df_data = df_data[(df_data["band_gap"] > 0) & (df_data["band_gap"] <= 2)]
    comp_list = df_data["pretty_formula"].unique()

    comp_feats = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(get_feature, formula) for formula in tqdm(comp_list)]
        for f in tqdm(futures):
            comp_feats.append(f.result())
    df_features = pd.DataFrame(comp_feats)

    input_list = []
    T_range = list(range(100, 1100, 100))
    complist_T = []
    for i, comp in tqdm(enumerate(comp_list)):
        for T in T_range:
            complist_T.append(comp)
            tmp = list(df_features.iloc[i, 0:])
            tmp.append(T)
            input_list.append(tmp)
    df_input = pd.DataFrame(input_list)
    df_input = df_input.rename(columns={df_input.shape[1] - 1: "Temperature"})
    X = df_input
    X.columns = X.columns.astype(str)
    predictions = X.copy()

    with open("models/scaler_X_final.pkl", "rb") as f:
        scaler_x = pickle.load(f)

    with open("models/scaler_y_final.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    X = scaler_x.transform(X)

    loaded_model = load_model(args.model_path)
    print(loaded_model.summary())
    y_pred = loaded_model.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred)
    predictions["results"] = y_pred.T[0]
    predictions["composition"] = complist_T
    df_te_mat = predictions[["composition", "Temperature", "results"]]
    df_te_mat = df_te_mat.sort_values(by="results", ascending=False)
    df_te_mat.to_csv("results/pred_table.csv", index=False)

    dict_formula = {}
    dict_results = {}
    for T in T_range:
        dict_formula[T] = index_tmp = list(
            df_te_mat[df_te_mat["Temperature"] == T].sort_values(
                by="results", ascending=False
            )["composition"]
        )
        dict_results[T] = results_tmp = list(
            df_te_mat[df_te_mat["Temperature"] == T].sort_values(
                by="results", ascending=False
            )["results"]
        )
    df_formula = pd.DataFrame(dict_formula)
    df_formula = df_formula.reset_index(drop=True)
    df_formula.index = df_formula.index + 1
    df_results = pd.DataFrame(dict_results)
    df_results = df_results.reset_index(drop=True)
    df_results.index = df_results.index + 1

    fig = plt.figure(figsize=(12, 7), dpi=400, facecolor="w", edgecolor="k")
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(pad=1)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(bottom="off", top="off")
    ax.tick_params(left="off")
    ax.tick_params(bottom=False, left=False, right=False, top=False)
    rank = 50
    sns.heatmap(
        df_results.iloc[:rank],
        cmap="jet",
        annot=df_formula.iloc[:rank],
        fmt="",
        # vmin=0.5,
        # vmax=1,
        annot_kws={"size": 7},
        cbar_kws={"pad": 0.01},
    )
    plt.tight_layout()
    plt.savefig("results/pred_table.png")


if __name__ == "__main__":
    seed_value = 10
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="datasets/mp_all.csv",
        type=str,
        help="Data path (default: datasets/mp_all.csv)",
    )
    parser.add_argument(
        "--formula",
        default="NaAlB14",
        type=str,
        help="formula (default: NaAlB14)",
    )
    parser.add_argument(
        "--model-path",
        default="models/final_model.keras",
        type=str,
        help="Model path (default: models/final_model.keras)",
    )
    args = parser.parse_args()
    main(args)
