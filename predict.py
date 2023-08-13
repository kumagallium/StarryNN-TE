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

    target = args.formula
    target_idx = np.where(comp_list == target)[0][0]
    T_list = list(range(300, 1100, 100))
    input_list = []
    for T in T_list:
        tmp = list(df_features.iloc[target_idx, 0:])
        tmp.append(T)
        input_list.append(tmp)
    df_input = pd.DataFrame(input_list).dropna()
    df_input = df_input.rename(columns={df_input.shape[1] - 1: "Temperature"})
    X = df_input
    X.columns = X.columns.astype(str)

    with open("models/scaler_X_final.pkl", "rb") as f:
        scaler_x = pickle.load(f)
    with open("models/scaler_y_final.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    X = scaler_x.transform(X)

    loaded_model = load_model(args.model_path)
    print(loaded_model.summary())
    y_pred = loaded_model.predict(X)
    print(dict(zip(T_list, y_pred.T[0])))
    y_pred = scaler_y.inverse_transform(y_pred)
    print(dict(zip(T_list, y_pred.T[0])))


if __name__ == "__main__":
    seed_value = 0
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
