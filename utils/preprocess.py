import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import pymatgen.core as mg
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from utils import utils


def get_elements(formula):
    try:
        comp = mg.Composition(formula)
        elements = [el.symbol for el in comp.elements]
        return elements
    except:
        return []


def convert_fractional_composition(formula):
    try:
        comp = mg.Composition(formula).fractional_composition.formula
        return comp
    except:
        pass


def get_train_test_dataset(df_data, outputprop):
    filtercols = [
        "sid",
        "composition",
        "Temperature",
        "Thermal conductivity",
        "Seebeck coefficient",
        "Electrical conductivity",
        "ZT",
    ]
    df_data = df_data[filtercols]
    df_data["ZT_calc"] = (
        (df_data["Seebeck coefficient"] ** 2)
        * df_data["Electrical conductivity"]
        * df_data["Temperature"]
    ) / df_data["Thermal conductivity"]
    df_data["PF_calc"] = (
        (df_data["Seebeck coefficient"] ** 2)
        * df_data["Electrical conductivity"]
        * 10**3
    )
    df_data["ZT_RAE"] = np.abs((df_data["ZT_calc"] - df_data["ZT"]) / df_data["ZT"])
    df_data = df_data[(df_data["ZT_RAE"] > 0) & (df_data["ZT_RAE"] < 0.1)].dropna()
    df_data["Z"] = df_data["ZT"] / df_data["Temperature"]
    df_data = df_data[(df_data["ZT"] > 0)].dropna()
    df_data["Seebeck coefficient"] = np.abs(df_data["Seebeck coefficient"]) * 10**6
    df_data["Temperature_input"] = df_data["Temperature"]

    elements_list = []
    for comp in df_data["composition"].unique():
        elements_list.extend(get_elements(comp))
    with open("datasets/starry_elements.pkl", "wb") as f:
        pickle.dump(elements_list, f)

    frac_comp_list = []
    for comp in df_data["composition"].unique():
        frac_comp_list.append(convert_fractional_composition(comp))
    with open("datasets/starry_comosition.pkl", "wb") as f:
        pickle.dump(frac_comp_list, f)

    comp_feats = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [
            executor.submit(utils.get_feature, formula)
            for formula in tqdm(df_data["composition"])
        ]
        for f in tqdm(futures):
            comp_feats.append(f.result())

    df_features = pd.DataFrame(comp_feats)
    unique_key = "sid"
    inputprop = ["Temperature_input", unique_key]

    df_features = df_features.reset_index(drop=True)
    df_data = df_data.reset_index(drop=True)
    df_input = pd.concat(
        [df_features.iloc[:, 0:], df_data[inputprop], df_data[outputprop]], axis=1
    )
    df_input = df_input.dropna()

    train_keys, test_keys = train_test_split(
        df_input[unique_key].unique(), test_size=0.2, random_state=0
    )
    train_df = df_input[df_input[unique_key].isin(train_keys)]
    test_df = df_input[df_input[unique_key].isin(test_keys)]
    x_train = train_df.iloc[:, : -1 * (len(outputprop) + 1)]
    x_test = test_df.iloc[:, : -1 * (len(outputprop) + 1)]
    x_train.columns = x_train.columns.astype(str)
    x_test.columns = x_test.columns.astype(str)
    y_train = train_df.iloc[:, -1 * (len(outputprop)) :]
    y_test = test_df.iloc[:, -1 * (len(outputprop)) :]
    y_train = y_train.values.reshape(-1, len(outputprop))
    y_test = y_test.values.reshape(-1, len(outputprop))
    input_dim = x_train.shape[1]

    x_combined = np.concatenate([x_train, x_test], axis=0)
    x_combined = np.vstack([x_train, x_test])
    y_combined = np.concatenate([y_train, y_test], axis=0)
    y_combined = np.vstack([y_train, y_test])

    return input_dim, x_train, x_test, y_train, y_test, x_combined, y_combined


with open("datasets/starry_elements.pkl", "rb") as f:
    starry_elements = pickle.load(f)


def filter_elements(comp_str):
    comp = mg.Composition(comp_str)
    elements = [el.symbol for el in comp.elements]
    return all(el in starry_elements for el in elements)


with open("datasets/starry_comosition.pkl", "rb") as f:
    frac_comp_list = pickle.load(f)


def filter_composition(comp_str):
    comp = mg.Composition(comp_str).fractional_composition.formula
    return comp not in frac_comp_list


def get_mp_features(df_data):
    df_data = df_data[df_data["e_above_hull"] == 0]
    df_data = df_data[(df_data["band_gap"] > 0) & (df_data["band_gap"] <= 2)]

    el_filters = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [
            executor.submit(filter_elements, formula)
            for formula in tqdm(df_data["pretty_formula"])
        ]
        for f in tqdm(futures):
            el_filters.append(f.result())
    df_data = df_data[el_filters]
    comp_filters = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [
            executor.submit(filter_composition, formula)
            for formula in tqdm(df_data["pretty_formula"])
        ]
        for f in tqdm(futures):
            comp_filters.append(f.result())
    df_data = df_data[comp_filters]

    comp_list = df_data["pretty_formula"].unique()

    comp_feats = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [
            executor.submit(utils.get_feature, formula) for formula in tqdm(comp_list)
        ]
        for f in tqdm(futures):
            comp_feats.append(f.result())
    df_features = pd.DataFrame(comp_feats)

    return df_features, comp_list
