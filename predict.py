import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import argparse
from utils import preprocess
import matminer.featurizers.composition.composite as composite

magpie_preset = composite.ElementProperty.from_preset("magpie")
import warnings

warnings.filterwarnings("ignore")


def main(args):
    data_path = args.data_path
    df_data = pd.read_csv(data_path)

    df_features, comp_list = preprocess.get_mp_features(df_data)

    target = args.formula
    target_idx = np.where(comp_list == target)[0][0]
    T_list = list(range(300, 1100, 100))
    input_list = []
    for T in T_list:
        tmp = list(df_features.iloc[target_idx, :])
        tmp.append(T)
        input_list.append(tmp)
    df_input = pd.DataFrame(input_list).dropna()
    df_input = df_input.rename(columns={df_input.shape[1] - 1: "Temperature"})
    X = df_input
    X.columns = X.columns.astype(str)

    with open("models/scaler_x_final.pkl", "rb") as f:
        scaler_x = pickle.load(f)
    with open("models/scaler_y_final.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    X = scaler_x.transform(X)

    loaded_model = load_model(args.model_path)
    print(loaded_model.summary())
    y_pred = loaded_model.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred)
    outputprop = [
        "Seebeck coefficient",
        "Electrical conductivity",
        "Thermal conductivity",
        "PF_calc",
        "ZT",
    ]
    for idx, prop in enumerate(outputprop):
        if prop == "Thermal conductivity":
            unit = "[Wm-1K-1]"
        elif prop == "Seebeck coefficient":
            unit = "[uVK-1]"
        elif prop == "Electrical conductivity":
            unit = "[Ω-1m-1]"
        elif prop == "PF_calc":
            unit = "[mWm-1K-2]"
        elif prop == "ZT":
            unit = ""
        print(prop + unit)
        print(dict(zip(T_list, y_pred.T[idx])))
        print("\n")


if __name__ == "__main__":
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
