import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import argparse
from utils import utils
import matminer.featurizers.composition.composite as composite

magpie_preset = composite.ElementProperty.from_preset("magpie")
import warnings

warnings.filterwarnings("ignore")


def main(args):
    target = args.formula

    input_list = []
    T_list = list(range(300, 1100, 100))
    for T in T_list:
        tmp = list(utils.get_feature(target))
        tmp.append(T)
        input_list.append(tmp)

    df_input = pd.DataFrame(input_list)
    df_input = df_input.rename(columns={df_input.shape[1] - 1: "Temperature_input"})
    mp_x = df_input
    mp_x.columns = mp_x.columns.astype(str)

    with open("models/scaler_x_final.pkl", "rb") as f:
        scaler_x = pickle.load(f)
    with open("models/scaler_y_final.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    mp_x = scaler_x.transform(mp_x)

    loaded_model = load_model(args.model_path)
    print(loaded_model.summary())
    y_pred = loaded_model.predict(mp_x)
    y_pred = scaler_y.inverse_transform(y_pred)
    outputprop = [
        "Seebeck coefficient",
        "Electrical conductivity",
        "Thermal conductivity",
        "PF_calc",
        "ZT",
        "Temperature",
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
