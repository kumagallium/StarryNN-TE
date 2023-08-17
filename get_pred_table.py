import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tqdm import tqdm
import argparse
from utils import preprocess
from utils import viz
import matminer.featurizers.composition.composite as composite

magpie_preset = composite.ElementProperty.from_preset("magpie")
import warnings

warnings.filterwarnings("ignore")


def main(args):
    data_path = args.data_path
    df_data = pd.read_csv(data_path)

    df_features, comp_list = preprocess.get_mp_features(df_data)
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
    mp_x = df_input
    mp_x.columns = mp_x.columns.astype(str)

    predictions = mp_x.copy()

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
    ]
    predictions = pd.concat(
        [predictions, pd.DataFrame(y_pred, columns=outputprop)], axis=1
    )
    predictions["composition"] = complist_T

    columns = ["composition", "Temperature"]
    columns.extend(outputprop)
    df_te_mat = predictions[columns]
    df_te_mat = df_te_mat.sort_values(by="ZT", ascending=False)
    df_te_mat.to_csv("results/pred_table.csv", index=False)

    for prop in outputprop:
        dict_formula = {}
        dict_results = {}
        for T in T_range:
            dict_formula[T] = list(
                df_te_mat[df_te_mat["Temperature"] == T].sort_values(
                    by=prop, ascending=False
                )["composition"]
            )
            dict_results[T] = list(
                df_te_mat[df_te_mat["Temperature"] == T].sort_values(
                    by=prop, ascending=False
                )[prop]
            )
        df_formula = pd.DataFrame(dict_formula)
        df_formula = df_formula.reset_index(drop=True)
        df_formula.index = df_formula.index + 1
        df_results = pd.DataFrame(dict_results)
        df_results = df_results.reset_index(drop=True)
        df_results.index = df_results.index + 1

        viz.pred_table(df_results, df_formula, prop)


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
