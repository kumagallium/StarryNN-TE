import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import argparse
import pickle
import models
from utils import viz
from utils import preprocess
from utils import models
import matminer.featurizers.composition.composite as composite

magpie_preset = composite.ElementProperty.from_preset("magpie")
import warnings

warnings.filterwarnings("ignore")


def main(args):
    data_path = args.data_path
    df_data = pd.read_csv(data_path)

    outputprop = [
        "Seebeck coefficient",
        "Electrical conductivity",
        "Thermal conductivity",
        "PF_calc",
        "ZT",
        "Temperature",
    ]

    (
        input_dim,
        x_train,
        x_test,
        y_train,
        y_test,
        x_combined,
        y_combined,
    ) = preprocess.get_train_test_dataset(df_data, outputprop)

    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    model = models.get_model(
        args.is_tuning, input_dim, x_train, y_train, x_test, y_test
    )

    y_pred = model.predict(x_test)

    scaler_x_final = StandardScaler()
    x_final = scaler_x_final.fit_transform(x_combined)
    scaler_y_final = StandardScaler()
    y_final = scaler_y_final.fit_transform(y_combined)

    with open("models/scaler_x_final.pkl", "wb") as f:
        pickle.dump(scaler_x_final, f)
    with open("models/scaler_y_final.pkl", "wb") as f:
        pickle.dump(scaler_y_final, f)

    y_test_check = scaler_y.inverse_transform(y_test)
    y_pred_check = scaler_y.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_check, y_pred_check)
    r2 = r2_score(y_test_check, y_pred_check)
    rmse = np.sqrt(mse)
    print(f"Test MSE: {mse}")
    print(f"Test R^2 score: {r2}")
    print(f"Test RMSE: {rmse}")

    for idx, tg in enumerate(outputprop):
        viz.parity_plot(tg, y_test_check.T[idx], y_pred_check.T[idx])

    final_model = models.get_final_model(args.is_tuning, input_dim, x_final, y_final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="datasets/20230406_interpolated_data.csv",
        type=str,
        help="Data path (default: datasets/20230406_interpolated_data.csv)",
    )
    parser.add_argument(
        "--is_tuning",
        default=0,
        type=int,
        help="tuning flug (default: 0)",
    )
    args = parser.parse_args()
    main(args)
