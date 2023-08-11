import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matminer.featurizers.composition.composite as composite
magpie_preset = composite.ElementProperty.from_preset("magpie")
import pymatgen.core as mg
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse

plt.rcParams['font.size'] = 11
plt.rcParams['font.family']= 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.direction'] = 'in'       # 目盛り線の向き、内側"in"か外側"out"かその両方"inout"か
plt.rcParams['ytick.direction'] = 'in'       # 目盛り線の向き、内側"in"か外側"out"かその両方"inout"か
plt.rcParams['xtick.major.width'] = 1.2      # x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.2      # y軸主目盛り線の線幅
plt.rcParams['xtick.major.size'] = 3         # x軸主目盛り線の長さ
plt.rcParams['ytick.major.size'] = 3         # y軸主目盛り線の長さ
#plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.grid']= False#True
plt.rcParams["axes.edgecolor"] = 'black'
plt.rcParams['grid.linestyle']= '--'
plt.rcParams['grid.linewidth'] = 0.3
plt.rcParams["legend.markerscale"] = 2
plt.rcParams["legend.fancybox"] = False      # Trueを指定すると凡例の枠の角が丸くなる
plt.rcParams["legend.framealpha"] = 1        # 判例の透明度
plt.rcParams["legend.edgecolor"] = 'black'


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


def build_model(hp):
    model = Sequential()
    model.add(
        Dense(
            units=hp.Int("input_units", min_value=32, max_value=512, step=32),
            input_dim=n_features,
            activation="sigmoid",
        )
    )  # 入力層
    model.add(
        Dropout(rate=hp.Float("dropout_1", min_value=0.0, max_value=0.5, step=0.01))
    )
    model.add(
        Dense(
            units=hp.Int("hidden_units", min_value=32, max_value=512, step=32),
            activation="sigmoid",
        )
    )  # 隠れ層
    model.add(
        Dropout(rate=hp.Float("dropout_2", min_value=0.0, max_value=0.5, step=0.01))
    )
    model.add(Dense(1, activation="linear"))  # 出力層
    model.compile(
        optimizer=Adam(
            learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="mean_squared_error",
    )


def get_tuned_model(X_train, y_train, X_test, y_test):
    tuner = kt.Hyperband(
        build_model,
        objective="val_loss",
        max_epochs=100,
        directory="output_dir",
        project_name="keras_tuning",
    )
    tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(
        f"""
    The hyperparameter search is complete. The optimal number of units in the input layer is {best_hps.get('input_units')}. 
    The optimal number of units in the hidden layer is {best_hps.get('hidden_units')}.
    The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
    The optimal dropout rate for the first and second dropout layers are {best_hps.get('dropout_1')} and {best_hps.get('dropout_2')} respectively.
    """
    )

    # Re-build the model with the optimal hyperparameters
    model = build_model(best_hps)
    history = model.fit(
        X_train, y_train, epochs=100, batch_size=1024, validation_data=(X_test, y_test)
    )
    model.save("models/tuned_model.keras")

    return model


def get_model(n_features, X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(288, input_dim=n_features, activation="sigmoid"))
    model.add(Dropout(0.12))
    model.add(Dense(224, activation="sigmoid"))
    model.add(Dropout(0.42))
    model.add(Dense(1, activation="linear"))
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    history = model.fit(
        X_train, y_train, epochs=100, batch_size=1024, validation_data=(X_test, y_test)
    )
    model.save("models/model.keras")

    return model


def get_final_model(n_features, X_final, y_final):
    model = Sequential()
    model.add(Dense(288, input_dim=n_features, activation="sigmoid"))
    model.add(Dropout(0.12))
    model.add(Dense(224, activation="sigmoid"))
    model.add(Dropout(0.42))
    model.add(Dense(1, activation="linear"))
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    filnal_history = model.fit(X_final, y_final, epochs=100, batch_size=1024)
    model.save("models/final_model.keras")

    return model


def main(args):
    data_path = args.data_path
    df_data = pd.read_csv(data_path)

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
    df_data["PF_calc"] = (df_data["Seebeck coefficient"] ** 2) * df_data[
        "Electrical conductivity"
    ]
    df_data["ZT_RAE"] = np.abs((df_data["ZT_calc"] - df_data["ZT"]) / df_data["ZT"])
    df_data = df_data[(df_data["ZT_RAE"] > 0) & (df_data["ZT_RAE"] < 0.4)].dropna()
    df_data["Z"] = df_data["ZT"] / df_data["Temperature"]
    df_data = df_data[(df_data["ZT"] > 0)].dropna()

    comp_feats = []
    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [
            executor.submit(get_feature, formula)
            for formula in tqdm(df_data["composition"])
        ]
        for f in tqdm(futures):
            comp_feats.append(f.result())

    df_features = pd.DataFrame(comp_feats)
    unique_key = "sid"
    inputprop = ["Temperature", unique_key]
    outputprop = [args.target]

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
    X_train = train_df.iloc[:, :-2]
    X_test = test_df.iloc[:, :-2]
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    y_train = train_df.iloc[:, -1]
    y_test = test_df.iloc[:, -1]
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)
    n_features = X_train.shape[1]

    X_combined = np.concatenate([X_train, X_test], axis=0)
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test], axis=0)
    y_combined = np.vstack([y_train, y_test])

    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X_train)
    X_test = scaler_x.transform(X_test)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    model = get_model(n_features, X_train, y_train, X_test, y_test)
    y_pred = model.predict(X_test)
    # スケーリングを元に戻す
    y_test_check = scaler_y.inverse_transform(y_test)
    y_pred_check = scaler_y.inverse_transform(y_pred)
    
    mse = mean_squared_error(y_test_check , y_pred_check )
    r2 = r2_score(y_test_check , y_pred_check )
    rmse = np.sqrt(mse)
    print(f"Test MSE: {mse}")
    print(f"Test R^2 score: {r2}")
    print(f"Test RMSE: {rmse}")

    fig = plt.figure(figsize=(4.5,4),dpi=300,facecolor='w',edgecolor='k')
    ax = fig.add_subplot(1,1,1)
    max_val = max(max(y_test_check), max(y_pred_check))
    min_val = min(min(y_test_check), min(y_pred_check))
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.set_xlim(min_val,max_val)
    ax.set_ylim(min_val,max_val)

    ax.scatter(y_test_check, y_pred_check)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.grid(True)
    ax.plot([min_val, max_val], [min_val, max_val], color='red')
    plt.savefig("results/parity_plot.png")

    scaler_x_final = StandardScaler()
    X_final = scaler_x_final.fit_transform(X_combined)
    scaler_y_final = StandardScaler()
    y_final = scaler_y_final.fit_transform(y_combined)

    with open('models/scaler_X_final.pkl', 'wb') as f:
        pickle.dump(scaler_x_final, f)
    with open('models/scaler_y_final.pkl', 'wb') as f:
        pickle.dump(scaler_y_final, f)

    final_model = get_final_model(n_features, X_final, y_final)


if __name__ == "__main__":
    seed_value = 0
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="datasets/20230406_interpolated_data.csv",
        type=str,
        help="Data path (default: datasets/20230406_interpolated_data.csv)",
    )
    parser.add_argument(
        "--target",
        default="ZT",
        type=str,
        help="target property (default: ZT)",
    )
    args = parser.parse_args()
    main(args)
