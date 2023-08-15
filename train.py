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
import os

plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["xtick.major.size"] = 3
plt.rcParams["ytick.major.size"] = 3
# plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.grid"] = False  # True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.3
plt.rcParams["legend.markerscale"] = 2
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
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

    base_material = {el: amt for el, amt in composition.items() if amt >= 0.1}
    dopants = {el: amt for el, amt in composition.items() if amt < 0.1}

    return {"base_material": base_material, "dopants": dopants}


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


def build_model(hp, input_dim):
    model = Sequential()

    model.add(
        Dense(
            units=hp.Int(
                "input_units", min_value=32, max_value=512, step=32, default=288
            ),
            input_dim=input_dim,
            activation="sigmoid",
        )
    )

    model.add(
        Dropout(
            rate=hp.Float(
                "dropout_1", min_value=0.0, max_value=0.5, default=0.12, step=0.01
            )
        )
    )

    model.add(
        Dense(
            units=hp.Int(
                "hidden_units", min_value=32, max_value=512, step=32, default=224
            ),
            activation="sigmoid",
        )
    )

    model.add(
        Dropout(
            rate=hp.Float(
                "dropout_2", min_value=0.0, max_value=0.5, default=0.42, step=0.01
            )
        )
    )

    model.add(Dense(5, activation="linear"))

    model.compile(
        optimizer=Adam(
            learning_rate=hp.Choice(
                "learning_rate", values=[1e-2, 1e-3, 1e-4], default=0.01
            )
        ),
        loss="mean_squared_error",
    )

    return model


def tuning(input_dim, X_train, y_train, X_test, y_test):
    print("tuninig:", input_dim)
    tuner = kt.Hyperband(
        lambda hp: build_model(hp, input_dim=input_dim),
        objective="val_loss",
        max_epochs=100,
        directory="models/output_dir",
        project_name="keras_tuning",
    )
    tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    with open("models/best_hps.pkl", "wb") as f:
        pickle.dump(best_hps, f)
    print(
        f"""
    The hyperparameter search is complete. The optimal number of units in the input layer is {best_hps.get('input_units')}. 
    The optimal number of units in the hidden layer is {best_hps.get('hidden_units')}.
    The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
    The optimal dropout rate for the first and second dropout layers are {best_hps.get('dropout_1')} and {best_hps.get('dropout_2')} respectively.
    """
    )

    return best_hps


def set_nn(input_dim):
    model = Sequential()
    model.add(Dense(288, input_dim=input_dim, activation="sigmoid"))
    model.add(Dropout(0.17))
    model.add(Dense(416, activation="sigmoid"))
    model.add(Dropout(0.32))
    model.add(Dense(5, activation="linear"))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mean_squared_error")

    return model


def get_model(is_tuning, input_dim, X_train, y_train, X_test, y_test):
    if is_tuning == 1:
        best_hps = tuning(input_dim, X_train, y_train, X_test, y_test)
        model = build_model(best_hps, input_dim)
    else:
        if os.path.exists("models/best_hps.pkl"):
            with open("models/best_hps.pkl", "rb") as f:
                loaded_best_hps = pickle.load(f)
            print(
                f"""
            The hyperparameter search is complete. The optimal number of units in the input layer is {loaded_best_hps.get('input_units')}. 
            The optimal number of units in the hidden layer is {loaded_best_hps.get('hidden_units')}.
            The optimal learning rate for the optimizer is {loaded_best_hps.get('learning_rate')}.
            The optimal dropout rate for the first and second, third dropout layers are {loaded_best_hps.get('dropout_1')} and {loaded_best_hps.get('dropout_2')} respectively.
            """
            )
            model = build_model(loaded_best_hps, input_dim)
        else:
            model = set_nn(input_dim)
    history = model.fit(
        X_train, y_train, epochs=100, batch_size=1024, validation_data=(X_test, y_test)
    )

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = list(range(1, len(train_loss) + 1))

    fig = plt.figure(figsize=(4, 4), dpi=300, facecolor="w", edgecolor="k")
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.set_xlim(min(epochs), max(epochs))
    ax.set_ylim(0, max(max(train_loss), max(val_loss)))

    ax.plot(epochs, train_loss, "bo", label="Training loss")
    ax.plot(epochs, val_loss, "b", label="Validation loss")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig("results/loss_plot.png")

    if is_tuning == 1:
        model.save("models/tuned_model.keras")
    else:
        model.save("models/model.keras")

    return model


def get_final_model(is_tuning, input_dim, X_train, y_train):
    if is_tuning == 1:
        with open("models/best_hps.pkl", "rb") as f:
            loaded_best_hps = pickle.load(f)
        model = build_model(loaded_best_hps, input_dim)
    else:
        if os.path.exists("models/best_hps.pkl"):
            with open("models/best_hps.pkl", "rb") as f:
                loaded_best_hps = pickle.load(f)
            model = build_model(loaded_best_hps, input_dim)
        else:
            model = set_nn(input_dim)
    history = model.fit(X_train, y_train, epochs=100, batch_size=1024)
    if is_tuning == 1:
        model.save("models/final_tuned_model.keras")
    else:
        model.save("models/final_model.keras")

    return model


def viz_parity_plot(target, true, pred):
    fig = plt.figure(figsize=(4, 4), dpi=300, facecolor="w", edgecolor="k")
    ax = fig.add_subplot(1, 1, 1)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    ax.scatter(true, pred)

    if target == "Thermal conductivity":
        ax.set_xlabel("Experimental $\u03BA$ [Wm$^{-1}$K$^{-1}$]")
        ax.set_ylabel("Predicted  $\u03BA$ [Wm$^{-1}$K$^{-1}$]")
        ax_min = 0
        ax_max = 20
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
    elif target == "Seebeck coefficient":
        ax.set_xlabel("Experimental $S$ [\u03BCVK$^{-1}$]")
        ax.set_ylabel("Predicted $S$ [\u03BCVK$^{-1}$]")
        ax_min = 0
        ax_max = 1000
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
    elif target == "Electrical conductivity":
        ax.set_xlabel("Experimental $\u03C3$ [\u03A9$^{-1}$m$^{-1}$]")
        ax.set_ylabel("Predicted  $\u03C3$ [\u03A9$^{-1}$m$^{-1}$]")
        ax_min = 0
        ax_max = 1000000
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
    elif target == "PF_calc":
        ax.set_xlabel("Experimental $PF_{ \mathrm{calc}}$ [mWm$^{-1}$K$^{-2}$]")
        ax.set_ylabel("Predicted $PF_{ \mathrm{calc}}$ [mWm$^{-1}$K$^{-2}$]")
        ax_min = 0
        ax_max = 10
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
    elif target == "ZT":
        ax.set_xlabel("Experimental $ZT$")
        ax.set_ylabel("Predicted $ZT$")
        ax_min = 0
        ax_max = 2
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)

    ax.grid(True)
    ax.plot([ax_min, ax_max], [ax_min, ax_max], color="red")
    plt.savefig("results/parity_plot_" + target.replace(" ", "_") + ".png")


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
    df_data["PF_calc"] = (
        (df_data["Seebeck coefficient"] ** 2)
        * df_data["Electrical conductivity"]
        * 10**3
    )
    df_data["ZT_RAE"] = np.abs((df_data["ZT_calc"] - df_data["ZT"]) / df_data["ZT"])
    df_data = df_data[(df_data["ZT_RAE"] > 0) & (df_data["ZT_RAE"] < 0.2)].dropna()
    df_data["Z"] = df_data["ZT"] / df_data["Temperature"]
    df_data = df_data[(df_data["ZT"] > 0)].dropna()
    df_data["Seebeck coefficient"] = np.abs(df_data["Seebeck coefficient"]) * 10**6

    elements_list = []
    for comp in df_data["composition"].unique():
        elements_list.extend(get_elements(comp))
    with open("models/starry_elements.pkl", "wb") as f:
        pickle.dump(elements_list, f)

    frac_comp_list = []
    for comp in df_data["composition"].unique():
        frac_comp_list.append(convert_fractional_composition(comp))
    with open("models/starry_comosition.pkl", "wb") as f:
        pickle.dump(frac_comp_list, f)

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
    outputprop = [
        "Seebeck coefficient",
        "Electrical conductivity",
        "Thermal conductivity",
        "PF_calc",
        "ZT",
    ]

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
    X_train = train_df.iloc[:, : -1 * (len(outputprop) + 1)]
    X_test = test_df.iloc[:, : -1 * (len(outputprop) + 1)]
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    y_train = train_df.iloc[:, -1 * (len(outputprop)) :]
    y_test = test_df.iloc[:, -1 * (len(outputprop)) :]
    y_train = y_train.values.reshape(-1, len(outputprop))
    y_test = y_test.values.reshape(-1, len(outputprop))
    input_dim = X_train.shape[1]

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

    model = get_model(args.is_tuning, input_dim, X_train, y_train, X_test, y_test)

    y_pred = model.predict(X_test)
    y_test_check = scaler_y.inverse_transform(y_test)
    y_pred_check = scaler_y.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_check, y_pred_check)
    r2 = r2_score(y_test_check, y_pred_check)
    rmse = np.sqrt(mse)
    print(f"Test MSE: {mse}")
    print(f"Test R^2 score: {r2}")
    print(f"Test RMSE: {rmse}")

    for idx, tg in enumerate(outputprop):
        viz_parity_plot(tg, y_test_check.T[idx], y_pred_check.T[idx])

    scaler_x_final = StandardScaler()
    X_final = scaler_x_final.fit_transform(X_combined)
    scaler_y_final = StandardScaler()
    y_final = scaler_y_final.fit_transform(y_combined)

    with open("models/scaler_X_final.pkl", "wb") as f:
        pickle.dump(scaler_x_final, f)
    with open("models/scaler_y_final.pkl", "wb") as f:
        pickle.dump(scaler_y_final, f)

    final_model = get_final_model(args.is_tuning, input_dim, X_final, y_final)


if __name__ == "__main__":
    seed_value = 0
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="datasets/20210216_interpolated_data.csv",
        type=str,
        help="Data path (default: datasets/20210216_interpolated_data.csv)",
    )
    parser.add_argument(
        "--is_tuning",
        default=0,
        type=int,
        help="tuning flug (default: 0)",
    )
    args = parser.parse_args()
    main(args)
