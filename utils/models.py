import numpy as np
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import pickle
from utils import viz
import tensorflow as tf
import random

seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


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
    model.add(Dense(6, activation="linear"))
    model.compile(
        optimizer=Adam(
            learning_rate=hp.Choice(
                "learning_rate", values=[1e-2, 1e-3, 1e-4], default=0.01
            )
        ),
        loss="mean_squared_error",
    )

    return model


def tuning(input_dim, x_train, y_train, x_test, y_test):
    print("tuninig:", input_dim)
    tuner = kt.Hyperband(
        lambda hp: build_model(hp, input_dim=input_dim),
        objective="val_loss",
        max_epochs=100,
        directory="models/output_dir",
        project_name="keras_tuning",
    )
    tuner.search(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
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
    model.add(Dense(64, input_dim=input_dim, activation="sigmoid"))
    model.add(Dropout(0.02))
    model.add(Dense(480, activation="sigmoid"))
    model.add(Dropout(0.42))
    model.add(Dense(6, activation="linear"))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mean_squared_error")

    return model


def get_model(is_tuning, input_dim, x_train, y_train, x_test, y_test):
    if is_tuning == 1:
        best_hps = tuning(input_dim, x_train, y_train, x_test, y_test)
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
        x_train, y_train, epochs=100, batch_size=1024, validation_data=(x_test, y_test)
    )

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = list(range(1, len(train_loss) + 1))

    viz.loss_plot(epochs, train_loss, val_loss)

    if is_tuning == 1:
        model.save("models/tuned_model.keras")
        model.save("models/model.keras")
    else:
        model.save("models/model.keras")

    return model


def get_final_model(is_tuning, input_dim, x_train, y_train):
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
    history = model.fit(x_train, y_train, epochs=100, batch_size=1024)
    if is_tuning == 1:
        model.save("models/final_tuned_model.keras")
        model.save("models/final_model.keras")
    else:
        model.save("models/final_model.keras")

    return model
