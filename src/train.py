import numpy as np
import os
import pandas as pd
import yaml
from tensorflow import keras

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")
x_val = np.load("data/processed/x_val.npy")
y_val = np.load("data/processed/y_val.npy")

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Flatten(),
    keras.layers.Dense(params["dense_units"], activation="relu"),
    keras.layers.Dropout(params["dropout_rate"]),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
)

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")
pd.DataFrame(history.history).to_csv("models/history.csv", index=False)

print("Saved models/model.h5 and models/history.csv")