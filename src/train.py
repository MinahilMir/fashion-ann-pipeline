import numpy as np
import os
import yaml
from tensorflow import keras

with open("params.yaml") as f:
    params = yaml.safe_load(f)

epochs = params["train"]["epochs"]
batch_size = params["train"]["batch_size"]
learning_rate = params["train"]["learning_rate"]

x_train = np.load("data/processed/x_train.npy")
y_train = np.load("data/processed/y_train.npy")

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

os.makedirs("model", exist_ok=True)
model.save("model/ann_model.h5")

print("Model trained and saved to model/ann_model.h5")