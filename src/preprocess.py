import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

x_train = np.load("data/raw/x_train.npy")
x_test = np.load("data/raw/x_test.npy")
y_train = np.load("data/raw/y_train.npy")
y_test = np.load("data/raw/y_test.npy")

# Normalize pixel values to [0,1] and flatten
x_train = x_train.reshape(x_train.shape[0], -1).astype("float32") / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).astype("float32") / 255.0

np.save("data/processed/x_train.npy", x_train)
np.save("data/processed/x_test.npy", x_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)

print(f"Preprocessed data: x_train={x_train.shape}, x_test={x_test.shape}")