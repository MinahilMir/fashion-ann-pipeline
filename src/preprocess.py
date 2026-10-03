import numpy as np
import os
from sklearn.model_selection import train_test_split

os.makedirs("data/processed", exist_ok=True)

x_train = np.load("data/raw/x_train.npy")
x_test = np.load("data/raw/x_test.npy")
y_train = np.load("data/raw/y_train.npy")
y_test = np.load("data/raw/y_test.npy")

x_train = (x_train.reshape(x_train.shape[0], -1).astype("float32") - 127.5) / 127.5
x_test = (x_test.reshape(x_test.shape[0], -1).astype("float32") - 127.5) / 127.5

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

np.save("data/processed/x_train.npy", x_train)
np.save("data/processed/x_val.npy", x_val)
np.save("data/processed/x_test.npy", x_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_val.npy", y_val)
np.save("data/processed/y_test.npy", y_test)

print(f"train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")