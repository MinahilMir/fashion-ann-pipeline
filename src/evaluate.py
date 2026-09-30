import numpy as np
import json
from tensorflow import keras

x_test = np.load("data/processed/x_test.npy")
y_test = np.load("data/processed/y_test.npy")

model = keras.models.load_model("model/ann_model.h5")

loss, accuracy = model.evaluate(x_test, y_test)

metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy)}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")