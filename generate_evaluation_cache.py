import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

MODEL_PATH = "model/gru_model.keras"
SCALER_PATH = "model/scaler.pkl"
DATA_PATH = "data/TCS_stock.csv"
OUTPUT_PATH = "data/model_evaluation.json"
SEQUENCE_LENGTH = 60

print("Loading model and data...")
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
data = pd.read_csv(DATA_PATH)
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
data = data.dropna(subset=["Date", "Close"])
data = data.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

data_values = data["Close"].values.astype(np.float32).reshape(-1, 1)
train_size = int(len(data_values) * 0.80)
train_data = data_values[:train_size]
test_data = data_values[train_size:]

train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)
combined = np.concatenate([train_scaled[-SEQUENCE_LENGTH:], test_scaled])

X_test = []
for i in range(SEQUENCE_LENGTH, len(combined)):
    X_test.append(combined[i-SEQUENCE_LENGTH:i])

X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(test_scaled, dtype=np.float32)

print(f"Running evaluation once on {len(X_test)} test sequences...")
predicted_scaled = model.predict(X_test, batch_size=32, verbose=0)
predicted = scaler.inverse_transform(predicted_scaled).flatten()
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae = float(mean_absolute_error(actual, predicted))
rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
mask = actual != 0
mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if np.any(mask) else 100.0
accuracy = float(max(0.0, 100.0 - mape))

test_dates = data["Date"].iloc[train_size:].reset_index(drop=True)

result = {
    "mae": mae,
    "rmse": rmse,
    "accuracy": accuracy,
    "actual_prices": [float(x) for x in actual],
    "predicted_prices": [float(x) for x in predicted],
    "test_dates": [d.strftime("%Y-%m-%d") for d in test_dates]
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, separators=(",", ":"))

print(f"Saved: {OUTPUT_PATH}")
print(f"MAE: ₹{mae:.2f}")
print(f"RMSE: ₹{rmse:.2f}")
print(f"Accuracy: {accuracy:.2f}%")
