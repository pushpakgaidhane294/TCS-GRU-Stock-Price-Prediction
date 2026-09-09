import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/TCS_stock.csv"
MODEL_PATH = "model/gru_model.keras"
SCALER_PATH = "model/scaler.pkl"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.dropna(subset=["Close"])

df = df.reset_index(drop=True)


prices = df["Close"].values.reshape(-1, 1)


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_size = int(len(prices) * 0.80)

train_prices = prices[:train_size]

test_prices = prices[train_size:]


# ============================================================
# SCALE DATA
# ============================================================

train_scaled = scaler.transform(train_prices)

test_scaled = scaler.transform(test_prices)


# ============================================================
# CREATE TEST SEQUENCES
# ============================================================

sequence_length = 60


combined_test = np.concatenate(
    (
        train_scaled[-sequence_length:],
        test_scaled
    ),
    axis=0
)


X_test = []
y_test = []


for i in range(
    sequence_length,
    len(combined_test)
):

    X_test.append(
        combined_test[
            i-sequence_length:i
        ]
    )

    y_test.append(
        combined_test[i]
    )


X_test = np.array(X_test)

y_test = np.array(y_test)


# ============================================================
# PREDICT
# ============================================================

print("Generating predictions...")


predictions_scaled = model.predict(
    X_test,
    verbose=0
)


# Convert back to actual prices

predictions = scaler.inverse_transform(
    predictions_scaled
)

actual = scaler.inverse_transform(
    y_test
)


# ============================================================
# CREATE ACTUAL VS PREDICTED GRAPH
# ============================================================

test_dates = df["Date"].iloc[train_size:].reset_index(
    drop=True
)


plt.figure(figsize=(12, 6))


plt.plot(
    test_dates,
    actual.flatten(),
    label="Actual Price"
)


plt.plot(
    test_dates,
    predictions.flatten(),
    label="Predicted Price"
)


plt.title(
    "TCS Actual vs Predicted Stock Price - GRU"
)


plt.xlabel("Date")

plt.ylabel("Closing Price (₹)")


plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()


plt.savefig(
    "static/actual_vs_predicted.png",
    dpi=150
)


plt.close()


print("\nActual vs Predicted graph regenerated successfully!")

print(
    "Saved to: static/actual_vs_predicted.png"
)


# ============================================================
# HISTORICAL GRAPH
# ============================================================

plt.figure(figsize=(12, 6))


plt.plot(
    df["Date"],
    df["Close"]
)


plt.title(
    "TCS Historical Closing Price"
)


plt.xlabel("Date")

plt.ylabel("Closing Price (₹)")


plt.xticks(rotation=45)

plt.tight_layout()


plt.savefig(
    "static/tcs_historical_price.png",
    dpi=150
)


plt.close()


print(
    "Historical graph regenerated successfully!"
)


print(
    "Saved to: static/tcs_historical_price.png"
)


print("\nAll graphs updated successfully!")