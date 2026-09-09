import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout


# ============================================================
# LOAD DATA
# ============================================================

file_path = "data/TCS_stock.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.dropna(subset=["Close"])

df = df.reset_index(drop=True)


prices = df["Close"].values.reshape(-1, 1)


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

train_size = int(len(prices) * 0.80)

train_prices = prices[:train_size]

test_prices = prices[train_size:]


print("=" * 50)
print("GRU STOCK PRICE PREDICTION")
print("=" * 50)

print("\nTotal rows:", len(prices))

print("Training rows:", len(train_prices))

print("Testing rows:", len(test_prices))


# ============================================================
# SCALE DATA
# ============================================================

scaler = MinMaxScaler(feature_range=(0, 1))

# IMPORTANT:
# Fit scaler ONLY on training data
scaler.fit(train_prices)

train_scaled = scaler.transform(train_prices)

test_scaled = scaler.transform(test_prices)


# ============================================================
# CREATE TRAINING SEQUENCES
# ============================================================

sequence_length = 60

X_train = []
y_train = []


for i in range(
    sequence_length,
    len(train_scaled)
):

    X_train.append(
        train_scaled[
            i-sequence_length:i
        ]
    )

    y_train.append(
        train_scaled[i]
    )


X_train = np.array(X_train)

y_train = np.array(y_train)


# ============================================================
# CREATE TEST SEQUENCES
# ============================================================

# We need the last 60 training days as context
# for the first test prediction.

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


print("\nSequence shapes:")

print("X_train:", X_train.shape)

print("y_train:", y_train.shape)

print("X_test:", X_test.shape)

print("y_test:", y_test.shape)


# ============================================================
# BUILD GRU MODEL
# ============================================================

model = Sequential()


model.add(
    GRU(
        64,
        return_sequences=True,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )
)


model.add(
    Dropout(0.2)
)


model.add(
    GRU(32)
)


model.add(
    Dropout(0.2)
)


model.add(
    Dense(1)
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\nStarting GRU training...")


history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating test predictions...")


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
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    actual,
    predictions
)


rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)


print("\n" + "=" * 50)

print("MODEL RESULTS")

print("=" * 50)

print(
    "MAE:",
    round(mae, 2)
)

print(
    "RMSE:",
    round(rmse, 2)
)


# ============================================================
# NEXT DAY PREDICTION
# ============================================================

latest_60 = prices[-60:]


latest_60_scaled = scaler.transform(
    latest_60
)


latest_60_scaled = latest_60_scaled.reshape(
    1,
    60,
    1
)


next_prediction_scaled = model.predict(
    latest_60_scaled,
    verbose=0
)


next_prediction = scaler.inverse_transform(
    next_prediction_scaled
)


next_price = float(
    next_prediction[0][0]
)


print(
    "\nPredicted next trading day price:",
    round(next_price, 2)
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs("model", exist_ok=True)


model.save(
    "model/gru_model.keras"
)


joblib.dump(
    scaler,
    "model/scaler.pkl"
)


print("\nModel saved:")
print("model/gru_model.keras")

print("\nScaler saved:")
print("model/scaler.pkl")


print("\nTraining completed successfully!")