import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


# ============================================================
# LOAD DATA
# ============================================================

file_path = "data/TCS_stock.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.dropna(subset=["Close"])

df = df.reset_index(drop=True)


print("=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)

print("\nTotal rows:", len(df))


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

prices = df["Close"].values.reshape(-1, 1)

train_size = int(len(prices) * 0.80)

train_prices = prices[:train_size]

test_prices = prices[train_size:]


print("\nTraining rows:", len(train_prices))
print("Testing rows:", len(test_prices))


# ============================================================
# FIT SCALER ONLY ON TRAINING DATA
# ============================================================

scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(train_prices)


# Transform training and testing data
train_scaled = scaler.transform(train_prices)

test_scaled = scaler.transform(test_prices)


print("\nScaler fitted ONLY on training data.")

print("Training scaled minimum:",
      train_scaled.min())

print("Training scaled maximum:",
      train_scaled.max())


# ============================================================
# SAVE SCALER
# ============================================================

os.makedirs("model", exist_ok=True)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)


print("\nScaler saved successfully!")

print("Location: model/scaler.pkl")