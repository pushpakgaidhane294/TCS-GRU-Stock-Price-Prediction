"""Generate the latest TCS next-trading-day GRU prediction.

Run this script locally from the project root before deploying a new
prediction. It refreshes the last three months of TCS data, merges it
with data/TCS_stock.csv, runs the trained GRU model once, and saves the
result to data/latest_prediction.json.
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model


TICKER = "TCS.NS"
DATA_PATH = "data/TCS_stock.csv"
MODEL_PATH = "model/gru_model.keras"
SCALER_PATH = "model/scaler.pkl"
OUTPUT_PATH = "data/latest_prediction.json"
SEQUENCE_LENGTH = 60


def clean_data(data):
    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Date" not in data.columns or "Close" not in data.columns:
        raise ValueError("Data must contain Date and Close columns.")

    data = data[["Date", "Close"]].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Close"])
    data = data.drop_duplicates(subset=["Date"], keep="last")
    data = data.sort_values("Date").reset_index(drop=True)

    return data


def refresh_data():
    """Refresh recent TCS data and update the local CSV."""
    print("Refreshing recent TCS market data...")

    existing = pd.read_csv(DATA_PATH)
    existing = clean_data(existing)

    try:
        recent = yf.download(
            TICKER,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=15,
        )

        if recent.empty:
            raise ValueError("Yahoo Finance returned no recent data.")

        recent = recent.reset_index()
        recent = clean_data(recent)

        combined = pd.concat(
            [existing, recent],
            ignore_index=True,
        )
        combined = clean_data(combined)
        combined.to_csv(DATA_PATH, index=False)

        print(f"Updated {DATA_PATH}: {len(combined)} rows")
        return combined

    except Exception as exc:
        print(f"Yahoo Finance refresh failed: {exc}")
        print("Using the existing local CSV instead.")
        return existing


def generate_prediction(data):
    if len(data) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least {SEQUENCE_LENGTH} closing prices for prediction."
        )

    print("Loading GRU model...")
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)

    recent_prices = (
        data["Close"]
        .tail(SEQUENCE_LENGTH)
        .astype(np.float32)
        .to_numpy()
    )

    latest_price = float(recent_prices[-1])
    latest_date = data["Date"].iloc[-1].strftime("%d %b %Y")

    scaled_prices = scaler.transform(
        recent_prices.reshape(-1, 1)
    )

    X_latest = scaled_prices.reshape(
        1,
        SEQUENCE_LENGTH,
        1,
    ).astype(np.float32)

    print("Running GRU prediction...")

    # One prediction only. This runs locally, not on Render.
    predicted_scaled = model.predict(
        X_latest,
        batch_size=1,
        verbose=0,
    )

    predicted_price = float(
        scaler.inverse_transform(
            np.asarray(predicted_scaled).reshape(-1, 1)
        )[0][0]
    )

    predicted_price = round(predicted_price, 2)
    difference = round(predicted_price - latest_price, 2)

    percentage_change = round(
        (difference / latest_price) * 100,
        2,
    ) if latest_price else 0.0

    if difference > 0:
        direction = "UP"
        direction_symbol = "📈"
    elif difference < 0:
        direction = "DOWN"
        direction_symbol = "📉"
    else:
        direction = "NEUTRAL"
        direction_symbol = "➖"

    prediction = {
        "ticker": TICKER,
        "latest_date": latest_date,
        "latest_price": round(latest_price, 2),
        "predicted_price": predicted_price,
        "difference": difference,
        "percentage_change": percentage_change,
        "direction": direction,
        "direction_symbol": direction_symbol,
        "sequence_length": SEQUENCE_LENGTH,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Write to a temporary file first, then replace the old result.
    temp_path = OUTPUT_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(prediction, file, indent=2, ensure_ascii=False)

    os.replace(temp_path, OUTPUT_PATH)

    return prediction


def main():
    data = refresh_data()
    prediction = generate_prediction(data)

    print("\n========================================")
    print("LATEST TCS GRU PREDICTION")
    print("========================================")
    print(f"Latest Date       : {prediction['latest_date']}")
    print(f"Latest Price      : ₹{prediction['latest_price']:.2f}")
    print(f"Predicted Price   : ₹{prediction['predicted_price']:.2f}")
    print(f"Expected Change   : {prediction['percentage_change']:.2f}%")
    print(f"Direction          : {prediction['direction_symbol']} {prediction['direction']}")
    print(f"Saved To           : {OUTPUT_PATH}")
    print("========================================")


if __name__ == "__main__":
    main()
