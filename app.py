from flask import Flask, render_template, request, redirect, url_for, session, send_file
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

import sqlite3
import os
import io
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from tensorflow.keras.models import load_model  # type: ignore[import-not-found,reportMissingModuleSource]
except ImportError:  # pragma: no cover
    load_model = None  # type: ignore[assignment]


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = "gru_stock_prediction_secret_key"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ============================================================
# PROJECT PATHS
# ============================================================

MODEL_PATH = "model/gru_model.keras"
SCALER_PATH = "model/scaler.pkl"
DATA_PATH = "data/TCS_stock.csv"
DATABASE = "database.db"

SEQUENCE_LENGTH = 60
TICKER = "TCS.NS"


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

print("Loading GRU model...")

model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

print("GRU model loaded successfully.")


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

historical_data = pd.read_csv(DATA_PATH)

historical_data["Date"] = pd.to_datetime(
    historical_data["Date"]
)

historical_data["Close"] = pd.to_numeric(
    historical_data["Close"],
    errors="coerce"
)

historical_data = historical_data.dropna(
    subset=["Close"]
).reset_index(drop=True)


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn


def init_db():

    conn = get_db_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT
        )
        """
    )

    columns = [
        row["name"]
        for row in conn.execute(
            "PRAGMA table_info(users)"
        ).fetchall()
    ]

    if "name" not in columns:
        conn.execute(
            "ALTER TABLE users ADD COLUMN name TEXT"
        )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction_date TEXT NOT NULL,
            latest_price REAL NOT NULL,
            predicted_price REAL NOT NULL,
            difference REAL NOT NULL,
            percentage_change REAL NOT NULL,
            direction TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


init_db()


# ============================================================
# USER CLASS
# ============================================================

class User(UserMixin):

    def __init__(self, user_id, username, name):

        self.id = user_id
        self.username = username
        self.name = name


@login_manager.user_loader
def load_user(user_id):

    conn = get_db_connection()

    columns = [
        row["name"]
        for row in conn.execute(
            "PRAGMA table_info(users)"
        ).fetchall()
    ]

    if "username" in columns:

        user = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

        if user:

            username = user["username"]

        else:

            conn.close()
            return None

    elif "email" in columns:

        user = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

        if user:

            username = user["email"]

        else:

            conn.close()
            return None

    else:

        conn.close()
        return None

    conn.close()

    if user and "name" in user.keys():
        full_name = user["name"]
    elif user:
        full_name = username
    else:
        return None

    return User(
        user["id"],
        username,
        full_name
    )


# ============================================================
# PREDICTION ACCURACY
# ============================================================

def calculate_prediction_accuracy(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0

    if not np.any(mask):
        return 0.0

    mape = np.mean(
        np.abs((actual[mask] - predicted[mask]) / actual[mask])
    ) * 100

    return max(0.0, 100.0 - mape)


# ============================================================
# PREDICTION HISTORY HELPERS
# ============================================================

def save_prediction_history(
    user_id,
    prediction_date,
    latest_price,
    predicted_price,
    difference,
    percentage_change,
    direction
):
    conn = get_db_connection()

    existing = conn.execute(
        """
        SELECT id
        FROM prediction_history
        WHERE user_id = ?
        AND prediction_date = ?
        """,
        (user_id, prediction_date)
    ).fetchone()

    if existing:
        conn.close()
        return

    conn.execute(
        """
        INSERT INTO prediction_history
        (
            user_id,
            prediction_date,
            latest_price,
            predicted_price,
            difference,
            percentage_change,
            direction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            prediction_date,
            latest_price,
            predicted_price,
            difference,
            percentage_change,
            direction
        )
    )

    conn.commit()
    conn.close()


def get_prediction_history(user_id, limit=10):
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT prediction_date, latest_price, predicted_price, difference, percentage_change, direction
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit)
    ).fetchall()
    conn.close()
    return rows


# ============================================================
# MODEL EVALUATION
# ============================================================

def calculate_model_results():

    data = historical_data["Close"].values.reshape(-1, 1)

    total_rows = len(data)

    train_size = int(total_rows * 0.80)

    train_data = data[:train_size]

    test_data = data[train_size:]

    # --------------------------------------------------------
    # Scaler is already fitted only on training data
    # --------------------------------------------------------

    train_scaled = scaler.transform(train_data)

    test_scaled = scaler.transform(test_data)

    # --------------------------------------------------------
    # TRAIN SEQUENCES
    # --------------------------------------------------------

    X_train = []
    y_train = []

    for i in range(
        SEQUENCE_LENGTH,
        len(train_scaled)
    ):

        X_train.append(
            train_scaled[
                i - SEQUENCE_LENGTH:i
            ]
        )

        y_train.append(
            train_scaled[i]
        )

    X_train = np.array(X_train)

    y_train = np.array(y_train)

    # --------------------------------------------------------
    # TEST SEQUENCES
    # --------------------------------------------------------

    combined_test = np.concatenate(
        (
            train_scaled[-SEQUENCE_LENGTH:],
            test_scaled
        )
    )

    X_test = []
    y_test = []

    for i in range(
        SEQUENCE_LENGTH,
        len(combined_test)
    ):

        X_test.append(
            combined_test[
                i - SEQUENCE_LENGTH:i
            ]
        )

        y_test.append(
            combined_test[i]
        )

    X_test = np.array(X_test)

    y_test = np.array(y_test)

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    )

    # --------------------------------------------------------
    # CONVERT BACK TO ORIGINAL PRICE
    # --------------------------------------------------------

    predictions = scaler.inverse_transform(
        predictions_scaled
    ).flatten()

    actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TEST DATES
    # --------------------------------------------------------

    test_dates = historical_data[
        "Date"
    ].iloc[train_size:].reset_index(drop=True)

    return (
        mae,
        rmse,
        actual,
        predictions,
        test_dates
    )


# ============================================================
# GET LATEST TCS DATA
# ============================================================

def get_latest_tcs_data():

    try:

        print(
            "Fetching latest TCS data from Yahoo Finance..."
        )

        latest_data = yf.download(
            TICKER,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=10
        )

        if latest_data.empty:

            raise Exception(
                "Yahoo Finance returned empty data."
            )

        # ----------------------------------------------------
        # Handle MultiIndex columns
        # ----------------------------------------------------

        if isinstance(
            latest_data.columns,
            pd.MultiIndex
        ):

            latest_data.columns = [
                column[0]
                for column in latest_data.columns
            ]

        latest_data = latest_data.reset_index()

        # ----------------------------------------------------
        # Find date column
        # ----------------------------------------------------

        if "Date" not in latest_data.columns:

            if "Datetime" in latest_data.columns:

                latest_data.rename(
                    columns={
                        "Datetime": "Date"
                    },
                    inplace=True
                )

        # ----------------------------------------------------
        # Clean Close column
        # ----------------------------------------------------

        latest_data["Close"] = pd.to_numeric(
            latest_data["Close"],
            errors="coerce"
        )

        latest_data = latest_data.dropna(
            subset=["Close"]
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # Need at least 60 records
        # ----------------------------------------------------

        if len(latest_data) < SEQUENCE_LENGTH:

            raise Exception(
                "Not enough recent data."
            )

        # ----------------------------------------------------
        # Latest price
        # ----------------------------------------------------

        latest_price = float(
            latest_data["Close"].iloc[-1]
        )

        latest_date = latest_data[
            "Date"
        ].iloc[-1]

        latest_date = pd.to_datetime(
            latest_date
        ).strftime("%d %b %Y")

        # ----------------------------------------------------
        # Last 60 closing prices
        # ----------------------------------------------------

        recent_prices = latest_data[
            "Close"
        ].tail(SEQUENCE_LENGTH).values

        # ----------------------------------------------------
        # Recent chart data
        # ----------------------------------------------------

        # Send the last 1 year of data to the dashboard
        chart_data = latest_data.tail(252)

        chart_dates = [
            pd.to_datetime(date).strftime("%d %b %Y")
            for date in chart_data["Date"]
        ]

        chart_prices = [
            round(float(price), 2)
            for price in chart_data["Close"]
        ]

        return (
            latest_price,
            latest_date,
            recent_prices,
            chart_dates,
            chart_prices,
            True
        )

    except Exception as error:

        print(
            "Could not fetch latest Yahoo Finance data:"
        )

        print(error)

        print(
            "Using stored historical data instead."
        )

        # ----------------------------------------------------
        # FALLBACK TO CSV
        # ----------------------------------------------------

        latest_price = float(
            historical_data["Close"].iloc[-1]
        )

        latest_date = historical_data[
            "Date"
        ].iloc[-1].strftime("%d %b %Y")

        recent = historical_data.tail(252)

        recent_prices = recent[
            "Close"
        ].values

        chart_dates = [
            date.strftime("%d %b %Y")
            for date in recent["Date"]
        ]

        chart_prices = [
            round(float(price), 2)
            for price in recent["Close"]
        ]

        return (
            latest_price,
            latest_date,
            recent_prices,
            chart_dates,
            chart_prices,
            False
        )


# ============================================================
# MAKE LATEST PREDICTION
# ============================================================

def make_latest_prediction(recent_prices):

    recent_prices = np.array(
        recent_prices
    ).reshape(-1, 1)

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    scaled_prices = scaler.transform(
        recent_prices
    )

    # --------------------------------------------------------
    # Reshape for GRU
    # Shape = (1, 60, 1)
    # --------------------------------------------------------

    X_latest = scaled_prices.reshape(
        1,
        SEQUENCE_LENGTH,
        1
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_scaled = model.predict(
        X_latest,
        verbose=0
    )

    # --------------------------------------------------------
    # Inverse transform
    # --------------------------------------------------------

    predicted_price = scaler.inverse_transform(
        predicted_scaled
    )[0][0]

    predicted_price = float(
        predicted_price
    )

    latest_price = float(
        recent_prices[-1][0]
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    price_difference = (
        predicted_price -
        latest_price
    )

    # --------------------------------------------------------
    # Percentage change
    # --------------------------------------------------------

    percentage_change = (
        price_difference /
        latest_price
    ) * 100

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if percentage_change > 0.05:

        direction = "UP"
        direction_symbol = "📈"

    elif percentage_change < -0.05:

        direction = "DOWN"
        direction_symbol = "📉"

    else:

        direction = "NEUTRAL"
        direction_symbol = "➖"

    return (
        predicted_price,
        price_difference,
        percentage_change,
        direction,
        direction_symbol
    )


# ------------------------------------------------------------
# ADD EXPECTED MOVEMENT TO SESSION (used by dashboard card)
# ------------------------------------------------------------

def save_prediction_movement(session_data, prediction_price, latest_price):

    difference = round(
        prediction_price - latest_price,
        2
    )

    if difference > 0:
        direction = "UP"
    else:
        direction = "DOWN"

    percentage_change = round(
        (difference / latest_price) * 100,
        2
    ) if latest_price else 0.0

    session_data["difference"] = difference
    session_data["direction"] = direction
    session_data["percentage_change"] = percentage_change

    return session_data


# ============================================================
# CALCULATE MODEL RESULTS
# ============================================================

(
    MAE,
    RMSE,
    actual_prices,
    predicted_prices,
    test_dates
) = calculate_model_results()


print(
    f"Model MAE: ₹{MAE:.2f}"
)

print(
    f"Model RMSE: ₹{RMSE:.2f}"
)

PREDICTION_ACCURACY = calculate_prediction_accuracy(
    actual_prices,
    predicted_prices
)

print(
    f"Prediction Accuracy: {PREDICTION_ACCURACY:.2f}%"
)


# ============================================================
# COMMON DASHBOARD DATA
# ============================================================

def get_dashboard_data():

    (
        latest_price,
        latest_date,
        recent_prices,
        chart_dates,
        chart_prices,
        live_data_available
    ) = get_latest_tcs_data()

    # --------------------------------------------------------
    # PRICE STATISTICS
    # --------------------------------------------------------

    highest_price = round(
        float(historical_data["Close"].max()),
        2
    )

    lowest_price = round(
        float(historical_data["Close"].min()),
        2
    )

    average_price = round(
        float(historical_data["Close"].mean()),
        2
    )

    (
        predicted_price,
        price_difference,
        percentage_change,
        direction,
        direction_symbol
    ) = make_latest_prediction(
        recent_prices
    )

    # --------------------------------------------------------
    # Historical chart data
    # --------------------------------------------------------

    historical_chart_dates = [
        date.strftime("%d %b %Y")
        for date in historical_data["Date"]
    ]

    historical_chart_prices = [
        round(float(price), 2)
        for price in historical_data["Close"]
    ]

    # --------------------------------------------------------
    # Actual vs predicted chart data
    # --------------------------------------------------------

    comparison_dates = [
        date.strftime("%d %b %Y")
        for date in test_dates
    ]

    actual_chart_prices = [
        round(float(price), 2)
        for price in actual_prices
    ]

    predicted_chart_prices = [
        round(float(price), 2)
        for price in predicted_prices
    ]

    return {

        "latest_price": latest_price,

        "latest_date": latest_date,

        "predicted_price": predicted_price,

        "price_difference": price_difference,

        "percentage_change": percentage_change,

        "direction": direction,

        "direction_symbol": direction_symbol,

        "mae": MAE,

        "rmse": RMSE,

        "prediction_accuracy": PREDICTION_ACCURACY,

        "chart_dates": chart_dates,

        "chart_prices": chart_prices,

        "historical_chart_dates":
            historical_chart_dates,

        "historical_chart_prices":
            historical_chart_prices,

        "comparison_dates":
            comparison_dates,

        "actual_chart_prices":
            actual_chart_prices,

        "predicted_chart_prices":
            predicted_chart_prices,

        "live_data_available":
            live_data_available,

        "highest_price":
            highest_price,

        "lowest_price":
            lowest_price,

        "average_price":
            average_price
    }


# ============================================================
# ROUTE - HOME
# ============================================================

@app.route("/")
def index():

    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )

    return redirect(
        url_for("login")
    )


# ============================================================
# ROUTE - LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        conn = get_db_connection()

        columns = [
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
        ]

        if "username" in columns:

            user = conn.execute(
                """
                SELECT * FROM users
                WHERE username = ?
                """,
                (email,)
            ).fetchone()

            username = (
                user["username"]
                if user
                else None
            )

        elif "email" in columns:

            user = conn.execute(
                """
                SELECT * FROM users
                WHERE email = ?
                """,
                (email,)
            ).fetchone()

            username = (
                user["email"]
                if user
                else None
            )

        else:

            conn.close()

            return render_template(
                "login.html",
                error="Database structure is invalid."
            )

        conn.close()

        if user and check_password_hash(
            user["password"],
            password
        ):

            display_name = (
                user["name"]
                if "name" in user.keys() and user["name"]
                else username
            )

            user_obj = User(
                user["id"],
                username,
                display_name
            )

            login_user(user_obj)

            return redirect(
                url_for("dashboard")
            )

        return render_template(
            "login.html",
            error="Invalid email or password."
        )

    return render_template(
        "login.html"
    )


# ============================================================
# ROUTE - REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not name or not email or not password:

            return render_template(
                "register.html",
                error="Please fill all fields."
            )

        hashed_password = generate_password_hash(
            password
        )

        conn = get_db_connection()

        columns = [
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(users)"
            ).fetchall()
        ]

        try:

            if "username" in columns:

                conn.execute(
                    """
                    INSERT INTO users
                    (username, password, name)
                    VALUES (?, ?, ?)
                    """,
                    (
                        email,
                        hashed_password,
                        name
                    )
                )

            elif "email" in columns:

                conn.execute(
                    """
                    INSERT INTO users
                    (email, password)
                    VALUES (?, ?)
                    """,
                    (
                        email,
                        hashed_password
                    )
                )

            else:

                conn.close()

                return render_template(
                    "register.html",
                    error="Database structure is invalid."
                )

            conn.commit()
            conn.close()

            return redirect(
                url_for("login")
            )

        except sqlite3.IntegrityError:

            conn.close()

            return render_template(
                "register.html",
                error="Email already exists."
            )

    return render_template(
        "register.html"
    )


# ============================================================
# ROUTE - LOGOUT
# ============================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(
        url_for("login")
    )


# ============================================================
# ROUTE - DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    dashboard_data = get_dashboard_data()
    prediction_history = get_prediction_history(current_user.id)

    return render_template(
        "dashboard.html",
        current_user=current_user,
        prediction_done=False,
        prediction_history=prediction_history,
        **dashboard_data
    )


# ============================================================
# ROUTE - PREDICT
# ============================================================

# ============================================================
# ROUTE - REFRESH MARKET DATA
# ============================================================

@app.route("/refresh-market")
@login_required
def refresh_market():

    try:

        get_latest_tcs_data()

        return redirect(
            url_for("dashboard")
        )

    except Exception as e:

        print("Refresh error:", e)

        return redirect(
            url_for("dashboard")
        )


# ============================================================
# ROUTE - PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
@login_required
def predict():

    dashboard_data = get_dashboard_data()

    prediction = round(
        float(dashboard_data.get("predicted_price", 0)),
        2
    )

    session["predicted_price"] = prediction

    latest_price = float(
        dashboard_data.get("latest_price", 0)
    )

    difference = round(
        prediction - latest_price,
        2
    )

    if difference > 0:
        direction = "UP"
    else:
        direction = "DOWN"

    percentage_change = round(
        (difference / latest_price) * 100,
        2
    ) if latest_price else 0.0

    session["difference"] = difference
    session["direction"] = direction
    session["percentage_change"] = percentage_change

    dashboard_data["predicted_price"] = prediction

    prediction_date = dashboard_data.get("latest_date")

    save_prediction_history(
        current_user.id,
        prediction_date,
        latest_price,
        prediction,
        difference,
        percentage_change,
        direction
    )

    prediction_history = get_prediction_history(current_user.id)

    return render_template(
        "dashboard.html",

        current_user=current_user,

        prediction_done=True,

        prediction_history=prediction_history,

        **dashboard_data
    )


# ============================================================
# ROUTE - DOWNLOAD HISTORICAL DATA
# ============================================================

@app.route("/download/historical")
@login_required
def download_historical():

    output = io.BytesIO()

    historical_data.to_csv(
        output,
        index=False
    )

    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="TCS_historical_data.csv"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)