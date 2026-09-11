from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
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

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "gru_stock_prediction_secret_key"
)

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

if os.environ.get("RENDER"):
    app.config["SESSION_COOKIE_SECURE"] = True
else:
    app.config["SESSION_COOKIE_SECURE"] = False

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

# Original model training information
MODEL_TRAIN_SIZE = 1580
ORIGINAL_TEST_END = 1976
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


def reload_historical_data():

    global historical_data

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

    print(
        f"Reloaded TCS data: {len(historical_data)} rows"
    )


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

    def __init__(self, user_id, username, name=None):

        self.id = str(user_id)
        self.username = username
        self.name = name or username


@login_manager.user_loader
def load_user(user_id):

    print(f"Loading user from session: {user_id}")

    conn = get_db_connection()

    try:

        user = conn.execute(
            """
            SELECT id, username, name
            FROM users
            WHERE id = ?
            """,
            (user_id,)
        ).fetchone()

        conn.close()

        if user:

            print(
                f"Session user found: {user['username']}"
            )

            return User(
                user["id"],
                user["username"],
                user["name"]
            )

        print(
            f"Session user NOT found: {user_id}"
        )

        return None

    except Exception as e:

        conn.close()

        print(
            f"load_user error: {e}"
        )

        return None


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


def calculate_extended_predictions():
    """
    Generate Actual vs Predicted values from the original model
    training boundary up to the latest available trading date.

    The GRU model is NOT retrained.
    """

    global historical_data

    if historical_data is None or len(historical_data) <= MODEL_TRAIN_SIZE:
        return [], [], []

    close_prices = historical_data["Close"].values.reshape(-1, 1)

    # Keep the original model training period fixed
    train_data = close_prices[:MODEL_TRAIN_SIZE]

    # Everything after original training becomes out-of-sample data
    future_data = close_prices[MODEL_TRAIN_SIZE:]

    # Use the SAME scaler that was used during model training
    train_scaled = scaler.transform(train_data)
    future_scaled = scaler.transform(future_data)

    # Last 60 training values are needed to predict the first test value
    combined_scaled = np.concatenate(
        [train_scaled[-SEQUENCE_LENGTH:], future_scaled]
    )

    X_extended = []

    for i in range(SEQUENCE_LENGTH, len(combined_scaled)):
        X_extended.append(
            combined_scaled[i-SEQUENCE_LENGTH:i]
        )

    X_extended = np.array(X_extended)

    if len(X_extended) == 0:
        return [], [], []

    # Use existing trained GRU model
    predicted_scaled = model.predict(
        X_extended,
        verbose=0
    )

    # Convert predictions back to actual ₹ prices
    predicted_prices_extended = scaler.inverse_transform(
        predicted_scaled
    ).flatten()

    actual_prices_extended = scaler.inverse_transform(
        future_scaled
    ).flatten()

    # Corresponding dates
    test_dates_extended = (
        historical_data["Date"]
        .iloc[MODEL_TRAIN_SIZE:]
        .reset_index(drop=True)
    )

    return (
        actual_prices_extended,
        predicted_prices_extended,
        test_dates_extended
    )


# ============================================================
# GET MARKET DATA
# ============================================================

def get_stored_tcs_data():

    print("Using stored TCS data from CSV...")

    latest_price = float(
        historical_data["Close"].iloc[-1]
    )

    latest_date = historical_data[
        "Date"
    ].iloc[-1].strftime("%d %b %Y")

    recent = historical_data.tail(252)

    recent_prices = historical_data[
        "Close"
    ].tail(SEQUENCE_LENGTH).values

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


def get_latest_tcs_data():
    """
    Get latest TCS data from the locally stored CSV.

    Yahoo Finance is only called when the user explicitly
    clicks Refresh Market Data.
    """

    global historical_data

    if historical_data is None or historical_data.empty:
        return {
            "latest_price": 0,
            "latest_date": "",
            "recent_prices": [],
            "chart_dates": [],
            "chart_prices": [],
            "live": False
        }

    latest_price = float(
        historical_data["Close"].iloc[-1]
    )

    latest_date = str(
        historical_data["Date"].iloc[-1]
    )

    recent_prices = (
        historical_data["Close"]
        .tail(60)
        .tolist()
    )

    chart_data = historical_data.tail(252)

    chart_dates = (
        chart_data["Date"]
        .astype(str)
        .tolist()
    )

    chart_prices = (
        chart_data["Close"]
        .astype(float)
        .tolist()
    )

    return {
        "latest_price": latest_price,
        "latest_date": latest_date,
        "recent_prices": recent_prices,
        "chart_dates": chart_dates,
        "chart_prices": chart_prices,
        "live": False
    }


def refresh_market_data():
    """
    Download latest TCS data from Yahoo Finance and MERGE it
    with the existing full historical dataset.

    This prevents the original 2018-present history from
    being overwritten by only the latest 3 months.
    """

    global historical_data
    global actual_prices
    global predicted_prices
    global test_dates

    try:
        print("Refreshing TCS market data...")

        latest_data = yf.download(
            TICKER,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=10
        )

        if latest_data.empty:
            print("Yahoo Finance returned no data.")
            return False

        # Handle Yahoo Finance MultiIndex columns
        if isinstance(latest_data.columns, pd.MultiIndex):
            latest_data.columns = latest_data.columns.get_level_values(0)

        latest_data = latest_data.reset_index()

        # Keep only required columns
        latest_data = latest_data[["Date", "Close"]].copy()

        latest_data["Date"] = pd.to_datetime(
            latest_data["Date"]
        )

        latest_data["Close"] = pd.to_numeric(
            latest_data["Close"],
            errors="coerce"
        )

        latest_data.dropna(
            subset=["Date", "Close"],
            inplace=True
        )

        # Load existing full historical data
        existing_data = pd.read_csv(DATA_PATH)

        existing_data["Date"] = pd.to_datetime(
            existing_data["Date"]
        )

        existing_data["Close"] = pd.to_numeric(
            existing_data["Close"],
            errors="coerce"
        )

        existing_data.dropna(
            subset=["Date", "Close"],
            inplace=True
        )

        # Merge old + new data
        combined_data = pd.concat(
            [
                existing_data[["Date", "Close"]],
                latest_data[["Date", "Close"]]
            ],
            ignore_index=True
        )

        # Remove duplicate trading dates
        combined_data.drop_duplicates(
            subset=["Date"],
            keep="last",
            inplace=True
        )

        # Sort chronologically
        combined_data.sort_values(
            "Date",
            inplace=True
        )

        combined_data.reset_index(
            drop=True,
            inplace=True
        )

        # Save FULL historical dataset
        combined_data.to_csv(
            DATA_PATH,
            index=False
        )

        # Reload data
        historical_data = combined_data.copy()

        # Recalculate Actual vs Predicted graph
        (
            actual_prices,
            predicted_prices,
            test_dates
        ) = calculate_extended_predictions()

        print(
            "Market data refreshed successfully."
        )

        print(
            "Latest date:",
            historical_data["Date"].iloc[-1]
        )

        print(
            "Total rows:",
            len(historical_data)
        )

        return True

    except Exception as e:
        print(
            "Market refresh error:",
            str(e)
        )

        return False


# ============================================================
# MAKE LATEST PREDICTION
# ============================================================

def make_latest_prediction(recent_prices):
    """
    Predict the next trading day's TCS closing price
    using the latest 60 closing prices.
    """

    try:
        recent_prices = pd.to_numeric(
            pd.Series(recent_prices),
            errors="coerce"
        ).dropna().values

        if len(recent_prices) < SEQUENCE_LENGTH:
            raise ValueError(
                f"Need at least {SEQUENCE_LENGTH} prices for prediction."
            )

        recent_prices = recent_prices[-SEQUENCE_LENGTH:]

        scaled_prices = scaler.transform(
            recent_prices.reshape(-1, 1)
        )

        X_latest = scaled_prices.reshape(
            1,
            SEQUENCE_LENGTH,
            1
        )

        predicted_scaled = model.predict(
            X_latest,
            verbose=0
        )

        predicted_price = scaler.inverse_transform(
            predicted_scaled
        )[0][0]

        return float(predicted_price)

    except Exception as e:
        print(
            "Latest prediction error:",
            str(e)
        )
        return None


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

# Extended Actual vs Predicted graph
(
    actual_prices,
    predicted_prices,
    test_dates
) = calculate_extended_predictions()


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

    global actual_prices
    global predicted_prices
    global test_dates

    latest_data = get_latest_tcs_data()

    latest_price = latest_data["latest_price"]
    latest_date = latest_data["latest_date"]
    recent_prices = latest_data["recent_prices"]
    chart_dates = latest_data["chart_dates"]
    chart_prices = latest_data["chart_prices"]
    live_data_available = latest_data["live"]

    current_price = float(latest_price)

    predicted_price = make_latest_prediction(
        recent_prices
    )

    if predicted_price is not None:
        if predicted_price > current_price:
            prediction_direction = "UP"
        elif predicted_price < current_price:
            prediction_direction = "DOWN"
        else:
            prediction_direction = "UNCHANGED"
    else:
        prediction_direction = "N/A"

    # NOTE: prediction history is saved only in the /predict route.

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

    predicted_price = make_latest_prediction(
        recent_prices
    )

    current_price = float(latest_price)

    if predicted_price is not None:
        price_difference = predicted_price - current_price
        percentage_change = (
            (price_difference / current_price) * 100
        ) if current_price else 0.0

        if predicted_price > current_price:
            direction = "UP"
            direction_symbol = "📈"
        elif predicted_price < current_price:
            direction = "DOWN"
            direction_symbol = "📉"
        else:
            direction = "NEUTRAL"
            direction_symbol = "➖"
    else:
        predicted_price = current_price
        price_difference = 0.0
        percentage_change = 0.0
        direction = "N/A"
        direction_symbol = "➖"

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

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        print(
            f"LOGIN ATTEMPT: {email}"
        )

        if not email or not password:

            return render_template(
                "login.html",
                error="Please enter email and password."
            )

        conn = get_db_connection()

        try:

            user = conn.execute(
                """
                SELECT id, username, password, name
                FROM users
                WHERE LOWER(username) = ?
                """,
                (email,)
            ).fetchone()

            conn.close()

            if user is None:

                print(
                    f"LOGIN FAILED: user not found - {email}"
                )

                return render_template(
                    "login.html",
                    error="Invalid email or password."
                )

            password_valid = check_password_hash(
                user["password"],
                password
            )

            if not password_valid:

                print(
                    f"LOGIN FAILED: wrong password - {email}"
                )

                return render_template(
                    "login.html",
                    error="Invalid email or password."
                )

            logged_user = User(
                user["id"],
                user["username"],
                user["name"]
            )

            login_user(
                logged_user,
                remember=False
            )

            print(
                f"LOGIN SUCCESS: {email}"
            )

            print(
                f"AUTHENTICATED BEFORE REDIRECT: "
                f"{current_user.is_authenticated}"
            )

            return redirect(
                url_for("dashboard")
            )

        except Exception as e:

            print(
                f"LOGIN ERROR: {e}"
            )

            try:
                conn.close()
            except Exception:
                pass

            return render_template(
                "login.html",
                error="Login failed. Please try again."
            )

    return render_template("login.html")


# ============================================================
# ROUTE - REGISTER
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not name or not email or not password:
            return render_template(
                "register.html",
                error="Please fill all fields."
            )

        if len(password) < 6:
            return render_template(
                "register.html",
                error="Password must be at least 6 characters."
            )

        conn = get_db_connection()

        try:
            existing_user = conn.execute(
                "SELECT id FROM users WHERE LOWER(username) = ?",
                (email,)
            ).fetchone()

            if existing_user:
                conn.close()
                return render_template(
                    "register.html",
                    error="Email already exists. Please use another email."
                )

            hashed_password = generate_password_hash(password)

            conn.execute(
                """
                INSERT INTO users (username, password, name)
                VALUES (?, ?, ?)
                """,
                (email, hashed_password, name)
            )

            conn.commit()
            conn.close()

            print(f"Registration successful: {email}")

            return redirect(url_for("login"))

        except Exception as e:

            conn.rollback()
            conn.close()

            print(f"Registration error: {e}")

            return render_template(
                "register.html",
                error="Registration failed. Please try again."
            )

    return render_template("register.html")


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

    print(
        "DASHBOARD REQUEST"
    )

    print(
        f"Authenticated: "
        f"{current_user.is_authenticated}"
    )

    print(
        f"User ID: "
        f"{current_user.id}"
    )

    print(
        f"Username: "
        f"{current_user.username}"
    )

    dashboard_data = get_dashboard_data()
    prediction_history = get_prediction_history(
        current_user.id
    )

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
    success = refresh_market_data()

    if success:
        flash(
            "Market data refreshed successfully.",
            "success"
        )
    else:
        flash(
            "Unable to refresh market data. Showing stored data.",
            "warning"
        )

    return redirect(url_for("dashboard"))


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