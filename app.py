from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_file,
    flash
)

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

import sqlite3
import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from tensorflow.keras.models import load_model


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


# ============================================================
# LOGIN CONFIGURATION
# ============================================================

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
EVALUATION_CACHE_PATH = "data/model_evaluation.json"

MODEL_TRAIN_SIZE = 1580
ORIGINAL_TEST_END = 1976
SEQUENCE_LENGTH = 60
TICKER = "TCS.NS"


# ============================================================
# LOAD GRU MODEL
# ============================================================

print("Loading GRU model...")

# compile=False is enough because this application only predicts.
# It reduces unnecessary model loading overhead.
model = load_model(
    MODEL_PATH,
    compile=False
)

scaler = joblib.load(SCALER_PATH)

print("GRU model loaded successfully.")


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_historical_data():

    data = pd.read_csv(DATA_PATH)

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Date", "Close"]
    )

    data = data.sort_values(
        "Date"
    )

    data = data.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    data = data.reset_index(
        drop=True
    )

    return data


historical_data = load_historical_data()


# ============================================================
# GLOBAL CACHE
# ============================================================

# This prevents TensorFlow prediction from running every time
# the dashboard page is opened.

LATEST_PREDICTION_CACHE = {
    "date": None,
    "price": None
}


# Model evaluation results are calculated once when the server
# starts instead of repeatedly during requests.

MAE = 0.0
RMSE = 0.0
PREDICTION_ACCURACY = 0.0

actual_prices = np.array([])
predicted_prices = np.array([])
test_dates = pd.Series(dtype="datetime64[ns]")


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():

    conn = sqlite3.connect(
        DATABASE,
        timeout=10
    )

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

    # Remove old duplicate rows, then enforce one prediction per user/date.
    conn.execute(
        """
        DELETE FROM prediction_history
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM prediction_history
            GROUP BY user_id, prediction_date
        )
        """
    )

    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_prediction_history_user_date
        ON prediction_history(user_id, prediction_date)
        """
    )

    conn.commit()
    conn.close()


init_db()


# ============================================================
# USER CLASS
# ============================================================

class User(UserMixin):

    def __init__(
        self,
        user_id,
        username,
        name=None
    ):

        self.id = str(user_id)
        self.username = username
        self.name = name or username


@login_manager.user_loader
def load_user(user_id):

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

            return User(
                user["id"],
                user["username"],
                user["name"]
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

def calculate_prediction_accuracy(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    mask = actual != 0

    if not np.any(mask):
        return 0.0

    mape = np.mean(
        np.abs(
            (
                actual[mask] -
                predicted[mask]
            )
            /
            actual[mask]
        )
    ) * 100

    return max(
        0.0,
        100.0 - mape
    )


# ============================================================
# LOAD PRECOMPUTED MODEL EVALUATION
# ============================================================

def load_evaluation_cache():
    """Load evaluation results without running TensorFlow."""
    global MAE, RMSE, PREDICTION_ACCURACY
    global actual_prices, predicted_prices, test_dates

    if not os.path.exists(EVALUATION_CACHE_PATH):
        print("Evaluation cache not found; comparison chart will be empty.")
        return

    try:
        with open(EVALUATION_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)

        MAE = float(cached.get("mae", 0.0))
        RMSE = float(cached.get("rmse", 0.0))
        PREDICTION_ACCURACY = float(cached.get("accuracy", 0.0))
        actual_prices = np.asarray(cached.get("actual_prices", []), dtype=float)
        predicted_prices = np.asarray(cached.get("predicted_prices", []), dtype=float)
        test_dates = pd.to_datetime(
            pd.Series(cached.get("test_dates", [])), errors="coerce"
        ).dropna().reset_index(drop=True)

        print(f"Model MAE: ₹{MAE:.2f}")
        print(f"Model RMSE: ₹{RMSE:.2f}")
        print(f"Prediction Accuracy: {PREDICTION_ACCURACY:.2f}%")
    except Exception as e:
        print(f"Evaluation cache load error: {e}")


# ============================================================
# PREDICTION HISTORY
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
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO prediction_history
            (
                user_id, prediction_date, latest_price,
                predicted_price, difference, percentage_change, direction
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, prediction_date, latest_price,
                predicted_price, difference, percentage_change, direction
            )
        )
        conn.commit()
    finally:
        conn.close()


def get_prediction_history(
    user_id,
    limit=10
):

    conn = get_db_connection()

    rows = conn.execute(
        """
        SELECT
            prediction_date,
            latest_price,
            predicted_price,
            difference,
            percentage_change,
            direction
        FROM prediction_history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (
            user_id,
            limit
        )
    ).fetchall()

    conn.close()

    return rows


# ============================================================
# MODEL EVALUATION
# ============================================================

def calculate_model_results():

    global historical_data

    data = historical_data[
        "Close"
    ].values.astype(
        np.float32
    ).reshape(
        -1,
        1
    )

    total_rows = len(data)

    train_size = int(
        total_rows * 0.80
    )

    train_data = data[
        :train_size
    ]

    test_data = data[
        train_size:
    ]

    # --------------------------------------------------------
    # SCALE DATA
    # --------------------------------------------------------

    train_scaled = scaler.transform(
        train_data
    )

    test_scaled = scaler.transform(
        test_data
    )

    # --------------------------------------------------------
    # ONLY CREATE TEST SEQUENCES
    #
    # The old code created X_train even though it was never
    # used for prediction. That unnecessary work is removed.
    # --------------------------------------------------------

    combined_test = np.concatenate(
        (
            train_scaled[
                -SEQUENCE_LENGTH:
            ],
            test_scaled
        )
    )

    X_test = []

    for i in range(
        SEQUENCE_LENGTH,
        len(combined_test)
    ):

        X_test.append(
            combined_test[
                i - SEQUENCE_LENGTH:i
            ]
        )

    X_test = np.asarray(
        X_test,
        dtype=np.float32
    )

    y_test = np.asarray(
        test_scaled,
        dtype=np.float32
    )

    if len(X_test) == 0:

        return (
            0.0,
            0.0,
            np.array([]),
            np.array([]),
            pd.Series(dtype="datetime64[ns]")
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions_scaled = model.predict(
        X_test,
        batch_size=32,
        verbose=0
    )

    # --------------------------------------------------------
    # INVERSE TRANSFORM
    # --------------------------------------------------------

    predictions = scaler.inverse_transform(
        predictions_scaled
    ).flatten()

    actual = scaler.inverse_transform(
        y_test.reshape(
            -1,
            1
        )
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
    # DATES
    # --------------------------------------------------------

    test_dates_result = historical_data[
        "Date"
    ].iloc[
        train_size:
    ].reset_index(
        drop=True
    )

    return (
        float(mae),
        float(rmse),
        actual,
        predictions,
        test_dates_result
    )


# ============================================================
# EXTENDED ACTUAL VS PREDICTED
# ============================================================

def calculate_extended_predictions():

    global historical_data

    if (
        historical_data is None
        or
        len(historical_data) <= MODEL_TRAIN_SIZE
    ):

        return (
            np.array([]),
            np.array([]),
            pd.Series(dtype="datetime64[ns]")
        )

    close_prices = historical_data[
        "Close"
    ].values.astype(
        np.float32
    ).reshape(
        -1,
        1
    )

    # Original training period remains fixed.
    train_data = close_prices[
        :MODEL_TRAIN_SIZE
    ]

    future_data = close_prices[
        MODEL_TRAIN_SIZE:
    ]

    train_scaled = scaler.transform(
        train_data
    )

    future_scaled = scaler.transform(
        future_data
    )

    combined_scaled = np.concatenate(
        [
            train_scaled[
                -SEQUENCE_LENGTH:
            ],
            future_scaled
        ]
    )

    X_extended = []

    for i in range(
        SEQUENCE_LENGTH,
        len(combined_scaled)
    ):

        X_extended.append(
            combined_scaled[
                i - SEQUENCE_LENGTH:i
            ]
        )

    X_extended = np.asarray(
        X_extended,
        dtype=np.float32
    )

    if len(X_extended) == 0:

        return (
            np.array([]),
            np.array([]),
            pd.Series(dtype="datetime64[ns]")
        )

    predicted_scaled = model.predict(
        X_extended,
        batch_size=32,
        verbose=0
    )

    predicted_prices_extended = (
        scaler.inverse_transform(
            predicted_scaled
        )
        .flatten()
    )

    actual_prices_extended = (
        scaler.inverse_transform(
            future_scaled
        )
        .flatten()
    )

    test_dates_extended = (
        historical_data[
            "Date"
        ]
        .iloc[
            MODEL_TRAIN_SIZE:
        ]
        .reset_index(
            drop=True
        )
    )

    return (
        actual_prices_extended,
        predicted_prices_extended,
        test_dates_extended
    )


# ============================================================
# GET STORED TCS DATA
# ============================================================

def get_latest_tcs_data():

    global historical_data

    if (
        historical_data is None
        or
        historical_data.empty
    ):

        return {
            "latest_price": 0.0,
            "latest_date": "",
            "recent_prices": [],
            "chart_dates": [],
            "chart_prices": [],
            "live": False
        }

    latest_price = float(
        historical_data[
            "Close"
        ].iloc[-1]
    )

    latest_date = (
        historical_data[
            "Date"
        ].iloc[-1]
        .strftime(
            "%d %b %Y"
        )
    )

    recent_prices = (
        historical_data[
            "Close"
        ]
        .tail(SEQUENCE_LENGTH)
        .astype(float)
        .tolist()
    )

    chart_data = historical_data.tail(
        252
    )

    chart_dates = [
        date.strftime(
            "%d %b %Y"
        )
        for date in chart_data[
            "Date"
        ]
    ]

    chart_prices = [
        round(
            float(price),
            2
        )
        for price in chart_data[
            "Close"
        ]
    ]

    return {
        "latest_price": latest_price,
        "latest_date": latest_date,
        "recent_prices": recent_prices,
        "chart_dates": chart_dates,
        "chart_prices": chart_prices,
        "live": False
    }


# ============================================================
# REFRESH MARKET DATA
# ============================================================

def refresh_market_data():

    global historical_data
    global actual_prices
    global predicted_prices
    global test_dates
    global LATEST_PREDICTION_CACHE

    try:

        print(
            "Refreshing TCS market data..."
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

            print(
                "Yahoo Finance returned no data."
            )

            return False

        # ----------------------------------------------------
        # HANDLE MULTIINDEX
        # ----------------------------------------------------

        if isinstance(
            latest_data.columns,
            pd.MultiIndex
        ):

            latest_data.columns = (
                latest_data.columns
                .get_level_values(0)
            )

        latest_data = (
            latest_data
            .reset_index()
        )

        latest_data = latest_data[
            ["Date", "Close"]
        ].copy()

        latest_data["Date"] = (
            pd.to_datetime(
                latest_data["Date"],
                errors="coerce"
            )
        )

        latest_data["Close"] = (
            pd.to_numeric(
                latest_data["Close"],
                errors="coerce"
            )
        )

        latest_data.dropna(
            subset=[
                "Date",
                "Close"
            ],
            inplace=True
        )

        # ----------------------------------------------------
        # LOAD EXISTING FULL DATA
        # ----------------------------------------------------

        existing_data = pd.read_csv(
            DATA_PATH
        )

        existing_data["Date"] = (
            pd.to_datetime(
                existing_data["Date"],
                errors="coerce"
            )
        )

        existing_data["Close"] = (
            pd.to_numeric(
                existing_data["Close"],
                errors="coerce"
            )
        )

        existing_data.dropna(
            subset=[
                "Date",
                "Close"
            ],
            inplace=True
        )

        # ----------------------------------------------------
        # MERGE OLD + NEW
        # ----------------------------------------------------

        combined_data = pd.concat(
            [
                existing_data[
                    ["Date", "Close"]
                ],
                latest_data[
                    ["Date", "Close"]
                ]
            ],
            ignore_index=True
        )

        combined_data.drop_duplicates(
            subset=["Date"],
            keep="last",
            inplace=True
        )

        combined_data.sort_values(
            "Date",
            inplace=True
        )

        combined_data.reset_index(
            drop=True,
            inplace=True
        )

        # ----------------------------------------------------
        # SAVE FULL DATASET
        # ----------------------------------------------------

        combined_data.to_csv(
            DATA_PATH,
            index=False
        )

        historical_data = (
            combined_data.copy()
        )

        # ----------------------------------------------------
        # CLEAR LATEST PREDICTION CACHE
        # ----------------------------------------------------

        LATEST_PREDICTION_CACHE = {
            "date": None,
            "price": None
        }

        print(
            "Market data refreshed successfully."
        )

        print(
            "Latest date:",
            historical_data[
                "Date"
            ].iloc[-1]
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

def make_latest_prediction(
    recent_prices
):

    try:

        recent_prices = pd.to_numeric(
            pd.Series(
                recent_prices
            ),
            errors="coerce"
        ).dropna().values.astype(
            np.float32
        )

        if len(recent_prices) < SEQUENCE_LENGTH:

            raise ValueError(
                f"Need at least "
                f"{SEQUENCE_LENGTH} prices "
                f"for prediction."
            )

        recent_prices = (
            recent_prices[
                -SEQUENCE_LENGTH:
            ]
        )

        scaled_prices = scaler.transform(
            recent_prices.reshape(
                -1,
                1
            )
        )

        X_latest = scaled_prices.reshape(
            1,
            SEQUENCE_LENGTH,
            1
        ).astype(
            np.float32
        )

        predicted_scaled = model.predict(
            X_latest,
            batch_size=1,
            verbose=0
        )

        predicted_price = (
            scaler.inverse_transform(
                predicted_scaled
            )[0][0]
        )

        return float(
            predicted_price
        )

    except Exception as e:

        print(
            "Latest prediction error:",
            str(e)
        )

        return None


# ============================================================
# CACHED LATEST PREDICTION
# ============================================================

def get_cached_latest_prediction(
    recent_prices,
    latest_date
):

    global LATEST_PREDICTION_CACHE

    # --------------------------------------------------------
    # RETURN EXISTING PREDICTION
    # --------------------------------------------------------

    if (
        LATEST_PREDICTION_CACHE["date"]
        == latest_date
        and
        LATEST_PREDICTION_CACHE["price"]
        is not None
    ):

        return float(
            LATEST_PREDICTION_CACHE[
                "price"
            ]
        )

    # --------------------------------------------------------
    # RUN GRU ONLY ONCE FOR THIS DATE
    # --------------------------------------------------------

    prediction = make_latest_prediction(
        recent_prices
    )

    if prediction is not None:

        LATEST_PREDICTION_CACHE = {
            "date": latest_date,
            "price": float(prediction)
        }

    return prediction


# ============================================================
# DASHBOARD DATA
# ============================================================

def get_dashboard_data():

    global historical_data
    global actual_prices
    global predicted_prices
    global test_dates

    latest_data = get_latest_tcs_data()

    latest_price = (
        latest_data[
            "latest_price"
        ]
    )

    latest_date = (
        latest_data[
            "latest_date"
        ]
    )

    recent_prices = (
        latest_data[
            "recent_prices"
        ]
    )

    chart_dates = (
        latest_data[
            "chart_dates"
        ]
    )

    chart_prices = (
        latest_data[
            "chart_prices"
        ]
    )

    live_data_available = (
        latest_data[
            "live"
        ]
    )

    current_price = float(
        latest_price
    )

    # --------------------------------------------------------
    # IMPORTANT: dashboard never runs TensorFlow inference.
    # Prediction is performed only by the /predict route.
    # --------------------------------------------------------

    predicted_price = None

    if session.get("prediction_date") == latest_date:
        try:
            predicted_price = float(session.get("predicted_price"))
        except (TypeError, ValueError):
            predicted_price = None

    if predicted_price is None and (
        LATEST_PREDICTION_CACHE["date"] == latest_date
        and LATEST_PREDICTION_CACHE["price"] is not None
    ):
        predicted_price = float(LATEST_PREDICTION_CACHE["price"])


    # --------------------------------------------------------
    # PRICE MOVEMENT
    # --------------------------------------------------------

    if predicted_price is not None:

        price_difference = (
            predicted_price -
            current_price
        )

        percentage_change = (
            (
                price_difference /
                current_price
            ) * 100
            if current_price
            else 0.0
        )

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
    # PRICE STATISTICS
    # --------------------------------------------------------

    highest_price = round(
        float(
            historical_data[
                "Close"
            ].max()
        ),
        2
    )

    lowest_price = round(
        float(
            historical_data[
                "Close"
            ].min()
        ),
        2
    )

    average_price = round(
        float(
            historical_data[
                "Close"
            ].mean()
        ),
        2
    )

    # --------------------------------------------------------
    # HISTORICAL CHART
    #
    # Keep 252 points instead of sending the entire dataset.
    # --------------------------------------------------------

    historical_chart_data = (
        historical_data.tail(252)
    )

    historical_chart_dates = [
        date.strftime(
            "%d %b %Y"
        )
        for date in historical_chart_data[
            "Date"
        ]
    ]

    historical_chart_prices = [
        round(
            float(price),
            2
        )
        for price in historical_chart_data[
            "Close"
        ]
    ]

    # --------------------------------------------------------
    # ACTUAL VS PREDICTED
    # --------------------------------------------------------

    comparison_dates = [
        date.strftime(
            "%d %b %Y"
        )
        for date in test_dates
    ]

    actual_chart_prices = [
        round(
            float(price),
            2
        )
        for price in actual_prices
    ]

    predicted_chart_prices = [
        round(
            float(price),
            2
        )
        for price in predicted_prices
    ]

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "latest_price":
            latest_price,

        "latest_date":
            latest_date,

        "predicted_price":
            round(
                float(predicted_price),
                2
            ),

        "price_difference":
            round(
                float(price_difference),
                2
            ),

        "percentage_change":
            round(
                float(percentage_change),
                2
            ),

        "direction":
            direction,

        "direction_symbol":
            direction_symbol,

        "mae":
            MAE,

        "rmse":
            RMSE,

        "prediction_accuracy":
            PREDICTION_ACCURACY,

        "chart_dates":
            chart_dates,

        "chart_prices":
            chart_prices,

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
# LOAD MODEL EVALUATION WITHOUT RUNNING TENSORFLOW
# ============================================================

load_evaluation_cache()

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
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        if not email or not password:

            return render_template(
                "login.html",
                error=(
                    "Please enter "
                    "email and password."
                )
            )

        conn = get_db_connection()

        try:

            user = conn.execute(
                """
                SELECT
                    id,
                    username,
                    password,
                    name
                FROM users
                WHERE LOWER(username) = ?
                """,
                (email,)
            ).fetchone()

            conn.close()

            if user is None:

                return render_template(
                    "login.html",
                    error=(
                        "Invalid email "
                        "or password."
                    )
                )

            password_valid = (
                check_password_hash(
                    user["password"],
                    password
                )
            )

            if not password_valid:

                return render_template(
                    "login.html",
                    error=(
                        "Invalid email "
                        "or password."
                    )
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

            return redirect(
                url_for("dashboard")
            )

        except Exception as e:

            try:
                conn.close()
            except Exception:
                pass

            print(
                f"Login error: {e}"
            )

            return render_template(
                "login.html",
                error=(
                    "Login failed. "
                    "Please try again."
                )
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
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        ).strip()

        if not name or not email or not password:

            return render_template(
                "register.html",
                error="Please fill all fields."
            )

        if len(password) < 6:

            return render_template(
                "register.html",
                error=(
                    "Password must be "
                    "at least 6 characters."
                )
            )

        conn = get_db_connection()

        try:

            existing_user = conn.execute(
                """
                SELECT id
                FROM users
                WHERE LOWER(username) = ?
                """,
                (email,)
            ).fetchone()

            if existing_user:

                conn.close()

                return render_template(
                    "register.html",
                    error=(
                        "Email already exists. "
                        "Please use another email."
                    )
                )

            hashed_password = (
                generate_password_hash(
                    password
                )
            )

            conn.execute(
                """
                INSERT INTO users
                (
                    username,
                    password,
                    name
                )
                VALUES (?, ?, ?)
                """,
                (
                    email,
                    hashed_password,
                    name
                )
            )

            conn.commit()
            conn.close()

            return redirect(
                url_for("login")
            )

        except Exception as e:

            conn.rollback()
            conn.close()

            print(
                f"Registration error: {e}"
            )

            return render_template(
                "register.html",
                error=(
                    "Registration failed. "
                    "Please try again."
                )
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

    dashboard_data = (
        get_dashboard_data()
    )

    prediction_history = (
        get_prediction_history(
            current_user.id
        )
    )

    return render_template(
        "dashboard.html",
        current_user=current_user,
        prediction_done=False,
        prediction_history=prediction_history,
        **dashboard_data
    )


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
            "Unable to refresh market data. "
            "Showing stored data.",
            "warning"
        )

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
    latest_data = get_latest_tcs_data()

    latest_price = float(latest_data.get("latest_price", 0.0))
    latest_date = latest_data.get("latest_date", "")
    recent_prices = latest_data.get("recent_prices", [])

    if not latest_date or latest_price <= 0:
        flash("Market data is not available for prediction.", "warning")
        return redirect(url_for("dashboard"))

    # The GRU model runs here, only when the user requests a prediction.
    prediction = make_latest_prediction(recent_prices)

    if prediction is None:
        flash("Prediction failed. Please try again.", "warning")
        return redirect(url_for("dashboard"))

    prediction = round(float(prediction), 2)
    difference = round(prediction - latest_price, 2)

    if difference > 0:
        direction = "UP"
        direction_symbol = "📈"
    elif difference < 0:
        direction = "DOWN"
        direction_symbol = "📉"
    else:
        direction = "NEUTRAL"
        direction_symbol = "➖"

    percentage_change = round(
        (difference / latest_price) * 100, 2
    ) if latest_price else 0.0

    global LATEST_PREDICTION_CACHE
    LATEST_PREDICTION_CACHE = {
        "date": latest_date,
        "price": prediction
    }

    session["predicted_price"] = prediction
    session["prediction_date"] = latest_date
    session["difference"] = difference
    session["direction"] = direction
    session["percentage_change"] = percentage_change

    save_prediction_history(
        current_user.id,
        latest_date,
        latest_price,
        prediction,
        difference,
        percentage_change,
        direction
    )

    dashboard_data = get_dashboard_data()
    dashboard_data["predicted_price"] = prediction
    dashboard_data["price_difference"] = difference
    dashboard_data["percentage_change"] = percentage_change
    dashboard_data["direction"] = direction
    dashboard_data["direction_symbol"] = direction_symbol

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

@app.route(
    "/download/historical"
)
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
        download_name=(
            "TCS_historical_data.csv"
        )
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():
    return "OK", 200


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )