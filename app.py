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
import numpy as np
import pandas as pd
import yfinance as yf

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

DATA_PATH = "data/TCS_stock.csv"
DATABASE = "database.db"
EVALUATION_CACHE_PATH = "data/model_evaluation.json"
LATEST_PREDICTION_PATH = "data/latest_prediction.json"

MODEL_TRAIN_SIZE = 1580
ORIGINAL_TEST_END = 1976
SEQUENCE_LENGTH = 60
TICKER = "TCS.NS"


# ============================================================
# NOTE ABOUT GRU MODEL
# ============================================================
# The GRU model is intentionally NOT loaded by the Render Flask
# server. TensorFlow inference caused worker timeouts/memory kills
# on the small Render instance.
#
# The model is still used by generate_latest_prediction.py on the
# local machine to create data/latest_prediction.json. The Flask
# app only reads that lightweight JSON result.
# ============================================================

print("Starting Flask application without TensorFlow inference...")


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
# MODEL EVALUATION NOTE
# ============================================================
# Evaluation metrics and actual-vs-predicted chart data are loaded
# from data/model_evaluation.json. The Flask server never recalculates
# them with TensorFlow.


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
# LOAD LATEST PRECOMPUTED PREDICTION
# ============================================================

def load_latest_prediction():
    """Read the latest GRU prediction generated locally."""
    if not os.path.exists(LATEST_PREDICTION_PATH):
        print("Latest prediction file not found.")
        return None

    try:
        with open(LATEST_PREDICTION_PATH, "r", encoding="utf-8") as f:
            prediction = json.load(f)

        required = [
            "latest_date",
            "latest_price",
            "predicted_price",
            "difference",
            "percentage_change",
            "direction"
        ]

        if not all(key in prediction for key in required):
            print("Latest prediction JSON is incomplete.")
            return None

        return prediction

    except Exception as e:
        print("Latest prediction file error:", str(e))
        return None


def refresh_prediction_cache_from_file():
    """Load the JSON prediction into the small in-process cache."""
    global LATEST_PREDICTION_CACHE

    prediction = load_latest_prediction()

    if prediction is None:
        LATEST_PREDICTION_CACHE = {
            "date": None,
            "price": None
        }
        return None

    try:
        LATEST_PREDICTION_CACHE = {
            "date": str(prediction["latest_date"]),
            "price": float(prediction["predicted_price"])
        }
    except (TypeError, ValueError, KeyError):
        LATEST_PREDICTION_CACHE = {
            "date": None,
            "price": None
        }
        return None

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
    # PRECOMPUTED GRU PREDICTION
    # --------------------------------------------------------

    prediction_data = refresh_prediction_cache_from_file()
    predicted_price = None

    if prediction_data is not None:
        try:
            predicted_price = float(prediction_data["predicted_price"])
            price_difference = float(prediction_data["difference"])
            percentage_change = float(prediction_data["percentage_change"])
            direction = str(prediction_data["direction"])
            direction_symbol = str(
                prediction_data.get(
                    "direction_symbol",
                    "📈" if direction == "UP" else "📉" if direction == "DOWN" else "➖"
                )
            )
        except (TypeError, ValueError, KeyError):
            predicted_price = None

    if predicted_price is None:
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
refresh_prediction_cache_from_file()

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
    """Display the latest GRU prediction generated offline."""
    latest_data = get_latest_tcs_data()

    latest_price = float(latest_data.get("latest_price", 0.0))
    latest_date = latest_data.get("latest_date", "")

    prediction_data = load_latest_prediction()

    if prediction_data is None:
        flash(
            "No GRU prediction is available yet. Run "
            "generate_latest_prediction.py locally, then deploy again.",
            "warning"
        )
        return redirect(url_for("dashboard"))

    try:
        prediction_date = str(prediction_data["latest_date"])
        prediction_source_price = float(prediction_data["latest_price"])
        prediction = round(float(prediction_data["predicted_price"]), 2)
        difference = round(float(prediction_data["difference"]), 2)
        percentage_change = round(
            float(prediction_data["percentage_change"]),
            2
        )
        direction = str(prediction_data["direction"])
        direction_symbol = str(
            prediction_data.get(
                "direction_symbol",
                "📈" if direction == "UP" else "📉" if direction == "DOWN" else "➖"
            )
        )
    except (TypeError, ValueError, KeyError) as e:
        print("Prediction JSON parsing error:", str(e))
        flash(
            "The saved GRU prediction file is invalid. "
            "Please regenerate it locally.",
            "warning"
        )
        return redirect(url_for("dashboard"))

    # Keep the prediction result internally consistent with the price
    # used by the GRU generator. This prevents a mismatched percentage
    # when the market CSV on Render is newer than the JSON file.
    latest_price = prediction_source_price

    global LATEST_PREDICTION_CACHE
    LATEST_PREDICTION_CACHE = {
        "date": prediction_date,
        "price": prediction
    }

    session["predicted_price"] = prediction
    session["prediction_date"] = prediction_date
    session["difference"] = difference
    session["direction"] = direction
    session["percentage_change"] = percentage_change

    save_prediction_history(
        current_user.id,
        prediction_date,
        latest_price,
        prediction,
        difference,
        percentage_change,
        direction
    )

    dashboard_data = get_dashboard_data()
    dashboard_data["latest_price"] = latest_price
    dashboard_data["latest_date"] = prediction_date
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